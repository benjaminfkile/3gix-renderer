//! The client against the v2 mock hub, with the core library's example
//! module as the only compiler (`client-compute.md` sections 1 to 6,
//! `space-model.md` section 11's first two acceptance points):
//!
//! - a first client with an empty cache computes and uploads the registry
//!   and every section it needs;
//! - a second client with a fresh cache streams every one of them and runs
//!   the module zero times, ending with the same matter;
//! - a third client reusing that cache reads everything locally;
//! - a `missing` section is computed and offered once with the section
//!   2.5 headers, and stored; a `local` one is computed and never offered;
//! - a cached copy whose hash differs from the `sha256` of a `ready`
//!   answer is discarded and the new version streamed;
//! - a revoked key's offers are counted as failed and never retried;
//! - with a hub that never answers uploads, every wanted compute still
//!   runs on the pool and finishes, and at most four uploads are in
//!   flight at once;
//! - two clients at once both compute what they need; the hub stores the
//!   first copy and matches the second; directions go out in batches of
//!   at most 512 keys;
//! - checks (section 2.6): at rate 0 nothing is checked, at rate 1 every
//!   downloaded section is recomputed once the view's computes are done
//!   and reported `match`; a section the hub stores wrongly is replaced
//!   by the computed one and reported `disputed`; while wanted computes
//!   wait no check runs;
//! - a headless render against the mock hub, entered at now and at the
//!   origin, produces a non-empty image each time, saved as
//!   `target/tmp/mock-hub-home-now.png` and `target/tmp/mock-hub-home-origin.png`.
//!
//! The space has two layers: `frames`, a registry-only module declaring the
//! two frames of the shared scene (`common`), and `matter`, the example
//! module on every frame up to depth 1 with its source tiles.

mod common;

use gx_core::frames::FrameSystem;
use gx_core::key::CellKey;
use gx_core::timescale::utc_ms_to_tdb_seconds;
use gx_core::units::Seconds;
use gx_worker::MAX_UPLOADS_IN_FLIGHT;
use renderer::app::{check_origin, matter_step, render_headless_view, HeadlessRun};
use renderer::camera::Camera;
use renderer::config::{ApiKey, View};
use renderer::extract::default_threads;
use renderer::hub::HubClient;
use renderer::mock_hub::{
    example_matter_layer, registry_layer, MockHub, MockHubOptions, MOCK_API_KEY, MOCK_SPACE_ID,
};
use renderer::protocol::Routes;
use renderer::seen::Seen;
use renderer::sim::{SimClock, Simulation};
use renderer::stream::{CellState, FetchEvent, Fetcher, FetcherOptions, FetcherStats};
use renderer::world::World;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

const EXAMPLE_MODULE: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../testdata/example-module/module.wasm"
));

const WIDTH: u32 = 320;
const HEIGHT: u32 = 180;

/// The deepest cell the matter layer fills.
const MATTER_DEPTH: u8 = 1;

fn space() -> MockHubOptions {
    space_with_registry(common::registry_bytes())
}

fn space_with_registry(registry: Vec<u8>) -> MockHubOptions {
    let (matter, tiles) = example_matter_layer(
        "matter",
        EXAMPLE_MODULE,
        &[
            (common::ROOT, common::ROOT_EXTENT),
            (common::CHILD, common::CHILD_EXTENT),
        ],
        MATTER_DEPTH,
        20.0,
        6000.0,
    );
    MockHubOptions::new(vec![registry_layer("frames", &registry), matter], tiles)
}

fn cache_dir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("gx-renderer-cache-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    d
}

struct Client {
    fetcher: Fetcher,
    system: FrameSystem,
}

/// Connects a client to `hub`, with a cache in `cache` (or none), and
/// loads the registry.
fn connect(rt: &tokio::runtime::Runtime, hub: &MockHub, cache: Option<&PathBuf>) -> Client {
    let client = rt
        .block_on(HubClient::open(
            Routes::new(&hub.url(), None, MOCK_SPACE_ID),
            ApiKey::new(MOCK_API_KEY),
            None,
        ))
        .expect("the mock hub answers");
    let cache = cache.map(|d| Arc::new(gx_worker::DiskCache::open(d, 1 << 30).unwrap()));
    let mut fetcher = Fetcher::new(
        rt.handle().clone(),
        Arc::new(client),
        FetcherOptions {
            cache,
            worker_threads: 2,
            ..FetcherOptions::default()
        },
    )
    .unwrap();
    let system = fetcher.load_registry(Duration::from_secs(60)).unwrap();
    Client { fetcher, system }
}

/// Streams the `Home` view's selection and pinned cells until every one is
/// resolved, then waits for the uploads to finish. Returns the world.
fn stream_home(c: &mut Client) -> World {
    let world = settle_home(c);
    let start = Instant::now();
    while c.fetcher.stats().worker.uploads_pending > 0 {
        assert!(start.elapsed() < Duration::from_secs(30), "uploads hang");
        std::thread::sleep(Duration::from_millis(10));
    }
    world
}

/// Streams the `Home` view's selection and pinned cells until every one is
/// resolved, without waiting for uploads. Returns the world.
fn settle_home(c: &mut Client) -> World {
    let mut camera = Camera::home(&c.system);
    camera.update_parent(&c.system);
    let mut world = World::new(default_threads());
    let start = Instant::now();
    let seen = Seen::instant(&c.system, &camera);
    loop {
        let now = start.elapsed().as_secs_f64();
        matter_step(&mut world, &mut c.fetcher, &seen, (WIDTH, HEIGHT), now);
        if world.settled() {
            break;
        }
        assert!(now < 90.0, "the view did not settle");
        c.fetcher.wait_for_events(Duration::from_millis(20));
    }
    world
}

/// The composited matter of every ready cell, as sample densities, so two
/// runs can be compared.
fn matter(world: &World) -> BTreeMap<CellKey, Vec<f64>> {
    world
        .cache()
        .entries()
        .filter_map(|(k, e)| match &e.state {
            CellState::Ready(cell) => {
                let s = cell.section.samples()?;
                Some((*k, (0..s.len()).map(|i| s.density(i).value()).collect()))
            }
            _ => None,
        })
        .collect()
}

#[test]
fn first_run_computes_and_uploads_second_run_streams_with_zero_module_runs() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let hub = rt.block_on(MockHub::start(space())).unwrap();
    assert_eq!(
        hub.origin_utc_ms(),
        None,
        "no origin before the first client"
    );

    // First client: everything is computed here and uploaded.
    let dir_a = cache_dir("a");
    let mut a = connect(&rt, &hub, Some(&dir_a));
    assert!(
        hub.origin_utc_ms().is_some(),
        "the first directions set the origin"
    );
    assert_eq!(a.system.tree().frames().len(), 2);
    // The matter layer's config raises both frames to its depth.
    for f in a.system.tree().frames() {
        assert_eq!(f.max_depth, MATTER_DEPTH, "frame {}", f.frame_id);
    }
    let world_a = stream_home(&mut a);
    let sa: FetcherStats = a.fetcher.stats();
    let after_a = hub.counts();
    let cells_a = matter(&world_a);
    assert!(!cells_a.is_empty(), "the view has matter");
    // Every address the hub directed this client to compute was computed
    // here, uploaded, and stored; nothing was streamed.
    assert!(after_a.missing_answers >= 3, "{after_a:?}");
    assert_eq!(
        after_a.uploads_stored, after_a.missing_answers,
        "{after_a:?}"
    );
    assert_eq!(after_a.uploads_refused, 0);
    assert_eq!(sa.worker.sections_computed, after_a.missing_answers);
    assert_eq!(sa.worker.uploads_offered, after_a.missing_answers);
    assert_eq!(sa.worker.uploads_stored, after_a.missing_answers);
    assert_eq!(sa.worker.uploads_failed + sa.worker.uploads_dropped, 0);
    assert_eq!(
        sa.worker.modules_loaded, 2,
        "the registry module and the example module"
    );
    assert_eq!(sa.worker.module_runs, 2 * after_a.missing_answers);
    assert_eq!(
        sa.directions.sections_streamed + sa.directions.sections_cached,
        0
    );
    assert_eq!(after_a.section_reads, 0);
    // Every tile the module read was fetched once and cached.
    assert!(
        after_a.source_reads >= 1 && after_a.manifest_reads == 1,
        "{after_a:?}"
    );
    let stored = hub.sections();
    assert_eq!(stored.len() as u64, after_a.uploads_stored);
    drop(a);

    // Second client, fresh cache: everything streams, the module never runs
    // (no download is chosen for a check at rate 0).
    hub.set_check_rate(Some(0.0));
    let dir_b = cache_dir("b");
    let mut b = connect(&rt, &hub, Some(&dir_b));
    let world_b = stream_home(&mut b);
    let sb = b.fetcher.stats();
    let after_b = hub.counts();
    assert_eq!(sb.worker.module_runs, 0, "{sb:?}");
    assert_eq!(sb.worker.modules_loaded, 0);
    assert_eq!(sb.worker.sections_computed, 0);
    assert_eq!(sb.directions.sections_streamed, stored.len() as u64);
    assert_eq!(after_b.missing_answers, after_a.missing_answers);
    assert_eq!(after_b.uploads_stored, after_a.uploads_stored);
    assert_eq!(
        after_b.module_reads, after_a.module_reads,
        "no module fetched"
    );
    assert_eq!(after_b.section_reads, stored.len() as u64);
    assert_eq!(matter(&world_b), cells_a, "the same matter either way");
    drop(b);

    // Third client on the second's cache: nothing leaves the machine but
    // directions.
    let mut c = connect(&rt, &hub, Some(&dir_b));
    let world_c = stream_home(&mut c);
    let sc = c.fetcher.stats();
    assert_eq!(sc.directions.sections_cached, stored.len() as u64);
    assert_eq!(sc.directions.sections_streamed, 0);
    assert_eq!(hub.counts().section_reads, after_b.section_reads);
    assert_eq!(matter(&world_c), cells_a);

    let _ = std::fs::remove_dir_all(&dir_a);
    let _ = std::fs::remove_dir_all(&dir_b);
}

#[test]
fn missing_is_computed_and_offered_with_the_section_headers() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let hub = rt.block_on(MockHub::start(space())).unwrap();
    let mut a = connect(&rt, &hub, None);
    let world = stream_home(&mut a);
    assert!(!matter(&world).is_empty());
    let n = hub.counts();
    let s = a.fetcher.stats().worker;
    assert!(n.missing_answers > 0);
    // The mock recomputes each address from `X-Layer`, `X-Key`, and
    // `X-Span-Index` and checks the body against `X-Content-Sha256`; any
    // wrong header is a refusal.
    assert_eq!(n.uploads_refused, 0, "{n:?}");
    assert_eq!(n.uploads_stored, n.missing_answers, "{n:?}");
    assert_eq!(s.uploads_offered, n.missing_answers);
    assert_eq!(
        (s.uploads_stored, s.uploads_failed, s.uploads_pending),
        (n.uploads_stored, 0, 0)
    );
}

#[test]
fn local_is_computed_and_never_offered() {
    let mut options = space();
    for l in &mut options.layers {
        if l.id == "matter" {
            l.shared = false;
        }
    }
    let rt = tokio::runtime::Runtime::new().unwrap();
    let hub = rt.block_on(MockHub::start(options)).unwrap();
    let mut a = connect(&rt, &hub, None);
    let world = stream_home(&mut a);
    assert!(!matter(&world).is_empty());
    let n = hub.counts();
    let s = a.fetcher.stats();
    assert!(n.local_answers > 0, "{n:?}");
    assert_eq!(s.directions.local_answers, n.local_answers);
    // Only the shared registry layer's section was offered and stored.
    assert_eq!(n.missing_answers, 1, "{n:?}");
    assert_eq!(s.worker.uploads_offered, 1);
    assert_eq!(n.uploads_stored, 1);
    assert_eq!(hub.sections().len(), 1);
    assert_eq!(
        s.worker.sections_computed,
        n.local_answers + n.missing_answers
    );
    drop(a);
    // A second client computes the local layer again.
    let mut b = connect(&rt, &hub, None);
    stream_home(&mut b);
    let sb = b.fetcher.stats();
    assert!(sb.worker.sections_computed > 0);
    assert_eq!(sb.worker.uploads_offered, 0);
}

#[test]
fn a_stale_cached_copy_is_discarded_on_a_hash_change() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let hub = rt.block_on(MockHub::start(space())).unwrap();
    let dir = cache_dir("stale");
    drop(connect(&rt, &hub, Some(&dir)));
    let first = hub.counts();
    // The hub replaces the frames layer's registry after a dispute: same
    // address, other bytes, so a new hash in the directions.
    let registry = common::registry_bytes();
    let frames = registry_layer("frames", &registry);
    let address = gx_core::sha256::to_hex(&gx_core::address::section_address(
        &gx_core::sha256::sha256(&frames.module),
        None,
        0,
        0,
        0,
        &gx_core::key::ChunkKey::Registry,
    ));
    assert!(hub.sections().contains_key(&address));
    let replaced = gx_core::registry::encode(&common::registry_at(Seconds::new(1.0)));
    assert_ne!(replaced, registry);
    hub.replace_section(&address, replaced.clone());
    let b = connect(&rt, &hub, Some(&dir));
    let s = b.fetcher.stats();
    // The matter layer's registry section is still read from the cache;
    // the replaced one is streamed again from the edge.
    assert_eq!(s.directions.sections_streamed, 1, "{s:?}");
    assert_eq!(s.directions.sections_cached, 1, "{s:?}");
    assert_eq!(hub.counts().section_reads, first.section_reads + 1);
    let cache = gx_worker::DiskCache::open(&dir, 1 << 30).unwrap();
    assert_eq!(
        cache.get(gx_worker::BlobKind::Section, &address).as_deref(),
        Some(&replaced[..])
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_revoked_key_counts_failures_and_never_retries() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let hub = rt.block_on(MockHub::start(space())).unwrap();
    hub.revoke();
    let mut a = connect(&rt, &hub, None);
    let world = stream_home(&mut a);
    assert!(!matter(&world).is_empty(), "drawing never waits on uploads");
    let n = hub.counts();
    let s = a.fetcher.stats().worker;
    assert_eq!(s.uploads_offered, n.missing_answers);
    assert_eq!(s.uploads_failed, s.uploads_offered, "{s:?}");
    assert_eq!(n.uploads_refused, s.uploads_offered, "one try each");
    assert!(hub.sections().is_empty());
}

#[test]
fn a_hub_that_never_answers_uploads_holds_no_compute_thread() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let hub = rt.block_on(MockHub::start(space())).unwrap();
    hub.hold_uploads();
    let mut a = connect(&rt, &hub, None);
    // Every missing section is computed on the pool's two threads while
    // the first uploads hang; with uploads on those threads the view
    // would never settle.
    let world = settle_home(&mut a);
    assert!(!matter(&world).is_empty(), "drawing never waits on uploads");
    let n = hub.counts();
    let s = a.fetcher.stats().worker;
    assert!(
        n.missing_answers > MAX_UPLOADS_IN_FLIGHT as u64,
        "enough sections to fill the upload slots: {n:?}"
    );
    assert!(s.sections_computed >= n.missing_answers, "{s:?}");
    assert_eq!(s.jobs_failed, 0, "{s:?}");
    assert_eq!(s.uploads_offered, n.missing_answers, "{s:?}");
    // The first four reach the hub and hang there; the rest wait in the
    // lane.
    let start = Instant::now();
    while hub.counts().uploads_held < MAX_UPLOADS_IN_FLIGHT as u64 {
        assert!(
            start.elapsed() < Duration::from_secs(30),
            "uploads never started"
        );
        std::thread::sleep(Duration::from_millis(10));
    }
    std::thread::sleep(Duration::from_millis(200));
    let n = hub.counts();
    assert_eq!(n.most_uploads_held, MAX_UPLOADS_IN_FLIGHT as u64, "{n:?}");
    assert_eq!(n.uploads_stored + n.uploads_refused, 0, "{n:?}");
    let s = a.fetcher.stats().worker;
    assert_eq!(
        s.uploads_pending,
        s.uploads_offered - s.uploads_dropped,
        "{s:?}"
    );
    assert_eq!(s.uploads_stored + s.uploads_failed, 0, "{s:?}");
}

#[test]
fn two_clients_at_once_both_compute_and_the_hub_keeps_the_first() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let hub = Arc::new(rt.block_on(MockHub::start(space())).unwrap());
    let rt = Arc::new(rt);
    let runs: Vec<_> = (0..2)
        .map(|_| {
            let (rt, hub) = (rt.clone(), hub.clone());
            std::thread::spawn(move || {
                let mut c = connect(&rt, &hub, None);
                let world = stream_home(&mut c);
                (matter(&world), c.fetcher.stats())
            })
        })
        .collect();
    let results: Vec<_> = runs.into_iter().map(|r| r.join().unwrap()).collect();
    let n = hub.counts();
    assert_eq!(results[0].0, results[1].0, "both see the same matter");
    let computed: u64 = results.iter().map(|r| r.1.worker.sections_computed).sum();
    let offered: u64 = results.iter().map(|r| r.1.worker.uploads_offered).sum();
    assert_eq!(computed, n.missing_answers, "{n:?}");
    assert_eq!(offered, n.missing_answers);
    assert_eq!(n.uploads_stored, hub.sections().len() as u64);
    // Deterministic modules: a second copy always matches the first.
    assert_eq!(n.uploads_stored + n.uploads_matched, offered, "{n:?}");
    assert_eq!((n.uploads_disputed, n.uploads_refused), (0, 0));
}

#[test]
fn directions_are_batched_and_empty_cells_need_nothing() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let hub = rt.block_on(MockHub::start(space())).unwrap();
    let mut c = connect(&rt, &hub, None);
    let before = hub.counts();
    // 600 cells deeper than any layer fills: every layer is empty there.
    let extent = c.system.tree().get(common::ROOT).unwrap().root_extent;
    let keys: Vec<CellKey> = (0..600)
        .map(|i| CellKey::new(common::ROOT, 5, i % 32, (i / 32) % 32, i / 1024).unwrap())
        .collect();
    for k in &keys {
        c.fetcher.request(*k, extent);
    }
    // Asking twice for a key in flight sends nothing more.
    c.fetcher.request(keys[0], extent);
    let mut done = 0;
    let start = Instant::now();
    while done < keys.len() {
        for e in c.fetcher.wait(Duration::from_millis(50)) {
            if let FetchEvent::Completed { outcome, .. } = e {
                assert_eq!(outcome, renderer::stream::FetchOutcome::Decoded(None));
                done += 1;
            }
        }
        assert!(start.elapsed() < Duration::from_secs(30));
    }
    let after = hub.counts();
    assert_eq!(after.directions_requests - before.directions_requests, 2);
    assert_eq!(after.largest_batch, 512);
    assert_eq!(after.missing_answers, before.missing_answers);
    assert_eq!(
        c.fetcher.stats().directions.keys_asked,
        601,
        "600 cells and the registry"
    );
}

#[test]
fn headless_render_at_now_and_at_the_origin() {
    // A space whose origin is two days ago, with the registry at the
    // origin, as the system module emits it (`matter-format.md` 5.2).
    let origin_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as i64
        - 2 * 86_400_000;
    let epoch = Seconds::new(utc_ms_to_tdb_seconds(origin_ms));
    let registry = gx_core::registry::encode(&common::registry_at(epoch));
    let mut options = space_with_registry(registry);
    options.origin_utc_ms = Some(origin_ms);
    let rt = tokio::runtime::Runtime::new().unwrap();
    let hub = rt.block_on(MockHub::start(options)).unwrap();
    let dir = cache_dir("render");
    let mut c = connect(&rt, &hub, Some(&dir));
    assert_eq!(c.system.tree().epoch(), epoch);
    assert_eq!(c.fetcher.origin_utc_ms(), Some(origin_ms));
    assert!(check_origin(&c.system, c.fetcher.origin_utc_ms()));
    assert!(!check_origin(&c.system, Some(origin_ms + 60_000)));
    let run = HeadlessRun {
        overlay: false,
        wait_ready_seconds: Some(90.0),
        view: View::Home,
        ..HeadlessRun::home(WIDTH, HEIGHT)
    };
    for (name, at_origin) in [("now", false), ("origin", true)] {
        let now = Seconds::new(utc_ms_to_tdb_seconds(c.fetcher.now_utc_ms()));
        let clock = if at_origin {
            SimClock::starting_at(epoch, now, epoch, 1.0)
        } else {
            SimClock::at_now(epoch, now)
        };
        let mut sim = Simulation::new(c.system.clone(), clock);
        if at_origin {
            assert_eq!(sim.time(), epoch);
        } else {
            assert!(sim.clock().is_at_now());
            assert!(sim.time() >= epoch + Seconds::new(2.0 * 86_400.0 - 60.0));
        }
        let frame = render_headless_view(&mut sim, Some(&mut c.fetcher), &run)
            .expect("a wgpu adapter, software is fine");
        assert!(frame.stats.ready, "{name}: every selected cell arrived");
        let m = &frame.stats.overlay.matter;
        assert!(m.modules_loaded == 2, "{name}: {m:?}");
        assert!(m.volumes_drawn > 0, "{name}: {m:?}");
        let lit = frame
            .image
            .rgba
            .chunks(4)
            .filter(|p| p[0] > 8 || p[1] > 8 || p[2] > 8)
            .count();
        assert!(lit > 100, "{name}: {lit} lit pixels");
        let out = std::path::Path::new(env!("CARGO_TARGET_TMPDIR"))
            .join(format!("mock-hub-home-{name}.png"));
        frame.image.write_png(&out).unwrap();
        let json = frame.stats.to_json();
        assert_eq!(json["modules_loaded"], 2);
        assert_eq!(json["at_now"], !at_origin);
        assert!(m.directions_requests > 0);
        // The budget statistics (`rendering.md` section 5.3): the view
        // settled, and the frame steps and selections were counted.
        let settle = json["settle_seconds"].as_f64().expect("settled");
        assert!(settle > 0.0 && settle <= json["stream_seconds"].as_f64().unwrap() + 1.0);
        assert!(json["frame_steps"].as_u64().unwrap() > 0);
        let p50 = json["frame_step_ms_p50"].as_f64().unwrap();
        let p95 = json["frame_step_ms_p95"].as_f64().unwrap();
        let max = json["frame_step_ms_max"].as_f64().unwrap();
        assert!(0.0 < p50 && p50 <= p95 && p95 <= max, "{p50} {p95} {max}");
        assert!(json["selection_rounds"].as_u64().unwrap() > 0);
        assert!(json["cells_requested_ahead"].as_u64().is_some());
    }
    assert!(c.fetcher.stats().worker.sections_computed > 0);
    let _ = std::fs::remove_dir_all(&dir);
}

/// Waits until every queued check has run and been reported.
fn wait_for_checks(c: &mut Client) {
    let start = Instant::now();
    loop {
        let w = c.fetcher.stats().worker;
        if w.checks_matched + w.checks_disputed + w.checks_failed >= w.checks_queued {
            return;
        }
        assert!(
            start.elapsed() < Duration::from_secs(60),
            "checks hang: {w:?}"
        );
        c.fetcher.wait_for_events(Duration::from_millis(20));
    }
}

#[test]
fn rate_zero_never_checks_and_rate_one_checks_every_download() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let hub = rt.block_on(MockHub::start(space())).unwrap();
    drop(stream_home(&mut connect(&rt, &hub, None)));
    let stored = hub.sections().len() as u64;

    hub.set_check_rate(Some(0.0));
    let mut a = connect(&rt, &hub, None);
    stream_home(&mut a);
    let s = a.fetcher.stats();
    assert!(s.directions.sections_streamed > 1, "{s:?}");
    assert_eq!(s.worker.checks_queued + s.worker.checks_skipped, 0, "{s:?}");
    assert_eq!(s.worker.module_runs, 0);
    drop(a);

    hub.set_check_rate(Some(1.0));
    let mut b = connect(&rt, &hub, None);
    let world = stream_home(&mut b);
    wait_for_checks(&mut b);
    let s = b.fetcher.stats();
    // Every downloaded cell section is checked (the registry key is not).
    let cells = s.directions.sections_streamed - 2;
    assert!(cells >= 1 && stored >= cells, "{s:?}");
    assert_eq!(s.worker.checks_queued, cells, "{s:?}");
    assert_eq!(s.worker.checks_run, cells, "{s:?}");
    assert_eq!(s.worker.checks_matched, cells, "{s:?}");
    assert_eq!(s.worker.checks_disputed + s.worker.checks_failed, 0);
    assert_eq!(hub.counts().checks_matched, cells);
    assert!(!matter(&world).is_empty());
}

#[test]
fn a_lying_section_is_replaced_by_the_computed_one_and_disputed() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let hub = rt.block_on(MockHub::start(space())).unwrap();
    let truth = matter(&stream_home(&mut connect(&rt, &hub, None)));
    // Replace one stored cell section with valid bytes that differ: the
    // same section with every density doubled.
    let (address, lie) = hub
        .sections()
        .into_iter()
        .find_map(|(a, bytes)| {
            let (k, s) = truth
                .keys()
                .find_map(|k| gx_core::matter::decode(k, &bytes).ok().map(|s| (*k, s)))?;
            let mut doubled = s.samples()?.clone();
            for i in 0..doubled.len() {
                let mut x = doubled.get(i)?;
                x.density = gx_core::units::Density::new(x.density.value() * 2.0);
                doubled.set(i, x);
            }
            let lie =
                gx_core::matter::Section::new(k, s.origin(), s.edge(), s.resolution(), doubled)
                    .ok()?;
            let lie = gx_core::matter::encode(&lie, gx_core::matter::Compression::None);
            (lie != bytes).then_some((a, lie))
        })
        .expect("a stored cell section");
    hub.replace_section(&address, lie);

    hub.set_check_rate(Some(1.0));
    let mut b = connect(&rt, &hub, None);
    stream_home(&mut b);
    wait_for_checks(&mut b);
    let s = b.fetcher.stats();
    assert_eq!(s.worker.checks_disputed, 1, "{s:?}");
    assert_eq!(hub.counts().checks_disputed, 1);
    // The replaced key completes again with the computed section.
    let mut world = settle_home(&mut b);
    let start = Instant::now();
    while matter(&world) != truth {
        assert!(
            start.elapsed() < Duration::from_secs(30),
            "the lie stayed drawn"
        );
        let mut camera = Camera::home(&b.system);
        camera.update_parent(&b.system);
        let seen = Seen::instant(&b.system, &camera);
        matter_step(
            &mut world,
            &mut b.fetcher,
            &seen,
            (WIDTH, HEIGHT),
            start.elapsed().as_secs_f64(),
        );
        b.fetcher.wait_for_events(Duration::from_millis(20));
    }
}

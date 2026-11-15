//! The client against the v2 mock hub, with the core library's example
//! module as the only compiler (`client-compute.md` sections 1 to 6,
//! `space-model.md` section 11's first two acceptance points):
//!
//! - a first client with an empty cache computes and uploads the registry
//!   and every section it needs;
//! - a second client with a fresh cache streams every one of them and runs
//!   the module zero times, ending with the same matter;
//! - a third client reusing that cache reads everything locally;
//! - `pending` claims of another client are waited out (honoring
//!   `retryMs`) and then computed; two clients at once compute each
//!   section once between them; directions go out in batches of at most
//!   512 keys;
//! - a headless render against the mock hub produces a non-empty image,
//!   saved as `target/tmp/mock-hub-home.png`.
//!
//! The space has two layers: `frames`, a registry-only module declaring the
//! two frames of the shared scene (`common`), and `matter`, the example
//! module on every frame up to depth 1 with its source tiles.

mod common;

use gx_core::frames::FrameSystem;
use gx_core::key::CellKey;
use renderer::app::{render_headless_view, stream_step, HeadlessRun};
use renderer::camera::Camera;
use renderer::config::{ApiKey, View};
use renderer::extract::default_threads;
use renderer::hub::HubClient;
use renderer::mock_hub::{
    example_matter_layer, registry_layer, MockHub, MockHubOptions, MOCK_API_KEY, MOCK_SPACE_ID,
};
use renderer::protocol::Routes;
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

fn space(foreign_claim_ms: Option<u64>) -> MockHubOptions {
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
    let mut options = MockHubOptions::new(
        vec![registry_layer("frames", &common::registry_bytes()), matter],
        tiles,
    );
    options.foreign_claim_ms = foreign_claim_ms;
    options
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
    let mut camera = Camera::home(&c.system);
    camera.update_parent(&c.system);
    let mut world = World::new(default_threads());
    let start = Instant::now();
    world.select(&c.system, &camera, (WIDTH, HEIGHT), 0.0, true);
    loop {
        let now = start.elapsed().as_secs_f64();
        let events = c.fetcher.wait(Duration::from_millis(20));
        stream_step(&mut world, &mut c.fetcher, &c.system, events, now);
        if world.settled() {
            break;
        }
        assert!(now < 90.0, "the view did not settle");
    }
    let start = Instant::now();
    while c.fetcher.stats().worker.uploads_pending > 0 {
        assert!(start.elapsed() < Duration::from_secs(30), "uploads hang");
        std::thread::sleep(Duration::from_millis(10));
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
    let hub = rt.block_on(MockHub::start(space(None))).unwrap();
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
    assert!(after_a.compute_answers >= 3, "{after_a:?}");
    assert_eq!(
        after_a.uploads_stored, after_a.compute_answers,
        "{after_a:?}"
    );
    assert_eq!(after_a.uploads_refused, 0);
    assert_eq!(sa.worker.sections_computed, after_a.compute_answers);
    assert_eq!(sa.worker.uploads_done, after_a.compute_answers);
    assert_eq!(sa.worker.uploads_failed, 0);
    assert_eq!(
        sa.worker.modules_loaded, 2,
        "the registry module and the example module"
    );
    assert_eq!(sa.worker.module_runs, 2 * after_a.compute_answers);
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

    // Second client, fresh cache: everything streams, the module never runs.
    let dir_b = cache_dir("b");
    let mut b = connect(&rt, &hub, Some(&dir_b));
    let world_b = stream_home(&mut b);
    let sb = b.fetcher.stats();
    let after_b = hub.counts();
    assert_eq!(sb.worker.module_runs, 0, "{sb:?}");
    assert_eq!(sb.worker.modules_loaded, 0);
    assert_eq!(sb.worker.sections_computed, 0);
    assert_eq!(sb.directions.sections_streamed, stored.len() as u64);
    assert_eq!(after_b.compute_answers, after_a.compute_answers);
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
fn pending_claims_are_waited_out_then_computed() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    // Every address is first held by another client for 300 ms.
    let hub = rt.block_on(MockHub::start(space(Some(300)))).unwrap();
    let mut a = connect(&rt, &hub, None);
    let world = stream_home(&mut a);
    assert!(!matter(&world).is_empty());
    let n = hub.counts();
    let s = a.fetcher.stats();
    assert!(n.pending_answers >= n.compute_answers, "{n:?}");
    assert!(s.directions.pending_answers > 0);
    assert_eq!(n.uploads_stored, n.compute_answers);
    assert_eq!(s.worker.sections_computed, n.compute_answers);
}

#[test]
fn two_clients_at_once_compute_each_section_once() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let hub = Arc::new(rt.block_on(MockHub::start(space(None))).unwrap());
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
    assert_eq!(computed, n.compute_answers, "{n:?}");
    assert_eq!(n.uploads_stored, hub.sections().len() as u64);
    assert_eq!(
        n.uploads_stored, n.compute_answers,
        "each section computed once"
    );
    assert_eq!(n.uploads_refused, 0);
}

#[test]
fn directions_are_batched_and_empty_cells_need_nothing() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let hub = rt.block_on(MockHub::start(space(None))).unwrap();
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
    assert_eq!(after.compute_answers, before.compute_answers);
    assert_eq!(
        c.fetcher.stats().directions.keys_asked,
        601,
        "600 cells and the registry"
    );
}

#[test]
fn headless_render_against_the_mock_hub_is_not_empty() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let hub = rt.block_on(MockHub::start(space(None))).unwrap();
    let dir = cache_dir("render");
    let mut c = connect(&rt, &hub, Some(&dir));
    let mut sim = Simulation::new(
        c.system.clone(),
        SimClock::new(gx_core::units::Seconds::new(0.0), 1.0),
    );
    let run = HeadlessRun {
        overlay: false,
        wait_ready_seconds: Some(90.0),
        view: View::Home,
        ..HeadlessRun::home(WIDTH, HEIGHT)
    };
    let frame = render_headless_view(&mut sim, Some(&mut c.fetcher), &run)
        .expect("a wgpu adapter, software is fine");
    assert!(frame.stats.ready, "every selected cell arrived");
    let m = &frame.stats.overlay.matter;
    assert!(m.sections_computed > 0 && m.modules_loaded == 2, "{m:?}");
    assert!(m.volumes_drawn > 0, "{m:?}");
    let lit = frame
        .image
        .rgba
        .chunks(4)
        .filter(|p| p[0] > 8 || p[1] > 8 || p[2] > 8)
        .count();
    assert!(lit > 100, "{lit} lit pixels");
    let out = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("mock-hub-home.png");
    frame.image.write_png(&out).unwrap();
    let json = frame.stats.to_json();
    assert_eq!(json["sections_computed"], m.sections_computed);
    assert_eq!(json["modules_loaded"], 2);
    let _ = std::fs::remove_dir_all(&dir);
}

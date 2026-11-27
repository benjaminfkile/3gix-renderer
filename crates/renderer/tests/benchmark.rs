//! The benchmark start picked by rule (`rendering.md` section 5.1,
//! [`renderer::benchmark`]), against made-up universes served by the mock
//! hub.
//!
//! The first universe has a root without a surface, frame 2 without one,
//! and frames 3 and 5 with flat surfaces carrying a few raised vertices:
//! frame 3 has two level-6 tiles raised alike (a tie), a lower one, and a
//! higher raise at level 5 that the rule must not see; frame 5, a higher id,
//! has the roughest tile of all. The pick is frame 3's tied tile with the
//! smaller key; the rule does not depend on arrival order
//! (`renderer::benchmark` unit tests). The second universe has no surface
//! at all and fails with the message.

#[path = "common/universe.rs"]
mod universe;

use gx_core::key::SurfaceKey;
use gx_core::matter::Compression;
use gx_core::registry::ROOT_PARENT;
use gx_core::surface::{vertex_direction, SurfaceTile, HEIGHT_SIDE, PARENT_SIDE, VERTEX_COUNT};
use renderer::benchmark::{benchmark_start, BENCHMARK_LEVEL, NO_SURFACE};
use renderer::config::ApiKey;
use renderer::hub::HubClient;
use renderer::mock_hub::{MockHub, MockHubOptions, SurfaceMaker, MOCK_API_KEY, MOCK_SPACE_ID};
use renderer::protocol::Routes;
use renderer::stream::{Fetcher, FetcherOptions};
use std::sync::Arc;
use std::time::{Duration, Instant};
use universe::{layers_of, rolling, Body, SurfaceSpec};

/// The reference radii of every surface here, meters: made up.
const RADII: (f64, f64) = (3.1e5, 2.9e5);

fn key(frame_id: u64, face: u8, level: u8, x: u32, y: u32) -> SurfaceKey {
    SurfaceKey::new(frame_id, face, level, x, y).unwrap()
}

/// The raised vertex (10, 12) of each listed tile and its height, meters;
/// every other height is zero.
fn raised() -> Vec<(SurfaceKey, f32)> {
    vec![
        (key(3, 2, BENCHMARK_LEVEL, 10, 20), 500.0),
        (key(3, 1, BENCHMARK_LEVEL, 40, 7), 500.0),
        (key(3, 0, BENCHMARK_LEVEL, 1, 1), 300.0),
        (key(3, 0, BENCHMARK_LEVEL - 1, 3, 3), 9000.0),
        (key(5, 0, BENCHMARK_LEVEL, 0, 0), 2000.0),
    ]
}

/// A flat tile of `key`, with vertex (10, 12) at `peak` meters.
fn tile(key: &SurfaceKey, peak: f32) -> Vec<u8> {
    let mut heights = vec![0.0f32; HEIGHT_SIDE * HEIGHT_SIDE];
    heights[11 + HEIGHT_SIDE * 13] = peak;
    let t = SurfaceTile {
        key: *key,
        radius_xy: RADII.0,
        radius_z: RADII.1,
        height_floor: 0.0,
        heights,
        parent_heights: vec![0.0; PARENT_SIDE * PARENT_SIDE],
        albedo: vec![[120, 120, 120]; VERTEX_COUNT],
        roughness: vec![200; VERTEX_COUNT],
        temperature: vec![250.0; VERTEX_COUNT],
    };
    gx_core::surface::encode(&t, Compression::None)
}

fn spec() -> SurfaceSpec {
    SurfaceSpec {
        radius_xy: RADII.0,
        radius_z: RADII.1,
        height: rolling(0.0),
        max_height: 0.0,
        max_level: BENCHMARK_LEVEL,
        temperature: 250.0,
    }
}

/// A root, frame 2 without a surface, and frames 3 and 5 with one when
/// `surfaces`.
fn bodies(surfaces: bool) -> Vec<Body> {
    let mut out = vec![Body::new(1, ROOT_PARENT, 1.0e9, 0.0, [0.0; 3])];
    for (id, x) in [(2, 1.0e7), (3, 2.0e7), (5, 3.0e7)] {
        let mut b = Body::new(id, 1, 1.0e6, 1.0e3, [x, 0.0, 0.0]);
        if surfaces && id != 2 {
            b.surface = Some(spec());
        }
        out.push(b);
    }
    out
}

/// Serves `bodies` (with the raised tiles when they have surfaces) and
/// runs the pick on a fresh client.
fn pick(bodies: &[Body]) -> anyhow::Result<String> {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut layers = layers_of(bodies);
    let with_surface: Vec<u64> = (bodies.iter())
        .filter(|b| b.surface.is_some())
        .map(|b| b.id)
        .collect();
    let main = layers.last_mut().unwrap();
    main.stored_surfaces = Some(SurfaceMaker(Arc::new(move |k: &SurfaceKey| {
        if !with_surface.contains(&k.frame_id) {
            return gx_core::surface::encode_empty(k);
        }
        let peak = raised()
            .into_iter()
            .find(|(r, _)| r == k)
            .map_or(0.0, |(_, h)| h);
        tile(k, peak)
    })));
    let hub = rt
        .block_on(MockHub::start(MockHubOptions::new(layers, Vec::new())))
        .unwrap();
    let client = rt
        .block_on(HubClient::open(
            Routes::new(&hub.url(), None, MOCK_SPACE_ID),
            ApiKey::new(MOCK_API_KEY),
            None,
        ))
        .expect("the mock hub answers");
    let mut fetcher = Fetcher::new(
        rt.handle().clone(),
        Arc::new(client),
        FetcherOptions {
            cache: None,
            worker_threads: 1,
            ..FetcherOptions::default()
        },
    )
    .unwrap();
    let system = fetcher.load_registry(Duration::from_secs(60)).unwrap();
    // Looser than the tool's own deadline: a debug build on a busy machine
    // takes about a minute for one frame's 24576 tiles.
    benchmark_start(
        &mut fetcher,
        &system,
        Instant::now() + Duration::from_secs(600),
    )
}

#[test]
fn the_pick_is_the_roughest_tile_of_the_lowest_frame_with_a_surface() {
    let tied = key(3, 1, BENCHMARK_LEVEL, 40, 7);
    let d = vertex_direction(&tied, 32, 32);
    let expected = format!("3:{:.9},{:.9},{:.9}", d[0], d[1], d[2]);
    assert_eq!(pick(&bodies(true)).unwrap(), expected);
}

#[test]
fn a_universe_without_a_surface_fails_with_the_message() {
    let e = pick(&bodies(false)).unwrap_err();
    assert!(format!("{e:#}").contains(NO_SURFACE), "{e:#}");
}

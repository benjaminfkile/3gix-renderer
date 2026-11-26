//! Surface overlays against the mock hub (`rendering.md` section 7 items 6
//! and 7, `matter-format.md` section 12, `client-compute.md` section 2.2).
//!
//! The space has the shared scene's registry layer (`frames`), a layer
//! whose surface tiles are stored on the hub (`main`), the core
//! library's example overlay module (`paint`, an overlay layer computed by
//! the client), and a second overlay layer stored on the hub (`stain`).
//! The tile of the overlay module's frame is drawn as soon as its base
//! arrives, with the overlays it has; the overlays that arrive later
//! update its material, in layer order, and never its geometry. A hidden
//! overlay is neither fetched nor composited; once shown it is fetched at
//! once for the tiles drawn and drawn on them.

mod common;

use gx_core::key::SurfaceKey;
use gx_core::matter::Compression;
use gx_core::overlay::{self, OverlayTile, PARENT_COUNT};
use gx_core::surface::{SurfaceTile, HEIGHT_SIDE, PARENT_SIDE, VERTEX_COUNT};
use renderer::config::ApiKey;
use renderer::hub::HubClient;
use renderer::mock_hub::{
    registry_layer, stored_layer, MockHub, MockHubOptions, MockLayer, SectionMaker, SurfaceMaker,
    MOCK_API_KEY, MOCK_SPACE_ID,
};
use renderer::protocol::{Attachment, Routes};
use renderer::stream::{FetchEvent, Fetcher, FetcherOptions};
use renderer::tiles::{composite_tile, DecodedTile, Surfaces};
use std::sync::Arc;
use std::time::{Duration, Instant};

const OVERLAY_MODULE: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../testdata/example-module-overlay/module.wasm"
));

/// The frame the example overlay module paints.
const PAINTED: u64 = 7;

/// A flat tile of `key` with a uniform material.
fn base_tile(key: &SurfaceKey) -> Vec<u8> {
    let tile = SurfaceTile {
        key: *key,
        radius_xy: 4.6e6,
        radius_z: 4.6e6,
        height_floor: 0.0,
        heights: vec![0.0; HEIGHT_SIDE * HEIGHT_SIDE],
        parent_heights: vec![0.0; PARENT_SIDE * PARENT_SIDE],
        albedo: vec![[40, 90, 160]; VERTEX_COUNT],
        roughness: vec![200; VERTEX_COUNT],
        temperature: vec![290.0; VERTEX_COUNT],
    };
    gx_core::surface::encode(&tile, Compression::None)
}

/// Half coverage of a dark material over the first half of the vertices:
/// partial coverage in value and in extent.
fn stain_tile(key: &SurfaceKey) -> OverlayTile {
    let coverage = (0..VERTEX_COUNT)
        .map(|v| if v < VERTEX_COUNT / 2 { 128 } else { 0 })
        .collect();
    OverlayTile {
        key: *key,
        coverage,
        albedo: vec![[10, 10, 10]; VERTEX_COUNT],
        roughness: vec![30; VERTEX_COUNT],
        parent_coverage: vec![128; PARENT_COUNT],
        parent_albedo: vec![[10, 10, 10]; PARENT_COUNT],
        parent_roughness: vec![30; PARENT_COUNT],
    }
}

fn space(stain_shown: bool) -> MockHubOptions {
    MockHubOptions::new(space_layers(stain_shown), Vec::new())
}

/// The layers of [`space`]: the registry, the `main` layer, and the
/// `paint` and `stain` overlays.
fn space_layers(stain_shown: bool) -> Vec<MockLayer> {
    let mut base = stored_layer("main", 4, SectionMaker(Arc::new(|_| Vec::new())));
    base.stored_surfaces = Some(SurfaceMaker(Arc::new(base_tile)));
    let paint = MockLayer {
        id: "paint".into(),
        module: OVERLAY_MODULE.to_vec(),
        manifest: None,
        attachment: Attachment::All,
        exclusive: false,
        overlay: true,
        shown: true,
        stored_surfaces: None,
        max_depth: [("*".to_string(), 4)].into(),
        span_seconds: 0,
        uses_origin: false,
        stored: None,
        shared: true,
        inputs: Vec::new(),
    };
    let mut stain = stored_layer("stain", 4, SectionMaker(Arc::new(|_| Vec::new())));
    stain.overlay = true;
    // A custom section (`stain`) makes the module, and so every address of
    // the layer, differ from `main`'s.
    stain
        .module
        .extend_from_slice(&[0, 6, 5, b's', b't', b'a', b'i', b'n']);
    stain.shown = stain_shown;
    stain.stored_surfaces = Some(SurfaceMaker(Arc::new(|k| {
        overlay::encode(&stain_tile(k), Compression::None)
    })));
    let layers = vec![
        registry_layer("frames", &common::registry_bytes()),
        base,
        paint,
        stain,
    ];
    layers
}

fn connect(rt: &tokio::runtime::Runtime, hub: &MockHub) -> Fetcher {
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
            worker_threads: 2,
            ..FetcherOptions::default()
        },
    )
    .unwrap();
    fetcher.load_registry(Duration::from_secs(60)).unwrap();
    fetcher
}

/// Requests `key` and applies every completion to `surfaces` until the
/// last one, checking each step: the tile is drawn from its first
/// completion on, and its geometry never changes. Returns each
/// completion's overlay layers.
fn fetch(fetcher: &mut Fetcher, surfaces: &mut Surfaces, key: SurfaceKey) -> Vec<Vec<usize>> {
    fetcher.request_surface(key);
    let start = Instant::now();
    let mut steps = Vec::new();
    let mut positions = None;
    loop {
        assert!(start.elapsed() < Duration::from_secs(120), "the tile hangs");
        for e in fetcher.wait(Duration::from_millis(50)) {
            let FetchEvent::Surface {
                key: k,
                outcome,
                partial,
                ..
            } = e
            else {
                continue;
            };
            // A late completion of the tile fetched before this one may
            // still arrive (the fetcher decodes each on its own thread);
            // it is not this tile's.
            if k != key {
                continue;
            }
            let tile = outcome.expect("a valid tile");
            assert!(tile.base().is_some(), "the base came first");
            steps.push(tile.overlays.iter().map(|(l, _)| *l).collect());
            surfaces.complete_layers(key, Ok(tile), partial);
            let drawn = surfaces.drawn_geometry(&key).expect("drawn at once");
            let p = positions.get_or_insert_with(|| drawn.positions.clone());
            assert_eq!(&drawn.positions, p, "an overlay changes material only");
            if !partial {
                return steps;
            }
        }
    }
}

/// The base and the two overlays of `key`, made as the hub makes them.
fn expected(key: &SurfaceKey) -> (renderer::tiles::TileGeometry, OverlayTile, OverlayTile) {
    let decoded: DecodedTile =
        renderer::tiles::decode_tile(key, &[base_tile(key)], &[1], &Default::default()).unwrap();
    let base = Arc::unwrap_or_clone(decoded.base().cloned().unwrap());
    let paint = gx_module_overlay_tile(key);
    (base, paint, stain_tile(key))
}

/// The example overlay module's tile of `key`, decoded from the module's
/// own output through the client.
fn gx_module_overlay_tile(key: &SurfaceKey) -> OverlayTile {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let hub = rt.block_on(MockHub::start(space(true))).unwrap();
    let mut fetcher = connect(&rt, &hub);
    fetcher.request_surface(*key);
    let start = Instant::now();
    loop {
        assert!(start.elapsed() < Duration::from_secs(120), "the tile hangs");
        for e in fetcher.wait(Duration::from_millis(50)) {
            if let FetchEvent::Surface {
                outcome,
                partial: false,
                ..
            } = e
            {
                let tile = outcome.unwrap();
                let (_, o) = tile.overlays.into_iter().find(|(l, _)| *l == 2).unwrap();
                return Arc::unwrap_or_clone(o);
            }
        }
    }
}

#[test]
fn overlays_draw_in_layer_order_as_they_arrive() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let hub = rt.block_on(MockHub::start(space(true))).unwrap();
    let mut fetcher = connect(&rt, &hub);
    let mut surfaces = Surfaces::new();

    // A frame the module does not paint: only the stain.
    let plain = SurfaceKey::new(1, 0, 0, 0, 0).unwrap();
    let steps = fetch(&mut fetcher, &mut surfaces, plain);
    assert_eq!(steps.last().unwrap(), &vec![3]);

    // The painted frame: both overlays in the end, in layer order, and
    // the material is the base with the paint and then the stain over it.
    let key = SurfaceKey::new(PAINTED, 2, 0, 0, 0).unwrap();
    let steps = fetch(&mut fetcher, &mut surfaces, key);
    assert_eq!(steps.last().unwrap(), &vec![2, 3], "{steps:?}");
    let (base, paint, stain) = expected(&key);
    let drawn = Arc::clone(surfaces.drawn_geometry(&key).unwrap());
    let want = composite_tile(&base, &[&paint, &stain]);
    assert_eq!(drawn.albedo, want.albedo);
    assert_eq!(drawn.roughness, want.roughness);
    // The paint covers some vertices and not others; the stain covers
    // half of them: the material differs from the base where covered.
    assert_ne!(drawn.albedo, base.albedo);
    let last = VERTEX_COUNT - 1;
    let paint_only = composite_tile(&base, &[&paint]);
    assert_eq!(drawn.albedo[last], paint_only.albedo[last]);

    // Hiding both overlays restores the base material; showing the paint
    // again draws it alone.
    surfaces.set_hidden_layers([2, 3].into());
    let hidden = surfaces.drawn_geometry(&key).unwrap();
    assert_eq!(hidden.albedo, base.albedo);
    assert_eq!(hidden.roughness, base.roughness);
    surfaces.set_hidden_layers([3].into());
    assert_eq!(
        surfaces.drawn_geometry(&key).unwrap().albedo,
        paint_only.albedo
    );
}

#[test]
fn a_hidden_base_layer_gives_way_to_the_next() {
    // A second plain layer with a surface: `second`, layer 3 (the build
    // orders layers by id).
    let mut layers = space_layers(true);
    let mut second = stored_layer("second", 4, SectionMaker(Arc::new(|_| Vec::new())));
    second
        .module
        .extend_from_slice(&[0, 5, 4, b'a', b'l', b't', b'2']);
    second.stored_surfaces = Some(SurfaceMaker(Arc::new(base_tile)));
    layers.push(second);
    let rt = tokio::runtime::Runtime::new().unwrap();
    let hub = rt
        .block_on(MockHub::start(MockHubOptions::new(layers, Vec::new())))
        .unwrap();
    let mut fetcher = connect(&rt, &hub);
    let mut surfaces = Surfaces::new();
    let mut left = std::collections::BTreeSet::new();
    for face in 0..6 {
        let key = SurfaceKey::new(PAINTED, face, 0, 0, 0).unwrap();
        fetcher.request_surface(key);
        left.insert(key);
    }
    let start = Instant::now();
    while !left.is_empty() {
        assert!(start.elapsed() < Duration::from_secs(120), "the tiles hang");
        for e in fetcher.wait(Duration::from_millis(50)) {
            if let FetchEvent::Surface {
                key,
                outcome,
                partial,
                ..
            } = e
            {
                surfaces.complete_layers(key, outcome, partial);
                if !partial {
                    left.remove(&key);
                }
            }
        }
    }
    let key = SurfaceKey::new(PAINTED, 2, 0, 0, 0).unwrap();
    assert!(surfaces.has_surface(PAINTED));
    let shown = Arc::clone(surfaces.drawn_geometry(&key).unwrap());

    // Hiding `main` draws `second`'s tile, with the overlays over it.
    surfaces.set_hidden_layers([1].into());
    let second = Arc::clone(surfaces.drawn_geometry(&key).unwrap());
    assert!(!Arc::ptr_eq(&second, &shown));
    assert_eq!(second.positions, shown.positions);
    assert!(surfaces.has_surface(PAINTED));

    // Hiding both bases draws no tile: the frame has no surface, and its
    // cells are drawn instead.
    surfaces.set_hidden_layers([1, 3].into());
    assert!(surfaces.drawn_geometry(&key).is_none());
    assert!(!surfaces.has_surface(PAINTED));

    // Showing them again draws `main` with its overlays as before.
    surfaces.set_hidden_layers(Default::default());
    let again = surfaces.drawn_geometry(&key).unwrap();
    assert_eq!(again.albedo, shown.albedo);
    assert_eq!(again.positions, shown.positions);
    assert!(surfaces.has_surface(PAINTED));
}

#[test]
fn a_hidden_overlay_is_not_fetched_until_shown() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let hub = rt.block_on(MockHub::start(space(false))).unwrap();
    let mut fetcher = connect(&rt, &hub);
    let mut surfaces = Surfaces::new();
    surfaces.set_hidden_layers([3].into());
    let key = SurfaceKey::new(PAINTED, 2, 0, 0, 0).unwrap();
    // A tile the selection draws, wanted at its priority.
    fetcher.set_wanted_surfaces(vec![(key, 100.0)]);
    let run = |fetcher: &mut Fetcher, surfaces: &mut Surfaces| {
        let start = Instant::now();
        loop {
            assert!(start.elapsed() < Duration::from_secs(120), "the tile hangs");
            for e in fetcher.wait(Duration::from_millis(50)) {
                if let FetchEvent::Surface {
                    key: k,
                    outcome,
                    partial,
                    ..
                } = e
                {
                    let tile = outcome.expect("a valid tile");
                    let layers: Vec<usize> = tile.overlays.iter().map(|(l, _)| *l).collect();
                    surfaces.complete_layers(k, Ok(tile), partial);
                    if !partial {
                        return layers;
                    }
                }
            }
        }
    };
    // Not fetched while hidden.
    assert_eq!(run(&mut fetcher, &mut surfaces), vec![2]);
    let before = Arc::clone(surfaces.drawn_geometry(&key).unwrap());
    // Shown: fetched at once for the drawn key, and drawn on the tile
    // already on screen.
    fetcher.set_hidden_layers(Default::default());
    surfaces.set_hidden_layers(Default::default());
    assert_eq!(run(&mut fetcher, &mut surfaces), vec![2, 3]);
    let after = surfaces.drawn_geometry(&key).unwrap();
    assert_eq!(after.positions, before.positions);
    assert_ne!(after.albedo, before.albedo, "the stain is drawn");
}

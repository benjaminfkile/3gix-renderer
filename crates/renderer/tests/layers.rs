//! Hidden plain layers in cells against the mock hub (`rendering.md`
//! section 7 item 7).
//!
//! The space has a registry layer and two plain layers stored on the hub:
//! `low`, a dark slab below the frame's middle plane, and `high`, a light
//! slab reaching higher. Each fetched cell keeps both layers' sections, so
//! hiding `high` composites the drawn cells again from `low` alone and
//! showing it restores the first composite, through the same
//! [`matter_step`] every build uses, with nothing fetched again.

use gx_core::frames::FrameSystem;
use gx_core::key::CellKey;
use gx_core::matter::{encode, Compression, Sample, Samples, Section, State};
use gx_core::registry::{self, Frame, FrameTree, Registry, ROOT_PARENT};
use gx_core::units::{Attenuation, Density, Kelvin, Kilograms, Meters, Quat, Ratio, Seconds, Vec3};
use renderer::app::matter_step;
use renderer::camera::{look_rotation, Camera};
use renderer::config::ApiKey;
use renderer::hub::HubClient;
use renderer::mock_hub::{
    registry_layer, stored_layer, MockHub, MockHubOptions, SectionMaker, MOCK_API_KEY,
    MOCK_SPACE_ID,
};
use renderer::protocol::Routes;
use renderer::seen::Seen;
use renderer::stream::{Fetcher, FetcherOptions};
use renderer::world::World;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// The one frame.
const FRAME: u64 = 3;

/// Its cube edge, meters.
const EXTENT: f64 = 200.0;

/// Samples per axis of every section.
const RESOLUTION: u8 = 8;

fn registry_bytes() -> Vec<u8> {
    let frame = Frame {
        frame_id: FRAME,
        parent_frame_id: ROOT_PARENT,
        root_extent: Meters::new(EXTENT),
        max_depth: 0,
        mass: Kilograms::new(1.0e12),
        position: Vec3::zero(),
        velocity: Vec3::zero(),
        orientation: Quat::identity(),
        angular_velocity: Vec3::zero(),
        j2: 0.0,
        reference_radius: Meters::new(0.0),
        principal_moments: Vec3::zero(),
    };
    let reg = Registry::new(Seconds::new(0.0), vec![frame]).expect("a valid registry");
    FrameTree::from_registries(std::slice::from_ref(&reg)).expect("a valid tree");
    registry::encode(&reg)
}

/// The section of `key` for a slab solid below `top` (meters, frame z)
/// with reflectance `albedo`, its occupancy ramping across one sample.
fn slab(key: &CellKey, top: f64, albedo: f64) -> Vec<u8> {
    let g = key.geometry(Meters::new(EXTENT));
    let n = RESOLUTION;
    let h = g.edge.value() / f64::from(n);
    let samples = Samples::from_fn(n, |_, _, z| {
        let at = g.origin.z + (f64::from(z) + 0.5) * h;
        let occupancy = (0.5 - (at - top) / h).clamp(0.0, 1.0);
        if occupancy < 1e-6 {
            Sample::VACUUM
        } else {
            Sample {
                density: Density::new(2000.0 * occupancy),
                state: State::Solid,
                temperature: Kelvin::new(290.0),
                albedo: [Ratio::new(albedo); 3],
                roughness: Ratio::new(0.7),
                attenuation: Attenuation::new(0.0),
                radiating_temperature: Kelvin::new(290.0),
                occupancy: Ratio::new(occupancy),
            }
        }
    });
    let section = Section::new(*key, g.origin, g.edge, n, samples).expect("valid");
    encode(&section, Compression::None)
}

/// What a frame draws: each mesh's key, positions, and albedo.
type Drawn = Vec<(CellKey, Vec<[f64; 3]>, Vec<[f32; 3]>)>;

/// Steps until nothing is in flight and every mark is extracted, then
/// returns what the next step draws.
fn settle(world: &mut World, fetcher: &mut Fetcher, seen: &Seen, now: &mut f64) -> Drawn {
    let view = (640, 360);
    let start = Instant::now();
    loop {
        assert!(start.elapsed() < Duration::from_secs(120), "the view hangs");
        *now += 0.016;
        matter_step(world, fetcher, seen, view, *now);
        world.wait_selection();
        matter_step(world, fetcher, seen, view, *now);
        let c = world.cache();
        if c.at_target() && c.outstanding() == 0 && c.counts().in_flight == 0 && !fetcher.busy() {
            break;
        }
        fetcher.wait_for_events(Duration::from_millis(5));
    }
    world.finish_extraction();
    *now += 0.016;
    let draw = matter_step(world, fetcher, seen, view, *now);
    draw.meshes
        .iter()
        .map(|m| {
            let albedo = m.vertices.iter().map(|v| v.albedo).collect();
            (m.key, m.positions.clone(), albedo)
        })
        .collect()
}

#[test]
fn hiding_a_plain_layer_composites_its_cells_again_without_a_fetch() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let low = stored_layer(
        "low",
        1,
        SectionMaker(Arc::new(|k: &CellKey| slab(k, 0.0, 0.1))),
    );
    let mut high = stored_layer(
        "high",
        1,
        SectionMaker(Arc::new(|k: &CellKey| slab(k, 30.0, 0.9))),
    );
    // A custom section makes the module, and so every address of the
    // layer, differ from `low`'s.
    high.module
        .extend_from_slice(&[0, 5, 4, b'h', b'i', b'g', b'h']);
    let hub = rt
        .block_on(MockHub::start(MockHubOptions::new(
            vec![registry_layer("frames", &registry_bytes()), low, high],
            Vec::new(),
        )))
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
    let system: FrameSystem = fetcher.load_registry(Duration::from_secs(60)).unwrap();
    let mut world = World::new(1);
    world.start_layers(fetcher.client().build(), None);
    fetcher.set_hidden_layers(world.hidden_layers().clone());
    let position = Vec3::new(150.0, 40.0, 160.0);
    let camera = Camera {
        frame_id: FRAME,
        position,
        orientation: look_rotation(Vec3::zero() - position, Vec3::new(0.0, 0.0, 1.0)),
        speed: 0.0,
        cruise: 0.0,
    };
    let seen = Seen::instant(&system, &camera);
    let mut now = 0.0;

    let both = settle(&mut world, &mut fetcher, &seen, &mut now);
    assert!(!both.is_empty(), "the slabs are drawn");
    let reads = hub.counts();
    let cache = world.cache();
    println!(
        "layers: cell cache {} ready cells, composites {} bytes, with kept layer sections {} bytes",
        cache.ready_cells().count(),
        cache.composite_bytes(),
        cache.sample_bytes()
    );

    // The build orders layers by id.
    let high = (fetcher.client().build().layers.iter())
        .position(|l| l.layer == "high")
        .expect("the high layer");
    world.set_hidden_layers([high].into());
    fetcher.set_hidden_layers(world.hidden_layers().clone());
    let low_only = settle(&mut world, &mut fetcher, &seen, &mut now);
    assert!(!low_only.is_empty(), "the low slab is drawn");
    assert_ne!(low_only, both, "hiding a layer changes the drawn composite");
    for (_, _, albedo) in &low_only {
        assert!(
            albedo.iter().all(|a| a[0] < 0.5),
            "only the dark slab is drawn: {:?}",
            &albedo[..albedo.len().min(4)]
        );
    }

    world.set_hidden_layers(Default::default());
    fetcher.set_hidden_layers(world.hidden_layers().clone());
    let restored = settle(&mut world, &mut fetcher, &seen, &mut now);
    assert_eq!(restored, both, "showing it restores the composite");

    let after = hub.counts();
    assert_eq!(
        after.section_reads, reads.section_reads,
        "no section fetched"
    );
    assert_eq!(
        after.directions_requests, reads.directions_requests,
        "no directions asked"
    );
}

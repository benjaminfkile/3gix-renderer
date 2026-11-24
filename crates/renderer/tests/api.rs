//! The renderer API against the mock hub (`docs/api.md` sections 3, 4, 5,
//! 7, and 8, proof in section 11).
//!
//! Each test loads the shared scene's registry from the mock hub and runs
//! the window's tick order without a window: the consumer, the queued
//! commands ([`Api::begin_tick`]), the flight input ([`Api::flight`] over
//! the keys), the simulation and the camera ([`renderer::camera::advance`]),
//! and the snapshot and events ([`Api::end_tick`]).

mod common;

use gx_core::units::Seconds;
use renderer::api::{Api, Controls, Event, Exposure, FlightInput, Kind, MeshMode, Pose, Target};
use renderer::app::{matter_step, set_fetch_time};
use renderer::camera::{Camera, SPEED_LIMITS};
use renderer::config::ApiKey;
use renderer::controls::Controls as Keys;
use renderer::hub::HubClient;
use renderer::mock_hub::{
    example_matter_layer, registry_layer, MockHub, MockHubOptions, MOCK_API_KEY, MOCK_SPACE_ID,
};
use renderer::protocol::Routes;
use renderer::seen::Seen;
use renderer::sim::Simulation;
use renderer::spawn::Spawner;
use renderer::stream::{Fetcher, FetcherOptions};
use renderer::world::World;
use std::sync::Arc;
use std::time::Duration;
use winit::keyboard::KeyCode;

/// Seconds per test tick.
const DT: f64 = 1.0 / 60.0;

/// The view, pixels.
const VIEW: (u32, u32) = (320, 180);

/// The core example module the matter layer runs.
const EXAMPLE_MODULE: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../testdata/example-module/module.wasm"
));

/// The window's loop state without the window, streaming from a mock hub
/// with the shared scene's registry and a matter layer.
struct Loop {
    api: Api,
    sim: Simulation,
    camera: Camera,
    spawner: Spawner,
    keys: Keys,
    world: World,
    fetcher: Fetcher,
    hub: MockHub,
    now: Seconds,
    t: f64,
    _rt: tokio::runtime::Runtime,
}

impl Loop {
    fn new() -> Loop {
        let (matter, tiles) = example_matter_layer(
            "matter",
            EXAMPLE_MODULE,
            &[
                (common::ROOT, common::ROOT_EXTENT),
                (common::CHILD, common::CHILD_EXTENT),
            ],
            1,
            20.0,
            6000.0,
        );
        Loop::with(
            vec![registry_layer("frames", &common::registry_bytes()), matter],
            tiles,
        )
    }

    /// The loop over a mock hub serving `layers` and source `tiles`.
    fn with(layers: Vec<renderer::mock_hub::MockLayer>, tiles: Vec<Vec<u8>>) -> Loop {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let hub = rt
            .block_on(MockHub::start(MockHubOptions::new(layers, tiles)))
            .unwrap();
        let client = rt
            .block_on(HubClient::open(
                Routes::new(&hub.url(), None, MOCK_SPACE_ID),
                ApiKey::new(MOCK_API_KEY),
                None,
            ))
            .expect("the mock hub answers");
        let build = client.build().clone();
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
        let now = system.tree().epoch() + Seconds::new(1.0e6);
        let camera = Camera::home(&system);
        let mut world = World::new(1);
        world.start_layers(&build, None);
        fetcher.set_hidden_layers(world.hidden_layers().clone());
        Loop {
            api: Api::new(system.tree(), &camera)
                .with_build(&build)
                .with_compute_budget(fetcher.max_compute_budget(), None),
            sim: Simulation::at_now(system, now),
            camera,
            spawner: Spawner::default(),
            keys: Keys::new(None),
            world,
            fetcher,
            hub,
            now,
            t: 0.0,
            _rt: rt,
        }
    }

    /// A key, as the window passes it on: only with the built-in controls.
    fn key(&mut self, code: KeyCode, pressed: bool) {
        if self.api.controls() == Controls::Builtin {
            self.keys
                .key(code, pressed, false, &mut self.sim, &mut self.camera);
        }
    }

    /// One tick in the window's order, without streaming.
    fn tick(&mut self) {
        self.tick_with(false);
    }

    /// One tick in the window's order; with `stream`, the matter step
    /// streams as the window's does.
    fn tick_with(&mut self, stream: bool) {
        self.api.begin_tick(Target {
            sim: &mut self.sim,
            camera: &mut self.camera,
            spawner: &mut self.spawner,
            keys: &mut self.keys,
        });
        // As the window applies it (`crate::app`).
        if self.api.get_compute_budget() != self.fetcher.compute_budget() {
            self.fetcher
                .set_compute_budget(self.api.get_compute_budget());
        }
        if self.api.paused() {
            self.api.end_tick(&self.sim, &self.camera, &self.world);
            return;
        }
        let input = self.api.flight(self.keys.take_flight());
        renderer::camera::advance(
            &mut self.sim,
            &mut self.camera,
            DT,
            self.now,
            &input,
            self.keys.attached,
        );
        if self.world.toggle_layers(&self.keys.take_layer_toggles()) {
            (self.fetcher).set_hidden_layers(self.world.hidden_layers().clone());
        }
        if stream {
            self.t += DT;
            self.sim.prepare_light(&self.camera);
            let seen = Seen::instant(self.sim.system(), &self.camera);
            set_fetch_time(&mut self.fetcher, &self.sim);
            matter_step(&mut self.world, &mut self.fetcher, &seen, VIEW, self.t);
        }
        self.api.end_tick(&self.sim, &self.camera, &self.world);
    }

    /// What the mock hub has served: direction batches and section reads.
    fn fetches(&self) -> u64 {
        let c = self.hub.counts();
        c.directions_requests + c.section_reads
    }
}

fn pose(frame_id: u64, position: [f64; 3]) -> Pose {
    Pose {
        frame_id,
        position,
        orientation: [0.0, 0.0, 0.0, 1.0],
        speed: 1.0,
    }
}

#[test]
fn ready_comes_first_and_once() {
    let mut l = Loop::new();
    assert!(l.api.events().is_empty());
    l.tick();
    assert_eq!(l.api.events(), vec![Event::Ready]);
    l.tick();
    assert!(l.api.events().is_empty(), "nothing moved, nothing to say");
}

#[test]
fn set_camera_puts_the_camera_at_the_pose_next_tick() {
    let mut l = Loop::new();
    l.tick();
    let home = l.api.get_camera();
    let p = pose(common::ROOT, [-20.0, 20.0, 20.0]);
    l.api.set_camera(p, false).unwrap();
    assert_eq!(l.api.get_camera(), home, "getters read the last tick");
    l.tick();
    assert_eq!(l.api.get_camera(), p);
    assert_eq!(l.camera.cruise, 1.0);
}

#[test]
fn set_camera_errors() {
    let mut l = Loop::new();
    let unknown = l.api.set_camera(pose(99, [0.0; 3]), false);
    assert!(unknown.unwrap_err().0.contains("not in the registry"));
    let nan = l
        .api
        .set_camera(pose(common::ROOT, [f64::NAN, 0.0, 0.0]), false);
    assert!(nan.unwrap_err().0.contains("not finite"));
    let mut q = pose(common::ROOT, [0.0; 3]);
    q.orientation = [0.0, 0.0, 0.0, 1.0 + 2e-6];
    assert!(l
        .api
        .set_camera(q, false)
        .unwrap_err()
        .0
        .contains("unit quaternion"));
    q.orientation = [0.0, 0.0, 0.0, 1.0 + 5e-7];
    assert!(l.api.set_camera(q, false).is_ok(), "within 1e-6 is unit");
    let mut s = pose(common::ROOT, [0.0; 3]);
    s.speed = 0.0;
    assert!(l.api.set_camera(s, false).is_err());
}

#[test]
fn commands_apply_in_call_order_within_one_tick() {
    let mut l = Loop::new();
    l.tick();
    l.api
        .set_camera(pose(common::ROOT, [-20.0, 20.0, 20.0]), false)
        .unwrap();
    l.api.set_speed(5.0).unwrap();
    l.tick();
    assert_eq!(l.api.get_camera().speed, 5.0, "the speed after the pose");
    l.api.set_speed(7.0).unwrap();
    l.api
        .set_camera(pose(common::ROOT, [-21.0, 20.0, 20.0]), false)
        .unwrap();
    l.tick();
    assert_eq!(l.api.get_camera().speed, 1.0, "the pose after the speed");
    assert_eq!(l.api.get_camera().position, [-21.0, 20.0, 20.0]);
}

#[test]
fn set_speed_clamps_and_rejects_non_finite() {
    let mut l = Loop::new();
    l.api.set_speed(1e30).unwrap();
    l.tick();
    assert_eq!(l.api.get_camera().speed, SPEED_LIMITS.1);
    assert!(l.api.set_speed(f64::NAN).is_err());
}

#[test]
fn fov_is_set_and_read_in_degrees() {
    let mut l = Loop::new();
    assert!((l.api.get_fov() - 60.0).abs() < 1e-12);
    l.api.set_fov(45.0).unwrap();
    assert_eq!(l.api.get_fov(), 45.0);
    assert!(l.api.set_fov(0.0).is_err());
    assert!(l.api.set_fov(f64::NAN).is_err());
}

#[test]
fn fov_is_one_value_on_the_camera() {
    use renderer::camera::FIELD_OF_VIEW_Y;
    use renderer::render::scene::build_scene;
    let mut l = Loop::new();
    l.tick();
    assert_eq!(l.camera.fov_y, FIELD_OF_VIEW_Y, "the default is unchanged");
    let default = build_scene(l.sim.system(), &l.camera, 2.0);
    assert_eq!(default.fov_y, FIELD_OF_VIEW_Y as f32);
    assert_eq!(
        renderer::farfield::focal_px(720.0, l.camera.fov_y),
        720.0 / (2.0 * (0.5 * FIELD_OF_VIEW_Y).tan())
    );
    l.api.set_fov(30.0).unwrap();
    assert_eq!(l.camera.fov_y, FIELD_OF_VIEW_Y, "applied next tick");
    l.tick();
    assert!((l.camera.fov_y - 30f64.to_radians()).abs() < 1e-15);
    let narrow = build_scene(l.sim.system(), &l.camera, 2.0);
    assert_eq!(narrow.fov_y, 30f64.to_radians() as f32);
    l.api
        .set_camera(pose(l.camera.frame_id, [1.0, 2.0, 3.0]), false)
        .unwrap();
    l.tick();
    assert!(
        (l.camera.fov_y - 30f64.to_radians()).abs() < 1e-15,
        "a set pose keeps the field of view"
    );
}

#[test]
fn place_over_uses_the_start_on_placement() {
    let mut l = Loop::new();
    let s = renderer::config::StartOn::parse("2:1,0,0@+3^20").unwrap();
    let want = renderer::options::launch_camera_at(l.sim.system(), s).unwrap();
    l.api
        .place_over(common::CHILD, [2.0, 0.0, 0.0], 3.0, 20.0)
        .unwrap();
    l.tick();
    assert_eq!(l.spawner.pending, Some(s), "spawns as --start-on does");
    let got = l.api.get_camera();
    assert_eq!(got.frame_id, want.frame_id);
    assert_eq!(got.orientation, Pose::of(&want).orientation);
    assert!(l.api.place_over(99, [1.0, 0.0, 0.0], 3.0, 0.0).is_err());
    assert!(l.api.place_over(common::CHILD, [0.0; 3], 3.0, 0.0).is_err());
    assert!(l
        .api
        .place_over(common::CHILD, [1.0, 0.0, 0.0], 0.0, 0.0)
        .is_err());
    assert!(l
        .api
        .place_over(common::CHILD, [1.0, 0.0, 0.0], 1.0, 91.0)
        .is_err());
}

/// Flying forward with W for some ticks, by the key or by the API.
fn fly_forward(by_key: bool, controls: Controls) -> Camera {
    let mut l = Loop::new();
    l.api.set_controls(controls);
    l.api
        .set_camera(pose(common::ROOT, [-20.0, 20.0, 20.0]), false)
        .unwrap();
    l.tick();
    if by_key {
        l.key(KeyCode::KeyW, true);
        l.key(KeyCode::ShiftLeft, true);
    } else {
        l.api.set_flight_input(Some(FlightInput {
            r#move: [0.0, 0.0, -1.0],
            look: [0.0; 3],
            boost: true,
        }));
    }
    for _ in 0..30 {
        l.tick();
    }
    l.camera
}

#[test]
fn flight_input_moves_the_camera_as_the_keys_do() {
    let keys = fly_forward(true, Controls::Builtin);
    let api = fly_forward(false, Controls::Builtin);
    assert_eq!(keys, api);
    assert_ne!(keys.position.z, 20.0, "it moved");
    assert_eq!(fly_forward(false, Controls::None), api, "in both modes");
}

#[test]
fn flight_input_is_held_until_cleared() {
    let mut l = Loop::new();
    l.api
        .set_camera(pose(common::ROOT, [-20.0, 20.0, 20.0]), false)
        .unwrap();
    l.tick();
    l.api.set_flight_input(Some(FlightInput {
        r#move: [1.0, 0.0, 0.0],
        ..FlightInput::default()
    }));
    l.tick();
    let a = l.camera.position.x;
    l.tick();
    let b = l.camera.position.x;
    assert!(b > a && a > -20.0, "held over ticks");
    l.api.set_flight_input(None);
    l.tick();
    l.tick();
    assert_eq!(l.camera.position.x, l.api.get_camera().position[0]);
    let c = l.camera.position.x;
    l.tick();
    assert_eq!(l.camera.position.x, c, "cleared");
}

#[test]
fn controls_none_ignores_keys() {
    let still = {
        let mut l = Loop::new();
        l.api.set_controls(Controls::None);
        l.tick();
        l.key(KeyCode::KeyW, true);
        l.key(KeyCode::Home, true);
        l.key(KeyCode::Space, true);
        for _ in 0..10 {
            l.tick();
        }
        assert!(!l.sim.clock().paused(), "the clock key did nothing");
        l.api.get_camera()
    };
    let mut l = Loop::new();
    l.tick();
    assert_eq!(still, l.api.get_camera());
    // Keys held before switching to `None` stop flying the camera.
    l.key(KeyCode::KeyW, true);
    l.tick();
    l.api.set_controls(Controls::None);
    let before = l.camera.position;
    l.tick();
    assert_eq!(l.camera.position, before);
}

#[test]
fn camera_event_once_per_tick_only_on_change() {
    let mut l = Loop::new();
    l.tick();
    l.api.events();
    l.api
        .set_camera(pose(common::ROOT, [-20.0, 20.0, 20.0]), false)
        .unwrap();
    l.api
        .set_camera(pose(common::ROOT, [-21.0, 20.0, 20.0]), false)
        .unwrap();
    l.tick();
    let events = l.api.events();
    assert_eq!(
        events,
        vec![Event::Camera(pose(common::ROOT, [-21.0, 20.0, 20.0]))]
    );
    l.tick();
    assert!(l.api.events().is_empty());
    l.api.set_flight_input(Some(FlightInput {
        r#move: [0.0, 1.0, 0.0],
        ..FlightInput::default()
    }));
    l.tick();
    l.tick();
    let moved = l.api.events();
    assert_eq!(moved.len(), 2, "one per tick: {moved:?}");
    assert!(matches!(moved[1], Event::Camera(p) if p == l.api.get_camera()));
}

#[test]
fn frame_changed_on_reparent() {
    let mut l = Loop::new();
    l.tick();
    l.api.events();
    // At the child's origin in root axes: the next step reparents there.
    l.api
        .set_camera(pose(common::ROOT, common::CHILD_OFFSET), false)
        .unwrap();
    l.tick();
    let events = l.api.events();
    assert_eq!(l.api.get_camera().frame_id, common::CHILD);
    assert_eq!(
        events[0],
        Event::FrameChanged {
            from: common::ROOT,
            to: common::CHILD
        }
    );
    assert!(matches!(events[1], Event::Camera(p) if p.frame_id == common::CHILD));
    assert_eq!(events.len(), 2);
}

#[test]
fn time_reads_in_utc_and_set_time_clamps_as_the_start_options() {
    let mut l = Loop::new();
    assert_eq!(l.api.get_time(), None, "nothing before the first tick");
    l.tick();
    let t = l.api.get_time().unwrap();
    assert_eq!(t.utc_ms, t.now_utc_ms, "starts at now");
    assert_eq!(t.time_scale, 1.0);
    assert!(t.origin_utc_ms < t.now_utc_ms);
    let middle = (t.origin_utc_ms + t.now_utc_ms) / 2;
    l.api.set_time(middle).unwrap();
    assert_eq!(l.api.get_time(), Some(t), "getters read the last tick");
    l.tick();
    let got = l.api.get_time().unwrap();
    // Set at the start of the tick, then one tick played at rate 1.
    let played = (DT * 1000.0).ceil() as i64 + 1;
    assert!(
        (got.utc_ms - middle).abs() <= played,
        "{} {}",
        got.utc_ms,
        middle
    );
    l.api.set_time(t.origin_utc_ms - 1_000_000).unwrap();
    l.api.set_time_scale(-1.0).unwrap();
    l.tick();
    assert_eq!(
        l.api.get_time().unwrap().utc_ms,
        t.origin_utc_ms,
        "held at the origin"
    );
    l.api.set_time(t.now_utc_ms + 1_000_000).unwrap();
    l.tick();
    assert_eq!(l.api.get_time().unwrap().utc_ms, t.now_utc_ms);
    assert!(l.sim.clock().is_at_now(), "past now follows now");
}

#[test]
fn set_time_scale_runs_back_and_clamps() {
    let mut l = Loop::new();
    l.tick();
    l.api.set_time_scale(-4.0).unwrap();
    l.tick();
    let a = l.api.get_time().unwrap();
    assert_eq!(a.time_scale, -4.0);
    l.tick();
    assert!(l.api.get_time().unwrap().utc_ms < a.utc_ms, "runs back");
    let origin = l.api.get_time().unwrap().origin_utc_ms;
    l.api.set_time(origin).unwrap();
    l.api.set_time_scale(1e30).unwrap();
    l.tick();
    let (_, hi) = renderer::sim::RATE_LIMITS;
    assert_eq!(l.api.get_time().unwrap().time_scale, hi);
    assert!(l.api.set_time_scale(0.0).is_err());
    assert!(l.api.set_time_scale(f64::NAN).is_err());
}

#[test]
fn frames_list_the_registry_and_surfaces_once_arrived() {
    use gx_core::matter::Compression;
    use gx_core::surface::{SurfaceTile, HEIGHT_SIDE, PARENT_SIDE, VERTEX_COUNT};
    let mut l = Loop::new();
    l.tick();
    let frames = l.api.frames();
    let ids: Vec<u64> = frames.iter().map(|f| f.id).collect();
    let want: Vec<u64> = l
        .sim
        .system()
        .tree()
        .frames()
        .iter()
        .map(|f| f.frame_id)
        .collect();
    assert_eq!(ids, want);
    let child = frames.iter().find(|f| f.id == common::CHILD).unwrap();
    assert_eq!(child.parent_id, Some(common::ROOT));
    assert_eq!(frames[0].parent_id, None);
    assert!(!child.has_surface);
    assert_eq!(child.surface, None, "null before the tiles arrive");
    // The child's six level-0 tiles arrive.
    for face in 0..gx_core::key::SURFACE_FACES {
        let key = gx_core::key::SurfaceKey {
            frame_id: common::CHILD,
            face,
            level: 0,
            x: 0,
            y: 0,
        };
        let tile = SurfaceTile {
            key,
            radius_equatorial: 4.0,
            radius_polar: 3.9,
            height_floor: -0.25,
            heights: vec![0.0; HEIGHT_SIDE * HEIGHT_SIDE],
            parent_heights: vec![0.0; PARENT_SIDE * PARENT_SIDE],
            albedo: vec![[40, 90, 160]; VERTEX_COUNT],
            roughness: vec![200; VERTEX_COUNT],
            temperature: vec![290.0; VERTEX_COUNT],
        };
        let bytes = vec![gx_core::surface::encode(&tile, Compression::None)];
        let decoded = renderer::tiles::decode_tile(&key, &bytes, &[0], &Default::default());
        l.world
            .select_surfaces(l.sim.system(), &l.camera, VIEW, 0.0);
        l.world.complete_surface(key, decoded, false);
    }
    l.tick();
    let child = l
        .api
        .frames()
        .into_iter()
        .find(|f| f.id == common::CHILD)
        .unwrap();
    assert!(child.has_surface);
    let s = child.surface.expect("the surface once arrived");
    assert_eq!(
        (s.radius_equatorial, s.radius_polar, s.height_floor),
        (4.0, 3.9, -0.25)
    );
}

fn close(a: [f64; 3], b: [f64; 3]) -> bool {
    let scale = b.iter().map(|v| v.abs()).fold(1e-300, f64::max);
    a.iter().zip(&b).all(|(x, y)| (x - y).abs() <= 1e-9 * scale)
}

#[test]
fn transform_round_trips_points_and_directions() {
    let mut l = Loop::new();
    assert!(l
        .api
        .transform([1.0, 0.0, 0.0], common::ROOT, common::CHILD, Kind::Point)
        .is_err());
    l.tick();
    let p = [3.0, -2.0, 5.5];
    for kind in [Kind::Point, Kind::Direction] {
        let there = l
            .api
            .transform(p, common::ROOT, common::CHILD, kind)
            .unwrap();
        let back = l
            .api
            .transform(there, common::CHILD, common::ROOT, kind)
            .unwrap();
        assert!(close(back, p), "{kind:?}: {back:?} {p:?}");
    }
    // The child's origin is at its offset in the root.
    let origin = l
        .api
        .transform([0.0; 3], common::CHILD, common::ROOT, Kind::Point)
        .unwrap();
    let sys = l.sim.system();
    let want = sys.relative(
        gx_core::units::Vec3::zero(),
        common::CHILD,
        gx_core::units::Vec3::zero(),
        common::ROOT,
    );
    let want = sys.root_orientation(common::ROOT).conjugate().rotate(want);
    assert!(close(origin, [want.x, want.y, want.z]));
    let d = l
        .api
        .transform([0.0; 3], common::CHILD, common::ROOT, Kind::Direction)
        .unwrap();
    assert_eq!(d, [0.0; 3], "a direction does not move with the origin");
    assert!(l.api.transform(p, 99, common::ROOT, Kind::Point).is_err());
}

#[test]
fn layers_and_set_shown_take_the_chooser_path() {
    let mut l = Loop::new();
    l.tick();
    let layers = l.api.layers();
    let ids: Vec<&str> = layers.iter().map(|x| x.id.as_str()).collect();
    assert_eq!(ids, ["frames", "matter"]);
    assert!(layers.iter().all(|x| x.shown && !x.overlay));
    l.api.set_shown("matter", false).unwrap();
    l.tick();
    assert!(!l.api.layers()[1].shown);
    assert!(l.world.hidden_layers().contains(&1));
    l.api.set_shown("matter", true).unwrap();
    l.api.set_shown("matter", true).unwrap();
    l.tick();
    assert!(l.api.layers()[1].shown, "twice shown is shown");
    l.api.set_shown("nothing", true).unwrap();
    l.tick();
    let events = l.api.events();
    assert!(
        events
            .iter()
            .any(|e| matches!(e, Event::Error(m) if m.contains("no layer"))),
        "a failed queued call is an error event: {events:?}"
    );
}

#[test]
fn display_items_apply_as_the_options() {
    let mut l = Loop::new();
    l.api.set_exposure(Exposure::Fixed { stops: 3.0 }).unwrap();
    l.api.set_mesh_mode(MeshMode::Edges).unwrap();
    l.tick();
    assert_eq!(l.keys.exposure_fixed, Some(3.0));
    assert_eq!(l.keys.mesh_mode, MeshMode::Edges);
    l.api
        .set_exposure(Exposure::Auto { bias_stops: 1e9 })
        .unwrap();
    l.tick();
    assert_eq!(l.keys.exposure_fixed, None);
    assert_eq!(
        l.keys.exposure_bias,
        renderer::controls::EXPOSURE_BIAS_LIMITS.1
    );
    assert!(l
        .api
        .set_exposure(Exposure::Fixed { stops: f64::NAN })
        .is_err());
    assert_eq!(l.api.max_fps(), None);
    l.api.set_max_fps(Some(30)).unwrap();
    assert_eq!(l.api.max_fps(), Some(30));
    l.api.set_max_fps(None).unwrap();
    assert_eq!(l.api.max_fps(), None);
}

#[test]
fn hud_off_draws_no_overlay_text() {
    let mut l = Loop::new();
    assert!(l.api.hud());
    assert_eq!(l.api.overlay_text("t".into()).as_deref(), Some("t"));
    l.api.set_hud(false).unwrap();
    assert_eq!(l.api.overlay_text("t".into()), None);
    // The headless renderer with no overlay text draws the scene alone.
    let scene = renderer::render::scene::build_scene(l.sim.system(), &l.camera, 2.0);
    let mut gpu = renderer::render::headless::Headless::new(64, 32, None)
        .expect("a wgpu adapter, software is fine");
    let without = gpu.render(&scene, l.api.overlay_text("HUD".into()).as_deref());
    let blank = gpu.render(&scene, None);
    assert_eq!(without.unwrap().rgba, blank.unwrap().rgba);
    l.api.set_hud(true).unwrap();
    let with = gpu
        .render(&scene, l.api.overlay_text("HUD".into()).as_deref())
        .unwrap();
    assert_ne!(with.rgba, gpu.render(&scene, None).unwrap().rgba);
}

#[test]
fn pause_holds_time_and_fetches_nothing() {
    let mut l = Loop::new();
    l.api.set_time_scale(0.5).unwrap();
    l.tick();
    l.api.pause().unwrap();
    l.tick();
    let held = l.api.get_time().unwrap();
    let before = l.fetches();
    for _ in 0..30 {
        l.tick_with(true);
        std::thread::sleep(Duration::from_millis(2));
    }
    assert_eq!(l.api.get_time().unwrap(), held, "time held");
    assert_eq!(l.fetches(), before, "nothing fetched while paused");
    l.api.resume().unwrap();
    let start = std::time::Instant::now();
    while l.fetches() == before {
        l.tick_with(true);
        l.fetcher.wait_for_events(Duration::from_millis(20));
        assert!(
            start.elapsed() < Duration::from_secs(30),
            "no fetch after resume"
        );
    }
    assert!(l.api.get_time().unwrap().utc_ms >= held.utc_ms);
}

#[test]
fn resize_and_destroy() {
    let mut l = Loop::new();
    assert!(l.api.resize(0, 10, 1.0).is_err());
    assert!(l.api.resize(10, 10, 0.0).is_err());
    l.api.resize(640, 480, 2.0).unwrap();
    assert_eq!(l.api.take_size(), Some((640, 480, 2.0)));
    assert_eq!(l.api.take_size(), None);
    l.api.destroy().unwrap();
    assert!(l.api.destroyed());
    let e = l.api.set_speed(1.0).unwrap_err();
    assert!(e.0.contains("destroyed"));
    assert!(l.api.pause().is_err());
    assert!(l.api.destroy().is_err());
}

#[test]
fn flight_look_turns_by_radians() {
    let mut l = Loop::new();
    l.api
        .set_camera(pose(common::ROOT, [-20.0, 20.0, 20.0]), false)
        .unwrap();
    l.tick();
    l.api.set_flight_input(Some(FlightInput {
        look: [0.25, 0.0, 0.0],
        ..FlightInput::default()
    }));
    l.tick();
    let [x, y, z, w] = l.api.get_camera().orientation;
    let angle = 2.0 * (x * x + y * y + z * z).sqrt().atan2(w);
    assert!((angle - 0.25).abs() < 1e-12, "{angle}");
    assert!(y > 0.0, "positive yaw turns left, about the up axis");
}

mod compute_budget {
    //! `docs/api.md` section 7 (`setComputeBudget`, `getComputeBudget`)
    //! against the mock hub.

    use super::{common, Loop};
    use renderer::mock_hub::{registry_layer, stored_layer, SectionMaker};
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    /// Ticks with streaming for `seconds`, waiting for the fetcher between
    /// ticks.
    fn run_for(l: &mut Loop, seconds: f64) {
        let start = Instant::now();
        while start.elapsed().as_secs_f64() < seconds {
            l.tick_with(true);
            l.fetcher.wait_for_events(Duration::from_millis(20));
        }
    }

    /// Ticks until `done` holds, failing after 60 s.
    fn run_until(l: &mut Loop, what: &str, done: impl Fn(&Loop) -> bool) {
        let start = Instant::now();
        while !done(l) {
            l.tick_with(true);
            l.fetcher.wait_for_events(Duration::from_millis(20));
            assert!(
                start.elapsed() < Duration::from_secs(30),
                "{what}: {:?} {:?}",
                l.hub.counts(),
                l.world.cache().counts()
            );
        }
    }

    fn computed(l: &Loop) -> u64 {
        l.fetcher.stats().worker.sections_computed
    }

    /// A valid section with no matter, for either frame of the scene.
    fn empty_cell(key: &gx_core::key::CellKey) -> Vec<u8> {
        let extent = if key.frame_id == common::ROOT {
            common::ROOT_EXTENT
        } else {
            common::CHILD_EXTENT
        };
        let g = key.geometry(gx_core::units::Meters::new(extent));
        let section = gx_core::matter::Section::empty(*key, g.origin, g.edge).expect("valid");
        gx_core::matter::encode(&section, gx_core::matter::Compression::None)
    }

    #[test]
    fn get_and_set_within_the_default() {
        let mut l = Loop::new();
        let max = l.api.max_compute_budget();
        assert_eq!(l.api.get_compute_budget(), max, "the default");
        l.api.set_compute_budget(0).unwrap();
        assert_eq!(l.api.get_compute_budget(), 0);
        assert!(l.api.set_compute_budget(max + 1).is_err());
        assert_eq!(l.api.get_compute_budget(), 0);
        l.tick();
        assert_eq!(l.fetcher.compute_budget(), 0, "applied at the tick");
        l.api.set_compute_budget(max).unwrap();
        l.tick();
        assert_eq!(l.fetcher.compute_budget(), max);
        // The start option clamps to the default.
        let a = renderer::api::Api::new(l.sim.system().tree(), &l.camera)
            .with_compute_budget(3, Some(7));
        assert_eq!((a.get_compute_budget(), a.max_compute_budget()), (3, 3));
    }

    #[test]
    fn at_zero_stored_sections_stream_and_draw_with_nothing_computed() {
        let layers = vec![
            registry_layer("frames", &common::registry_bytes()),
            stored_layer("main", 1, SectionMaker(Arc::new(empty_cell))),
        ];
        let mut l = Loop::with(layers, Vec::new());
        l.api.set_compute_budget(0).unwrap();
        // The registry's own upload, sent before, lands first.
        run_for(&mut l, 0.5);
        let (jobs, uploads) = (computed(&l), l.hub.counts().uploads_stored);
        run_until(&mut l, "stored cells ready at budget 0", |l| {
            l.world.cache().counts().ready > 0
        });
        run_for(&mut l, 0.5);
        assert_eq!(computed(&l), jobs, "no compute job at budget 0");
        assert_eq!(l.hub.counts().uploads_stored, uploads, "no upload");
        assert!(l.hub.counts().section_reads > 0, "stored sections stream");
    }

    #[test]
    fn missing_stays_missing_at_zero_and_raising_starts_work() {
        let mut l = Loop::new();
        l.api.set_compute_budget(0).unwrap();
        // The registry's own upload, sent before, lands first.
        run_for(&mut l, 0.5);
        let (jobs, uploads) = (computed(&l), l.hub.counts().uploads_stored);
        run_until(&mut l, "directions answered", |l| {
            l.hub.counts().missing_answers > 0
        });
        run_for(&mut l, 1.0);
        assert_eq!(computed(&l), jobs, "no compute job at budget 0");
        assert_eq!(l.hub.counts().uploads_stored, uploads, "no upload");
        assert_eq!(l.world.cache().counts().ready, 0, "missing stays missing");
        l.api.set_compute_budget(1).unwrap();
        run_until(&mut l, "computing after raising the budget", |l| {
            computed(l) > jobs && l.hub.counts().uploads_stored > uploads
        });
    }

    #[test]
    fn lowering_mid_run_never_sends_a_partial_upload() {
        let mut l = Loop::new();
        run_until(&mut l, "a module run", |l| {
            l.fetcher.stats().worker.module_runs > 0
        });
        l.api.set_compute_budget(0).unwrap();
        run_for(&mut l, 1.0);
        let jobs = computed(&l);
        let c = l.hub.counts();
        run_for(&mut l, 1.0);
        let after = l.hub.counts();
        assert_eq!(computed(&l), jobs, "nothing starts after lowering");
        assert_eq!(after.uploads_stored, c.uploads_stored, "nothing sent at 0");
        // The mock checks every upload against its address: a partial
        // section would be refused.
        assert_eq!(after.uploads_refused, 0, "{after:?}");
    }
}

mod queries {
    //! `docs/api.md` sections 3 (`create`), 6 (`surfaceHeight`,
    //! `raycast`, `project`), and 8 (`settled`, `stats`) on synthetic
    //! surface tiles.

    use super::common;
    use gx_core::key::SurfaceKey;
    use gx_core::matter::Compression;
    use gx_core::surface::{SurfaceTile, HEIGHT_SIDE, PARENT_SIDE, TILE_QUADS, VERTEX_COUNT};
    use gx_core::units::{Quat, Vec3};
    use renderer::api::query::{ellipsoid_height, morphed_vertex};
    use renderer::api::{create, Api, Consumer, CreateTarget, Event, HitKind, Picture, Pose};
    use renderer::camera::Camera;
    use renderer::config::{parse_flags, Config};
    use renderer::mock_hub::{
        registry_layer, stored_layer, MockHub, MockHubOptions, SectionMaker, SurfaceMaker,
        MOCK_API_KEY, MOCK_SPACE_ID,
    };
    use renderer::tiles::{Anchor, TileDrawInfo, TileGeometry};
    use renderer::world::DrawList;
    use std::sync::{Arc, Mutex};
    use std::time::Instant;

    /// Radius of the synthetic ball, meters.
    const RADIUS: f64 = 1.0e6;

    /// The view, pixels.
    const SIZE: (u32, u32) = (320, 180);

    /// A synthetic tile: heights varying over the tile, parent heights
    /// different from them so the morph moves vertices.
    fn tile(key: SurfaceKey, a: f64, c: f64) -> SurfaceTile {
        let h = |n: usize| 200.0 + 50.0 * ((n % 13) as f32) + 7.0 * ((n / 11 % 5) as f32);
        SurfaceTile {
            key,
            radius_equatorial: a,
            radius_polar: c,
            height_floor: 0.0,
            heights: (0..HEIGHT_SIDE * HEIGHT_SIDE).map(h).collect(),
            parent_heights: (0..PARENT_SIDE * PARENT_SIDE)
                .map(|n| h(n) + 300.0)
                .collect(),
            albedo: vec![[100, 100, 100]; VERTEX_COUNT],
            roughness: vec![200; VERTEX_COUNT],
            temperature: vec![290.0; VERTEX_COUNT],
        }
    }

    /// A flat tile: every height `h`.
    fn flat(key: SurfaceKey, h: f32) -> SurfaceTile {
        SurfaceTile {
            heights: vec![h; HEIGHT_SIDE * HEIGHT_SIDE],
            parent_heights: vec![h; PARENT_SIDE * PARENT_SIDE],
            ..tile(key, RADIUS, RADIUS)
        }
    }

    fn key(level: u8, x: u32, y: u32) -> SurfaceKey {
        SurfaceKey {
            frame_id: common::CHILD,
            face: 0,
            level,
            x,
            y,
        }
    }

    fn drawn(t: &SurfaceTile, morph: [f32; 2]) -> TileDrawInfo {
        TileDrawInfo {
            geometry: Arc::new(TileGeometry::new(t)),
            anchor: Anchor {
                position: [0.0; 3],
                generation: 0,
            },
            morph,
        }
    }

    /// The camera at `position` in the tile frame, looking along -x.
    fn camera(position: [f64; 3]) -> Camera {
        let s = std::f64::consts::FRAC_1_SQRT_2;
        Camera {
            frame_id: common::CHILD,
            position: Vec3::new(position[0], position[1], position[2]),
            orientation: Quat::new(0.0, s, 0.0, s),
            speed: 1.0,
            cruise: 1.0,
            fov_y: 60f64.to_radians(),
        }
    }

    fn picture(tiles: Vec<TileDrawInfo>, camera: &Camera) -> Picture {
        let system = common::system();
        let draw = DrawList {
            tiles,
            ..DrawList::default()
        };
        Picture::new(&system, camera, &draw, SIZE)
    }

    fn unit(p: [f64; 3]) -> [f64; 3] {
        let l = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
        p.map(|c| c / l)
    }

    #[test]
    fn surface_height_is_the_vertex_height_at_vertices() {
        let (a, c) = (RADIUS, 0.98 * RADIUS);
        let t = tile(key(0, 0, 0), a, c);
        let d = drawn(&t, [f32::INFINITY; 2]);
        let cam = camera([3.0 * RADIUS, 0.0, 0.0]);
        let p = picture(vec![d.clone()], &cam);
        for (i, j) in [(32, 32), (10, 40), (63, 1), (5, 5)] {
            let n = i + (TILE_QUADS + 1) * j;
            let v = d.geometry.positions[n];
            let got = p
                .surface_height(common::CHILD, unit(v))
                .expect("a tile is drawn");
            let want = f64::from(t.height(i as i64, j as i64));
            assert!(
                (got.height - want).abs() < 1e-4,
                "{i},{j}: {} vs {want}",
                got.height
            );
            assert_eq!(got.level, 0);
            assert!(!got.finer);
        }
        assert_eq!(p.surface_height(common::CHILD, [-1.0, 0.0, 0.0]), None);
        assert_eq!(p.surface_height(common::ROOT, [1.0, 0.0, 0.0]), None);
    }

    #[test]
    fn surface_height_is_morphed_between_levels() {
        let t = tile(key(1, 1, 1), RADIUS, RADIUS);
        let cam = camera([RADIUS + 50_000.0, 0.0, 0.0]);
        let morph = [30_000.0f32, 80_000.0];
        let d = drawn(&t, morph);
        let p = picture(vec![d.clone()], &cam);
        let c = [RADIUS + 50_000.0, 0.0, 0.0];
        let mut moved = 0;
        for (i, j) in [(1, 1), (3, 2), (20, 21), (40, 40)] {
            let n = i + (TILE_QUADS + 1) * j;
            let (v, q) = (d.geometry.positions[n], d.geometry.targets[n]);
            // `vs_tile`'s weights, written out.
            let dist = (0..3).map(|a| (v[a] - c[a]).powi(2)).sum::<f64>().sqrt();
            let k = ((dist - 30_000.0) / 50_000.0).clamp(0.0, 1.0);
            let m: [f64; 3] = std::array::from_fn(|a| v[a] + k * (q[a] - v[a]));
            assert_eq!(morphed_vertex(&d, n, c), m);
            let want = ellipsoid_height(m, RADIUS, RADIUS);
            let got = p.surface_height(common::CHILD, unit(m)).unwrap();
            assert!((got.height - want).abs() < 1e-4, "{} vs {want}", got.height);
            if k > 0.0 && k < 1.0 {
                moved += 1;
                let own = f64::from(t.height(i as i64, j as i64));
                assert!((got.height - own).abs() > 1.0, "the morph moved the vertex");
            }
        }
        assert!(moved > 0, "some vertex is part way through the morph");
    }

    #[test]
    fn raycast_down_project_back_and_occlusion() {
        let h0 = 500.0f32;
        let t = flat(key(0, 0, 0), h0);
        let height = 12_345.0;
        let cam = camera([RADIUS + f64::from(h0) + height, 0.0, 0.0]);
        let p = picture(vec![drawn(&t, [f32::INFINITY; 2])], &cam);
        let hit = p
            .raycast(f64::from(SIZE.0) / 2.0, f64::from(SIZE.1) / 2.0)
            .expect("straight down hits the tile");
        assert_eq!(hit.kind, HitKind::Surface);
        assert_eq!(hit.frame_id, common::CHILD);
        assert!(
            ((hit.distance - height) / height).abs() < 1e-6,
            "{} vs {height}",
            hit.distance
        );
        let system = common::system();
        for (x, y) in [(160.0, 90.0), (100.5, 70.25), (250.0, 30.0), (20.0, 170.0)] {
            let hit = p.raycast(x, y).expect("the tile fills the view");
            let back = p
                .project(&system, &cam, hit.frame_id, hit.position)
                .expect("a hit is in view");
            assert!(
                (back.x - x).abs() < 0.5 && (back.y - y).abs() < 0.5,
                "{back:?} vs {x},{y}"
            );
            assert!(back.depth > 0.0 && back.depth <= hit.distance * (1.0 + 1e-9));
        }
        // Through the ball: the near side of the drawn tile hides it.
        let far = [-(RADIUS + f64::from(h0)), 0.0, 0.0];
        assert_eq!(p.project(&system, &cam, common::CHILD, far), None);
        // Above the surface, in front: shown.
        let near = [RADIUS + f64::from(h0) + 0.5 * height, 0.0, 0.0];
        assert!(p.project(&system, &cam, common::CHILD, near).is_some());
        // Behind the camera.
        let behind = [RADIUS + f64::from(h0) + 2.0 * height, 0.0, 0.0];
        assert_eq!(p.project(&system, &cam, common::CHILD, behind), None);
    }

    #[test]
    fn raycast_time_per_call() {
        // A view filled by a 4 x 4 block of level-2 tiles, about the
        // number a close view draws of one face.
        let mut tiles = Vec::new();
        for y in 0..4 {
            for x in 0..4 {
                tiles.push(drawn(
                    &tile(key(2, x, y), RADIUS, RADIUS),
                    [f32::INFINITY; 2],
                ));
            }
        }
        let cam = camera([RADIUS + 200_000.0, 0.0, 0.0]);
        let p = picture(tiles, &cam);
        let n = 2000;
        let start = Instant::now();
        let mut hits = 0;
        for k in 0..n {
            let x = f64::from(k % 320);
            let y = f64::from((k * 7) % 180);
            hits += usize::from(p.raycast(x, y).is_some());
        }
        let per = start.elapsed().as_secs_f64() / f64::from(n);
        eprintln!(
            "raycast: {:.1} us per call over {n} calls, {hits} hits",
            per * 1e6
        );
        assert!(hits > 0);
    }

    /// A consumer that records events and destroys the renderer once
    /// `until` says so, or after a deadline.
    struct Recorder {
        events: Arc<Mutex<Vec<Event>>>,
        heights: Arc<Mutex<Vec<Option<f64>>>>,
        pose: Option<Pose>,
        started: Instant,
        until: fn(&[Event]) -> bool,
    }

    impl Consumer for Recorder {
        fn tick(&mut self, api: &mut Api) {
            if let Some(pose) = self.pose.take() {
                api.set_camera(pose, false).unwrap();
            }
            let new = api.events();
            let mut events = self.events.lock().unwrap();
            for e in &new {
                if matches!(e, Event::Settled { .. }) {
                    let h = api
                        .surface_height(common::CHILD, [1.0, 0.0, 0.0])
                        .unwrap()
                        .map(|s| s.height);
                    self.heights.lock().unwrap().push(h);
                }
            }
            events.extend(new);
            if (self.until)(&events) || self.started.elapsed().as_secs() > 120 {
                api.destroy().unwrap();
            }
        }
    }

    fn mock_config(hub: &MockHub) -> Config {
        let mut env = std::collections::BTreeMap::new();
        env.insert(renderer::config::ENV_HUB_URL.into(), hub.url());
        env.insert(renderer::config::ENV_API_KEY.into(), MOCK_API_KEY.into());
        env.insert(renderer::config::ENV_SPACE_ID.into(), MOCK_SPACE_ID.into());
        let flags = parse_flags(["--workers", "1"]).unwrap();
        Config::resolve(&env, &flags).unwrap()
    }

    fn run(
        layers: Vec<renderer::mock_hub::MockLayer>,
        pose: Option<Pose>,
        until: fn(&[Event]) -> bool,
    ) -> (Vec<Event>, Vec<Option<f64>>) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let hub = rt
            .block_on(MockHub::start(MockHubOptions::new(layers, Vec::new())))
            .unwrap();
        let config = mock_config(&hub);
        let events = Arc::new(Mutex::new(Vec::new()));
        let heights = Arc::new(Mutex::new(Vec::new()));
        let recorder = Recorder {
            events: events.clone(),
            heights: heights.clone(),
            pose,
            started: Instant::now(),
            until,
        };
        create(
            CreateTarget::Headless {
                width: SIZE.0,
                height: SIZE.1,
            },
            &config,
            config.options.clone(),
            Box::new(recorder),
        )
        .expect("create runs the loop");
        let e = events.lock().unwrap().clone();
        let h = heights.lock().unwrap().clone();
        (e, h)
    }

    #[test]
    fn create_runs_the_loop_until_destroy() {
        let layers = vec![registry_layer("frames", &common::registry_bytes())];
        let (events, _) = run(layers, None, |e| e.contains(&Event::Ready));
        assert_eq!(events.first(), Some(&Event::Ready));
    }

    /// A consumer that records the compute budget it sees on its first
    /// tick and then stops the loop.
    struct BudgetProbe(Arc<Mutex<Option<usize>>>);

    impl Consumer for BudgetProbe {
        fn tick(&mut self, api: &mut Api) {
            *self.0.lock().unwrap() = Some(api.get_compute_budget());
            api.destroy().unwrap();
        }
    }

    #[test]
    fn create_takes_the_start_compute_budget() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let hub = rt
            .block_on(MockHub::start(MockHubOptions::new(
                vec![registry_layer("frames", &common::registry_bytes())],
                Vec::new(),
            )))
            .unwrap();
        // Two workers and a start budget of one: below the default, and
        // above 0 so the registry, which the mock hub does not store, is
        // still computed.
        let mut config = mock_config(&hub);
        config.workers = Some(2);
        let mut options = config.options.clone();
        options.compute_budget = Some(1);
        let seen = Arc::new(Mutex::new(None));
        create(
            CreateTarget::Headless {
                width: SIZE.0,
                height: SIZE.1,
            },
            &config,
            options,
            Box::new(BudgetProbe(seen.clone())),
        )
        .expect("create runs the loop");
        assert_eq!(*seen.lock().unwrap(), Some(1));
    }

    /// The radius of the served tiles, meters.
    const SERVED_RADIUS: f64 = 5.0;

    fn served(key: &SurfaceKey) -> Vec<u8> {
        let t = SurfaceTile {
            key: *key,
            radius_equatorial: SERVED_RADIUS,
            radius_polar: SERVED_RADIUS,
            height_floor: 0.0,
            heights: vec![0.1; HEIGHT_SIDE * HEIGHT_SIDE],
            parent_heights: vec![0.1; PARENT_SIDE * PARENT_SIDE],
            albedo: vec![[100, 100, 100]; VERTEX_COUNT],
            roughness: vec![200; VERTEX_COUNT],
            temperature: vec![290.0; VERTEX_COUNT],
        };
        gx_core::surface::encode(&t, Compression::None)
    }

    /// A valid section with no matter, for either frame of the scene.
    fn empty_cell(key: &gx_core::key::CellKey) -> Vec<u8> {
        let extent = if key.frame_id == common::ROOT {
            common::ROOT_EXTENT
        } else {
            common::CHILD_EXTENT
        };
        let g = key.geometry(gx_core::units::Meters::new(extent));
        let section = gx_core::matter::Section::empty(*key, g.origin, g.edge).expect("valid");
        gx_core::matter::encode(&section, Compression::None)
    }

    #[test]
    fn settled_fires_after_the_last_tile_arrives_and_stats_once_a_second() {
        let mut main = stored_layer("main", 2, SectionMaker(Arc::new(empty_cell)));
        main.stored_surfaces = Some(SurfaceMaker(Arc::new(served)));
        let layers = vec![registry_layer("frames", &common::registry_bytes()), main];
        let s = std::f64::consts::FRAC_1_SQRT_2;
        let pose = Pose {
            frame_id: common::CHILD,
            position: [3.0 * SERVED_RADIUS, 0.0, 0.0],
            orientation: [0.0, s, 0.0, s],
            speed: 1.0,
        };
        let (events, heights) = run(layers, Some(pose), |e| {
            e.iter().any(|e| matches!(e, Event::Settled { .. }))
                && e.iter().filter(|e| matches!(e, Event::Stats(_))).count() >= 2
        });
        let settled = events
            .iter()
            .position(|e| matches!(e, Event::Settled { .. }))
            .expect("the view settles");
        // At the settle the surface is drawn and answers queries.
        let h = heights[0].expect("a tile is drawn under +x at the settle");
        assert!((h - 0.1).abs() < 0.01, "{h}");
        let stats: Vec<_> = events
            .iter()
            .filter_map(|e| match e {
                Event::Stats(s) => Some(*s),
                _ => None,
            })
            .collect();
        assert!(stats.len() >= 2);
        assert!(stats.last().unwrap().tiles_drawn > 0);
        assert!(settled > 0);
    }
}

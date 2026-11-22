//! The renderer API's camera items and events against the mock hub
//! (`docs/api.md` sections 4 and 8, proof in section 11).
//!
//! Each test loads the shared scene's registry from the mock hub and runs
//! the window's tick order without a window: the consumer, the queued
//! commands ([`Api::begin_tick`]), the flight input ([`Api::flight`] over
//! the keys), the simulation and the camera ([`renderer::camera::advance`]),
//! and the snapshot and events ([`Api::end_tick`]).

mod common;

use gx_core::frames::FrameSystem;
use gx_core::units::Seconds;
use renderer::api::{Api, Controls, Event, Pose};
use renderer::camera::{Camera, FlightInput, SPEED_LIMITS};
use renderer::config::ApiKey;
use renderer::controls::Controls as Keys;
use renderer::hub::HubClient;
use renderer::mock_hub::{registry_layer, MockHub, MockHubOptions, MOCK_API_KEY, MOCK_SPACE_ID};
use renderer::protocol::Routes;
use renderer::sim::Simulation;
use renderer::spawn::Spawner;
use renderer::stream::{Fetcher, FetcherOptions};
use std::sync::Arc;
use std::time::Duration;
use winit::keyboard::KeyCode;

/// Seconds per test tick.
const DT: f64 = 1.0 / 60.0;

/// The frame system of the shared scene, loaded from the mock hub.
fn hub_system() -> FrameSystem {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let hub = rt
        .block_on(MockHub::start(MockHubOptions::new(
            vec![registry_layer("frames", &common::registry_bytes())],
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
    fetcher.load_registry(Duration::from_secs(60)).unwrap()
}

/// The window's loop state without the window.
struct Loop {
    api: Api,
    sim: Simulation,
    camera: Camera,
    spawner: Spawner,
    keys: Keys,
    now: Seconds,
}

impl Loop {
    fn new() -> Loop {
        let system = hub_system();
        let now = system.tree().epoch();
        let camera = Camera::home(&system);
        Loop {
            api: Api::new(system.tree(), &camera),
            sim: Simulation::at_now(system, now),
            camera,
            spawner: Spawner::default(),
            keys: Keys::new(None),
            now,
        }
    }

    /// A key, as the window passes it on: only with the built-in controls.
    fn key(&mut self, code: KeyCode, pressed: bool) {
        if self.api.controls() == Controls::Builtin {
            self.keys
                .key(code, pressed, false, &mut self.sim, &mut self.camera);
        }
    }

    /// One tick in the window's order.
    fn tick(&mut self) {
        self.api.begin_tick(
            &self.sim,
            &mut self.camera,
            &mut self.spawner,
            &mut self.keys.attached,
        );
        let input = self.api.flight(self.keys.take_flight());
        renderer::camera::advance(
            &mut self.sim,
            &mut self.camera,
            DT,
            self.now,
            &input,
            self.keys.attached,
        );
        self.api.end_tick(&self.camera);
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
    l.api.set_camera(p).unwrap();
    assert_eq!(l.api.get_camera(), home, "getters read the last tick");
    l.tick();
    assert_eq!(l.api.get_camera(), p);
    assert_eq!(l.camera.cruise, 1.0);
}

#[test]
fn set_camera_errors() {
    let mut l = Loop::new();
    let unknown = l.api.set_camera(pose(99, [0.0; 3]));
    assert!(unknown.unwrap_err().0.contains("not in the registry"));
    let nan = l.api.set_camera(pose(common::ROOT, [f64::NAN, 0.0, 0.0]));
    assert!(nan.unwrap_err().0.contains("not finite"));
    let mut q = pose(common::ROOT, [0.0; 3]);
    q.orientation = [0.0, 0.0, 0.0, 1.0 + 2e-6];
    assert!(l
        .api
        .set_camera(q)
        .unwrap_err()
        .0
        .contains("unit quaternion"));
    q.orientation = [0.0, 0.0, 0.0, 1.0 + 5e-7];
    assert!(l.api.set_camera(q).is_ok(), "within 1e-6 is unit");
    let mut s = pose(common::ROOT, [0.0; 3]);
    s.speed = 0.0;
    assert!(l.api.set_camera(s).is_err());
}

#[test]
fn commands_apply_in_call_order_within_one_tick() {
    let mut l = Loop::new();
    l.tick();
    l.api
        .set_camera(pose(common::ROOT, [-20.0, 20.0, 20.0]))
        .unwrap();
    l.api.set_speed(5.0).unwrap();
    l.tick();
    assert_eq!(l.api.get_camera().speed, 5.0, "the speed after the pose");
    l.api.set_speed(7.0).unwrap();
    l.api
        .set_camera(pose(common::ROOT, [-21.0, 20.0, 20.0]))
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
        .set_camera(pose(common::ROOT, [-20.0, 20.0, 20.0]))
        .unwrap();
    l.tick();
    if by_key {
        l.key(KeyCode::KeyW, true);
        l.key(KeyCode::ShiftLeft, true);
    } else {
        l.api.set_flight_input(Some(FlightInput {
            forward: true,
            boost: true,
            ..FlightInput::default()
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
        .set_camera(pose(common::ROOT, [-20.0, 20.0, 20.0]))
        .unwrap();
    l.tick();
    l.api.set_flight_input(Some(FlightInput {
        right: true,
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
        .set_camera(pose(common::ROOT, [-20.0, 20.0, 20.0]))
        .unwrap();
    l.api
        .set_camera(pose(common::ROOT, [-21.0, 20.0, 20.0]))
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
        up: true,
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
        .set_camera(pose(common::ROOT, common::CHILD_OFFSET))
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

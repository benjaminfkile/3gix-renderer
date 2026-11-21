//! Light travel time in the client (`space-model.md` section 6.4), against
//! a mock registry of massless frames in straight-line motion, so every
//! state at every time is known in closed form:
//!
//! - a frame one light-hour from the observer is drawn where it was an
//!   hour earlier, both its position and its orientation, and its marker
//!   sits there too;
//! - a lit frame's lit side faces where the emitter was one light-travel
//!   time before the lit frame is seen, not where the observer sees the
//!   emitter;
//! - the frame nearest the crosshair reports its light time;
//! - through the whole draw path (selection, streaming, extraction, the
//!   draw list, and the scene), a mesh is lit from the emitter's retarded
//!   position for the mesh's frame.

mod common;

use gx_core::frames::FrameSystem;
use gx_core::gravity::C;
use gx_core::registry::{Frame, FrameTree, Registry, ROOT_PARENT};
use gx_core::units::{Kilograms, Meters, Quat, Seconds, Vec3};
use renderer::app::matter_scene;
use renderer::camera::{look_rotation, Camera};
use renderer::light::{irradiance, lights_for, PointLight};
use renderer::render::scene::build_scene;
use renderer::seen::Seen;
use renderer::sim::Simulation;
use renderer::stream::{decode_cell, FetchOutcome};
use renderer::world::World;

/// The registry epoch, seconds of TDB since J2000.
const EPOCH: f64 = 8.0e8;
/// One light-hour, meters.
const LIGHT_HOUR: f64 = C * 3600.0;
/// The far frame's speed, meters per second.
const SPEED: f64 = 1.0e5;
/// The far frame's spin, radians per second.
const SPIN: f64 = 1.0e-4;
/// The emitter's speed, meters per second.
const EMITTER_SPEED: f64 = 1.0e6;

const ROOT: u64 = 1;
const FAR: u64 = 2;
const EMITTER: u64 = 3;

fn frame(id: u64, parent: u64, position: Vec3, velocity: Vec3, spin: f64) -> Frame {
    Frame {
        frame_id: id,
        parent_frame_id: parent,
        root_extent: Meters::new(1.0e9),
        max_depth: 2,
        mass: Kilograms::new(0.0),
        position,
        velocity,
        orientation: Quat::identity(),
        angular_velocity: Vec3::new(0.0, 0.0, spin),
        j2: 0.0,
        reference_radius: Meters::new(0.0),
        principal_moments: Vec3::zero(),
    }
}

/// A massless root, the far frame one light-hour `D` out along `+x` moving
/// along `+y` and spinning about `z`, and an emitter one light-hour from it
/// along `-y`, at `(D, -D, 0)`, moving along `+x`. Nothing has mass, so
/// everything moves in straight lines.
fn system() -> FrameSystem {
    let reg = Registry::new(
        Seconds::new(EPOCH),
        vec![
            frame(ROOT, ROOT_PARENT, Vec3::zero(), Vec3::zero(), 0.0),
            frame(
                FAR,
                ROOT,
                Vec3::new(LIGHT_HOUR, 0.0, 0.0),
                Vec3::new(0.0, SPEED, 0.0),
                SPIN,
            ),
            frame(
                EMITTER,
                ROOT,
                Vec3::new(LIGHT_HOUR, -LIGHT_HOUR, 0.0),
                Vec3::new(EMITTER_SPEED, 0.0, 0.0),
                0.0,
            ),
        ],
    )
    .unwrap();
    FrameSystem::from_tree(FrameTree::from_registries(&[reg]).unwrap())
}

fn far_at(t: f64) -> Vec3 {
    Vec3::new(LIGHT_HOUR, SPEED * (t - EPOCH), 0.0)
}

fn emitter_at(t: f64) -> Vec3 {
    Vec3::new(LIGHT_HOUR + EMITTER_SPEED * (t - EPOCH), -LIGHT_HOUR, 0.0)
}

/// Solves `s = t - |x(s) - from| / c` by fixed-point iteration, the exact
/// answer for straight-line motion.
fn retarded(t: f64, from: Vec3, x: impl Fn(f64) -> Vec3) -> f64 {
    let mut s = t;
    for _ in 0..50 {
        s = t - (x(s) - from).length() / C;
    }
    s
}

/// The observer at the root origin looking along `+x`.
fn camera() -> Camera {
    Camera {
        frame_id: ROOT,
        position: Vec3::zero(),
        orientation: look_rotation(Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0)),
        speed: 1.0,
        cruise: 1.0,
    }
}

/// Four hours after the epoch, at now: late enough that the light from
/// the emitter that reaches the far frame an hour back left after the
/// origin.
fn sim() -> Simulation {
    Simulation::at_now(system(), Seconds::new(EPOCH + 14_400.0))
}

#[test]
fn a_frame_one_light_hour_away_is_drawn_an_hour_earlier() {
    let mut sim = sim();
    let t = sim.time().value();
    assert_eq!(t, EPOCH + 14_400.0);
    let camera = camera();
    sim.prepare_light(&camera);
    let seen = sim.seen(&camera);

    // The light time is an hour (plus the little the motion adds).
    let lt = seen.light_time(FAR).unwrap().value();
    let want = t - retarded(t, Vec3::zero(), far_at);
    assert!((lt - want).abs() < 1e-6, "{lt} vs {want}");
    assert!((lt - 3600.0).abs() < 1e-2, "{lt}");

    // Drawn where it was an hour ago, not where it is.
    let drawn = seen.root_position(FAR);
    let then = far_at(t - lt);
    assert!((drawn - then).length() < 1.0, "{drawn:?} vs {then:?}");
    let now = sim.system().root_position(FAR);
    assert!((now - far_at(t)).length() < 1.0);
    assert!((drawn - now).length() > 0.99 * SPEED * 3600.0);

    // Turned as it was an hour ago: the spin angle at the retarded time.
    let x_axis = seen
        .system()
        .root_orientation(FAR)
        .rotate(Vec3::new(1.0, 0.0, 0.0));
    let angle = x_axis.y.atan2(x_axis.x);
    let want = SPIN * (t - lt - EPOCH);
    assert!((angle - want).abs() < 1e-9, "{angle} vs {want}");

    // The marker sits at the drawn position, relative to the camera.
    let scene = build_scene(seen.system(), seen.camera(), 16.0 / 9.0);
    let marker = scene.markers.iter().find(|m| m.frame_id == FAR).unwrap();
    let rel = drawn - camera.root_position(sim.system());
    for (got, want) in marker.position.iter().zip([rel.x, rel.y, rel.z]) {
        assert!(
            (f64::from(*got) - want).abs() <= 1e-6 * LIGHT_HOUR,
            "{got} {want}"
        );
    }

    // The frame nearest the crosshair is the far one, an hour away.
    let (frame, light) = seen.nearest_to_crosshair().unwrap();
    assert_eq!(frame, FAR);
    assert_eq!(light.value(), lt);

    // With instantaneous light it would be drawn where it is now.
    let instant = Seen::instant(sim.system(), &camera);
    assert!((instant.root_position(FAR) - now).length() < 1.0);
}

#[test]
fn the_day_side_faces_the_emitters_retarded_position() {
    let mut sim = sim();
    let t = sim.time().value();
    let camera = camera();
    sim.prepare_light(&camera);
    let seen = sim.seen(&camera);
    let observer = camera.root_position(sim.system());

    // Where the observer sees the emitter, and the light at it.
    let emitter_seen = seen.root_position(EMITTER);
    let t_e = t - seen.light_time(EMITTER).unwrap().value();
    assert!((emitter_seen - emitter_at(t_e)).length() < 1.0);
    let rel = emitter_seen - observer;
    let light = PointLight {
        frame_id: EMITTER,
        position: [rel.x as f32, rel.y as f32, rel.z as f32],
        intensity: [1.0e26; 3],
    };

    // The lit frame is seen an hour back; the light reaching it then left
    // the emitter one more light-hour before that.
    let lit_root = seen.root_position(FAR);
    let t_lit = t - seen.light_time(FAR).unwrap().value();
    let t_emit = retarded(t_lit, lit_root, emitter_at);
    assert!((t_lit - t_emit - 3600.0).abs() < 10.0, "{t_lit} {t_emit}");
    let source = emitter_at(t_emit);

    let lit = lights_for(&[light], FAR, |l, e| seen.emitter_shift(l, e));
    let p = lit[0].position;
    let from_lit =
        Vec3::new(f64::from(p[0]), f64::from(p[1]), f64::from(p[2])) - (lit_root - observer);
    let want = source - lit_root;
    let angle = |a: Vec3, b: Vec3| {
        (a.dot(b) / (a.length() * b.length()))
            .clamp(-1.0, 1.0)
            .acos()
    };
    // The f32 positions hold about 1e-7 of a light-hour.
    assert!(angle(from_lit, want) < 1e-6, "{}", angle(from_lit, want));
    // That is not where the observer sees the emitter, nor where it is.
    let seen_dir = emitter_seen - lit_root;
    let now_dir = sim.system().root_position(EMITTER) - lit_root;
    assert!(angle(want, seen_dir) > 1e-3, "{}", angle(want, seen_dir));
    assert!(angle(want, now_dir) > 1e-3, "{}", angle(want, now_dir));

    // The lit side: a surface facing the retarded emitter takes the full
    // irradiance, one facing the observer's emitter a little less, and
    // the side turned away none.
    let point = lit_root - observer;
    let light_at = point + from_lit;
    let toward = want.normalized().unwrap();
    let full = irradiance([1.0e26; 3], light_at, point, toward)[1];
    let off = irradiance([1.0e26; 3], light_at, point, seen_dir.normalized().unwrap())[1];
    let unlit = irradiance([1.0e26; 3], light_at, point, toward.scale(-1.0))[1];
    assert!(full > off && off > 0.0, "{full} {off}");
    assert_eq!(unlit, 0.0);

    // Without light time there is no shift.
    let instant = Seen::instant(sim.system(), &camera);
    assert_eq!(instant.emitter_shift(FAR, EMITTER), Vec3::zero());
    assert_eq!(seen.emitter_shift(EMITTER, EMITTER), Vec3::zero());
}

#[test]
fn just_after_the_origin_frames_are_seen_from_before_it() {
    let t = EPOCH + 600.0;
    let mut sim = Simulation::at_now(system(), Seconds::new(t));
    let camera = camera();
    sim.prepare_light(&camera);
    let seen = sim.seen(&camera);
    // The light now arriving left about an hour before the origin. The
    // origin only starts the recorded time, so the frame is drawn where it
    // was then, not clamped to the origin.
    let f = seen.frame(FAR).unwrap();
    assert!(!f.light.clamped);
    let tr = f.light.time.value();
    assert!(
        tr < EPOCH - 2900.0 && tr > EPOCH - 3100.0,
        "retarded time {tr}"
    );
    // The distance the light crossed is the one to where the frame was then.
    let d = far_at(tr).length(); // the camera is at the root origin
    assert!((t - tr - d / C).abs() < 1e-6);
    assert!((seen.root_position(FAR) - far_at(tr)).length() < 1e-3);
}

#[test]
fn meshes_are_lit_from_the_emitters_retarded_position() {
    // The shared scene's hot blob (frame 1) and solid blob (frame 2), here
    // under a massless root a light-hour apart, the hot one moving.
    let hot = common::ROOT;
    let solid = common::CHILD;
    let root = 9;
    let extent = |id| match id {
        1 => common::ROOT_EXTENT,
        2 => common::CHILD_EXTENT,
        _ => 1.0e3,
    };
    let mut frames = vec![
        frame(root, ROOT_PARENT, Vec3::zero(), Vec3::zero(), 0.0),
        frame(
            hot,
            root,
            Vec3::new(LIGHT_HOUR, -LIGHT_HOUR, 0.0),
            Vec3::new(EMITTER_SPEED, 0.0, 0.0),
            0.0,
        ),
        frame(
            solid,
            root,
            Vec3::new(LIGHT_HOUR, 0.0, 0.0),
            Vec3::zero(),
            0.0,
        ),
    ];
    for f in &mut frames {
        f.root_extent = Meters::new(extent(f.frame_id));
        f.max_depth = 0;
    }
    // Only frames with mass take part in the far field, so the hot frame
    // gets one kilogram: its pull over a light-hour is far below anything
    // the straight-line answer below could notice.
    frames[1].mass = Kilograms::new(1.0);
    let reg = Registry::new(Seconds::new(EPOCH), frames).unwrap();
    let system = FrameSystem::from_tree(FrameTree::from_registries(&[reg]).unwrap());
    let mut sim = Simulation::at_now(system, Seconds::new(EPOCH + 14_400.0));
    let t = sim.time().value();
    // 30 m above the solid blob, looking down at it.
    let camera = Camera {
        frame_id: solid,
        position: Vec3::new(0.0, 0.0, 30.0),
        orientation: look_rotation(Vec3::new(0.0, 0.0, -1.0), Vec3::new(0.0, 1.0, 0.0)),
        speed: 1.0,
        cruise: 1.0,
    };
    sim.prepare_light(&camera);
    let seen = sim.seen(&camera);
    let size = (320, 180);
    let chunks = common::chunks();
    let mut world = World::new(2);
    world.select(seen.system(), seen.camera(), size, 0.0, true);
    for round in 0..4 {
        for key in world.take_requests(f64::from(round)) {
            let extent = seen.system().tree().get(key.frame_id).unwrap().root_extent;
            let outcome = match chunks.get(&key) {
                Some(bytes) => decode_cell(&key, bytes, extent),
                None => FetchOutcome::Decoded(None),
            };
            world.complete(key, outcome, f64::from(round));
        }
        world.finish_extraction();
    }
    assert!(world.settled());
    let scene = matter_scene(&mut world, &seen, size, 0.0);
    assert_eq!(scene.lights.len(), 1);
    assert_eq!(scene.lights[0].frame_id, hot);
    let surface = scene
        .surfaces
        .iter()
        .find(|d| d.mesh.key.frame_id == solid)
        .expect("the solid blob is drawn");

    let observer = camera.root_position(sim.system());
    let lit_root = seen.root_position(solid);
    let t_lit = t - seen.light_time(solid).unwrap().value();
    let source = emitter_at(retarded(t_lit, lit_root, emitter_at));
    let v = |p: [f32; 3]| Vec3::new(f64::from(p[0]), f64::from(p[1]), f64::from(p[2]));
    let angle = |a: Vec3, b: Vec3| {
        (a.dot(b) / (a.length() * b.length()))
            .clamp(-1.0, 1.0)
            .acos()
    };
    let lit_rel = lit_root - observer;
    let mesh_light = v(surface.lights[0]) - lit_rel;
    let want = source - lit_root;
    assert!(
        angle(mesh_light, want) < 1e-6,
        "{}",
        angle(mesh_light, want)
    );
    // That is an hour of the emitter's motion away from where it is now.
    let now_dir = sim.system().root_position(hot) - lit_root;
    assert!(angle(now_dir, want) > 1e-3, "{}", angle(now_dir, want));
    // The observer floats next to the solid blob, so it sees the emitter
    // where the blob is lit from: the scene's light agrees.
    let seen_light = v(scene.lights[0].position) - lit_rel;
    assert!(
        angle(seen_light, want) < 1e-6,
        "{}",
        angle(seen_light, want)
    );
}

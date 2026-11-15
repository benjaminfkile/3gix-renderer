//! Walk mode and spawning end to end, without a hub: a mock layer answers
//! every cell the world requests at once, and the observer runs through
//! the same frame loop as the window ([`Observer::update`]).
//!
//! - A solid ball: spawn, come to rest within 2 s, walk 10 m with `W`, in contact
//!   the whole way.
//! - A gas body whose density rises toward its center: spawn, sink, and
//!   come to rest where its density equals the observer's.
//! - Toggling between fly and walk keeps the eye and the velocity.

use gx_core::frames::FrameSystem;
use gx_core::gravity::G;
use gx_core::key::CellKey;
use gx_core::matter::{encode, Compression, Sample, Samples, Section, State};
use gx_core::registry::{Frame, FrameTree, Registry, ROOT_PARENT};
use gx_core::units::{Attenuation, Density, Kelvin, Kilograms, Meters, Quat, Ratio, Seconds, Vec3};
use renderer::camera::{look_rotation, Camera, FlightInput};
use renderer::sim::{SimClock, Simulation};
use renderer::stream::decode_cell;
use renderer::walk::{spawn_by_index, spawn_frames, Observer, EYE_HEIGHT_FRACTION, WALK_SPEED};
use renderer::world::World;

/// The frame holding the matter.
const FRAME: u64 = 1;

/// Samples per axis of every mock section.
const RES: u8 = 16;

/// Frame duration of the loop, seconds.
const DT: f64 = 1.0 / 60.0;

/// A frame of edge `extent` and `max_depth`, with `mass` and a spin of
/// `spin` radians per second about its z axis, as the only frame.
fn system(extent: f64, max_depth: u8, mass: f64, spin: f64) -> FrameSystem {
    let frame = Frame {
        frame_id: FRAME,
        parent_frame_id: ROOT_PARENT,
        root_extent: Meters::new(extent),
        max_depth,
        mass: Kilograms::new(mass),
        position: Vec3::zero(),
        velocity: Vec3::zero(),
        orientation: Quat::identity(),
        angular_velocity: Vec3::new(0.0, 0.0, spin),
        j2: 0.0,
        reference_radius: Meters::new(0.0),
        principal_moments: Vec3::zero(),
    };
    let r = Registry::new(Seconds::new(0.0), vec![frame]).unwrap();
    FrameSystem::from_tree(FrameTree::from_registries(&[r]).unwrap())
}

/// A simulation in the past of a far now, playing at rate 1.
fn simulation(system: FrameSystem) -> (Simulation, Seconds) {
    let now = Seconds::new(1.0e9);
    let clock = SimClock::starting_at(Seconds::new(0.0), now, Seconds::new(0.0), 1.0);
    (Simulation::new(system, clock), now)
}

fn matter(density: f64, state: State) -> Sample {
    Sample {
        density: Density::new(density),
        state,
        temperature: Kelvin::new(300.0),
        albedo: [Ratio::new(0.4); 3],
        roughness: Ratio::new(0.8),
        attenuation: Attenuation::new(1.0e-3),
    }
}

/// A mock layer: the section of `key`, from the density and state `field`
/// gives at each sample center for samples of edge `step`. Cells farther
/// than `reach` from the origin at every point are empty.
fn section(
    key: CellKey,
    extent: f64,
    reach: f64,
    field: &dyn Fn(Vec3, f64) -> (f64, State),
) -> Section {
    let g = key.geometry(Meters::new(extent));
    let e = g.edge.value();
    let gap = |o: f64| o.max(-(o + e)).max(0.0);
    let nearest = Vec3::new(gap(g.origin.x), gap(g.origin.y), gap(g.origin.z)).length();
    if nearest > reach {
        return Section::empty(key, g.origin, g.edge).unwrap();
    }
    let step = e / f64::from(RES);
    let samples = Samples::from_fn(RES, |x, y, z| {
        let c = |o: f64, n: u32| o + (f64::from(n) + 0.5) * step;
        let p = Vec3::new(c(g.origin.x, x), c(g.origin.y, y), c(g.origin.z, z));
        match field(p, step) {
            (d, state) if d > 0.0 => matter(d, state),
            _ => Sample::VACUUM,
        }
    });
    Section::new(key, g.origin, g.edge, RES, samples).unwrap()
}

/// Answers every cell the world wants with the mock layer.
fn serve(world: &mut World, extent: f64, layer: &dyn Fn(CellKey) -> Section, now: f64) {
    let keys = world.take_requests(now);
    answer(world, extent, layer, keys, now);
}

/// Answers the requests for `keys` with the mock layer.
fn answer(
    world: &mut World,
    extent: f64,
    layer: &dyn Fn(CellKey) -> Section,
    keys: Vec<CellKey>,
    now: f64,
) {
    for key in keys {
        let bytes = encode(&layer(key), Compression::None);
        let outcome = decode_cell(&key, &[bytes], Meters::new(extent));
        world.complete(key, outcome, now);
    }
}

/// A camera in the frame at `position`, looking at the frame origin.
fn camera_at(position: Vec3) -> Camera {
    Camera {
        frame_id: FRAME,
        position,
        orientation: look_rotation(-position, Vec3::new(0.0, 1.0, 0.0)),
        speed: 1.0,
        cruise: 1.0,
    }
}

/// Everything one test runs.
struct Run<'a> {
    sim: Simulation,
    now: Seconds,
    world: World,
    observer: Observer,
    camera: Camera,
    extent: f64,
    layer: &'a dyn Fn(CellKey) -> Section,
    t: f64,
    /// The layer answers requests; while `false` they stay in flight.
    serving: bool,
    /// Requests the layer has not answered.
    in_flight: Vec<CellKey>,
}

impl Run<'_> {
    /// Surveys the frames and spawns onto the first with matter.
    fn spawn(&mut self) {
        self.world.survey(self.sim.system(), self.t);
        serve(&mut self.world, self.extent, self.layer, self.t);
        let frames = spawn_frames(self.world.cache(), self.sim.system().tree());
        assert_eq!(frames, vec![FRAME]);
        let (system, cache) = (self.sim.system(), self.world.cache());
        assert!(spawn_by_index(
            &mut self.observer,
            system,
            cache,
            0,
            &mut self.camera
        ));
        assert!(self.observer.walking());
    }

    /// One frame of `dt` seconds with `input`, as the window loop runs it.
    fn frame(&mut self, input: &FlightInput, dt: f64) {
        self.observer.update(
            &mut self.sim,
            dt,
            self.now,
            self.world.cache(),
            input,
            false,
            &mut self.camera,
        );
        self.t += dt;
        let priority = self.observer.priority_cells(self.sim.system());
        self.world.set_priority(&priority, self.t);
        self.in_flight.extend(self.world.take_requests(self.t));
        if self.serving {
            let keys = std::mem::take(&mut self.in_flight);
            answer(&mut self.world, self.extent, self.layer, keys, self.t);
        }
    }

    fn feet(&self) -> Vec3 {
        let w = self.observer.walker.as_ref().unwrap();
        w.feet_in_frame(self.sim.system())
    }
}

/// Radius of the solid ball, meters.
const BALL_RADIUS: f64 = 50.0;
/// Surface gravity of the ball, m/s^2.
const BALL_GRAVITY: f64 = 9.8;

/// The ball's mock layer: the mean density over a sample is the share of
/// it inside the ball, ramping across two sample edges at the surface.
fn ball_layer(extent: f64) -> impl Fn(CellKey) -> Section {
    let ball = |p: Vec3, step: f64| {
        let inside = (0.5 - (p.length() - BALL_RADIUS) / (2.0 * step)).clamp(0.0, 1.0);
        (3000.0 * inside, State::Solid)
    };
    move |k: CellKey| section(k, extent, BALL_RADIUS + 1.0, &ball)
}

#[test]
fn waits_in_place_until_matter_arrives() {
    let extent = 128.0;
    let mass = BALL_GRAVITY * BALL_RADIUS * BALL_RADIUS / G;
    let (sim, now) = simulation(system(extent, 4, mass, 1.0e-3));
    let layer = ball_layer(extent);
    let mut run = Run {
        sim,
        now,
        world: World::new(0),
        observer: Observer::new(),
        camera: camera_at(Vec3::new(0.0, -40.0, 300.0)),
        extent,
        layer: &layer,
        t: 0.0,
        serving: false,
        in_flight: Vec::new(),
    };
    run.spawn();
    let start = run.feet();
    let idle = FlightInput::default();
    for _ in 0..60 {
        run.frame(&idle, DT);
    }
    let w = run.observer.walker.as_ref().unwrap();
    assert!(w.held);
    assert!((run.feet() - start).length() < 1e-9, "moved while held");
    let lines = run.observer.status_lines(run.sim.system());
    assert!(
        lines.iter().any(|l| l.contains("waiting for matter")),
        "{lines:?}"
    );
    // The cells around the feet are asked for first, coarsest first.
    let priority = run.observer.priority_cells(run.sim.system());
    assert!((0..=4).all(|d| priority.iter().any(|k| k.depth == d)));
    assert_eq!(priority.iter().filter(|k| k.depth == 4).count(), 8);
    let asked = run.world.take_requests(run.t);
    assert!(asked.is_empty(), "everything is in flight already");
    for k in priority.iter().filter(|k| k.depth > 1) {
        let state = run.world.cache().state(k);
        assert_eq!(state, Some(&renderer::stream::CellState::Requested), "{k}");
    }

    // The requests in flight are answered by the next frame.
    run.serving = true;
    for _ in 0..(2.0 / DT) as usize {
        run.frame(&idle, DT);
    }
    let w = run.observer.walker.as_ref().unwrap();
    assert!(!w.held);
    assert!(w.state.contact.is_some());
    let r = run.feet().length();
    assert!((r - BALL_RADIUS).abs() < 0.1, "rests at radius {r}");
}

#[test]
fn spawns_settles_and_walks_on_a_solid_ball() {
    let extent = 128.0;
    let mass = BALL_GRAVITY * BALL_RADIUS * BALL_RADIUS / G;
    let (sim, now) = simulation(system(extent, 4, mass, 1.0e-3));
    let layer = ball_layer(extent);
    let mut run = Run {
        sim,
        now,
        world: World::new(0),
        observer: Observer::new(),
        camera: camera_at(Vec3::new(40.0, 10.0, 300.0)),
        extent,
        layer: &layer,
        t: 0.0,
        serving: true,
        in_flight: Vec::new(),
    };
    run.spawn();
    let start = run.feet().length();
    assert!(
        (start - BALL_RADIUS - 2.0).abs() < 0.5,
        "spawned at radius {start}"
    );

    // Come down and settle within 2 s.
    let idle = FlightInput::default();
    for _ in 0..(2.0 / DT) as usize {
        run.frame(&idle, DT);
    }
    let w = run.observer.walker.as_ref().unwrap();
    assert!(!w.held, "still waiting for matter");
    assert!(w.state.contact.is_some(), "not in contact after 2 s");
    let r = run.feet().length();
    assert!((r - BALL_RADIUS).abs() < 0.1, "rests at radius {r}");
    let speed = w.speed_over_matter(run.sim.system());
    assert!(speed < 0.05, "still moving at {speed} m/s");

    // The camera is at eye height above the feet.
    let eye = run.camera.position;
    let height = (eye - run.feet()).length();
    let expected = EYE_HEIGHT_FRACTION * w.params.height.value();
    assert!(
        (height - expected).abs() < 1e-6,
        "eye {height} m above the feet"
    );

    // Walk 10 m with W, in contact every frame.
    let from = run.feet();
    let walk = FlightInput {
        forward: true,
        ..FlightInput::default()
    };
    let mut frames = 0;
    while (run.feet() - from).length() < 10.0 {
        run.frame(&walk, DT);
        frames += 1;
        let w = run.observer.walker.as_ref().unwrap();
        assert!(
            w.state.contact.is_some(),
            "lost contact after {frames} frames"
        );
        let r = run.feet().length();
        assert!((r - BALL_RADIUS).abs() < 0.1, "walking at radius {r}");
        assert!(frames < 1200, "did not walk 10 m in 20 s");
    }
    let speed = run
        .observer
        .walker
        .as_ref()
        .unwrap()
        .speed_over_matter(run.sim.system());
    assert!((speed - WALK_SPEED).abs() < 0.05, "walking at {speed} m/s");
}

/// Outer radius of the gas body, meters.
const GAS_RADIUS: f64 = 200.0;
/// Density at the gas body's center, kilograms per cubic meter.
const GAS_CENTER_DENSITY: f64 = 1970.0;

#[test]
fn sinks_into_gas_until_buoyancy_holds_it() {
    let extent = 512.0;
    // Gravity of 10 m/s^2 at half the radius, where the density is half the
    // center's, 985 kg/m^3: the observer's density.
    let r_eq = 0.5 * GAS_RADIUS;
    let mass = 10.0 * r_eq * r_eq / G;
    let (sim, now) = simulation(system(extent, 3, mass, 0.0));
    let gas = |p: Vec3, _: f64| {
        let r = p.length();
        let d = (GAS_CENTER_DENSITY * (1.0 - r / GAS_RADIUS)).max(0.0);
        (d, State::Gas)
    };
    let layer = |k: CellKey| section(k, extent, GAS_RADIUS + 1.0, &gas);
    let mut run = Run {
        sim,
        now,
        world: World::new(0),
        observer: Observer::new(),
        camera: camera_at(Vec3::new(0.0, 0.0, 2000.0)),
        extent,
        layer: &layer,
        t: 0.0,
        serving: true,
        in_flight: Vec::new(),
    };
    run.spawn();
    let density = run.observer.params.density.value();
    assert_eq!(density, 0.5 * GAS_CENTER_DENSITY);
    let start = run.feet().length();
    assert!(start > GAS_RADIUS, "spawned at radius {start}");

    let idle = FlightInput::default();
    let step = 0.25;
    let mut deepest = start;
    for _ in 0..(900.0 / step) as usize {
        run.frame(&idle, step);
        deepest = deepest.min(run.feet().length());
    }
    assert!(deepest < GAS_RADIUS - 50.0, "never sank: {deepest}");
    let w = run.observer.walker.as_ref().unwrap();
    assert!(w.state.contact.is_none());
    let r = run.feet().length();
    // Depth below the outer radius where the density is the observer's.
    let depth = GAS_RADIUS - r;
    let expected = GAS_RADIUS - r_eq;
    assert!(
        (depth - expected).abs() <= 0.01 * expected,
        "rests {depth} m deep, expected {expected} m"
    );
    let speed = w.speed_over_matter(run.sim.system());
    assert!(speed < 0.01, "still moving at {speed} m/s");
}

#[test]
fn toggling_keeps_the_eye_and_the_velocity() {
    // A massless root and a moving, turned child frame.
    let mut frames = vec![
        Frame {
            frame_id: 1,
            parent_frame_id: ROOT_PARENT,
            root_extent: Meters::new(1.0e6),
            max_depth: 0,
            mass: Kilograms::new(0.0),
            position: Vec3::zero(),
            velocity: Vec3::zero(),
            orientation: Quat::identity(),
            angular_velocity: Vec3::zero(),
            j2: 0.0,
            reference_radius: Meters::new(0.0),
            principal_moments: Vec3::zero(),
        };
        2
    ];
    frames[1].frame_id = 2;
    frames[1].parent_frame_id = 1;
    frames[1].root_extent = Meters::new(1.0e3);
    frames[1].position = Vec3::new(1.0e5, 0.0, 0.0);
    frames[1].velocity = Vec3::new(1000.0, 0.0, 0.0);
    frames[1].mass = Kilograms::new(1.0e12);
    frames[1].orientation = Quat::new(0.0, 0.0, 0.5f64.sqrt(), 0.5f64.sqrt());
    let registry = Registry::new(Seconds::new(0.0), frames).unwrap();
    let system = FrameSystem::from_tree(FrameTree::from_registries(&[registry]).unwrap());

    let mut camera = Camera {
        frame_id: 2,
        position: Vec3::new(10.0, 20.0, 30.0),
        orientation: look_rotation(Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0)),
        speed: 3.0,
        cruise: 3.0,
    };
    let eye = camera.root_position(&system);
    let mut observer = Observer {
        drift: Vec3::new(5.0, 6.0, 7.0),
        ..Observer::new()
    };
    let keys = FlightInput {
        forward: true,
        ..FlightInput::default()
    };
    let flight = camera.flight_velocity(&system, &keys);
    assert!((flight.length() - 3.0).abs() < 1e-12);
    let velocity = observer.drift + flight;

    observer.toggle(&system, &mut camera, flight);
    let w = observer.walker.clone().expect("walking");
    assert!((camera.root_position(&system) - eye).length() < 1e-9);
    assert!((w.root_velocity(&system) - velocity).length() < 1e-9);
    let relative = velocity - system.root_velocity(2);
    assert!((w.state.velocity - relative).length() < 1e-9);
    assert_eq!(observer.drift, Vec3::zero());
    let below = w.eye(&system) - w.state.position;
    assert!((below.length() - EYE_HEIGHT_FRACTION * w.params.height.value()).abs() < 1e-9);

    observer.toggle(&system, &mut camera, Vec3::zero());
    assert!(!observer.walking());
    assert!((camera.root_position(&system) - eye).length() < 1e-9);
    assert!((observer.drift - velocity).length() < 1e-9);
}

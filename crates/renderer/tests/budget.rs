//! The performance budget measured in CI without a hub (`rendering.md`
//! section 5.3): synthetic worlds served by the mock hub with zero latency
//! and every section already stored, streamed through the real fetcher
//! into the same [`matter_step`] the window and the headless run use.
//!
//! - **Frame step.** At 4096 selected cells the frame thread's matter work
//!   per step is at most 4 ms at the 95th percentile, over at least 100
//!   steps of a moving camera (`rendering.md` section 4). Only a release
//!   build is timed: `cargo test --release -p renderer --test budget`, as
//!   `scripts/ci.sh` runs it. A debug build runs the same steps and checks
//!   everything else.
//! - **Rounds.** Refinement to depth `D` takes at most `ceil(D / 3) + 2`
//!   selection rounds (section 3.2), for a plane, a solid ball, a hollow
//!   shell, a ring, and an overhang. One round is one step that selects
//!   and requests, then everything requested arriving.
//! - **No shape model.** Once settled, no drawn cell is held back by
//!   anything but the selection rule (section 3.1): every drawn cell the
//!   selection's geometry would refine is one whose occupancy does not
//!   cross 0.5. In particular the far side of the hollow shell, behind its
//!   near side, and the ring's far rim, seen through its hole, are refined
//!   like any other surface.
//!
//! - **Transvoxel regular cells.** Extraction of one resolution 32 cell
//!   crossed by a plane, with its 26 same-depth neighbors, timed over 50
//!   runs; the median and 95th percentile are printed, no limit is
//!   asserted yet.
//! - **Transvoxel transition cells.** Extraction of one resolution 32 cell
//!   crossed by a plane, with neighbors one depth finer across one face
//!   and across three faces, timed over 50 runs each; the median and 95th
//!   percentile are printed, no limit is asserted yet.
//!
//! The watertightness proof of section 2.5 belongs to surface extraction
//! and comes with it.
//!
//! The shapes are signed distance functions in the frame's coordinates;
//! each sample's occupancy ramps linearly from 1 to 0 across one sample
//! around the boundary, so the 0.5 level is exactly the shape's surface
//! (`matter-format.md` section 3.3). Nothing here is shaped like any
//! particular body.

use gx_core::frames::FrameSystem;
use gx_core::key::CellKey;
use gx_core::matter::{encode, Compression, Sample, Samples, Section, State};
use gx_core::registry::{self, Frame, FrameTree, Registry, ROOT_PARENT};
use gx_core::units::{Attenuation, Density, Kelvin, Kilograms, Meters, Quat, Ratio, Seconds, Vec3};
use renderer::app::matter_step;
use renderer::camera::{look_rotation, Camera};
use renderer::config::ApiKey;
use renderer::extract::default_threads;
use renderer::hub::HubClient;
use renderer::mock_hub::{
    registry_layer, stored_layer, MockHub, MockHubOptions, SectionMaker, MOCK_API_KEY,
    MOCK_SPACE_ID,
};
use renderer::protocol::Routes;
use renderer::seen::Seen;
use renderer::stream::{wants_refinement, Fetcher, FetcherOptions, MAX_CELLS_PER_FRAME};
use renderer::world::World;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// The one frame of every synthetic world.
const FRAME: u64 = 7;

/// Its cube edge, meters.
const EXTENT: f64 = 200_000.0;

/// Samples per axis of every section.
const RESOLUTION: u8 = 16;

/// The window size of the views, pixels.
const VIEW: (u32, u32) = (1280, 720);

/// Tests share the machine's cores; the timed one must run alone.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

/// A shape: the signed distance to its surface, meters, negative inside.
/// Every function here is exact or underestimates the distance, so a cell
/// farther from the surface than its half diagonal is uniform.
#[derive(Copy, Clone, Debug)]
enum Shape {
    /// Matter below a plane at `z = h`.
    Plane,
    /// A solid ball.
    Ball,
    /// A hollow shell: matter between two concentric spheres.
    Shell,
    /// A ring around the z axis (a torus).
    Ring,
    /// Matter below a plane, and a slab above it held up by a pillar.
    Overhang,
}

/// A point off the grid's planes, so no surface lies on a cell face.
const CENTER: [f64; 3] = [0.0123 * EXTENT, -0.0217 * EXTENT, 0.0171 * EXTENT];

impl Shape {
    fn distance(self, p: Vec3) -> f64 {
        let c = Vec3::new(CENTER[0], CENTER[1], CENTER[2]);
        let q = p - c;
        match self {
            Shape::Plane => q.z,
            Shape::Ball => q.length() - 0.3 * EXTENT,
            Shape::Shell => (q.length() - 0.3 * EXTENT).abs() - 0.04 * EXTENT,
            Shape::Ring => {
                let r = q.x.hypot(q.y) - 0.3 * EXTENT;
                r.hypot(q.z) - 0.06 * EXTENT
            }
            Shape::Overhang => {
                let ground = q.z + 0.2 * EXTENT;
                let slab = boxed(q - Vec3::new(0.0, 0.0, 0.08 * EXTENT), [0.3, 0.3, 0.035]);
                let pillar = boxed(
                    q - Vec3::new(-0.2 * EXTENT, 0.0, -0.06 * EXTENT),
                    [0.05, 0.05, 0.15],
                );
                ground.min(slab).min(pillar)
            }
        }
    }

    fn name(self) -> &'static str {
        match self {
            Shape::Plane => "plane",
            Shape::Ball => "ball",
            Shape::Shell => "shell",
            Shape::Ring => "ring",
            Shape::Overhang => "overhang",
        }
    }
}

/// The signed distance to a box of half extents `half` (fractions of
/// [`EXTENT`]) centered on the origin.
fn boxed(p: Vec3, half: [f64; 3]) -> f64 {
    let d = [
        p.x.abs() - half[0] * EXTENT,
        p.y.abs() - half[1] * EXTENT,
        p.z.abs() - half[2] * EXTENT,
    ];
    let outside = Vec3::new(d[0].max(0.0), d[1].max(0.0), d[2].max(0.0)).length();
    outside + d[0].max(d[1]).max(d[2]).min(0.0)
}

/// The section of `key` for `shape`: solid where the shape is, occupancy
/// ramping across one sample at its surface.
fn section_bytes(shape: Shape, key: &CellKey) -> Vec<u8> {
    let g = key.geometry(Meters::new(EXTENT));
    let e = g.edge.value();
    let n = RESOLUTION;
    let h = e / f64::from(n);
    let half = 0.5 * e;
    let center = g.origin + Vec3::new(half, half, half);
    let d = shape.distance(center);
    let reach = half * 3f64.sqrt() + h;
    let solid = |occupancy: f64| Sample {
        density: Density::new(2500.0 * occupancy),
        state: State::Solid,
        temperature: Kelvin::new(290.0),
        albedo: [Ratio::new(0.4); 3],
        roughness: Ratio::new(0.8),
        attenuation: Attenuation::new(0.0),
        radiating_temperature: Kelvin::new(290.0),
        occupancy: Ratio::new(occupancy),
    };
    let section = if d > reach {
        Section::empty(*key, g.origin, g.edge).expect("valid")
    } else if d < -reach {
        Section::new(*key, g.origin, g.edge, n, Samples::filled(n, solid(1.0))).expect("valid")
    } else {
        let samples = Samples::from_fn(n, |x, y, z| {
            let at = |o: f64, i: u32| o + (f64::from(i) + 0.5) * h;
            let p = Vec3::new(at(g.origin.x, x), at(g.origin.y, y), at(g.origin.z, z));
            let occupancy = (0.5 - shape.distance(p) / h).clamp(0.0, 1.0);
            if occupancy < 1e-6 {
                Sample::VACUUM
            } else {
                solid(occupancy)
            }
        });
        Section::new(*key, g.origin, g.edge, n, samples).expect("valid")
    };
    encode(&section, Compression::Zstd)
}

fn registry_bytes() -> Vec<u8> {
    let frame = Frame {
        frame_id: FRAME,
        parent_frame_id: ROOT_PARENT,
        root_extent: Meters::new(EXTENT),
        max_depth: 0,
        mass: Kilograms::new(1.0e15),
        position: Vec3::zero(),
        velocity: Vec3::zero(),
        orientation: Quat::identity(),
        angular_velocity: Vec3::zero(),
        j2: 0.0,
        reference_radius: Meters::new(0.0),
        principal_moments: Vec3::zero(),
    };
    let reg = Registry::new(Seconds::new(0.0), vec![frame]).expect("a valid registry");
    // The frame system the client builds checks this registry too.
    FrameTree::from_registries(std::slice::from_ref(&reg)).expect("a valid tree");
    registry::encode(&reg)
}

/// A mock hub serving `shape` to depth `max_depth`, and a client of it.
struct Rig {
    _hub: MockHub,
    fetcher: Fetcher,
    system: FrameSystem,
    _rt: tokio::runtime::Runtime,
}

fn rig(shape: Shape, max_depth: u8) -> Rig {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let make = SectionMaker(Arc::new(move |k: &CellKey| section_bytes(shape, k)));
    let options = MockHubOptions::new(
        vec![
            registry_layer("frames", &registry_bytes()),
            stored_layer("matter", max_depth, make),
        ],
        Vec::new(),
    );
    let hub = rt.block_on(MockHub::start(options)).unwrap();
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
    assert_eq!(system.tree().get(FRAME).unwrap().max_depth, max_depth);
    Rig {
        _hub: hub,
        fetcher,
        system,
        _rt: rt,
    }
}

/// A camera at `position` (frame coordinates) looking at `target`.
fn camera(position: Vec3, target: Vec3) -> Camera {
    let up = if (target - position)
        .normalized()
        .is_some_and(|f| f.z.abs() > 0.99)
    {
        Vec3::new(0.0, 1.0, 0.0)
    } else {
        Vec3::new(0.0, 0.0, 1.0)
    };
    Camera {
        frame_id: FRAME,
        position,
        orientation: look_rotation(target - position, up),
        speed: 0.0,
        cruise: 0.0,
    }
}

/// Where each shape is looked at from, and the depth the view reaches: the
/// camera is placed a fraction of a deepest cell's edge from the surface,
/// so the cells around it refine to the frame's limit.
fn view_of(shape: Shape, depth: u8) -> Camera {
    let c = Vec3::new(CENTER[0], CENTER[1], CENTER[2]);
    let deepest = EXTENT / f64::from(1u32 << depth);
    let gap = 0.5 * deepest;
    match shape {
        Shape::Plane => camera(
            c + Vec3::new(0.0, 0.0, gap),
            c + Vec3::new(0.3 * EXTENT, 0.1 * EXTENT, -0.05 * EXTENT),
        ),
        Shape::Ball | Shape::Shell => {
            // Outside, just above the outer surface, looking across the
            // ball's center: the shell's far side lies behind its near
            // side.
            let outer = if matches!(shape, Shape::Shell) {
                0.34 * EXTENT
            } else {
                0.3 * EXTENT
            };
            camera(
                c + Vec3::new(outer + gap, 0.0, 0.0),
                c + Vec3::new(-0.1 * EXTENT, 0.05 * EXTENT, 0.0),
            )
        }
        Shape::Ring => {
            // Just above the ring's top, looking down through its hole at
            // the far rim.
            camera(
                c + Vec3::new(0.3 * EXTENT, 0.0, 0.06 * EXTENT + gap),
                c + Vec3::new(-0.3 * EXTENT, 0.0, -0.04 * EXTENT),
            )
        }
        Shape::Overhang => {
            // Under the slab, next to the pillar, looking out.
            camera(
                c + Vec3::new(-0.1 * EXTENT, 0.0, 0.045 * EXTENT - gap),
                c + Vec3::new(0.4 * EXTENT, 0.1 * EXTENT, -0.1 * EXTENT),
            )
        }
    }
}

/// Waits until nothing requested is still on its way.
fn wait_idle(rig: &mut Rig) {
    let start = Instant::now();
    while rig.fetcher.busy() {
        rig.fetcher.wait_for_events(Duration::from_millis(5));
        assert!(
            start.elapsed() < Duration::from_secs(120),
            "the fetcher hangs"
        );
    }
}

/// Steps in rounds until the drawn selection is the target with nothing
/// outstanding. One round: a step applies what arrived and starts a
/// selection, the selection lands, a second step at the same moment asks
/// for what it needs, and everything asked for arrives. Returns the
/// selection rounds it took.
fn settle_in_rounds(rig: &mut Rig, world: &mut World, cam: &Camera, view: (u32, u32)) -> u64 {
    let seen = Seen::instant(&rig.system, cam);
    let mut now = 0.0;
    for _ in 0..64 {
        now += 0.016;
        matter_step(world, &mut rig.fetcher, &seen, view, now);
        world.wait_selection();
        matter_step(world, &mut rig.fetcher, &seen, view, now);
        let c = world.cache();
        if c.at_target() && c.outstanding() == 0 && c.counts().in_flight == 0 {
            return world.selection_rounds();
        }
        wait_idle(rig);
    }
    panic!("the selection did not settle in 64 rounds");
}

/// Every drawn cell that the selection's geometry would refine does not
/// cross 0.5: nothing but the selection rule holds a cell back.
fn assert_nothing_held_back(rig: &Rig, world: &World, cam: &Camera, view: (u32, u32)) {
    let seen = Seen::instant(&rig.system, cam);
    let view = (f64::from(view.0), f64::from(view.1));
    let c = world.cache();
    assert!(
        c.selection().len() < MAX_CELLS_PER_FRAME,
        "the cell cap held cells back"
    );
    for s in c.selection() {
        let wanted = wants_refinement(seen.system(), seen.camera(), view, &s.key);
        assert!(
            !(wanted && c.refinable(&s.key)),
            "cell {} could be refined and is not",
            s.key
        );
    }
}

/// Rounds per shape to the frame's depth limit, and nothing held back.
#[test]
fn refinement_rounds_and_no_shape_model() {
    let _alone = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    for depth in [5u8, 9, 12] {
        rounds_at_depth(depth);
    }
}

/// [`refinement_rounds_and_no_shape_model`] for frames `depth` deep.
fn rounds_at_depth(depth: u8) {
    let bound = u64::from(depth.div_ceil(3)) + 2;
    for shape in [
        Shape::Plane,
        Shape::Ball,
        Shape::Shell,
        Shape::Ring,
        Shape::Overhang,
    ] {
        let mut rig = rig(shape, depth);
        let cam = view_of(shape, depth);
        let mut world = World::new(default_threads());
        let rounds = settle_in_rounds(&mut rig, &mut world, &cam, VIEW);
        let c = world.cache();
        let deepest = world.deepest_selected().unwrap();
        println!(
            "budget: {} rounds {rounds} (bound {bound}) depth {deepest} cells {} requested ahead {}",
            shape.name(),
            c.selection().len(),
            c.requested_ahead()
        );
        assert_eq!(deepest, depth, "{} reaches the depth limit", shape.name());
        assert!(
            rounds <= bound,
            "{}: {rounds} rounds to depth {depth}, at most {bound}",
            shape.name()
        );
        assert_nothing_held_back(&rig, &world, &cam, VIEW);
        if let Shape::Shell | Shape::Ring = shape {
            // The far side, beyond the shape's center from the camera, is
            // selected where the surface crosses it, and refined as far as
            // the pixel error and the view cone ask: nothing nearer hides
            // it from the selection.
            let center = Vec3::new(CENTER[0], CENTER[1], CENTER[2]);
            let seen = Seen::instant(&rig.system, &cam);
            let view = (f64::from(VIEW.0), f64::from(VIEW.1));
            let far: Vec<CellKey> = c
                .selection()
                .iter()
                .map(|s| s.key)
                .filter(|k| {
                    let g = k.geometry(Meters::new(EXTENT));
                    let h = 0.5 * g.edge.value();
                    let mid = g.origin + Vec3::new(h, h, h);
                    (mid - cam.position).length() > (center - cam.position).length()
                        && shape.distance(mid).abs() < h * 3f64.sqrt()
                        && c.refinable(k)
                })
                .collect();
            println!(
                "budget: {} far side: {} surface cells, deepest {:?}",
                shape.name(),
                far.len(),
                far.iter().map(|k| k.depth).max()
            );
            assert!(
                !far.is_empty(),
                "{}: the far side is selected",
                shape.name()
            );
            for k in &far {
                assert!(
                    k.depth >= 2 && !wants_refinement(seen.system(), seen.camera(), view, k),
                    "{}: far side cell {k} is held back",
                    shape.name()
                );
            }
        }
    }
}

/// The frame thread's matter step at 4096 selected cells, over a moving
/// camera with results arriving as they come.
#[test]
fn frame_step_at_4096_cells() {
    let _alone = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let depth = 16u8;
    let mut rig = rig(Shape::Plane, depth);
    // A tall view puts the selection at its cap.
    let view = (5120, 2880);
    let c = Vec3::new(CENTER[0], CENTER[1], CENTER[2]);
    let height = 300.0;
    let place = |t: f64| {
        let along = Vec3::new(1.0, 0.25, 0.0).scale(t);
        camera(
            c + along + Vec3::new(0.0, 0.0, height),
            c + along + Vec3::new(0.2 * EXTENT, 0.05 * EXTENT, -0.02 * EXTENT),
        )
    };
    let mut world = World::new(default_threads());
    settle_in_rounds(&mut rig, &mut world, &place(0.0), view);
    let settled_cells = world.cache().selection().len();
    let before = world.step_times().count();
    let steps = 200usize;
    let mut cells = Vec::with_capacity(steps);
    let mut drawn = Vec::with_capacity(steps);
    let mut now = 10.0;
    for i in 0..steps {
        now += 1.0 / 60.0;
        // 5 m per step, 300 m/s at 60 steps per second, 300 m above the
        // plane: the camera moves every step, so every step selects.
        let cam = place(5.0 * i as f64);
        let seen = Seen::instant(&rig.system, &cam);
        matter_step(&mut world, &mut rig.fetcher, &seen, view, now);
        cells.push(world.cache().target().len());
        drawn.push(world.cache().selection().len());
        rig.fetcher.wait_for_events(Duration::from_millis(1));
    }
    let measured = (world.step_times().count() - before) as usize;
    assert_eq!(measured, steps);
    let p50 = world.step_times().recent_percentile(steps, 50.0).unwrap();
    let p95 = world.step_times().recent_percentile(steps, 95.0).unwrap();
    let max = world.step_times().recent_percentile(steps, 100.0).unwrap();
    let least = *cells.iter().min().unwrap();
    let most = *cells.iter().max().unwrap();
    println!(
        "budget: frame step over {steps} steps at {least} to {most} selected cells, {} to {} drawn (settled at {settled_cells}): p50 {p50:.3} ms, p95 {p95:.3} ms, max {max:.3} ms",
        drawn.iter().min().unwrap(),
        drawn.iter().max().unwrap()
    );
    assert!(
        least >= MAX_CELLS_PER_FRAME * 9 / 10 && most <= MAX_CELLS_PER_FRAME,
        "the view selects about 4096 cells ({least} to {most})"
    );
    world.finish_extraction();
    println!(
        "budget: extraction per cell over {} cells: p50 {:.3} ms, p95 {:.3} ms",
        world.extraction_times().count(),
        world.extraction_times().percentile(50.0).unwrap_or(0.0),
        world.extraction_times().percentile(95.0).unwrap_or(0.0)
    );
    if cfg!(debug_assertions) {
        println!("budget: debug build, the frame step is not timed");
    } else {
        assert!(p95 <= 4.0, "frame step p95 {p95:.3} ms, at most 4 ms");
    }
}

/// Times [`renderer::transvoxel::extract_regular`] on one resolution 32
/// cell crossed by a tilted plane (module docs). Prints, asserts no limit.
#[test]
fn transvoxel_regular_cell_timing() {
    use renderer::transvoxel::{extract_regular, Neighborhood, Slot};
    const N: u8 = 32;
    let edge = 32.0;
    let h = edge / f64::from(N);
    let normal: [f64; 3] = [0.21, -0.34, 1.0];
    let len = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
    let distance =
        |p: [f64; 3]| (normal[0] * p[0] + normal[1] * p[1] + normal[2] * p[2] - 0.37) / len;
    let section = |c: [i32; 3]| {
        let key = CellKey::new(
            FRAME,
            5,
            (8 + c[0]) as u32,
            (8 + c[1]) as u32,
            (8 + c[2]) as u32,
        )
        .expect("a valid key");
        let o = c.map(|v| f64::from(v) * edge - 0.5 * edge);
        let samples = Samples::from_fn(N, |x, y, z| {
            let i = [x, y, z];
            let at: [f64; 3] = std::array::from_fn(|a| o[a] + (f64::from(i[a]) + 0.5) * h);
            let occupancy = (0.5 - distance(at) / h).clamp(0.0, 1.0);
            if occupancy <= 0.0 {
                return Sample::VACUUM;
            }
            Sample {
                density: Density::new(2500.0 * occupancy),
                state: State::Solid,
                temperature: Kelvin::new(290.0),
                albedo: [Ratio::new(0.4); 3],
                roughness: Ratio::new(0.8),
                attenuation: Attenuation::new(0.0),
                radiating_temperature: Kelvin::new(290.0),
                occupancy: Ratio::new(occupancy),
            }
        });
        Arc::new(
            Section::new(
                key,
                Vec3::new(o[0], o[1], o[2]),
                Meters::new(edge),
                N,
                samples,
            )
            .expect("valid"),
        )
    };
    let mut hood = Neighborhood::new(section([0, 0, 0]));
    for dz in -1..=1 {
        for dy in -1..=1 {
            for dx in -1..=1 {
                hood.slots[Neighborhood::slot_index(dx, dy, dz)] =
                    Slot::Present(section([dx, dy, dz]));
            }
        }
    }
    let mut times = Vec::new();
    let mut triangles = 0;
    for _ in 0..50 {
        let start = Instant::now();
        let mesh = extract_regular(&hood).expect("every neighbor present");
        times.push(start.elapsed());
        triangles = mesh.triangle_count();
    }
    assert!(triangles > 0);
    times.sort();
    let ms = |d: Duration| d.as_secs_f64() * 1e3;
    println!(
        "transvoxel regular cell, resolution {N}, {triangles} triangles: median {:.3} ms, p95 {:.3} ms over {} runs",
        ms(times[times.len() / 2]),
        ms(times[(times.len() * 95).div_ceil(100) - 1]),
        times.len()
    );
}

/// Times [`renderer::transvoxel::extract_mixed`] on one resolution 32
/// cell crossed by a tilted plane, with its neighbors drawn one depth finer
/// across one face (+x) and across three faces (+x, +y, +z). Prints the
/// median and 95th percentile, asserts no limit.
#[test]
fn transvoxel_transition_cell_timing() {
    use renderer::transvoxel::{extract_mixed, CellAt, MixedField};
    const N: usize = 32;
    /// A tilted plane through the timed cell, the root cell split once,
    /// and the depth 1 cells in `finer` split once more.
    struct Plane {
        finer: Vec<[i64; 3]>,
    }
    const EDGE: f64 = 64.0;
    impl MixedField for Plane {
        fn resolution(&self) -> usize {
            N
        }
        fn root_origin(&self) -> [f64; 3] {
            [0.0; 3]
        }
        fn root_edge(&self) -> f64 {
            EDGE
        }
        fn drawn_depth(&self, d: u8, c: [i64; 3]) -> Option<u8> {
            let per = (N as i64) << d;
            if c.iter().any(|&v| !(0..per).contains(&v)) {
                return None;
            }
            // The depth 1 cell holding the sample center.
            let one = c.map(|v| (2 * v + 1) * 2 / (2 * per));
            Some(if self.finer.contains(&one) { 2 } else { 1 })
        }
        fn occupancy(&self, d: u8, c: [i64; 3]) -> f64 {
            let per = (N as i64) << d;
            if c.iter().any(|&v| !(0..per).contains(&v)) {
                return 0.0;
            }
            let s = EDGE / per as f64;
            let p = c.map(|v| (v as f64 + 0.5) * s);
            let n: [f64; 3] = [0.21, -0.34, 1.0];
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            let distance =
                (n[0] * (p[0] - 16.0) + n[1] * (p[1] - 16.0) + n[2] * (p[2] - 16.0)) / len;
            (0.5 - distance / s).clamp(0.0, 1.0)
        }
    }
    let cell = CellAt {
        depth: 1,
        index: [0, 0, 0],
    };
    for (label, finer) in [
        ("one face", vec![[1, 0, 0]]),
        ("three faces", vec![[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    ] {
        let field = Plane { finer };
        let mut times = Vec::new();
        let mut triangles = 0;
        let mut faces = 0;
        for _ in 0..50 {
            let start = Instant::now();
            let mesh = extract_mixed(&field, cell);
            times.push(start.elapsed());
            triangles = mesh.indices.len() / 3;
            faces = mesh.transition_faces.iter().filter(|&&f| f).count();
        }
        assert!(triangles > 0);
        times.sort();
        let ms = |d: Duration| d.as_secs_f64() * 1e3;
        println!(
            "transvoxel cell with transitions on {label} ({faces}), resolution {N}, {triangles} triangles: median {:.3} ms, p95 {:.3} ms over {} runs",
            ms(times[times.len() / 2]),
            ms(times[(times.len() * 95).div_ceil(100) - 1]),
            times.len()
        );
    }
}

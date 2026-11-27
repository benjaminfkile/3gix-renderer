//! Synthetic universes for any reference radii and frame tree
//! (`rendering.md` sections 2, 3, 6, and 7, `matter-format.md` section 10),
//! served by the mock hub and streamed and drawn headless the way the
//! window does it.
//!
//! A [`Body`] is one frame: its place in the tree, its spin, and
//! optionally a surface (reference radii and a height function of the
//! ellipsoid normal) and a depth-0 cell of matter (solid inside a surface,
//! or hot plasma for a light). Every value is made up by the test that
//! builds it. Nothing here is named after anything.

#![allow(dead_code)]

use gx_core::key::{CellKey, SurfaceKey};
use gx_core::matter::{encode, Compression, Sample, Samples, Section, State};
use gx_core::registry::{self, Frame, Registry};
use gx_core::surface::{
    ellipsoid_point, vertex_direction, SurfaceTile, HEIGHT_SIDE, PARENT_SIDE, TILE_QUADS,
    VERTEX_COUNT,
};
use gx_core::units::{Attenuation, Density, Kelvin, Kilograms, Meters, Quat, Ratio, Seconds, Vec3};
use renderer::api::query::ellipsoid_height;
use renderer::api::{Api, Picture, Target};
use renderer::app::{api_picture, matter_step, set_fetch_time, view_settled};
use renderer::camera::Camera;
use renderer::config::ApiKey;
use renderer::controls::Controls as Keys;
use renderer::hub::HubClient;
use renderer::mock_hub::{
    registry_layer, stored_layer, MockHub, MockHubOptions, MockLayer, SectionMaker, SurfaceMaker,
    MOCK_API_KEY, MOCK_SPACE_ID,
};
use renderer::protocol::Routes;
use renderer::render::scene::{scene_with, Scene};
use renderer::seen::Seen;
use renderer::sim::{SimClock, Simulation};
use renderer::spawn::Spawner;
use renderer::stream::{Fetcher, FetcherOptions};
use renderer::world::{DrawList, World};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// A height function: meters above the reference ellipsoid at the
/// ellipsoid's outward unit normal.
pub type HeightFn = Arc<dyn Fn([f64; 3]) -> f64 + Send + Sync>;

/// A surface: reference radii, a height function, and the deepest level
/// served (finer tiles are empty).
#[derive(Clone)]
pub struct SurfaceSpec {
    /// Radius in the frame's x-y plane, meters.
    pub radius_xy: f64,
    /// Radius along the frame's z axis, meters.
    pub radius_z: f64,
    /// The heights, meters.
    pub height: HeightFn,
    /// The largest height the function gives, meters.
    pub max_height: f64,
    /// The deepest level with tiles.
    pub max_level: u8,
    /// The tiles' temperature, kelvin.
    pub temperature: f32,
}

/// What a body's depth-0 cell holds.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Matter {
    /// Nothing.
    None,
    /// A solid ball of this radius, meters.
    Solid(f64),
    /// A hot plasma ball of this radius, meters, at this temperature.
    Hot(f64, f64),
}

/// One frame of a synthetic universe.
#[derive(Clone)]
pub struct Body {
    /// Frame id.
    pub id: u64,
    /// Parent frame id, [`ROOT_PARENT`] for the root.
    pub parent: u64,
    /// Root cube edge, meters.
    pub extent: f64,
    /// Mass, kilograms.
    pub mass: f64,
    /// Origin in the parent frame, meters.
    pub position: [f64; 3],
    /// Angular velocity about the frame axes, radians per second.
    pub spin: [f64; 3],
    /// The surface, if any.
    pub surface: Option<SurfaceSpec>,
    /// The depth-0 cell's matter.
    pub matter: Matter,
}

impl Body {
    /// A frame with no surface and no matter, not spinning.
    pub fn new(id: u64, parent: u64, extent: f64, mass: f64, position: [f64; 3]) -> Body {
        Body {
            id,
            parent,
            extent,
            mass,
            position,
            spin: [0.0; 3],
            surface: None,
            matter: Matter::None,
        }
    }
}

/// The smooth made-up height function used by the tests: between 0 and
/// `amplitude` meters, varying slowly over the body.
pub fn rolling(amplitude: f64) -> HeightFn {
    Arc::new(move |n: [f64; 3]| {
        amplitude * (0.5 + 0.25 * (3.0 * n[0] + 1.0).sin() * (2.0 * n[1]).cos() + 0.25 * n[2])
    })
}

/// The registry of `bodies`, at epoch 0.
pub fn registry_of(bodies: &[Body]) -> Registry {
    let frames = bodies
        .iter()
        .map(|b| Frame {
            frame_id: b.id,
            parent_frame_id: b.parent,
            root_extent: Meters::new(b.extent),
            max_depth: 0,
            mass: Kilograms::new(b.mass),
            position: Vec3::new(b.position[0], b.position[1], b.position[2]),
            velocity: Vec3::zero(),
            orientation: Quat::identity(),
            angular_velocity: Vec3::new(b.spin[0], b.spin[1], b.spin[2]),
            j2: 0.0,
            reference_radius: Meters::new(0.0),
            principal_moments: Vec3::zero(),
        })
        .collect();
    Registry::new(Seconds::new(0.0), frames).expect("a valid registry")
}

/// The unit normal of the ellipsoid of radii `a`, `c` at its point along
/// unit direction `d`.
pub fn normal_at(d: [f64; 3], a: f64, c: f64) -> [f64; 3] {
    ellipsoid_point(d, a, c).1
}

/// The ellipsoid normal under frame point `q`: the normal of the foot of
/// the perpendicular from `q`, by a few fixed point steps.
pub fn normal_under(q: [f64; 3], a: f64, c: f64) -> [f64; 3] {
    let h = ellipsoid_height(q, a, c);
    let grad = |p: [f64; 3]| unit([p[0] / (a * a), p[1] / (a * a), p[2] / (c * c)]);
    let mut n = grad(q);
    for _ in 0..8 {
        let foot: [f64; 3] = std::array::from_fn(|k| q[k] - h * n[k]);
        n = grad(foot);
    }
    n
}

/// `p` scaled to unit length.
pub fn unit(p: [f64; 3]) -> [f64; 3] {
    let l = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
    p.map(|v| v / l)
}

/// The tile of `key` sampled from `s`, encoded as the hub sends it; empty
/// past [`SurfaceSpec::max_level`].
pub fn tile_bytes(s: &SurfaceSpec, key: &SurfaceKey) -> Vec<u8> {
    if key.level > s.max_level {
        return gx_core::surface::encode_empty(key);
    }
    let (a, c) = (s.radius_xy, s.radius_z);
    let h = |i: i64, j: i64| (s.height)(normal_at(vertex_direction(key, i, j), a, c)) as f32;
    let side = HEIGHT_SIDE as i64;
    let heights = (0..side * side)
        .map(|n| h(n % side - 1, n / side - 1))
        .collect();
    let pside = PARENT_SIDE as i64;
    let parent_heights = (0..pside * pside)
        .map(|n| h(2 * (n % pside - 1), 2 * (n / pside - 1)))
        .collect();
    let tile = SurfaceTile {
        key: *key,
        radius_xy: a,
        radius_z: c,
        height_floor: -1.0,
        heights,
        parent_heights,
        albedo: vec![[150, 150, 150]; VERTEX_COUNT],
        roughness: vec![200; VERTEX_COUNT],
        temperature: vec![s.temperature; VERTEX_COUNT],
    };
    debug_assert_eq!(TILE_QUADS + 3, HEIGHT_SIDE);
    gx_core::surface::encode(&tile, Compression::None)
}

/// The section of `key` of `body`: its matter at depth 0, empty deeper.
pub fn cell_bytes(body: &Body, key: &CellKey) -> Vec<u8> {
    let g = key.geometry(Meters::new(body.extent));
    let (radius, sample) = match body.matter {
        Matter::None => (0.0, Sample::VACUUM),
        Matter::Solid(r) => (
            r,
            Sample {
                density: Density::new(2000.0),
                state: State::Solid,
                temperature: Kelvin::new(250.0),
                albedo: [Ratio::new(0.5); 3],
                roughness: Ratio::new(0.8),
                attenuation: Attenuation::new(0.0),
                radiating_temperature: Kelvin::new(250.0),
                occupancy: Ratio::new(1.0),
            },
        ),
        Matter::Hot(r, t) => (
            r,
            Sample {
                density: Density::new(0.01),
                state: State::Plasma,
                temperature: Kelvin::new(t),
                albedo: [Ratio::new(0.0); 3],
                roughness: Ratio::new(1.0),
                attenuation: Attenuation::new(20.0),
                radiating_temperature: Kelvin::new(t),
                occupancy: Ratio::new(1.0),
            },
        ),
    };
    if key.depth > 0 || radius <= 0.0 {
        let s = Section::empty(*key, g.origin, g.edge).expect("valid");
        return encode(&s, Compression::None);
    }
    let res = 16u8;
    let step = g.edge.value() / f64::from(res);
    let samples = Samples::from_fn(res, |x, y, z| {
        let c = |o: f64, n: u32| o + (f64::from(n) + 0.5) * step;
        let p = Vec3::new(c(g.origin.x, x), c(g.origin.y, y), c(g.origin.z, z));
        if p.length() < radius {
            sample
        } else {
            Sample::VACUUM
        }
    });
    let s = Section::new(*key, g.origin, g.edge, res, samples).expect("a valid section");
    encode(&s, Compression::None)
}

/// The mock hub's layers for `bodies`: the registry, one stored layer of
/// every frame's cells and surface tiles (empty for a frame without a
/// surface), filled to each frame's [`SurfaceSpec::max_level`].
pub fn layers_of(bodies: &[Body]) -> Vec<MockLayer> {
    let reg = registry::encode(&registry_of(bodies));
    let mut layers = vec![registry_layer("frames", &reg)];
    let all: Arc<Vec<Body>> = Arc::new(bodies.to_vec());
    let (cells, surfaces) = (all.clone(), all);
    let mut layer = stored_layer(
        "main",
        0,
        SectionMaker(Arc::new(move |key: &CellKey| {
            let body = cells
                .iter()
                .find(|b| b.id == key.frame_id)
                .expect("a frame");
            cell_bytes(body, key)
        })),
    );
    layer.stored_surfaces = Some(SurfaceMaker(Arc::new(move |key: &SurfaceKey| {
        let body = surfaces
            .iter()
            .find(|b| b.id == key.frame_id)
            .expect("a frame");
        match &body.surface {
            Some(s) => tile_bytes(s, key),
            None => gx_core::surface::encode_empty(key),
        }
    })));
    // The hub applies a layer's maximum depth to surface levels too: each
    // frame's is its deepest surface level (its cells are empty past 0).
    layer.max_depth = bodies
        .iter()
        .map(|b| {
            let levels = b.surface.as_ref().map_or(0, |s| s.max_level);
            (b.id.to_string(), levels)
        })
        .collect();
    layers.push(layer);
    layers
}

/// The window's loop without the window, streaming a synthetic universe
/// from a mock hub, as `tests/api.rs` runs it.
pub struct Universe {
    /// The bodies served.
    pub bodies: Vec<Body>,
    /// The API.
    pub api: Api,
    /// The simulation.
    pub sim: Simulation,
    /// The camera.
    pub camera: Camera,
    /// The spawner (`placeOver`).
    pub spawner: Spawner,
    /// The keys.
    pub keys: Keys,
    /// The world.
    pub world: World,
    /// The fetcher.
    pub fetcher: Fetcher,
    /// The mock hub.
    pub hub: MockHub,
    /// Simulated seconds per tick.
    pub dt: f64,
    /// Seconds of the loop.
    pub t: f64,
    now: Seconds,
    _rt: tokio::runtime::Runtime,
}

/// The view, pixels: small.
pub const VIEW: (u32, u32) = (160, 90);

impl Universe {
    /// Starts a mock hub serving `bodies` and loads its registry.
    pub fn start(bodies: Vec<Body>) -> Universe {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let hub = rt
            .block_on(MockHub::start(MockHubOptions::new(
                layers_of(&bodies),
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
        let now = system.tree().epoch();
        let camera = Camera::home(&system);
        let mut world = World::new(1);
        world.start_layers(&build, None);
        fetcher.set_hidden_layers(world.hidden_layers().clone());
        Universe {
            bodies,
            api: Api::new(system.tree(), &camera)
                .with_build(&build)
                .with_compute_budget(fetcher.max_compute_budget(), None),
            sim: Simulation::new(system, SimClock::starting_at(now, now, now, 1.0)),
            camera,
            spawner: Spawner::default(),
            keys: Keys::new(None),
            world,
            fetcher,
            hub,
            dt: 1.0 / 60.0,
            t: 0.0,
            now,
            _rt: rt,
        }
    }

    /// The body `id`.
    pub fn body(&self, id: u64) -> &Body {
        self.bodies.iter().find(|b| b.id == id).expect("a body")
    }

    /// One tick in the window's order, streaming.
    pub fn tick(&mut self) {
        self.api.begin_tick(Target {
            sim: &mut self.sim,
            camera: &mut self.camera,
            spawner: &mut self.spawner,
            keys: &mut self.keys,
        });
        let input = self.api.flight(self.keys.take_flight());
        // The window's order: the spawner after the flight input, before
        // the camera advances, at the loop's session time.
        self.spawner.frame(
            self.sim.system(),
            &self.world,
            &mut self.camera,
            &mut self.keys,
            &input,
            self.t,
        );
        // The hub's clock runs with the loop's, as the window's does.
        self.now = self.now + Seconds::new(self.dt);
        renderer::camera::advance(
            &mut self.sim,
            &mut self.camera,
            self.dt,
            self.now,
            &input,
            self.keys.attached,
        );
        self.t += self.dt;
        self.sim.prepare_light(&self.camera);
        let seen = Seen::instant(self.sim.system(), &self.camera);
        set_fetch_time(&mut self.fetcher, &self.sim);
        let draw = matter_step(&mut self.world, &mut self.fetcher, &seen, VIEW, self.t);
        let picture = api_picture(&self.world, &draw, &seen, VIEW);
        self.api.set_picture(picture);
        self.api.end_tick(&self.sim, &self.camera, &self.world);
    }

    /// Ticks until the view settles; panics after two minutes.
    pub fn settle(&mut self) {
        let start = Instant::now();
        let mut ticks = 0;
        loop {
            self.tick();
            ticks += 1;
            if ticks > 3 && view_settled(&self.world, &self.spawner) {
                // A second look: the settle holds.
                self.tick();
                if view_settled(&self.world, &self.spawner) {
                    return;
                }
            }
            assert!(
                start.elapsed() < Duration::from_secs(120),
                "the view never settled"
            );
            std::thread::sleep(Duration::from_millis(2));
        }
    }

    /// What is drawn now: the draw list, the queries' picture, and the
    /// scene for the GPU.
    pub fn drawn(&mut self) -> (DrawList, Picture, Scene) {
        self.sim.prepare_light(&self.camera);
        let seen = Seen::instant(self.sim.system(), &self.camera);
        let draw = self.world.draw_list(&seen, VIEW, self.t);
        let picture = api_picture(&self.world, &draw, &seen, VIEW);
        let scene = scene_with(&draw, &seen, VIEW);
        (draw, picture, scene)
    }
}

/// Every value the scene hands the GPU is finite.
pub fn assert_finite(scene: &Scene) {
    let ok3 = |v: &[f32; 3]| v.iter().all(|x| x.is_finite());
    for t in &scene.tiles {
        assert!(ok3(&t.offset), "tile offset {:?}", t.offset);
        assert!(t.rotation.iter().all(ok3), "tile rotation");
        assert!(t.lights.iter().all(ok3), "tile lights");
        assert!(t.weight.is_finite());
        let g = &t.tile.geometry;
        assert!(g.positions.iter().all(|p| p.iter().all(|x| x.is_finite())));
        assert!(g.normals.iter().all(ok3));
        assert!(g.target_normals.iter().all(ok3));
        assert!(t.tile.morph.iter().all(|m| !m.is_nan()));
        // The camera-relative vertex positions the GPU gets.
        for p in g.positions.iter().step_by(97) {
            let rel: [f32; 3] =
                std::array::from_fn(|k| (p[k] - t.tile.anchor.position[k]) as f32 + t.offset[k]);
            assert!(ok3(&rel), "relative vertex {rel:?}");
        }
    }
    for s in &scene.surfaces {
        assert!(ok3(&s.offset), "cell offset");
        assert!(s.lights.iter().all(ok3), "cell lights");
    }
    for l in &scene.lights {
        assert!(ok3(&l.position) && ok3(&l.intensity), "light {l:?}");
    }
    for v in &scene.volumes {
        assert!(ok3(&v.offset), "volume offset");
    }
}

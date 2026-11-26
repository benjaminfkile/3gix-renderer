//! Bodies nothing like any one universe's (`rendering.md` sections 2, 3, 6,
//! and 7, `matter-format.md` section 10): a small body, a very flattened
//! one, a fast spinner, a system of three lights around a surface, a deep
//! frame tree with ids out of order, and a very large body, each served by
//! the mock hub from made-up values and drawn headless at a small size.
//!
//! Each view is checked for: the view settles, the drawn shown cells have
//! no open edges ([`renderer::world::World::open_edges`]) and the drawn
//! tiles leave no hole a pixel ray through the reference ellipsoid could
//! pass, `surfaceHeight` and `raycast` agree with each other and with the
//! height function, nothing non-finite reaches the GPU, and the picture is
//! not empty.

#[path = "common/universe.rs"]
mod universe;

use gx_core::registry::ROOT_PARENT;
use renderer::api::query::ellipsoid_height;
use renderer::api::{HitKind, Kind};
use renderer::camera::look_rotation;
use renderer::render::headless::Headless;
use universe::{
    assert_finite, normal_at, normal_under, rolling, unit, Body, Matter, SurfaceSpec, Universe,
    VIEW,
};

use gx_core::units::Vec3;

/// The root of every case.
const ROOT: u64 = 1;

/// `surfaceHeight` and the height of the `raycast` hit agree to this,
/// meters, plus this part of the radius (rounding of the vertex positions).
const QUERY_TOLERANCE: f64 = 1e-3;
const QUERY_TOLERANCE_RADIUS: f64 = 1e-9;
/// The drawn height agrees with the height function to this part of its
/// amplitude: the function is smooth and the mesh interpolates it linearly
/// between vertices.
const HEIGHT_TOLERANCE: f64 = 0.02;

fn surface(a: f64, c: f64, amplitude: f64, max_level: u8) -> SurfaceSpec {
    SurfaceSpec {
        radius_xy: a,
        radius_z: c,
        height: rolling(amplitude),
        max_height: amplitude,
        max_level,
        temperature: 250.0,
    }
}

/// A massless root of edge `extent` and one body of surface `s` at its
/// origin.
fn one_body(extent: f64, body_extent: f64, s: SurfaceSpec) -> Vec<Body> {
    let r = s.radius_xy.min(s.radius_z);
    let mut body = Body::new(2, ROOT, body_extent, 1.0e3, [0.0; 3]);
    body.matter = Matter::Solid(0.5 * r);
    body.surface = Some(s);
    vec![Body::new(ROOT, ROOT_PARENT, extent, 0.0, [0.0; 3]), body]
}

/// Puts the camera in `frame` `height` meters above the surface of body
/// `body` along unit direction `d` (frame axes), looking down and a little
/// ahead.
fn over(u: &mut Universe, body: u64, d: [f64; 3], height: f64) {
    let s = u.body(body).surface.clone().unwrap();
    let (a, c) = (s.radius_xy, s.radius_z);
    let n = normal_at(d, a, c);
    let p = gx_core::surface::ellipsoid_point(d, a, c).0;
    let h = (s.height)(n);
    let pos: [f64; 3] = std::array::from_fn(|k| p[k] + (h + height) * n[k]);
    let side = unit(cross(
        n,
        if n[2].abs() < 0.9 {
            [0.0, 0.0, 1.0]
        } else {
            [1.0, 0.0, 0.0]
        },
    ));
    let fwd = unit(std::array::from_fn(|k| -n[k] + 0.3 * side[k]));
    place(u, body, pos, fwd, n);
}

/// Puts the camera `distance` meters from the origin of `body` along `d`,
/// looking at the origin.
fn away(u: &mut Universe, body: u64, d: [f64; 3], distance: f64) {
    let pos = d.map(|v| v * distance);
    let up = if d[2].abs() < 0.9 {
        [0.0, 0.0, 1.0]
    } else {
        [1.0, 0.0, 0.0]
    };
    place(u, body, pos, d.map(|v| -v), up);
}

fn place(u: &mut Universe, frame: u64, pos: [f64; 3], fwd: [f64; 3], up: [f64; 3]) {
    let v = |a: [f64; 3]| Vec3::new(a[0], a[1], a[2]);
    // Frame axes into root axes for the orientation.
    u.tick();
    let fwd = u
        .api
        .transform(fwd, frame, root(u), Kind::Direction)
        .unwrap();
    let up = u
        .api
        .transform(up, frame, root(u), Kind::Direction)
        .unwrap();
    u.camera.frame_id = frame;
    u.camera.position = v(pos);
    u.camera.orientation = look_rotation(v(fwd), v(up));
    u.camera.update_parent(u.sim.system());
}

/// The root frame id of the universe.
fn root(u: &Universe) -> u64 {
    u.sim.system().tree().root().frame_id
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Where the ray from `o` along unit `d` first meets the ellipsoid of
/// radii `a`, `c`, if it does.
fn ray_ellipsoid(o: [f64; 3], d: [f64; 3], a: f64, c: f64) -> Option<f64> {
    let s = [1.0 / a, 1.0 / a, 1.0 / c];
    let (o, d): ([f64; 3], [f64; 3]) = (
        std::array::from_fn(|k| o[k] * s[k]),
        std::array::from_fn(|k| d[k] * s[k]),
    );
    let dot = |x: [f64; 3], y: [f64; 3]| x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
    let (qa, qb, qc) = (dot(d, d), 2.0 * dot(o, d), dot(o, o) - 1.0);
    let disc = qb * qb - 4.0 * qa * qc;
    if disc < 0.0 {
        return None;
    }
    let t = (-qb - disc.sqrt()) / (2.0 * qa);
    (t > 0.0).then_some(t)
}

/// What one view of one body shows.
struct Checked {
    /// Pixels whose ray hit the body.
    hits: usize,
    /// Lit pixels of the picture.
    lit: usize,
}

/// Settles the view and runs every check on body `body`.
fn check(u: &mut Universe, body: u64, name: &str) -> Checked {
    u.settle();
    let (_, picture, scene) = u.drawn();
    assert_finite(&scene);
    for b in &u.bodies {
        assert_eq!(
            u.world.open_edges(b.id, b.extent, 1e-6 * b.extent),
            0,
            "{name}"
        );
    }
    assert!(
        scene
            .tiles
            .iter()
            .any(|t| t.tile.geometry.key.frame_id == body),
        "{name}: the body's tiles are drawn"
    );
    let s = u.body(body).surface.clone().unwrap();
    let (a, c) = (s.radius_xy, s.radius_z);
    let p = u.camera.position;
    let cam = u
        .api
        .transform([p.x, p.y, p.z], u.camera.frame_id, body, Kind::Point)
        .unwrap();
    let mut hits = 0;
    let (nx, ny) = (16, 9);
    for j in 0..ny {
        for i in 0..nx {
            let x = (f64::from(i) + 0.5) * f64::from(VIEW.0) / f64::from(nx);
            let y = (f64::from(j) + 0.5) * f64::from(VIEW.1) / f64::from(ny);
            let ray = picture.pixel_ray(x, y);
            let d = u
                .api
                .transform(ray, root(u), body, Kind::Direction)
                .unwrap();
            let through = ray_ellipsoid(cam, d, a - 2.0, c - 2.0).is_some();
            let hit = picture.raycast(x, y);
            match hit {
                Some(hit) if hit.frame_id == body && hit.kind == HitKind::Surface => {
                    hits += 1;
                    let q = hit.position;
                    let mesh = ellipsoid_height(q, a, c);
                    let sh = picture
                        .surface_height(body, unit(q))
                        .unwrap_or_else(|| panic!("{name}: surfaceHeight under a hit"));
                    let tol = QUERY_TOLERANCE + QUERY_TOLERANCE_RADIUS * a.max(c);
                    assert!(
                        (sh.height - mesh).abs() < tol,
                        "{name} ({x}, {y}): surfaceHeight {} vs raycast {mesh}",
                        sh.height
                    );
                    let want = (s.height)(normal_under(q, a, c));
                    assert!(
                        (mesh - want).abs() < HEIGHT_TOLERANCE * s.max_height + tol,
                        "{name} ({x}, {y}): drawn {mesh} vs height function {want}"
                    );
                }
                _ => assert!(
                    !through,
                    "{name} ({x}, {y}): a ray through the body misses the drawn tiles: {hit:?}"
                ),
            }
        }
    }
    let mut headless = Headless::new(VIEW.0, VIEW.1, None).expect("a wgpu adapter");
    // Radiance before exposure: a body lit by no light still has its own
    // thermal emission, far below what the display shows.
    headless.render(&scene, None).unwrap();
    let radiance = headless.read_radiance().unwrap();
    let mut lit = 0;
    for y in 0..VIEW.1 {
        for x in 0..VIEW.0 {
            lit += usize::from(radiance.pixel(x, y)[..3].iter().any(|v| *v > 0.0));
        }
    }
    println!(
        "{name}: {hits} hits, {lit} lit pixels, {} tiles",
        scene.tiles.len()
    );
    assert!(lit > 0, "{name}: the picture is empty");
    Checked { hits, lit }
}

#[test]
fn a_small_body() {
    let s = surface(1500.0, 1500.0, 400.0, 3);
    let mut u = Universe::start(one_body(1.0e6, 8000.0, s));
    over(&mut u, 2, unit([0.3, -0.5, 0.4]), 100.0);
    let near = check(&mut u, 2, "small, 100 m above");
    assert!(near.hits > 0);
    away(&mut u, 2, unit([0.3, -0.5, 0.4]), 20_000.0);
    let far = check(&mut u, 2, "small, 20 km away");
    assert!(far.hits > 0 && far.lit > 0);
}

#[test]
fn a_very_flattened_body() {
    let (a, c) = (9.0e7, 5.4e7);
    let s = surface(a, c, 5000.0, 4);
    let mut u = Universe::start(one_body(1.0e12, 2.5 * a, s));
    // The point whose normal is 45 degrees from the x-y plane.
    let t = (a / c).powi(2);
    let d45 = unit([1.0, 0.0, 1.0 / t]);
    let n = normal_at(d45, a, c);
    assert!((n[2] - n[0]).abs() < 1e-9, "{n:?}");
    over(&mut u, 2, d45, 2.0e5);
    check(&mut u, 2, "flattened, 45 degrees");
    over(&mut u, 2, [0.0, 0.0, 1.0], 2.0e5);
    check(&mut u, 2, "flattened, on the z axis");
}

#[test]
fn a_fast_spinner_keeps_the_camera_over_its_point() {
    let s = surface(3000.0, 2900.0, 200.0, 3);
    let mut bodies = one_body(1.0e6, 1.6e4, s);
    bodies[1].spin = [0.0, 0.0, 2.0 * std::f64::consts::PI / 200.0];
    let mut u = Universe::start(bodies);
    u.tick();
    u.api.place_over(2, [1.0, 0.5, 0.2], 100.0, 30.0).unwrap();
    u.settle();
    assert_eq!(u.camera.frame_id, 2, "the camera rides the body");
    let start = u.camera.position;
    let start_root = u
        .api
        .transform([start.x, start.y, start.z], 2, ROOT, Kind::Point)
        .unwrap();
    u.dt = 0.25;
    for _ in 0..240 {
        u.tick();
    }
    let p = u.camera.position;
    let moved = (p - start).length();
    assert!(moved < 1.0, "the camera moved {moved} m over the body");
    let now_root = u
        .api
        .transform([p.x, p.y, p.z], 2, ROOT, Kind::Point)
        .unwrap();
    let turned = (0..3)
        .map(|k| (now_root[k] - start_root[k]).powi(2))
        .sum::<f64>()
        .sqrt();
    assert!(turned > 100.0, "the body turned under the root: {turned} m");
    u.dt = 1.0 / 60.0;
    check(&mut u, 2, "spinner, after 60 s");
}

#[test]
fn three_lights_add_on_a_surface() {
    let s = SurfaceSpec {
        temperature: 3.0,
        ..surface(1000.0, 1000.0, 50.0, 2)
    };
    let mut bodies = one_body(1.0e6, 6000.0, s);
    for (id, d) in [
        (31, [1.0, 0.2, 0.3]),
        (17, [0.2, 1.0, -0.3]),
        (52, [0.6, 0.6, 0.9]),
    ] {
        let p = unit(d).map(|v| v * 8000.0);
        let mut light = Body::new(id, ROOT, 1000.0, 1.0, p);
        light.matter = Matter::Hot(200.0, 7000.0);
        bodies.push(light);
    }
    let mut u = Universe::start(bodies);
    away(&mut u, 2, unit([1.0, 1.0, 0.6]), 4000.0);
    check(&mut u, 2, "three lights");
    let (_, _, scene) = u.drawn();
    assert_eq!(scene.lights.len(), 3, "three lights: {:?}", scene.lights);
    // The lights draw only as cells.
    for t in &scene.tiles {
        assert_eq!(t.tile.geometry.key.frame_id, 2, "only the body has tiles");
    }
    for l in [31, 17, 52] {
        assert!(
            !u.world.surfaces().has_surface(l),
            "light {l} has no surface"
        );
    }
    let mut headless = Headless::new(VIEW.0, VIEW.1, None).unwrap();
    let (cx, cy) = (VIEW.0 / 2, VIEW.1 / 2);
    headless.render(&scene, None).unwrap();
    let all = headless.read_radiance().unwrap().pixel(cx, cy);
    let mut sum = [0.0f32; 3];
    for k in 0..3 {
        let mut one = scene.clone();
        for (j, l) in one.lights.iter_mut().enumerate() {
            if j != k {
                l.intensity = [0.0; 3];
            }
        }
        headless.render(&one, None).unwrap();
        let p = headless.read_radiance().unwrap().pixel(cx, cy);
        for b in 0..3 {
            sum[b] += p[b];
        }
    }
    for b in 0..3 {
        assert!(all[b] > 0.0, "the surface is lit: {all:?}");
        let rel = (all[b] - sum[b]).abs() / all[b];
        assert!(rel < 1e-3, "band {b}: {} vs the sum {}", all[b], sum[b]);
    }
}

#[test]
fn a_deep_tree_with_ids_out_of_order() {
    // Root 11; 40 under it; 7 under 40; 93 under 7. A surface on each.
    let spec = |r: f64| surface(r, 0.97 * r, 0.02 * r, 2);
    let mut bodies = vec![Body::new(11, ROOT_PARENT, 1.0e9, 0.0, [0.0; 3])];
    for (id, parent, r, at) in [
        (40u64, 11u64, 2.0e5, [0.0, 0.0, 0.0]),
        (7, 40, 2.0e4, [1.0e6, 0.0, 0.0]),
        (93, 7, 2.0e3, [0.0, 1.0e5, 0.0]),
    ] {
        let mut b = Body::new(id, parent, 4.0 * r, 1.0e3, at);
        b.surface = Some(spec(r));
        b.matter = Matter::Solid(0.5 * r);
        bodies.push(b);
    }
    // The root holds a surface of its own too, at its origin.
    bodies[0].surface = Some(spec(5.0e4));
    bodies[0].matter = Matter::Solid(2.5e4);
    bodies[1].position = [0.0, 0.0, 3.0e7];
    let ids = [11u64, 40, 7, 93];
    let mut u = Universe::start(bodies);
    for id in [40u64, 7, 93] {
        let r = u.body(id).surface.as_ref().unwrap().radius_xy;
        away(&mut u, id, unit([0.4, -0.7, 0.3]), 4.0 * r);
        check(&mut u, id, &format!("frame {id}"));
    }
    let frames = u.api.frames();
    let listed: Vec<u64> = frames.iter().map(|f| f.id).collect();
    for id in ids {
        assert!(listed.contains(&id), "frames() lists {id}: {listed:?}");
        let parent = frames.iter().find(|f| f.id == id).unwrap().parent_id;
        let want = u.body(id).parent;
        assert_eq!(parent, (want != ROOT_PARENT).then_some(want), "frame {id}");
    }
    // `transform` round trips between every pair.
    for from in ids {
        for to in ids {
            let p = [12.5, -3.0, 7.25];
            let there = u.api.transform(p, from, to, Kind::Point).unwrap();
            let back = u.api.transform(there, to, from, Kind::Point).unwrap();
            for k in 0..3 {
                assert!((back[k] - p[k]).abs() < 1e-6, "{from} -> {to}: {back:?}");
            }
        }
    }
    // 93's origin, seen from the root: the sum of the offsets down the tree.
    let o = u.api.transform([0.0; 3], 93, 11, Kind::Point).unwrap();
    let want = [1.0e6, 1.0e5, 3.0e7];
    for k in 0..3 {
        assert!((o[k] - want[k]).abs() < 1e-3, "{o:?}");
    }
    // The spawn chooser offers every frame with matter but the root.
    let tree = u.sim.system().tree().clone();
    let offered = renderer::spawn::spawn_frames(u.world.cache(), &tree);
    for id in [40u64, 7, 93] {
        assert!(
            offered.contains(&id),
            "the chooser offers {id}: {offered:?}"
        );
    }
}

#[test]
fn a_very_large_body() {
    let r = 7.0e8;
    let s = surface(r, r, 3000.0, 6);
    let mut u = Universe::start(one_body(1.0e13, 2.5 * r, s));
    let d = unit([0.2, 0.9, -0.3]);
    over(&mut u, 2, d, 1000.0);
    let near = check(&mut u, 2, "large, 1 km above");
    assert!(near.hits > 0);
    // Straight down from the camera: the hit is 1 km below, to the
    // precision of the f64 positions.
    let (_, picture, _) = u.drawn();
    let n = normal_at(d, r, r);
    let down = u
        .api
        .transform(n.map(|v| -v), 2, ROOT, Kind::Direction)
        .unwrap();
    let hit = picture.cast(down).expect("straight down hits the surface");
    assert!(
        (hit.distance - 1000.0).abs() < HEIGHT_TOLERANCE * 3000.0 + 1.0,
        "{}",
        hit.distance
    );
}

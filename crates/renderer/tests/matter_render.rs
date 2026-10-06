//! Headless render of matter: a solid blob lit by a hot blob beside it, and
//! the full scene with a gas cloud added.
//!
//! The cells go through the same path as on the desktop (selection, the
//! cell cache, decoding and compositing, extraction on the worker pool,
//! volumes, lights from hot matter), fed from in-memory chunk containers
//! instead of the hub. The images are checked by pixel statistics and saved
//! as `target/tmp/matter-render.png` and `target/tmp/full-scene.png` with
//! the statistics beside them in `.txt` files of the same name.

mod common;

use common::*;
use gx_core::units::{Meters, Vec3};
use renderer::app::matter_scene;
use renderer::camera::FIELD_OF_VIEW_Y;
use renderer::render::headless::{Headless, Image};
use renderer::render::scene::Scene;
use renderer::stream::{decode_cell, CellState, FetchOutcome};
use renderer::world::World;

const WIDTH: u32 = 640;
const HEIGHT: u32 = 360;

fn luma(p: [u8; 4]) -> f64 {
    0.2126 * f64::from(p[0]) + 0.7152 * f64::from(p[1]) + 0.0722 * f64::from(p[2])
}

/// Streams every selected cell from the fixture's containers.
fn stream_all(world: &mut World) {
    stream_from(world, &system(), &chunks(), 2);
}

/// Streams every selected cell of `system` from `chunks`, expecting
/// `expected` requests.
fn stream_from(
    world: &mut World,
    system: &gx_core::frames::FrameSystem,
    chunks: &std::collections::BTreeMap<gx_core::key::CellKey, Vec<u8>>,
    expected: usize,
) {
    let camera = camera();
    world.select(system, &camera, (WIDTH, HEIGHT), 0.0, true);
    let requests = world.take_requests(0.0);
    assert_eq!(requests.len(), expected, "{requests:?}");
    for key in requests {
        let extent = system.tree().get(key.frame_id).unwrap().root_extent;
        let outcome = match chunks.get(&key) {
            Some(bytes) => decode_cell(&key, bytes, extent),
            None => FetchOutcome::NotFound,
        };
        world.complete(key, outcome, 0.0);
    }
    world.finish_extraction();
    assert!(world.settled());
}

/// A disc in the image: projected center and radius in pixels.
struct Disc {
    x: f64,
    y: f64,
    r: f64,
}

fn disc(scene: &Scene, center: Vec3, radius: f64) -> Disc {
    let c = [center.x as f32, center.y as f32, center.z as f32];
    let (x, y) = scene.project(c, WIDTH, HEIGHT).expect("in front");
    let focal = f64::from(HEIGHT) / (2.0 * (0.5 * FIELD_OF_VIEW_Y).tan());
    Disc {
        x: f64::from(x),
        y: f64::from(y),
        r: radius / center.length() * focal,
    }
}

/// Mean luma and pixel count over the pixels inside `fraction` of the disc
/// radius that pass `keep(dx)`, with `dx` the offset from the disc center.
fn mean_in(image: &Image, d: &Disc, fraction: f64, keep: impl Fn(f64) -> bool) -> (f64, usize) {
    let mut sum = 0.0;
    let mut n = 0;
    for y in 0..image.height {
        for x in 0..image.width {
            let dx = f64::from(x) + 0.5 - d.x;
            let dy = f64::from(y) + 0.5 - d.y;
            if dx * dx + dy * dy <= (fraction * d.r).powi(2) && keep(dx) {
                sum += luma(image.pixel(x, y));
                n += 1;
            }
        }
    }
    (sum / n.max(1) as f64, n)
}

#[test]
fn solid_blob_is_lit_from_the_side_of_the_hot_blob() {
    let mut world = World::new(2);
    stream_all(&mut world);
    let system = system();
    let camera = camera();
    let scene = matter_scene(&mut world, &system, &camera, (WIDTH, HEIGHT), 0.0);
    // The solid blob is a mesh; the plasma blob is a volume.
    assert_eq!(scene.surfaces.len(), 1);
    assert_eq!(scene.volumes.len(), 1);
    assert_eq!(scene.lights.len(), 1);
    assert_eq!(scene.lights[0].frame_id, ROOT);

    let mut headless =
        Headless::new(WIDTH, HEIGHT, None).expect("a wgpu adapter, software is fine");
    let first = headless.render(&scene, None).unwrap();
    let stats = headless.stats();
    let second = headless.render(&scene, None).unwrap();
    assert!(first == second, "two renders of the same state differ");
    assert_eq!(stats.meshes, 1);
    assert_eq!(stats.volumes, 1);
    assert_eq!(stats.lights, 1);

    let hot = disc(
        &scene,
        camera.relative(&system, Vec3::zero(), ROOT),
        HOT_RADIUS,
    );
    let solid = disc(
        &scene,
        camera.relative(&system, Vec3::zero(), CHILD),
        SOLID_RADIUS,
    );
    assert!(hot.x < solid.x, "the hot blob is on the left");

    let (hot_mean, hot_n) = mean_in(&first, &hot, 0.85, |_| true);
    // The half of the solid disc toward the hot blob, and the other half.
    let (near_mean, near_n) = mean_in(&first, &solid, 0.85, |dx| dx < 0.0);
    let (far_mean, far_n) = mean_in(&first, &solid, 0.85, |dx| dx > 0.0);
    let lit_n = {
        let mut lit = 0;
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                let dx = f64::from(x) + 0.5 - solid.x;
                let dy = f64::from(y) + 0.5 - solid.y;
                if dx * dx + dy * dy <= (0.85 * solid.r).powi(2) && luma(first.pixel(x, y)) > 8.0 {
                    lit += 1;
                }
            }
        }
        lit
    };
    // The brightest pixel anywhere, and the brightest outside the hot disc.
    let mut max_all = 0.0f64;
    let mut max_outside_hot = 0.0f64;
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let l = luma(first.pixel(x, y));
            max_all = max_all.max(l);
            let dx = f64::from(x) + 0.5 - hot.x;
            let dy = f64::from(y) + 0.5 - hot.y;
            if dx * dx + dy * dy > (1.2 * hot.r).powi(2) {
                max_outside_hot = max_outside_hot.max(l);
            }
        }
    }

    let report = format!(
        "image {WIDTH} x {HEIGHT}\n\
         hot blob disc: center ({:.1}, {:.1}) radius {:.1} px, {hot_n} px, mean luma {hot_mean:.2}\n\
         solid disc: center ({:.1}, {:.1}) radius {:.1} px\n\
         solid half toward the hot blob: {near_n} px, mean luma {near_mean:.2}\n\
         solid half away from the hot blob: {far_n} px, mean luma {far_mean:.2}\n\
         solid pixels with luma above 8: {lit_n}\n\
         brightest luma anywhere {max_all:.2}, outside the hot disc {max_outside_hot:.2}\n\
         meshes {}, triangles {}, volumes {}, lights {}\n",
        hot.x,
        hot.y,
        hot.r,
        solid.x,
        solid.y,
        solid.r,
        stats.meshes,
        stats.triangles,
        stats.volumes,
        stats.lights
    );
    println!("{report}");
    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR"));
    first.write_png(&dir.join("matter-render.png")).unwrap();
    std::fs::write(dir.join("matter-render.txt"), &report).unwrap();

    // A lit disc: a good part of it is visibly lit.
    assert!(lit_n * 3 > near_n + far_n, "{report}");
    // Brighter toward the hot blob.
    assert!(near_mean > 20.0, "{report}");
    assert!(near_mean > 2.0 * far_mean, "{report}");
    // The hot blob is the brightest region.
    assert!(hot_mean > near_mean, "{report}");
    assert!(hot_mean > 150.0, "{report}");
    // No pixel outside it is as bright as its mean.
    assert!(max_outside_hot < hot_mean, "{report}");
}

#[test]
fn empty_composite_and_registry_geometry() {
    let chunks = chunks();
    let key = empty_key();
    assert_eq!(
        decode_cell(&key, &chunks[&key], Meters::new(CHILD_EXTENT)),
        FetchOutcome::Decoded(None)
    );
    let mut world = World::new(0);
    stream_all(&mut world);
    assert!(matches!(
        world.cache().state(&solid_key()),
        Some(CellState::Ready(_))
    ));
    let mesh = world.mesh(&solid_key()).expect("extracted");
    assert!(mesh.triangle_count() > 1000);
}

/// Luma along row `y` from column `x0` for `len` pixels toward `step`.
fn scan(image: &Image, x0: u32, y: u32, len: u32, step: i64) -> Vec<f64> {
    (0..len)
        .map(|i| {
            let x = (i64::from(x0) + step * i64::from(i)).clamp(0, i64::from(image.width) - 1);
            luma(image.pixel(x as u32, y))
        })
        .collect()
}

/// The brightest pixel of row `y` within `reach` pixels of column `x`.
fn brightest_near(image: &Image, x: f64, y: u32, reach: i64) -> u32 {
    let c = x as i64;
    (c - reach..=c + reach)
        .map(|x| x.clamp(0, i64::from(image.width) - 1) as u32)
        .max_by(|a, b| {
            luma(image.pixel(*a, y))
                .total_cmp(&luma(image.pixel(*b, y)))
                .then(b.cmp(a))
        })
        .unwrap()
}

/// Largest luma change that is 8-bit quantization rather than a rise: one
/// step in one channel moves luma by at most 0.72.
const QUANTUM: f64 = 1.0;

/// Number of places a sequence goes up by more than [`QUANTUM`], and its
/// largest rise of any size.
fn rises(values: &[f64]) -> (usize, f64) {
    let mut n = 0;
    let mut max = 0.0f64;
    for w in values.windows(2) {
        if w[1] > w[0] + QUANTUM {
            n += 1;
        }
        max = max.max(w[1] - w[0]);
    }
    (n, max)
}

fn describe(values: &[f64]) -> String {
    values
        .iter()
        .map(|v| format!("{v:.0}"))
        .collect::<Vec<_>>()
        .join(" ")
}

#[test]
fn full_scene_draws_soft_glowing_volumes() {
    let system = full_system();
    let chunks = full_chunks();
    let camera = camera();
    let mut world = World::new(2);
    stream_from(&mut world, &system, &chunks, 3);
    let mut scene = matter_scene(&mut world, &system, &camera, (WIDTH, HEIGHT), 0.0);
    // Matter only: the frame markers and lines are a debug aid drawn on top.
    scene.markers.clear();
    scene.lines.clear();
    assert_eq!(scene.surfaces.len(), 1);
    assert_eq!(scene.volumes.len(), 2);
    // Back to front: the gas cloud is farther than the hot blob.
    assert_eq!(scene.volumes[0].grid.key, gas_key());
    assert_eq!(scene.volumes[1].grid.key, hot_key());
    assert!(scene.volumes[0].distance > scene.volumes[1].distance);
    // Hot blob and gas cloud both emit light.
    assert_eq!(scene.lights.len(), 2);

    let mut headless =
        Headless::new(WIDTH, HEIGHT, None).expect("a wgpu adapter, software is fine");
    let first = headless.render(&scene, None).unwrap();
    let stats = headless.stats();
    let second = headless.render(&scene, None).unwrap();
    assert!(first == second, "two renders of the same state differ");
    assert_eq!((stats.meshes, stats.volumes), (1, 2));

    let hot = disc(
        &scene,
        camera.relative(&system, Vec3::zero(), ROOT),
        HOT_RADIUS,
    );
    let gas = disc(
        &scene,
        camera.relative(&system, Vec3::zero(), GAS),
        GAS_RADIUS,
    );
    let mut report = format!("image {WIDTH} x {HEIGHT}\n");
    let mut checks = Vec::new();
    for (name, d) in [("hot blob", &hot), ("gas cloud", &gas)] {
        let y = d.y as u32;
        let cx = brightest_near(&first, d.x, y, 3);
        let len = (1.5 * d.r).ceil() as u32;
        let right = scan(&first, cx, y, len, 1);
        let left = scan(&first, cx, y, len, -1);
        let (up_r, max_r) = rises(&right);
        let (up_l, max_l) = rises(&left);
        // Soft edge: pixels strictly between the center value and black.
        let partial = right
            .iter()
            .chain(&left)
            .filter(|&&v| v > 4.0 && v < 0.9 * right[0])
            .count();
        report += &format!(
            "{name}: disc center ({:.1}, {:.1}) radius {:.1} px; scan row {y} from column {cx}\n\
             \x20 right: {}\n\
             \x20 left: {}\n\
             \x20 rises right {up_r} (largest {max_r:.2}), left {up_l} (largest {max_l:.2}); \
             partial pixels {partial} of {}\n",
            d.x,
            d.y,
            d.r,
            describe(&right),
            describe(&left),
            2 * len
        );
        checks.push((name, right, left, partial));
    }
    println!("{report}");
    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR"));
    first.write_png(&dir.join("full-scene.png")).unwrap();
    std::fs::write(dir.join("full-scene.txt"), &report).unwrap();

    for (name, right, left, partial) in checks {
        // Bright at the center, black past the edge.
        assert!(right[0] > 100.0, "{name}\n{report}");
        assert!(
            *right.last().unwrap() < 4.0 && *left.last().unwrap() < 4.0,
            "{name}\n{report}"
        );
        // Monotonic decrease from the center outward on both sides, to
        // within one 8-bit step: the nearest-sample march ripples by less
        // than that where the profile is flat.
        assert_eq!(rises(&right).0, 0, "{name}\n{report}");
        assert_eq!(rises(&left).0, 0, "{name}\n{report}");
        // A glow, not a hard-edged disc: the falloff spans several pixels.
        assert!(partial >= 6, "{name}\n{report}");
    }
}

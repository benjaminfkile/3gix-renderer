//! Headless render of matter: a solid blob lit by a hot blob beside it.
//!
//! The cells go through the same path as on the desktop (selection, the
//! cell cache, decoding and compositing, extraction on the worker pool,
//! lights from hot matter), fed from in-memory chunk containers instead of
//! the hub. The image is checked by pixel statistics and saved as
//! `target/tmp/matter-render.png` with the statistics beside it in
//! `target/tmp/matter-render.txt`.

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
    let system = system();
    let camera = camera();
    let chunks = chunks();
    world.select(&system, &camera, (WIDTH, HEIGHT), 0.0, true);
    let requests = world.take_requests(0.0);
    assert_eq!(requests.len(), 2, "{requests:?}");
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
    assert_eq!(scene.surfaces.len(), 2);
    assert_eq!(scene.lights.len(), 1);
    assert_eq!(scene.lights[0].frame_id, ROOT);

    let mut headless = Headless::new(WIDTH, HEIGHT).expect("a wgpu adapter, software is fine");
    let first = headless.render(&scene, None).unwrap();
    let stats = headless.stats();
    let second = headless.render(&scene, None).unwrap();
    assert!(first == second, "two renders of the same state differ");
    assert_eq!(stats.meshes, 2);
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
         meshes {}, triangles {}, lights {}\n",
        hot.x, hot.y, hot.r, solid.x, solid.y, solid.r, stats.meshes, stats.triangles, stats.lights
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

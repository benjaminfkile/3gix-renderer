//! Far field sprites through the whole pipeline: selection, the pinned
//! depth-0 cell, the draw list, and the sprite pass.
//!
//! A massive frame 2000 m away with a `root_extent / 8` region of 1 m
//! projects to well under 2 pixels, so its cells are not selected and it is
//! drawn as one point sprite worked out from its depth-0 cell.

use gx_core::container::encode_chunk;
use gx_core::frames::FrameSystem;
use gx_core::key::CellKey;
use gx_core::matter::{encode, Compression, Sample, Samples, Section, State};
use gx_core::registry::{Frame, FrameTree, Registry, ROOT_PARENT};
use gx_core::units::{Attenuation, Density, Kelvin, Kilograms, Meters, Quat, Ratio, Seconds, Vec3};
use renderer::app::matter_scene;
use renderer::camera::{look_rotation, Camera};
use renderer::farfield::{depth_zero_key, far_frames};
use renderer::render::gpu::{fixed_exposure, Exposure, REFERENCE_LUMINANCE};
use renderer::render::headless::Headless;
use renderer::render::scene::Scene;
use renderer::stream::{decode_cell, FetchOutcome};
use renderer::world::World;
use std::collections::BTreeMap;

const WIDTH: u32 = 320;
const HEIGHT: u32 = 180;
const ROOT: u64 = 1;
const EXTENT: f64 = 8.0;

fn frame(id: u64, parent: u64, extent: f64, mass: f64, position: Vec3) -> Frame {
    Frame {
        frame_id: id,
        parent_frame_id: parent,
        root_extent: Meters::new(extent),
        max_depth: 0,
        mass: Kilograms::new(mass),
        position,
        velocity: Vec3::zero(),
        orientation: Quat::identity(),
        angular_velocity: Vec3::zero(),
    }
}

/// A root of mass 0 around the camera and massive frames at the given
/// positions, ids 2 up.
fn system(far: &[Vec3]) -> FrameSystem {
    let mut frames = vec![frame(ROOT, ROOT_PARENT, 100.0, 0.0, Vec3::zero())];
    for (i, p) in far.iter().enumerate() {
        frames.push(frame(i as u64 + 2, ROOT, EXTENT, 1.0e6, *p));
    }
    let reg = Registry::new(Seconds::new(0.0), frames).unwrap();
    FrameSystem::from_tree(FrameTree::from_registries(&[reg]).unwrap())
}

/// A depth-0 cell filled at its center with one kind of matter.
fn cell(frame_id: u64, sample: Sample) -> Vec<u8> {
    let key = depth_zero_key(frame_id);
    let g = key.geometry(Meters::new(EXTENT));
    let samples = Samples::from_fn(4, |x, y, z| {
        if (1..3).contains(&x) && (1..3).contains(&y) && (1..3).contains(&z) {
            sample
        } else {
            Sample::VACUUM
        }
    });
    let s = Section::new(key, g.origin, g.edge, 4, samples).unwrap();
    encode_chunk(&[&encode(&s, Compression::None)], &["layer-a"])
}

fn hot() -> Sample {
    Sample {
        density: Density::new(1.0),
        state: State::Plasma,
        temperature: Kelvin::new(6000.0),
        albedo: [Ratio::new(0.0); 3],
        roughness: Ratio::new(1.0),
        attenuation: Attenuation::new(1.0),
    }
}

fn cold() -> Sample {
    Sample {
        density: Density::new(3000.0),
        state: State::Solid,
        temperature: Kelvin::new(300.0),
        albedo: [Ratio::new(0.4); 3],
        roughness: Ratio::new(0.8),
        attenuation: Attenuation::new(0.0),
    }
}

/// At the root origin looking along `+y`, `+z` up.
fn camera() -> Camera {
    Camera {
        frame_id: ROOT,
        position: Vec3::zero(),
        orientation: look_rotation(Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, 0.0, 1.0)),
        speed: 1.0,
    }
}

/// Runs selection and fetching until every selected and pinned cell is
/// resolved, serving `chunks` and `404` for anything else, and returns the
/// matter-only scene and the keys that were requested.
fn stream(system: &FrameSystem, chunks: &BTreeMap<CellKey, Vec<u8>>) -> (Scene, Vec<CellKey>) {
    let camera = camera();
    let mut world = World::new(0);
    world.select(system, &camera, (WIDTH, HEIGHT), 0.0, true);
    let mut requested = Vec::new();
    for round in 0..8 {
        let keys = world.take_requests(f64::from(round));
        for key in keys {
            let extent = system.tree().get(key.frame_id).unwrap().root_extent;
            let outcome = match chunks.get(&key) {
                Some(bytes) => decode_cell(&key, bytes, extent),
                None => FetchOutcome::NotFound,
            };
            world.complete(key, outcome, f64::from(round));
            requested.push(key);
        }
        world.finish_extraction();
        if world.settled() {
            break;
        }
    }
    assert!(world.settled());
    let mut scene = matter_scene(&mut world, system, &camera, (WIDTH, HEIGHT), 1.0);
    scene.markers.clear();
    scene.lines.clear();
    (scene, requested)
}

#[test]
fn far_hot_frame_is_a_sprite_where_it_projects() {
    let system = system(&[Vec3::new(30.0, 2000.0, 15.0)]);
    let camera = camera();
    let far = far_frames(&system, &camera, f64::from(HEIGHT));
    assert_eq!(far.len(), 1);
    assert!(far[0].sprite_only(), "{far:?}");
    assert!(far[0].projected_px < 2.0);

    let chunks: BTreeMap<CellKey, Vec<u8>> = [(depth_zero_key(2), cell(2, hot()))].into();
    let (scene, requested) = stream(&system, &chunks);
    // The far frame's depth-0 cell was fetched for its brightness, and no
    // other cell of that frame was selected.
    assert!(requested.contains(&depth_zero_key(2)));
    assert_eq!(requested.iter().filter(|k| k.frame_id == 2).count(), 1);
    assert!(scene.surfaces.is_empty() && scene.volumes.is_empty());
    assert_eq!(scene.sprites.len(), 1);
    // A hot far frame still lights the rest.
    assert_eq!(scene.lights.len(), 1);
    let sprite = scene.sprites[0];
    assert!(sprite.size_px >= 2.0 && sprite.size_px <= 6.0);

    let (x, y) = scene
        .project(sprite.position, WIDTH, HEIGHT)
        .expect("in front of the camera");
    let (px, py) = (x as u32, y as u32);
    let mut headless = Headless::new(WIDTH, HEIGHT).expect("a wgpu adapter, software is fine");
    let image = headless.render(&scene, None).unwrap();
    assert_eq!(headless.stats().sprites, 1);
    let p = image.pixel(px, py);
    println!(
        "sprite at ({x:.1}, {y:.1}), {} px, pixel {p:?}",
        sprite.size_px
    );
    assert!(p[0] > 0 && p[1] > 0 && p[2] > 0, "{p:?}");
    // Nothing anywhere else: every lit pixel is within the sprite.
    for yy in 0..HEIGHT {
        for xx in 0..WIDTH {
            let q = image.pixel(xx, yy);
            if q[0] > 0 || q[1] > 0 || q[2] > 0 {
                let d =
                    (f64::from(xx) + 0.5 - f64::from(x)).hypot(f64::from(yy) + 0.5 - f64::from(y));
                assert!(
                    d <= f64::from(sprite.size_px),
                    "lit pixel ({xx}, {yy}) at {d} px"
                );
            }
        }
    }
    // Two renders of one state are one image.
    assert!(image == headless.render(&scene, None).unwrap());
}

#[test]
fn fixed_exposure_is_relative_to_the_reference_scene() {
    let system = system(&[Vec3::new(30.0, 2000.0, 15.0)]);
    let chunks: BTreeMap<CellKey, Vec<u8>> = [(depth_zero_key(2), cell(2, hot()))].into();
    let (mut scene, _) = stream(&system, &chunks);
    let (x, y) = scene
        .project(scene.sprites[0].position, WIDTH, HEIGHT)
        .expect("in front of the camera");
    let (px, py) = (x as u32, y as u32);
    let mut headless = Headless::new(WIDTH, HEIGHT).expect("a wgpu adapter, software is fine");
    let fixed = |stops: f64| Exposure {
        fixed_stops: Some(stops),
        ..Exposure::default()
    };
    for radiance in [1.0f32, 1.0e-4] {
        // A grey sprite is the only covered thing, so the automatic
        // exposure maps its radiance to middle grey. A fixed exposure of
        // log2(1 W m^-2 sr^-1 / radiance) stops does the same: 0.18 before
        // the curve, 0.267 after it, 141 in sRGB.
        scene.sprites[0].radiance = [radiance; 3];
        let stops = (REFERENCE_LUMINANCE / f64::from(radiance)).log2();
        let auto = headless.render(&scene, None).unwrap().pixel(px, py);
        let same = headless
            .render_with(&scene, None, fixed(stops))
            .unwrap()
            .pixel(px, py);
        println!("radiance {radiance}: stops {stops:.3}, auto {auto:?}, fixed {same:?}");
        for c in 0..3 {
            assert!((139..=143).contains(&auto[c]), "{auto:?}");
            assert!(auto[c].abs_diff(same[c]) <= 1, "{auto:?} {same:?}");
        }
        // One stop up is brighter, twenty down is black: nothing adapts.
        let up = headless
            .render_with(&scene, None, fixed(stops + 1.0))
            .unwrap()
            .pixel(px, py);
        let down = headless
            .render_with(&scene, None, fixed(stops - 20.0))
            .unwrap()
            .pixel(px, py);
        assert!(up[1] > same[1] + 20, "{up:?}");
        assert_eq!(&down[..3], &[0, 0, 0], "{down:?}");
    }
    assert_eq!(fixed_exposure(0.0), 0.18);
    assert_eq!(fixed_exposure(2.0), 0.72);
}

#[test]
fn far_cold_frame_needs_light_from_its_side() {
    let cold_at = Vec3::new(0.0, 2000.0, 0.0);
    let chunks: BTreeMap<CellKey, Vec<u8>> = [(depth_zero_key(2), cell(2, cold()))].into();

    // No light anywhere: nothing at all.
    let system_dark = system(&[cold_at]);
    let (scene, requested) = stream(&system_dark, &chunks);
    assert!(requested.contains(&depth_zero_key(2)));
    assert!(scene.sprites.is_empty());
    assert!(scene.lights.is_empty());
    let mut headless = Headless::new(WIDTH, HEIGHT).expect("a wgpu adapter, software is fine");
    let image = headless.render(&scene, None).unwrap();
    assert!(image.rgba.chunks(4).all(|p| p[..3] == [0, 0, 0]));
    assert!(headless
        .read_radiance()
        .unwrap()
        .texels
        .iter()
        .all(|t| *t == [0.0; 4]));

    // A far hot frame behind the camera lights the side the camera sees:
    // a faint dot. Beyond the cold frame, it lights the far side: nothing.
    let mut lit_chunks = chunks.clone();
    lit_chunks.insert(depth_zero_key(3), cell(3, hot()));
    let behind = system(&[cold_at, Vec3::new(0.0, -3000.0, 0.0)]);
    let (scene, _) = stream(&behind, &lit_chunks);
    assert_eq!(scene.lights.len(), 1);
    let cold_sprite = scene
        .sprites
        .iter()
        .find(|s| s.frame_id == 2)
        .expect("the lit side faces the camera");
    assert!(cold_sprite.radiance.iter().all(|&l| l > 0.0));
    let beyond = system(&[cold_at, Vec3::new(0.0, 5000.0, 0.0)]);
    let (scene, _) = stream(&beyond, &lit_chunks);
    assert_eq!(scene.lights.len(), 1);
    assert!(scene.sprites.iter().all(|s| s.frame_id != 2));
}

#[test]
fn transition_blends_the_sprite_out_as_the_cells_blend_in() {
    // Region 1 m at 31.2 m: about 5 px, halfway through the transition.
    let system = system(&[Vec3::new(0.0, 31.2, 0.0)]);
    let far = far_frames(&system, &camera(), f64::from(HEIGHT));
    assert_eq!(far.len(), 1);
    let w = far[0].sprite_weight;
    assert!(!far[0].sprite_only() && w > 0.4 && w < 0.6, "{far:?}");

    let chunks: BTreeMap<CellKey, Vec<u8>> = [(depth_zero_key(2), cell(2, hot()))].into();
    let (scene, _) = stream(&system, &chunks);
    assert_eq!(scene.volumes.len(), 1);
    assert_eq!(scene.sprites.len(), 1);
    assert!((f64::from(scene.volumes[0].weight) - (1.0 - w)).abs() < 1e-6);
    let mut headless = Headless::new(WIDTH, HEIGHT).expect("a wgpu adapter, software is fine");
    headless.render(&scene, None).unwrap();
    let stats = headless.stats();
    assert_eq!((stats.volumes, stats.sprites), (1, 1));
}

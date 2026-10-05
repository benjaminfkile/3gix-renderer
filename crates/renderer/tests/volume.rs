//! The ray marched volume on the GPU against the CPU reference.
//!
//! One cell of uniform plasma at 6000 K, seen along an axis through its
//! center. The center pixel of the float radiance target (before exposure)
//! must match the analytic result of the same emission and extinction model,
//! `E * (1 - T)` with `T` from `gx_core::extinction::transmittance` over the
//! cube depth, within 2 percent in every band, and the CPU mirror of the
//! march ([`VolumeGrid::march`]) much more closely.

use gx_core::frames::FrameSystem;
use gx_core::key::CellKey;
use gx_core::matter::{Sample, Samples, Section, State};
use gx_core::radiance::emitted_band_radiance;
use gx_core::registry::{Frame, FrameTree, Registry, ROOT_PARENT};
use gx_core::units::{Attenuation, Density, Kelvin, Kilograms, Meters, Quat, Ratio, Seconds, Vec3};
use renderer::camera::{look_rotation, Camera};
use renderer::render::headless::Headless;
use renderer::render::scene::{add_matter, build_scene};
use renderer::volume::{uniform_radiance, VolumeGrid, VOLUME_STEPS};
use renderer::world::DrawList;
use std::collections::BTreeMap;
use std::sync::Arc;

/// Odd, so one pixel sits exactly on the axis.
const SIZE: u32 = 65;
const EDGE: f64 = 8.0;
const DENSITY: f64 = 0.5;
/// Extinction 0.125 per meter: optical depth 1 across the cube.
const ATTENUATION: f64 = 0.25;
const ALBEDO: f32 = 0.2;
const DISTANCE: f64 = 20.0;

fn system() -> FrameSystem {
    let root = Frame {
        frame_id: 1,
        parent_frame_id: ROOT_PARENT,
        root_extent: Meters::new(EDGE),
        max_depth: 0,
        mass: Kilograms::new(0.0),
        position: Vec3::zero(),
        velocity: Vec3::zero(),
        orientation: Quat::identity(),
        angular_velocity: Vec3::zero(),
    };
    let reg = Registry::new(Seconds::new(0.0), vec![root]).unwrap();
    FrameSystem::from_tree(FrameTree::from_registries(&[reg]).unwrap())
}

fn plasma_cell() -> Section {
    let key = CellKey::new(1, 0, 0, 0, 0).unwrap();
    let g = key.geometry(Meters::new(EDGE));
    let sample = Sample {
        density: Density::new(DENSITY),
        state: State::Plasma,
        temperature: Kelvin::new(6000.0),
        albedo: [Ratio::new(f64::from(ALBEDO)); 3],
        roughness: Ratio::new(0.0),
        attenuation: Attenuation::new(ATTENUATION),
    };
    Section::new(key, g.origin, g.edge, 8, Samples::filled(8, sample)).unwrap()
}

#[test]
fn center_pixel_matches_the_cpu_integration() {
    let system = system();
    // On the +z axis looking at the cell center, +y up.
    let camera = Camera {
        frame_id: 1,
        position: Vec3::new(0.0, 0.0, DISTANCE),
        orientation: look_rotation(Vec3::new(0.0, 0.0, -1.0), Vec3::new(0.0, 1.0, 0.0)),
        speed: 1.0,
    };
    let grid = Arc::new(VolumeGrid::from_section(&plasma_cell()).expect("plasma is a volume"));
    let mut scene = build_scene(&system, &camera, 1.0);
    scene.markers.clear();
    scene.lines.clear();
    let draw = DrawList {
        meshes: Vec::new(),
        volumes: vec![grid.clone()],
        lights: Vec::new(),
        sprites: Vec::new(),
        cell_weights: BTreeMap::new(),
    };
    add_matter(&mut scene, &system, &camera, &draw);
    assert_eq!(scene.volumes.len(), 1);

    let mut headless = Headless::new(SIZE, SIZE).expect("a wgpu adapter, software is fine");
    headless.render(&scene, None).unwrap();
    assert_eq!(headless.stats().volumes, 1);
    let radiance = headless.read_radiance().unwrap();
    let center = radiance.pixel(SIZE / 2, SIZE / 2);

    // The analytic reference: emitted band radiance (albedo as stored, f32)
    // times one minus the transmittance over the cube depth.
    let albedo = [Ratio::new(f64::from(ALBEDO)); 3];
    let emitted = emitted_band_radiance(Kelvin::new(6000.0), albedo);
    let want = uniform_radiance(
        emitted,
        Density::new(DENSITY),
        Attenuation::new(ATTENUATION),
        Meters::new(EDGE),
    );
    // The CPU mirror of the march.
    let cpu = grid.march(
        camera.position,
        Vec3::new(0.0, 0.0, -1.0),
        f64::INFINITY,
        VOLUME_STEPS,
    );
    let t = gx_core::extinction::transmittance(
        Density::new(DENSITY),
        Attenuation::new(ATTENUATION),
        Meters::new(EDGE),
    );
    println!(
        "gpu {center:?}\nanalytic {want:?}\ncpu march {:?} transmittance {} (analytic {t})",
        cpu.radiance, cpu.transmittance
    );
    for b in 0..3 {
        let gpu = f64::from(center[b]);
        let rel = (gpu - want[b]).abs() / want[b];
        assert!(
            rel < 0.02,
            "band {b}: gpu {gpu} analytic {} ({rel})",
            want[b]
        );
        let rel_cpu = (gpu - cpu.radiance[b]).abs() / cpu.radiance[b];
        assert!(
            rel_cpu < 1e-3,
            "band {b}: gpu {gpu} cpu march {} ({rel_cpu})",
            cpu.radiance[b]
        );
    }
    // Coverage for the exposure pass: the opacity 1 - T.
    assert!(
        (f64::from(center[3]) - (1.0 - t)).abs() < 1e-3,
        "{center:?}"
    );
    // Outside the cube's projection nothing is drawn.
    assert_eq!(radiance.pixel(0, 0), [0.0; 4]);
}

#[test]
fn rays_stop_at_a_surface_inside_the_volume() {
    // The same plasma cell with a cold solid block in its middle, samples 2
    // to 5 on every axis: z from -2 m to 2 m. The cell draws both a mesh and
    // a volume, and the ray from the camera on the +z axis stops at the
    // block's face at z = 2 m, 18 m away, after 2 m of plasma.
    let system = system();
    let camera = Camera {
        frame_id: 1,
        position: Vec3::new(0.0, 0.0, DISTANCE),
        orientation: look_rotation(Vec3::new(0.0, 0.0, -1.0), Vec3::new(0.0, 1.0, 0.0)),
        speed: 1.0,
    };
    let plasma = plasma_cell();
    let mut samples = plasma.samples().unwrap().clone();
    for z in 2..6 {
        for y in 2..6 {
            for x in 2..6 {
                samples.set(
                    x + 8 * (y + 8 * z),
                    Sample {
                        density: Density::new(3000.0),
                        state: State::Solid,
                        temperature: Kelvin::new(3.0),
                        albedo: [Ratio::new(0.5); 3],
                        roughness: Ratio::new(1.0),
                        attenuation: Attenuation::new(0.0),
                    },
                );
            }
        }
    }
    let mixed = Section::new(plasma.key(), plasma.origin(), plasma.edge(), 8, samples).unwrap();
    let mesh = Arc::new(renderer::extract::extract(&mixed));
    assert!(!mesh.is_empty());
    assert_eq!(mesh.bounds_max[2], 2.0);
    let grid = Arc::new(VolumeGrid::from_section(&mixed).unwrap());

    let mut scene = build_scene(&system, &camera, 1.0);
    scene.markers.clear();
    scene.lines.clear();
    let draw = DrawList {
        meshes: vec![mesh],
        volumes: vec![grid.clone()],
        lights: Vec::new(),
        sprites: Vec::new(),
        cell_weights: BTreeMap::new(),
    };
    add_matter(&mut scene, &system, &camera, &draw);
    let mut headless = Headless::new(SIZE, SIZE).expect("a wgpu adapter, software is fine");
    headless.render(&scene, None).unwrap();
    assert_eq!((headless.stats().meshes, headless.stats().volumes), (1, 1));
    let center = headless.read_radiance().unwrap().pixel(SIZE / 2, SIZE / 2);

    let cpu = grid.march(
        camera.position,
        Vec3::new(0.0, 0.0, -1.0),
        DISTANCE - 2.0,
        VOLUME_STEPS,
    );
    let full = grid.march(
        camera.position,
        Vec3::new(0.0, 0.0, -1.0),
        f64::INFINITY,
        VOLUME_STEPS,
    );
    println!("gpu {center:?}\ncpu to the surface {cpu:?}\ncpu through the cell {full:?}");
    for (b, (cpu, full)) in cpu.radiance.iter().zip(full.radiance).enumerate() {
        let gpu = f64::from(center[b]);
        let rel = (gpu - cpu).abs() / cpu;
        assert!(rel < 1e-3, "band {b}: gpu {gpu} cpu {cpu} ({rel})");
        // Less than the march through the whole cell would give.
        assert!(gpu < 0.9 * full);
    }
    // The surface behind covers the pixel.
    assert_eq!(center[3], 1.0);
}

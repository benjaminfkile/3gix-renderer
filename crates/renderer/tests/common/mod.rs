//! The synthetic scene shared by the matter render test and the mock hub
//! test: one root frame (mass 0) and one child frame, a hot plasma blob at
//! 6000 K in the root frame's depth 0 cell (drawn as a volume), a dense
//! solid blob at 300 K in the child frame's depth 0 cell, and an empty cell.
//!
//! Every blob is built numerically from a radius test on the sample
//! centers. Nothing here is named after anything.

#![allow(dead_code)]

use gx_core::container::encode_chunk;
use gx_core::frames::FrameSystem;
use gx_core::key::CellKey;
use gx_core::matter::{encode, Compression, Sample, Samples, Section, State};
use gx_core::registry::{self, Frame, FrameTree, Registry, ROOT_PARENT};
use gx_core::units::{Attenuation, Density, Kelvin, Kilograms, Meters, Quat, Ratio, Seconds, Vec3};
use renderer::camera::{look_rotation, Camera};
use std::collections::BTreeMap;

/// The root frame: holds the hot blob at its origin.
pub const ROOT: u64 = 1;
/// The child frame: holds the solid blob at its origin.
pub const CHILD: u64 = 2;
/// Root frame cube edge, meters.
pub const ROOT_EXTENT: f64 = 32.0;
/// Child frame cube edge, meters.
pub const CHILD_EXTENT: f64 = 12.0;
/// Child frame origin relative to the root, meters.
pub const CHILD_OFFSET: [f64; 3] = [14.0, 0.0, 0.0];
/// Radius of the hot blob, meters.
pub const HOT_RADIUS: f64 = 2.5;
/// Radius of the solid blob, meters.
pub const SOLID_RADIUS: f64 = 4.0;
/// Temperature of the hot blob, kelvin.
pub const HOT_TEMPERATURE: f64 = 6000.0;
/// Temperature of the solid blob, kelvin.
pub const SOLID_TEMPERATURE: f64 = 300.0;

fn frame(id: u64, parent: u64, extent: f64, mass: f64, position: [f64; 3]) -> Frame {
    Frame {
        frame_id: id,
        parent_frame_id: parent,
        root_extent: Meters::new(extent),
        max_depth: 0,
        mass: Kilograms::new(mass),
        position: Vec3::new(position[0], position[1], position[2]),
        velocity: Vec3::zero(),
        orientation: Quat::identity(),
        angular_velocity: Vec3::zero(),
    }
}

/// The registry: the root (mass 0) and the child.
pub fn registry() -> Registry {
    Registry::new(
        Seconds::new(0.0),
        vec![
            frame(ROOT, ROOT_PARENT, ROOT_EXTENT, 0.0, [0.0; 3]),
            frame(CHILD, ROOT, CHILD_EXTENT, 1.0e3, CHILD_OFFSET),
        ],
    )
    .expect("a valid registry")
}

/// The registry chunk container, as the hub serves it.
pub fn registry_chunk() -> Vec<u8> {
    encode_chunk(&[&registry::encode(&registry())], &["layer-a"])
}

/// The frame system at the epoch.
pub fn system() -> FrameSystem {
    FrameSystem::from_tree(FrameTree::from_registries(&[registry()]).expect("valid union"))
}

/// The hot blob's cell.
pub fn hot_key() -> CellKey {
    CellKey::new(ROOT, 0, 0, 0, 0).unwrap()
}

/// The solid blob's cell.
pub fn solid_key() -> CellKey {
    CellKey::new(CHILD, 0, 0, 0, 0).unwrap()
}

/// A cell whose only section is empty.
pub fn empty_key() -> CellKey {
    CellKey::new(CHILD, 1, 1, 0, 1).unwrap()
}

/// Points per axis of the radius test inside one sample.
const SUBSAMPLES: u32 = 4;

/// A section for `key` of a frame of edge `extent` at `res` samples per
/// axis holding a blob of `radius` around the frame origin. Each sample's
/// density is the mean over its sub-cube, as the format defines it: the
/// matter's density times the fraction of a 4 x 4 x 4 grid of points in the
/// sub-cube that pass the radius test.
fn blob(key: CellKey, extent: f64, res: u8, radius: f64, matter: Sample) -> Section {
    let g = key.geometry(Meters::new(extent));
    let step = g.edge.value() / f64::from(res);
    let sub = step / f64::from(SUBSAMPLES);
    let samples = Samples::from_fn(res, |x, y, z| {
        let mut inside = 0u32;
        for k in 0..SUBSAMPLES {
            for j in 0..SUBSAMPLES {
                for i in 0..SUBSAMPLES {
                    let c = |o: f64, n: u32, m: u32| {
                        o + f64::from(n) * step + (f64::from(m) + 0.5) * sub
                    };
                    let p = Vec3::new(
                        c(g.origin.x, x, i),
                        c(g.origin.y, y, j),
                        c(g.origin.z, z, k),
                    );
                    inside += u32::from(p.length() < radius);
                }
            }
        }
        if inside == 0 {
            return Sample::VACUUM;
        }
        let fraction = f64::from(inside) / f64::from(SUBSAMPLES.pow(3));
        Sample {
            density: Density::new(matter.density.value() * fraction),
            ..matter
        }
    });
    Section::new(key, g.origin, g.edge, res, samples).expect("a valid section")
}

/// Density of the hot blob, kilograms per cubic meter.
pub const HOT_DENSITY: f64 = 0.01;
/// Mass attenuation of the hot blob, square meters per kilogram: an
/// extinction of 0.4 per meter, optical depth 2 through its center.
pub const HOT_ATTENUATION: f64 = 20.0;

/// The hot blob section: plasma at 6000 K, emissivity 1, drawn as a volume.
pub fn hot_section() -> Section {
    blob(
        hot_key(),
        ROOT_EXTENT,
        64,
        HOT_RADIUS,
        Sample {
            density: Density::new(HOT_DENSITY),
            state: State::Plasma,
            temperature: Kelvin::new(HOT_TEMPERATURE),
            albedo: [Ratio::new(0.0); 3],
            roughness: Ratio::new(1.0),
            attenuation: Attenuation::new(HOT_ATTENUATION),
        },
    )
}

/// The solid blob section: dense, 300 K, grey.
pub fn solid_section() -> Section {
    blob(
        solid_key(),
        CHILD_EXTENT,
        48,
        SOLID_RADIUS,
        Sample {
            density: Density::new(3000.0),
            state: State::Solid,
            temperature: Kelvin::new(SOLID_TEMPERATURE),
            albedo: [Ratio::new(0.5); 3],
            roughness: Ratio::new(0.8),
            attenuation: Attenuation::new(0.0),
        },
    )
}

/// The chunk containers by key, as the hub would serve them. The solid
/// cell has two layers: the blob and an empty section from another
/// compiler, which the compositing must ignore.
pub fn chunks() -> BTreeMap<CellKey, Vec<u8>> {
    let mut out = BTreeMap::new();
    let hot = encode(&hot_section(), Compression::Zstd);
    out.insert(hot_key(), encode_chunk(&[&hot], &["layer-a"]));
    let solid = encode(&solid_section(), Compression::None);
    let g = solid_key().geometry(Meters::new(CHILD_EXTENT));
    let nothing = encode(
        &Section::empty(solid_key(), g.origin, g.edge).unwrap(),
        Compression::None,
    );
    out.insert(
        solid_key(),
        encode_chunk(&[&solid, &nothing], &["layer-a", "layer-b"]),
    );
    let g = empty_key().geometry(Meters::new(CHILD_EXTENT));
    let empty = encode(
        &Section::empty(empty_key(), g.origin, g.edge).unwrap(),
        Compression::None,
    );
    out.insert(empty_key(), encode_chunk(&[&empty], &["layer-a"]));
    out
}

/// The fixed camera: in the root frame, below the line between the two
/// blobs, looking along `+y` with `+z` up, so the hot blob is on the left of
/// the image and the solid blob on the right.
pub fn camera() -> Camera {
    Camera {
        frame_id: ROOT,
        position: Vec3::new(7.0, -26.0, 0.0),
        orientation: look_rotation(Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, 0.0, 1.0)),
        speed: 1.0,
    }
}

/// The gas frame of the full scene: a child of the root holding a cloud of
/// warm gas at its origin. Mass 0, so it never leaves for the far field.
pub const GAS: u64 = 3;
/// Gas frame cube edge, meters.
pub const GAS_EXTENT: f64 = 8.0;
/// Gas frame origin relative to the root, meters: up and to the left of
/// the hot blob as the fixed camera sees it.
pub const GAS_OFFSET: [f64; 3] = [-12.0, 6.0, 7.0];
/// Radius of the gas cloud, meters.
pub const GAS_RADIUS: f64 = 3.0;
/// Temperature of the gas cloud, kelvin.
pub const GAS_TEMPERATURE: f64 = 4000.0;

/// The registry of the full scene: the D2 frames plus the gas frame.
pub fn full_registry() -> Registry {
    Registry::new(
        Seconds::new(0.0),
        vec![
            frame(ROOT, ROOT_PARENT, ROOT_EXTENT, 0.0, [0.0; 3]),
            frame(CHILD, ROOT, CHILD_EXTENT, 1.0e3, CHILD_OFFSET),
            frame(GAS, ROOT, GAS_EXTENT, 0.0, GAS_OFFSET),
        ],
    )
    .expect("a valid registry")
}

/// The frame system of the full scene at the epoch.
pub fn full_system() -> FrameSystem {
    FrameSystem::from_tree(FrameTree::from_registries(&[full_registry()]).expect("valid union"))
}

/// The gas cloud's cell.
pub fn gas_key() -> CellKey {
    CellKey::new(GAS, 0, 0, 0, 0).unwrap()
}

/// The gas cloud section: gas at 2500 K, extinction 0.5 per meter.
pub fn gas_section() -> Section {
    blob(
        gas_key(),
        GAS_EXTENT,
        32,
        GAS_RADIUS,
        Sample {
            density: Density::new(0.05),
            state: State::Gas,
            temperature: Kelvin::new(GAS_TEMPERATURE),
            albedo: [Ratio::new(0.3); 3],
            roughness: Ratio::new(0.0),
            attenuation: Attenuation::new(10.0),
        },
    )
}

/// The chunk containers of the full scene: the D2 cells plus the gas cell.
pub fn full_chunks() -> BTreeMap<CellKey, Vec<u8>> {
    let mut out = chunks();
    let gas = encode(&gas_section(), Compression::Zstd);
    out.insert(gas_key(), encode_chunk(&[&gas], &["layer-c"]));
    out
}

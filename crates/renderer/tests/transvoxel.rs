//! Regular-cell Transvoxel extraction over synthetic matter
//! (`rendering.md` sections 2.1 to 2.5, same depth only).
//!
//! - **Watertightness.** A 3 x 3 x 3 block of cells of one depth, each with
//!   its 26 neighbors, over a plane, a solid ball, a hollow shell, a ring,
//!   and an overhang: the union of the 27 meshes has no open edge except on
//!   the block's outer boundary, and every inner edge is used once in each
//!   direction (consistent winding).
//! - **Accuracy.** Occupancy that ramps linearly across one sample around
//!   the boundary: every vertex lies within a tenth of a sample spacing of
//!   the true boundary.
//! - **Determinism.** The same bytes for the same inputs whatever the
//!   number of threads extracting.
//!
//! The shapes are signed distance functions in meters; each sample's
//! occupancy is `clamp(0.5 - d / h, 0, 1)` at its center, so the 0.5 level
//! is the shape's surface (`matter-format.md` section 3.3). Cells are 8 m
//! with 8 samples per axis, so every lattice position is exact in `f64`.

use gx_core::key::CellKey;
use gx_core::matter::{Sample, Samples, Section, State};
use gx_core::units::{Attenuation, Density, Kelvin, Meters, Ratio, Vec3};
use renderer::extract::SurfaceMesh;
use renderer::transvoxel::{extract_regular, Neighborhood, Slot};
use std::collections::BTreeMap;
use std::sync::Arc;

/// Edge of one cell, meters.
const EDGE: f64 = 8.0;
/// Samples per cell axis.
const RES: u8 = 8;

/// A synthetic shape.
#[derive(Copy, Clone, Debug)]
enum Shape {
    Plane,
    Ball,
    Shell,
    Ring,
    Overhang,
}

const SHAPES: [Shape; 5] = [
    Shape::Plane,
    Shape::Ball,
    Shape::Shell,
    Shape::Ring,
    Shape::Overhang,
];

/// Signed distance to an axis-aligned box of half extents `h` at `c`.
fn boxed(p: [f64; 3], c: [f64; 3], h: [f64; 3]) -> f64 {
    let d: [f64; 3] = std::array::from_fn(|a| (p[a] - c[a]).abs() - h[a]);
    let out = d.map(|v| v.max(0.0));
    (out[0] * out[0] + out[1] * out[1] + out[2] * out[2]).sqrt() + d[0].max(d[1]).max(d[2]).min(0.0)
}

impl Shape {
    /// Signed distance from `p`, negative inside.
    fn distance(self, p: [f64; 3]) -> f64 {
        let [x, y, z] = p;
        let r = (x * x + y * y + z * z).sqrt();
        match self {
            Shape::Plane => {
                let n: [f64; 3] = [0.3, -0.2, 1.0];
                let l: f64 = n[0] * n[0] + n[1] * n[1] + n[2] * n[2];
                (n[0] * x + n[1] * y + n[2] * z - 1.37) / l.sqrt()
            }
            Shape::Ball => r - 9.3,
            Shape::Shell => (r - 8.2).abs() - 2.1,
            Shape::Ring => {
                let q = (x * x + y * y).sqrt() - 7.5;
                (q * q + z * z).sqrt() - 2.6
            }
            Shape::Overhang => boxed(p, [0.0, 0.0, -3.0], [2.5, 2.5, 8.0]).min(boxed(
                p,
                [1.3, 0.7, 5.5],
                [8.0, 7.0, 1.5],
            )),
        }
    }
}

fn solid(occupancy: f64) -> Sample {
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
}

/// The section of cell `c` (cell indices, origin `c * EDGE - EDGE / 2`).
fn section(distance: &dyn Fn([f64; 3]) -> f64, c: [i32; 3]) -> Arc<Section> {
    let key = CellKey::new(
        1,
        4,
        (8 + c[0]) as u32,
        (8 + c[1]) as u32,
        (8 + c[2]) as u32,
    )
    .unwrap();
    let o = c.map(|v| f64::from(v) * EDGE - 0.5 * EDGE);
    let h = EDGE / f64::from(RES);
    let samples = Samples::from_fn(RES, |x, y, z| {
        let p = [x, y, z];
        let at: [f64; 3] = std::array::from_fn(|a| o[a] + (f64::from(p[a]) + 0.5) * h);
        let occupancy = (0.5 - distance(at) / h).clamp(0.0, 1.0);
        if occupancy <= 0.0 {
            Sample::VACUUM
        } else {
            solid(occupancy)
        }
    });
    Arc::new(
        Section::new(
            key,
            Vec3::new(o[0], o[1], o[2]),
            Meters::new(EDGE),
            RES,
            samples,
        )
        .unwrap(),
    )
}

/// Every section of a 5 x 5 x 5 block of cells centered on cell 0.
fn block(distance: &dyn Fn([f64; 3]) -> f64) -> BTreeMap<[i32; 3], Arc<Section>> {
    let mut out = BTreeMap::new();
    for z in -2..=2 {
        for y in -2..=2 {
            for x in -2..=2 {
                out.insert([x, y, z], section(distance, [x, y, z]));
            }
        }
    }
    out
}

fn hood(sections: &BTreeMap<[i32; 3], Arc<Section>>, c: [i32; 3]) -> Neighborhood {
    let mut h = Neighborhood::new(sections[&c].clone());
    for dz in -1..=1 {
        for dy in -1..=1 {
            for dx in -1..=1 {
                let k = [c[0] + dx, c[1] + dy, c[2] + dz];
                h.slots[Neighborhood::slot_index(dx, dy, dz)] = Slot::Present(sections[&k].clone());
            }
        }
    }
    h
}

/// The meshes of the inner 3 x 3 x 3 cells.
fn inner_meshes(sections: &BTreeMap<[i32; 3], Arc<Section>>) -> Vec<SurfaceMesh> {
    let mut out = Vec::new();
    for z in -1..=1 {
        for y in -1..=1 {
            for x in -1..=1 {
                out.push(extract_regular(&hood(sections, [x, y, z])).unwrap());
            }
        }
    }
    out
}

/// Bit pattern of a position, to weld vertices across meshes exactly.
fn bits(p: [f64; 3]) -> [u64; 3] {
    p.map(f64::to_bits)
}

#[test]
fn same_depth_block_is_watertight() {
    let outer = 1.5 * EDGE;
    for shape in SHAPES {
        let meshes = inner_meshes(&block(&|p| shape.distance(p)));
        let mut ids: BTreeMap<[u64; 3], usize> = BTreeMap::new();
        let mut points = Vec::new();
        let mut directed: BTreeMap<(usize, usize), u32> = BTreeMap::new();
        let mut triangles = 0;
        let mut outward = 0;
        for m in &meshes {
            assert_eq!(m.positions.len(), m.vertices.len());
            // Indexed: no position twice within one mesh.
            let mut seen = std::collections::BTreeSet::new();
            for p in &m.positions {
                assert!(seen.insert(bits(*p)), "{shape:?}: duplicate vertex");
            }
            let id: Vec<usize> = m
                .positions
                .iter()
                .map(|p| {
                    let next = ids.len();
                    let i = *ids.entry(bits(*p)).or_insert(next);
                    if i == points.len() {
                        points.push(*p);
                    }
                    i
                })
                .collect();
            for t in m.indices.chunks(3) {
                let v = [id[t[0] as usize], id[t[1] as usize], id[t[2] as usize]];
                triangles += 1;
                for k in 0..3 {
                    *directed.entry((v[k], v[(k + 1) % 3])).or_default() += 1;
                }
                // Winding against the occupancy normal.
                let [a, b, c] = v.map(|i| points[i]);
                let u: [f64; 3] = std::array::from_fn(|i| b[i] - a[i]);
                let w: [f64; 3] = std::array::from_fn(|i| c[i] - a[i]);
                let cross = [
                    u[1] * w[2] - u[2] * w[1],
                    u[2] * w[0] - u[0] * w[2],
                    u[0] * w[1] - u[1] * w[0],
                ];
                let nrm = m.vertices[t[0] as usize].normal;
                let dot: f64 = (0..3).map(|i| cross[i] * f64::from(nrm[i])).sum();
                if dot > 0.0 {
                    outward += 1;
                }
            }
        }
        assert!(triangles > 100, "{shape:?}: {triangles} triangles");
        assert!(
            outward * 100 >= triangles * 99,
            "{shape:?}: {outward} of {triangles} face outward"
        );
        let on_boundary = |p: [f64; 3]| p.iter().any(|v| v.abs() == outer);
        let mut open = 0;
        for (&(a, b), &count) in &directed {
            assert_eq!(count, 1, "{shape:?}: directed edge used {count} times");
            if directed.contains_key(&(b, a)) {
                continue;
            }
            let (pa, pb) = (points[a], points[b]);
            let same_face = (0..3).any(|i| pa[i] == pb[i] && pa[i].abs() == outer);
            assert!(
                on_boundary(pa) && on_boundary(pb) && same_face,
                "{shape:?}: open edge {pa:?} {pb:?} inside the block"
            );
            open += 1;
        }
        println!("{shape:?}: {triangles} triangles, {open} open edges on the block boundary");
    }
}

#[test]
fn vertices_lie_on_the_ramp_boundary() {
    let h = EDGE / f64::from(RES);
    let planes: [[f64; 4]; 4] = [
        [0.0, 0.0, 1.0, 0.31],
        [0.3, -0.2, 1.0, 1.37],
        [1.0, 1.0, 1.0, -0.77],
        [-0.6, 0.8, 0.1, 2.05],
    ];
    for q in planes {
        let l = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2]).sqrt();
        let d = move |p: [f64; 3]| (q[0] * p[0] + q[1] * p[1] + q[2] * p[2] - q[3]) / l;
        let meshes = inner_meshes(&block(&d));
        let mut worst: f64 = 0.0;
        let mut count = 0;
        for m in &meshes {
            for p in &m.positions {
                worst = worst.max(d(*p).abs());
                count += 1;
            }
        }
        assert!(count > 0);
        println!(
            "plane {q:?}: {count} vertices, worst distance {} samples",
            worst / h
        );
        assert!(
            worst <= 0.1 * h,
            "plane {q:?}: vertex {worst} m off the boundary"
        );
    }
}

/// The bytes of a mesh, every field in order.
fn mesh_bytes(m: &SurfaceMesh) -> Vec<u8> {
    let mut out = Vec::new();
    for p in &m.positions {
        for v in p {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    for v in &m.vertices {
        for f in v
            .normal
            .iter()
            .chain(&v.albedo)
            .chain([&v.roughness, &v.temperature, &v.density])
        {
            out.extend_from_slice(&f.to_le_bytes());
        }
        out.extend_from_slice(&v.state.to_le_bytes());
    }
    for i in &m.indices {
        out.extend_from_slice(&i.to_le_bytes());
    }
    for b in m.bounds_min.iter().chain(&m.bounds_max) {
        out.extend_from_slice(&b.to_le_bytes());
    }
    out
}

#[test]
fn identical_bytes_across_thread_counts() {
    let sections = Arc::new(block(&|p| Shape::Shell.distance(p)));
    let cells: Vec<[i32; 3]> = (0..27)
        .map(|i| [i % 3 - 1, (i / 3) % 3 - 1, i / 9 - 1])
        .collect();
    let run = |threads: usize| -> Vec<Vec<u8>> {
        let mut out = vec![Vec::new(); cells.len()];
        std::thread::scope(|s| {
            let handles: Vec<_> = (0..threads)
                .map(|t| {
                    let sections = sections.clone();
                    let cells = &cells;
                    s.spawn(move || {
                        (t..cells.len())
                            .step_by(threads)
                            .map(|i| {
                                (
                                    i,
                                    mesh_bytes(
                                        &extract_regular(&hood(&sections, cells[i])).unwrap(),
                                    ),
                                )
                            })
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            for h in handles {
                for (i, b) in h.join().unwrap() {
                    out[i] = b;
                }
            }
        });
        out
    };
    let one = run(1);
    assert!(one.iter().any(|b| !b.is_empty()));
    for threads in [2, 3, 8] {
        assert_eq!(run(threads), one, "{threads} threads");
    }
}

#[test]
fn missing_neighbor_is_refused() {
    let sections = block(&|p| Shape::Ball.distance(p));
    let mut h = hood(&sections, [0, 0, 0]);
    h.slots[Neighborhood::slot_index(1, 0, -1)] = Slot::Missing;
    assert!(extract_regular(&h).is_err());
    h.slots[Neighborhood::slot_index(1, 0, -1)] = Slot::Empty;
    assert!(extract_regular(&h).is_ok());
}

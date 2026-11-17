//! Mixed-depth Transvoxel extraction over synthetic matter (`rendering.md`
//! sections 2.2, 2.3, and 2.5).
//!
//! - **Watertightness.** Random 2:1 balanced selections (across faces,
//!   edges, and corners) of cells at depths 1 to 4 of a 32 m root cell,
//!   each mixing at least three depths, over a plane, a solid ball, a
//!   hollow shell, a ring, and an overhang. Every selected cell is
//!   extracted from the selection and the samples alone. The union of the
//!   meshes has no open edge except on the root cell's boundary, every
//!   inner edge is used once in each direction, and no triangle of a cell
//!   lies outside that cell (so no two meshes overlap).
//! - **Accuracy.** A plane whose occupancy ramps linearly over a width
//!   wider than the root cell, so every depth's field is the same linear
//!   function: vertices on transition faces lie within a tenth of the finer
//!   sample spacing of the plane.
//!
//! Shapes are signed distance functions in meters; a sample of spacing `s`
//! has occupancy `clamp(0.5 - d / s, 0, 1)` at its center.

use renderer::transvoxel::{extract_mixed, CellAt, MixedField, MixedMesh};
use std::collections::{BTreeMap, BTreeSet};

/// Root cell edge, meters.
const ROOT: f64 = 32.0;
/// Samples per cell axis.
const RES: usize = 4;
/// Deepest depth a selection uses.
const DEEPEST: u8 = 4;
/// Cells per axis at the deepest depth.
const FINEST: i64 = 1 << DEEPEST;

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

/// A selection: leaves of the octree under the root.
struct Selection {
    leaves: BTreeSet<CellAt>,
    /// Depth of the leaf over every deepest-depth cell, x fastest.
    depth_at: Vec<u8>,
}

impl Selection {
    fn new(leaves: BTreeSet<CellAt>) -> Self {
        let mut s = Selection {
            leaves,
            depth_at: Vec::new(),
        };
        s.index();
        s
    }

    fn index(&mut self) {
        let f = FINEST as usize;
        self.depth_at = vec![0; f * f * f];
        for l in &self.leaves {
            let k = 1i64 << (DEEPEST - l.depth);
            for z in 0..k {
                for y in 0..k {
                    for x in 0..k {
                        let c = [l.index[0] * k + x, l.index[1] * k + y, l.index[2] * k + z];
                        self.depth_at[(c[0] + FINEST * (c[1] + FINEST * c[2])) as usize] = l.depth;
                    }
                }
            }
        }
    }

    fn depth_of(&self, c: [i64; 3]) -> Option<u8> {
        if c.iter().any(|&v| !(0..FINEST).contains(&v)) {
            return None;
        }
        Some(self.depth_at[(c[0] + FINEST * (c[1] + FINEST * c[2])) as usize])
    }

    /// Splits leaves until no two touching leaves are more than one depth
    /// apart, across faces, edges, or corners.
    fn balance(&mut self) {
        loop {
            let mut split = Vec::new();
            for l in &self.leaves {
                let k = 1i64 << (DEEPEST - l.depth);
                let lo = l.index.map(|v| v * k - 1);
                let hi = l.index.map(|v| (v + 1) * k);
                let mut bad = false;
                'scan: for z in lo[2]..=hi[2] {
                    for y in lo[1]..=hi[1] {
                        for x in lo[0]..=hi[0] {
                            let edge = [x, y, z]
                                .iter()
                                .enumerate()
                                .any(|(a, &v)| v == lo[a] || v == hi[a]);
                            if !edge {
                                continue;
                            }
                            if self.depth_of([x, y, z]).is_some_and(|d| d > l.depth + 1) {
                                bad = true;
                                break 'scan;
                            }
                        }
                    }
                }
                if bad {
                    split.push(*l);
                }
            }
            if split.is_empty() {
                return;
            }
            for l in split {
                self.split(l);
            }
            self.index();
        }
    }

    fn split(&mut self, l: CellAt) {
        self.leaves.remove(&l);
        for o in 0..8 {
            self.leaves.insert(CellAt {
                depth: l.depth + 1,
                index: [
                    2 * l.index[0] + (o & 1),
                    2 * l.index[1] + ((o >> 1) & 1),
                    2 * l.index[2] + ((o >> 2) & 1),
                ],
            });
        }
    }
}

/// Deterministic pseudo random numbers (a 64 bit linear congruential
/// generator).
struct Lcg(u64);

impl Lcg {
    fn unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// A random balanced selection mixing at least three depths.
fn random_selection(rng: &mut Lcg) -> Selection {
    loop {
        let mut leaves = BTreeSet::new();
        for o in 0..8i64 {
            leaves.insert(CellAt {
                depth: 1,
                index: [o & 1, (o >> 1) & 1, (o >> 2) & 1],
            });
        }
        let mut s = Selection::new(leaves);
        for (depth, p) in [(1u8, 0.6), (2, 0.25), (3, 0.12)] {
            let at: Vec<CellAt> = s
                .leaves
                .iter()
                .copied()
                .filter(|l| l.depth == depth)
                .collect();
            for l in at {
                if rng.unit() < p {
                    s.split(l);
                }
            }
        }
        s.index();
        s.balance();
        let depths: BTreeSet<u8> = s.leaves.iter().map(|l| l.depth).collect();
        if depths.len() >= 3 {
            return s;
        }
    }
}

/// The samples of a shape at every depth and a selection.
struct Field<'a> {
    shape: &'a dyn Fn([f64; 3]) -> f64,
    selection: &'a Selection,
    /// Width of the occupancy ramp, meters; the sample spacing if `None`.
    ramp: Option<f64>,
}

/// Sample spacing at depth `d`, meters.
fn sample_spacing(d: u8) -> f64 {
    ROOT / (RES as f64 * (1u64 << d) as f64)
}

impl MixedField for Field<'_> {
    fn resolution(&self) -> usize {
        RES
    }

    fn root_origin(&self) -> [f64; 3] {
        [-0.5 * ROOT; 3]
    }

    fn root_edge(&self) -> f64 {
        ROOT
    }

    fn drawn_depth(&self, d: u8, c: [i64; 3]) -> Option<u8> {
        // The deepest-depth cell holding the center (c + 1/2) of depth d.
        let n = RES as i64;
        let (num, den) = if d <= DEEPEST {
            (1i64 << (DEEPEST - d), 2 * n)
        } else {
            (1, 2 * n * (1i64 << (d - DEEPEST)))
        };
        let cell = c.map(|v| (2 * v + 1) * num).map(|v| v.div_euclid(den));
        self.selection.depth_of(cell)
    }

    fn occupancy(&self, d: u8, c: [i64; 3]) -> f64 {
        let s = sample_spacing(d);
        let limit = RES as i64 * (1i64 << d);
        if c.iter().any(|&v| !(0..limit).contains(&v)) {
            return 0.0;
        }
        let at = c.map(|v| -0.5 * ROOT + (v as f64 + 0.5) * s);
        (0.5 - (self.shape)(at) / self.ramp.unwrap_or(s)).clamp(0.0, 1.0)
    }
}

fn bits(p: [f64; 3]) -> [u64; 3] {
    p.map(f64::to_bits)
}

/// Bounds of a cell, meters.
fn cell_box(l: CellAt) -> ([f64; 3], [f64; 3]) {
    let e = ROOT / (1u64 << l.depth) as f64;
    let lo = l.index.map(|v| -0.5 * ROOT + v as f64 * e);
    (lo, lo.map(|v| v + e))
}

/// Checks one selection; returns the number of transition faces seen.
fn check(shape: Shape, s: &Selection) -> usize {
    let distance = |p: [f64; 3]| shape.distance(p);
    let field = Field {
        shape: &distance,
        selection: s,
        ramp: None,
    };
    let mut directed: BTreeMap<([u64; 3], [u64; 3]), usize> = BTreeMap::new();
    let mut transitions = 0;
    let mut triangles = 0;
    for &l in &s.leaves {
        let m: MixedMesh = extract_mixed(&field, l);
        transitions += m.transition_faces.iter().filter(|&&f| f).count();
        let (lo, hi) = cell_box(l);
        let tol = 1e-9 * ROOT;
        for t in m.indices.chunks(3) {
            triangles += 1;
            let p = [t[0], t[1], t[2]].map(|i| m.positions[i as usize]);
            for a in 0..3 {
                let c = (p[0][a] + p[1][a] + p[2][a]) / 3.0;
                assert!(
                    c >= lo[a] - tol && c <= hi[a] + tol,
                    "{shape:?}: triangle of {l:?} outside its cell"
                );
            }
            for k in 0..3 {
                let e = (bits(p[k]), bits(p[(k + 1) % 3]));
                *directed.entry(e).or_insert(0) += 1;
            }
        }
    }
    assert!(triangles > 0, "{shape:?}: no surface");
    let on_root = |p: [u64; 3]| {
        p.iter()
            .any(|&b| (f64::from_bits(b).abs() - 0.5 * ROOT).abs() < 1e-12)
    };
    for (&(a, b), &count) in &directed {
        if on_root(a) && on_root(b) {
            continue;
        }
        let back = directed.get(&(b, a)).copied().unwrap_or(0);
        assert!(
            count == 1 && back == 1,
            "{shape:?}: edge {:?} -> {:?} used {count} times, {back} back ({} cells)",
            a.map(f64::from_bits),
            b.map(f64::from_bits),
            s.leaves.len()
        );
    }
    transitions
}

#[test]
fn mixed_depth_selections_are_watertight() {
    let mut rng = Lcg(0x5EED);
    let mut coverage: BTreeMap<String, (usize, usize, BTreeSet<u8>)> = BTreeMap::new();
    for i in 0..200 {
        let s = random_selection(&mut rng);
        let shape = SHAPES[i % SHAPES.len()];
        let faces = check(shape, &s);
        let e = coverage.entry(format!("{shape:?}")).or_default();
        e.0 += 1;
        e.1 += faces;
        e.2.extend(s.leaves.iter().map(|l| l.depth));
    }
    for (shape, (selections, faces, depths)) in &coverage {
        println!(
            "proof: {shape}: {selections} selections, depths {depths:?}, {faces} transition faces"
        );
    }
}

#[test]
fn transition_vertices_lie_on_the_ramp_boundary() {
    // One depth 1 cell next to depth 2 cells: every other depth 1 cell
    // split, then balanced.
    let mut leaves = BTreeSet::new();
    for o in 0..8i64 {
        leaves.insert(CellAt {
            depth: 1,
            index: [o & 1, (o >> 1) & 1, (o >> 2) & 1],
        });
    }
    let mut s = Selection::new(leaves);
    for l in [[1, 0, 0], [0, 1, 0], [1, 1, 1], [0, 0, 1]] {
        s.split(CellAt { depth: 1, index: l });
    }
    s.index();
    s.balance();
    let n: [f64; 3] = [0.3, -0.2, 1.0];
    let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    let plane = |p: [f64; 3]| (n[0] * p[0] + n[1] * p[1] + n[2] * p[2] - 1.37) / len;
    let field = Field {
        shape: &plane,
        selection: &s,
        ramp: Some(64.0),
    };
    let mut checked = 0;
    for &l in &s.leaves {
        let m = extract_mixed(&field, l);
        let (lo, hi) = cell_box(l);
        let fine = sample_spacing(l.depth + 1);
        for (f, &on) in m.transition_faces.iter().enumerate() {
            if !on {
                continue;
            }
            let a = f / 2;
            let face = if f % 2 == 1 { hi[a] } else { lo[a] };
            for p in &m.positions {
                let inner = p
                    .iter()
                    .all(|v| v.abs() < 0.5 * ROOT - sample_spacing(l.depth));
                if p[a] == face && inner {
                    let d = plane(*p).abs();
                    assert!(d <= 0.1 * fine, "vertex {p:?} is {d} m off the plane");
                    checked += 1;
                }
            }
        }
    }
    assert!(checked > 0, "no vertex on a transition face");
}

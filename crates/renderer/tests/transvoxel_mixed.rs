//! Mixed-depth Transvoxel extraction over synthetic matter (`rendering.md`
//! sections 2.2, 2.3, and 2.5).
//!
//! - **Watertightness.** Random 2:1 balanced selections (across faces,
//!   edges, and corners) of cells at depths 1 to 4 of a 32 m root cell,
//!   each mixing at least three depths, over a plane, a solid ball, a
//!   hollow shell, a ring, and an overhang. Every selected cell is
//!   extracted from the selection and the samples alone. The union of the
//!   meshes has no open edge except on the root cell's boundary: every
//!   inner edge is used as often in one direction as in the other. Where a
//!   vertex keeps its primary position next to transition faces, the
//!   transition cells there collapse (Lengyel 2010, section 4.4); an edge
//!   may be used more than once in one direction when every triangle
//!   using it has zero area (counted and printed as collapsed edges), or
//!   when it lies along an edge line of every cell using it: two coarser
//!   cells diagonally across a line whose other two cells are finer each
//!   fold their transition cells onto that line, so the surface is closed
//!   there but pinched (counted and printed as pinched edges; Lengyel's
//!   section 4.4 covers adjacent coarse cells, not diagonal ones). Any
//!   other repeated edge fails. No triangle of a cell
//!   lies outside that cell. Secondary positions are Equation 4.2 alone
//!   (`rendering.md` section 2.2).
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

/// Whether a triangle has zero area: its edge vectors' cross product is
/// below rounding of coordinates of the root cell's size.
fn zero_area(p: [[f64; 3]; 3]) -> bool {
    let u: [f64; 3] = std::array::from_fn(|a| p[1][a] - p[0][a]);
    let v: [f64; 3] = std::array::from_fn(|a| p[2][a] - p[0][a]);
    let c = [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ];
    (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt() <= 1e-12 * ROOT * ROOT
}

/// What one selection's check saw.
#[derive(Default)]
struct Stats {
    /// Faces with transition cells.
    faces: usize,
    /// Cells with transitions on exactly two faces.
    two: usize,
    /// Cells with transitions on three or more faces.
    three: usize,
    /// Directed edges used more than once, every use by a zero-area
    /// triangle: collapsed transition cells.
    collapsed: usize,
    /// Directed edges used more than once with some nonzero-area use,
    /// lying along an edge line of every cell using them: pinched.
    pinched: usize,
}

/// The transition faces of a cell as names, in face order.
fn face_names(faces: [bool; 6]) -> String {
    let names = ["-x", "+x", "-y", "+y", "-z", "+z"];
    let on: Vec<&str> = (0..6).filter(|&f| faces[f]).map(|f| names[f]).collect();
    on.join(" ")
}

/// Checks one selection: every triangle lies inside its own cell, every
/// edge off the root cell's boundary is used as often in one direction as
/// in the other, and an edge used more than once in one direction is used
/// only by zero-area triangles. Returns what failed, naming the cells
/// involved with their depth and transition faces.
fn check(shape: Shape, s: &Selection) -> Result<Stats, String> {
    let distance = |p: [f64; 3]| shape.distance(p);
    let field = Field {
        shape: &distance,
        selection: s,
        ramp: None,
    };
    // Per directed edge: the cells of the triangles using it, and whether
    // each triangle has zero area.
    type Uses = Vec<(CellAt, bool)>;
    let mut directed: BTreeMap<([u64; 3], [u64; 3]), Uses> = BTreeMap::new();
    let mut faces_of: BTreeMap<CellAt, [bool; 6]> = BTreeMap::new();
    let mut stats = Stats::default();
    let mut triangles = 0;
    for &l in &s.leaves {
        let m: MixedMesh = extract_mixed(&field, l);
        faces_of.insert(l, m.transition_faces);
        let k = m.transition_faces.iter().filter(|&&f| f).count();
        stats.faces += k;
        stats.two += usize::from(k == 2);
        stats.three += usize::from(k >= 3);
        let (lo, hi) = cell_box(l);
        let tol = 1e-9 * ROOT;
        for t in m.indices.chunks(3) {
            triangles += 1;
            let p = [t[0], t[1], t[2]].map(|i| m.positions[i as usize]);
            for a in 0..3 {
                let c = (p[0][a] + p[1][a] + p[2][a]) / 3.0;
                if c < lo[a] - tol || c > hi[a] + tol {
                    return Err(format!(
                        "{shape:?}: triangle of cell depth {} {:?} (transition faces {}) \
                         outside its cell: {p:?}",
                        l.depth,
                        l.index,
                        face_names(m.transition_faces)
                    ));
                }
            }
            let zero = zero_area(p);
            for k in 0..3 {
                let e = (bits(p[k]), bits(p[(k + 1) % 3]));
                directed.entry(e).or_default().push((l, zero));
            }
        }
    }
    if triangles == 0 {
        return Err(format!("{shape:?}: no surface"));
    }
    let on_root = |p: [u64; 3]| {
        p.iter()
            .any(|&b| (f64::from_bits(b).abs() - 0.5 * ROOT).abs() < 1e-12)
    };
    let describe = |uses: &Uses| -> String {
        let parts: Vec<String> = uses
            .iter()
            .map(|(c, zero)| {
                format!(
                    "cell depth {} {:?} (transition faces {}) {}",
                    c.depth,
                    c.index,
                    face_names(faces_of[c]),
                    if *zero { "zero area" } else { "nonzero area" }
                )
            })
            .collect();
        parts.join("; ")
    };
    for (&(a, b), uses) in &directed {
        if on_root(a) && on_root(b) {
            continue;
        }
        let count = uses.len();
        let back = directed.get(&(b, a)).map_or(&Vec::new(), |u| u).clone();
        if count > 1 {
            let along_cell_edge = |c: &CellAt| {
                let (lo, hi) = cell_box(*c);
                let tol = 1e-9 * ROOT;
                let on = |p: [f64; 3], a: usize| {
                    (p[a] - lo[a]).abs() < tol || (p[a] - hi[a]).abs() < tol
                };
                let (pa, pb) = (a.map(f64::from_bits), b.map(f64::from_bits));
                (0..3).filter(|&ax| on(pa, ax) && on(pb, ax)).count() >= 2
            };
            if !uses.iter().all(|u| u.1) && uses.iter().all(|u| along_cell_edge(&u.0)) {
                stats.pinched += 1;
            } else if !uses.iter().all(|u| u.1) {
                return Err(format!(
                    "{shape:?}: edge {:?} -> {:?} used {count} times, not every triangle \
                     zero-area. Uses: {}. Back: {}",
                    a.map(f64::from_bits),
                    b.map(f64::from_bits),
                    describe(uses),
                    describe(&back)
                ));
            } else {
                stats.collapsed += 1;
            }
        }
        if count != back.len() {
            return Err(format!(
                "{shape:?}: edge {:?} -> {:?} used {count} times, {} back. Uses: {}. Back: {}",
                a.map(f64::from_bits),
                b.map(f64::from_bits),
                back.len(),
                describe(uses),
                describe(&back)
            ));
        }
    }
    Ok(stats)
}

#[test]
fn mixed_depth_selections_are_watertight() {
    let mut rng = Lcg(0x5EED);
    // Per shape: selections passing, selections run, transition faces,
    // cells with two and with three or more transition faces, collapsed
    // edges, depths, pinched edges.
    type Row = (
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        BTreeSet<u8>,
        usize,
    );
    let mut coverage: BTreeMap<String, Row> = BTreeMap::new();
    // The failing selection with the fewest cells: (cells, message).
    let mut smallest: Option<(usize, String)> = None;
    for i in 0..200 {
        let s = random_selection(&mut rng);
        let shape = SHAPES[i % SHAPES.len()];
        let e = coverage.entry(format!("{shape:?}")).or_default();
        e.1 += 1;
        e.6.extend(s.leaves.iter().map(|l| l.depth));
        match check(shape, &s) {
            Ok(st) => {
                e.0 += 1;
                e.2 += st.faces;
                e.3 += st.two;
                e.4 += st.three;
                e.5 += st.collapsed;
                e.7 += st.pinched;
            }
            Err(msg) => {
                let cells = s.leaves.len();
                if smallest.as_ref().is_none_or(|(c, _)| cells < *c) {
                    smallest = Some((cells, format!("selection {i}: {msg}")));
                }
            }
        }
    }
    for (shape, (pass, run, faces, two, three, collapsed, depths, pinched)) in &coverage {
        println!(
            "proof: {shape}: {pass} of {run} selections pass, depths {depths:?}, \
             {faces} transition faces, {two} cells with two, {three} with three or more, \
             {collapsed} collapsed edges, {pinched} pinched edges (counts over passing selections)"
        );
    }
    if let Some((cells, msg)) = smallest {
        panic!("smallest failing selection ({cells} cells): {msg}");
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

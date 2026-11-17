//! Cells between depths: ownership of transition faces, transition cells,
//! and the shrinking of regular cells next to them (`rendering.md`
//! sections 2.2, 2.3, and 2.5).
//!
//! # One value per point
//!
//! [`lattice_value`] is the one function every cell calls for a lattice
//! point, so two cells that share the point get the same bits. The value of
//! a point is the mean of the eight sample centers around it at the depth
//! `D` of the finest drawn cell that touches the point (section 2.3). A
//! sample center that lies in a drawn cell of depth `D` is that cell's
//! sample. One that lies in a drawn cell of depth `D - 1` (the coarser side
//! of a transition face) is the coarser field resampled there: trilinear
//! over the eight coarser sample centers around it, with the weights 1/4
//! and 3/4 per axis, summed in a fixed order. Points on a face between two
//! cells of one depth are therefore the plain mean of section 2.3, and
//! points on a face one depth apart are owned by the finer side.
//!
//! # Transition cells
//!
//! On every face of a cell whose neighbor across it is drawn one depth
//! finer, each lattice square of the face becomes a Transvoxel transition
//! cell: nine values on the full resolution face (the finer lattice points,
//! from [`lattice_value`]) and four on the half resolution face, which take
//! the values of the four corners of the square.
//!
//! # Where vertices go (Lengyel 2010, section 4.4)
//!
//! A cell of the selection is a block of the dissertation. Every vertex on
//! a coarse lattice edge of the cell (all regular cell vertices, and the
//! transition cell vertices on the half resolution face) has a primary
//! position, its plain position on the edge, and a secondary position,
//! used when transition cells are drawn next to it. The vertex is near a
//! face of the cell when its primary position lies within one lattice step
//! of that face. It takes the secondary position only when no face it is
//! near has a neighbor drawn at the same depth; otherwise it keeps the
//! primary position. Vertices on the full resolution face of a transition
//! cell are never moved: they are the finer cells' vertices.
//!
//! The secondary position (Equation 4.2) scales the boundary lattice step
//! of each axis whose near face has a transition, toward the cell interior,
//! leaving room [`TRANSITION_WIDTH`] wide for the transition cell. In local
//! lattice coordinates `u` (0 to `n`), with `w` the width in lattice steps:
//! next to a low transition face `u` in `[0, 1]` goes to `w + (1 - w) u`,
//! and next to a high one `u` in `[n - 1, n]` goes to
//! `(n - 1) + (1 - w) (u - n + 1)`; other coordinates are unchanged. A
//! vertex on the half resolution face (at `u = 0` or `u = n`) lands at `w`
//! from the face, so the regular cells and the transition cells meet. The
//! tangent-plane projection of Equation 4.3 is not used (`rendering.md`
//! section 2.2): it moves vertices outside their cell.
//!
//! Where a vertex keeps its primary position next to a transition face,
//! the half resolution face of the transition cell coincides with the cell
//! face, and the transition cell there has zero width: its triangles
//! collapse, as the dissertation's figure 4.13 shows at corners where
//! blocks of one depth meet.
//!
//! The width is half of the adjacent finer regular cell (dissertation,
//! figure 4.11 caption), a quarter of this cell's lattice step.
//!
//! # Determinism
//!
//! A vertex on a lattice edge is computed from the edge's lower end in
//! global lattice coordinates of the edge's depth, so a finer cell's
//! regular cell and a coarser cell's transition cell produce the same bits
//! for a vertex on their shared face.

use super::regular::SURFACE_OCCUPANCY;
use super::tables::{
    transition_class, REGULAR_CELL_CLASS, REGULAR_CELL_DATA, REGULAR_VERTEX_DATA,
    TRANSITION_CELL_DATA, TRANSITION_VERTEX_DATA,
};
use std::collections::BTreeMap;

/// Width of a transition cell as a fraction of the cell's lattice step:
/// half of the adjacent finer regular cell, whose step is half of this
/// cell's (module docs).
pub const TRANSITION_WIDTH: f64 = 0.25;

/// What extraction of a cell reads: the selection around it and the
/// samples of every depth it touches.
///
/// Sample indices are global at their depth: sample `c` at depth `d` has its
/// center at `root_origin + (c + 1/2) * root_edge / (n * 2^d)`.
pub trait MixedField {
    /// Samples per cell axis, the same at every depth.
    fn resolution(&self) -> usize;
    /// Minimum corner of the root cell, meters.
    fn root_origin(&self) -> [f64; 3];
    /// Edge of the root cell, meters.
    fn root_edge(&self) -> f64;
    /// Depth of the drawn cell whose interior holds the center of sample
    /// `c` at depth `d`; `None` outside the selection.
    fn drawn_depth(&self, d: u8, c: [i64; 3]) -> Option<u8>;
    /// Occupancy of sample `c` at depth `d`: its occupancy if solid or
    /// fluid, 0 otherwise or outside the root cell.
    fn occupancy(&self, d: u8, c: [i64; 3]) -> f64;
    /// [`lattice_value`] at point `q` of depth `d` when the field can give
    /// it without the general search, or `None`. An implementation returns
    /// `Some` only where the result is the same bits: a point strictly
    /// inside a drawn cell of depth `d`, whose eight samples are that
    /// cell's own, summed in the order of [`lattice_value`].
    fn interior_value(&self, _d: u8, _q: [i64; 3]) -> Option<f64> {
        None
    }
}

/// A drawn cell: its depth and index at that depth.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CellAt {
    /// Depth of the cell.
    pub depth: u8,
    /// Index of the cell at its depth.
    pub index: [i64; 3],
}

/// One extracted cell: positions in meters, frame coordinates, and
/// counterclockwise triangles seen from outside the matter.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MixedMesh {
    /// Vertex positions, each held once.
    pub positions: Vec<[f64; 3]>,
    /// Outward normals, the negated occupancy gradient.
    pub normals: Vec<[f32; 3]>,
    /// Three indices per triangle.
    pub indices: Vec<u32>,
    /// Faces with a transition, in the order -x, +x, -y, +y, -z, +z.
    pub transition_faces: [bool; 6],
}

/// Global sample index at depth `d` of the sample around lattice point `q`
/// (depth `d`) at offset `o` (bits x, y, z).
fn around(q: [i64; 3], o: usize) -> [i64; 3] {
    [
        q[0] - 1 + (o & 1) as i64,
        q[1] - 1 + ((o >> 1) & 1) as i64,
        q[2] - 1 + ((o >> 2) & 1) as i64,
    ]
}

/// The value at lattice point `q` of depth `d` (module docs). Every cell
/// calls this for every point it reads.
pub fn lattice_value(field: &dyn MixedField, d: u8, q: [i64; 3]) -> f64 {
    if let Some(v) = field.interior_value(d, q) {
        return v;
    }
    // The finest drawn depth touching the point; 2:1 balance keeps it at
    // most one deeper than any touching cell.
    let (mut d, mut q) = (d, q);
    loop {
        let fine = q.map(|v| 2 * v);
        let deeper = (0..8).any(|o| {
            field
                .drawn_depth(d + 1, around(fine, o))
                .is_some_and(|x| x > d)
        });
        if !deeper {
            break;
        }
        d += 1;
        q = fine;
    }
    let mut sum = 0.0;
    for o in 0..8 {
        let c = around(q, o);
        sum += match field.drawn_depth(d, c) {
            Some(x) if x + 1 == d => resampled(field, d, c),
            _ => field.occupancy(d, c),
        };
    }
    sum / 8.0
}

/// The field of depth `d - 1` at the center of sample `c` of depth `d`:
/// trilinear over the coarser sample centers.
fn resampled(field: &dyn MixedField, d: u8, c: [i64; 3]) -> f64 {
    // Per axis: the lower coarser center and the weight of the upper one.
    let lower_and_weight = |v: i64| {
        let m = v.div_euclid(2);
        if v.rem_euclid(2) == 0 {
            (m - 1, 0.75)
        } else {
            (m, 0.25)
        }
    };
    let axes = c.map(lower_and_weight);
    let mut sum = 0.0;
    for o in 0..8 {
        let mut idx = [0i64; 3];
        let mut w = 1.0;
        for a in 0..3 {
            let bit = (o >> a) & 1;
            idx[a] = axes[a].0 + bit as i64;
            w *= if bit == 1 { axes[a].1 } else { 1.0 - axes[a].1 };
        }
        sum += w * field.occupancy(d - 1, idx);
    }
    sum
}

/// Whether the neighbor of `cell` across face `f` (-x, +x, -y, +y, -z,
/// +z) is drawn one depth finer.
pub fn finer_across(field: &dyn MixedField, cell: CellAt, f: usize) -> bool {
    let n = field.resolution() as i64;
    let a = f / 2;
    let high = f % 2 == 1;
    // A sample of depth + 1 just beyond the face, in its middle.
    let fine_n = 2 * n;
    let mut c = cell.index.map(|v| v * fine_n + n);
    c[a] = if high {
        (cell.index[a] + 1) * fine_n
    } else {
        cell.index[a] * fine_n - 1
    };
    field
        .drawn_depth(cell.depth + 1, c)
        .is_some_and(|x| x > cell.depth)
}

/// Whether the neighbor of `cell` across face `f` is drawn at the same
/// depth.
pub fn same_across(field: &dyn MixedField, cell: CellAt, f: usize) -> bool {
    let n = field.resolution() as i64;
    let a = f / 2;
    // A sample of the cell's depth just beyond the face, in its middle.
    let mut c = cell.index.map(|v| v * n + n / 2);
    c[a] = if f % 2 == 1 {
        (cell.index[a] + 1) * n
    } else {
        cell.index[a] * n - 1
    };
    field.drawn_depth(cell.depth, c) == Some(cell.depth)
}

/// Where the vertices on the coarse lattice edges of one cell go (module
/// docs, Lengyel's primary and secondary positions).
struct Placement {
    /// Lattice steps per cell axis.
    n: i64,
    /// Depth of the cell.
    depth: u8,
    /// Lattice step, meters.
    h: f64,
    /// Minimum corner of the root cell, meters.
    root: [f64; 3],
    /// Global lattice index of the cell's minimum corner.
    base: [i64; 3],
    /// Faces with transition cells.
    faces: [bool; 6],
    /// Faces whose neighbor is drawn at the same depth.
    same: [bool; 6],
}

impl Placement {
    /// The vertex at `t` (lattice steps from `lo`) along the coarse
    /// lattice edge from global point `lo` on `axis`.
    fn vertex(&self, b: &mut Builder<'_>, lo: [i64; 3], axis: usize, t: f64) -> u32 {
        let (h, root) = (self.h, self.root);
        // Primary position, from global coordinates: shared bits.
        let mut p: [f64; 3] = std::array::from_fn(|i| root[i] + lo[i] as f64 * h);
        p[axis] = root[axis] + (lo[axis] as f64 + t) * h;
        let mut hi = lo;
        hi[axis] += 1;
        let g0 = b.normal(self.depth, lo);
        let g1 = b.normal(self.depth, hi);
        let g: [f64; 3] = std::array::from_fn(|i| g0[i] + t * (g1[i] - g0[i]));
        let mut local: [f64; 3] = std::array::from_fn(|i| (lo[i] - self.base[i]) as f64);
        local[axis] += t;
        let nf = self.n as f64;
        // Near faces, in face order.
        let near: [bool; 6] = std::array::from_fn(|f| {
            let u = local[f / 2];
            if f % 2 == 1 {
                u > nf - 1.0
            } else {
                u < 1.0
            }
        });
        let secondary = (0..6).all(|f| !near[f] || !self.same[f]);
        let moved = (0..6).any(|f| near[f] && self.faces[f]);
        if secondary && moved {
            // Equation 4.2: scale the boundary step toward the interior.
            let w = TRANSITION_WIDTH;
            let delta: [f64; 3] = std::array::from_fn(|a| {
                let u = local[a];
                let s = if near[2 * a] && self.faces[2 * a] {
                    w + (1.0 - w) * u
                } else if near[2 * a + 1] && self.faces[2 * a + 1] {
                    nf - 1.0 + (1.0 - w) * (u - nf + 1.0)
                } else {
                    u
                };
                (s - u) * h
            });
            for (pi, di) in p.iter_mut().zip(delta) {
                *pi += di;
            }
        }
        b.vertex(p, unit(g))
    }
}

/// Builds meshes with vertices shared by exact position.
struct Builder<'a> {
    field: &'a dyn MixedField,
    mesh: MixedMesh,
    index: BTreeMap<[u64; 3], u32>,
}

impl Builder<'_> {
    fn vertex(&mut self, p: [f64; 3], normal: [f32; 3]) -> u32 {
        let key = p.map(f64::to_bits);
        if let Some(&i) = self.index.get(&key) {
            return i;
        }
        let i = self.mesh.positions.len() as u32;
        self.mesh.positions.push(p);
        self.mesh.normals.push(normal);
        self.index.insert(key, i);
        i
    }

    fn triangle(&mut self, t: [u32; 3]) {
        if t[0] != t[1] && t[1] != t[2] && t[0] != t[2] {
            self.mesh.indices.extend_from_slice(&t);
        }
    }

    /// Negated gradient at point `q` of depth `d`, central differences.
    fn normal(&self, d: u8, q: [i64; 3]) -> [f64; 3] {
        std::array::from_fn(|a| {
            let mut lo = q;
            let mut hi = q;
            lo[a] -= 1;
            hi[a] += 1;
            lattice_value(self.field, d, lo) - lattice_value(self.field, d, hi)
        })
    }
}

/// Lattice spacing at depth `d`, meters.
fn spacing(field: &dyn MixedField, d: u8) -> f64 {
    field.root_edge() / (field.resolution() as f64 * (1u64 << d) as f64)
}

/// One end of a lattice edge: global point at a depth and its value.
#[derive(Copy, Clone)]
struct End {
    q: [i64; 3],
    v: f64,
}

/// The surface crossing on the edge from `a` to `b` (depth `d`, differing
/// on one axis): `(axis, lower point, t)` with `t` measured from the lower
/// end, so both cells sharing the edge get the same bits.
fn crossing(a: End, b: End) -> ([i64; 3], usize, f64, f64) {
    let axis = (0..3).find(|&i| a.q[i] != b.q[i]).unwrap_or(0);
    let (lo, hi) = if a.q[axis] < b.q[axis] {
        (a, b)
    } else {
        (b, a)
    };
    let t = ((SURFACE_OCCUPANCY - lo.v) / (hi.v - lo.v)).clamp(0.0, 1.0);
    (lo.q, axis, t, (hi.q[axis] - lo.q[axis]) as f64)
}

/// Normalizes an unnormalized normal to `f32`.
fn unit(g: [f64; 3]) -> [f32; 3] {
    let len = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt();
    if len > 0.0 {
        g.map(|v| (v / len) as f32)
    } else {
        [0.0; 3]
    }
}

/// Extracts `cell` with regular cells and, on every face with a finer
/// neighbor, transition cells (module docs).
pub fn extract_mixed(field: &dyn MixedField, cell: CellAt) -> MixedMesh {
    let n = field.resolution() as i64;
    let d = cell.depth;
    let h = spacing(field, d);
    let root = field.root_origin();
    let base = cell.index.map(|v| v * n);
    let faces: [bool; 6] = std::array::from_fn(|f| finer_across(field, cell, f));
    let same: [bool; 6] = std::array::from_fn(|f| same_across(field, cell, f));
    let place = Placement {
        n,
        depth: d,
        h,
        root,
        base,
        faces,
        same,
    };
    let mut b = Builder {
        field,
        mesh: MixedMesh {
            transition_faces: faces,
            ..MixedMesh::default()
        },
        index: BTreeMap::new(),
    };
    let nu = (n + 1) as usize;
    let mut values = vec![0.0; nu * nu * nu];
    for z in 0..=n {
        for y in 0..=n {
            for x in 0..=n {
                let q = [base[0] + x, base[1] + y, base[2] + z];
                values[x as usize + nu * (y as usize + nu * z as usize)] =
                    lattice_value(field, d, q);
            }
        }
    }
    let val = |p: [i64; 3]| values[p[0] as usize + nu * (p[1] as usize + nu * p[2] as usize)];
    // Regular cells.
    let mut ids = [0u32; 12];
    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                let corner = |c: usize| {
                    [
                        x + (c & 1) as i64,
                        y + ((c >> 1) & 1) as i64,
                        z + ((c >> 2) & 1) as i64,
                    ]
                };
                let mut case = 0usize;
                for c in 0..8 {
                    if val(corner(c)) > SURFACE_OCCUPANCY {
                        case |= 1 << c;
                    }
                }
                if case == 0 || case == 255 {
                    continue;
                }
                let data = &REGULAR_CELL_DATA[usize::from(REGULAR_CELL_CLASS[case])];
                for (k, id) in ids.iter_mut().enumerate().take(data.vertex_count()) {
                    let code = REGULAR_VERTEX_DATA[case][k];
                    let (ca, cb) = (usize::from((code >> 4) & 0x0F), usize::from(code & 0x0F));
                    let (pa, pb) = (corner(ca), corner(cb));
                    let ea = End {
                        q: std::array::from_fn(|i| base[i] + pa[i]),
                        v: val(pa),
                    };
                    let eb = End {
                        q: std::array::from_fn(|i| base[i] + pb[i]),
                        v: val(pb),
                    };
                    let (lo, axis, t, _) = crossing(ea, eb);
                    *id = place.vertex(&mut b, lo, axis, t);
                }
                for tri in 0..data.triangle_count() {
                    let i = &data.vertex_index[3 * tri..3 * tri + 3];
                    b.triangle([
                        ids[usize::from(i[0])],
                        ids[usize::from(i[1])],
                        ids[usize::from(i[2])],
                    ]);
                }
            }
        }
    }
    // Transition cells.
    for (f, &on) in faces.iter().enumerate() {
        if on {
            transition_face(&mut b, cell, f, &place);
        }
    }
    b.mesh
}

/// Tangent axes `(u, v)` of face `f`, chosen so that `(u, v, inward)` is
/// right handed.
fn face_axes(f: usize) -> (usize, usize) {
    let a = f / 2;
    let (u, v) = ((a + 1) % 3, (a + 2) % 3);
    // For the high face inward is -a, which flips the handedness.
    if f % 2 == 1 {
        (v, u)
    } else {
        (u, v)
    }
}

/// The transition cells of face `f` of `cell`.
fn transition_face(b: &mut Builder<'_>, cell: CellAt, f: usize, place: &Placement) {
    let field = b.field;
    let n = field.resolution() as i64;
    let a = f / 2;
    let high = f % 2 == 1;
    let (ua, va) = face_axes(f);
    let fd = cell.depth + 1;
    let hf = spacing(field, fd);
    let root = field.root_origin();
    let fine_base = cell.index.map(|v| v * 2 * n);
    let face_fine = if high { 2 * n } else { 0 };
    let mut ids = [0u32; 12];
    for sv in 0..n {
        for su in 0..n {
            // Full resolution points 0..8, row major in (u, v): fine local
            // coordinates.
            let fine_local = |i: usize| {
                let mut p = [0i64; 3];
                p[a] = face_fine;
                p[ua] = 2 * su + (i % 3) as i64;
                p[va] = 2 * sv + (i / 3) as i64;
                p
            };
            let global = |p: [i64; 3]| -> [i64; 3] { std::array::from_fn(|i| fine_base[i] + p[i]) };
            let mut vals = [0.0; 13];
            for (i, v) in vals.iter_mut().enumerate().take(9) {
                *v = lattice_value(field, fd, global(fine_local(i)));
            }
            vals[9] = vals[0];
            vals[10] = vals[2];
            vals[11] = vals[6];
            vals[12] = vals[8];
            let inside = |i: usize| u16::from(vals[i] > SURFACE_OCCUPANCY);
            let case = inside(0)
                | inside(1) << 1
                | inside(2) << 2
                | inside(5) << 3
                | inside(8) << 4
                | inside(7) << 5
                | inside(6) << 6
                | inside(3) << 7
                | inside(4) << 8;
            if case == 0 || case == 511 {
                continue;
            }
            let (class, inverted) = transition_class(case);
            let data = &TRANSITION_CELL_DATA[class];
            for (k, id) in ids.iter_mut().enumerate().take(data.vertex_count()) {
                let code = TRANSITION_VERTEX_DATA[usize::from(case)][k];
                let (ca, cb) = (usize::from((code >> 4) & 0x0F), usize::from(code & 0x0F));
                if ca < 9 {
                    // Full resolution face: shared with the finer cells.
                    let ea = End {
                        q: global(fine_local(ca)),
                        v: vals[ca],
                    };
                    let eb = End {
                        q: global(fine_local(cb)),
                        v: vals[cb],
                    };
                    let (lo, axis, t, len) = crossing(ea, eb);
                    let mut p: [f64; 3] = std::array::from_fn(|i| root[i] + lo[i] as f64 * hf);
                    p[axis] = root[axis] + (lo[axis] as f64 + t * len) * hf;
                    let mut hi = lo;
                    hi[axis] += 1;
                    let g0 = b.normal(fd, lo);
                    let g1 = b.normal(fd, hi);
                    let g = std::array::from_fn(|i| g0[i] + t * (g1[i] - g0[i]));
                    *id = b.vertex(p, unit(g));
                } else {
                    // Half resolution face: corners 9..12 copy 0, 2, 6, 8.
                    let full = |c: usize| [0, 2, 6, 8][c - 9];
                    let la = fine_local(full(ca));
                    let lb = fine_local(full(cb));
                    let (va, vb) = (vals[ca], vals[cb]);
                    let axis = (0..3).find(|&i| la[i] != lb[i]).unwrap_or(0);
                    let (lo, vlo, vhi) = if la[axis] < lb[axis] {
                        (la, va, vb)
                    } else {
                        (lb, vb, va)
                    };
                    let t = ((SURFACE_OCCUPANCY - vlo) / (vhi - vlo)).clamp(0.0, 1.0);
                    let lo_g: [i64; 3] = std::array::from_fn(|i| cell.index[i] * n + lo[i] / 2);
                    *id = place.vertex(b, lo_g, axis, t);
                }
            }
            for tri in 0..data.triangle_count() {
                let i = &data.vertex_index[3 * tri..3 * tri + 3];
                let t = [
                    ids[usize::from(i[0])],
                    ids[usize::from(i[1])],
                    ids[usize::from(i[2])],
                ];
                b.triangle(if inverted { [t[0], t[2], t[1]] } else { t });
            }
        }
    }
}

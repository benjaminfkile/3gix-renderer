//! Regular cells: marching cubes over the corner lattice with the
//! Transvoxel regular cell tables (`rendering.md` sections 2.2 and 2.4).
//!
//! Every cube of the lattice is classified by which of its eight corners
//! are inside the matter (occupancy above 0.5) and triangulated with
//! [`REGULAR_CELL_CLASS`], [`REGULAR_CELL_DATA`], and
//! [`REGULAR_VERTEX_DATA`]. A vertex lies on a cube edge where the
//! occupancy, linear along the edge, is 0.5; its position is computed in
//! `f64` frame coordinates from the edge's lower corner toward its upper
//! one, so two cells that share the edge get the same bits. A corner whose
//! value is exactly 0.5 is itself the vertex of every edge that ends there.
//!
//! Vertices are shared through an index on lattice edges and corners, so the
//! mesh is indexed and holds each vertex once. Triangles that collapse onto
//! one corner are dropped. Normals are the negated occupancy gradient
//! ([`CornerLattice::gradient`]) interpolated along the edge. The other
//! attributes are those of the edge's inside corner
//! ([`CornerLattice::attributes`]), or of the corner itself for a vertex on
//! a corner.

use super::lattice::{CornerLattice, LatticeError, Neighborhood};
use super::tables::{REGULAR_CELL_CLASS, REGULAR_CELL_DATA, REGULAR_VERTEX_DATA};
use crate::extract::{SurfaceMesh, SurfaceVertex};

/// The surface level of occupancy (`matter-format.md` section 3.3).
pub const SURFACE_OCCUPANCY: f64 = 0.5;

/// Marker for "no vertex yet" in the vertex index.
const NONE: u32 = u32::MAX;

/// Extracts the regular-cell surface of `hood.center` (module docs).
pub fn extract_regular(hood: &Neighborhood) -> Result<SurfaceMesh, LatticeError> {
    let lattice = CornerLattice::build(hood)?;
    let mut mesh = SurfaceMesh {
        key: hood.center.key(),
        origin: hood.center.origin(),
        isovalue: SURFACE_OCCUPANCY,
        positions: Vec::new(),
        vertices: Vec::new(),
        indices: Vec::new(),
        bounds_min: [f64::INFINITY; 3],
        bounds_max: [f64::NEG_INFINITY; 3],
        skirt_from: 0,
    };
    march(&lattice, &mut mesh);
    mesh.skirt_from = mesh.positions.len();
    if mesh.positions.is_empty() {
        let o = lattice.origin;
        mesh.bounds_min = o;
        mesh.bounds_max = o;
    }
    Ok(mesh)
}

/// Builds the indexed mesh into `mesh`.
fn march(lattice: &CornerLattice, mesh: &mut SurfaceMesh) {
    let n = lattice.resolution;
    let n1 = n + 1;
    let corner_id = |c: [usize; 3]| c[0] + n1 * (c[1] + n1 * c[2]);
    // One slot per lattice corner, and per corner and axis for the edge
    // from that corner toward larger coordinates.
    let mut corner_vertex = vec![NONE; n1 * n1 * n1];
    let mut edge_vertex = vec![NONE; 3 * n1 * n1 * n1];
    let mut values = [0.0f64; 8];
    let mut ids = [0u32; 12];
    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                let mut case = 0usize;
                for (c, v) in values.iter_mut().enumerate() {
                    let p = cube_corner([x, y, z], c);
                    *v = lattice.value(p[0] as i64, p[1] as i64, p[2] as i64);
                    if *v > SURFACE_OCCUPANCY {
                        case |= 1 << c;
                    }
                }
                if case == 0 || case == 255 {
                    continue;
                }
                let data = &REGULAR_CELL_DATA[usize::from(REGULAR_CELL_CLASS[case])];
                for (k, id) in ids.iter_mut().enumerate().take(data.vertex_count()) {
                    let code = REGULAR_VERTEX_DATA[case][k];
                    let (a, b) = (usize::from((code >> 4) & 0x0F), usize::from(code & 0x0F));
                    // Corners differ in one bit; the lower index is the
                    // lower coordinate along the edge.
                    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
                    let axis = (hi ^ lo).trailing_zeros() as usize;
                    let p0 = cube_corner([x, y, z], lo);
                    let p1 = cube_corner([x, y, z], hi);
                    let (v0, v1) = (values[lo], values[hi]);
                    let t = (SURFACE_OCCUPANCY - v0) / (v1 - v0);
                    *id = if t <= 0.0 {
                        corner_or_new(
                            &mut corner_vertex[corner_id(p0)],
                            || corner_vertex_data(lattice, p0),
                            mesh,
                        )
                    } else if t >= 1.0 {
                        corner_or_new(
                            &mut corner_vertex[corner_id(p1)],
                            || corner_vertex_data(lattice, p1),
                            mesh,
                        )
                    } else {
                        corner_or_new(
                            &mut edge_vertex[3 * corner_id(p0) + axis],
                            || edge_vertex_data(lattice, p0, p1, axis, t, v0 > v1),
                            mesh,
                        )
                    };
                }
                for tri in 0..data.triangle_count() {
                    let i = &data.vertex_index[3 * tri..3 * tri + 3];
                    let t = [
                        ids[usize::from(i[0])],
                        ids[usize::from(i[1])],
                        ids[usize::from(i[2])],
                    ];
                    if t[0] == t[1] || t[1] == t[2] || t[0] == t[2] {
                        continue;
                    }
                    // With the case bits marking the inside, the published
                    // order is counterclockwise seen from outside.
                    mesh.indices.extend_from_slice(&t);
                }
            }
        }
    }
}

/// Lattice corner `c` (0 to 7) of the cube whose minimum corner is `p`.
fn cube_corner(p: [usize; 3], c: usize) -> [usize; 3] {
    [p[0] + (c & 1), p[1] + ((c >> 1) & 1), p[2] + ((c >> 2) & 1)]
}

/// Returns the vertex in `slot`, creating it from `make` first if needed.
fn corner_or_new(
    slot: &mut u32,
    make: impl FnOnce() -> ([f64; 3], SurfaceVertex),
    mesh: &mut SurfaceMesh,
) -> u32 {
    if *slot == NONE {
        let (p, v) = make();
        *slot = mesh.positions.len() as u32;
        for (a, v) in p.iter().enumerate() {
            mesh.bounds_min[a] = mesh.bounds_min[a].min(*v);
            mesh.bounds_max[a] = mesh.bounds_max[a].max(*v);
        }
        mesh.positions.push(p);
        mesh.vertices.push(v);
    }
    *slot
}

/// A vertex on lattice corner `c`.
fn corner_vertex_data(lattice: &CornerLattice, c: [usize; 3]) -> ([f64; 3], SurfaceVertex) {
    let g = lattice.gradient(c);
    (lattice.position(c), vertex(lattice, c, g))
}

/// A vertex at `t` along the edge from `p0` to `p1` (along `axis`);
/// `low_inside` says which end is inside.
fn edge_vertex_data(
    lattice: &CornerLattice,
    p0: [usize; 3],
    p1: [usize; 3],
    axis: usize,
    t: f64,
    low_inside: bool,
) -> ([f64; 3], SurfaceVertex) {
    let mut pos = lattice.position(p0);
    let end = lattice.position(p1)[axis];
    pos[axis] += t * (end - pos[axis]);
    let g0 = lattice.gradient(p0);
    let g1 = lattice.gradient(p1);
    let g = std::array::from_fn(|a| g0[a] + t * (g1[a] - g0[a]));
    let inside = if low_inside { p0 } else { p1 };
    (pos, vertex(lattice, inside, g))
}

/// Attributes of corner `c` with the normal from gradient `g`.
fn vertex(lattice: &CornerLattice, c: [usize; 3], g: [f64; 3]) -> SurfaceVertex {
    let a = lattice.attributes(c[0], c[1], c[2]);
    let len = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt();
    let normal = if len > 0.0 {
        g.map(|v| (-v / len) as f32)
    } else {
        [0.0; 3]
    };
    SurfaceVertex {
        normal,
        albedo: a.albedo.map(|v| v as f32),
        roughness: a.roughness as f32,
        temperature: a.temperature as f32,
        state: u32::from(a.state),
        density: a.density as f32,
    }
}

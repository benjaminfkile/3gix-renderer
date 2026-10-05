//! Surface extraction: marching cubes over a composited density grid.
//!
//! Implements the renderer's step "extracts surfaces" of `space-model.md`
//! section 2 over one composited section (`matter-format.md` sections 3.3
//! and 3.5). The renderer knows nothing about what the matter is: a dense
//! sample is inside a surface, and the surface is where the density crosses
//! half the largest density in the section.
//!
//! Only solid and fluid samples form surfaces ([`forms_surface`]). Gas and
//! plasma have no hard surface; they count as vacuum here and are drawn as a
//! volume instead ([`crate::volume`]). A cell that holds both gives a mesh
//! from its solid and fluid samples and a volume from the rest.
//!
//! # The grid
//!
//! The samples of a section of resolution `n` are values at the sample
//! centers, `origin + (i + 1/2) * e / n` per axis. Extraction runs over those
//! points padded with one layer of vacuum on every side, so the grid has
//! `n + 2` points per axis and every surface closes within half a sample
//! of the cell boundary. Matter that fills a cell to its boundary therefore
//! ends in a wall at the boundary instead of an open edge, and two filled
//! neighbors meet wall to wall: the seam between cells is never a hole.
//!
//! # The isovalue
//!
//! Half the largest density of the section's solid and fluid samples, per
//! section; gas and plasma samples take no part. A cell at a
//! coarse depth gives a coarse mesh: that is the compiler's chosen
//! complexity, and nothing here smooths it.
//!
//! # Cubes and triangles
//!
//! A corner is inside when its density is strictly above the isovalue. The
//! triangle table is not typed in: [`case_table`] derives it once from the
//! cube faces. On every face the inside corners are kept apart (an inside
//! run of corners gets its own segment), a rule that depends only on the
//! four corners of the face, so the two cubes sharing a face always cut it
//! the same way and the mesh is watertight. Segments are linked into loops
//! and each loop is fanned into triangles, wound counterclockwise as seen
//! from the outside (from lower density).
//!
//! # Attributes
//!
//! Positions are linear interpolations along cube edges, in `f64`, in the
//! frame's coordinates. Normals are the negated density gradient (central
//! differences, one-sided at the padded border), interpolated along the edge
//! and normalized, so they point from dense to thin. Albedo, roughness, and
//! temperature are density-weighted along the edge, the same rule the
//! compositing uses, so vacuum on one side never dims or cools the surface;
//! the state is that of the corner inside the surface, and the density is
//! the linear interpolation.
//!
//! # Determinism
//!
//! The grid is walked in index order (x fastest, then y, then z), vertices
//! are created in that walk and shared through an index table addressed by
//! grid edge, and every value is plain `f64` arithmetic with `sqrt`. The same
//! section always gives bit-identical vertex and index buffers. Extraction
//! may run on a worker pool ([`ExtractPool`]); each mesh depends only on its
//! own section, so which thread ran it never shows in the output.

use gx_core::key::CellKey;
use gx_core::matter::{Section, State};
use gx_core::units::Vec3;
use std::collections::VecDeque;
use std::sync::mpsc;
use std::sync::{Arc, Mutex, OnceLock};

/// Returns `true` for the states that form surfaces: solid and fluid.
/// Everything else (vacuum, gas, plasma) counts as vacuum for extraction.
pub fn forms_surface(state: State) -> bool {
    matches!(state, State::Solid | State::Fluid)
}

/// Per-vertex attributes other than the position.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SurfaceVertex {
    /// Unit normal, pointing from dense to thin matter, in the frame axes.
    pub normal: [f32; 3],
    /// Reflectance in the three bands, long to short wavelength.
    pub albedo: [f32; 3],
    /// Microfacet roughness, 0 to 1.
    pub roughness: f32,
    /// Temperature, kelvin.
    pub temperature: f32,
    /// State as its format integer (`matter-format.md` section 3.3).
    pub state: u32,
    /// Density, kilograms per cubic meter.
    pub density: f32,
}

/// The surface of one cell.
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceMesh {
    /// The cell the surface belongs to.
    pub key: CellKey,
    /// Minimum corner of the cell, meters, frame coordinates.
    pub origin: Vec3,
    /// The density the surface follows, kilograms per cubic meter: half the
    /// largest density of the solid and fluid samples.
    pub isovalue: f64,
    /// Vertex positions, meters, frame coordinates.
    pub positions: Vec<[f64; 3]>,
    /// Vertex attributes, one per position.
    pub vertices: Vec<SurfaceVertex>,
    /// Triangles, three indices each, counterclockwise from outside.
    pub indices: Vec<u32>,
    /// Smallest corner of the box around every position, frame coordinates.
    pub bounds_min: [f64; 3],
    /// Largest corner of the box around every position, frame coordinates.
    pub bounds_max: [f64; 3],
}

impl SurfaceMesh {
    /// Number of triangles.
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// Returns `true` when there is nothing to draw.
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }
}

/// Corner `c` of a cube is offset `(c & 1, (c >> 1) & 1, (c >> 2) & 1)`.
fn corner_offset(c: usize) -> [usize; 3] {
    [c & 1, (c >> 1) & 1, (c >> 2) & 1]
}

/// The twelve cube edges as `(low corner, axis)`: the edge runs from the
/// low corner to the corner with the `axis` bit set.
fn cube_edges() -> [(usize, usize); 12] {
    let mut out = [(0, 0); 12];
    let mut i = 0;
    for axis in 0..3 {
        for c in 0..8 {
            if c & (1 << axis) == 0 {
                out[i] = (c, axis);
                i += 1;
            }
        }
    }
    out
}

/// Index of the cube edge between corners `a` and `b` (adjacent).
fn edge_between(edges: &[(usize, usize); 12], a: usize, b: usize) -> usize {
    let low = a.min(b);
    let axis = (a ^ b).trailing_zeros() as usize;
    edges
        .iter()
        .position(|&e| e == (low, axis))
        .expect("adjacent corners share an edge")
}

/// Triangles for every corner configuration: entry `config` lists triangles
/// as triples of cube edge indices. Bit `c` of `config` is set when corner
/// `c`, at offset `(c & 1, (c >> 1) & 1, (c >> 2) & 1)`, is inside. Edges
/// are numbered by axis (x, then y, then z) and, within an axis, by the
/// index of their lower corner.
pub fn case_table() -> &'static [Vec<[u8; 3]>] {
    static TABLE: OnceLock<Vec<Vec<[u8; 3]>>> = OnceLock::new();
    TABLE.get_or_init(build_case_table)
}

fn build_case_table() -> Vec<Vec<[u8; 3]>> {
    let edges = cube_edges();
    (0..256usize)
        .map(|config| {
            let inside = |c: usize| config & (1 << c) != 0;
            // next[edge] = the edge the surface loop continues to.
            let mut next = [usize::MAX; 12];
            for axis in 0..3 {
                let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);
                for side in 0..2 {
                    // Corners counterclockwise seen from outside the face.
                    let mut ring: Vec<usize> = [(0, 0), (1, 0), (1, 1), (0, 1)]
                        .iter()
                        .map(|&(a, b)| (side << axis) | (a << u) | (b << v))
                        .collect();
                    if side == 0 {
                        ring.reverse();
                    }
                    // Crossings in ring order: (edge, entering).
                    let mut crossings = Vec::new();
                    for k in 0..4 {
                        let (a, b) = (ring[k], ring[(k + 1) % 4]);
                        if inside(a) != inside(b) {
                            crossings.push((edge_between(&edges, a, b), inside(b)));
                        }
                    }
                    // Each inside run is one segment: from the crossing that
                    // enters it to the next one, which leaves it.
                    for i in 0..crossings.len() {
                        let (from, entering) = crossings[i];
                        if entering {
                            let (to, _) = crossings[(i + 1) % crossings.len()];
                            next[from] = to;
                        }
                    }
                }
            }
            let mut seen = [false; 12];
            let mut triangles = Vec::new();
            for start in 0..12 {
                if next[start] == usize::MAX || seen[start] {
                    continue;
                }
                let mut ring = Vec::new();
                let mut e = start;
                while !seen[e] {
                    seen[e] = true;
                    ring.push(e as u8);
                    e = next[e];
                }
                for i in 1..ring.len() - 1 {
                    triangles.push([ring[0], ring[i], ring[i + 1]]);
                }
            }
            triangles
        })
        .collect()
}

/// The section's samples on the padded grid.
struct Grid<'a> {
    section: &'a Section,
    /// Points per axis: resolution plus two.
    m: usize,
    /// Density at every grid point, padded with zeros.
    density: Vec<f64>,
    /// Sample spacing, meters.
    step: f64,
    origin: Vec3,
}

impl Grid<'_> {
    fn index(&self, p: [usize; 3]) -> usize {
        p[0] + self.m * (p[1] + self.m * p[2])
    }

    fn d(&self, p: [usize; 3]) -> f64 {
        self.density[self.index(p)]
    }

    /// Frame coordinates of grid point `p`: sample `p - 1`'s center.
    fn position(&self, p: [usize; 3]) -> [f64; 3] {
        let o = [self.origin.x, self.origin.y, self.origin.z];
        core::array::from_fn(|a| o[a] + (p[a] as f64 - 0.5) * self.step)
    }

    /// Density gradient at grid point `p`, kilograms per cubic meter per
    /// meter.
    fn gradient(&self, p: [usize; 3]) -> [f64; 3] {
        core::array::from_fn(|a| {
            let mut lo = p;
            let mut hi = p;
            if p[a] > 0 {
                lo[a] -= 1;
            }
            if p[a] + 1 < self.m {
                hi[a] += 1;
            }
            let span = (hi[a] - lo[a]) as f64 * self.step;
            (self.d(hi) - self.d(lo)) / span
        })
    }

    /// The sample at grid point `p`, or `None` in the vacuum padding. Its
    /// weight in the attributes is the grid density, which is 0 for samples
    /// that do not form surfaces.
    fn sample(&self, p: [usize; 3]) -> Option<gx_core::matter::Sample> {
        let n = self.m - 2;
        if p.iter().any(|&c| c == 0 || c > n) {
            return None;
        }
        let i = (p[0] - 1) + n * ((p[1] - 1) + n * (p[2] - 1));
        self.section.samples().and_then(|s| s.get(i))
    }
}

/// Extracts the surface of the solid and fluid samples of a composited
/// section. An empty section, or one with no solid or fluid sample, gives an
/// empty mesh.
pub fn extract(section: &Section) -> SurfaceMesh {
    let key = section.key();
    let origin = section.origin();
    let empty = |isovalue| SurfaceMesh {
        key,
        origin,
        isovalue,
        positions: Vec::new(),
        vertices: Vec::new(),
        indices: Vec::new(),
        bounds_min: [origin.x, origin.y, origin.z],
        bounds_max: [origin.x, origin.y, origin.z],
    };
    let Some(samples) = section.samples() else {
        return empty(0.0);
    };
    let n = usize::from(section.resolution());
    let m = n + 2;
    let mut density = vec![0.0f64; m * m * m];
    let mut max = 0.0f64;
    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                let i = x + n * (y + n * z);
                if !forms_surface(samples.state(i)) {
                    continue;
                }
                let d = samples.density(i).value();
                density[(x + 1) + m * ((y + 1) + m * (z + 1))] = d;
                if d > max {
                    max = d;
                }
            }
        }
    }
    let isovalue = max / 2.0;
    if max <= 0.0 {
        return empty(isovalue);
    }
    let grid = Grid {
        section,
        m,
        density,
        step: section.edge().value() / n as f64,
        origin,
    };

    let table = case_table();
    let edges = cube_edges();
    // Vertex index by grid edge: point index * 3 + axis.
    let mut by_edge = vec![u32::MAX; m * m * m * 3];
    let mut positions: Vec<[f64; 3]> = Vec::new();
    let mut vertices: Vec<SurfaceVertex> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    for z in 0..m - 1 {
        for y in 0..m - 1 {
            for x in 0..m - 1 {
                let base = [x, y, z];
                let at = |c: usize| {
                    let o = corner_offset(c);
                    [base[0] + o[0], base[1] + o[1], base[2] + o[2]]
                };
                let mut config = 0usize;
                for c in 0..8 {
                    if grid.d(at(c)) > isovalue {
                        config |= 1 << c;
                    }
                }
                let tris = &table[config];
                if tris.is_empty() {
                    continue;
                }
                for tri in tris {
                    for &e in tri {
                        let (c, axis) = edges[usize::from(e)];
                        let a = at(c);
                        let slot = grid.index(a) * 3 + axis;
                        if by_edge[slot] == u32::MAX {
                            let mut b = a;
                            b[axis] += 1;
                            by_edge[slot] = positions.len() as u32;
                            let (p, v) = edge_vertex(&grid, a, b, isovalue);
                            positions.push(p);
                            vertices.push(v);
                        }
                        indices.push(by_edge[slot]);
                    }
                }
            }
        }
    }

    let mut bounds_min = [f64::INFINITY; 3];
    let mut bounds_max = [f64::NEG_INFINITY; 3];
    for p in &positions {
        for a in 0..3 {
            bounds_min[a] = bounds_min[a].min(p[a]);
            bounds_max[a] = bounds_max[a].max(p[a]);
        }
    }
    if positions.is_empty() {
        return empty(isovalue);
    }
    SurfaceMesh {
        key,
        origin,
        isovalue,
        positions,
        vertices,
        indices,
        bounds_min,
        bounds_max,
    }
}

/// The vertex where the surface crosses the grid edge from `a` to `b`.
fn edge_vertex(
    grid: &Grid<'_>,
    a: [usize; 3],
    b: [usize; 3],
    iso: f64,
) -> ([f64; 3], SurfaceVertex) {
    let (da, db) = (grid.d(a), grid.d(b));
    // Exactly one end is above the isovalue, so `da != db`.
    let t = (iso - da) / (db - da);
    let (pa, pb) = (grid.position(a), grid.position(b));
    let position = core::array::from_fn(|k| pa[k] + t * (pb[k] - pa[k]));

    let (ga, gb) = (grid.gradient(a), grid.gradient(b));
    let g: [f64; 3] = core::array::from_fn(|k| ga[k] + t * (gb[k] - ga[k]));
    let len = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt();
    let normal = if len > 0.0 {
        [
            (-g[0] / len) as f32,
            (-g[1] / len) as f32,
            (-g[2] / len) as f32,
        ]
    } else {
        // A flat gradient: point from the inside end to the outside end.
        let axis = (0..3).find(|&k| a[k] != b[k]).unwrap_or(0);
        let mut n = [0.0f32; 3];
        n[axis] = if da > db { 1.0 } else { -1.0 };
        n
    };

    // Density-weighted attributes; vacuum, gas, plasma, and the padding
    // weigh nothing.
    let (sa, sb) = (grid.sample(a), grid.sample(b));
    let wa = (1.0 - t) * da;
    let wb = t * db;
    let total = wa + wb;
    let mut albedo = [0.0f64; 3];
    let mut roughness = 0.0;
    let mut temperature = 0.0;
    for (s, w) in [(sa, wa), (sb, wb)] {
        if let (Some(s), true) = (s, w > 0.0) {
            for (acc, al) in albedo.iter_mut().zip(s.albedo) {
                *acc += w * al.value();
            }
            roughness += w * s.roughness.value();
            temperature += w * s.temperature.value();
        }
    }
    let inside = if da > db { sa } else { sb };
    let vertex = SurfaceVertex {
        normal,
        albedo: albedo.map(|v| (v / total) as f32),
        roughness: (roughness / total) as f32,
        temperature: (temperature / total) as f32,
        state: u32::from(inside.map_or(State::Vacuum, |s| s.state).as_u8()),
        density: (da + t * (db - da)) as f32,
    };
    (position, vertex)
}

/// A job for the pool: one cell's composited section.
type Job = (CellKey, Arc<Section>);

/// Runs [`extract`] on worker threads.
///
/// Jobs go in over a channel and finished meshes come back over another, in
/// whatever order the workers finish; each mesh carries its key, and its
/// contents depend only on its section. With zero threads (the browser
/// build, where `std::thread` is unavailable) extraction runs inline on
/// [`ExtractPool::submit`].
pub struct ExtractPool {
    jobs: Option<mpsc::Sender<Job>>,
    results: mpsc::Receiver<(CellKey, Arc<SurfaceMesh>)>,
    inline: VecDeque<(CellKey, Arc<SurfaceMesh>)>,
    workers: Vec<std::thread::JoinHandle<()>>,
}

/// The default worker count: the available parallelism, at most four, and
/// zero on the web where there are no threads.
pub fn default_threads() -> usize {
    if cfg!(target_arch = "wasm32") {
        0
    } else {
        std::thread::available_parallelism()
            .map_or(1, |n| n.get())
            .clamp(1, 4)
    }
}

impl ExtractPool {
    /// Starts `threads` workers (zero for inline extraction).
    pub fn new(threads: usize) -> ExtractPool {
        let (result_tx, results) = mpsc::channel();
        if threads == 0 {
            return ExtractPool {
                jobs: None,
                results,
                inline: VecDeque::new(),
                workers: Vec::new(),
            };
        }
        let (jobs, job_rx) = mpsc::channel::<Job>();
        let job_rx = Arc::new(Mutex::new(job_rx));
        let workers = (0..threads)
            .map(|i| {
                let rx = job_rx.clone();
                let tx = result_tx.clone();
                std::thread::Builder::new()
                    .name(format!("gx-extract-{i}"))
                    .spawn(move || loop {
                        let job = rx.lock().expect("job queue lock").recv();
                        let Ok((key, section)) = job else {
                            break;
                        };
                        if tx.send((key, Arc::new(extract(&section)))).is_err() {
                            break;
                        }
                    })
                    .expect("spawning an extraction worker")
            })
            .collect();
        ExtractPool {
            jobs: Some(jobs),
            results,
            inline: VecDeque::new(),
            workers,
        }
    }

    /// Queues one section for extraction.
    pub fn submit(&mut self, key: CellKey, section: Arc<Section>) {
        match &self.jobs {
            Some(tx) => {
                tx.send((key, section))
                    .expect("extraction workers are running");
            }
            None => self.inline.push_back((key, Arc::new(extract(&section)))),
        }
    }

    /// The meshes finished so far, without waiting.
    pub fn try_results(&mut self) -> Vec<(CellKey, Arc<SurfaceMesh>)> {
        let mut out: Vec<_> = self.inline.drain(..).collect();
        out.extend(self.results.try_iter());
        out
    }

    /// Waits for the next finished mesh. `None` if nothing can arrive.
    pub fn wait_result(&mut self) -> Option<(CellKey, Arc<SurfaceMesh>)> {
        if let Some(r) = self.inline.pop_front() {
            return Some(r);
        }
        self.jobs.as_ref()?;
        self.results.recv().ok()
    }
}

impl Drop for ExtractPool {
    fn drop(&mut self) {
        self.jobs = None;
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gx_core::matter::{Sample, Samples};
    use gx_core::units::{Attenuation, Density, Kelvin, Meters, Ratio};
    use std::collections::BTreeMap;

    const EDGE: f64 = 8.0;

    fn solid(density: f64) -> Sample {
        Sample {
            density: Density::new(density),
            state: State::Solid,
            temperature: Kelvin::new(300.0),
            albedo: [Ratio::new(0.6), Ratio::new(0.5), Ratio::new(0.4)],
            roughness: Ratio::new(0.7),
            attenuation: Attenuation::new(0.0),
        }
    }

    /// A radially symmetric blob of radius `radius` meters centered in a
    /// cell of edge [`EDGE`] at resolution `res`: a sample is matter when
    /// its center lies within the radius, with a density ramp so the
    /// gradient is well defined.
    fn blob(res: u8, radius: f64) -> Section {
        let step = EDGE / f64::from(res);
        let samples = Samples::from_fn(res, |x, y, z| {
            let c = |i: u32| (f64::from(i) + 0.5) * step - EDGE / 2.0;
            let r = (c(x) * c(x) + c(y) * c(y) + c(z) * c(z)).sqrt();
            if r < radius {
                solid(5000.0 * (1.0 + (radius - r) / radius))
            } else {
                Sample::VACUUM
            }
        });
        Section::new(
            CellKey::new(2, 0, 0, 0, 0).unwrap(),
            Vec3::new(-EDGE / 2.0, -EDGE / 2.0, -EDGE / 2.0),
            Meters::new(EDGE),
            res,
            samples,
        )
        .unwrap()
    }

    /// Every undirected edge is shared by exactly two triangles, and they
    /// traverse it in opposite directions.
    fn assert_closed(mesh: &SurfaceMesh) {
        let mut uses: BTreeMap<(u32, u32), i32> = BTreeMap::new();
        for t in mesh.indices.chunks(3) {
            for k in 0..3 {
                let (a, b) = (t[k], t[(k + 1) % 3]);
                *uses.entry((a.min(b), a.max(b))).or_default() += if a < b { 1 } else { -1 };
            }
        }
        let mut count: BTreeMap<(u32, u32), i32> = BTreeMap::new();
        for t in mesh.indices.chunks(3) {
            for k in 0..3 {
                let (a, b) = (t[k], t[(k + 1) % 3]);
                *count.entry((a.min(b), a.max(b))).or_default() += 1;
            }
        }
        for (e, n) in &count {
            assert_eq!(*n, 2, "edge {e:?} used {n} times");
            assert_eq!(uses[e], 0, "edge {e:?} traversed one way twice");
        }
    }

    #[test]
    fn table_cases() {
        let t = case_table();
        assert_eq!(t.len(), 256);
        assert!(t[0].is_empty() && t[255].is_empty());
        assert_eq!(t[1].len(), 1);
        // Complementary single corner: still one triangle.
        assert_eq!(t[254].len(), 1);
        // Two adjacent corners: a quad.
        assert_eq!(t[3].len(), 2);
        // Every triangle uses three distinct edges.
        assert!(t
            .iter()
            .flatten()
            .all(|tri| tri[0] != tri[1] && tri[1] != tri[2] && tri[0] != tri[2]));
    }

    #[test]
    fn blob_is_closed_and_normals_point_outward() {
        let section = blob(24, 2.5);
        let mesh = extract(&section);
        assert!(mesh.triangle_count() > 500, "{}", mesh.triangle_count());
        assert_eq!(mesh.positions.len(), mesh.vertices.len());
        assert_closed(&mesh);

        let mut dot_sum = 0.0;
        let mut wind_sum = 0.0;
        for (p, v) in mesh.positions.iter().zip(&mesh.vertices) {
            let r = Vec3::new(p[0], p[1], p[2]).normalized().unwrap();
            let n = Vec3::new(
                f64::from(v.normal[0]),
                f64::from(v.normal[1]),
                f64::from(v.normal[2]),
            );
            dot_sum += n.dot(r);
            assert_eq!(v.state, 1);
            assert!((v.temperature - 300.0).abs() < 1e-3);
            assert!((v.albedo[0] - 0.6).abs() < 1e-6);
            assert!((v.roughness - 0.7).abs() < 1e-6);
        }
        let mean = dot_sum / mesh.positions.len() as f64;
        assert!(mean > 0.8, "mean normal dot radial {mean}");

        // The winding agrees with the normals: counterclockwise from outside.
        for t in mesh.indices.chunks(3) {
            let p = |i: u32| {
                let q = mesh.positions[i as usize];
                Vec3::new(q[0], q[1], q[2])
            };
            let (a, b, c) = (p(t[0]), p(t[1]), p(t[2]));
            let face = (b - a).cross(c - a);
            let centroid = (a + b + c).scale(1.0 / 3.0);
            wind_sum += face.dot(centroid).signum();
        }
        assert!(
            wind_sum > 0.95 * mesh.triangle_count() as f64,
            "winding agrees on {wind_sum} of {}",
            mesh.triangle_count()
        );

        // Every vertex lies near the radius (within a sample).
        let step = EDGE / 24.0;
        for p in &mesh.positions {
            let r = Vec3::new(p[0], p[1], p[2]).length();
            assert!((r - 2.5).abs() < step * 1.5, "vertex at radius {r}");
        }
    }

    #[test]
    fn vertex_count_is_stable_and_buffers_identical() {
        let section = blob(20, 3.0);
        let a = extract(&section);
        let b = extract(&section);
        assert_eq!(a.positions.len(), b.positions.len());
        assert_eq!(a.indices, b.indices);
        let bits = |m: &SurfaceMesh| -> Vec<u64> {
            m.positions
                .iter()
                .flat_map(|p| p.map(f64::to_bits))
                .chain(m.vertices.iter().flat_map(|v| {
                    v.normal
                        .iter()
                        .chain(&v.albedo)
                        .chain([&v.roughness, &v.temperature, &v.density])
                        .map(|f| u64::from(f.to_bits()))
                        .chain([u64::from(v.state)])
                        .collect::<Vec<_>>()
                }))
                .collect()
        };
        assert_eq!(bits(&a), bits(&b));

        // The pool gives the same mesh as a direct call.
        let mut pool = ExtractPool::new(3);
        let key = section.key();
        let shared = Arc::new(section);
        for _ in 0..4 {
            pool.submit(key, shared.clone());
        }
        for _ in 0..4 {
            let (k, mesh) = pool.wait_result().unwrap();
            assert_eq!(k, key);
            assert_eq!(*mesh, a);
        }
    }

    #[test]
    fn filled_cell_closes_at_the_boundary() {
        let section = Section::new(
            CellKey::new(2, 0, 0, 0, 0).unwrap(),
            Vec3::new(0.0, 0.0, 0.0),
            Meters::new(4.0),
            4,
            Samples::filled(4, solid(1000.0)),
        )
        .unwrap();
        let mesh = extract(&section);
        assert_closed(&mesh);
        // Uniform density meets vacuum halfway between the last center and
        // the padding: exactly at the cell boundary.
        for a in 0..3 {
            assert_eq!(mesh.bounds_min[a], 0.0);
            assert_eq!(mesh.bounds_max[a], 4.0);
        }
    }

    #[test]
    fn gas_and_plasma_take_no_part() {
        // A solid core inside a much denser hot gas shell: the surface
        // follows the solid alone, at half the solid density.
        let res = 16;
        let step = EDGE / f64::from(res);
        let samples = Samples::from_fn(res, |x, y, z| {
            let c = |i: u32| (f64::from(i) + 0.5) * step - EDGE / 2.0;
            let r = (c(x) * c(x) + c(y) * c(y) + c(z) * c(z)).sqrt();
            if r < 2.0 {
                solid(1000.0)
            } else if r < 3.5 {
                Sample {
                    density: Density::new(5000.0),
                    state: State::Gas,
                    temperature: Kelvin::new(4000.0),
                    albedo: [Ratio::new(0.1); 3],
                    roughness: Ratio::new(0.0),
                    attenuation: Attenuation::new(1.0),
                }
            } else {
                Sample::VACUUM
            }
        });
        let section = Section::new(
            CellKey::new(2, 0, 0, 0, 0).unwrap(),
            Vec3::new(-EDGE / 2.0, -EDGE / 2.0, -EDGE / 2.0),
            Meters::new(EDGE),
            res,
            samples,
        )
        .unwrap();
        let mesh = extract(&section);
        assert_eq!(mesh.isovalue, 500.0);
        assert!(!mesh.is_empty());
        assert_closed(&mesh);
        for (p, v) in mesh.positions.iter().zip(&mesh.vertices) {
            let r = Vec3::new(p[0], p[1], p[2]).length();
            assert!(r < 2.0 + step, "vertex at radius {r}");
            assert_eq!(v.state, 1);
            assert!((v.temperature - 300.0).abs() < 1e-3);
        }

        // Gas alone gives no surface.
        let gas_only = Section::new(
            CellKey::new(2, 0, 0, 0, 0).unwrap(),
            Vec3::zero(),
            Meters::new(EDGE),
            2,
            Samples::filled(
                2,
                Sample {
                    density: Density::new(1.0),
                    state: State::Plasma,
                    temperature: Kelvin::new(6000.0),
                    albedo: [Ratio::new(0.0); 3],
                    roughness: Ratio::new(0.0),
                    attenuation: Attenuation::new(1.0),
                },
            ),
        )
        .unwrap();
        assert!(extract(&gas_only).is_empty());
    }

    #[test]
    fn empty_and_inline() {
        let key = CellKey::new(2, 0, 0, 0, 0).unwrap();
        let s = Section::empty(key, Vec3::zero(), Meters::new(1.0)).unwrap();
        assert!(extract(&s).is_empty());
        let mut pool = ExtractPool::new(0);
        pool.submit(key, Arc::new(blob(8, 2.0)));
        let r = pool.try_results();
        assert_eq!(r.len(), 1);
        assert!(!r[0].1.is_empty());
    }
}

//! The queries of the renderer API: `docs/api.md` section 6
//! (`surfaceHeight`, `raycast`, `project`).
//!
//! A [`Picture`] holds what one tick drew: the surface tiles with their
//! morph distances, the cell meshes, and where each of their frames is
//! relative to the camera. The queries run on the CPU against it, in `f64`
//! frame coordinates, with no GPU readback and no fetch. A tile vertex is
//! placed as `vs_tile` in `shaders/surface.wgsl` places it: moved from its
//! position toward its parent level's by the camera distance through the
//! tile's morph range, so every answer matches the picture.

use crate::camera::Camera;
use crate::extract::SurfaceMesh;
use crate::tiles::TileDrawInfo;
use crate::world::DrawList;
use gx_core::frames::FrameSystem;
use gx_core::surface::TILE_QUADS;
use gx_core::units::Vec3;
use std::collections::BTreeMap;
use std::sync::Arc;

/// Relative tolerance on distance below which a point counts as on the
/// nearest drawn thing rather than behind it ([`Picture::project`]).
pub const OCCLUSION_TOLERANCE: f64 = 1e-5;

/// What was drawn where: a `raycast` hit (`docs/api.md` section 6).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum HitKind {
    /// A surface tile.
    Surface,
    /// A cell mesh.
    Cells,
}

/// A `raycast` result.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Hit {
    /// The frame of the tile or mesh hit.
    pub frame_id: u64,
    /// The point hit, frame axes, meters from the frame origin.
    pub position: [f64; 3],
    /// Meters from the camera.
    pub distance: f64,
    /// A surface tile or a cell mesh.
    pub kind: HitKind,
}

/// A `surfaceHeight` result.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SurfaceHeight {
    /// Height of the drawn surface above the reference ellipsoid, meters,
    /// along the ellipsoid's normal.
    pub height: f64,
    /// The level of the tile the height came from.
    pub level: u8,
    /// A finer tile there is still on its way.
    pub finer: bool,
}

/// A `project` result.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Projected {
    /// Pixels from the left edge.
    pub x: f64,
    /// Pixels from the top edge.
    pub y: f64,
    /// Meters in front of the camera along its view axis.
    pub depth: f64,
}

/// Where a frame is for the camera at the tick drawn.
#[derive(Copy, Clone, Debug, PartialEq)]
struct Placement {
    /// The frame origin relative to the camera, root axes, meters.
    origin: [f64; 3],
    /// Columns of the rotation from frame axes into root axes.
    rotation: [[f64; 3]; 3],
}

impl Placement {
    /// A point in frame axes, relative to the camera in root axes.
    fn to_root(self, p: [f64; 3]) -> [f64; 3] {
        let r = &self.rotation;
        std::array::from_fn(|a| self.origin[a] + r[0][a] * p[0] + r[1][a] * p[1] + r[2][a] * p[2])
    }

    /// A direction in root axes, into frame axes.
    fn dir_to_frame(&self, d: [f64; 3]) -> [f64; 3] {
        std::array::from_fn(|c| dot(self.rotation[c], d))
    }

    /// A camera-relative point in root axes, into frame coordinates.
    fn point_to_frame(&self, p: [f64; 3]) -> [f64; 3] {
        self.dir_to_frame(sub(p, self.origin))
    }
}

/// What one tick drew, for the queries.
#[derive(Clone, Debug, Default)]
pub struct Picture {
    tiles: Vec<TileDrawInfo>,
    meshes: Vec<Arc<SurfaceMesh>>,
    frames: BTreeMap<u64, Placement>,
    /// Rows of the rotation from root axes into camera axes.
    view: [[f64; 3]; 3],
    fov_y: f64,
    size: (u32, u32),
    /// Tiles whose finer tiles the selection wants and has not drawn.
    coarse: Vec<gx_core::key::SurfaceKey>,
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn length(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

fn v3(v: Vec3) -> [f64; 3] {
    [v.x, v.y, v.z]
}

/// Möller and Trumbore: the ray parameter where `o + t d` meets the
/// triangle, both sides.
fn ray_triangle(o: [f64; 3], d: [f64; 3], a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> Option<f64> {
    let e1 = sub(b, a);
    let e2 = sub(c, a);
    let p = cross(d, e2);
    let det = dot(e1, p);
    if det == 0.0 || !det.is_finite() {
        return None;
    }
    let inv = 1.0 / det;
    let s = sub(o, a);
    let u = dot(s, p) * inv;
    // A small allowance so a ray through a shared edge or vertex hits.
    const EDGE: f64 = 1e-9;
    if !(-EDGE..=1.0 + EDGE).contains(&u) {
        return None;
    }
    let q = cross(s, e1);
    let v = dot(d, q) * inv;
    if v < -EDGE || u + v > 1.0 + EDGE {
        return None;
    }
    Some(dot(e2, q) * inv)
}

/// The ray parameters where `o + t d` is inside the box, if any.
fn ray_box(o: [f64; 3], d: [f64; 3], lo: [f64; 3], hi: [f64; 3]) -> Option<(f64, f64)> {
    let (mut t0, mut t1) = (f64::NEG_INFINITY, f64::INFINITY);
    for a in 0..3 {
        if d[a] == 0.0 {
            if o[a] < lo[a] || o[a] > hi[a] {
                return None;
            }
            continue;
        }
        let (x, y) = ((lo[a] - o[a]) / d[a], (hi[a] - o[a]) / d[a]);
        t0 = t0.max(x.min(y));
        t1 = t1.min(x.max(y));
    }
    // A little slack: the box is exact, the test is not.
    let slack = 1e-9 * (t0.abs().max(t1.abs()) + 1.0);
    (t0 <= t1 + slack).then_some((t0 - slack, t1 + slack))
}

/// The drawn position of tile vertex `n`: `vs_tile`'s morph by the camera
/// distance `|p - camera|` through the tile's morph range.
pub fn morphed_vertex(tile: &TileDrawInfo, n: usize, camera: [f64; 3]) -> [f64; 3] {
    let g = &tile.geometry;
    let (p, q) = (g.positions[n], g.targets[n]);
    let [m0, m1] = tile.morph.map(f64::from);
    if !m0.is_finite() {
        return p;
    }
    let d = length(sub(p, camera));
    let k = ((d - m0) / (m1 - m0).max(1e-6)).clamp(0.0, 1.0);
    std::array::from_fn(|a| p[a] + (q[a] - p[a]) * k)
}

/// Every ray parameter where `o + t d` (frame coordinates) meets the drawn
/// tile, with `camera` the camera in the same coordinates for the morph.
fn tile_hits(tile: &TileDrawInfo, o: [f64; 3], d: [f64; 3], camera: [f64; 3], out: &mut Vec<f64>) {
    let g = &tile.geometry;
    if ray_box(o, d, g.bounds.0, g.bounds.1).is_none() {
        return;
    }
    let side = TILE_QUADS + 1;
    // Blocks of 8 x 8 quads, each tested against the box of its positions
    // and targets first.
    const BLOCK: usize = 8;
    for bj in (0..TILE_QUADS).step_by(BLOCK) {
        for bi in (0..TILE_QUADS).step_by(BLOCK) {
            let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
            for j in bj..=bj + BLOCK {
                for i in bi..=bi + BLOCK {
                    let n = i + side * j;
                    for v in [g.positions[n], g.targets[n]] {
                        for a in 0..3 {
                            lo[a] = lo[a].min(v[a]);
                            hi[a] = hi[a].max(v[a]);
                        }
                    }
                }
            }
            if ray_box(o, d, lo, hi).is_none() {
                continue;
            }
            for j in bj..bj + BLOCK {
                for i in bi..bi + BLOCK {
                    let at = |i: usize, j: usize| morphed_vertex(tile, i + side * j, camera);
                    // Split from (i, j) to (i + 1, j + 1), as the index list.
                    let (a, b, c, e) = (at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1));
                    out.extend(ray_triangle(o, d, a, b, c));
                    out.extend(ray_triangle(o, d, a, c, e));
                }
            }
        }
    }
}

/// Every ray parameter where `o + t d` (frame coordinates) meets the mesh.
fn mesh_hits(mesh: &SurfaceMesh, o: [f64; 3], d: [f64; 3], out: &mut Vec<f64>) {
    if ray_box(o, d, mesh.bounds_min, mesh.bounds_max).is_none() {
        return;
    }
    for t in mesh.indices.as_chunks::<3>().0 {
        let p = |k: usize| mesh.positions[t[k] as usize];
        out.extend(ray_triangle(o, d, p(0), p(1), p(2)));
    }
}

/// The height of `p` (frame coordinates) above the ellipsoid of radii `a`
/// (equatorial) and `c` (polar), along the ellipsoid's normal.
pub fn ellipsoid_height(p: [f64; 3], a: f64, c: f64) -> f64 {
    if a == c {
        return length(p) - a;
    }
    let rho = (p[0] * p[0] + p[1] * p[1]).sqrt();
    let z = p[2];
    let e2 = 1.0 - (c * c) / (a * a);
    // The angle of the normal from the equatorial plane by fixed point
    // iteration, then the height in the form that holds at the poles too.
    let mut phi = z.atan2(rho * (1.0 - e2));
    let mut h = 0.0;
    for _ in 0..16 {
        let s = phi.sin();
        let n = a / (1.0 - e2 * s * s).sqrt();
        h = rho * phi.cos() + z * s - a * (1.0 - e2 * s * s).sqrt();
        let next = z.atan2(rho * (1.0 - e2 * n / (n + h)));
        if next == phi {
            break;
        }
        phi = next;
    }
    h
}

impl Picture {
    /// The picture of a tick: the tiles and meshes of `draw` as `camera`
    /// saw them in `system`, in a view of `size` pixels.
    pub fn new(
        system: &FrameSystem,
        camera: &Camera,
        draw: &DrawList,
        size: (u32, u32),
    ) -> Picture {
        let mut frames = BTreeMap::new();
        let ids = draw
            .tiles
            .iter()
            .map(|t| t.geometry.key.frame_id)
            .chain(draw.meshes.iter().map(|m| m.key.frame_id));
        for id in ids {
            frames.entry(id).or_insert_with(|| {
                let q = system.root_orientation(id);
                Placement {
                    origin: v3(camera.relative(system, Vec3::zero(), id)),
                    rotation: [
                        v3(q.rotate(Vec3::new(1.0, 0.0, 0.0))),
                        v3(q.rotate(Vec3::new(0.0, 1.0, 0.0))),
                        v3(q.rotate(Vec3::new(0.0, 0.0, 1.0))),
                    ],
                }
            });
        }
        let to_camera = camera.root_orientation(system).conjugate();
        let row = |v: Vec3| v3(to_camera.rotate(v));
        // Columns of the root-to-camera rotation, transposed below to rows.
        let cols = [
            row(Vec3::new(1.0, 0.0, 0.0)),
            row(Vec3::new(0.0, 1.0, 0.0)),
            row(Vec3::new(0.0, 0.0, 1.0)),
        ];
        let view = std::array::from_fn(|r| [cols[0][r], cols[1][r], cols[2][r]]);
        Picture {
            tiles: draw.tiles.clone(),
            meshes: draw.meshes.clone(),
            frames,
            view,
            fov_y: camera.fov_y,
            size: (size.0.max(1), size.1.max(1)),
            coarse: Vec::new(),
        }
    }

    /// The picture with `coarse` marked as tiles a finer tile is on its
    /// way for.
    pub fn with_coarse(mut self, coarse: Vec<gx_core::key::SurfaceKey>) -> Picture {
        self.coarse = coarse;
        self
    }

    /// The view size, pixels.
    pub fn size(&self) -> (u32, u32) {
        self.size
    }

    /// The camera-relative ray direction, root axes, through pixel
    /// `(x, y)` (top left origin).
    pub fn pixel_ray(&self, x: f64, y: f64) -> [f64; 3] {
        let (w, h) = (f64::from(self.size.0), f64::from(self.size.1));
        let f = (0.5 * self.fov_y).tan();
        let cam = [
            (2.0 * x / w - 1.0) * f * w / h,
            (1.0 - 2.0 * y / h) * f,
            -1.0,
        ];
        // Camera axes into root axes: the transpose of `view`.
        let v = &self.view;
        let d: [f64; 3] =
            std::array::from_fn(|a| v[0][a] * cam[0] + v[1][a] * cam[1] + v[2][a] * cam[2]);
        let l = length(d);
        d.map(|c| c / l)
    }

    /// The nearest drawn thing along the camera-relative ray `d` (unit,
    /// root axes).
    pub fn cast(&self, d: [f64; 3]) -> Option<Hit> {
        let mut best: Option<Hit> = None;
        let mut hits = Vec::new();
        let mut consider = |frame: u64, kind: HitKind, place: &Placement, hits: &mut Vec<f64>| {
            for &t in hits.iter() {
                if t > 0.0 && best.is_none_or(|b| t < b.distance) {
                    let o = place.point_to_frame([0.0; 3]);
                    let df = place.dir_to_frame(d);
                    best = Some(Hit {
                        frame_id: frame,
                        position: std::array::from_fn(|a| o[a] + t * df[a]),
                        distance: t,
                        kind,
                    });
                }
            }
            hits.clear();
        };
        for tile in &self.tiles {
            let frame = tile.geometry.key.frame_id;
            let Some(place) = self.frames.get(&frame) else {
                continue;
            };
            let o = place.point_to_frame([0.0; 3]);
            tile_hits(tile, o, place.dir_to_frame(d), o, &mut hits);
            consider(frame, HitKind::Surface, place, &mut hits);
        }
        for mesh in &self.meshes {
            let frame = mesh.key.frame_id;
            let Some(place) = self.frames.get(&frame) else {
                continue;
            };
            let o = place.point_to_frame([0.0; 3]);
            mesh_hits(mesh, o, place.dir_to_frame(d), &mut hits);
            consider(frame, HitKind::Cells, place, &mut hits);
        }
        best
    }

    /// `raycast`: the nearest drawn surface tile or cell mesh along the ray
    /// through pixel `(x, y)`.
    pub fn raycast(&self, x: f64, y: f64) -> Option<Hit> {
        self.cast(self.pixel_ray(x, y))
    }

    /// `surfaceHeight`: the drawn surface of `frame_id` along `direction`
    /// (frame axes, from the frame origin), from its drawn tiles.
    pub fn surface_height(&self, frame_id: u64, direction: [f64; 3]) -> Option<SurfaceHeight> {
        let l = length(direction);
        if !(l.is_finite() && l > 0.0) {
            return None;
        }
        let d = direction.map(|c| c / l);
        let place = self.frames.get(&frame_id)?;
        let camera = place.point_to_frame([0.0; 3]);
        let mut hits = Vec::new();
        let mut best: Option<(f64, &TileDrawInfo)> = None;
        for tile in self
            .tiles
            .iter()
            .filter(|t| t.geometry.key.frame_id == frame_id)
        {
            tile_hits(tile, [0.0; 3], d, camera, &mut hits);
            // The outermost crossing: the surface seen from outside.
            for &t in &hits {
                if t > 0.0 && best.is_none_or(|(b, _)| t > b) {
                    best = Some((t, tile));
                }
            }
            hits.clear();
        }
        let (t, tile) = best?;
        let s = tile.geometry.shape;
        let p = d.map(|c| c * t);
        let key = tile.geometry.key;
        Some(SurfaceHeight {
            height: ellipsoid_height(p, s.radius_equatorial, s.radius_polar),
            level: key.level,
            finer: self.coarse.contains(&key),
        })
    }

    /// `project`: `point` (frame axes of `frame_id`) in pixels, or `None`
    /// behind the camera, outside the view, or behind the nearest drawn
    /// thing along its ray. A frame that drew nothing is placed with
    /// `system` and `camera`.
    pub fn project(
        &self,
        system: &FrameSystem,
        camera: &Camera,
        frame_id: u64,
        point: [f64; 3],
    ) -> Option<Projected> {
        let rel = match self.frames.get(&frame_id) {
            Some(place) => place.to_root(point),
            None => v3(camera.relative(system, Vec3::new(point[0], point[1], point[2]), frame_id)),
        };
        let v = &self.view;
        let c = [dot(v[0], rel), dot(v[1], rel), dot(v[2], rel)];
        if c[2] >= 0.0 || c[2].is_nan() {
            return None;
        }
        let depth = -c[2];
        let (w, h) = (f64::from(self.size.0), f64::from(self.size.1));
        let f = (0.5 * self.fov_y).tan();
        let (nx, ny) = (c[0] / (depth * f * w / h), c[1] / (depth * f));
        let (x, y) = ((nx + 1.0) * 0.5 * w, (1.0 - ny) * 0.5 * h);
        if !(0.0..=w).contains(&x) || !(0.0..=h).contains(&y) {
            return None;
        }
        let distance = length(rel);
        if let Some(hit) = self.cast(rel.map(|a| a / distance)) {
            if hit.distance < distance * (1.0 - OCCLUSION_TOLERANCE) {
                return None;
            }
        }
        Some(Projected { x, y, depth })
    }
}

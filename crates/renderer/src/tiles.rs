//! Surface tiles: drawing a frame's surface from heights over a reference
//! shape (`matter-format.md` section 10, `rendering.md` section 7).
//!
//! A frame has a surface when any of its six level-0 tiles is not empty.
//! For such a frame the tiles are selected by distance from a quadtree per
//! cube face, the way continuous level of detail selects a height field: a tile
//! of level `l` is split into its four children while the camera is
//! within [`range`]`(l)` of it, so every drawn tile's quads are about
//! [`PIXELS_PER_QUAD`] pixels or smaller. Each vertex moves continuously
//! from its own position to its parent level's position (the morph) as
//! its distance from the camera goes from [`MORPH_START`] of its parent's
//! range to the whole range, so a tile meets a coarser neighbor with that
//! neighbor's exact shape and nothing changes when a tile is split or
//! joined. Nothing here knows what the frame is.
//!
//! Positions are computed once per tile in `f64` frame coordinates
//! ([`TileGeometry`]) by the core library's geometry, so a vertex shared by
//! two tiles has the same bits in both; they reach the GPU relative to one
//! anchor per frame ([`Anchor`]), so shared vertices stay identical there.

use crate::camera::{Camera, FIELD_OF_VIEW_Y};
use crate::farfield::focal_px;
use gx_core::frames::FrameSystem;
use gx_core::key::{SurfaceKey, SURFACE_FACES};
use gx_core::surface::{
    vertex_position, SurfaceTile, Tile, TILE_QUADS, TILE_VERTICES, VERTEX_COUNT,
};
use gx_core::units::Vec3;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

/// Largest projected quad edge, pixels, at the distance a tile is split.
pub const PIXELS_PER_QUAD: f64 = 4.0;

/// Fraction of a tile's parent range where its morph toward the parent
/// begins. A tile meets a coarser neighbor only where the neighbor is out
/// of the tile's parent range, so the tile is fully morphed there; the
/// neighbor must not have started its own morph there, which holds while
/// `(2 MORPH_START - 1) range(l)` exceeds a tile's diagonal (tested).
pub const MORPH_START: f64 = 0.85;

/// Children are asked for while the camera is within this multiple of a
/// tile's range, so they have usually arrived when they are needed.
pub const PREFETCH: f64 = 1.6;

/// Distance the camera may move from a frame's anchor before the anchor is
/// moved, meters: positions near the camera stay within millimeters in
/// `f32`.
pub const ANCHOR_REACH: f64 = 50_000.0;

/// Grid the anchor snaps to, meters.
pub const ANCHOR_GRID: f64 = 4096.0;

/// One tile ready to draw: positions and their morph targets in `f64` frame
/// coordinates, normals, and material, per vertex (row major, `j` outer).
#[derive(Clone, Debug, PartialEq)]
pub struct TileGeometry {
    /// The tile.
    pub key: SurfaceKey,
    /// Vertex positions, frame coordinates, meters.
    pub positions: Vec<[f64; 3]>,
    /// Where each vertex is one level coarser: the parent tile's surface at
    /// it (a parent vertex, or the middle of the parent's edge or diagonal).
    pub targets: Vec<[f64; 3]>,
    /// Unit normals of the tile's surface, frame axes.
    pub normals: Vec<[f32; 3]>,
    /// Unit normals of the parent's surface at each vertex.
    pub target_normals: Vec<[f32; 3]>,
    /// Albedo per vertex, three bands.
    pub albedo: Vec<[f32; 3]>,
    /// Roughness per vertex.
    pub roughness: Vec<f32>,
    /// Radiating temperature per vertex, kelvin.
    pub temperature: Vec<f32>,
    /// Smallest and largest corner of the box holding every position and
    /// target, frame coordinates.
    pub bounds: ([f64; 3], [f64; 3]),
    /// Length of the tile's edge along the reference shape, meters
    /// (approximately: a quarter of the equator over `2^level`).
    pub edge: f64,
    /// Radius of a ball about the frame origin that the surface never dips
    /// into: the reference shape's smaller radius minus [`OCCLUDER_DEPTH`].
    /// Tiles behind it from the camera are hidden.
    pub occluder: f64,
}

/// How far below the reference shape's smaller radius the occluding ball
/// is taken, meters: deeper than any surface height a tile may have.
pub const OCCLUDER_DEPTH: f64 = 20_000.0;

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn unit_cross(a: [f64; 3], b: [f64; 3]) -> [f32; 3] {
    let c = [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ];
    let l = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
    if l > 0.0 {
        c.map(|v| (v / l) as f32)
    } else {
        [0.0; 3]
    }
}

fn mean_unit(ns: &[[f32; 3]]) -> [f32; 3] {
    let mut s = [0.0f32; 3];
    for n in ns {
        for a in 0..3 {
            s[a] += n[a];
        }
    }
    let l = (s[0] * s[0] + s[1] * s[1] + s[2] * s[2]).sqrt();
    if l > 0.0 {
        s.map(|v| v / l)
    } else {
        ns[0]
    }
}

impl TileGeometry {
    /// Builds a tile's geometry (off the frame thread: about `5 * 67^2`
    /// square roots).
    pub fn new(t: &SurfaceTile) -> TileGeometry {
        let (a, c) = (t.radius_equatorial, t.radius_polar);
        let key = t.key;
        let q = TILE_QUADS as i64;
        let at = |i: i64, j: i64| vertex_position(&key, i, j, f64::from(t.height(i, j)), a, c);
        // Every height's position, border included.
        let side = (q + 3) as usize;
        let mut grid = Vec::with_capacity(side * side);
        for j in -1..=(q + 1) {
            for i in -1..=(q + 1) {
                grid.push(at(i, j));
            }
        }
        let g = |i: i64, j: i64| grid[(i + 1) as usize + side * (j + 1) as usize];
        // The parent's surface at the even vertices, border included.
        let half = q / 2;
        let pside = (half + 3) as usize;
        let mut parent = Vec::with_capacity(pside * pside);
        for m in -1..=(half + 1) {
            for k in -1..=(half + 1) {
                parent.push(vertex_position(
                    &key,
                    2 * k,
                    2 * m,
                    f64::from(t.parent_height(k, m)),
                    a,
                    c,
                ));
            }
        }
        let pg = |k: i64, m: i64| parent[(k + 1) as usize + pside * (m + 1) as usize];
        let parent_normal = |k: i64, m: i64| {
            unit_cross(
                sub(pg(k + 1, m), pg(k - 1, m)),
                sub(pg(k, m + 1), pg(k, m - 1)),
            )
        };
        let mut positions = Vec::with_capacity(VERTEX_COUNT);
        let mut targets = Vec::with_capacity(VERTEX_COUNT);
        let mut normals = Vec::with_capacity(VERTEX_COUNT);
        let mut target_normals = Vec::with_capacity(VERTEX_COUNT);
        let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
        for j in 0..=q {
            for i in 0..=q {
                let p = g(i, j);
                positions.push(p);
                normals.push(unit_cross(
                    sub(g(i + 1, j), g(i - 1, j)),
                    sub(g(i, j + 1), g(i, j - 1)),
                ));
                // The parent's triangles split each 2 x 2 block of quads
                // along the same diagonal as the tile's, from (2k, 2m) to
                // (2k + 2, 2m + 2): an odd vertex lies on the middle of a
                // parent edge or diagonal.
                let (k0, m0) = (i.div_euclid(2), j.div_euclid(2));
                let (oi, oj) = (i % 2 == 1, j % 2 == 1);
                let mid = |x: [f64; 3], y: [f64; 3]| -> [f64; 3] {
                    core::array::from_fn(|n| 0.5 * (x[n] + y[n]))
                };
                let (target, tn) = match (oi, oj) {
                    (false, false) => (pg(k0, m0), parent_normal(k0, m0)),
                    (true, false) => (
                        mid(pg(k0, m0), pg(k0 + 1, m0)),
                        mean_unit(&[parent_normal(k0, m0), parent_normal(k0 + 1, m0)]),
                    ),
                    (false, true) => (
                        mid(pg(k0, m0), pg(k0, m0 + 1)),
                        mean_unit(&[parent_normal(k0, m0), parent_normal(k0, m0 + 1)]),
                    ),
                    (true, true) => (
                        mid(pg(k0, m0), pg(k0 + 1, m0 + 1)),
                        mean_unit(&[parent_normal(k0, m0), parent_normal(k0 + 1, m0 + 1)]),
                    ),
                };
                targets.push(target);
                target_normals.push(tn);
                for v in [p, target] {
                    for n in 0..3 {
                        lo[n] = lo[n].min(v[n]);
                        hi[n] = hi[n].max(v[n]);
                    }
                }
            }
        }
        let albedo = t
            .albedo
            .iter()
            .map(|b| b.map(|v| f32::from(v) / 255.0))
            .collect();
        let roughness = t.roughness.iter().map(|&v| f32::from(v) / 255.0).collect();
        TileGeometry {
            key,
            positions,
            targets,
            normals,
            target_normals,
            albedo,
            roughness,
            temperature: t.temperature.clone(),
            bounds: (lo, hi),
            edge: std::f64::consts::FRAC_PI_2 * a / (1u64 << key.level) as f64,
            occluder: a.min(c) - OCCLUDER_DEPTH,
        }
    }

    /// The index list shared by every tile: two triangles per quad, split
    /// from `(i, j)` to `(i + 1, j + 1)`, counterclockwise seen from outside.
    pub fn indices() -> Vec<u32> {
        let n = TILE_VERTICES as u32;
        let mut out = Vec::with_capacity(TILE_QUADS * TILE_QUADS * 6);
        for j in 0..TILE_QUADS as u32 {
            for i in 0..TILE_QUADS as u32 {
                let v = |di: u32, dj: u32| (i + di) + n * (j + dj);
                out.extend_from_slice(&[v(0, 0), v(1, 0), v(1, 1), v(0, 0), v(1, 1), v(0, 1)]);
            }
        }
        out
    }
}

/// Decodes a surface key's sections (one per layer, in layer order) into
/// the first layer's surface, `None` when every layer's tile is empty.
pub fn decode_tile(
    key: &SurfaceKey,
    sections: &[impl AsRef<[u8]>],
) -> Result<Option<Arc<TileGeometry>>, String> {
    for bytes in sections {
        match gx_core::surface::decode(key, bytes.as_ref()) {
            Ok(Tile::Surface(t)) => return Ok(Some(Arc::new(TileGeometry::new(&t)))),
            Ok(Tile::Empty(_)) => {}
            Err(e) => return Err(format!("code {}: {}", e.code, e.reason)),
        }
    }
    Ok(None)
}

/// The distance within which a tile of `edge` meters is split: where its
/// quads (`edge / TILE_QUADS`) project to [`PIXELS_PER_QUAD`] pixels.
pub fn range(edge: f64, focal: f64) -> f64 {
    edge / TILE_QUADS as f64 * focal / PIXELS_PER_QUAD
}

/// Where a tile is in its fetch.
#[derive(Clone, Debug)]
enum State {
    Requested,
    Ready(Arc<TileGeometry>),
    Empty,
    Gone,
}

/// The point a frame's tiles are drawn relative to.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Anchor {
    /// Frame coordinates, meters.
    pub position: [f64; 3],
    /// Counts the anchor's moves, so GPU copies made for an older anchor
    /// are replaced.
    pub generation: u64,
}

/// One tile to draw: its geometry, the frame's anchor, and the distances
/// over which its vertices morph toward the parent level.
#[derive(Clone, Debug, PartialEq)]
pub struct TileDrawInfo {
    /// The tile.
    pub geometry: Arc<TileGeometry>,
    /// The anchor the GPU copy is made relative to.
    pub anchor: Anchor,
    /// Camera distances, meters, where the morph begins and where it is
    /// complete; both infinite at level 0.
    pub morph: [f32; 2],
}

/// Surface tiles of every frame: the cache, which frames have a surface,
/// the anchors, and the latest selection.
#[derive(Default)]
pub struct Surfaces {
    tiles: HashMap<SurfaceKey, State>,
    /// Per frame: `Some(true)` with a surface, `Some(false)` without one,
    /// absent until its six roots have arrived.
    has_surface: BTreeMap<u64, bool>,
    anchors: BTreeMap<u64, Anchor>,
    requests: Vec<SurfaceKey>,
    drawn: Vec<TileDrawInfo>,
}

impl Surfaces {
    /// No tiles.
    pub fn new() -> Surfaces {
        Surfaces::default()
    }

    /// The frames known to have a surface.
    pub fn frames_with_surface(&self) -> impl Iterator<Item = u64> + '_ {
        self.has_surface.iter().filter(|(_, &s)| s).map(|(&f, _)| f)
    }

    /// Whether `frame_id` is known to have a surface.
    pub fn has_surface(&self, frame_id: u64) -> bool {
        self.has_surface.get(&frame_id) == Some(&true)
    }

    /// Asks for `key` unless it is cached or asked for already.
    fn want(&mut self, key: SurfaceKey) {
        if let std::collections::hash_map::Entry::Vacant(e) = self.tiles.entry(key) {
            e.insert(State::Requested);
            self.requests.push(key);
        }
    }

    /// The keys to request now.
    pub fn take_requests(&mut self) -> Vec<SurfaceKey> {
        std::mem::take(&mut self.requests)
    }

    /// Applies a finished request.
    pub fn complete(
        &mut self,
        key: SurfaceKey,
        outcome: Result<Option<Arc<TileGeometry>>, String>,
    ) {
        let state = match outcome {
            Ok(Some(g)) => State::Ready(g),
            Ok(None) => State::Empty,
            Err(reason) => {
                tracing::warn!(key = %key, "surface tile is invalid: {reason}");
                State::Gone
            }
        };
        self.tiles.insert(key, state);
        if key.level == 0 {
            let roots: Vec<Option<&State>> = (0..SURFACE_FACES)
                .map(|f| {
                    self.tiles.get(&SurfaceKey {
                        frame_id: key.frame_id,
                        face: f,
                        level: 0,
                        x: 0,
                        y: 0,
                    })
                })
                .collect();
            if roots
                .iter()
                .all(|s| !matches!(s, None | Some(State::Requested)))
            {
                let any = roots.iter().any(|s| matches!(s, Some(State::Ready(_))));
                self.has_surface.insert(key.frame_id, any);
            }
        }
    }

    fn ready(&self, key: &SurfaceKey) -> Option<&Arc<TileGeometry>> {
        match self.tiles.get(key) {
            Some(State::Ready(g)) => Some(g),
            _ => None,
        }
    }

    /// Selects the tiles to draw for `camera` and asks for missing ones
    /// (module docs). `frames` are the frames worth looking at (in range,
    /// with mass); each gets its six roots asked for once.
    pub fn select(
        &mut self,
        system: &FrameSystem,
        camera: &Camera,
        view_px: (u32, u32),
        frames: &[u64],
    ) {
        let focal = focal_px(f64::from(view_px.1.max(1)));
        let aspect = f64::from(view_px.0.max(1)) / f64::from(view_px.1.max(1));
        let half_y = 0.5 * FIELD_OF_VIEW_Y;
        let half_x = (half_y.tan() * aspect).atan();
        let cone = (half_x.tan().hypot(half_y.tan())).atan() * 1.2;
        let mut drawn = Vec::new();
        for &frame in frames {
            for face in 0..SURFACE_FACES {
                self.want(SurfaceKey {
                    frame_id: frame,
                    face,
                    level: 0,
                    x: 0,
                    y: 0,
                });
            }
            if !self.has_surface(frame) {
                continue;
            }
            let cam = crate::stream::camera_in_frame(system, camera, frame);
            let cam = [cam.x, cam.y, cam.z];
            let forward = system.root_orientation(frame).conjugate().rotate(
                camera
                    .root_orientation(system)
                    .rotate(Vec3::new(0.0, 0.0, -1.0)),
            );
            let forward = [forward.x, forward.y, forward.z];
            self.select_in_frame(frame, cam, forward, focal, cone, &mut drawn);
        }
        self.drawn = drawn;
    }

    /// The selection of one frame with a surface (module docs): `cam` and
    /// the unit `forward` in frame coordinates, the view's focal length in
    /// pixels and the half angle of a cone holding the view.
    /// The selection of one frame with a surface (module docs): `cam` and
    /// the unit `forward` in frame coordinates, the view's focal length in
    /// pixels and the half angle of a cone holding the view.
    pub fn select_in_frame(
        &mut self,
        frame: u64,
        cam: [f64; 3],
        forward: [f64; 3],
        focal: f64,
        cone: f64,
        drawn: &mut Vec<TileDrawInfo>,
    ) {
        let anchor = self.anchor(frame, cam);
        let mut stack: Vec<SurfaceKey> = (0..SURFACE_FACES)
            .map(|face| SurfaceKey {
                frame_id: frame,
                face,
                level: 0,
                x: 0,
                y: 0,
            })
            .collect();
        while let Some(key) = stack.pop() {
            let Some(geo) = self.ready(&key).cloned() else {
                continue;
            };
            // Outside the view, or behind the reference shape as seen
            // from the camera, nothing is drawn or refined.
            if !in_cone(cam, forward, geo.bounds, cone)
                || below_horizon(cam, geo.bounds, geo.occluder)
            {
                continue;
            }
            let d = box_distance(cam, geo.bounds);
            let r = range(geo.edge, focal);
            let children = key.children();
            if let Some(ch) = children {
                if d < r * PREFETCH {
                    for c in ch {
                        self.want(c);
                    }
                }
            }
            let split = d < r
                && children.is_some_and(|ch| {
                    // Every child arrived, and at least one has a
                    // surface: deeper keys a module leaves empty end
                    // the refinement at this tile.
                    ch.iter()
                        .all(|c| matches!(self.tiles.get(c), Some(State::Ready(_) | State::Empty)))
                        && ch.iter().any(|c| self.ready(c).is_some())
                });
            if split {
                stack.extend(children.into_iter().flatten());
                continue;
            }
            let morph = if key.level == 0 {
                [f32::INFINITY; 2]
            } else {
                let parent_range = r * 2.0;
                [(MORPH_START * parent_range) as f32, parent_range as f32]
            };
            drawn.push(TileDrawInfo {
                geometry: geo,
                anchor,
                morph,
            });
        }
    }

    /// The tiles of the latest selection.
    pub fn drawn(&self) -> &[TileDrawInfo] {
        &self.drawn
    }

    /// Whether every root and every tile the latest selection would split
    /// toward has arrived: nothing requested is outstanding.
    pub fn settled(&self) -> bool {
        self.requests.is_empty() && !self.tiles.values().any(|s| matches!(s, State::Requested))
    }

    /// Cached tiles, for the overlay.
    pub fn cached(&self) -> usize {
        self.tiles.len()
    }

    /// The frame's anchor, moved (snapped to [`ANCHOR_GRID`]) when the
    /// camera is more than [`ANCHOR_REACH`] from it.
    fn anchor(&mut self, frame: u64, cam: [f64; 3]) -> Anchor {
        let snap = |v: f64| (v / ANCHOR_GRID).round() * ANCHOR_GRID;
        let next = |generation| Anchor {
            position: cam.map(snap),
            generation,
        };
        let a = *self.anchors.entry(frame).or_insert_with(|| next(0));
        let far = (0..3).any(|n| (cam[n] - a.position[n]).abs() > ANCHOR_REACH);
        if far {
            let moved = next(a.generation + 1);
            self.anchors.insert(frame, moved);
            moved
        } else {
            a
        }
    }

    /// Drops cached tiles the latest selection does not use, beyond
    /// `keep` tiles (roots and the drawn tiles' ancestors stay).
    pub fn evict(&mut self, keep: usize) {
        if self.tiles.len() <= keep {
            return;
        }
        let mut needed: std::collections::HashSet<SurfaceKey> = std::collections::HashSet::new();
        for d in &self.drawn {
            let mut k = Some(d.geometry.key);
            while let Some(x) = k {
                if !needed.insert(x) {
                    break;
                }
                // The children asked for around a drawn tile.
                if let Some(ch) = x.children() {
                    needed.extend(ch);
                }
                k = x.parent();
            }
        }
        self.tiles
            .retain(|k, s| k.level == 0 || matches!(s, State::Requested) || needed.contains(k));
    }
}

/// Whether every point of the box is hidden from `cam` behind the ball of
/// radius `r` about the frame origin: farther from the camera than the
/// camera's horizon on the ball plus the horizon of the box's highest
/// point.
fn below_horizon(cam: [f64; 3], (lo, hi): ([f64; 3], [f64; 3]), r: f64) -> bool {
    let len = |v: [f64; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    let rc = len(cam);
    if r <= 0.0 || rc <= r {
        return false;
    }
    let center: [f64; 3] = core::array::from_fn(|n| 0.5 * (lo[n] + hi[n]));
    let radius = 0.5 * len(sub(hi, lo));
    let top = len(center) + radius;
    if top <= r {
        return true;
    }
    let near = (len(sub(center, cam)) - radius).max(0.0);
    near > (rc * rc - r * r).sqrt() + (top * top - r * r).sqrt()
}

fn box_distance(p: [f64; 3], (lo, hi): ([f64; 3], [f64; 3])) -> f64 {
    let gap = |c: f64, l: f64, h: f64| (l - c).max(c - h).max(0.0);
    let d = [
        gap(p[0], lo[0], hi[0]),
        gap(p[1], lo[1], hi[1]),
        gap(p[2], lo[2], hi[2]),
    ];
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
}

/// Whether the box can be inside a cone of half angle `cone` around
/// `forward` from `cam`.
fn in_cone(cam: [f64; 3], forward: [f64; 3], (lo, hi): ([f64; 3], [f64; 3]), cone: f64) -> bool {
    let center: [f64; 3] = core::array::from_fn(|n| 0.5 * (lo[n] + hi[n]));
    let radius =
        0.5 * ((hi[0] - lo[0]).powi(2) + (hi[1] - lo[1]).powi(2) + (hi[2] - lo[2]).powi(2)).sqrt();
    let to = sub(center, cam);
    let d = (to[0] * to[0] + to[1] * to[1] + to[2] * to[2]).sqrt();
    if d <= radius {
        return true;
    }
    let cos = (to[0] * forward[0] + to[1] * forward[1] + to[2] * forward[2]) / d;
    cos.clamp(-1.0, 1.0).acos() <= cone + (radius / d).min(1.0).asin()
}

#[cfg(test)]
mod tests {
    use super::*;
    use gx_core::surface::{HEIGHT_SIDE, PARENT_SIDE};

    /// A tile whose heights are a smooth function of the vertex direction,
    /// with parent heights the same function at the parent's level (the
    /// same direction, so the same values at shared vertices).
    fn tile(key: SurfaceKey) -> SurfaceTile {
        let h = |k: &SurfaceKey, i: i64, j: i64| {
            let d = gx_core::surface::vertex_direction(k, i, j);
            (300.0 * (7.0 * d[0]).sin() * (5.0 * d[1]).cos() + 80.0 * (23.0 * d[2]).sin()) as f32
        };
        let q = TILE_QUADS as i64;
        let mut heights = Vec::with_capacity(HEIGHT_SIDE * HEIGHT_SIDE);
        for j in -1..=q + 1 {
            for i in -1..=q + 1 {
                heights.push(h(&key, i, j));
            }
        }
        let mut parent_heights = Vec::with_capacity(PARENT_SIDE * PARENT_SIDE);
        for m in -1..=q / 2 + 1 {
            for k in -1..=q / 2 + 1 {
                parent_heights.push(h(&key, 2 * k, 2 * m));
            }
        }
        SurfaceTile {
            key,
            radius_equatorial: 6_378_137.0,
            radius_polar: 6_356_752.3,
            heights,
            parent_heights,
            albedo: vec![[76, 71, 64]; VERTEX_COUNT],
            roughness: vec![230; VERTEX_COUNT],
            temperature: vec![288.0; VERTEX_COUNT],
        }
    }

    #[test]
    fn neighbors_share_edge_positions_and_targets() {
        let l = TileGeometry::new(&tile(SurfaceKey::new(4, 2, 7, 40, 90).unwrap()));
        let r = TileGeometry::new(&tile(SurfaceKey::new(4, 2, 7, 41, 90).unwrap()));
        let n = TILE_VERTICES;
        for j in 0..n {
            assert_eq!(l.positions[(n - 1) + n * j], r.positions[n * j]);
            assert_eq!(l.targets[(n - 1) + n * j], r.targets[n * j]);
        }
    }

    #[test]
    fn a_fully_morphed_edge_is_the_parents_edge() {
        // A child's edge, morphed completely, lies on its parent's edge:
        // even vertices are the parent's vertices, odd ones the middle of
        // the parent's edge between them.
        let parent_key = SurfaceKey::new(4, 2, 7, 40, 90).unwrap();
        let child_key = parent_key.children().unwrap()[0];
        let p = TileGeometry::new(&tile(parent_key));
        let c = TileGeometry::new(&tile(child_key));
        let n = TILE_VERTICES;
        // Child edge j = 0 (bottom), parent vertices (0..=32, 0).
        for i in 0..n {
            let t = c.targets[i];
            let expect = if i % 2 == 0 {
                p.positions[i / 2]
            } else {
                let (a, b) = (p.positions[i / 2], p.positions[i / 2 + 1]);
                core::array::from_fn(|k| 0.5 * (a[k] + b[k]))
            };
            assert_eq!(t, expect, "vertex {i}");
        }
    }

    /// Fills `s` with synthetic tiles (the [`tile`] surface, frame 4, levels
    /// up to `max_level`) until a selection from `cam` looking along
    /// `forward` asks for nothing more.
    fn settle(
        s: &mut Surfaces,
        cam: [f64; 3],
        forward: [f64; 3],
        max_level: u8,
    ) -> Vec<TileDrawInfo> {
        let focal = focal_px(720.0);
        let cone = 0.9;
        for face in 0..SURFACE_FACES {
            s.want(SurfaceKey::new(4, face, 0, 0, 0).unwrap());
        }
        for _ in 0..64 {
            for key in s.take_requests() {
                let outcome = if key.level <= max_level {
                    Ok(Some(Arc::new(TileGeometry::new(&tile(key)))))
                } else {
                    Ok(None)
                };
                s.complete(key, outcome);
            }
            let mut drawn = Vec::new();
            s.select_in_frame(4, cam, forward, focal, cone, &mut drawn);
            if s.requests.is_empty() {
                return drawn;
            }
        }
        panic!("the selection never settled");
    }

    /// A drawn tile's vertices where the GPU puts them (`vs_tile`): moved
    /// toward the parent level by the morph factor of their distance.
    fn morphed(t: &TileDrawInfo, cam: [f64; 3]) -> Vec<[f64; 3]> {
        let g = &t.geometry;
        let [m0, m1] = t.morph.map(f64::from);
        g.positions
            .iter()
            .zip(&g.targets)
            .map(|(p, q)| {
                let d = sub(*p, cam);
                let d = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                let k = if m0.is_finite() {
                    ((d - m0) / (m1 - m0)).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                core::array::from_fn(|n| p[n] + k * (q[n] - p[n]))
            })
            .collect()
    }

    /// The drawn tile other than `skip` whose face region holds the
    /// direction `dir`.
    fn locate_drawn(drawn: &[TileDrawInfo], dir: [f64; 3], skip: usize) -> Option<usize> {
        let ax = (0..3)
            .max_by(|&a, &b| dir[a].abs().total_cmp(&dir[b].abs()))
            .unwrap();
        let face = (2 * ax + usize::from(dir[ax] < 0.0)) as u8;
        let (n, u, v) = gx_core::surface::face_axes(face);
        let dn = dir[0] * n[0] + dir[1] * n[1] + dir[2] * n[2];
        let sc = (dir[0] * u[0] + dir[1] * u[1] + dir[2] * u[2]) / dn;
        let tc = (dir[0] * v[0] + dir[1] * v[1] + dir[2] * v[2]) / dn;
        drawn.iter().enumerate().position(|(i, t)| {
            let k = t.geometry.key;
            if i == skip || k.face != face {
                return false;
            }
            let side = (1u64 << k.level) as f64;
            let x = ((sc + 1.0) / 2.0 * side).floor();
            let y = ((tc + 1.0) / 2.0 * side).floor();
            x == f64::from(k.x) && y == f64::from(k.y)
        })
    }

    /// The seam proof for surface tiles (`rendering.md` 7): from cameras
    /// along a low flight over rough synthetic relief, every boundary
    /// vertex of every drawn tile, morphed as the GPU morphs it, lies on the
    /// boundary of another drawn tile (within a tenth of a millimeter), so
    /// tiles meet with no gap, whatever their levels.
    #[test]
    fn drawn_tiles_meet_without_gaps() {
        let (a, c) = (6_378_137.0, 6_356_752.3);
        let n = TILE_VERTICES;
        let boundary: Vec<usize> = (0..n)
            .flat_map(|i| [i, i + n * (n - 1), n * i, n * i + n - 1])
            .collect();
        let mut checked = 0usize;
        let mut levels = std::collections::BTreeSet::new();
        for step in 0..12 {
            let mut s = Surfaces::new();
            // 3 km up, moving along the surface, looking down 25 degrees.
            let d = gx_core::surface::vertex_direction(
                &SurfaceKey::new(4, 4, 0, 0, 0).unwrap(),
                20 + step,
                9,
            );
            let (p, nrm) = gx_core::surface::ellipsoid_point(d, a, c);
            let cam: [f64; 3] = core::array::from_fn(|k| p[k] + 3000.0 * nrm[k]);
            let east = unit_cross([0.0, 0.0, 1.0], nrm).map(f64::from);
            let forward: [f64; 3] = core::array::from_fn(|k| 0.9 * east[k] - 0.42 * nrm[k]);
            let drawn = settle(&mut s, cam, forward, 11);
            assert!(drawn.len() > 20, "{} tiles drawn", drawn.len());
            let placed: Vec<Vec<[f64; 3]>> = drawn.iter().map(|t| morphed(t, cam)).collect();
            for (ti, t) in drawn.iter().enumerate() {
                levels.insert(t.geometry.key.level);
                let key = t.geometry.key;
                for &v in &boundary {
                    let p = placed[ti][v];
                    let (vi, vj) = ((v % n) as i64, (v / n) as i64);
                    // Just outside the tile's edge (or corner) in face
                    // coordinates, crossing onto the next face if needed.
                    let q = TILE_QUADS as i64;
                    let out = |c: i64| {
                        if c == 0 {
                            -1.0
                        } else if c == q {
                            1.0
                        } else {
                            0.0
                        }
                    };
                    let (di, dj) = (out(vi), out(vj));
                    let eps = 1e-7;
                    let sc = gx_core::surface::face_coord(key.level, key.x, vi) + di * eps;
                    let tc = gx_core::surface::face_coord(key.level, key.y, vj) + dj * eps;
                    let (fnm, fu, fv) = gx_core::surface::face_axes(key.face);
                    let dir: [f64; 3] = core::array::from_fn(|k| fnm[k] + sc * fu[k] + tc * fv[k]);
                    let Some(ui) = locate_drawn(&drawn, dir, ti) else {
                        // Nothing drawn there: the edge of the drawn region.
                        continue;
                    };
                    let qv = &placed[ui];
                    let on = |idx: &dyn Fn(usize) -> usize| {
                        (0..n - 1).any(|i| {
                            let (x, y) = (qv[idx(i)], qv[idx(i + 1)]);
                            let e = sub(y, x);
                            let w = sub(p, x);
                            let ee = e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
                            let tt =
                                ((w[0] * e[0] + w[1] * e[1] + w[2] * e[2]) / ee).clamp(0.0, 1.0);
                            let r = sub(w, e.map(|x| x * tt));
                            (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt() < 1e-4
                        })
                    };
                    let ok = on(&|i| i)
                        || on(&|i| i + n * (n - 1))
                        || on(&|i| n * i)
                        || on(&|i| n * i + n - 1);
                    assert!(
                        ok,
                        "step {step}: vertex ({vi}, {vj}) of {key} (level {}) at {p:?} is not on the edge of its neighbor {} (level {})",
                        key.level,
                        drawn[ui].geometry.key,
                        drawn[ui].geometry.key.level
                    );
                    checked += 1;
                }
            }
        }
        assert!(checked > 10_000, "{checked} boundary vertices checked");
        assert!(levels.len() >= 5, "levels {levels:?}");
    }

    #[test]
    fn indices_cover_the_grid() {
        let idx = TileGeometry::indices();
        assert_eq!(idx.len(), TILE_QUADS * TILE_QUADS * 6);
        assert_eq!(*idx.iter().max().unwrap() as usize, VERTEX_COUNT - 1);
    }
}

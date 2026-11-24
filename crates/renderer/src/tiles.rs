//! Surface tiles: drawing a frame's surface from heights over a reference
//! shape (`matter-format.md` section 10, `rendering.md` section 7).
//!
//! A frame has a surface when any of its six level-0 tiles is not empty.
//! For such a frame the tiles are selected by distance from a quadtree per
//! cube face, the way continuous level of detail selects a height field: a tile
//! of level `l` is split into its four children while the camera is
//! within [`range`]`(l)` of it, so every drawn tile's quads are about
//! [`PIXELS_PER_QUAD`] pixels or smaller, or finer where the seamless
//! morph needs a longer range ([`range`]). Each vertex moves continuously
//! from its own position to its parent level's position (the morph) as
//! its distance from the camera goes from [`MORPH_START`] of its parent's
//! range to the whole range, so a tile meets a coarser neighbor with that
//! neighbor's exact shape and nothing changes when a tile is split or
//! joined. Nothing here knows what the frame is.
//!
//! A tile replaces its parent only when each child has its surface tile
//! and the overlays it waits for (`rendering.md` section 7 item 6): every
//! shown overlay layer has arrived, failed, or is not `required` and
//! [`OVERLAY_WAIT_SECONDS`] have passed since the child's surface tile
//! arrived. Level-0 tiles are drawn without waiting.
//!
//! Positions are computed once per tile in `f64` frame coordinates
//! ([`TileGeometry`]) by the core library's geometry, so a vertex shared by
//! two tiles has the same bits in both; they reach the GPU relative to one
//! anchor per frame ([`Anchor`]), so shared vertices stay identical there.

use crate::camera::Camera;
use crate::farfield::focal_px;
use gx_core::frames::FrameSystem;
use gx_core::key::{SurfaceKey, SURFACE_FACES};
use gx_core::overlay::{self, Overlay, OverlayTile};
use gx_core::surface::{
    vertex_position, SurfaceTile, Tile, TILE_QUADS, TILE_VERTICES, VERTEX_COUNT,
};
use gx_core::units::Vec3;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;

/// Largest projected quad edge, pixels, at the distance a tile is split
/// (unless the morph needs the split farther out, [`range`]).
pub const PIXELS_PER_QUAD: f64 = 8.0;

/// Fraction of a tile's parent range where its morph toward the parent
/// begins. A tile meets a coarser neighbor only where the neighbor is out
/// of the tile's parent range, so the tile is fully morphed there; the
/// neighbor must not have started its own morph there, which holds while
/// `(2 MORPH_START - 1) range(l)` exceeds a tile's diagonal ([`range`]
/// keeps it so; tested).
pub const MORPH_START: f64 = 0.9;

/// Margin on the tile diagonal in the morph condition ([`range`]).
pub const SEAM_MARGIN: f64 = 1.05;

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
    /// The longest a tile edge of this level can be, meters: at a face's
    /// center, where the cube's face coordinate is stretched least, on the
    /// larger radius: `2 max(a, c) / 2^level`. A function of the level and
    /// the frame only, so neighbors agree.
    pub max_edge: f64,
    /// Radius of a ball about the frame origin that the frame's surface
    /// never dips into: the reference shape's smaller radius plus the
    /// tile's `height_floor` (`matter-format.md` 10.3), the lowest height
    /// its compiler declares for the frame's whole surface.
    pub occluder: f64,
    /// The reference ellipsoid's equatorial and polar radii and the
    /// frame's `height_floor`, meters (`matter-format.md` section 10), as
    /// the tile declares them.
    pub shape: SurfaceShape,
}

/// A frame's surface as its tiles declare it (`matter-format.md` section
/// 10): the reference ellipsoid and the lowest height above it.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SurfaceShape {
    /// Equatorial radius of the reference ellipsoid, meters.
    pub radius_equatorial: f64,
    /// Polar radius of the reference ellipsoid, meters.
    pub radius_polar: f64,
    /// The lowest height above the ellipsoid the compiler declares for
    /// the whole surface, meters.
    pub height_floor: f64,
}

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
            max_edge: 2.0 * a.max(c) / (1u64 << key.level) as f64,
            occluder: a.min(c) + t.height_floor,
            shape: SurfaceShape {
                radius_equatorial: a,
                radius_polar: c,
                height_floor: t.height_floor,
            },
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

/// A surface key's sections decoded (`rendering.md` section 7 item 6):
/// the base surface and the overlays that have arrived.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DecodedTile {
    /// The non-overlay layers' non-empty surface tiles, each with its
    /// layer index, in layer order: the first shown one is the base
    /// (`rendering.md` section 7 item 7).
    pub bases: Vec<(usize, Arc<TileGeometry>)>,
    /// The overlay layers' non-empty overlay tiles, each with its layer
    /// index, in layer order.
    pub overlays: Vec<(usize, Arc<OverlayTile>)>,
    /// The overlay layers whose section is invalid, with why, in layer
    /// order: they
    /// contribute nothing and are not waited for (`rendering.md` section 7
    /// item 6).
    pub failed: Vec<(usize, String)>,
    /// The overlay layers still to come ([`crate::directions::Completion`]),
    /// in layer order; set by the driver with [`DecodedTile::waiting_on`].
    pub waiting: Vec<usize>,
}

/// Decodes a surface key's sections, `layers[n]` the layer index of
/// `sections[n]`, in layer order: the base from the non-overlay layers,
/// the overlays from the layers in `overlay_layers`.
pub fn decode_tile(
    key: &SurfaceKey,
    sections: &[impl AsRef<[u8]>],
    layers: &[usize],
    overlay_layers: &BTreeSet<usize>,
) -> Result<DecodedTile, String> {
    let mut out = DecodedTile::default();
    for (n, bytes) in sections.iter().enumerate() {
        let layer = layers.get(n).copied().unwrap_or(n);
        let bytes = bytes.as_ref();
        if overlay_layers.contains(&layer) {
            match overlay::decode(key, bytes) {
                Ok(Overlay::Tile(t)) => out.overlays.push((layer, Arc::new(t))),
                Ok(Overlay::Empty(_)) => {}
                // An invalid overlay fails alone: the tile is drawn
                // without it (logged once by [`Surfaces`]).
                Err(e) => out
                    .failed
                    .push((layer, format!("code {}: {}", e.code, e.reason))),
            }
            continue;
        }
        match gx_core::surface::decode(key, bytes) {
            Ok(Tile::Surface(t)) => out.bases.push((layer, Arc::new(geometry(&t)))),
            Ok(Tile::Empty(_)) => {}
            Err(e) => return Err(format!("code {}: {}", e.code, e.reason)),
        }
    }
    Ok(out)
}

impl DecodedTile {
    /// The first non-empty surface tile among the non-overlay layers, with
    /// every layer shown.
    pub fn base(&self) -> Option<&Arc<TileGeometry>> {
        self.bases.first().map(|(_, b)| b)
    }

    /// The tile with `waiting` as the overlay layers still to come.
    pub fn waiting_on(mut self, waiting: Vec<usize>) -> DecodedTile {
        self.waiting = waiting;
        self
    }
}

/// The base tile's geometry with its material composited with
/// `overlays`, in order, per vertex by [`overlay::composite`]
/// (`rendering.md` section 7 item 6). The parent values of each overlay
/// are composited the same way over the base at the tile's even vertices
/// ([`composite_parent`]). Positions and normals are the base's.
pub fn composite_tile(base: &TileGeometry, overlays: &[&OverlayTile]) -> TileGeometry {
    let mut out = base.clone();
    if overlays.is_empty() {
        return out;
    }
    let mut samples = Vec::with_capacity(overlays.len());
    for v in 0..out.albedo.len() {
        samples.clear();
        samples.extend(overlays.iter().map(|o| sample(o, v)));
        let (a, r) = overlay::composite(base.albedo[v], base.roughness[v], &samples);
        out.albedo[v] = a;
        out.roughness[v] = r;
    }
    out
}

fn sample(o: &OverlayTile, v: usize) -> overlay::OverlaySample {
    (
        overlay::unit(o.coverage[v]),
        o.albedo[v].map(overlay::unit),
        overlay::unit(o.roughness[v]),
    )
}

/// The material at the parent's values (`rendering.md` section 7 item 6):
/// per even vertex `(2k, 2m)`, row major with `m` outer, the base's
/// albedo and roughness there composited with each overlay's parent
/// values, in order.
pub fn composite_parent(base: &TileGeometry, overlays: &[&OverlayTile]) -> Vec<([f32; 3], f32)> {
    let mut out = Vec::with_capacity(overlay::PARENT_COUNT);
    for m in 0..overlay::PARENT_VERTICES {
        for k in 0..overlay::PARENT_VERTICES {
            let v = 2 * k + TILE_VERTICES * 2 * m;
            let p = k + overlay::PARENT_VERTICES * m;
            let samples: Vec<overlay::OverlaySample> = overlays
                .iter()
                .map(|o| {
                    (
                        overlay::unit(o.parent_coverage[p]),
                        o.parent_albedo[p].map(overlay::unit),
                        overlay::unit(o.parent_roughness[p]),
                    )
                })
                .collect();
            out.push(overlay::composite(
                base.albedo[v],
                base.roughness[v],
                &samples,
            ));
        }
    }
    out
}

/// The distance within which a tile is split: where its quads
/// (`edge / TILE_QUADS`) project to [`PIXELS_PER_QUAD`] pixels, but at
/// least far enough out that `(2 MORPH_START - 1)` of it exceeds the
/// longest tile diagonal of the level (`sqrt 2 max_edge`, times
/// [`SEAM_MARGIN`]), so tiles still meet without gaps. In small views the
/// second bound decides and quads end up finer than [`PIXELS_PER_QUAD`].
pub fn range(edge: f64, max_edge: f64, focal: f64) -> f64 {
    let pixels = edge / TILE_QUADS as f64 * focal / PIXELS_PER_QUAD;
    let seams = std::f64::consts::SQRT_2 * max_edge * SEAM_MARGIN / (2.0 * MORPH_START - 1.0);
    pixels.max(seams)
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

/// How long, seconds, a tile waits on an overlay that is not `required`
/// before it replaces its parent without it (`rendering.md` section 7 item
/// 6), counted from the arrival of the tile's surface tile on the clock
/// the selection runs on ([`Surfaces::select`]). The selection runs every
/// frame, so a still camera still splits once this has passed.
pub const OVERLAY_WAIT_SECONDS: f64 = 2.0;

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
    /// The latest selection's wanted set (`client-compute.md` section
    /// 2.8): every tile it asked for that has not arrived, with its
    /// priority, in the order asked. `None` once taken.
    wanted: Option<Vec<(SurfaceKey, f64)>>,
    drawn: Vec<TileDrawInfo>,
    /// Every tile the latest selection looked at: the tiles it walked and
    /// the children it checked or asked for. Eviction keeps them.
    used: std::collections::HashSet<SurfaceKey>,
    /// Every ready tile's base and overlays (`rendering.md` section 7
    /// item 6).
    layers: HashMap<SurfaceKey, TileLayers>,
    /// Ready tiles whose overlay layers are still to come: kept in the
    /// wanted set so they arrive.
    pending: std::collections::HashSet<SurfaceKey>,
    /// The hidden layers, by index (`rendering.md` section 7 item 7).
    hidden: BTreeSet<usize>,
    /// The `required` overlay layers, by index (`client-compute.md`
    /// section 2.2).
    required: BTreeSet<usize>,
    /// The time, seconds, of the latest selection: the clock tiles'
    /// arrivals and [`OVERLAY_WAIT_SECONDS`] are measured on.
    now: f64,
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

    /// The surface of `frame_id` as its level-0 tiles declare it, once one
    /// of them has arrived drawn; `None` before.
    pub fn shape(&self, frame_id: u64) -> Option<SurfaceShape> {
        (0..SURFACE_FACES).find_map(|face| {
            let key = SurfaceKey {
                frame_id,
                face,
                level: 0,
                x: 0,
                y: 0,
            };
            match self.tiles.get(&key) {
                Some(State::Ready(g)) => Some(g.shape),
                _ => None,
            }
        })
    }

    /// Asks for `key` unless it is cached or asked for already, and adds
    /// it to the wanted set with `priority` (projected size, pixels)
    /// unless it has arrived.
    fn want(&mut self, key: SurfaceKey, priority: f64) {
        match self.tiles.entry(key) {
            std::collections::hash_map::Entry::Vacant(e) => {
                e.insert(State::Requested);
                self.requests.push(key);
            }
            std::collections::hash_map::Entry::Occupied(e) => {
                if !matches!(e.get(), State::Requested) && !self.pending.contains(&key) {
                    return;
                }
            }
        }
        self.wanted
            .get_or_insert_with(Vec::new)
            .push((key, priority));
    }

    /// The hidden layers, by index (`rendering.md` section 7 item 7).
    pub fn hidden_layers(&self) -> &BTreeSet<usize> {
        &self.hidden
    }

    /// Hides the layers in `hidden` (indices) and shows every other one
    /// (`rendering.md` section 7 item 7): every arrived tile is drawn again
    /// from its first shown base, or not at all when every base is hidden,
    /// and its material composited again from its shown overlays, so no
    /// stale material is drawn. A frame whose level-0 tiles all lose their
    /// base has no surface while they are hidden, and its cells are drawn.
    pub fn set_hidden_layers(&mut self, hidden: BTreeSet<usize>) {
        if hidden == self.hidden {
            return;
        }
        self.hidden = hidden;
        let mut frames = BTreeSet::new();
        for (key, layers) in &self.layers {
            let state = match layers.drawn(&self.hidden) {
                Some(g) => State::Ready(g),
                None => State::Empty,
            };
            self.tiles.insert(*key, state);
            frames.insert(key.frame_id);
        }
        for frame in frames {
            self.update_has_surface(frame);
        }
    }

    /// The `required` overlay layers, by index (`client-compute.md`
    /// section 2.2, [`crate::protocol::BuildDescription::required_layers`]):
    /// a tile waits on these until they arrive or fail before it replaces
    /// its parent.
    pub fn set_required_layers(&mut self, required: BTreeSet<usize>) {
        self.required = required;
    }

    /// The geometry drawn for `key`, if it is ready.
    pub fn drawn_geometry(&self, key: &SurfaceKey) -> Option<&Arc<TileGeometry>> {
        self.ready(key)
    }

    /// The keys to request now.
    pub fn take_requests(&mut self) -> Vec<SurfaceKey> {
        std::mem::take(&mut self.requests)
    }

    /// The latest selection's wanted set, once per selection: every tile
    /// it wants that has not arrived, each with its projected size,
    /// pixels, as priority (level-0 roots at
    /// [`crate::directions::REQUESTED_PRIORITY`]). A tile asked for before
    /// and left out of this set is forgotten, since the fetcher drops it.
    pub fn take_wanted(&mut self) -> Option<Vec<(SurfaceKey, f64)>> {
        self.requests.clear();
        let wanted = self.wanted.take()?;
        let keep: std::collections::HashSet<SurfaceKey> = wanted.iter().map(|(k, _)| *k).collect();
        self.tiles
            .retain(|k, s| !matches!(s, State::Requested) || keep.contains(k));
        Some(wanted)
    }

    /// Applies a finished request without overlays.
    pub fn complete(
        &mut self,
        key: SurfaceKey,
        outcome: Result<Option<Arc<TileGeometry>>, String>,
    ) {
        self.complete_layers(
            key,
            outcome.map(|base| DecodedTile {
                bases: base.into_iter().map(|b| (0, b)).collect(),
                ..DecodedTile::default()
            }),
            false,
        );
    }

    /// Applies a finished or partly finished request
    /// (`rendering.md` section 7 item 6): the tile is drawn with the
    /// overlays it has, and a later completion with more overlays replaces
    /// its material (a new geometry, so its GPU copy is rebuilt).
    /// `partial` is `true` while overlays are still to come.
    ///
    /// Completions of one request may arrive out of order (the native
    /// fetcher decodes each on its own thread): a partial completion is
    /// ignored when the tile already holds the request's final one, or as
    /// many overlays (arrived or failed), so a stale partial never replaces a fuller tile.
    pub fn complete_layers(
        &mut self,
        key: SurfaceKey,
        outcome: Result<DecodedTile, String>,
        partial: bool,
    ) {
        if partial {
            if let (Some(held), Ok(d)) = (self.layers.get(&key), &outcome) {
                // Overlays that failed count as arrived: a later
                // completion holds them too.
                let had = held.overlays.len() + held.failed.len();
                if !self.pending.contains(&key) || had >= d.overlays.len() + d.failed.len() {
                    return;
                }
            }
        }
        if partial {
            self.pending.insert(key);
        } else {
            self.pending.remove(&key);
        }
        let held = self.layers.remove(&key);
        let state = match outcome {
            Ok(DecodedTile {
                bases,
                overlays,
                failed,
                waiting,
            }) if !bases.is_empty() => {
                let arrived_at = held.as_ref().map_or(self.now, |h| h.arrived_at);
                let mut known = held.map(|h| h.failed).unwrap_or_default();
                for (layer, reason) in failed {
                    if known.insert(layer) {
                        tracing::warn!(key = %key, layer, "overlay tile is invalid: {reason}");
                    }
                }
                let waiting = if partial { waiting } else { Vec::new() };
                let layers = TileLayers {
                    bases,
                    overlays,
                    failed: known,
                    waiting,
                    arrived_at,
                };
                let drawn = layers.drawn(&self.hidden);
                self.layers.insert(key, layers);
                match drawn {
                    Some(g) => State::Ready(g),
                    None => State::Empty,
                }
            }
            Ok(_) => State::Empty,
            Err(reason) => {
                tracing::warn!(key = %key, "surface tile is invalid: {reason}");
                self.pending.remove(&key);
                State::Gone
            }
        };
        self.tiles.insert(key, state);
        if key.level == 0 {
            self.update_has_surface(key.frame_id);
        }
    }

    /// Records whether `frame_id` has a surface once all six of its
    /// level-0 tiles have an answer: whether any of them is drawn.
    fn update_has_surface(&mut self, frame_id: u64) {
        {
            let roots: Vec<Option<&State>> = (0..SURFACE_FACES)
                .map(|f| {
                    self.tiles.get(&SurfaceKey {
                        frame_id,
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
                self.has_surface.insert(frame_id, any);
            }
        }
    }

    /// Whether `key` has arrived as far as a split needs (`rendering.md`
    /// section 7 item 6): it is empty, or it has its surface tile and
    /// every shown overlay layer has arrived for it, failed for it, or is
    /// not `required` and [`OVERLAY_WAIT_SECONDS`] have passed since its
    /// surface tile arrived. A hidden overlay is never waited for.
    fn arrived_for_split(&self, key: &SurfaceKey) -> bool {
        match self.tiles.get(key) {
            Some(State::Empty) => true,
            Some(State::Ready(_)) => self.layers.get(key).is_none_or(|l| {
                let waited = self.now - l.arrived_at >= OVERLAY_WAIT_SECONDS;
                l.waiting.iter().all(|layer| {
                    self.hidden.contains(layer)
                        || l.failed.contains(layer)
                        || (waited && !self.required.contains(layer))
                })
            }),
            _ => false,
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
    /// with mass); each gets its six roots asked for once. `now`, seconds,
    /// is the selection's clock ([`OVERLAY_WAIT_SECONDS`]).
    pub fn select(
        &mut self,
        system: &FrameSystem,
        camera: &Camera,
        view_px: (u32, u32),
        frames: &[u64],
        now: f64,
    ) {
        self.now = now;
        let focal = focal_px(f64::from(view_px.1.max(1)), camera.fov_y);
        let aspect = f64::from(view_px.0.max(1)) / f64::from(view_px.1.max(1));
        let half_y = 0.5 * camera.fov_y;
        let half_x = (half_y.tan() * aspect).atan();
        let cone = (half_x.tan().hypot(half_y.tan())).atan() * 1.2;
        let mut drawn = Vec::new();
        self.used.clear();
        self.wanted = Some(Vec::new());
        for &frame in frames {
            for face in 0..SURFACE_FACES {
                self.want(
                    SurfaceKey {
                        frame_id: frame,
                        face,
                        level: 0,
                        x: 0,
                        y: 0,
                    },
                    crate::directions::REQUESTED_PRIORITY,
                );
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
        // The ball the frame's surface never dips into, from its level-0
        // tiles' declared floor (`rendering.md` 7.2); none, no hiding.
        let occluder = stack
            .iter()
            .filter_map(|k| self.ready(k).map(|g| g.occluder))
            .fold(f64::INFINITY, f64::min);
        let occluder = if occluder.is_finite() { occluder } else { 0.0 };
        while let Some(key) = stack.pop() {
            self.used.insert(key);
            let Some(geo) = self.ready(&key).cloned() else {
                continue;
            };
            // Outside the view, or behind the reference shape as seen
            // from the camera, nothing is drawn or refined.
            if !in_cone(cam, forward, geo.bounds, cone) || below_horizon(cam, geo.bounds, occluder)
            {
                continue;
            }
            let d = box_distance(cam, geo.bounds);
            let r = range(geo.edge, geo.max_edge, focal);
            let children = key.children();
            if let Some(ch) = children {
                self.used.extend(ch);
                if d < r * PREFETCH {
                    // A child's projected edge: half the parent's.
                    let px = 0.5 * geo.edge * focal / d.max(f64::MIN_POSITIVE);
                    for c in ch {
                        self.want(c, px);
                    }
                }
            }
            let split = d < r
                && children.is_some_and(|ch| {
                    // Every child arrived with the overlays it waits for
                    // (`rendering.md` 7.6), and at least one has a
                    // surface: deeper keys a module leaves empty end
                    // the refinement at this tile. Level-0 tiles have no
                    // parent and are drawn without waiting.
                    ch.iter().all(|c| self.arrived_for_split(c))
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

    /// The drawn tiles a finer tile is on its way for: a child asked for
    /// and not yet arrived (`surfaceHeight`'s `finer`, `docs/api.md`
    /// section 6).
    pub fn coarse_drawn(&self) -> Vec<SurfaceKey> {
        self.drawn
            .iter()
            .map(|t| t.geometry.key)
            .filter(|k| {
                k.children().is_some_and(|ch| {
                    ch.iter().any(|c| {
                        matches!(self.tiles.get(c), Some(State::Requested))
                            || self.requests.contains(c)
                    })
                })
            })
            .collect()
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

    /// Drops cached tiles the latest selection did not look at, beyond
    /// `keep` tiles: roots, tiles asked for, and every tile the selection
    /// walked or checked as a child stay. Keeping only what is drawn and
    /// its ancestors is not enough: a split tile whose children are all
    /// hidden draws nothing, and dropping its children would make its
    /// parent draw coarse until they come back, again and again.
    pub fn evict(&mut self, keep: usize) {
        if self.tiles.len() <= keep {
            return;
        }
        let used = &self.used;
        self.tiles
            .retain(|k, s| k.level == 0 || matches!(s, State::Requested) || used.contains(k));
        let tiles = &self.tiles;
        self.layers.retain(|k, _| tiles.contains_key(k));
        self.pending.retain(|k| tiles.contains_key(k));
    }
}

/// A tile's base surfaces and the overlays that have arrived for it.
#[derive(Clone, Debug)]
struct TileLayers {
    /// Every non-empty base, with its layer index, in layer order.
    bases: Vec<(usize, Arc<TileGeometry>)>,
    overlays: Vec<(usize, Arc<OverlayTile>)>,
    /// Overlay layers whose section was invalid, already logged.
    failed: BTreeSet<usize>,
    /// Overlay layers still to come while the tile is pending.
    waiting: Vec<usize>,
    /// When the tile's surface tile arrived, on the selection's clock.
    arrived_at: f64,
}

impl TileLayers {
    /// The geometry to draw: the first shown base with the shown overlays
    /// composited, or that base itself when no overlay is shown; `None`
    /// when every base is hidden.
    fn drawn(&self, hidden: &BTreeSet<usize>) -> Option<Arc<TileGeometry>> {
        let (_, base) = self.bases.iter().find(|(l, _)| !hidden.contains(l))?;
        let shown: Vec<&OverlayTile> = (self.overlays.iter())
            .filter(|(l, _)| !hidden.contains(l))
            .map(|(_, o)| o.as_ref())
            .collect();
        Some(if shown.is_empty() {
            Arc::clone(base)
        } else {
            Arc::new(composite_tile(base, &shown))
        })
    }
}

/// [`TileGeometry::new`], timed for the memory report with the
/// `memory-report` feature.
fn geometry(t: &SurfaceTile) -> TileGeometry {
    timed!(TILE_GEOMETRY, TileGeometry::new(t))
}

/// [`decode_tile`], timed for the memory report with the `memory-report`
/// feature: what both fetchers call.
pub(crate) fn decode_tile_timed(
    key: &SurfaceKey,
    sections: &[impl AsRef<[u8]>],
    layers: &[usize],
    overlays: &BTreeSet<usize>,
) -> Result<DecodedTile, String> {
    timed!(DECODE_TILE, decode_tile(key, sections, layers, overlays))
}

#[cfg(feature = "memory-report")]
impl TileGeometry {
    /// Bytes this geometry holds: the struct and each array's capacity
    /// (the memory report, `docs/architecture.md`).
    #[cfg(feature = "memory-report")]
    pub fn heap_bytes(&self) -> u64 {
        fn cap<T>(v: &Vec<T>) -> usize {
            v.capacity() * std::mem::size_of::<T>()
        }
        (std::mem::size_of::<TileGeometry>()
            + cap(&self.positions)
            + cap(&self.targets)
            + cap(&self.normals)
            + cap(&self.target_normals)
            + cap(&self.albedo)
            + cap(&self.roughness)
            + cap(&self.temperature)) as u64
    }
}

/// Bytes an overlay tile holds: the struct and each array's capacity (the
/// memory report). A free function since the type is the core library's.
#[cfg(feature = "memory-report")]
pub fn overlay_heap_bytes(o: &OverlayTile) -> u64 {
    (std::mem::size_of::<OverlayTile>()
        + o.coverage.capacity()
        + o.albedo.capacity() * 3
        + o.roughness.capacity()
        + o.parent_coverage.capacity()
        + o.parent_albedo.capacity() * 3
        + o.parent_roughness.capacity()) as u64
}

#[cfg(feature = "memory-report")]
impl Surfaces {
    /// What the surface tiles hold ([`crate::memory::SurfaceMemory`]):
    /// entries by state, the selection's sizes, and the bytes of every
    /// distinct geometry and overlay tile, each `Arc` counted once.
    pub fn memory(&self) -> crate::memory::SurfaceMemory {
        use std::collections::BTreeMap as Distinct;
        let mut m = crate::memory::SurfaceMemory {
            used: self.used.len(),
            drawn: self.drawn.len(),
            ..Default::default()
        };
        let ptr = |g: &Arc<TileGeometry>| Arc::as_ptr(g) as usize;
        let mut geometries: Distinct<usize, u64> = Distinct::new();
        let mut bases: BTreeSet<usize> = BTreeSet::new();
        let mut overlays: Distinct<usize, u64> = Distinct::new();
        for state in self.tiles.values() {
            match state {
                State::Requested => m.requested += 1,
                State::Ready(g) => {
                    m.ready += 1;
                    geometries.insert(ptr(g), g.heap_bytes());
                }
                State::Empty => m.empty += 1,
                State::Gone => m.gone += 1,
            }
        }
        for l in self.layers.values() {
            for (_, b) in &l.bases {
                bases.insert(ptr(b));
                geometries.insert(ptr(b), b.heap_bytes());
            }
            for (_, o) in &l.overlays {
                overlays.insert(Arc::as_ptr(o) as usize, overlay_heap_bytes(o));
            }
        }
        let mut drawn: Distinct<usize, u64> = Distinct::new();
        let mut drawn_overlays: Distinct<usize, u64> = Distinct::new();
        for t in &self.drawn {
            let g = &t.geometry;
            geometries.insert(ptr(g), g.heap_bytes());
            drawn.insert(ptr(g), g.heap_bytes());
            if let Some(l) = self.layers.get(&g.key) {
                for (_, b) in &l.bases {
                    drawn.insert(ptr(b), b.heap_bytes());
                }
                for (_, o) in &l.overlays {
                    drawn_overlays.insert(Arc::as_ptr(o) as usize, overlay_heap_bytes(o));
                }
            }
        }
        m.geometries = geometries.len();
        m.geometry_bytes = geometries.values().sum();
        for (p, bytes) in &geometries {
            if !bases.contains(p) {
                m.composited += 1;
                m.composited_bytes += bytes;
            }
        }
        m.drawn_geometry_bytes = drawn.values().sum();
        m.overlay_bytes = overlays.values().sum();
        m.drawn_overlay_bytes = drawn_overlays.values().sum();
        m
    }

    /// Bytes of every distinct tile geometry held, for the overlay line.
    pub fn geometry_bytes(&self) -> u64 {
        self.memory().geometry_bytes
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
    use crate::camera::FIELD_OF_VIEW_Y;
    use gx_core::surface::{HEIGHT_SIDE, PARENT_SIDE};

    fn overlay_tile(key: SurfaceKey, coverage: u8, albedo: u8, roughness: u8) -> OverlayTile {
        let mut c = vec![coverage; VERTEX_COUNT];
        // Partial coverage: the first row is not covered.
        for v in c.iter_mut().take(TILE_VERTICES) {
            *v = 0;
        }
        OverlayTile {
            key,
            coverage: c,
            albedo: vec![[albedo; 3]; VERTEX_COUNT],
            roughness: vec![roughness; VERTEX_COUNT],
            parent_coverage: vec![coverage; overlay::PARENT_COUNT],
            parent_albedo: vec![[albedo; 3]; overlay::PARENT_COUNT],
            parent_roughness: vec![roughness; overlay::PARENT_COUNT],
        }
    }

    #[cfg(feature = "memory-report")]
    #[test]
    fn the_memory_report_counts_each_geometry_once_by_its_arrays() {
        let key = SurfaceKey::new(4, 0, 0, 0, 0).unwrap();
        let g = Arc::new(TileGeometry::new(&tile(key)));
        // Two f64 triples, three f32 triples, and two f32 per vertex.
        let per_vertex = 2 * 24 + 3 * 12 + 2 * 4;
        let one = (std::mem::size_of::<TileGeometry>() + VERTEX_COUNT * per_vertex) as u64;
        assert_eq!(g.heap_bytes(), one);

        // One geometry as a tile's state and its base, and as a second
        // tile's: counted once.
        let mut s = Surfaces::new();
        s.complete(key, Ok(Some(Arc::clone(&g))));
        let other = SurfaceKey::new(4, 1, 0, 0, 0).unwrap();
        s.complete(other, Ok(Some(Arc::clone(&g))));
        let m = s.memory();
        assert_eq!((m.ready, m.geometries, m.composited), (2, 1, 0));
        assert_eq!(m.geometry_bytes, one);

        // A shown overlay adds the composited copy and the overlay tile.
        let third = SurfaceKey::new(4, 2, 0, 0, 0).unwrap();
        let o = overlay_tile(third, 255, 128, 64);
        let o_bytes = (std::mem::size_of::<OverlayTile>()
            + VERTEX_COUNT * 5
            + overlay::PARENT_COUNT * 5) as u64;
        assert_eq!(overlay_heap_bytes(&o), o_bytes);
        let base = Arc::new(TileGeometry::new(&tile(third)));
        s.complete_layers(
            third,
            Ok(DecodedTile {
                bases: vec![(0, base)],
                overlays: vec![(1, Arc::new(o))],
                ..DecodedTile::default()
            }),
            false,
        );
        let m = s.memory();
        assert_eq!((m.geometries, m.composited), (3, 1));
        assert_eq!((m.geometry_bytes, m.composited_bytes), (3 * one, one));
        assert_eq!(m.overlay_bytes, o_bytes);
        assert_eq!(
            (m.drawn, m.drawn_geometry_bytes, m.drawn_overlay_bytes),
            (0, 0, 0)
        );
    }

    fn overlay_bytes(o: &OverlayTile) -> Vec<u8> {
        overlay::encode(o, gx_core::matter::Compression::None)
    }

    #[test]
    fn overlays_composite_in_layer_order_and_hide() {
        let key = SurfaceKey::new(4, 0, 0, 0, 0).unwrap();
        let base = gx_core::surface::encode(&tile(key), gx_core::matter::Compression::None);
        let first = overlay_tile(key, 255, 255, 255);
        let second = overlay_tile(key, 128, 0, 0);
        let overlays: BTreeSet<usize> = [1, 2].into();
        let decode = |sections: &[Vec<u8>], layers: &[usize]| {
            decode_tile(&key, sections, layers, &overlays).unwrap()
        };
        let mut s = Surfaces::new();
        // No overlay: the base material.
        let d = decode(std::slice::from_ref(&base), &[0]);
        let plain = d.base().cloned().unwrap();
        s.complete_layers(key, Ok(d), true);
        let drawn = Arc::clone(s.drawn_geometry(&key).unwrap());
        assert!(Arc::ptr_eq(&drawn, &plain));
        // The second overlay arrives first: drawn with it alone.
        let d = decode(&[base.clone(), overlay_bytes(&second)], &[0, 2]);
        s.complete_layers(key, Ok(d), true);
        let one = Arc::clone(s.drawn_geometry(&key).unwrap());
        assert!(!Arc::ptr_eq(&one, &drawn), "a new material is a new copy");
        assert_eq!(one.positions, plain.positions);
        let v = TILE_VERTICES + 1;
        let c = overlay::unit(128);
        let want = (1.0 - c) * plain.albedo[v][0];
        assert_eq!(one.albedo[v][0], want);
        // Partial coverage: the uncovered row keeps the base.
        assert_eq!(one.albedo[1], plain.albedo[1]);
        assert_eq!(one.roughness[1], plain.roughness[1]);
        // Both, in layer order: the first replaces, the second blends.
        let d = decode(
            &[base.clone(), overlay_bytes(&first), overlay_bytes(&second)],
            &[0, 1, 2],
        );
        s.complete_layers(key, Ok(d), false);
        let two = Arc::clone(s.drawn_geometry(&key).unwrap());
        assert_eq!(two.albedo[v], [1.0 - c; 3]);
        assert_eq!(two.roughness[v], 1.0 - c);
        let parent = composite_parent(&plain, &[&first, &second]);
        assert_eq!(parent[1], ([1.0 - c; 3], 1.0 - c));
        // Hiding the second leaves the first; hiding both restores the base.
        s.set_hidden_layers([2].into());
        let hidden = Arc::clone(s.drawn_geometry(&key).unwrap());
        assert_eq!(hidden.albedo[v], [1.0; 3]);
        s.set_hidden_layers([1, 2].into());
        let restored = s.drawn_geometry(&key).unwrap();
        assert_eq!(restored.albedo, plain.albedo);
        assert_eq!(restored.roughness, plain.roughness);
    }

    #[test]
    fn a_stale_partial_completion_never_replaces_a_fuller_tile() {
        let key = SurfaceKey::new(4, 0, 0, 0, 0).unwrap();
        let base = gx_core::surface::encode(&tile(key), gx_core::matter::Compression::None);
        let first = overlay_tile(key, 255, 255, 255);
        let second = overlay_tile(key, 128, 0, 0);
        let overlays: BTreeSet<usize> = [1, 2].into();
        let decode = |sections: &[Vec<u8>], layers: &[usize]| {
            decode_tile(&key, sections, layers, &overlays).unwrap()
        };
        let v = TILE_VERTICES + 1;
        let mut s = Surfaces::new();
        // The final completion lands before the partial one sent earlier.
        let full = decode(
            &[base.clone(), overlay_bytes(&first), overlay_bytes(&second)],
            &[0, 1, 2],
        );
        s.complete_layers(key, Ok(full), false);
        let done = Arc::clone(s.drawn_geometry(&key).unwrap());
        s.complete_layers(key, Ok(decode(std::slice::from_ref(&base), &[0])), true);
        assert!(Arc::ptr_eq(s.drawn_geometry(&key).unwrap(), &done));
        assert!(!s.pending.contains(&key));
        // Two partials out of order: the one with fewer overlays is
        // ignored, and the final completion still applies.
        let mut s = Surfaces::new();
        let one = decode(&[base.clone(), overlay_bytes(&second)], &[0, 2]);
        s.complete_layers(key, Ok(one), true);
        let with_one = Arc::clone(s.drawn_geometry(&key).unwrap());
        s.complete_layers(key, Ok(decode(std::slice::from_ref(&base), &[0])), true);
        assert!(Arc::ptr_eq(s.drawn_geometry(&key).unwrap(), &with_one));
        let full = decode(
            &[base.clone(), overlay_bytes(&first), overlay_bytes(&second)],
            &[0, 1, 2],
        );
        s.complete_layers(key, Ok(full), false);
        assert_eq!(s.drawn_geometry(&key).unwrap().albedo[v], done.albedo[v]);
    }

    /// A frame whose six roots have arrived plainly and whose face-0 root
    /// is split from a camera at its box center as soon as its children
    /// allow; every child arrives at `t = 0` with its surface tile and
    /// overlay layer 1 still to come.
    fn waiting_children(required: BTreeSet<usize>, hidden: BTreeSet<usize>) -> Surfaces {
        let mut s = Surfaces::new();
        s.set_required_layers(required);
        s.set_hidden_layers(hidden);
        for face in 0..SURFACE_FACES {
            let key = SurfaceKey::new(4, face, 0, 0, 0).unwrap();
            s.complete(key, Ok(Some(Arc::new(TileGeometry::new(&tile(key))))));
        }
        let root = SurfaceKey::new(4, 0, 0, 0, 0).unwrap();
        for c in root.children().unwrap() {
            s.complete_layers(c, Ok(waiting_child(c, None)), true);
        }
        s
    }

    /// Child `c` with its surface tile, overlay layer 1 still to come
    /// (`None`) or failed with an invalid section (`Some(reason)`).
    fn waiting_child(c: SurfaceKey, failed: Option<&str>) -> DecodedTile {
        DecodedTile {
            bases: vec![(0, Arc::new(TileGeometry::new(&tile(c))))],
            failed: failed.into_iter().map(|r| (1, r.to_string())).collect(),
            waiting: if failed.is_some() { vec![2] } else { vec![1] },
            ..DecodedTile::default()
        }
    }

    /// The levels drawn from the face-0 root's box center at time `now`.
    fn levels_at(s: &mut Surfaces, now: f64) -> BTreeSet<u8> {
        s.now = now;
        let root = SurfaceKey::new(4, 0, 0, 0, 0).unwrap();
        let (lo, hi) = s.drawn_geometry(&root).unwrap().bounds;
        let cam: [f64; 3] = core::array::from_fn(|n| 0.5 * (lo[n] + hi[n]));
        let len = (cam[0] * cam[0] + cam[1] * cam[1] + cam[2] * cam[2]).sqrt();
        let forward = cam.map(|v| v / len);
        let mut drawn = Vec::new();
        s.select_in_frame(
            4,
            cam,
            forward,
            focal_px(720.0, FIELD_OF_VIEW_Y),
            0.9,
            &mut drawn,
        );
        (drawn.iter())
            .filter(|t| t.geometry.key.face == 0)
            .map(|t| t.geometry.key.level)
            .collect()
    }

    #[test]
    fn a_required_overlay_keeps_the_parent_until_it_arrives() {
        let mut s = waiting_children([1].into(), BTreeSet::new());
        assert_eq!(levels_at(&mut s, 0.0), [0].into());
        assert_eq!(
            levels_at(&mut s, 60.0),
            [0].into(),
            "required: no time limit"
        );
        let root = SurfaceKey::new(4, 0, 0, 0, 0).unwrap();
        for c in root.children().unwrap() {
            let d = DecodedTile {
                bases: vec![(0, Arc::new(TileGeometry::new(&tile(c))))],
                ..DecodedTile::default()
            };
            s.complete_layers(c, Ok(d), false);
        }
        assert_eq!(levels_at(&mut s, 60.0), [1].into());
    }

    #[test]
    fn an_overlay_not_required_holds_the_split_at_most_the_wait() {
        let mut s = waiting_children(BTreeSet::new(), BTreeSet::new());
        assert_eq!(levels_at(&mut s, 0.0), [0].into());
        assert_eq!(levels_at(&mut s, OVERLAY_WAIT_SECONDS * 0.99), [0].into());
        // A still camera: the selection runs again and splits.
        assert_eq!(levels_at(&mut s, OVERLAY_WAIT_SECONDS), [1].into());
    }

    #[test]
    fn a_failed_required_overlay_releases_the_split() {
        let mut s = waiting_children([1].into(), BTreeSet::new());
        assert_eq!(levels_at(&mut s, 0.0), [0].into());
        let root = SurfaceKey::new(4, 0, 0, 0, 0).unwrap();
        for c in root.children().unwrap() {
            // Layer 1's section is invalid; layer 2, not required, is
            // still to come and has waited long enough.
            s.complete_layers(c, Ok(waiting_child(c, Some("code 1: bad"))), true);
        }
        assert_eq!(levels_at(&mut s, OVERLAY_WAIT_SECONDS), [1].into());
        let c = root.children().unwrap()[0];
        assert!(s.layers[&c].failed.contains(&1));
    }

    #[test]
    fn a_hidden_required_overlay_is_not_waited_for() {
        let mut s = waiting_children([1].into(), [1].into());
        assert_eq!(levels_at(&mut s, 0.0), [1].into());
    }

    #[test]
    fn level_zero_tiles_draw_with_no_overlay_yet() {
        let mut s = Surfaces::new();
        s.set_required_layers([1].into());
        for face in 0..SURFACE_FACES {
            let key = SurfaceKey::new(4, face, 0, 0, 0).unwrap();
            let d = DecodedTile {
                bases: vec![(0, Arc::new(TileGeometry::new(&tile(key))))],
                waiting: vec![1],
                ..DecodedTile::default()
            };
            s.complete_layers(key, Ok(d), true);
        }
        assert!(s.has_surface(4));
        assert_eq!(levels_at(&mut s, 0.0), [0].into());
    }

    /// A tile whose heights are a smooth function of the vertex direction,
    /// with parent heights the same function at the parent's level (the
    /// same direction, so the same values at shared vertices).
    fn tile(key: SurfaceKey) -> SurfaceTile {
        tile_with_floor(key, -400.0)
    }

    /// [`tile`] declaring `floor` as the frame's lowest height.
    fn tile_with_floor(key: SurfaceKey, floor: f64) -> SurfaceTile {
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
            height_floor: floor,
            heights,
            parent_heights,
            albedo: vec![[76, 71, 64]; VERTEX_COUNT],
            roughness: vec![230; VERTEX_COUNT],
            temperature: vec![288.0; VERTEX_COUNT],
        }
    }

    #[test]
    fn eviction_keeps_what_the_selection_uses() {
        // Low camera looking level: some split tiles have every child
        // hidden. Eviction at its most eager (keep nothing it may drop)
        // must not change what is drawn from one selection to the next.
        let (a, c) = (6_378_137.0, 6_356_752.3);
        let d =
            gx_core::surface::vertex_direction(&SurfaceKey::new(4, 0, 0, 0, 0).unwrap(), 32, 32);
        let (p, nrm) = gx_core::surface::ellipsoid_point(d, a, c);
        let cam: [f64; 3] = core::array::from_fn(|k| p[k] + 2000.0 * nrm[k]);
        let up_tangent =
            unit_cross(nrm, unit_cross([0.0, 0.0, 1.0], nrm).map(f64::from)).map(f64::from);
        let mut s = Surfaces::new();
        let first = settle(&mut s, cam, up_tangent, 11);
        let keys =
            |d: &[TileDrawInfo]| -> Vec<SurfaceKey> { d.iter().map(|t| t.geometry.key).collect() };
        let want = keys(&first);
        for round in 0..6 {
            s.evict(0);
            let mut drawn = Vec::new();
            s.select_in_frame(
                4,
                cam,
                up_tangent,
                focal_px(720.0, FIELD_OF_VIEW_Y),
                0.9,
                &mut drawn,
            );
            assert_eq!(keys(&drawn), want, "round {round}: what is drawn changed");
            // Serve whatever was asked for again, as the hub would.
            for key in s.take_requests() {
                let outcome = if key.level <= 11 {
                    Ok(Some(Arc::new(TileGeometry::new(&tile(key)))))
                } else {
                    Ok(None)
                };
                s.complete(key, outcome);
            }
        }
    }

    #[test]
    fn hiding_follows_the_declared_floor() {
        // The same heights, low camera looking level: with the floor the
        // compiler declares (just under the heights) tiles past the horizon
        // are hidden; declared far deeper, the occluding ball shrinks and
        // they are not. Nothing but the declared value differs.
        let (a, c) = (6_378_137.0, 6_356_752.3);
        let d =
            gx_core::surface::vertex_direction(&SurfaceKey::new(4, 0, 0, 0, 0).unwrap(), 32, 32);
        let (p, nrm) = gx_core::surface::ellipsoid_point(d, a, c);
        let cam: [f64; 3] = core::array::from_fn(|k| p[k] + 2000.0 * nrm[k]);
        let up_tangent =
            unit_cross(nrm, unit_cross([0.0, 0.0, 1.0], nrm).map(f64::from)).map(f64::from);
        let shallow = settle_with_floor(&mut Surfaces::new(), cam, up_tangent, 9, -400.0);
        let deep = settle_with_floor(&mut Surfaces::new(), cam, up_tangent, 9, -3.0e6);
        assert!(
            deep.len() > shallow.len(),
            "{} tiles with a deep floor, {} with the declared one",
            deep.len(),
            shallow.len()
        );
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
        settle_with_floor(s, cam, forward, max_level, -400.0)
    }

    fn settle_with_floor(
        s: &mut Surfaces,
        cam: [f64; 3],
        forward: [f64; 3],
        max_level: u8,
        floor: f64,
    ) -> Vec<TileDrawInfo> {
        settle_full(
            s,
            cam,
            forward,
            max_level,
            floor,
            focal_px(720.0, FIELD_OF_VIEW_Y),
        )
    }

    fn settle_full(
        s: &mut Surfaces,
        cam: [f64; 3],
        forward: [f64; 3],
        max_level: u8,
        floor: f64,
        focal: f64,
    ) -> Vec<TileDrawInfo> {
        let cone = 0.9;
        for face in 0..SURFACE_FACES {
            s.want(SurfaceKey::new(4, face, 0, 0, 0).unwrap(), 1.0);
        }
        for _ in 0..64 {
            for key in s.take_requests() {
                let outcome = if key.level <= max_level {
                    Ok(Some(Arc::new(TileGeometry::new(&tile_with_floor(
                        key, floor,
                    )))))
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
    /// along a low flight over rough synthetic heights, every boundary
    /// vertex of every drawn tile, morphed as the GPU morphs it, lies on the
    /// boundary of another drawn tile (within a tenth of a millimeter), so
    /// tiles meet with no gap, whatever their levels.
    #[test]
    fn drawn_tiles_meet_without_gaps() {
        // A 720 px view (the morph bound sets the ranges) and a 1620 px one
        // (the pixel bound does), along a face edge and over a face center,
        // where tiles are largest.
        for focal in [
            focal_px(720.0, FIELD_OF_VIEW_Y),
            focal_px(1620.0, FIELD_OF_VIEW_Y),
        ] {
            for (face, i0, j) in [(4u8, 20i64, 9i64), (0, 26, 32)] {
                meet_without_gaps(focal, face, i0, j);
            }
        }
    }

    fn meet_without_gaps(focal: f64, face: u8, i0: i64, j0: i64) {
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
                &SurfaceKey::new(4, face, 0, 0, 0).unwrap(),
                i0 + step,
                j0,
            );
            let (p, nrm) = gx_core::surface::ellipsoid_point(d, a, c);
            let cam: [f64; 3] = core::array::from_fn(|k| p[k] + 3000.0 * nrm[k]);
            let tangent = unit_cross([0.0, 0.0, 1.0], nrm).map(f64::from);
            let forward: [f64; 3] = core::array::from_fn(|k| 0.9 * tangent[k] - 0.42 * nrm[k]);
            let drawn = settle_full(&mut s, cam, forward, 11, -400.0, focal);
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

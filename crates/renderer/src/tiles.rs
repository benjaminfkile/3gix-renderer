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
//! A tile's shape ([`TileShape`]) is held once, behind an `Arc`, by its
//! base and every composite of it; a composite owns only its material
//! (`rendering.md` section 7 items 6 and 7).
//!
//! An arrived tile is held in one of two forms (`rendering.md` sections
//! 3.5 and 4): built, with its geometry, or compact, with only its
//! surface section bytes, its overlays, and a [`TileSummary`]. A tile is
//! built while it is drawn, while it is a level-0 tile, or while it is a
//! child of a tile the selection walks within that tile's [`range`];
//! every other arrived tile goes back to compact [`BUILT_GRACE_SECONDS`]
//! after it was last wanted built. A compact tile wanted built is handed
//! to the fetcher ([`Surfaces::take_builds`]), which builds it off the
//! frame thread ([`build_tile`]) and hands the geometry back
//! ([`Surfaces::complete_build`]). A child counts as arrived for a split
//! only once it is built, so its parent keeps drawing until then.

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

/// One tile ready to draw (`rendering.md` section 7 items 3 and 6): its
/// shape, shared through an `Arc` by the base and every composite of it,
/// and its own material per vertex (row major, `j` outer). The shape's
/// fields read through the geometry ([`std::ops::Deref`]), so
/// `geometry.positions` is the shape's.
#[derive(Clone, Debug, PartialEq)]
pub struct TileGeometry {
    /// Everything compositing leaves alone, built once per tile.
    pub tile_shape: Arc<TileShape>,
    /// Albedo per vertex, three bands: the base's, or the composite's.
    pub albedo: Vec<[f32; 3]>,
    /// Roughness per vertex: the base's, or the composite's.
    pub roughness: Vec<f32>,
}

/// A tile's shape: positions and their morph targets in `f64` frame
/// coordinates, normals, temperature, and what the selection reads, per
/// vertex (row major, `j` outer). Held once per tile however many
/// composites of it there are.
#[derive(Clone, Debug, PartialEq)]
pub struct TileShape {
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

impl std::ops::Deref for TileGeometry {
    type Target = TileShape;

    /// The tile's shared shape.
    fn deref(&self) -> &TileShape {
        &self.tile_shape
    }
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
        let tile_shape = Arc::new(TileShape {
            key,
            positions,
            targets,
            normals,
            target_normals,
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
        });
        TileGeometry {
            tile_shape,
            albedo,
            roughness,
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

/// What the selection needs of a tile without its geometry: taken from
/// the geometry the first time the tile is built and kept while the tile
/// is compact (module docs). A tile with a summary has a surface.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct TileSummary {
    /// The geometry's [`TileShape::bounds`].
    pub bounds: ([f64; 3], [f64; 3]),
    /// The geometry's [`TileShape::edge`].
    pub edge: f64,
    /// The geometry's [`TileShape::max_edge`].
    pub max_edge: f64,
    /// The geometry's [`TileShape::occluder`].
    pub occluder: f64,
    /// The frame's surface as the tile declares it.
    pub shape: SurfaceShape,
}

impl TileSummary {
    /// The summary of `g`.
    pub fn of(g: &TileGeometry) -> TileSummary {
        TileSummary {
            bounds: g.bounds,
            edge: g.edge,
            max_edge: g.max_edge,
            occluder: g.occluder,
            shape: g.shape,
        }
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
    /// The section bytes of each of [`DecodedTile::bases`], in the same
    /// order: what a compact tile keeps to be built again.
    pub sections: Vec<Arc<[u8]>>,
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
    decode_sections(key, sections, layers, overlay_layers, |b| {
        Arc::from(b.as_ref())
    })
}

/// [`decode_tile`] over sections already shared: a compact tile keeps the
/// fetcher's own `Arc` of each base section.
pub fn decode_shared_tile(
    key: &SurfaceKey,
    sections: &[Arc<[u8]>],
    layers: &[usize],
    overlay_layers: &BTreeSet<usize>,
) -> Result<DecodedTile, String> {
    decode_sections(key, sections, layers, overlay_layers, Arc::clone)
}

fn decode_sections<S: AsRef<[u8]>>(
    key: &SurfaceKey,
    sections: &[S],
    layers: &[usize],
    overlay_layers: &BTreeSet<usize>,
    share: impl Fn(&S) -> Arc<[u8]>,
) -> Result<DecodedTile, String> {
    let mut out = DecodedTile::default();
    for (n, section) in sections.iter().enumerate() {
        let layer = layers.get(n).copied().unwrap_or(n);
        let bytes = section.as_ref();
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
            Ok(Tile::Surface(t)) => {
                out.bases.push((layer, Arc::new(geometry(&t))));
                out.sections.push(share(section));
            }
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
/// ([`composite_parent`]). The shape is the base's own `Arc`: only the
/// material arrays are new.
pub fn composite_tile(base: &TileGeometry, overlays: &[&OverlayTile]) -> TileGeometry {
    if overlays.is_empty() {
        return base.clone();
    }
    let n = base.albedo.len();
    let mut albedo = Vec::with_capacity(n);
    let mut roughness = Vec::with_capacity(n);
    let mut samples = Vec::with_capacity(overlays.len());
    for v in 0..n {
        samples.clear();
        samples.extend(overlays.iter().map(|o| sample(o, v)));
        let (a, r) = overlay::composite(base.albedo[v], base.roughness[v], &samples);
        albedo.push(a);
        roughness.push(r);
    }
    TileGeometry {
        tile_shape: Arc::clone(&base.tile_shape),
        albedo,
        roughness,
    }
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
    /// Built: the geometry drawn.
    Ready(Arc<TileGeometry>),
    /// Arrived with a surface and not built (module docs).
    Compact(TileSummary),
    Empty,
    Gone,
}

/// How long, seconds, a built tile that is neither drawn nor wanted built
/// stays built before it goes back to compact (module docs, the
/// selection's clock). Without it a camera resting near a split distance
/// would drop and rebuild the same tiles every frame.
pub const BUILT_GRACE_SECONDS: f64 = 2.0;

/// A compact tile to build ([`Surfaces::take_builds`]): its base sections
/// and their layer indices, in layer order, and its projected size,
/// pixels, as priority (largest first).
#[derive(Clone, Debug)]
pub struct BuildRequest {
    /// The tile.
    pub key: SurfaceKey,
    /// The base layers' section bytes.
    pub sections: Vec<Arc<[u8]>>,
    /// The layer index of each section.
    pub layers: Vec<usize>,
    /// Projected size, pixels.
    pub priority: f64,
    /// The overlays that had arrived for the tile, with their layer
    /// indices, in layer order.
    pub overlays: Vec<(usize, Arc<OverlayTile>)>,
    /// The hidden layers when the request was made: the drawn geometry is
    /// composited for these.
    pub hidden: BTreeSet<usize>,
}

/// A finished [`build_tile`]: the geometry of each base with its layer
/// index, and the geometry to draw composited for the hidden set the
/// request was made with.
#[derive(Clone, Debug, PartialEq)]
pub struct BuiltTile {
    /// The built bases.
    pub bases: Vec<(usize, Arc<TileGeometry>)>,
    /// The hidden layers [`BuiltTile::drawn`] was composited for.
    pub hidden: BTreeSet<usize>,
    /// How many overlays the tile held when the request was made.
    pub overlays: usize,
    /// The first shown base with the shown overlays composited; `None`
    /// when every base is hidden.
    pub drawn: Option<Arc<TileGeometry>>,
}

/// What [`build_tile`] returns: the built tile, or why it is invalid.
pub type BuildOutcome = Result<BuiltTile, String>;

/// Builds a [`BuildRequest`] off the frame thread (what the fetchers
/// run): each base's geometry, and the geometry to draw composited with
/// the overlays shown under the request's hidden set, so the frame thread
/// only stores the result ([`Surfaces::complete_build`]).
pub fn build_tile(req: &BuildRequest) -> BuildOutcome {
    let bases = decode_shared_tile(&req.key, &req.sections, &req.layers, &BTreeSet::new())?.bases;
    let drawn = composite_shown(&bases, &req.overlays, &req.hidden);
    Ok(BuiltTile {
        bases,
        hidden: req.hidden.clone(),
        overlays: req.overlays.len(),
        drawn,
    })
}

/// The first base not in `hidden` with the overlays not in `hidden`
/// composited over it, or that base itself when no overlay is shown;
/// `None` when every base is hidden.
fn composite_shown(
    bases: &[(usize, Arc<TileGeometry>)],
    overlays: &[(usize, Arc<OverlayTile>)],
    hidden: &BTreeSet<usize>,
) -> Option<Arc<TileGeometry>> {
    let (_, g) = bases.iter().find(|(l, _)| !hidden.contains(l))?;
    let shown: Vec<&OverlayTile> = (overlays.iter())
        .filter(|(l, _)| !hidden.contains(l))
        .map(|(_, o)| o.as_ref())
        .collect();
    Some(if shown.is_empty() {
        Arc::clone(g)
    } else {
        Arc::new(composite_tile(g, &shown))
    })
}

/// Sorts `builds` largest projected size first, ties by key, so the
/// order is the same on every run.
pub fn sort_builds(builds: &mut [BuildRequest]) {
    builds.sort_by(|a, b| {
        b.priority
            .total_cmp(&a.priority)
            .then_with(|| a.key.cmp(&b.key))
    });
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
    /// The tiles the latest selection wants built (module docs).
    want_built: std::collections::HashSet<SurfaceKey>,
    /// Compact tiles wanted built and not yet handed to the fetcher, with
    /// their projected size, pixels.
    builds: Vec<(SurfaceKey, f64)>,
    /// Compact tiles wanted built whose build has not come back.
    building: std::collections::HashSet<SurfaceKey>,
    /// Over the memory budget: children are asked for only within the
    /// split distance, not out to [`PREFETCH`] times it (`crate::budget`).
    hold_ahead: bool,
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
                Some(State::Compact(s)) => Some(s.shape),
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
            self.tiles.insert(*key, layers.state(&self.hidden));
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

    /// The compact tiles wanted built since the last call, largest
    /// projected size first: the fetcher builds each off the frame thread
    /// ([`build_tile`]) and hands it back to [`Surfaces::complete_build`].
    pub fn take_builds(&mut self) -> Vec<BuildRequest> {
        let mut out: Vec<BuildRequest> = std::mem::take(&mut self.builds)
            .into_iter()
            .filter_map(|(key, priority)| {
                let l = self.layers.get(&key)?;
                let (layers, sections) = l
                    .bases
                    .iter()
                    .filter_map(|b| Some((b.layer, Arc::clone(b.section.as_ref()?))))
                    .unzip();
                Some(BuildRequest {
                    key,
                    sections,
                    layers,
                    priority,
                    overlays: l.overlays.clone(),
                    hidden: self.hidden.clone(),
                })
            })
            .collect();
        sort_builds(&mut out);
        out
    }

    /// Applies a finished build ([`build_tile`]): the tile is built again
    /// and drawn from the next selection with the geometry the build
    /// composited. Only when the hidden set or the tile's overlays changed
    /// since the request is it composited again here, as a layer toggle
    /// does. A build for a tile no longer held, or already built again by
    /// a later arrival, is dropped.
    pub fn complete_build(&mut self, key: SurfaceKey, outcome: BuildOutcome) {
        self.building.remove(&key);
        let Some(l) = self.layers.get_mut(&key) else {
            return;
        };
        if l.built() {
            return;
        }
        match outcome {
            Ok(built) => {
                for b in &mut l.bases {
                    if let Some((_, g)) = built.bases.iter().find(|(layer, _)| *layer == b.layer) {
                        b.geometry = Some(Arc::clone(g));
                    }
                }
                let same = built.hidden == self.hidden && l.overlays.len() == built.overlays;
                let state = match (same, built.drawn) {
                    (true, Some(g)) => State::Ready(g),
                    (true, None) => State::Empty,
                    (false, _) => l.state(&self.hidden),
                };
                self.tiles.insert(key, state);
            }
            Err(reason) => {
                tracing::warn!(key = %key, "surface tile is invalid: {reason}");
                self.layers.remove(&key);
                self.pending.remove(&key);
                self.tiles.insert(key, State::Gone);
            }
        }
    }

    /// Builds every compact tile wanted built on this thread: for tests
    /// and tools that run the selection without a fetcher.
    pub fn build_now(&mut self) {
        for req in self.take_builds() {
            let outcome = build_tile(&req);
            self.complete_build(req.key, outcome);
        }
    }

    /// Marks `key` wanted built by this selection and asks for its build
    /// when it is compact (module docs).
    fn want_built(&mut self, key: SurfaceKey, priority: f64) {
        self.want_built.insert(key);
        if let Some(l) = self.layers.get_mut(&key) {
            l.wanted_at = self.now;
        }
        if matches!(self.tiles.get(&key), Some(State::Compact(_))) && self.building.insert(key) {
            self.builds.push((key, priority));
        }
    }

    /// Returns every built tile that is not a level-0 tile and has not
    /// been wanted built for [`BUILT_GRACE_SECONDS`] to compact.
    fn compact_unwanted(&mut self) {
        let now = self.now;
        for (key, l) in &mut self.layers {
            if key.level == 0 || !l.built() || !l.can_compact() {
                continue;
            }
            if now - l.wanted_at >= BUILT_GRACE_SECONDS {
                l.compact();
                self.tiles.insert(*key, l.state(&self.hidden));
            }
        }
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
                sections,
                overlays,
                failed,
                waiting,
            }) if !bases.is_empty() => {
                let arrived_at = held.as_ref().map_or(self.now, |h| h.arrived_at);
                // Kept built only when wanted built now or within the
                // grace time (module docs); level-0 tiles always are.
                let wanted_at = if key.level == 0 || self.want_built.contains(&key) {
                    self.now
                } else {
                    held.as_ref().map_or(f64::NEG_INFINITY, |h| h.wanted_at)
                };
                let keep = self.now - wanted_at < BUILT_GRACE_SECONDS;
                let mut sections = sections.into_iter();
                let bases = bases
                    .into_iter()
                    .map(|(layer, g)| Base {
                        layer,
                        section: sections.next(),
                        summary: TileSummary::of(&g),
                        geometry: Some(g),
                    })
                    .collect();
                let mut known = held.map(|h| h.failed).unwrap_or_default();
                for (layer, reason) in failed {
                    if known.insert(layer) {
                        tracing::warn!(key = %key, layer, "overlay tile is invalid: {reason}");
                    }
                }
                let waiting = if partial { waiting } else { Vec::new() };
                let mut layers = TileLayers {
                    bases,
                    overlays,
                    failed: known,
                    waiting,
                    arrived_at,
                    wanted_at,
                    used_at: self.now,
                };
                if !keep && layers.can_compact() {
                    layers.compact();
                }
                let state = layers.state(&self.hidden);
                self.layers.insert(key, layers);
                state
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
                let any = roots
                    .iter()
                    .any(|s| matches!(s, Some(State::Ready(_) | State::Compact(_))));
                self.has_surface.insert(frame_id, any);
            }
        }
    }

    /// Whether `key` has arrived as far as a split needs (`rendering.md`
    /// section 7 item 6): it is empty, or it is built and
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
        self.want_built.clear();
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
        for k in &self.used {
            if let Some(l) = self.layers.get_mut(k) {
                l.used_at = now;
            }
        }
        self.compact_unwanted();
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
                // A child's projected edge: half the parent's.
                let px = 0.5 * geo.edge * focal / d.max(f64::MIN_POSITIVE);
                if d < r * if self.hold_ahead { 1.0 } else { PREFETCH } {
                    for c in ch {
                        self.want(c, px);
                    }
                }
                // Within the split distance the children are wanted
                // built, split or waiting on them (module docs).
                if d < r {
                    for c in ch {
                        self.want_built(c, px);
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
            self.want_built(key, f64::INFINITY);
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
                            || self.building.contains(c)
                    })
                })
            })
            .collect()
    }

    /// Whether every root and every tile the latest selection would split
    /// toward has arrived and is built: nothing requested or wanted built
    /// is outstanding.
    pub fn settled(&self) -> bool {
        self.requests.is_empty()
            && self.builds.is_empty()
            && self.building.is_empty()
            && !self.tiles.values().any(|s| matches!(s, State::Requested))
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

    /// Holds asking ahead ([`PREFETCH`]) while `hold` is `true`: the
    /// memory budget is exceeded (`crate::budget`).
    pub fn set_hold_ahead(&mut self, hold: bool) {
        self.hold_ahead = hold;
    }

    /// Bytes every held tile counts against the memory budget
    /// (`crate::budget`): each entry's key and state, and each arrived
    /// tile's sections, overlays, and geometry.
    pub fn held_bytes(&self) -> u64 {
        let entry = std::mem::size_of::<(SurfaceKey, State)>() as u64;
        let mut n = self.tiles.len() as u64 * entry;
        for (k, l) in &self.layers {
            if let Some(s) = self.tiles.get(k) {
                n += l.bytes(s);
            }
        }
        n
    }

    /// The drawn tiles and every ancestor the selection walks to reach
    /// them: never evicted, even over the budget.
    fn drawn_and_ancestors(&self) -> std::collections::HashSet<SurfaceKey> {
        let mut out = std::collections::HashSet::new();
        for t in &self.drawn {
            let mut k = Some(t.geometry.key);
            while let Some(key) = k {
                if !out.insert(key) {
                    break;
                }
                k = key.parent();
            }
        }
        out
    }

    /// The first step of eviction over the budget (`crate::budget`):
    /// while `over` bytes remain, built tiles that are not level-0, not
    /// drawn or walked to, and not wanted built go compact, least recently
    /// used first, the deepest on a tie. Takes the bytes freed off `over`
    /// and returns how many tiles went compact.
    pub fn compact_over(&mut self, over: &mut u64) -> usize {
        if *over == 0 {
            return 0;
        }
        let keep = self.drawn_and_ancestors();
        let mut items: Vec<crate::budget::Candidate<SurfaceKey>> = Vec::new();
        for (k, l) in &self.layers {
            if k.level == 0
                || !l.built()
                || !l.can_compact()
                || keep.contains(k)
                || self.want_built.contains(k)
            {
                continue;
            }
            items.push(crate::budget::Candidate {
                last_used: l.used_at,
                level: k.level,
                key: *k,
                bytes: 0,
            });
        }
        crate::budget::sort_candidates(&mut items);
        let mut done = 0;
        for c in items {
            if *over == 0 {
                break;
            }
            let Some(l) = self.layers.get_mut(&c.key) else {
                continue;
            };
            let before = self.tiles.get(&c.key).map_or(0, |s| l.bytes(s));
            l.compact();
            let state = l.state(&self.hidden);
            let after = l.bytes(&state);
            self.tiles.insert(c.key, state);
            *over = over.saturating_sub(before.saturating_sub(after));
            done += 1;
        }
        done
    }

    /// The tiles eviction over the budget may drop (`crate::budget`):
    /// every entry that is not level-0, not asked for, not drawn or walked
    /// to, and not built and wanted built, with its last use and bytes.
    pub fn eviction_candidates(&self) -> Vec<crate::budget::Candidate<SurfaceKey>> {
        let keep = self.drawn_and_ancestors();
        let entry = std::mem::size_of::<(SurfaceKey, State)>() as u64;
        let mut out = Vec::new();
        for (k, s) in &self.tiles {
            if k.level == 0 || matches!(s, State::Requested) || keep.contains(k) {
                continue;
            }
            let l = self.layers.get(k);
            if l.is_some_and(|l| l.built()) && self.want_built.contains(k) {
                continue;
            }
            out.push(crate::budget::Candidate {
                last_used: l.map_or(f64::NEG_INFINITY, |l| l.used_at),
                level: k.level,
                key: *k,
                bytes: entry + l.map_or(0, |l| l.bytes(s)),
            });
        }
        out
    }

    /// Drops `keys` from the cache (eviction over the budget).
    pub fn drop_tiles(&mut self, keys: &[SurfaceKey]) {
        for k in keys {
            self.tiles.remove(k);
            self.layers.remove(k);
            self.pending.remove(k);
            self.building.remove(k);
        }
        let tiles = &self.tiles;
        self.builds.retain(|(k, _)| tiles.contains_key(k));
    }
}

/// One non-empty base surface of a tile: its section, kept to build it
/// again, its summary, and its geometry while the tile is built.
#[derive(Clone, Debug)]
struct Base {
    layer: usize,
    /// The section bytes; `None` when the tile came without them, and
    /// then it is never made compact.
    section: Option<Arc<[u8]>>,
    summary: TileSummary,
    geometry: Option<Arc<TileGeometry>>,
}

/// A tile's base surfaces and the overlays that have arrived for it.
#[derive(Clone, Debug)]
struct TileLayers {
    /// Every non-empty base, in layer order.
    bases: Vec<Base>,
    overlays: Vec<(usize, Arc<OverlayTile>)>,
    /// Overlay layers whose section was invalid, already logged.
    failed: BTreeSet<usize>,
    /// Overlay layers still to come while the tile is pending.
    waiting: Vec<usize>,
    /// When the tile's surface tile arrived, on the selection's clock.
    arrived_at: f64,
    /// When the tile was last wanted built, on the selection's clock.
    wanted_at: f64,
    /// When a selection last looked at the tile, on the selection's
    /// clock: what eviction over the budget orders by.
    used_at: f64,
}

impl TileLayers {
    /// Bytes the tile holds (`crate::budget`): each base's section,
    /// summary, and geometry while built (material and shape), each
    /// overlay, and the drawn geometry's material when it is a composite.
    fn bytes(&self, state: &State) -> u64 {
        let mut n = std::mem::size_of::<TileLayers>() as u64;
        for b in &self.bases {
            n += std::mem::size_of::<Base>() as u64;
            n += b.section.as_ref().map_or(0, |s| s.len() as u64);
            if let Some(g) = &b.geometry {
                n += g.heap_bytes() + g.tile_shape.heap_bytes();
            }
        }
        n += self
            .overlays
            .iter()
            .map(|(_, o)| overlay_heap_bytes(o))
            .sum::<u64>();
        if let State::Ready(g) = state {
            let base = self
                .bases
                .iter()
                .any(|b| b.geometry.as_ref().is_some_and(|bg| Arc::ptr_eq(bg, g)));
            if !base {
                n += g.heap_bytes();
            }
        }
        n
    }

    /// The geometry to draw: the first shown base with the shown overlays
    /// composited, or that base itself when no overlay is shown; `None`
    /// when every base is hidden.
    fn state(&self, hidden: &BTreeSet<usize>) -> State {
        let Some(base) = self.bases.iter().find(|b| !hidden.contains(&b.layer)) else {
            return State::Empty;
        };
        let Some(g) = &base.geometry else {
            return State::Compact(base.summary);
        };
        match composite_shown(&[(base.layer, Arc::clone(g))], &self.overlays, hidden) {
            Some(g) => State::Ready(g),
            None => State::Empty,
        }
    }

    /// Whether any base holds its geometry.
    fn built(&self) -> bool {
        self.bases.iter().any(|b| b.geometry.is_some())
    }

    /// Whether every base keeps its section, so the tile can be built
    /// again.
    fn can_compact(&self) -> bool {
        self.bases.iter().all(|b| b.section.is_some())
    }

    /// Drops every base's geometry.
    fn compact(&mut self) {
        for b in &mut self.bases {
            b.geometry = None;
        }
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
    sections: &[Arc<[u8]>],
    layers: &[usize],
    overlays: &BTreeSet<usize>,
) -> Result<DecodedTile, String> {
    timed!(
        DECODE_TILE,
        decode_shared_tile(key, sections, layers, overlays)
    )
}

impl TileGeometry {
    /// Bytes this geometry's own material holds: the struct and each
    /// material array's capacity (the memory report,
    /// `docs/architecture.md`). The shape is counted apart
    /// ([`TileShape::heap_bytes`]), once however many geometries share it.
    pub fn heap_bytes(&self) -> u64 {
        (std::mem::size_of::<TileGeometry>() + cap(&self.albedo) + cap(&self.roughness)) as u64
    }
}

impl TileShape {
    /// Bytes this shape holds: the struct and each array's capacity (the
    /// memory report).
    pub fn heap_bytes(&self) -> u64 {
        (std::mem::size_of::<TileShape>()
            + cap(&self.positions)
            + cap(&self.targets)
            + cap(&self.normals)
            + cap(&self.target_normals)
            + cap(&self.temperature)) as u64
    }
}

fn cap<T>(v: &Vec<T>) -> usize {
    v.capacity() * std::mem::size_of::<T>()
}

/// Bytes an overlay tile holds: the struct and each array's capacity (the
/// memory report). A free function since the type is the core library's.
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
    /// distinct geometry's material, shape, and overlay tile, each `Arc`
    /// counted once: a base and its composites share one shape.
    pub fn memory(&self) -> crate::memory::SurfaceMemory {
        use std::collections::BTreeMap as Distinct;
        let mut m = crate::memory::SurfaceMemory {
            used: self.used.len(),
            drawn: self.drawn.len(),
            ..Default::default()
        };
        let ptr = |g: &Arc<TileGeometry>| Arc::as_ptr(g) as usize;
        let shape_ptr = |g: &TileGeometry| Arc::as_ptr(&g.tile_shape) as usize;
        let mut geometries: Distinct<usize, u64> = Distinct::new();
        let mut shapes: Distinct<usize, u64> = Distinct::new();
        let mut bases: BTreeSet<usize> = BTreeSet::new();
        let mut overlays: Distinct<usize, u64> = Distinct::new();
        let hold = |g: &Arc<TileGeometry>,
                    geometries: &mut Distinct<usize, u64>,
                    shapes: &mut Distinct<usize, u64>| {
            geometries.insert(ptr(g), g.heap_bytes());
            shapes.insert(shape_ptr(g), g.tile_shape.heap_bytes());
        };
        for state in self.tiles.values() {
            match state {
                State::Requested => m.requested += 1,
                State::Ready(g) => {
                    m.ready += 1;
                    hold(g, &mut geometries, &mut shapes);
                }
                State::Compact(_) => m.ready += 1,
                State::Empty => m.empty += 1,
                State::Gone => m.gone += 1,
            }
        }
        for (key, l) in &self.layers {
            for b in &l.bases {
                if let Some(g) = &b.geometry {
                    bases.insert(ptr(g));
                    hold(g, &mut geometries, &mut shapes);
                }
            }
            for (_, o) in &l.overlays {
                overlays.insert(Arc::as_ptr(o) as usize, overlay_heap_bytes(o));
            }
            // The tile's own bytes in its form: its sections, summaries,
            // and overlays, and when built its geometries.
            let mut bytes: u64 = l
                .bases
                .iter()
                .map(|b| {
                    (b.section.as_ref().map_or(0, |s| s.len()) + std::mem::size_of::<TileSummary>())
                        as u64
                })
                .sum::<u64>()
                + l.overlays
                    .iter()
                    .map(|(_, o)| overlay_heap_bytes(o))
                    .sum::<u64>();
            if l.built() {
                let mut own: Distinct<usize, u64> = Distinct::new();
                let mut own_shapes: Distinct<usize, u64> = Distinct::new();
                for g in l.bases.iter().filter_map(|b| b.geometry.as_ref()) {
                    hold(g, &mut own, &mut own_shapes);
                }
                if let Some(State::Ready(g)) = self.tiles.get(key) {
                    hold(g, &mut own, &mut own_shapes);
                }
                bytes += own.values().sum::<u64>() + own_shapes.values().sum::<u64>();
                m.built += 1;
                m.built_bytes += bytes;
            } else {
                m.compact += 1;
                m.compact_bytes += bytes;
            }
        }
        let mut drawn: Distinct<usize, u64> = Distinct::new();
        let mut drawn_shapes: Distinct<usize, u64> = Distinct::new();
        let mut drawn_overlays: Distinct<usize, u64> = Distinct::new();
        for t in &self.drawn {
            let g = &t.geometry;
            hold(g, &mut geometries, &mut shapes);
            hold(g, &mut drawn, &mut drawn_shapes);
            if let Some(l) = self.layers.get(&g.key) {
                for b in l.bases.iter().filter_map(|b| b.geometry.as_ref()) {
                    hold(b, &mut drawn, &mut drawn_shapes);
                }
                for (_, o) in &l.overlays {
                    drawn_overlays.insert(Arc::as_ptr(o) as usize, overlay_heap_bytes(o));
                }
            }
        }
        m.geometries = geometries.len();
        m.shapes = shapes.len();
        m.shape_bytes = shapes.values().sum();
        m.geometry_bytes = geometries.values().sum::<u64>() + m.shape_bytes;
        for (p, bytes) in &geometries {
            if !bases.contains(p) {
                m.composited += 1;
                m.composited_bytes += bytes;
            }
        }
        m.drawn_geometry_bytes = drawn.values().sum::<u64>() + drawn_shapes.values().sum::<u64>();
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
    fn the_memory_report_counts_each_shape_once_and_each_material_apart() {
        let key = SurfaceKey::new(4, 0, 0, 0, 0).unwrap();
        let g = Arc::new(TileGeometry::new(&tile(key)));
        // Material: one f32 triple and one f32 per vertex. Shape: two f64
        // triples, two f32 triples, and one f32 per vertex.
        let material = (std::mem::size_of::<TileGeometry>() + VERTEX_COUNT * (12 + 4)) as u64;
        let shape =
            (std::mem::size_of::<TileShape>() + VERTEX_COUNT * (2 * 24 + 2 * 12 + 4)) as u64;
        assert_eq!(
            (g.heap_bytes(), g.tile_shape.heap_bytes()),
            (material, shape)
        );

        // One geometry as a tile's state and its base, and as a second
        // tile's: counted once.
        let mut s = Surfaces::new();
        s.complete(key, Ok(Some(Arc::clone(&g))));
        let other = SurfaceKey::new(4, 1, 0, 0, 0).unwrap();
        s.complete(other, Ok(Some(Arc::clone(&g))));
        let m = s.memory();
        assert_eq!(
            (m.ready, m.geometries, m.shapes, m.composited),
            (2, 1, 1, 0)
        );
        assert_eq!((m.geometry_bytes, m.shape_bytes), (material + shape, shape));

        // A shown overlay adds the composited material, sharing the base's
        // shape, and the overlay tile.
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
                bases: vec![(0, Arc::clone(&base))],
                overlays: vec![(1, Arc::new(o))],
                ..DecodedTile::default()
            }),
            false,
        );
        let drawn = Arc::clone(s.drawn_geometry(&third).unwrap());
        assert!(Arc::ptr_eq(&drawn.tile_shape, &base.tile_shape));
        let m = s.memory();
        assert_eq!((m.geometries, m.shapes, m.composited), (3, 2, 1));
        assert_eq!(
            (m.geometry_bytes, m.composited_bytes),
            (3 * material + 2 * shape, material)
        );
        assert_eq!(m.overlay_bytes, o_bytes);
        assert_eq!(
            (m.drawn, m.drawn_geometry_bytes, m.drawn_overlay_bytes),
            (0, 0, 0)
        );
        // Before the split every geometry held its own whole copy: the
        // shape's struct and arrays and two material arrays.
        let whole = shape + material - (std::mem::size_of::<TileGeometry>() - 2 * 24) as u64;
        let before = 3 * whole;
        println!(
            "tile geometry bytes, these three geometries: before {before}, after {}",
            m.geometry_bytes
        );
        assert!(m.geometry_bytes < before);
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
    fn eviction_over_the_budget_keeps_what_is_drawn() {
        // Low camera looking level, looking at far more tiles than fit:
        // after each eviction pass the held bytes are at or under the
        // budget, or only what is drawn, walked to, and level-0 remains,
        // and every drawn tile stays.
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
        let full = s.held_bytes();
        let budget = full / 2;
        let mut over = full - budget;
        s.compact_over(&mut over);
        let mut items = s.eviction_candidates();
        crate::budget::sort_candidates(&mut items);
        let mut gone = Vec::new();
        for c in items {
            if s.held_bytes() <= budget {
                break;
            }
            gone.push(c.key);
            s.drop_tiles(&[c.key]);
        }
        assert!(!gone.is_empty(), "nothing was evicted from {full} bytes");
        let held = s.held_bytes();
        assert!(
            held <= budget || s.eviction_candidates().is_empty(),
            "{held} bytes held over a budget of {budget}"
        );
        for k in &want {
            assert!(
                s.ready(k).is_some() || matches!(s.tiles.get(k), Some(State::Compact(_))),
                "drawn tile {k} was evicted"
            );
            let mut up = k.parent();
            while let Some(p) = up {
                assert!(s.tiles.contains_key(&p), "ancestor {p} of {k} was evicted");
                up = p.parent();
            }
        }
    }

    #[test]
    fn asking_ahead_is_held_over_the_budget() {
        let (a, c) = (6_378_137.0, 6_356_752.3);
        let d =
            gx_core::surface::vertex_direction(&SurfaceKey::new(4, 0, 0, 0, 0).unwrap(), 32, 32);
        let (p, nrm) = gx_core::surface::ellipsoid_point(d, a, c);
        let cam: [f64; 3] = core::array::from_fn(|k| p[k] + 2000.0 * nrm[k]);
        let up_tangent =
            unit_cross(nrm, unit_cross([0.0, 0.0, 1.0], nrm).map(f64::from)).map(f64::from);
        let asked = |hold: bool| {
            let mut s = Surfaces::new();
            s.set_hold_ahead(hold);
            settle(&mut s, cam, up_tangent, 11);
            s.cached()
        };
        let (ahead, held) = (asked(false), asked(true));
        assert!(held < ahead, "{held} tiles held back, {ahead} asking ahead");
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

    /// One selection at `now` over tiles that arrive with their sections
    /// (so they may be compact), answering every request and every build
    /// at once, as [`Surfaces::select`] runs it.
    fn compact_round(
        s: &mut Surfaces,
        cam: [f64; 3],
        forward: [f64; 3],
        max_level: u8,
        now: f64,
    ) -> Vec<TileDrawInfo> {
        s.now = now;
        for key in s.take_requests() {
            let outcome = if key.level <= max_level {
                let bytes =
                    gx_core::surface::encode(&tile(key), gx_core::matter::Compression::None);
                decode_tile(&key, &[bytes], &[0], &BTreeSet::new())
            } else {
                Ok(DecodedTile::default())
            };
            s.complete_layers(key, outcome, false);
        }
        s.build_now();
        s.want_built.clear();
        let mut drawn = Vec::new();
        s.select_in_frame(
            4,
            cam,
            forward,
            focal_px(720.0, FIELD_OF_VIEW_Y),
            0.9,
            &mut drawn,
        );
        s.compact_unwanted();
        drawn
    }

    /// [`compact_round`] until nothing is asked for or wanted built.
    fn settle_compact(
        s: &mut Surfaces,
        cam: [f64; 3],
        forward: [f64; 3],
        max_level: u8,
        now: f64,
    ) -> Vec<TileDrawInfo> {
        for face in 0..SURFACE_FACES {
            s.want(SurfaceKey::new(4, face, 0, 0, 0).unwrap(), 1.0);
        }
        for _ in 0..64 {
            let drawn = compact_round(s, cam, forward, max_level, now);
            if s.requests.is_empty() && s.builds.is_empty() {
                return drawn;
            }
        }
        panic!("the selection never settled");
    }

    /// The camera `height` meters over a fixed point of face 4, looking
    /// down 25 degrees.
    fn low_camera(height: f64) -> ([f64; 3], [f64; 3]) {
        let d = gx_core::surface::vertex_direction(&SurfaceKey::new(4, 4, 0, 0, 0).unwrap(), 20, 9);
        let (p, nrm) = gx_core::surface::ellipsoid_point(d, 6_378_137.0, 6_356_752.3);
        let cam: [f64; 3] = core::array::from_fn(|k| p[k] + height * nrm[k]);
        let tangent = unit_cross([0.0, 0.0, 1.0], nrm).map(f64::from);
        let forward: [f64; 3] = core::array::from_fn(|k| 0.9 * tangent[k] - 0.42 * nrm[k]);
        (cam, forward)
    }

    fn by_key(drawn: &[TileDrawInfo]) -> BTreeMap<SurfaceKey, TileDrawInfo> {
        drawn.iter().map(|t| (t.geometry.key, t.clone())).collect()
    }

    fn built_keys(s: &Surfaces) -> BTreeSet<SurfaceKey> {
        (s.layers.iter())
            .filter(|(_, l)| l.built())
            .map(|(k, _)| *k)
            .collect()
    }

    #[test]
    fn a_descending_camera_draws_what_it_drew_when_every_tile_was_built() {
        // The same tiles, the same vertex data, with tiles built only when
        // about to be drawn as with every arrived tile built.
        let mut s = Surfaces::new();
        let mut levels = BTreeSet::new();
        for (step, height) in [2.0e6, 4.0e5, 6.0e4, 1.0e4, 3000.0].into_iter().enumerate() {
            let (cam, forward) = low_camera(height);
            let now = 10.0 * step as f64;
            let drawn = settle_compact(&mut s, cam, forward, 11, now);
            let mut all_built = Surfaces::new();
            let expected = settle(&mut all_built, cam, forward, 11);
            let (got, want) = (by_key(&drawn), by_key(&expected));
            assert_eq!(
                got.keys().collect::<Vec<_>>(),
                want.keys().collect::<Vec<_>>(),
                "{height} m"
            );
            for (k, t) in &got {
                assert_eq!(t.geometry, want[k].geometry, "{k} at {height} m");
                assert_eq!(t.morph, want[k].morph);
            }
            levels.extend(got.keys().map(|k| k.level));
            // Built: the drawn tiles and the children of the tiles within
            // their split distance (wanted built), and the roots.
            let built = built_keys(&s);
            let roots = built.iter().filter(|k| k.level == 0).count();
            assert!(built.len() <= drawn.len() + s.want_built.len() + roots);
            for k in &built {
                assert!(k.level == 0 || s.want_built.contains(k), "{k} built");
            }
            // Far more arrived than are built.
            if height < 1.0e5 {
                let compact = s.layers.values().filter(|l| !l.built()).count();
                assert!(compact > 0, "nothing compact at {height} m");
            }
        }
        assert!(levels.len() >= 4, "levels {levels:?}");
    }

    #[test]
    fn a_tile_out_of_view_goes_compact_and_is_rebuilt_the_same() {
        let mut s = Surfaces::new();
        let (cam, forward) = low_camera(1.0e4);
        let before = by_key(&settle_compact(&mut s, cam, forward, 11, 0.0));
        let deep: Vec<SurfaceKey> = before.keys().filter(|k| k.level > 1).copied().collect();
        assert!(!deep.is_empty());
        // Away from the surface: the deep tiles are neither drawn nor
        // wanted built, and stay built only for the grace time.
        let away = forward.map(|v| -v);
        compact_round(&mut s, cam, away, 11, 1.0);
        assert!(deep.iter().all(|k| built_keys(&s).contains(k)));
        compact_round(&mut s, cam, away, 11, 1.0 + BUILT_GRACE_SECONDS);
        for k in &deep {
            assert!(matches!(s.tiles.get(k), Some(State::Compact(_))), "{k}");
        }
        let after = by_key(&settle_compact(
            &mut s,
            cam,
            forward,
            11,
            2.0 + BUILT_GRACE_SECONDS,
        ));
        assert_eq!(
            before.keys().collect::<Vec<_>>(),
            after.keys().collect::<Vec<_>>()
        );
        for (k, t) in &before {
            assert_eq!(t.geometry, after[k].geometry, "{k}");
        }
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

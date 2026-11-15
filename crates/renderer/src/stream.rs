//! Cell streaming: which cells to fetch, the cache of fetched cells, which
//! cells to draw, and the native fetcher that brings sections in.
//!
//! Implements the renderer's steps "asks the hub what exists near the
//! camera, streams finished sections from the CDN, receives freshly
//! computed ones from the worker, composites layers" of `space-model.md`
//! section 2, over the octree cells of section 5, with the compositing rule
//! of section 8 and `matter-format.md` section 3.5. The hub side is the
//! loop of `client-compute.md` section 1, run by [`crate::directions`].
//!
//! # Selection
//!
//! Every 100 ms ([`SELECT_INTERVAL_SECONDS`]), every frame whose
//! `root_extent / 8` region is in the view or within `4 * root_extent` of
//! the camera is passed to [`gx_core::lod::select_cells`] with a pixel error
//! of 4, the viewport height, the vertical field of view, and at most 512
//! cells ([`select_all`]), except frames drawn as a sprite only. A frame's
//! depth limit is the deepest of its registry `max_depth` and its layers'
//! configs ([`crate::protocol::frame_system_from_registries`]).
//!
//! # The cache
//!
//! [`CellCache`] holds one [`CellState`] per cell key. Selected cells start
//! `Missing`; at most [`MAX_IN_FLIGHT`] (one directions batch) are
//! `Requested` at a time, largest projected size first. A request stays
//! `Requested` until the fetcher has every layer's section, however long the
//! hub keeps it `pending`; the fetcher retries on its own. The sections are
//! decoded and validated per layer and composited with
//! [`gx_core::matter::composite`] into `Ready` or, when the composite holds
//! no matter, `Empty`. Sections that fail validation, or a module that
//! fails, make a cell `Gone`. Cells not selected for 60 s are evicted, and
//! the cache never holds more than 4096 cells it could drop.
//!
//! While the observer walks, the cells around its feet at every depth of
//! its frame are priority cells ([`CellCache::set_priority`], chosen by
//! [`crate::probe::feet_cells`]): requested before anything else, coarsest
//! first, so contact soon uses the finest matter any layer offers there.
//!
//! A frame that is a sprite only in the far field ([`crate::farfield`]) has
//! no cells selected. The depth-0 cell of every frame that may need a sprite
//! is pinned ([`CellCache::pin`]): fetched once like a selected cell and
//! never evicted, so the sprite's brightness is always at hand.
//!
//! # Depth transitions
//!
//! [`CellCache::draw_set`] never leaves a hole. A selected cell that cannot
//! be drawn yet is covered by its loaded children if all of them can be
//! drawn, or else by its nearest drawable ancestor. A drawn ancestor hides
//! every drawn cell below it, so a parent keeps drawing while any of its
//! children is not ready, and stops once all of them are.
//!
//! # The fetcher
//!
//! [`Fetcher`] (native) runs [`crate::directions::Coordinator`] on the
//! frame loop's thread and carries out its actions on a tokio runtime:
//! directions through [`crate::hub::HubClient`], sections from the local
//! cache ([`gx_worker::DiskCache`]) or the edge, `compute` jobs on the
//! worker pool ([`gx_worker::WorkerPool`]), which hands each section over
//! at once and uploads it with its claim in the background, and readiness
//! notices from the socket. The browser has its own fetcher with the same
//! coordinator (`crate::web`).

use crate::camera::{Camera, FIELD_OF_VIEW_Y};
use crate::farfield::focal_px;
use crate::farfield::{far_frames, mean_albedo, FarFieldCell};
use crate::light::cell_emitter;
use crate::volume::VolumeGrid;
use gx_core::emission::Emitter;
use gx_core::frames::FrameSystem;
use gx_core::key::CellKey;
use gx_core::lod::{select_cells, SelectionParams};
use gx_core::matter::{self, composite, Section};
use gx_core::units::{Meters, Vec3};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

/// Seconds between two selections.
pub const SELECT_INTERVAL_SECONDS: f64 = 0.1;

/// Largest projected cell edge, pixels, that is not refined further.
pub const PIXEL_ERROR: f64 = 4.0;

/// Most cells selected per frame.
pub const MAX_CELLS_PER_FRAME: usize = 512;

/// Most cells requested at once: one directions batch
/// ([`crate::protocol::MAX_DIRECTION_KEYS`]).
pub const MAX_IN_FLIGHT: usize = crate::protocol::MAX_DIRECTION_KEYS;

/// Seconds a cell may go unselected before it is evicted.
pub const EVICT_AFTER_SECONDS: f64 = 60.0;

/// Most cells the cache keeps once eviction has dropped what it can.
pub const MAX_CACHED_CELLS: usize = 4096;

/// How far below a selected cell [`CellCache::draw_set`] looks for loaded
/// descendants to cover it.
pub const COVER_LEVELS: u8 = 2;

/// One selected cell and its projected size.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SelectedCell {
    /// The cell.
    pub key: CellKey,
    /// Projected edge in pixels, the request priority (largest first).
    pub projected_px: f64,
}

/// A fetched and composited cell.
#[derive(Clone, Debug, PartialEq)]
pub struct ReadyCell {
    /// The composite of every layer's section for the cell.
    pub section: Arc<Section>,
    /// The light escaping the cell's matter as one emitter, if any
    /// ([`crate::light::cell_emitter`]). Computed for depth-0 cells only,
    /// the ones that hold a frame's whole matter and so the only ones
    /// whose boundary is the frame's boundary with space; `None` for every
    /// other cell.
    pub emitter: Option<Emitter>,
    /// The cell's gas and plasma, if any, drawn as a volume
    /// ([`crate::volume`]).
    pub volume: Option<Arc<VolumeGrid>>,
    /// Mass-weighted mean albedo, for the far field
    /// ([`crate::farfield::mean_albedo`]).
    pub mean_albedo: [f64; 3],
    /// Some sample has a density above 0. A composite can hold samples and
    /// still no matter; such a cell counts as empty for
    /// [`CellCache::known_empty`].
    pub has_matter: bool,
    /// The density of the drawn surface, kilograms per cubic meter, the
    /// isovalue of [`crate::extract`] ([`crate::extract::surface_isovalue`]);
    /// `None` without solid or fluid samples. The
    /// observer's matter probe ([`crate::probe`]) puts the solid boundary
    /// at the same isovalue, so the observer stands on what is drawn.
    pub surface_isovalue: Option<f64>,
}

impl ReadyCell {
    /// Derives everything the renderer keeps about a composited section:
    /// its emitter, its volume, and its mean albedo.
    pub fn new(section: Section) -> ReadyCell {
        ReadyCell {
            emitter: if section.key().depth == 0 {
                cell_emitter(&section)
            } else {
                None
            },
            volume: VolumeGrid::from_section(&section).map(Arc::new),
            mean_albedo: mean_albedo(&section),
            has_matter: section.samples().is_some_and(|s| {
                (0..s.len()).any(|i| s.get(i).is_some_and(|x| x.density.value() > 0.0))
            }),
            surface_isovalue: crate::extract::surface_isovalue(&section),
            section: Arc::new(section),
        }
    }

    /// What the far field needs from the cell.
    pub fn far_field(&self) -> FarFieldCell {
        FarFieldCell {
            emitter: self.emitter,
            mean_albedo: self.mean_albedo,
        }
    }
}

/// Where a cell is in its fetch.
#[derive(Clone, Debug, PartialEq)]
pub enum CellState {
    /// Selected, not yet requested.
    Missing,
    /// The fetcher is resolving it: directions, streams, or a module run.
    Requested,
    /// Fetched and composited.
    Ready(Arc<ReadyCell>),
    /// No matter: every layer is empty, or the composite holds none. Draws
    /// nothing.
    Empty,
    /// Sections that fail validation, or a module that failed. Never drawn.
    Gone,
}

/// What the cache knows of the matter in a face neighbor
/// ([`CellCache::face_neighbors`]).
#[derive(Clone, Debug, PartialEq)]
pub enum NeighborMatter<'a> {
    /// The neighbor is `Ready`.
    Ready(&'a Arc<ReadyCell>),
    /// The neighbor holds no matter ([`CellCache::known_empty`]).
    Empty,
}

/// The result of one fetch, decoded.
#[derive(Clone, Debug, PartialEq)]
pub enum FetchOutcome {
    /// Every layer's section, decoded and composited; `None` when there is
    /// no matter.
    Decoded(Option<Arc<ReadyCell>>),
    /// A section failed validation or the module failed, with the reason.
    /// Never asked for again in this session.
    Invalid(String),
}

/// Decodes and validates each layer's section for `key` (`matter-format.md`
/// section 4), checks each section's geometry against the registry's
/// `root_extent` (section 3.1), and composites them (section 3.5). The
/// sections are the non-empty layers' in the build's layer order; none at
/// all is a cell without matter.
pub fn decode_cell<S: AsRef<[u8]>>(
    key: &CellKey,
    sections: &[S],
    root_extent: Meters,
) -> FetchOutcome {
    let mut decoded = Vec::with_capacity(sections.len());
    for (i, bytes) in sections.iter().enumerate() {
        match matter::decode(key, bytes.as_ref()) {
            Ok(s) => decoded.push(s),
            Err(e) => {
                return FetchOutcome::Invalid(format!("section {i}: code {}: {}", e.code, e.reason))
            }
        }
    }
    let g = key.geometry(root_extent);
    if let Some(i) = decoded
        .iter()
        .position(|s| s.origin() != g.origin || s.edge() != g.edge)
    {
        return FetchOutcome::Invalid(format!(
            "section {i}: cell geometry does not match the registry"
        ));
    }
    if decoded.is_empty() {
        return FetchOutcome::Decoded(None);
    }
    let refs: Vec<&Section> = decoded.iter().collect();
    match composite(&refs) {
        Ok(c) if c.is_empty() => FetchOutcome::Decoded(None),
        Ok(c) => FetchOutcome::Decoded(Some(Arc::new(ReadyCell::new(c)))),
        Err(e) => FetchOutcome::Invalid(format!("code {}: {}", e.code, e.reason)),
    }
}

/// The camera position in `frame_id`'s coordinates (relative to its
/// origin, in its axes).
pub fn camera_in_frame(system: &FrameSystem, camera: &Camera, frame_id: u64) -> Vec3 {
    let v = system.relative(camera.position, camera.frame_id, Vec3::zero(), frame_id);
    system.root_orientation(frame_id).conjugate().rotate(v)
}

/// Returns `true` when `frame_id`'s `root_extent / 8` region is within
/// `4 * root_extent` of the camera or inside the view cone.
pub fn frame_in_range(system: &FrameSystem, camera: &Camera, frame_id: u64, aspect: f64) -> bool {
    let Some(frame) = system.tree().get(frame_id) else {
        return false;
    };
    let extent = frame.root_extent.value();
    let region = extent / 8.0;
    let rel = camera.relative(system, Vec3::zero(), frame_id);
    let d = rel.length();
    if d - region <= 4.0 * extent {
        return true;
    }
    // Camera axes: looking along -z.
    let c = camera.root_orientation(system).conjugate().rotate(rel);
    let half_y = 0.5 * FIELD_OF_VIEW_Y;
    let half_x = (half_y.tan() * aspect.max(1.0)).atan();
    let half_diag = (half_x.tan().hypot(half_y.tan())).atan();
    let off_axis = (-c.z / d).clamp(-1.0, 1.0).acos();
    off_axis <= half_diag + (region / d).min(1.0).asin()
}

/// Selects the cells of every frame in range ([`frame_in_range`]) that is
/// not a sprite only in the far field ([`crate::farfield::FarFrame::sprite_only`]),
/// in ascending key order, with their projected sizes.
pub fn select_all(
    system: &FrameSystem,
    camera: &Camera,
    view_width_px: f64,
    view_height_px: f64,
) -> Vec<SelectedCell> {
    let aspect = view_width_px / view_height_px.max(1.0);
    let params = SelectionParams {
        pixel_error: PIXEL_ERROR,
        view_height_px,
        vertical_fov_rad: FIELD_OF_VIEW_Y,
        max_cells: MAX_CELLS_PER_FRAME,
    };
    let focal = focal_px(view_height_px);
    let sprites: BTreeSet<u64> = far_frames(system, camera, view_height_px)
        .iter()
        .filter(|f| f.sprite_only())
        .map(|f| f.frame_id)
        .collect();
    let mut out = Vec::new();
    for frame in system.tree().frames() {
        if sprites.contains(&frame.frame_id)
            || !frame_in_range(system, camera, frame.frame_id, aspect)
        {
            continue;
        }
        let cam = camera_in_frame(system, camera, frame.frame_id);
        for key in select_cells(frame, cam, &params).cells {
            let g = key.geometry(frame.root_extent);
            let e = g.edge.value();
            let gap = |c: f64, o: f64| (o - c).max(c - (o + e)).max(0.0);
            let d = Vec3::new(
                gap(cam.x, g.origin.x),
                gap(cam.y, g.origin.y),
                gap(cam.z, g.origin.z),
            )
            .length();
            out.push(SelectedCell {
                key,
                projected_px: e / d.max(e) * focal,
            });
        }
    }
    out
}

/// One cache entry.
#[derive(Clone, Debug, PartialEq)]
pub struct CellEntry {
    /// Fetch state.
    pub state: CellState,
    /// In the latest selection.
    pub selected: bool,
    /// Kept for the far field: requested like a selected cell, never
    /// evicted.
    pub pinned: bool,
    /// Around the walking observer's feet ([`CellCache::set_priority`]):
    /// requested before every other cell, never evicted while it is.
    pub priority: bool,
    /// Session time the cell was last selected or drawn, seconds.
    pub last_used: f64,
    /// Projected size at the latest selection, pixels.
    pub projected_px: f64,
    /// Session time of the latest request, seconds.
    pub requested_at: f64,
}

impl CellEntry {
    fn new(now: f64) -> CellEntry {
        CellEntry {
            state: CellState::Missing,
            selected: false,
            pinned: false,
            priority: false,
            last_used: now,
            projected_px: 0.0,
            requested_at: f64::NEG_INFINITY,
        }
    }
}

/// Counts for the overlay.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct CacheCounts {
    /// Cells in the latest selection.
    pub selected: usize,
    /// Selected cells that are `Ready` or `Empty`.
    pub ready: usize,
    /// Selected cells that are `Requested`.
    pub pending: usize,
    /// Requests in flight.
    pub in_flight: usize,
    /// Pinned cells (see [`CellCache::pin`]).
    pub pinned: usize,
    /// Every cell in the cache.
    pub cached: usize,
}

/// The cell cache: one entry per key, in key order.
#[derive(Clone, Debug, Default)]
pub struct CellCache {
    entries: BTreeMap<CellKey, CellEntry>,
    selection: Vec<SelectedCell>,
}

impl CellCache {
    /// An empty cache.
    pub fn new() -> CellCache {
        CellCache::default()
    }

    /// The entry for a key.
    pub fn entry(&self, key: &CellKey) -> Option<&CellEntry> {
        self.entries.get(key)
    }

    /// The state of a key, `None` if it is not cached.
    pub fn state(&self, key: &CellKey) -> Option<&CellState> {
        self.entries.get(key).map(|e| &e.state)
    }

    /// Every entry, in key order.
    pub fn entries(&self) -> impl Iterator<Item = (&CellKey, &CellEntry)> {
        self.entries.iter()
    }

    /// What the cache knows of the cells across the six faces of `key` at
    /// the same depth, in the order of [`crate::extract::face_index`]
    /// (`-x, +x, -y, +y, -z, +z`): the `Ready` ones and the ones known to
    /// hold no matter ([`CellCache::known_empty`]). A face at the edge of
    /// the root cube, or whose neighbor is neither, is `None`. Extraction
    /// pads with these ([`crate::extract`]).
    pub fn face_neighbors(&self, key: &CellKey) -> [Option<(CellKey, NeighborMatter<'_>)>; 6] {
        face_neighbor_keys(key).map(|k| {
            let k = k?;
            match self.state(&k) {
                Some(CellState::Ready(cell)) => Some((k, NeighborMatter::Ready(cell))),
                _ if self.known_empty(&k) => Some((k, NeighborMatter::Empty)),
                _ => None,
            }
        })
    }

    /// Returns `true` when the cache knows that `key` holds no matter: it
    /// holds none itself (`Empty`, or `Ready` without a sample of density
    /// above 0), an ancestor holds none, or its descendants down to
    /// [`COVER_LEVELS`] levels below hold none, covering it. The cell itself
    /// need not be cached: the selection may hold its region at another
    /// depth.
    pub fn known_empty(&self, key: &CellKey) -> bool {
        let empty = |k: &CellKey| self.holds_no_matter(k);
        let mut up = Some(*key);
        while let Some(k) = up {
            if empty(&k) {
                return true;
            }
            up = k.parent();
        }
        self.empty_below(key, COVER_LEVELS)
    }

    /// `key` itself is fetched and holds no matter.
    fn holds_no_matter(&self, key: &CellKey) -> bool {
        match self.state(key) {
            Some(CellState::Empty) => true,
            Some(CellState::Ready(cell)) => !cell.has_matter,
            _ => false,
        }
    }

    /// `key` holds no matter, or every child holds none, recursively within
    /// `levels`.
    fn empty_below(&self, key: &CellKey, levels: u8) -> bool {
        if self.holds_no_matter(key) {
            return true;
        }
        if levels == 0 {
            return false;
        }
        match key.children() {
            Some(children) => children.iter().all(|c| self.empty_below(c, levels - 1)),
            None => false,
        }
    }

    /// The latest selection.
    pub fn selection(&self) -> &[SelectedCell] {
        &self.selection
    }

    /// Replaces the selection. New keys start `Missing`; `Missing` keys no
    /// longer selected or pinned are dropped, since there is nothing to
    /// keep.
    pub fn set_selection(&mut self, cells: Vec<SelectedCell>, now: f64) {
        for e in self.entries.values_mut() {
            e.selected = false;
        }
        for c in &cells {
            let e = self
                .entries
                .entry(c.key)
                .or_insert_with(|| CellEntry::new(now));
            e.selected = true;
            e.last_used = now;
            e.projected_px = c.projected_px;
        }
        self.entries
            .retain(|_, e| e.selected || e.pinned || e.priority || e.state != CellState::Missing);
        self.selection = cells;
    }

    /// Replaces the priority cells: the cells around the walking observer's
    /// feet ([`crate::probe::feet_cells`]). They are requested before any
    /// other cell, whatever their projected size, and kept while they are
    /// priority cells. `Missing` keys that are no longer wanted at all are
    /// dropped, as in [`CellCache::set_selection`].
    pub fn set_priority(&mut self, keys: &[CellKey], now: f64) {
        for e in self.entries.values_mut() {
            e.priority = false;
        }
        for k in keys {
            let e = self
                .entries
                .entry(*k)
                .or_insert_with(|| CellEntry::new(now));
            e.priority = true;
            e.last_used = now;
        }
        self.entries
            .retain(|_, e| e.selected || e.pinned || e.priority || e.state != CellState::Missing);
    }

    /// Pins a cell: it is requested like a selected cell (if it is not
    /// cached yet) and never evicted. Returns `true` if it was not pinned
    /// before.
    pub fn pin(&mut self, key: CellKey, now: f64) -> bool {
        let e = self
            .entries
            .entry(key)
            .or_insert_with(|| CellEntry::new(now));
        !std::mem::replace(&mut e.pinned, true)
    }

    /// The pinned cells, in key order.
    pub fn pinned(&self) -> impl Iterator<Item = &CellKey> {
        self.entries
            .iter()
            .filter(|(_, e)| e.pinned)
            .map(|(k, _)| k)
    }

    /// Picks the next requests: priority `Missing` cells first, from the
    /// coarsest depth down, then selected or pinned `Missing` cells,
    /// largest projected size first (ties in key order), up to
    /// [`MAX_IN_FLIGHT`] in flight. Marks them `Requested`.
    pub fn take_requests(&mut self, now: f64) -> Vec<CellKey> {
        let in_flight = self
            .entries
            .values()
            .filter(|e| e.state == CellState::Requested)
            .count();
        let room = MAX_IN_FLIGHT.saturating_sub(in_flight);
        if room == 0 {
            return Vec::new();
        }
        // Priority cells sort ahead of everything, coarsest first.
        let rank = |e: &CellEntry| {
            if e.priority {
                f64::INFINITY
            } else {
                e.projected_px
            }
        };
        let mut due: Vec<(f64, u8, CellKey)> = self
            .entries
            .iter()
            .filter(|(_, e)| {
                (e.selected || e.pinned || e.priority) && e.state == CellState::Missing
            })
            .map(|(k, e)| (rank(e), if e.priority { k.depth } else { 0 }, *k))
            .collect();
        due.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
        due.truncate(room);
        due.into_iter()
            .map(|(_, _, k)| {
                let e = self.entries.get_mut(&k).expect("listed above");
                e.state = CellState::Requested;
                e.requested_at = now;
                k
            })
            .collect()
    }

    /// Applies the result of a request for `key`.
    pub fn complete(&mut self, key: CellKey, outcome: FetchOutcome, now: f64) {
        let e = self
            .entries
            .entry(key)
            .or_insert_with(|| CellEntry::new(now));
        e.state = match outcome {
            FetchOutcome::Decoded(Some(cell)) => CellState::Ready(cell),
            FetchOutcome::Decoded(None) => CellState::Empty,
            FetchOutcome::Invalid(reason) => {
                tracing::warn!(key = %key, "cell is invalid: {reason}");
                CellState::Gone
            }
        };
    }

    /// Marks cells as used now (drawn as stand-ins), so eviction keeps them.
    pub fn touch(&mut self, keys: &[CellKey], now: f64) {
        for k in keys {
            if let Some(e) = self.entries.get_mut(k) {
                e.last_used = now;
            }
        }
    }

    /// Evicts cells not used for [`EVICT_AFTER_SECONDS`], then, while more
    /// than [`MAX_CACHED_CELLS`] remain, the least recently used (ties in key
    /// order). Selected, pinned, and priority cells, and requests in flight,
    /// are never evicted.
    /// Returns the evicted keys in key order.
    pub fn evict(&mut self, now: f64) -> Vec<CellKey> {
        let droppable = |e: &CellEntry| {
            !e.selected && !e.pinned && !e.priority && e.state != CellState::Requested
        };
        let mut gone: BTreeSet<CellKey> = self
            .entries
            .iter()
            .filter(|(_, e)| droppable(e) && now - e.last_used >= EVICT_AFTER_SECONDS)
            .map(|(k, _)| *k)
            .collect();
        let remaining = self.entries.len() - gone.len();
        if remaining > MAX_CACHED_CELLS {
            let mut lru: Vec<(f64, CellKey)> = self
                .entries
                .iter()
                .filter(|(k, e)| droppable(e) && !gone.contains(k))
                .map(|(k, e)| (e.last_used, *k))
                .collect();
            lru.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            gone.extend(
                lru.into_iter()
                    .take(remaining - MAX_CACHED_CELLS)
                    .map(|(_, k)| k),
            );
        }
        for k in &gone {
            self.entries.remove(k);
        }
        gone.into_iter().collect()
    }

    /// Counts for the overlay.
    pub fn counts(&self) -> CacheCounts {
        let mut c = CacheCounts {
            selected: self.selection.len(),
            cached: self.entries.len(),
            ..CacheCounts::default()
        };
        for e in self.entries.values() {
            if e.state == CellState::Requested {
                c.in_flight += 1;
            }
            if e.pinned {
                c.pinned += 1;
            }
            if !e.selected {
                continue;
            }
            match e.state {
                CellState::Ready(_) | CellState::Empty => c.ready += 1,
                CellState::Requested => c.pending += 1,
                _ => {}
            }
        }
        c
    }

    /// The cells to draw for the latest selection, in key order, applying
    /// the depth transition rule of the module docs. `drawable` says whether
    /// a cell can be drawn now (its mesh is ready, or it is `Empty`).
    pub fn draw_set(&self, drawable: impl Fn(&CellKey) -> bool) -> Vec<CellKey> {
        let mut set: BTreeSet<CellKey> = BTreeSet::new();
        for s in &self.selection {
            if drawable(&s.key) {
                set.insert(s.key);
                continue;
            }
            if let Some(cover) = cover_below(&s.key, COVER_LEVELS, &drawable) {
                set.extend(cover);
                continue;
            }
            let mut up = s.key.parent();
            while let Some(p) = up {
                if drawable(&p) {
                    set.insert(p);
                    break;
                }
                up = p.parent();
            }
        }
        // A drawn ancestor hides everything below it.
        let hidden: Vec<CellKey> = set
            .iter()
            .filter(|k| {
                let mut up = k.parent();
                while let Some(p) = up {
                    if set.contains(&p) {
                        return true;
                    }
                    up = p.parent();
                }
                false
            })
            .copied()
            .collect();
        for k in hidden {
            set.remove(&k);
        }
        set.into_iter().collect()
    }
}

/// The keys of the cells across the six faces of `key` at the same depth,
/// in the order of [`crate::extract::face_index`] (`-x, +x, -y, +y, -z,
/// +z`); `None` where the face is the boundary of the root cube.
pub fn face_neighbor_keys(key: &CellKey) -> [Option<CellKey>; 6] {
    core::array::from_fn(|face| {
        let (axis, high) = (face / 2, face % 2 == 1);
        let mut c = [key.x, key.y, key.z];
        c[axis] = if high {
            c[axis].checked_add(1)?
        } else {
            c[axis].checked_sub(1)?
        };
        CellKey::new(key.frame_id, key.depth, c[0], c[1], c[2]).ok()
    })
}

/// Drawable descendants within `levels` that cover all of `key`, or `None`.
fn cover_below(
    key: &CellKey,
    levels: u8,
    drawable: &impl Fn(&CellKey) -> bool,
) -> Option<Vec<CellKey>> {
    if levels == 0 {
        return None;
    }
    let mut out = Vec::new();
    for child in key.children()? {
        if drawable(&child) {
            out.push(child);
        } else {
            out.extend(cover_below(&child, levels - 1, drawable)?);
        }
    }
    Some(out)
}

/// Something a fetcher reports back to the frame loop.
#[derive(Clone, Debug, PartialEq)]
pub enum FetchEvent {
    /// A cell request finished.
    Completed {
        /// The cell.
        key: CellKey,
        /// What came back, decoded.
        outcome: FetchOutcome,
        /// Section bytes received, streamed or computed.
        bytes: u64,
        /// Time from the request to having every layer's section.
        round_trip: std::time::Duration,
    },
    /// The registry key finished: every non-empty layer's registry section
    /// in layer order, or why it never can.
    Registry(Result<Vec<Arc<[u8]>>, String>),
}

/// Running totals over finished cell requests, for the overlay and the
/// headless statistics. Shared by the desktop and the browser fetchers.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct FetchTally {
    /// Cell requests finished.
    pub requests: u64,
    /// Cell requests that ended with decoded sections (with or without
    /// matter).
    pub bodies: u64,
    /// Section bytes received for cells.
    pub bytes: u64,
    /// Time to complete the first finished request.
    pub first_round_trip: Option<std::time::Duration>,
    /// Time to complete the latest finished request.
    pub last_round_trip: Option<std::time::Duration>,
}

impl FetchTally {
    /// Adds the finished requests among `events`, in order.
    pub fn record(&mut self, events: &[FetchEvent]) {
        for e in events {
            if let FetchEvent::Completed {
                outcome,
                bytes,
                round_trip,
                ..
            } = e
            {
                self.requests += 1;
                if matches!(outcome, FetchOutcome::Decoded(_)) {
                    self.bodies += 1;
                }
                self.bytes += bytes;
                self.first_round_trip.get_or_insert(*round_trip);
                self.last_round_trip = Some(*round_trip);
            }
        }
    }
}

/// Everything a fetcher counts: finished cells, the directions loop, and
/// the worker.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct FetcherStats {
    /// Finished cell requests.
    pub tally: FetchTally,
    /// The directions loop ([`crate::directions::DirectionStats`]).
    pub directions: crate::directions::DirectionStats,
    /// The worker ([`gx_worker::WorkerSnapshot`]).
    pub worker: gx_worker::WorkerSnapshot,
}

/// The moment to ask for directions at, UTC milliseconds: `wanted` (the
/// drawn moment) clamped to the hub's now and, once known, the origin.
/// Before anything is drawn, the hub's now.
pub fn directions_time(wanted: Option<i64>, origin: Option<i64>, now: i64) -> i64 {
    let t = wanted.unwrap_or(now).min(now);
    match origin {
        Some(o) => t.max(o),
        None => t,
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub use native::{Fetcher, FetcherOptions};

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use super::{decode_cell, directions_time, FetchEvent, FetchOutcome, FetchTally, FetcherStats};
    use crate::directions::{Action, Coordinator, Origin, StreamResult};
    use crate::hub::{HubClient, ReadySubscription};
    use crate::protocol::DirectionsResponse;
    use anyhow::{anyhow, Result};
    use gx_core::frames::FrameSystem;
    use gx_core::key::{CellKey, ChunkKey};
    use gx_core::sha256::to_hex;
    use gx_core::units::Meters;
    use gx_module_host::HostConfig;
    use gx_worker::{
        BlobKind, Blobs, ComputeJob, DiskCache, UploadTicket, Uploader, WorkerError, WorkerPool,
    };
    use std::collections::BTreeMap;
    use std::sync::mpsc;
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    /// How the fetcher's worker and cache are set up.
    #[derive(Clone, Debug)]
    pub struct FetcherOptions {
        /// The local cache, or `None` to keep nothing on disk.
        pub cache: Option<Arc<DiskCache>>,
        /// Worker threads ([`gx_worker::default_threads`] by default).
        pub worker_threads: usize,
        /// Memory and time limits of each module call.
        pub host: HostConfig,
    }

    impl Default for FetcherOptions {
        fn default() -> FetcherOptions {
            FetcherOptions {
                cache: None,
                worker_threads: gx_worker::default_threads(),
                host: HostConfig::default(),
            }
        }
    }

    /// The worker's view of the edge: blocking reads on the runtime.
    struct EdgeBlobs {
        handle: tokio::runtime::Handle,
        client: Arc<HubClient>,
    }

    impl Blobs for EdgeBlobs {
        fn fetch(&self, kind: BlobKind, hash: &[u8; 32]) -> Result<Vec<u8>, WorkerError> {
            let name = to_hex(hash);
            match self.handle.block_on(self.client.get_blob(kind, &name)) {
                Ok(Some(b)) => Ok(b),
                Ok(None) => Err(WorkerError::transient(format!("{kind} {name}: not found"))),
                Err(e) => Err(WorkerError::transient(format!("{e:#}"))),
            }
        }
    }

    /// Uploads on the runtime; failures are logged and never wait.
    struct HubUploader {
        handle: tokio::runtime::Handle,
        client: Arc<HubClient>,
    }

    impl Uploader for HubUploader {
        fn upload(&self, job: &ComputeJob, section: Arc<[u8]>, ticket: UploadTicket) {
            let client = self.client.clone();
            let (address, claim, key) = (job.address, job.claim.clone(), job.key());
            self.handle.spawn(async move {
                match client.upload(&address, &claim, section.to_vec()).await {
                    Ok(status) => {
                        tracing::debug!(key = %key, status, "section uploaded");
                        ticket.finish(true);
                    }
                    Err(e) => {
                        tracing::warn!(key = %key, "upload failed: {e:#}");
                        ticket.finish(false);
                    }
                }
            });
        }
    }

    /// What the fetcher's tasks report back.
    enum Internal {
        Directions(Vec<ChunkKey>, Result<DirectionsResponse, String>),
        Stream {
            key: ChunkKey,
            layer: usize,
            address: [u8; 32],
            result: StreamResult,
        },
        Compute {
            key: ChunkKey,
            layer: usize,
            address: [u8; 32],
            result: Result<Arc<[u8]>, WorkerError>,
        },
        Ready([u8; 32]),
        Event(FetchEvent),
    }

    /// Fetches cells and the registry through the hub on a tokio runtime,
    /// computing what nobody has computed yet, and reports back over a
    /// queue the frame loop drains without blocking.
    pub struct Fetcher {
        handle: tokio::runtime::Handle,
        client: Arc<HubClient>,
        cache: Option<Arc<DiskCache>>,
        pool: WorkerPool,
        coordinator: Coordinator,
        socket: ReadySubscription,
        tx: mpsc::Sender<Internal>,
        rx: mpsc::Receiver<Internal>,
        extents: BTreeMap<CellKey, Meters>,
        out: Vec<FetchEvent>,
        decoding: usize,
        started: Instant,
        wanted_utc_ms: Option<i64>,
        tally: FetchTally,
    }

    impl Fetcher {
        /// Starts a fetcher on `handle`: the worker pool, the readiness
        /// socket, and the task that keeps the hub's clock measured
        /// ([`HubClient::keep_clock`]).
        pub fn new(
            handle: tokio::runtime::Handle,
            client: Arc<HubClient>,
            options: FetcherOptions,
        ) -> Result<Fetcher> {
            let pool = WorkerPool::new(
                options.worker_threads,
                options.host,
                Arc::new(EdgeBlobs {
                    handle: handle.clone(),
                    client: client.clone(),
                }),
                options.cache.clone(),
                Arc::new(HubUploader {
                    handle: handle.clone(),
                    client: client.clone(),
                }),
            )
            .map_err(|e| anyhow!("starting the worker: {e}"))?;
            tracing::info!(threads = pool.size(), "worker started");
            let (tx, rx) = mpsc::channel();
            let notices = tx.clone();
            let socket =
                client.subscribe_ready(&handle, move |a| notices.send(Internal::Ready(a)).is_ok());
            client.keep_clock(&handle);
            Ok(Fetcher {
                handle,
                client,
                cache: options.cache,
                pool,
                coordinator: Coordinator::new(),
                socket,
                tx,
                rx,
                extents: BTreeMap::new(),
                out: Vec::new(),
                decoding: 0,
                started: Instant::now(),
                wanted_utc_ms: None,
                tally: FetchTally::default(),
            })
        }

        /// The hub client.
        pub fn client(&self) -> &Arc<HubClient> {
            &self.client
        }

        fn now(&self) -> f64 {
            self.started.elapsed().as_secs_f64()
        }

        /// Sets the moment being drawn, UTC milliseconds; directions are
        /// asked for it, clamped between the origin and the hub's now.
        pub fn set_time_utc_ms(&mut self, t: i64) {
            self.wanted_utc_ms = Some(t);
        }

        /// The hub's now, UTC milliseconds ([`HubClient::now_utc_ms`]).
        pub fn now_utc_ms(&self) -> i64 {
            self.client.now_utc_ms()
        }

        /// The space's origin, UTC milliseconds, once a directions answer
        /// or the hub's clock has reported it.
        pub fn origin_utc_ms(&self) -> Option<i64> {
            self.coordinator
                .origin_utc_ms()
                .or_else(|| self.client.clock().origin_utc_ms())
        }

        /// Queues a request for `key`, a cell of a frame of edge
        /// `root_extent`. Requests queued before the next
        /// [`Fetcher::flush`], [`Fetcher::drain`], or [`Fetcher::wait`] go
        /// out together, in batches of at most 512 keys. The result arrives
        /// through [`Fetcher::drain`]. A key already in flight is not asked
        /// for twice.
        pub fn request(&mut self, key: CellKey, root_extent: Meters) {
            self.extents.insert(key, root_extent);
            let now = self.now();
            self.coordinator.request(ChunkKey::Cell(key), now);
        }

        /// Sends what is queued and applies what has arrived, without
        /// waiting.
        pub fn flush(&mut self) {
            self.pump();
        }

        /// Starts a request for the registry key. The result arrives as
        /// [`FetchEvent::Registry`].
        pub fn request_registry(&mut self) {
            let now = self.now();
            self.coordinator.request(ChunkKey::Registry, now);
            self.pump();
        }

        fn apply(&mut self, msg: Internal) -> Vec<Action> {
            let now = self.now();
            match msg {
                Internal::Directions(keys, r) => self.coordinator.directions_done(&keys, r, now),
                Internal::Stream {
                    key,
                    layer,
                    address,
                    result,
                } => self
                    .coordinator
                    .stream_done(&key, layer, address, result, now),
                Internal::Compute {
                    key,
                    layer,
                    address,
                    result,
                } => self
                    .coordinator
                    .compute_done(&key, layer, address, result, now),
                Internal::Ready(a) => self.coordinator.ready_notice(&a),
                Internal::Event(e) => {
                    self.decoding = self.decoding.saturating_sub(1);
                    self.out.push(e);
                    Vec::new()
                }
            }
        }

        /// Applies every report so far and carries out what is due.
        fn pump(&mut self) {
            let mut actions = Vec::new();
            while let Ok(msg) = self.rx.try_recv() {
                actions.extend(self.apply(msg));
            }
            let now = self.now();
            actions.extend(self.coordinator.poll(now));
            while !actions.is_empty() {
                for a in std::mem::take(&mut actions) {
                    self.execute(a);
                }
                actions = self.coordinator.poll(self.now());
            }
        }

        fn execute(&mut self, action: Action) {
            match action {
                Action::Directions(keys) => {
                    let t = directions_time(
                        self.wanted_utc_ms,
                        self.coordinator.origin_utc_ms(),
                        self.client.now_utc_ms(),
                    );
                    let client = self.client.clone();
                    let tx = self.tx.clone();
                    self.handle.spawn(async move {
                        let r = client
                            .directions(t, &keys)
                            .await
                            .map_err(|e| format!("{e:#}"));
                        let _ = tx.send(Internal::Directions(keys, r));
                    });
                }
                Action::Stream {
                    key,
                    layer,
                    address,
                } => {
                    let client = self.client.clone();
                    let cache = self.cache.clone();
                    let tx = self.tx.clone();
                    self.handle.spawn(async move {
                        let result = stream_section(&client, cache, key, address).await;
                        let _ = tx.send(Internal::Stream {
                            key,
                            layer,
                            address,
                            result,
                        });
                    });
                }
                Action::Compute { key, layer, job } => {
                    let tx = self.tx.clone();
                    let address = job.address;
                    self.pool.submit(job, move |result| {
                        let _ = tx.send(Internal::Compute {
                            key,
                            layer,
                            address,
                            result,
                        });
                    });
                }
                Action::Subscribe(address) => self.socket.subscribe(address),
                Action::Complete(done) => match done.key {
                    ChunkKey::Registry => self.out.push(FetchEvent::Registry(Ok(done.sections))),
                    ChunkKey::Cell(key) => {
                        let Some(extent) = self.extents.remove(&key) else {
                            return;
                        };
                        let round_trip =
                            Duration::from_secs_f64((self.now() - done.requested_at).max(0.0));
                        let bytes = done.sections.iter().map(|s| s.len() as u64).sum();
                        let tx = self.tx.clone();
                        self.decoding += 1;
                        self.handle.spawn_blocking(move || {
                            let outcome = decode_cell(&key, &done.sections, extent);
                            let _ = tx.send(Internal::Event(FetchEvent::Completed {
                                key,
                                outcome,
                                bytes,
                                round_trip,
                            }));
                        });
                    }
                },
                Action::Failed { key, reason } => match key {
                    ChunkKey::Registry => self.out.push(FetchEvent::Registry(Err(reason))),
                    ChunkKey::Cell(key) => {
                        self.extents.remove(&key);
                        self.out.push(FetchEvent::Completed {
                            key,
                            outcome: FetchOutcome::Invalid(reason),
                            bytes: 0,
                            round_trip: Duration::ZERO,
                        });
                    }
                },
            }
        }

        fn take_events(&mut self) -> Vec<FetchEvent> {
            let events = std::mem::take(&mut self.out);
            self.tally.record(&events);
            events
        }

        /// Every event so far, without waiting.
        pub fn drain(&mut self) -> Vec<FetchEvent> {
            self.pump();
            self.take_events()
        }

        /// Waits up to `timeout` for at least one event, then drains.
        pub fn wait(&mut self, timeout: Duration) -> Vec<FetchEvent> {
            let deadline = Instant::now() + timeout;
            loop {
                self.pump();
                if !self.out.is_empty() {
                    return self.take_events();
                }
                let now = Instant::now();
                if now >= deadline {
                    return Vec::new();
                }
                let mut wait = deadline - now;
                if let Some(t) = self.coordinator.next_wake() {
                    let due = Duration::from_secs_f64((t - self.now()).max(0.0));
                    wait = wait.min(due.max(Duration::from_millis(1)));
                }
                if let Ok(msg) = self.rx.recv_timeout(wait) {
                    let actions = self.apply(msg);
                    for a in actions {
                        self.execute(a);
                    }
                }
            }
        }

        /// `true` while any key is in flight or being decoded.
        pub fn busy(&self) -> bool {
            self.coordinator.in_flight() > 0 || self.decoding > 0
        }

        /// The totals over every finished request this session, the
        /// directions loop, and the worker.
        pub fn stats(&self) -> FetcherStats {
            FetcherStats {
                tally: self.tally,
                directions: self.coordinator.stats(),
                worker: self.pool.stats(),
            }
        }

        /// The totals over every finished cell request this session.
        pub fn tally(&self) -> FetchTally {
            self.tally
        }

        /// Section bytes received for cells this session.
        pub fn bytes_fetched(&self) -> u64 {
            self.tally.bytes
        }

        /// Time to complete the latest finished cell request.
        pub fn last_round_trip(&self) -> Option<Duration> {
            self.tally.last_round_trip
        }

        /// Loads the frame registry (`client-compute.md`, the registry is
        /// just another key): asks for it like any key, waits up to
        /// `deadline` for every layer's registry section, and builds the
        /// frame system from their union with the build's layer depths. The
        /// integrator starts only after this.
        pub fn load_registry(&mut self, deadline: Duration) -> Result<FrameSystem> {
            self.request_registry();
            let start = Instant::now();
            loop {
                let left = deadline.saturating_sub(start.elapsed());
                if left.is_zero() {
                    return Err(anyhow!(
                        "the registry did not arrive within {} s",
                        deadline.as_secs_f64()
                    ));
                }
                for e in self.wait(left.min(Duration::from_millis(200))) {
                    if let FetchEvent::Registry(r) = e {
                        let sections = r.map_err(|e| anyhow!("registry: {e}"))?;
                        let system = crate::protocol::frame_system_from_registries(
                            &sections,
                            self.client.build(),
                        )?;
                        tracing::info!(frames = system.tree().frames().len(), "registry loaded");
                        return Ok(system);
                    }
                }
            }
        }
    }

    /// Gets a stored section: the local cache if it holds a valid copy,
    /// else the edge, validating and caching what comes back.
    async fn stream_section(
        client: &HubClient,
        cache: Option<Arc<DiskCache>>,
        key: ChunkKey,
        address: [u8; 32],
    ) -> StreamResult {
        let name = to_hex(&address);
        let name_for_cache = name.clone();
        let cached = match &cache {
            Some(c) => {
                let c = c.clone();
                tokio::task::spawn_blocking(move || {
                    let bytes = c.get(BlobKind::Section, &name_for_cache)?;
                    if gx_core::validate(&key.to_string(), &bytes).is_ok() {
                        Some(bytes)
                    } else {
                        c.remove(BlobKind::Section, &name_for_cache);
                        None
                    }
                })
                .await
                .ok()
                .flatten()
            }
            None => None,
        };
        if let Some(bytes) = cached {
            return StreamResult::Bytes(Arc::from(bytes), Origin::Cache);
        }
        match client.get_blob(BlobKind::Section, &name).await {
            Ok(Some(bytes)) => {
                let checked = tokio::task::spawn_blocking(move || {
                    gx_core::validate(&key.to_string(), &bytes)
                        .map_err(|e| format!("section {name} from the edge is invalid: {e}"))?;
                    if let Some(c) = &cache {
                        if let Err(e) = c.put(BlobKind::Section, &name, &bytes) {
                            tracing::warn!(name, "could not cache the section: {e}");
                        }
                    }
                    Ok::<_, String>(bytes)
                })
                .await;
                match checked {
                    Ok(Ok(bytes)) => StreamResult::Bytes(Arc::from(bytes), Origin::Edge),
                    Ok(Err(e)) => StreamResult::Failed(e),
                    Err(e) => StreamResult::Failed(format!("validation task: {e}")),
                }
            }
            Ok(None) => StreamResult::NotFound,
            Err(e) => StreamResult::Failed(format!("{e:#}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(d: u8, x: u32, y: u32, z: u32) -> CellKey {
        CellKey::new(5, d, x, y, z).unwrap()
    }

    fn sel(keys: &[CellKey]) -> Vec<SelectedCell> {
        keys.iter()
            .enumerate()
            .map(|(i, &key)| SelectedCell {
                key,
                projected_px: i as f64,
            })
            .collect()
    }

    fn ready() -> FetchOutcome {
        use gx_core::matter::{Sample, Samples, State};
        use gx_core::units::{Attenuation, Density, Kelvin, Ratio};
        let s = Section::new(
            key(0, 0, 0, 0),
            Vec3::zero(),
            Meters::new(1.0),
            1,
            Samples::filled(
                1,
                Sample {
                    density: Density::new(1.0),
                    state: State::Solid,
                    temperature: Kelvin::new(1.0),
                    albedo: [Ratio::new(0.0); 3],
                    roughness: Ratio::new(0.0),
                    attenuation: Attenuation::new(0.0),
                    radiating_temperature: Kelvin::new(1.0),
                },
            ),
        )
        .unwrap();
        FetchOutcome::Decoded(Some(Arc::new(ReadyCell::new(s))))
    }

    #[test]
    fn requests_are_capped_and_prioritized() {
        let mut c = CellCache::new();
        let n = MAX_IN_FLIGHT as u32 + 8;
        let keys: Vec<CellKey> = (0..n).map(|i| key(10, i, 0, 0)).collect();
        c.set_selection(sel(&keys), 0.0);
        let first = c.take_requests(0.0);
        assert_eq!(first.len(), MAX_IN_FLIGHT);
        // Largest projected size first: the last keys had the largest.
        assert_eq!(first[0], keys[n as usize - 1]);
        assert!(c.take_requests(0.0).is_empty());
        c.complete(first[0], FetchOutcome::Decoded(None), 0.1);
        assert_eq!(c.take_requests(0.1).len(), 1);
        assert_eq!(c.counts().in_flight, MAX_IN_FLIGHT);
    }

    #[test]
    fn priority_cells_go_first_coarsest_first_and_are_kept() {
        let mut c = CellCache::new();
        let n = MAX_IN_FLIGHT as u32 + 8;
        let keys: Vec<CellKey> = (0..n).map(|i| key(10, i, 0, 0)).collect();
        c.set_selection(sel(&keys), 0.0);
        let (fine, coarse) = (key(12, 0, 0, 0), key(2, 0, 0, 0));
        c.set_priority(&[fine, coarse], 0.0);
        let first = c.take_requests(0.0);
        assert_eq!(first.len(), MAX_IN_FLIGHT);
        assert_eq!(&first[..2], &[coarse, fine]);
        c.complete(coarse, ready(), 0.0);
        c.set_selection(Vec::new(), 0.0);
        assert!(c.evict(1000.0).iter().all(|k| *k != coarse));
        assert!(c.entry(&coarse).is_some_and(|e| e.priority));
        // No longer a priority: an unused cell like any other.
        c.set_priority(&[], 1000.0);
        assert!(c.evict(2000.0).contains(&coarse));
    }

    #[test]
    fn invalid_cells_are_gone_and_not_asked_again() {
        let mut c = CellCache::new();
        let (a, b) = (key(1, 0, 0, 0), key(1, 1, 0, 0));
        c.set_selection(sel(&[a, b]), 0.0);
        assert_eq!(c.take_requests(0.0).len(), 2);
        c.complete(a, FetchOutcome::Invalid("x".into()), 0.0);
        c.complete(b, FetchOutcome::Decoded(None), 0.0);
        assert!(c.take_requests(5.0).is_empty());
        assert_eq!(c.state(&a), Some(&CellState::Gone));
        assert_eq!(c.state(&b), Some(&CellState::Empty));
        assert_eq!(c.counts().ready, 1);
    }

    #[test]
    fn tally_counts_requests_bodies_and_round_trips() {
        use std::time::Duration;
        let done = |outcome: FetchOutcome, bytes: u64, ms: u64| FetchEvent::Completed {
            key: key(1, 0, 0, 0),
            outcome,
            bytes,
            round_trip: Duration::from_millis(ms),
        };
        let mut t = FetchTally::default();
        t.record(&[
            done(FetchOutcome::Invalid("x".into()), 0, 30),
            FetchEvent::Registry(Ok(Vec::new())),
            done(ready(), 100, 5),
        ]);
        t.record(&[done(FetchOutcome::Decoded(None), 72, 2)]);
        assert_eq!(t.requests, 3);
        assert_eq!(t.bodies, 2);
        assert_eq!(t.bytes, 172);
        assert_eq!(t.first_round_trip, Some(Duration::from_millis(30)));
        assert_eq!(t.last_round_trip, Some(Duration::from_millis(2)));
    }

    #[test]
    fn directions_time_is_clamped_to_origin_and_now() {
        assert_eq!(directions_time(None, None, 50), 50);
        assert_eq!(directions_time(Some(80), Some(10), 50), 50);
        assert_eq!(directions_time(Some(5), Some(10), 50), 10);
        assert_eq!(directions_time(Some(20), Some(10), 50), 20);
    }

    #[test]
    fn decode_rejects_geometry_that_disagrees_with_the_registry() {
        use gx_core::matter::{encode, Compression};
        let k = key(1, 1, 0, 0);
        let g = k.geometry(Meters::new(8.0));
        let empty = Section::empty(k, g.origin, g.edge).unwrap();
        let bytes = encode(&empty, Compression::None);
        assert_eq!(
            decode_cell(&k, &[&bytes], Meters::new(8.0)),
            FetchOutcome::Decoded(None)
        );
        assert!(matches!(
            decode_cell(&k, &[&bytes], Meters::new(16.0)),
            FetchOutcome::Invalid(_)
        ));
        assert!(matches!(
            decode_cell(&k, &[&[1u8, 2][..]], Meters::new(8.0)),
            FetchOutcome::Invalid(_)
        ));
        let none: [&[u8]; 0] = [];
        assert_eq!(
            decode_cell(&k, &none, Meters::new(8.0)),
            FetchOutcome::Decoded(None)
        );
    }

    #[test]
    fn eviction_by_age_and_cap() {
        let mut c = CellCache::new();
        let a = key(1, 0, 0, 0);
        c.set_selection(sel(&[a]), 0.0);
        c.take_requests(0.0);
        c.complete(a, ready(), 0.0);
        c.set_selection(Vec::new(), 1.0);
        assert!(c.evict(59.0).is_empty());
        assert_eq!(c.evict(60.0), vec![a]);

        let many: Vec<CellKey> = (0..(MAX_CACHED_CELLS as u32 + 10))
            .map(|i| CellKey::new(5, 13, i % 8192, i / 8192, 0).unwrap())
            .collect();
        for (i, k) in many.iter().enumerate() {
            c.complete(*k, FetchOutcome::Decoded(None), i as f64 * 1e-3);
        }
        let gone = c.evict(10.0);
        assert_eq!(gone.len(), 10);
        assert_eq!(gone, many[..10].to_vec());
        assert_eq!(c.counts().cached, MAX_CACHED_CELLS);
    }

    #[test]
    fn pinned_cells_are_requested_and_kept() {
        let mut c = CellCache::new();
        let k = key(0, 0, 0, 0);
        assert!(c.pin(k, 0.0));
        assert!(!c.pin(k, 0.0));
        // Not selected, still requested.
        c.set_selection(Vec::new(), 0.0);
        assert_eq!(c.take_requests(0.0), vec![k]);
        c.complete(k, ready(), 0.1);
        c.set_selection(Vec::new(), 0.2);
        assert!(c.evict(1000.0).is_empty());
        assert!(matches!(c.state(&k), Some(CellState::Ready(_))));
        assert_eq!(c.pinned().copied().collect::<Vec<_>>(), vec![k]);
    }

    #[test]
    fn parent_draws_until_every_child_is_ready() {
        let mut c = CellCache::new();
        let parent = key(1, 0, 0, 0);
        let mut kids = parent.children().unwrap();
        kids.sort();
        c.set_selection(sel(&kids), 0.0);
        // Only the parent is drawable: it stands in for all children.
        let mut drawable: BTreeSet<CellKey> = [parent].into();
        let d = |c: &CellCache, s: &BTreeSet<CellKey>| c.draw_set(|k| s.contains(k));
        assert_eq!(d(&c, &drawable), vec![parent]);
        // Seven of eight children ready: still the parent alone.
        drawable.extend(kids[..7].iter().copied());
        assert_eq!(d(&c, &drawable), vec![parent]);
        // All eight: the children, without the parent.
        drawable.insert(kids[7]);
        assert_eq!(d(&c, &drawable), kids.to_vec());

        // Moving away: the parent is selected but not ready, its children
        // cover it.
        c.set_selection(sel(&[parent]), 1.0);
        drawable.remove(&parent);
        assert_eq!(d(&c, &drawable), kids.to_vec());
        // Nothing drawable anywhere: nothing drawn.
        assert!(d(&c, &BTreeSet::new()).is_empty());
    }

    #[test]
    fn face_neighbors_at_the_same_depth() {
        let k = key(2, 0, 1, 3);
        assert_eq!(
            face_neighbor_keys(&k),
            [
                None,
                Some(key(2, 1, 1, 3)),
                Some(key(2, 0, 0, 3)),
                Some(key(2, 0, 2, 3)),
                Some(key(2, 0, 1, 2)),
                None,
            ]
        );
        assert_eq!(face_neighbor_keys(&key(0, 0, 0, 0)), [None; 6]);

        let mut c = CellCache::new();
        c.set_selection(sel(&[k, key(2, 1, 1, 3), key(2, 0, 0, 3)]), 0.0);
        c.complete(key(2, 1, 1, 3), ready(), 0.0);
        c.complete(key(2, 0, 0, 3), FetchOutcome::Decoded(None), 0.0);
        // A Ready cell one depth down touching the +y face is not a neighbor.
        c.complete(key(3, 1, 4, 6), ready(), 0.0);
        let n = c.face_neighbors(&k);
        assert!(matches!(n[1], Some((_, NeighborMatter::Ready(_)))));
        assert_eq!(n[2], Some((key(2, 0, 0, 3), NeighborMatter::Empty)));
        assert!(n[0].is_none() && n[3].is_none() && n[4].is_none() && n[5].is_none());

        // The -z neighbor (2, 0, 1, 2) is not cached, but every child of it
        // is fetched and empty: known empty. One child unknown: not.
        let below = key(2, 0, 1, 2).children().unwrap();
        for child in &below[1..] {
            c.complete(*child, FetchOutcome::Decoded(None), 0.0);
        }
        assert!(c.face_neighbors(&k)[4].is_none());
        c.complete(below[0], FetchOutcome::Decoded(None), 0.0);
        assert_eq!(
            c.face_neighbors(&k)[4],
            Some((key(2, 0, 1, 2), NeighborMatter::Empty))
        );
        // An empty ancestor covers everything under it.
        c.complete(key(1, 0, 0, 1), FetchOutcome::Decoded(None), 0.0);
        assert!(c.known_empty(&key(3, 1, 2, 5)));
        assert!(!c.known_empty(&key(3, 1, 2, 3)));
    }

    #[test]
    fn a_composite_of_vacuum_holds_no_matter() {
        use gx_core::matter::{Sample, Samples};
        let k = key(2, 1, 1, 1);
        let zero = Sample::VACUUM;
        let s = Section::new(
            k,
            Vec3::zero(),
            Meters::new(1.0),
            2,
            Samples::filled(2, zero),
        );
        let cell = ReadyCell::new(s.unwrap());
        assert!(!cell.has_matter);
        let mut c = CellCache::new();
        c.complete(k, FetchOutcome::Decoded(Some(Arc::new(cell))), 0.0);
        assert!(c.known_empty(&k));
        assert!(c.known_empty(&key(3, 2, 3, 2)));
        if let FetchOutcome::Decoded(Some(full)) = ready() {
            assert!(full.has_matter);
        }
    }
}

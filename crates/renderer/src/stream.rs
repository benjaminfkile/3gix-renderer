//! Cell streaming: which cells to fetch, the cache of fetched cells, which
//! cells to draw, and the native fetcher that brings sections in.
//!
//! Implements the renderer's steps "asks the hub what exists near the
//! camera, streams finished sections from the CDN, receives freshly
//! computed ones from the worker, composites layers" of `space-model.md`
//! section 2, over the octree cells of section 5, with the compositing rule
//! of section 8 and `matter-format.md` section 3.5, and the selection and
//! streaming rules of `rendering.md` section 3. The hub side is the loop of
//! `client-compute.md` section 1, run by [`crate::directions`].
//!
//! # Selection (`rendering.md` 3.1)
//!
//! Every frame whose `root_extent / 8` region is in the view or within
//! `4 * root_extent` of the camera, except frames drawn as a sprite only,
//! is passed to [`gx_core::lod::select_cells_with`] with a pixel error of
//! [`PIXEL_ERROR`], the viewport height, the vertical field of view, and at
//! most [`MAX_CELLS_PER_FRAME`] cells ([`select_all`]). A cell is refined
//! only inside the view cone ([`VIEW_CONE_MARGIN`] wider than the view's
//! diagonal) and only where it is refinable: a solid or fluid cell whose
//! occupancy crosses 0.5, or a cell holding gas or plasma
//! ([`ReadyCell::refinable`]). Nothing else limits it: there is no
//! occlusion culling and no assumption about the shape of a frame's matter
//! (`rendering.md` 1.1). A frame's depth limit is the deepest of its
//! registry `max_depth` and its layers' configs
//! ([`crate::protocol::frame_system_from_registries`]).
//!
//! Each selection makes two trees of cells per frame:
//!
//! - **The target**: what the frame would be drawn with if every cell had
//!   arrived. A cell that has not arrived counts as refinable when the
//!   occupancy of its nearest arrived ancestor, at most
//!   [`AHEAD_DEPTHS`]` - 1` depths up, says the 0.5 level crosses its region
//!   with one sample of margin ([`ReadyCell::ahead`]). The target is then
//!   balanced 2:1 ([`balance_target`]): it is split until no two cells that
//!   touch across a face, an edge, or a corner differ by more than one
//!   depth.
//! - **The drawn selection**: the target, pruned from the root down, so that
//!   a cell is split only when every child has arrived and every cell
//!   around it at its own depth is split or outside the target
//!   ([`drawn_from_target`]). It is balanced by construction and holds only
//!   arrived cells, and it is the target once everything has arrived.
//!
//! # Asking ahead (`rendering.md` 3.2) and switching depth (3.3)
//!
//! Every cell of the target, leaves and the cells above them, that has not
//! arrived is wanted: on every selection the cache hands the fetcher the
//! wanted set, the largest [`MAX_IN_FLIGHT`] of those cells with each
//! one's projected size as its priority
//! ([`CellCache::take_wanted`], `client-compute.md` section 2.8), and the
//! coordinator asks and streams largest first, dropping cells the view has
//! left. Because the target reaches up to
//! [`AHEAD_DEPTHS`] depths below the arrived cells, refinement takes about
//! one round trip per three depths. The drawn selection switches a region to
//! finer cells only when everything it needs there has arrived; the coarser
//! cells draw until then.
//!
//! # When the selection runs (`rendering.md` 3.4)
//!
//! [`crate::world::World`] selects when results arrive, when a first mesh is
//! extracted, when the camera or the view changes, and otherwise every
//! [`SELECT_INTERVAL_SECONDS`] as the frames move.
//!
//! # The cache
//!
//! [`CellCache`] holds one [`CellEntry`] per cell key, in a hash map keyed
//! by cell ([`CellMap`]): every lookup the selection and the extraction
//! inputs make is constant time. A request stays `Requested` until the
//! fetcher has every layer's section; the fetcher retries on its own. Sections are decoded once on
//! their way in, off the frame thread (`rendering.md` 3.5), and composited
//! with [`gx_core::matter::composite`] into `Ready` or, when the composite
//! holds no matter, `Empty`. Sections that fail validation, or a module
//! that fails, make a cell `Gone`. Cells of no target for 60 s are evicted,
//! and the cache never holds more than 4096 cells it could drop.
//!
//! A frame that is a sprite only in the far field ([`crate::farfield`]) has
//! no cells selected. The depth-0 cell of every frame that may need a sprite
//! is pinned ([`CellCache::pin`]): fetched once like a selected cell and
//! never evicted, so the sprite's brightness is always at hand.
//!
//! # Drawing
//!
//! [`CellCache::draw_set`] draws the drawn selection. A selected cell whose
//! mesh is not extracted yet is covered by its drawable children if all of
//! them are, or else by its nearest drawable ancestor; a drawn ancestor
//! hides every drawn cell below it, so nothing is drawn twice.
//!
//! # The fetcher
//!
//! [`Fetcher`] (native) runs [`crate::directions::Coordinator`] on the
//! frame loop's thread and carries out its actions on a tokio runtime:
//! directions through [`crate::hub::HubClient`]; `ready` sections from the
//! local cache ([`gx_worker::DiskCache`]) or the edge (`GET
//! /section/{address}?sha256={hex}`), checked against the hash the
//! directions carried (a cached copy that differs is discarded and fetched
//! again) and decoded on a blocking task; `missing` and `local` jobs on the
//! worker pool ([`gx_worker::WorkerPool`]), which hands each section over
//! decoded at once. A computed `missing` section then enters the upload
//! lane ([`gx_worker::UploadLane`], `client-compute.md` sections 2.5 and
//! 2.8): bounded at 64, oldest dropped when full, and offered once with
//! the section 2.5 headers on the runtime as soon as it arrives, never on
//! a pool thread and never waiting for computes, with at most four in flight
//! ([`gx_worker::MAX_UPLOADS_IN_FLIGHT`]), a failure counted and never
//! retried. Compute jobs start highest priority first, and on every new
//! wanted set the waiting ones are reprioritized and those whose key left
//! it dropped before they start. A `local` section is never offered. Uploads never delay a
//! result reaching the renderer. The frame loop hands the fetcher the
//! wanted cells and surface tiles on every selection
//! ([`Fetcher::set_wanted_cells`], [`Fetcher::set_wanted_surfaces`]).
//!
//! A layer with inputs (`client-compute.md` section 10) takes its slots and
//! frame records from the build: the records are fetched by hash before
//! the registry is asked for, and every compute job is completed from the
//! build before it is queued. The pool resolves its input sections through
//! the fetcher's input source (section 10.3, [`gx_worker::InputSource`]):
//! input jobs from the build, directions for one key, and streams, as
//! blocking calls on the runtime from pool threads, never from the frame
//! thread. Checks of such a layer resolve their inputs the same way.
//! Uploads and check reports carry `X-Inputs-Sha256` (section 10.4); a
//! `409` is counted, never retried. The
//! browser has its own fetcher with the same coordinator (`crate::web`);
//! it does not resolve layer inputs yet.

use crate::camera::{Camera, FIELD_OF_VIEW_Y};
use crate::directions::Arrived;
use crate::farfield::focal_px;
use crate::farfield::{far_frames, matter_radius, mean_albedo, FarFieldCell};
use crate::light::cell_emitter;
use crate::volume::VolumeGrid;
use gx_core::emission::Emitter;
use gx_core::frames::FrameSystem;
use gx_core::key::CellKey;
use gx_core::lod::{select_cells_with, SelectionParams};
use gx_core::matter::{self, composite, Section, State};
use gx_core::registry::Frame;
use gx_core::units::{Meters, Vec3};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};
use std::sync::Arc;

/// Seconds between two selections when nothing else asks for one: the
/// frames move and turn, and the camera with its frame.
pub const SELECT_INTERVAL_SECONDS: f64 = 0.1;

/// Largest projected cell edge, pixels, that is not refined further.
/// Justified by the screen and the budget (`rendering.md` 1.2 and 5.2):
/// at 512 a cell of 32 samples per axis draws its samples at most about 16
/// pixels apart, and a 1280 x 720 view stays within [`MAX_CELLS_PER_FRAME`]
/// cells and 2,000,000 triangles.
pub const PIXEL_ERROR: f64 = 512.0;

/// Factor on the half angle of the view cone inside which cells are
/// refined ([`select_all`]): a little wider than the view
/// (`rendering.md` 3.1), so turning slightly does not show coarse cells.
pub const VIEW_CONE_MARGIN: f64 = 1.15;

/// Most cells selected per frame: the budget's limit (`rendering.md` 5.2).
pub const MAX_CELLS_PER_FRAME: usize = 4096;

/// Percent of [`MAX_CELLS_PER_FRAME`] the balanced target of a frame aims
/// at when [`select_all`] adjusts a frame's cell budget: close to the
/// limit, with room for the target to grow a little between selections
/// without passing it.
pub const BUDGET_AIM: usize = 97;

/// Most cells requested at once: as many as the directions loop asks the
/// hub about at once, [`crate::directions::MAX_DIRECTIONS_IN_FLIGHT`]
/// batches of [`crate::protocol::MAX_DIRECTION_KEYS`] keys. Asking ahead
/// three depths at a time can need more than one batch in one round trip
/// (`rendering.md` 3.2); more than this would only wait in the directions
/// queue, where a moving camera can no longer reorder it.
pub const MAX_IN_FLIGHT: usize =
    crate::directions::MAX_DIRECTIONS_IN_FLIGHT * crate::protocol::MAX_DIRECTION_KEYS;

/// Seconds a cell may be outside every target before it is evicted.
pub const EVICT_AFTER_SECONDS: f64 = 60.0;

/// Most cells the cache keeps once eviction has dropped what it can.
pub const MAX_CACHED_CELLS: usize = 4096;

/// How far below a cell [`CellCache::known_empty`] and
/// [`CellCache::draw_set`] look for arrived descendants that cover it.
pub const COVER_LEVELS: u8 = 2;

/// How many depths below an arrived cell the selection predicts and asks
/// for (`rendering.md` 3.2).
pub const AHEAD_DEPTHS: u8 = 3;

/// Samples of margin around a descendant's region when predicting whether
/// the 0.5 level crosses it (`rendering.md` 3.2).
pub const AHEAD_MARGIN_SAMPLES: usize = 1;

/// The level of occupancy that is a solid or fluid surface
/// (`matter-format.md` section 3.3).
pub const SURFACE_OCCUPANCY: f64 = 0.5;

/// A fast, deterministic hasher for cell keys: the same keys hash the same
/// on every run and every machine (FxHash's multiply and rotate).
#[derive(Copy, Clone, Debug, Default)]
pub struct CellHasher(u64);

impl CellHasher {
    #[inline]
    fn add(&mut self, v: u64) {
        self.0 = (self.0.rotate_left(5) ^ v).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }
}

impl Hasher for CellHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.add(u64::from(b));
        }
    }

    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.add(u64::from(i));
    }

    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.add(u64::from(i));
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.add(i);
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.add(i as u64);
    }
}

/// A hash map keyed by cell with [`CellHasher`]. Iteration order is never
/// used for anything that is drawn or sent.
pub type CellMap<V> = HashMap<CellKey, V, BuildHasherDefault<CellHasher>>;

/// A hash set of cells with [`CellHasher`].
pub type CellSet = HashSet<CellKey, BuildHasherDefault<CellHasher>>;

/// One selected cell and its projected size.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SelectedCell {
    /// The cell.
    pub key: CellKey,
    /// Projected edge in pixels, the request priority (largest first).
    pub projected_px: f64,
}

/// Which descendants of a cell, one to [`AHEAD_DEPTHS`] depths down, the
/// cell's occupancy says are refinable: the 0.5 level of its solid and
/// fluid samples crosses the descendant's region of the samples, with
/// [`AHEAD_MARGIN_SAMPLES`] of margin, or gas or plasma is there
/// (`rendering.md` 3.2). One bit per descendant: 8 at one depth down, 64 at
/// two, 512 at three.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AheadBits([u64; 10]);

impl AheadBits {
    /// Bit offset of the descendants `levels` depths down.
    fn base(levels: u8) -> usize {
        match levels {
            1 => 0,
            2 => 8,
            _ => 72,
        }
    }

    /// The bit index of descendant `(x, y, z)` (relative to the cell, each
    /// `0..2^levels`) `levels` depths down.
    fn index(levels: u8, x: u32, y: u32, z: u32) -> usize {
        let side = 1usize << levels;
        Self::base(levels) + x as usize + side * (y as usize + side * z as usize)
    }

    fn set(&mut self, i: usize) {
        self.0[i / 64] |= 1 << (i % 64);
    }

    fn get(&self, i: usize) -> bool {
        self.0[i / 64] & (1 << (i % 64)) != 0
    }

    /// No descendant is refinable.
    pub fn none() -> AheadBits {
        AheadBits([0; 10])
    }

    /// Computes the bits from a section's samples.
    pub fn of(section: &Section) -> AheadBits {
        let mut bits = AheadBits::none();
        let Some(s) = section.samples() else {
            return bits;
        };
        let n = usize::from(section.resolution());
        let flags: Vec<u8> = (0..s.len()).map(|i| sample_flags(s, n, i)).collect();
        let deepest = AHEAD_DEPTHS;
        let side = 1u32 << deepest;
        // The deepest level box by box; the coarser levels are the union of
        // their children, whose boxes with margin cover the parent's box
        // with margin.
        let mut masks = vec![0u8; (side * side * side) as usize];
        for z in 0..side {
            for y in 0..side {
                for x in 0..side {
                    let range = |c: u32| {
                        let lo = (n * c as usize) / side as usize;
                        let hi = (n * (c as usize + 1)).div_ceil(side as usize);
                        (
                            lo.saturating_sub(AHEAD_MARGIN_SAMPLES),
                            (hi + AHEAD_MARGIN_SAMPLES).min(n),
                        )
                    };
                    let (x0, x1) = range(x);
                    let (y0, y1) = range(y);
                    let (z0, z1) = range(z);
                    let mut m = 0u8;
                    'scan: for k in z0..z1 {
                        for j in y0..y1 {
                            let row = n * (j + n * k);
                            for i in x0..x1 {
                                m |= flags[row + i];
                                if crossing(m) {
                                    break 'scan;
                                }
                            }
                        }
                    }
                    masks[(x + side * (y + side * z)) as usize] = m;
                }
            }
        }
        for levels in (1..=deepest).rev() {
            let side = 1u32 << levels;
            for z in 0..side {
                for y in 0..side {
                    for x in 0..side {
                        let m = masks[(x + side * (y + side * z)) as usize];
                        if crossing(m) {
                            bits.set(Self::index(levels, x, y, z));
                        }
                    }
                }
            }
            if levels > 1 {
                let half = side / 2;
                let mut up = vec![0u8; (half * half * half) as usize];
                for z in 0..side {
                    for y in 0..side {
                        for x in 0..side {
                            up[(x / 2 + half * (y / 2 + half * (z / 2))) as usize] |=
                                masks[(x + side * (y + side * z)) as usize];
                        }
                    }
                }
                masks = up;
            }
        }
        bits
    }

    /// Whether `descendant`, 1 to [`AHEAD_DEPTHS`] depths below `cell`, is
    /// predicted refinable. `false` for any other key.
    pub fn predicts(&self, cell: &CellKey, descendant: &CellKey) -> bool {
        let Some(levels) = descendant.depth.checked_sub(cell.depth) else {
            return false;
        };
        if !(1..=AHEAD_DEPTHS).contains(&levels) || descendant.frame_id != cell.frame_id {
            return false;
        }
        let local = |d: u32, c: u32| d.checked_sub(c << levels).filter(|&l| l < (1 << levels));
        match (
            local(descendant.x, cell.x),
            local(descendant.y, cell.y),
            local(descendant.z, cell.z),
        ) {
            (Some(x), Some(y), Some(z)) => self.get(Self::index(levels, x, y, z)),
            _ => false,
        }
    }
}

/// Flag bits of sample `i` of a section of resolution `n` for
/// [`crossing`]: 1 when its surface occupancy is below 0.5, 2 when at or
/// above, 4 for gas or plasma. The surface occupancy of a solid or fluid
/// sample is its occupancy; every other sample has none.
///
/// A partly filled solid or fluid sample on a face of the section counts as
/// both sides of 0.5. Its sub-cube is cut by the boundary of its matter
/// (occupancy is the filled share of the sub-cube, `matter-format.md`
/// section 3.3), and the part of that sub-cube between the sample's center
/// and the face lies past the last sample center: the 0.5 level may cross
/// there, and the sample on the other side of the face that would show it
/// belongs to the neighbor. A fully filled or empty face sample holds no
/// boundary in its sub-cube, so nothing crosses there.
fn sample_flags(s: &gx_core::matter::Samples, n: usize, i: usize) -> u8 {
    match s.state(i) {
        State::Solid | State::Fluid => {
            let occupancy = s.occupancy(i).value();
            let (x, y, z) = (i % n, (i / n) % n, i / (n * n));
            let on_face = [x, y, z].iter().any(|&c| c == 0 || c + 1 == n);
            if on_face && occupancy < 1.0 {
                3
            } else if occupancy >= SURFACE_OCCUPANCY {
                2
            } else {
                1
            }
        }
        State::Gas | State::Plasma => 4 | 1,
        State::Vacuum => 1,
    }
}

/// A region with these [`sample_flags`] is refinable: it holds gas or
/// plasma, or surface occupancy on both sides of 0.5.
fn crossing(m: u8) -> bool {
    m & 4 != 0 || m & 3 == 3
}

/// A fetched and composited cell.
#[derive(Clone, Debug, PartialEq)]
pub struct ReadyCell {
    /// The composite of the shown layers' sections for the cell.
    pub section: Arc<Section>,
    /// Every non-empty layer's decoded section, with its layer index, in
    /// layer order, kept so the cell can be composited again from the shown
    /// layers when the hidden set changes (`rendering.md` section 7 item
    /// 7) without fetching anything. A cell with one layer shares that
    /// section with [`ReadyCell::section`]. Empty for a cell built from a
    /// composite alone ([`ReadyCell::new`]).
    pub layers: Vec<(usize, Arc<Section>)>,
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
    /// Radius of the ball of the volume its matter fills, for the far field
    /// ([`crate::farfield::matter_radius`]).
    pub radius: f64,
    /// Some sample has a density above 0. A composite can hold samples and
    /// still no matter; such a cell counts as empty for
    /// [`CellCache::known_empty`].
    pub has_matter: bool,
    /// The cell's children can show more than it does (`rendering.md`
    /// 3.1): it is a solid or fluid cell whose occupancy crosses 0.5 (some
    /// solid or fluid sample at or above 0.5 and some sample below it, or a
    /// partly filled one on a face of the cell, see [`sample_flags`]), or
    /// it holds gas or plasma. Only such cells are refined ([`select_all`]).
    pub refinable: bool,
    /// Which descendants up to [`AHEAD_DEPTHS`] down are predicted
    /// refinable, for asking ahead.
    pub ahead: AheadBits,
}

impl ReadyCell {
    /// Derives everything the renderer keeps about a composited section:
    /// its emitter, its volume, its mean albedo, whether it and its
    /// descendants are refinable. Runs off the frame thread.
    pub fn new(section: Section) -> ReadyCell {
        ReadyCell::from_section(Arc::new(section))
    }

    /// [`ReadyCell::new`] for a section already shared.
    pub fn from_section(section: Arc<Section>) -> ReadyCell {
        let volume = VolumeGrid::from_section(&section).map(Arc::new);
        let n = usize::from(section.resolution());
        let refinable = section.samples().is_some_and(|s| {
            let mut m = 0u8;
            for i in 0..s.len() {
                m |= sample_flags(s, n, i);
                if crossing(m) {
                    return true;
                }
            }
            false
        });
        ReadyCell {
            emitter: if section.key().depth == 0 {
                cell_emitter(&section)
            } else {
                None
            },
            volume,
            mean_albedo: mean_albedo(&section),
            radius: matter_radius(&section),
            has_matter: section
                .samples()
                .is_some_and(|s| (0..s.len()).any(|i| s.density(i).value() > 0.0)),
            refinable,
            ahead: AheadBits::of(&section),
            layers: Vec::new(),
            section,
        }
    }

    /// The layers this cell's composite is made of under `hidden`: its
    /// layers not in `hidden`, by index.
    pub fn shown_layers(&self, hidden: &BTreeSet<usize>) -> Vec<usize> {
        (self.layers.iter())
            .map(|(l, _)| *l)
            .filter(|l| !hidden.contains(l))
            .collect()
    }

    /// The cell composited again from its layers not in `hidden`
    /// (`rendering.md` section 7 item 7), keeping every layer. With every
    /// layer hidden it holds an empty section and no matter.
    pub fn with_hidden(&self, hidden: &BTreeSet<usize>) -> Result<ReadyCell, String> {
        let shown: Vec<&Section> = (self.layers.iter())
            .filter(|(l, _)| !hidden.contains(l))
            .map(|(_, s)| s.as_ref())
            .collect();
        let section = match shown.as_slice() {
            [] => {
                let s = &self.section;
                Arc::new(
                    Section::empty(s.key(), s.origin(), s.edge())
                        .map_err(|e| format!("code {}: {}", e.code, e.reason))?,
                )
            }
            [_] => (self.layers.iter())
                .find(|(l, _)| !hidden.contains(l))
                .map(|(_, s)| Arc::clone(s))
                .expect("one shown layer"),
            many => {
                Arc::new(composite(many).map_err(|e| format!("code {}: {}", e.code, e.reason))?)
            }
        };
        let mut cell = ReadyCell::from_section(section);
        cell.layers = self.layers.clone();
        Ok(cell)
    }

    /// Bytes held by the cell's decoded samples: its composite and every
    /// layer's section not shared with it, at [`SAMPLE_BYTES`] per sample.
    pub fn sample_bytes(&self) -> usize {
        let len = |s: &Section| s.samples().map_or(0, |v| v.len());
        let own = len(&self.section);
        let layers: usize = (self.layers.iter())
            .filter(|(_, s)| !Arc::ptr_eq(s, &self.section))
            .map(|(_, s)| len(s))
            .sum();
        (own + layers) * SAMPLE_BYTES
    }

    /// What the far field needs from the cell.
    pub fn far_field(&self) -> FarFieldCell {
        FarFieldCell {
            emitter: self.emitter,
            mean_albedo: self.mean_albedo,
            radius: self.radius,
        }
    }
}

/// Bytes one decoded sample takes in memory: density, state, temperature,
/// three albedo channels, roughness, attenuation, radiating temperature,
/// and occupancy (eight 4-byte values and one byte).
pub const SAMPLE_BYTES: usize = 37;

/// Where a cell is in its fetch.
#[derive(Clone, Debug, PartialEq)]
pub enum CellState {
    /// Known, not yet requested.
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

impl CellState {
    /// The cell has arrived: `Ready` or `Empty`.
    pub fn arrived(&self) -> bool {
        matches!(self, CellState::Ready(_) | CellState::Empty)
    }
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
/// section 4), then composites them ([`cell_from_sections`]). For bytes
/// nobody has decoded yet; sections that arrive through a fetcher come
/// decoded ([`cell_from_arrived`]).
pub fn decode_cell<S: AsRef<[u8]>>(
    key: &CellKey,
    sections: &[S],
    root_extent: Meters,
) -> FetchOutcome {
    let mut decoded = Vec::with_capacity(sections.len());
    for (i, bytes) in sections.iter().enumerate() {
        match matter::decode(key, bytes.as_ref()) {
            Ok(s) => decoded.push(Arc::new(s)),
            Err(e) => {
                return FetchOutcome::Invalid(format!("section {i}: code {}: {}", e.code, e.reason))
            }
        }
    }
    let layers: Vec<usize> = (0..decoded.len()).collect();
    cell_from_sections(key, &decoded, &layers, root_extent)
}

/// The cell from every layer's arrived section: the decoded section where
/// the stream or the worker decoded it on the way in, else decoded here.
/// Each section is validated once (`rendering.md` 3.5).
/// `layers` gives each section's layer index in the build, parallel to
/// `sections`.
pub fn cell_from_arrived(
    key: &CellKey,
    sections: &[Arrived],
    layers: &[usize],
    root_extent: Meters,
) -> FetchOutcome {
    let mut decoded = Vec::with_capacity(sections.len());
    for (i, a) in sections.iter().enumerate() {
        match &a.decoded {
            Some(s) if s.key() == *key => decoded.push(s.clone()),
            Some(_) => {
                return FetchOutcome::Invalid(format!("section {i}: decoded for another key"))
            }
            None => match matter::decode(key, &a.bytes) {
                Ok(s) => decoded.push(Arc::new(s)),
                Err(e) => {
                    return FetchOutcome::Invalid(format!(
                        "section {i}: code {}: {}",
                        e.code, e.reason
                    ))
                }
            },
        }
    }
    cell_from_sections(key, &decoded, layers, root_extent)
}

/// Checks each decoded section's geometry against the registry's
/// `root_extent` (`matter-format.md` section 3.1) and composites them
/// (section 3.5). The sections are the non-empty layers' in the build's
/// layer order, each with its layer index in `layers` (parallel); none at
/// all is a cell without matter. The cell keeps every layer's section
/// ([`ReadyCell::layers`]).
pub fn cell_from_sections(
    key: &CellKey,
    decoded: &[Arc<Section>],
    layers: &[usize],
    root_extent: Meters,
) -> FetchOutcome {
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
    let kept: Vec<(usize, Arc<Section>)> = (decoded.iter().enumerate())
        .map(|(i, s)| (layers.get(i).copied().unwrap_or(i), Arc::clone(s)))
        .collect();
    let section = if let [one] = decoded {
        Arc::clone(one)
    } else {
        let refs: Vec<&Section> = decoded.iter().map(|s| s.as_ref()).collect();
        match composite(&refs) {
            Ok(c) => Arc::new(c),
            Err(e) => return FetchOutcome::Invalid(format!("code {}: {}", e.code, e.reason)),
        }
    };
    // A composite with no samples is a cell without matter, unless some
    // layer holds samples a hidden layer could stop covering.
    if section.is_empty() && kept.iter().all(|(_, s)| s.is_empty()) {
        return FetchOutcome::Decoded(None);
    }
    let mut cell = ReadyCell::from_section(section);
    cell.layers = kept;
    FetchOutcome::Decoded(Some(Arc::new(cell)))
}

/// The camera position in `frame_id`'s coordinates (relative to its
/// origin, in its axes).
pub fn camera_in_frame(system: &FrameSystem, camera: &Camera, frame_id: u64) -> Vec3 {
    let v = system.relative(camera.position, camera.frame_id, Vec3::zero(), frame_id);
    system.root_orientation(frame_id).conjugate().rotate(v)
}

/// Returns `true` when the ball around `frame_id`'s cube (all its matter is
/// inside it, [`crate::farfield::unknown_radius`]) is within
/// `4 * root_extent` of the camera or inside the view cone.
pub fn frame_in_range(system: &FrameSystem, camera: &Camera, frame_id: u64, aspect: f64) -> bool {
    let Some(frame) = system.tree().get(frame_id) else {
        return false;
    };
    let extent = frame.root_extent.value();
    // The frame's matter lies inside its cube: the ball around the cube.
    let region = crate::farfield::unknown_radius(frame);
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

/// What one selection decided ([`select_all`]).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Selection {
    /// The drawn selection: arrived cells, balanced, in `(frame, depth, x,
    /// y, z)` order, with their projected sizes.
    pub drawn: Vec<SelectedCell>,
    /// The cells of [`Selection::drawn`], for lookups.
    pub drawn_cells: CellSet,
    /// The target's leaves, in the same order.
    pub target: Vec<SelectedCell>,
    /// Every cell of the target, leaves and the cells above them.
    pub target_cells: CellSet,
    /// The cells of the target that have not arrived and are not known to
    /// hold no matter, largest projected size first (ties in key order):
    /// what to request.
    pub needed: Vec<SelectedCell>,
}

/// Selects the cells of every frame in range ([`frame_in_range`]) that is
/// not a sprite only in the far field ([`crate::farfield::FarFrame::sprite_only`]):
/// the target and the drawn selection of the module docs. `budgets` is the
/// cell budget per frame for [`gx_core::lod::select_cells_with`]
/// (default [`MAX_CELLS_PER_FRAME`]); when balancing makes a frame's target
/// larger than [`MAX_CELLS_PER_FRAME`] it is scaled toward [`BUDGET_AIM`]
/// and the frame selected again, and while the target stays below the aim
/// it is scaled up for the next selection.
pub fn select_all(
    system: &FrameSystem,
    camera: &Camera,
    (view_width_px, view_height_px): (f64, f64),
    cache: &impl CellStates,
    budgets: &mut BTreeMap<u64, usize>,
    skip_frames: &BTreeSet<u64>,
) -> Selection {
    let aspect = view_width_px / view_height_px.max(1.0);
    let focal = focal_px(view_height_px);
    let sprites: BTreeSet<u64> = far_frames(system, camera, view_height_px, |f| {
        depth_zero_radius(cache, f)
    })
    .iter()
    .filter(|f| f.sprite_only())
    .map(|f| f.frame_id)
    .collect();
    let mut out = Selection::default();
    for frame in system.tree().frames() {
        if sprites.contains(&frame.frame_id)
            || skip_frames.contains(&frame.frame_id)
            || !frame_in_range(system, camera, frame.frame_id, aspect)
        {
            continue;
        }
        let cam = camera_in_frame(system, camera, frame.frame_id);
        // The view direction in the frame's axes, and the half angle of a
        // cone holding the whole view (its diagonal), a little wider.
        let forward = system.root_orientation(frame.frame_id).conjugate().rotate(
            camera
                .root_orientation(system)
                .rotate(Vec3::new(0.0, 0.0, -1.0)),
        );
        let half_y = 0.5 * FIELD_OF_VIEW_Y;
        let half_x = (half_y.tan() * aspect).atan();
        let cone = (half_x.tan().hypot(half_y.tan())).atan() * VIEW_CONE_MARGIN;
        let view = FrameView {
            frame,
            cam,
            forward,
            cone,
            focal,
        };
        let budget = budgets
            .get(&frame.frame_id)
            .copied()
            .unwrap_or(MAX_CELLS_PER_FRAME);
        let params = SelectionParams {
            pixel_error: PIXEL_ERROR,
            view_height_px,
            vertical_fov_rad: FIELD_OF_VIEW_Y,
            max_cells: budget,
        };
        let mut tree = target_tree(&view, &params, cache);
        let leaves = tree.leaves.len();
        // Balancing adds cells in about a fixed proportion: scale the
        // budget by how far the balanced target is from the limit, aiming
        // a little below it.
        let scaled =
            |n: usize| (budget * MAX_CELLS_PER_FRAME * BUDGET_AIM / (100 * n.max(1))).max(1);
        if leaves > MAX_CELLS_PER_FRAME && budget > 1 {
            // Over the limit: select again with the smaller budget.
            let lower = scaled(leaves).min(budget - 1);
            let params = SelectionParams {
                max_cells: lower,
                ..params
            };
            tree = target_tree(&view, &params, cache);
            budgets.insert(frame.frame_id, lower);
        } else if budget < MAX_CELLS_PER_FRAME
            && leaves * 100 < MAX_CELLS_PER_FRAME * BUDGET_AIM - MAX_CELLS_PER_FRAME
        {
            // More than a point below the aim: raise the budget for next
            // time.
            budgets.insert(
                frame.frame_id,
                scaled(leaves).clamp(budget + 1, MAX_CELLS_PER_FRAME),
            );
        }
        let drawn = drawn_from_target(&tree, cache);
        out.drawn_cells.extend(drawn.iter().copied());
        for key in drawn {
            out.drawn.push(view.selected(key));
        }
        let mut leaves: Vec<CellKey> = tree.leaves.iter().copied().collect();
        leaves.sort_unstable_by_key(|k| (k.depth, k.x, k.y, k.z));
        out.target
            .extend(leaves.into_iter().map(|k| view.selected(k)));
        out.needed.extend(
            needed_cells(&tree, cache)
                .into_iter()
                .map(|k| view.selected(k)),
        );
        out.target_cells.extend(tree.leaves);
        out.target_cells.extend(tree.internal);
    }
    out.needed.sort_by(|a, b| {
        b.projected_px
            .total_cmp(&a.projected_px)
            .then(a.key.cmp(&b.key))
    });
    out
}

/// Whether the selection's geometry asks to refine `key` for this camera
/// and view: its frame is in range and not a sprite only, the cell is
/// inside the view cone, larger on screen than [`PIXEL_ERROR`], and above
/// its frame's depth limit. Together with [`ReadyCell::refinable`] this is
/// the whole of the selection rule (`rendering.md` 3.1); tests use it to
/// show that nothing else holds a cell back.
pub fn wants_refinement(
    system: &FrameSystem,
    camera: &Camera,
    (view_width_px, view_height_px): (f64, f64),
    cache: &impl CellStates,
    key: &CellKey,
) -> bool {
    let aspect = view_width_px / view_height_px.max(1.0);
    let Some(frame) = system.tree().get(key.frame_id) else {
        return false;
    };
    let sprite = far_frames(system, camera, view_height_px, |f| {
        depth_zero_radius(cache, f)
    })
    .iter()
    .any(|f| f.frame_id == key.frame_id && f.sprite_only());
    if sprite || !frame_in_range(system, camera, key.frame_id, aspect) {
        return false;
    }
    let forward = system.root_orientation(frame.frame_id).conjugate().rotate(
        camera
            .root_orientation(system)
            .rotate(Vec3::new(0.0, 0.0, -1.0)),
    );
    let half_y = 0.5 * FIELD_OF_VIEW_Y;
    let half_x = (half_y.tan() * aspect).atan();
    let view = FrameView {
        frame,
        cam: camera_in_frame(system, camera, frame.frame_id),
        forward,
        cone: (half_x.tan().hypot(half_y.tan())).atan() * VIEW_CONE_MARGIN,
        focal: focal_px(view_height_px),
    };
    key.depth < frame.max_depth
        && view.in_view(*key)
        && view.selected(*key).projected_px > PIXEL_ERROR
}

/// One frame as the selection sees it.
struct FrameView<'a> {
    frame: &'a Frame,
    /// The camera in the frame's coordinates.
    cam: Vec3,
    /// The view direction in the frame's axes.
    forward: Vec3,
    /// Half angle of the view cone, radians.
    cone: f64,
    /// Focal length, pixels.
    focal: f64,
}

impl FrameView<'_> {
    /// Inside the view cone, or holding the camera.
    fn in_view(&self, key: CellKey) -> bool {
        let g = key.geometry(self.frame.root_extent);
        let half = 0.5 * g.edge.value();
        let center = g.origin + Vec3::new(half, half, half);
        let to = center - self.cam;
        let d = to.length();
        let radius = half * 3.0f64.sqrt();
        if d <= radius {
            return true;
        }
        let angle = (to.dot(self.forward) / d).clamp(-1.0, 1.0).acos();
        angle <= self.cone + (radius / d).asin()
    }

    /// The cell with its projected edge, pixels.
    fn selected(&self, key: CellKey) -> SelectedCell {
        let g = key.geometry(self.frame.root_extent);
        let e = g.edge.value();
        let gap = |c: f64, o: f64| (o - c).max(c - (o + e)).max(0.0);
        let d = Vec3::new(
            gap(self.cam.x, g.origin.x),
            gap(self.cam.y, g.origin.y),
            gap(self.cam.z, g.origin.z),
        )
        .length();
        SelectedCell {
            key,
            projected_px: e / d.max(e) * self.focal,
        }
    }
}

/// A tree of cells: its leaves and the cells above them (split).
#[derive(Clone, Debug, Default)]
pub struct CellTree {
    /// The leaves.
    pub leaves: CellSet,
    /// Every strict ancestor of a leaf.
    pub internal: CellSet,
}

impl CellTree {
    /// The tree whose leaves are `leaves`.
    pub fn from_leaves(leaves: impl IntoIterator<Item = CellKey>) -> CellTree {
        let leaves: CellSet = leaves.into_iter().collect();
        let mut internal = CellSet::default();
        for k in &leaves {
            let mut up = k.parent();
            while let Some(p) = up {
                if !internal.insert(p) {
                    break;
                }
                up = p.parent();
            }
        }
        CellTree { leaves, internal }
    }

    /// `key` is a leaf or above one.
    pub fn contains(&self, key: &CellKey) -> bool {
        self.leaves.contains(key) || self.internal.contains(key)
    }
}

/// The balanced target of one frame (module docs).
fn target_tree(
    view: &FrameView<'_>,
    params: &SelectionParams,
    cache: &impl CellStates,
) -> CellTree {
    let refine = |key: CellKey| view.in_view(key) && cache.refinable_or_predicted(&key);
    let cells = select_cells_with(view.frame, view.cam, params, refine).cells;
    let mut tree = CellTree::from_leaves(cells);
    balance_target(&mut tree);
    tree
}

/// The offsets of the 26 cells around a cell: every `(dx, dy, dz)` in
/// `{-1, 0, 1}^3` but zero, in ascending order of `dz`, then `dy`, then
/// `dx`.
pub const AROUND: [[i8; 3]; 26] = {
    let mut out = [[0i8; 3]; 26];
    let mut i = 0;
    let mut dz = -1i8;
    while dz <= 1 {
        let mut dy = -1i8;
        while dy <= 1 {
            let mut dx = -1i8;
            while dx <= 1 {
                if !(dx == 0 && dy == 0 && dz == 0) {
                    out[i] = [dx, dy, dz];
                    i += 1;
                }
                dx += 1;
            }
            dy += 1;
        }
        dz += 1;
    }
    out
};

/// The parents of the 26 cells around `key` at its depth, other than its
/// own parent: at most 7 distinct cells, the corner of the 27 cells around
/// the parent that `key` sits in. A cell around `key` across the low side
/// of an axis has the parent's low neighbor as its parent when `key` is the
/// low child along that axis, and the parent itself otherwise; the high
/// side the same way. `None` past the root cube, and for a depth-0 key.
pub fn around_parents(key: &CellKey) -> [Option<CellKey>; 7] {
    let Some(parent) = key.parent() else {
        return [None; 7];
    };
    let step = |c: u32| if c & 1 == 1 { 1i8 } else { -1i8 };
    let s = [step(key.x), step(key.y), step(key.z)];
    core::array::from_fn(|i| {
        let bits = i + 1;
        let o = core::array::from_fn(|a| if bits >> a & 1 == 1 { s[a] } else { 0 });
        offset_key(&parent, o)
    })
}

/// Splits `tree` until it is balanced 2:1: no two leaves that touch across
/// a face, an edge, or a corner differ by more than one depth. A tree is
/// balanced exactly when, for every split cell, the 26 cells around it at
/// its own depth are cells of the tree (their parents are split), or lie
/// where the tree has no cell (outside the root cube, or a child the
/// selection culled by distance).
///
/// One pass over the split cells, deepest first: where a cell around a
/// split cell lies inside a coarser leaf, that leaf is split down to it.
/// What a split adds is shallower than the cell being checked, so it is
/// checked later in the same pass; nothing is checked twice. Splitting
/// never needs anything loaded: the target is what would be drawn if every
/// cell had arrived. The result is the smallest balanced tree holding
/// `tree`, whatever the order of the cells within one depth.
pub fn balance_target(tree: &mut CellTree) {
    let mut buckets: Vec<Vec<CellKey>> = vec![Vec::new(); 32];
    for k in &tree.internal {
        buckets[usize::from(k.depth)].push(*k);
    }
    for depth in (1..32usize).rev() {
        let mut i = 0;
        while i < buckets[depth].len() {
            let n = buckets[depth][i];
            i += 1;
            for p in around_parents(&n).into_iter().flatten() {
                if tree.internal.contains(&p) {
                    continue;
                }
                // The leaf holding `p`, at or above it; none means the
                // region is not in the tree.
                let mut leaf = Some(p);
                while let Some(a) = leaf {
                    if tree.leaves.contains(&a) {
                        break;
                    }
                    if tree.internal.contains(&a) {
                        leaf = None;
                        break;
                    }
                    leaf = a.parent();
                }
                let Some(mut a) = leaf else {
                    continue;
                };
                // Split from that leaf down to `p`.
                loop {
                    tree.leaves.remove(&a);
                    tree.internal.insert(a);
                    buckets[usize::from(a.depth)].push(a);
                    let Some(children) = a.children() else {
                        break;
                    };
                    tree.leaves.extend(children);
                    if a == p {
                        break;
                    }
                    let shift = p.depth - a.depth - 1;
                    match children
                        .into_iter()
                        .find(|c| (p.x >> shift, p.y >> shift, p.z >> shift) == (c.x, c.y, c.z))
                    {
                        Some(c) => a = c,
                        None => break,
                    }
                }
            }
        }
    }
}

/// The drawn selection from a balanced target (module docs): from the root
/// down, a split cell of the target stays split when every child has
/// arrived (or is known to hold no matter) and every cell around it at its
/// depth is split here too, or lies where the target has no cell. Returns
/// the leaves in `(depth, x, y, z)` order.
pub fn drawn_from_target(tree: &CellTree, cache: &impl CellStates) -> Vec<CellKey> {
    // Shallowest first: a cell is decided after its parent and after the
    // parents of the cells around it.
    let internal: Vec<CellKey> = by_depth(tree.internal.iter()).concat();
    // Split cells, with whether they or an ancestor hold no matter.
    let mut split: CellMap<bool> = CellMap::default();
    for n in &internal {
        let parent_empty = match n.parent() {
            Some(p) => match split.get(&p) {
                Some(&e) => e,
                None => continue,
            },
            None => false,
        };
        let empty = parent_empty || cache.holds_no_matter(n);
        let Some(children) = n.children() else {
            continue;
        };
        let arrived = empty
            || children
                .iter()
                .all(|c| !tree.contains(c) || cache.arrived_or_empty(c));
        if !arrived {
            continue;
        }
        let balanced = around_parents(n)
            .into_iter()
            .flatten()
            .all(|p| split.contains_key(&p) || !tree.internal.contains(&p));
        if balanced {
            split.insert(*n, empty);
        }
    }
    let mut out: Vec<CellKey> = Vec::new();
    match internal.first() {
        Some(root) if root.depth == 0 && split.contains_key(root) => {}
        _ => {
            // Nothing split: the root alone, if the target has one.
            out.extend(
                tree.leaves
                    .iter()
                    .chain(tree.internal.iter())
                    .filter(|k| k.depth == 0),
            );
            return out;
        }
    }
    for n in &internal {
        if !split.contains_key(n) {
            continue;
        }
        for c in n.children().into_iter().flatten() {
            if tree.contains(&c) && !split.contains_key(&c) {
                out.push(c);
            }
        }
    }
    out.sort_unstable_by_key(|k| (k.depth, k.x, k.y, k.z));
    out
}

/// The cells of `tree` that have not arrived and are not known to hold no
/// matter, shallowest first. A cell under one that holds no matter is known
/// to hold none. Every cell's parent is in the tree, so one pass from the
/// root down, depth by depth, knows each parent's answer before its
/// children's.
fn needed_cells(tree: &CellTree, cache: &impl CellStates) -> Vec<CellKey> {
    let mut empty: CellSet = CellSet::default();
    let mut out = Vec::new();
    for bucket in by_depth(tree.internal.iter().chain(tree.leaves.iter())) {
        for k in bucket {
            let under_empty = k.parent().is_some_and(|p| empty.contains(&p));
            if under_empty || cache.holds_no_matter(&k) {
                empty.insert(k);
                continue;
            }
            if !matches!(
                cache.cell_state(&k),
                Some(CellState::Ready(_) | CellState::Empty | CellState::Gone)
            ) {
                out.push(k);
            }
        }
    }
    out
}

/// `keys` grouped by depth, shallowest first, each group in no particular
/// order.
fn by_depth<'a>(keys: impl Iterator<Item = &'a CellKey>) -> Vec<Vec<CellKey>> {
    let mut buckets: Vec<Vec<CellKey>> = Vec::new();
    for k in keys {
        let d = usize::from(k.depth);
        if buckets.len() <= d {
            buckets.resize_with(d + 1, Vec::new);
        }
        buckets[d].push(*k);
    }
    buckets
}

/// The matter radius of `frame_id` from its depth-0 cell in `cache`, once
/// it has arrived ([`crate::farfield::matter_radius`]).
pub fn depth_zero_radius(cache: &impl CellStates, frame_id: u64) -> Option<f64> {
    match cache.cell_state(&crate::farfield::depth_zero_key(frame_id)) {
        Some(CellState::Ready(c)) => Some(c.radius),
        _ => None,
    }
}

/// What the selection reads of the cells: their fetch states. The cache
/// ([`CellCache`]) provides it on the frame thread, and the selection's own
/// copy of the states ([`CellMap`] of [`CellState`]) where the selection
/// runs on its own thread ([`crate::world`]).
pub trait CellStates {
    /// The state of `key`, `None` if it is not known.
    fn cell_state(&self, key: &CellKey) -> Option<&CellState>;

    /// Returns `true` when `key` may be refined in the target: it is
    /// `Ready` and refinable ([`ReadyCell::refinable`]), or it has not
    /// arrived and its nearest arrived ancestor, at most
    /// [`AHEAD_DEPTHS`]` - 1` depths up, predicts it refinable
    /// ([`ReadyCell::ahead`]).
    fn refinable_or_predicted(&self, key: &CellKey) -> bool {
        match self.cell_state(key) {
            Some(CellState::Ready(c)) => return c.refinable,
            Some(CellState::Empty | CellState::Gone) => return false,
            _ => {}
        }
        let mut up = *key;
        for _ in 1..AHEAD_DEPTHS {
            let Some(a) = up.parent() else {
                return false;
            };
            match self.cell_state(&a) {
                Some(CellState::Ready(c)) => return c.ahead.predicts(&a, key),
                Some(CellState::Empty | CellState::Gone) => return false,
                _ => up = a,
            }
        }
        false
    }

    /// Returns `true` when `key` has arrived (`Ready` or `Empty`).
    fn arrived_or_empty(&self, key: &CellKey) -> bool {
        self.cell_state(key).is_some_and(CellState::arrived)
    }

    /// `key` itself is fetched and holds no matter.
    fn holds_no_matter(&self, key: &CellKey) -> bool {
        match self.cell_state(key) {
            Some(CellState::Empty) => true,
            Some(CellState::Ready(cell)) => !cell.has_matter,
            _ => false,
        }
    }
}

impl CellStates for CellCache {
    fn cell_state(&self, key: &CellKey) -> Option<&CellState> {
        self.state(key)
    }
}

impl CellStates for CellMap<CellState> {
    fn cell_state(&self, key: &CellKey) -> Option<&CellState> {
        self.get(key)
    }
}

/// One cache entry.
#[derive(Clone, Debug, PartialEq)]
pub struct CellEntry {
    /// Fetch state.
    pub state: CellState,
    /// Kept for the far field: requested like a selected cell, never
    /// evicted.
    pub pinned: bool,
    /// Session time the cell was last in a target or drawn, seconds.
    pub last_used: f64,
    /// Session time of the latest request, seconds.
    pub requested_at: f64,
}

impl CellEntry {
    fn new(now: f64) -> CellEntry {
        CellEntry {
            state: CellState::Missing,
            pinned: false,
            last_used: now,
            requested_at: f64::NEG_INFINITY,
        }
    }
}

/// Counts for the overlay.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct CacheCounts {
    /// Cells in the latest drawn selection.
    pub selected: usize,
    /// Selected cells that are `Ready` or `Empty`.
    pub ready: usize,
    /// Cells of the target not arrived yet: requested, or waiting for room
    /// to be requested.
    pub pending: usize,
    /// Requests in flight.
    pub in_flight: usize,
    /// Pinned cells (see [`CellCache::pin`]).
    pub pinned: usize,
    /// Every cell in the cache.
    pub cached: usize,
}

/// The cell cache: one entry per key, the latest selection, and what it
/// needs.
#[derive(Clone, Debug, Default)]
pub struct CellCache {
    entries: CellMap<CellEntry>,
    selection: Vec<SelectedCell>,
    selection_cells: CellSet,
    target: Vec<SelectedCell>,
    target_cells: CellSet,
    needed: Vec<SelectedCell>,
    in_flight: usize,
    pinned: BTreeSet<CellKey>,
    requested_ahead: u64,
    /// A selection or a pin changed what is wanted since
    /// [`CellCache::take_wanted`] last ran.
    wanted_changed: bool,
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

    /// Every entry, in no particular order.
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
        face_neighbor_keys(key).map(|k| self.neighbor_at(k?))
    }

    /// The cells across the 12 edges and 8 corners of `key`, in the order
    /// of [`diagonal_offsets`], found the same way as
    /// [`CellCache::face_neighbors`].
    pub fn diagonal_neighbors(&self, key: &CellKey) -> Vec<Option<(CellKey, NeighborMatter<'_>)>> {
        diagonal_offsets()
            .iter()
            .map(|&o| self.neighbor_at(offset_key(key, o)?))
            .collect()
    }

    /// What is known of cell `k` for padding: `Ready`, known empty, or the
    /// nearest `Ready` coarser cell holding it.
    fn neighbor_at(&self, k: CellKey) -> Option<(CellKey, NeighborMatter<'_>)> {
        match self.state(&k) {
            Some(CellState::Ready(cell)) => Some((k, NeighborMatter::Ready(cell))),
            _ if self.known_empty(&k) => Some((k, NeighborMatter::Empty)),
            _ => {
                // The nearest loaded coarser cell holding the region across
                // the face: extraction reads it at the grid points'
                // positions, so a depth transition joins.
                let mut up = k.parent();
                while let Some(a) = up {
                    if let Some(CellState::Ready(cell)) = self.state(&a) {
                        return Some((a, NeighborMatter::Ready(cell)));
                    }
                    up = a.parent();
                }
                None
            }
        }
    }

    /// Returns `true` when `key` is `Ready` and its children can show more
    /// than it does ([`ReadyCell::refinable`]).
    pub fn refinable(&self, key: &CellKey) -> bool {
        matches!(self.state(key), Some(CellState::Ready(c)) if c.refinable)
    }

    /// Returns `true` when `key` has arrived (`Ready` or `Empty`) or is
    /// known to hold no matter: the selection can stand on it.
    pub fn loaded(&self, key: &CellKey) -> bool {
        self.arrived_or_empty(key) || self.empty_above(key)
    }

    /// Returns `true` when the cache knows that `key` holds no matter: it
    /// holds none itself (`Empty`, or `Ready` without a sample of density
    /// above 0), an ancestor holds none, or its descendants down to
    /// [`COVER_LEVELS`] levels below hold none, covering it. The cell itself
    /// need not be cached: the selection may hold its region at another
    /// depth.
    pub fn known_empty(&self, key: &CellKey) -> bool {
        self.empty_above(key) || self.empty_below(key, COVER_LEVELS)
    }

    /// `key` or one of its ancestors is fetched and holds no matter.
    pub fn empty_above(&self, key: &CellKey) -> bool {
        let mut up = Some(*key);
        while let Some(k) = up {
            if self.holds_no_matter(&k) {
                return true;
            }
            up = k.parent();
        }
        false
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

    /// The latest drawn selection.
    pub fn selection(&self) -> &[SelectedCell] {
        &self.selection
    }

    /// The latest target's leaves.
    pub fn target(&self) -> &[SelectedCell] {
        &self.target
    }

    /// Every cell of the latest target, leaves and the cells above them.
    pub fn target_cells(&self) -> &CellSet {
        &self.target_cells
    }

    /// The latest drawn selection is the target: every cell the target
    /// needs has arrived and the drawn selection has switched to it.
    pub fn at_target(&self) -> bool {
        self.selection.len() == self.target.len()
            && self
                .selection
                .iter()
                .zip(&self.target)
                .all(|(a, b)| a.key == b.key)
    }

    /// Cells of the target that have not arrived: requested, or waiting
    /// for room.
    pub fn outstanding(&self) -> usize {
        self.needed
            .iter()
            .filter(
                |c| !matches!(self.state(&c.key), Some(s) if s.arrived() || *s == CellState::Gone),
            )
            .count()
    }

    /// Cells requested before their parent had arrived, this session: what
    /// asking ahead requested (`rendering.md` 3.2).
    pub fn requested_ahead(&self) -> u64 {
        self.requested_ahead
    }

    /// Replaces the selection with `selection` (see [`select_all`]).
    pub fn set_selection(&mut self, selection: Selection) {
        self.selection = selection.drawn;
        self.selection_cells = selection.drawn_cells;
        self.target = selection.target;
        self.target_cells = selection.target_cells;
        self.needed = selection.needed;
        self.wanted_changed = true;
    }

    /// Replaces the selection with these cells as both the drawn selection
    /// and the target, all of them needed: for tests and tools that pick
    /// cells by hand.
    pub fn set_cells(&mut self, cells: Vec<SelectedCell>) {
        let mut needed = cells.clone();
        needed.sort_by(|a, b| {
            b.projected_px
                .total_cmp(&a.projected_px)
                .then(a.key.cmp(&b.key))
        });
        let tree = CellTree::from_leaves(cells.iter().map(|c| c.key));
        self.set_selection(Selection {
            drawn_cells: cells.iter().map(|c| c.key).collect(),
            drawn: cells.clone(),
            target: cells,
            target_cells: tree.leaves.into_iter().chain(tree.internal).collect(),
            needed,
        });
    }

    /// Pins a cell: it is requested like a selected cell (if it is not
    /// cached yet) and never evicted. Returns `true` if it was not pinned
    /// before.
    pub fn pin(&mut self, key: CellKey, now: f64) -> bool {
        let e = self
            .entries
            .entry(key)
            .or_insert_with(|| CellEntry::new(now));
        self.pinned.insert(key);
        self.wanted_changed = true;
        !std::mem::replace(&mut e.pinned, true)
    }

    /// The pinned cells, in key order.
    pub fn pinned(&self) -> impl Iterator<Item = &CellKey> {
        self.pinned.iter()
    }

    /// Picks the next requests: the target's cells that have not arrived,
    /// largest projected size first (ties in key order), then pinned cells
    /// not yet fetched, up to [`MAX_IN_FLIGHT`] in flight. Marks them
    /// `Requested`.
    pub fn take_requests(&mut self, now: f64) -> Vec<CellKey> {
        let mut room = MAX_IN_FLIGHT.saturating_sub(self.in_flight);
        let mut out = Vec::new();
        if room == 0 {
            return out;
        }
        let candidates: Vec<CellKey> = self
            .needed
            .iter()
            .map(|c| c.key)
            .chain(self.pinned.iter().copied())
            .collect();
        for k in candidates {
            if room == 0 {
                break;
            }
            let ahead = k
                .parent()
                .is_some_and(|p| !self.state(&p).is_some_and(CellState::arrived));
            let e = self.entries.entry(k).or_insert_with(|| CellEntry::new(now));
            if e.state != CellState::Missing {
                continue;
            }
            e.state = CellState::Requested;
            e.requested_at = now;
            e.last_used = now;
            self.in_flight += 1;
            room -= 1;
            if ahead {
                self.requested_ahead += 1;
            }
            out.push(k);
        }
        out
    }

    /// The wanted set (`client-compute.md` section 2.8), once per new
    /// selection or pin: the needed cells of the target that have not
    /// arrived, largest projected size first with it as priority, then
    /// pinned cells not yet fetched at
    /// [`crate::directions::REQUESTED_PRIORITY`], at most [`MAX_IN_FLIGHT`]
    /// of them. The cap keeps a selection from asking for more than one
    /// round trip brings in, which the per-frame budget of [`select_all`]
    /// adapts to (`tests/budget.rs`); the rest is wanted by a later
    /// selection once these have arrived. Marks
    /// them `Requested`; a cell requested before and left out now goes
    /// back to `Missing`, since the fetcher drops it. `None` when nothing
    /// changed since the last call.
    pub fn take_wanted(&mut self, now: f64) -> Option<Vec<(CellKey, f64)>> {
        if !std::mem::take(&mut self.wanted_changed) {
            return None;
        }
        let mut out: Vec<(CellKey, f64)> = Vec::new();
        let mut seen = CellSet::default();
        let candidates: Vec<(CellKey, f64)> = self
            .needed
            .iter()
            .map(|c| (c.key, c.projected_px))
            .chain(
                self.pinned
                    .iter()
                    .map(|k| (*k, crate::directions::REQUESTED_PRIORITY)),
            )
            .collect();
        for (k, p) in candidates {
            if out.len() >= MAX_IN_FLIGHT {
                break;
            }
            if !seen.insert(k) {
                continue;
            }
            let ahead = k
                .parent()
                .is_some_and(|p| !self.state(&p).is_some_and(CellState::arrived));
            let e = self.entries.entry(k).or_insert_with(|| CellEntry::new(now));
            match e.state {
                CellState::Missing => {
                    e.state = CellState::Requested;
                    e.requested_at = now;
                    self.in_flight += 1;
                    if ahead {
                        self.requested_ahead += 1;
                    }
                }
                CellState::Requested => {}
                _ => continue,
            }
            e.last_used = now;
            out.push((k, p));
        }
        let mut dropped = 0;
        for (k, e) in self.entries.iter_mut() {
            if e.state == CellState::Requested && !seen.contains(k) {
                e.state = CellState::Missing;
                dropped += 1;
            }
        }
        self.in_flight = self.in_flight.saturating_sub(dropped);
        Some(out)
    }

    /// Applies the result of a request for `key`.
    pub fn complete(&mut self, key: CellKey, outcome: FetchOutcome, now: f64) {
        let e = self
            .entries
            .entry(key)
            .or_insert_with(|| CellEntry::new(now));
        if e.state == CellState::Requested {
            self.in_flight = self.in_flight.saturating_sub(1);
        }
        e.state = match outcome {
            FetchOutcome::Decoded(Some(cell)) => CellState::Ready(cell),
            FetchOutcome::Decoded(None) => CellState::Empty,
            FetchOutcome::Invalid(reason) => {
                tracing::warn!(key = %key, "cell is invalid: {reason}");
                CellState::Gone
            }
        };
        e.last_used = now;
    }

    /// Puts `cell` in place of `key`'s `Ready` cell, composited again
    /// from its shown layers (`rendering.md` section 7 item 7). Does
    /// nothing when `key` is not `Ready`.
    pub fn replace_ready(&mut self, key: &CellKey, cell: Arc<ReadyCell>) {
        if let Some(e) = self.entries.get_mut(key) {
            if matches!(e.state, CellState::Ready(_)) {
                e.state = CellState::Ready(cell);
            }
        }
    }

    /// The `Ready` cells, in key order.
    pub fn ready_cells(&self) -> impl Iterator<Item = (&CellKey, &Arc<ReadyCell>)> {
        self.entries.iter().filter_map(|(k, e)| match &e.state {
            CellState::Ready(c) => Some((k, c)),
            _ => None,
        })
    }

    /// Bytes held by the cached cells' decoded samples
    /// ([`ReadyCell::sample_bytes`]).
    pub fn sample_bytes(&self) -> usize {
        self.ready_cells().map(|(_, c)| c.sample_bytes()).sum()
    }

    /// Bytes held by the cached cells' composites alone, at
    /// [`SAMPLE_BYTES`] per sample: what the cache held before cells kept
    /// their layers' sections.
    pub fn composite_bytes(&self) -> usize {
        let len = |c: &ReadyCell| c.section.samples().map_or(0, |v| v.len());
        self.ready_cells().map(|(_, c)| len(c)).sum::<usize>() * SAMPLE_BYTES
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
    /// order). Cells of the latest target (which holds the drawn selection
    /// and every ancestor of it), pinned cells, and requests in flight are
    /// never evicted and count as used now. Returns the evicted keys in key
    /// order.
    pub fn evict(&mut self, now: f64) -> Vec<CellKey> {
        for k in self
            .target_cells
            .iter()
            .chain(self.selection.iter().map(|s| &s.key))
        {
            if let Some(e) = self.entries.get_mut(k) {
                e.last_used = now;
            }
        }
        let keep = |k: &CellKey, e: &CellEntry, cells: &CellSet| {
            e.pinned || e.state == CellState::Requested || cells.contains(k)
        };
        let mut gone: Vec<CellKey> = self
            .entries
            .iter()
            .filter(|(k, e)| {
                !keep(k, e, &self.target_cells) && now - e.last_used >= EVICT_AFTER_SECONDS
            })
            .map(|(k, _)| *k)
            .collect();
        let remaining = self.entries.len() - gone.len();
        if remaining > MAX_CACHED_CELLS {
            let dropped: CellSet = gone.iter().copied().collect();
            let mut lru: Vec<(f64, CellKey)> = self
                .entries
                .iter()
                .filter(|(k, e)| !keep(k, e, &self.target_cells) && !dropped.contains(k))
                .map(|(k, e)| (e.last_used, *k))
                .collect();
            lru.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            gone.extend(
                lru.into_iter()
                    .take(remaining - MAX_CACHED_CELLS)
                    .map(|(_, k)| k),
            );
        }
        gone.sort();
        for k in &gone {
            self.entries.remove(k);
        }
        gone
    }

    /// Counts for the overlay.
    pub fn counts(&self) -> CacheCounts {
        CacheCounts {
            selected: self.selection.len(),
            ready: self
                .selection
                .iter()
                .filter(|s| self.loaded(&s.key))
                .count(),
            pending: self.outstanding(),
            in_flight: self.in_flight,
            pinned: self.pinned.len(),
            cached: self.entries.len(),
        }
    }

    /// The cells to draw for the latest selection, in key order, applying
    /// the rule of the module docs. `drawable` says whether a cell can be
    /// drawn now (its mesh is ready, or it holds no matter).
    ///
    /// A selected cell that cannot be drawn is drawn by a stand-in: its
    /// drawable descendants within [`COVER_LEVELS`] if they cover it, else
    /// its nearest drawable ancestor. An ancestor drawn as a stand-in hides
    /// every selected cell below it, with their stand-ins, and every
    /// ancestor stand-in below it. Selected cells never hold one another,
    /// so the work beyond one look per selected cell is proportional to
    /// what the ancestor stand-ins hide: the cells below such an ancestor
    /// are found by walking down the target from it, which holds every
    /// cell above a selected one and stops at the selected ones.
    pub fn draw_set(&self, drawable: impl Fn(&CellKey) -> bool) -> Vec<CellKey> {
        let mut out: Vec<CellKey> = Vec::with_capacity(self.selection.len());
        let mut covers: Vec<(CellKey, Vec<CellKey>)> = Vec::new();
        let mut above: CellSet = CellSet::default();
        let mut walked: CellMap<Option<CellKey>> = CellMap::default();
        for s in &self.selection {
            if drawable(&s.key) {
                out.push(s.key);
                continue;
            }
            if let Some(cover) = cover_below(&s.key, COVER_LEVELS, &drawable) {
                covers.push((s.key, cover));
                continue;
            }
            // Siblings share the walk up: remember it per parent.
            let Some(parent) = s.key.parent() else {
                continue;
            };
            let found = *walked.entry(parent).or_insert_with(|| {
                let mut up = Some(parent);
                while let Some(p) = up {
                    if drawable(&p) {
                        return Some(p);
                    }
                    up = p.parent();
                }
                None
            });
            if let Some(p) = found {
                above.insert(p);
            }
        }
        if !above.is_empty() {
            let mut hidden: CellSet = CellSet::default();
            for a in &above {
                let mut up = a.parent();
                while let Some(p) = up {
                    if above.contains(&p) {
                        hidden.insert(*a);
                        break;
                    }
                    up = p.parent();
                }
                let mut stack: Vec<CellKey> = vec![*a];
                while let Some(n) = stack.pop() {
                    for c in n.children().into_iter().flatten() {
                        if self.selection_cells.contains(&c) {
                            hidden.insert(c);
                        } else if self.target_cells.contains(&c) {
                            stack.push(c);
                        }
                    }
                }
            }
            out.retain(|k| !hidden.contains(k));
            covers.retain(|(s, _)| !hidden.contains(s));
            out.extend(above.iter().filter(|k| !hidden.contains(k)));
        }
        out.extend(covers.into_iter().flat_map(|(_, cover)| cover));
        out.sort_unstable();
        out
    }
}

/// The ancestor of `key` at `depth` (at most `key.depth`), or `None` deeper.
pub fn ancestor_at(key: &CellKey, depth: u8) -> Option<CellKey> {
    let shift = key.depth.checked_sub(depth)?;
    CellKey::new(
        key.frame_id,
        depth,
        key.x >> shift,
        key.y >> shift,
        key.z >> shift,
    )
    .ok()
}

/// The offsets of the 12 edge and 8 corner neighbors of a cell: every
/// `(dx, dy, dz)` in `{-1, 0, 1}^3` with two or three nonzero, in
/// ascending order of `dz`, then `dy`, then `dx`.
pub fn diagonal_offsets() -> Vec<[i8; 3]> {
    AROUND
        .into_iter()
        .filter(|o| o.iter().filter(|&&d| d != 0).count() >= 2)
        .collect()
}

/// The cell at offset `o` from `key` at the same depth, or `None` past the
/// root cube.
pub fn offset_key(key: &CellKey, o: [i8; 3]) -> Option<CellKey> {
    let c = [key.x, key.y, key.z];
    let mut n = [0u32; 3];
    for a in 0..3 {
        n[a] = match o[a] {
            0 => c[a],
            d if d > 0 => c[a].checked_add(1)?,
            _ => c[a].checked_sub(1)?,
        };
    }
    CellKey::new(key.frame_id, key.depth, n[0], n[1], n[2]).ok()
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
    /// A surface tile request finished (`matter-format.md` sections 10
    /// and 12): the base surface ready to draw (`None` when every
    /// non-overlay layer's tile is empty) with the overlays so far, or why
    /// it is invalid.
    Surface {
        /// The tile.
        key: gx_core::key::SurfaceKey,
        /// What came back, decoded into geometry and overlays.
        outcome: Result<crate::tiles::DecodedTile, String>,
        /// `true` while overlay layers are still to come: the tile draws
        /// now and the request finishes again when one arrives.
        partial: bool,
        /// Section bytes received.
        bytes: u64,
    },
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
            // Surface tiles count toward the bytes received.
            if let FetchEvent::Surface { bytes, .. } = e {
                self.bytes += bytes;
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
pub use native::{frame_records, Fetcher, FetcherOptions};

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use super::{
        cell_from_arrived, directions_time, FetchEvent, FetchOutcome, FetchTally, FetcherStats,
    };
    use crate::directions::{Action, Arrived, Coordinator, Origin, StreamResult};
    use crate::hub::HubClient;
    use crate::protocol::DirectionsResponse;
    use anyhow::{anyhow, Result};
    use gx_core::frames::FrameSystem;
    use gx_core::key::{CellKey, ChunkKey};
    use gx_core::sha256::sha256;
    use gx_core::sha256::to_hex;
    use gx_core::units::Meters;
    use gx_module_host::HostConfig;
    use gx_worker::job::check_blob;
    use gx_worker::{
        BlobKind, Blobs, CheckJob, CheckLane, CheckOutcome, CheckRunner, ComputeJob, Computed,
        DiskCache, Lane, LaneSource, UploadLane, UploadOffer, UploadOutcome, Uploader, WorkerError,
        WorkerPool, WorkerStats, MAX_UPLOADS_IN_FLIGHT,
    };
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::mpsc;
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};

    /// How the fetcher's worker and cache are set up.
    #[derive(Clone, Debug)]
    pub struct FetcherOptions {
        /// The local cache, or `None` to keep nothing on disk.
        pub cache: Option<Arc<DiskCache>>,
        /// Worker threads ([`crate::extract::worker_threads`] by default).
        pub worker_threads: usize,
        /// Memory and time limits of each module call.
        pub host: HostConfig,
    }

    impl Default for FetcherOptions {
        fn default() -> FetcherOptions {
            FetcherOptions {
                cache: None,
                worker_threads: crate::extract::worker_threads(),
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

    /// The worker's view of the hub for input sections (`client-compute.md`
    /// section 10.3): input jobs from the build and the frame records,
    /// directions for one key, and streams, as blocking calls on the
    /// runtime from the pool's threads, never from the frame thread.
    struct HubInputs {
        handle: tokio::runtime::Handle,
        client: Arc<HubClient>,
        /// The frame records the build lists, by hash, once the registry
        /// has loaded.
        records: Mutex<Arc<crate::protocol::FrameRecords>>,
        /// The space's origin, once directions have told it.
        origin: Mutex<Option<i64>>,
    }

    impl HubInputs {
        fn origin(&self) -> Result<i64, WorkerError> {
            self.origin
                .lock()
                .expect("origin")
                .ok_or_else(|| WorkerError::transient("the origin is not known yet"))
        }
    }

    impl gx_worker::InputSource for HubInputs {
        fn input_job(
            &self,
            reader: &ComputeJob,
            slot: u32,
            key: ChunkKey,
        ) -> Result<ComputeJob, WorkerError> {
            let records = Arc::clone(&self.records.lock().expect("records"));
            self.client
                .build()
                .input_job(reader, slot, key, self.origin()?, &records)
                .map_err(|e| WorkerError::permanent(format!("input {key} of slot {slot}: {e:#}")))
        }

        fn direct(&self, job: &ComputeJob) -> Result<gx_worker::InputDirection, WorkerError> {
            let t = crate::protocol::input_directions_time(
                job,
                self.origin()?,
                self.client.now_utc_ms(),
            );
            let key = job.key();
            let response = self
                .handle
                .block_on(self.client.directions(t, &[key]))
                .map_err(|e| {
                    WorkerError::transient(format!("directions for input {key}: {e:#}"))
                })?;
            crate::protocol::input_direction(&response, job)
        }

        fn stream(&self, address: &[u8; 32], sha256: &[u8; 32]) -> Result<Vec<u8>, WorkerError> {
            match self
                .handle
                .block_on(self.client.get_section(address, sha256))
            {
                Ok(Some(b)) => Ok(b),
                Ok(None) => Err(WorkerError::transient(format!(
                    "input section {}: not found",
                    to_hex(address)
                ))),
                Err(e) => Err(WorkerError::transient(format!("{e:#}"))),
            }
        }
    }

    /// Puts computed `missing` sections into the upload lane and starts
    /// sending ([`UploadSender::pump`]). Never blocks the worker.
    struct HubUploader {
        lane: Arc<UploadLane>,
        sender: Arc<UploadSender>,
    }

    impl Uploader for HubUploader {
        fn offer(&self, job: &ComputeJob, section: Arc<[u8]>) {
            self.lane.offer(UploadOffer {
                job: job.clone(),
                bytes: section,
            });
            self.sender.pump();
        }
    }

    /// Sends the upload lane (`client-compute.md` section 2.8): each offer
    /// sent once, at most [`MAX_UPLOADS_IN_FLIGHT`] at a time, on the async
    /// runtime. An upload is network I/O and never takes a pool thread, so
    /// it does not wait for computes: waiting let a burst of computes
    /// overflow the bounded lane and drop most offers, and the hub never
    /// filled. [`UploadSender::pump`] runs when an offer arrives and when a
    /// request ends.
    struct UploadSender {
        handle: tokio::runtime::Handle,
        client: Arc<HubClient>,
        lane: Arc<UploadLane>,
        in_flight: AtomicUsize,
    }

    /// Frees an upload slot when the request ends, however it ends
    /// (answered, failed, or cancelled with the runtime), and sends the
    /// next offer.
    struct UploadSlot {
        sender: Arc<UploadSender>,
    }

    impl Drop for UploadSlot {
        fn drop(&mut self) {
            self.sender.in_flight.fetch_sub(1, Ordering::SeqCst);
            UploadSender::pump(&self.sender);
        }
    }

    impl UploadSender {
        /// Starts offers until [`MAX_UPLOADS_IN_FLIGHT`] are in flight or
        /// the lane is empty.
        fn pump(self: &Arc<Self>) {
            loop {
                let reserved = self
                    .in_flight
                    .try_update(Ordering::SeqCst, Ordering::SeqCst, |n| {
                        (n < MAX_UPLOADS_IN_FLIGHT).then_some(n + 1)
                    })
                    .is_ok();
                if !reserved {
                    return;
                }
                let Some((offer, ticket)) = self.lane.take() else {
                    self.in_flight.fetch_sub(1, Ordering::SeqCst);
                    return;
                };
                let slot = UploadSlot {
                    sender: Arc::clone(self),
                };
                let client = self.client.clone();
                self.handle.spawn(async move {
                    let _slot = slot;
                    let key = offer.job.key();
                    let outcome = match client.upload(&offer.job, offer.bytes.to_vec()).await {
                        Ok(status) => {
                            tracing::debug!(key = %key, status, "section offered");
                            UploadOutcome::from_status(status)
                        }
                        Err(e) => {
                            tracing::debug!(key = %key, "upload failed, not retried: {e:#}");
                            UploadOutcome::Failed
                        }
                    };
                    ticket.finish(outcome);
                });
            }
        }
    }

    /// The pool's check lane (`client-compute.md` sections 2.6 and 2.8):
    /// taken only when no compute and no upload waits. The recompute runs
    /// on the pool thread, the report on the runtime. A recompute whose
    /// hash differs from the drawn bytes is handed back to the fetcher to
    /// take the drawn section's place, whatever the hub answers.
    struct CheckSender {
        handle: tokio::runtime::Handle,
        client: Arc<HubClient>,
        lane: Arc<CheckLane>,
        runner: CheckRunner,
        tx: Mutex<mpsc::Sender<Internal>>,
    }

    impl LaneSource for CheckSender {
        fn next(&self) -> Option<Box<dyn FnOnce() + Send>> {
            let check = self.lane.take()?;
            let (handle, client, runner) = (
                self.handle.clone(),
                self.client.clone(),
                self.runner.clone(),
            );
            let stats = Arc::clone(self.lane.stats());
            let tx = self.tx.lock().expect("check sender").clone();
            Some(Box::new(move || {
                let key = check.job.key();
                let computed = match runner.run(&check.job) {
                    Ok(c) => c,
                    Err(e) => {
                        tracing::debug!(key = %key, "check recompute failed: {e}");
                        stats.check_skipped();
                        return;
                    }
                };
                stats.check_run();
                let hash = sha256(&computed.bytes);
                let mut check = check;
                check.job.inputs_sha256 = computed.inputs_sha256;
                if hash != check.drawn {
                    tracing::info!(key = %key, "a checked section differs; drawing the computed one");
                    let _ = tx.send(Internal::Replace {
                        key,
                        layer: check.layer_index,
                        address: check.job.address,
                        sha256: check.drawn,
                        computed,
                    });
                }
                handle.spawn(async move {
                    let outcome = match client.check(&check.job, &hash).await {
                        Ok(status) => CheckOutcome::from_status(status),
                        Err(e) => {
                            tracing::debug!(key = %key, "check report failed: {e:#}");
                            CheckOutcome::Failed
                        }
                    };
                    stats.check_finished(outcome);
                });
            }))
        }
    }

    /// What the fetcher's tasks report back.
    enum Internal {
        Replace {
            key: ChunkKey,
            layer: usize,
            address: [u8; 32],
            sha256: [u8; 32],
            computed: Computed,
        },
        Directions(Vec<ChunkKey>, Result<DirectionsResponse, String>),
        Stream {
            key: ChunkKey,
            layer: usize,
            address: [u8; 32],
            sha256: [u8; 32],
            result: StreamResult,
        },
        Compute {
            key: ChunkKey,
            layer: usize,
            address: [u8; 32],
            result: Result<Computed, WorkerError>,
        },
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
        checks: Arc<CheckLane>,
        inputs: Arc<HubInputs>,
        /// The directions time each key in flight was last asked about at.
        asked_at: BTreeMap<ChunkKey, i64>,
        /// The frame edge of each checked cell, to ask for it again when
        /// its check replaces a section.
        checked_extents: BTreeMap<CellKey, Meters>,
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
        /// Starts a fetcher on `handle`: the worker pool, the upload lane
        /// and its task, and the task that keeps the hub's clock measured
        /// ([`HubClient::keep_clock`]).
        pub fn new(
            handle: tokio::runtime::Handle,
            client: Arc<HubClient>,
            options: FetcherOptions,
        ) -> Result<Fetcher> {
            let stats = WorkerStats::new();
            let lane = Arc::new(UploadLane::new(Arc::clone(&stats)));
            let checks = Arc::new(CheckLane::new(Arc::clone(&stats)));
            let pool = WorkerPool::with_stats(
                options.worker_threads,
                options.host,
                Arc::new(EdgeBlobs {
                    handle: handle.clone(),
                    client: client.clone(),
                }),
                options.cache.clone(),
                Arc::new(HubUploader {
                    lane: Arc::clone(&lane),
                    sender: Arc::new(UploadSender {
                        handle: handle.clone(),
                        client: client.clone(),
                        lane: Arc::clone(&lane),
                        in_flight: AtomicUsize::new(0),
                    }),
                }),
                stats,
            )
            .map_err(|e| anyhow!("starting the worker: {e}"))?;
            let (tx, rx) = mpsc::channel();
            pool.set_lane(
                Lane::Check,
                Arc::new(CheckSender {
                    handle: handle.clone(),
                    client: client.clone(),
                    lane: Arc::clone(&checks),
                    runner: pool.check_runner(),
                    tx: Mutex::new(tx.clone()),
                }),
            );
            let inputs = Arc::new(HubInputs {
                handle: handle.clone(),
                client: client.clone(),
                records: Mutex::new(Arc::default()),
                origin: Mutex::new(None),
            });
            pool.set_inputs(Arc::clone(&inputs) as Arc<dyn gx_worker::InputSource>);
            tracing::info!(threads = pool.size(), "worker started");
            let mut coordinator = Coordinator::new();
            coordinator.set_overlay_layers(client.build().overlay_layers());
            coordinator.set_hidden_layers(client.build().initially_hidden_layers(), 0.0);
            // Seeded once per run, never from an address: it only chooses
            // which downloads to check and never touches bytes.
            coordinator.set_check_seed(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_or(0, |d| d.as_nanos() as u64)
                    ^ u64::from(std::process::id()),
            );
            client.keep_clock(&handle);
            Ok(Fetcher {
                handle,
                client,
                cache: options.cache,
                pool,
                coordinator,
                checks,
                inputs,
                asked_at: BTreeMap::new(),
                checked_extents: BTreeMap::new(),
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

        /// Replaces the wanted cells (`client-compute.md` section 2.8) with
        /// `cells`, each a cell of a frame of edge `root_extent` with its
        /// priority (projected size, pixels). Directions and streams follow
        /// the wanted set, highest priority first; a cell left out that has
        /// nothing in flight is dropped.
        pub fn set_wanted_cells(&mut self, cells: Vec<(CellKey, Meters, f64)>) {
            let now = self.now();
            let mut wanted = Vec::with_capacity(cells.len());
            for (key, root_extent, priority) in cells {
                self.extents.insert(key, root_extent);
                wanted.push((key, priority));
            }
            self.coordinator.set_wanted_cells(wanted, now);
            self.follow_wanted();
        }

        /// Gives the worker's waiting jobs their keys' new priorities and
        /// drops those whose key left the wanted set; each dropped job
        /// reports back through the queue like any other.
        fn follow_wanted(&mut self) {
            let coordinator = &self.coordinator;
            self.pool.reprioritize(|k| coordinator.compute_priority(k));
        }

        /// Replaces the wanted surface tiles with `tiles`, each with its
        /// priority (projected size, pixels), like
        /// [`Self::set_wanted_cells`].
        pub fn set_wanted_surfaces(&mut self, tiles: Vec<(gx_core::key::SurfaceKey, f64)>) {
            let now = self.now();
            self.coordinator.set_wanted_surfaces(tiles, now);
            self.follow_wanted();
        }

        /// The hidden layers, by index (`rendering.md` section 7 item 7):
        /// a hidden overlay is not fetched for drawing, and one shown again
        /// is wanted at once for every wanted surface tile
        /// ([`Coordinator::set_hidden_layers`]).
        pub fn set_hidden_layers(&mut self, hidden: std::collections::BTreeSet<usize>) {
            let now = self.now();
            self.coordinator.set_hidden_layers(hidden, now);
            self.pump();
        }

        /// Queues a request for a surface tile; the result arrives as
        /// [`FetchEvent::Surface`] through [`Fetcher::drain`].
        pub fn request_surface(&mut self, key: gx_core::key::SurfaceKey) {
            let now = self.now();
            self.coordinator.request(ChunkKey::Surface(key), now);
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
                Internal::Replace {
                    key,
                    layer,
                    address,
                    sha256,
                    computed,
                } => {
                    if let ChunkKey::Cell(cell) = key {
                        if let Some(e) = self.checked_extents.remove(&cell) {
                            self.extents.entry(cell).or_insert(e);
                        }
                    }
                    self.coordinator
                        .replace(key, layer, address, sha256, computed.into(), now)
                }
                Internal::Directions(keys, r) => self.coordinator.directions_done(&keys, r, now),
                Internal::Stream {
                    key,
                    layer,
                    address,
                    sha256,
                    result,
                } => self
                    .coordinator
                    .stream_done(&key, layer, address, sha256, result, now),
                Internal::Compute {
                    key,
                    layer,
                    address,
                    result,
                } => self
                    .coordinator
                    .compute_done(&key, layer, address, result, now),
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
            // Input work on the pool needs the origin (section 10.3).
            if let Some(origin) = self.coordinator.origin_utc_ms() {
                *self.inputs.origin.lock().expect("origin") = Some(origin);
            }
            if let Action::Complete(crate::directions::Completion { key, .. })
            | Action::Failed { key, .. } = &action
            {
                self.asked_at.remove(key);
            }
            match action {
                Action::Directions(keys) => {
                    let t = directions_time(
                        self.wanted_utc_ms,
                        self.coordinator.origin_utc_ms(),
                        self.client.now_utc_ms(),
                    );
                    for k in &keys {
                        self.asked_at.insert(*k, t);
                    }
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
                    sha256,
                } => {
                    let client = self.client.clone();
                    let cache = self.cache.clone();
                    let tx = self.tx.clone();
                    let resolver = self.pool.input_resolver();
                    self.handle.spawn(async move {
                        let result =
                            shared_stream(&resolver, &client, cache, key, address, sha256).await;
                        let _ = tx.send(Internal::Stream {
                            key,
                            layer,
                            address,
                            sha256,
                            result,
                        });
                    });
                }
                Action::Compute {
                    key,
                    layer,
                    job,
                    priority,
                } => {
                    let mut job = *job;
                    let tx = self.tx.clone();
                    let address = job.address;
                    // A layer with inputs takes its slots and frame records
                    // from the build (section 10).
                    let records = Arc::clone(&self.inputs.records.lock().expect("records"));
                    let origin = self.coordinator.origin_utc_ms().unwrap_or(0);
                    if let Err(e) = self.client.build().complete_job(&mut job, origin, &records) {
                        let _ = tx.send(Internal::Compute {
                            key,
                            layer,
                            address,
                            result: Err(WorkerError::permanent(format!("{e:#}"))),
                        });
                        return;
                    }
                    self.pool.submit(job, priority, move |result| {
                        let _ = tx.send(Internal::Compute {
                            key,
                            layer,
                            address,
                            result,
                        });
                    });
                }
                Action::Check {
                    key,
                    layer,
                    address,
                    sha256,
                } => {
                    let records = Arc::clone(&self.inputs.records.lock().expect("records"));
                    let job = match (self.asked_at.get(&key), self.coordinator.origin_utc_ms()) {
                        (Some(&t), Some(origin)) => crate::protocol::check_job(
                            self.client.build(),
                            layer,
                            key,
                            address,
                            t,
                            origin,
                            Some(&records),
                        ),
                        _ => None,
                    };
                    if let ChunkKey::Cell(cell) = key {
                        if let Some(e) = self.extents.get(&cell) {
                            self.checked_extents.insert(cell, *e);
                        }
                    }
                    match job {
                        Some(job) => {
                            if self.checks.push(CheckJob {
                                job,
                                layer_index: layer,
                                drawn: sha256,
                            }) {
                                self.pool.wake();
                            }
                        }
                        None => self.pool.stats_handle().check_skipped(),
                    }
                }
                Action::Complete(done) => match done.key {
                    ChunkKey::Registry => self.out.push(FetchEvent::Registry(Ok(done
                        .sections
                        .into_iter()
                        .map(|a| a.bytes)
                        .collect()))),
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
                            let outcome =
                                cell_from_arrived(&key, &done.sections, &done.layers, extent);
                            let _ = tx.send(Internal::Event(FetchEvent::Completed {
                                key,
                                outcome,
                                bytes,
                                round_trip,
                            }));
                        });
                    }
                    ChunkKey::Surface(key) => {
                        let bytes = done.sections.iter().map(|s| s.len() as u64).sum();
                        let tx = self.tx.clone();
                        let overlays = self.client.build().overlay_layers();
                        self.decoding += 1;
                        self.handle.spawn_blocking(move || {
                            let sections: Vec<Arc<[u8]>> =
                                done.sections.iter().map(|a| a.bytes.clone()).collect();
                            let outcome =
                                crate::tiles::decode_tile(&key, &sections, &done.layers, &overlays);
                            let _ = tx.send(Internal::Event(FetchEvent::Surface {
                                key,
                                outcome,
                                partial: done.partial,
                                bytes,
                            }));
                        });
                    }

                    ChunkKey::Frame(_) => {}
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
                    ChunkKey::Surface(key) => self.out.push(FetchEvent::Surface {
                        key,
                        outcome: Err(reason),
                        partial: false,
                        bytes: 0,
                    }),
                    // The renderer never asks for a frame key: frame
                    // records come from the build (section 10.2).
                    ChunkKey::Frame(_) => {}
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

        /// Waits up to `timeout` until at least one event can be drained,
        /// without taking it: the next [`Fetcher::drain`] gets it.
        pub fn wait_for_events(&mut self, timeout: Duration) {
            let deadline = Instant::now() + timeout;
            loop {
                self.pump();
                let now = Instant::now();
                if !self.out.is_empty() || now >= deadline {
                    return;
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
            // Frame records come from the build by hash, before anything
            // runs: a layer with a frames slot passes them to its module,
            // for the registry key too (section 10.2).
            let records = self
                .handle
                .block_on(frame_records(&self.client, self.cache.as_deref()))?;
            *self.inputs.records.lock().expect("records") = Arc::new(
                records
                    .iter()
                    .map(|(_, b)| (sha256(b), Arc::clone(b)))
                    .collect(),
            );
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
                        crate::protocol::check_frame_records(&sections, &records).map_err(|e| {
                            tracing::error!("refusing the build: {e}");
                            anyhow!("refusing the build: {e}")
                        })?;
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

    /// Every frame record the build lists (`client-compute.md` section
    /// 10.2), as `(frame id, bytes)`: the local cache if it holds bytes
    /// with the right hash, else `GET /frame-record/{hash}` from the edge,
    /// checked against the hash and cached like any content-addressed
    /// bytes. Fails for a record that is missing or does not hash to its
    /// name.
    pub async fn frame_records(
        client: &HubClient,
        cache: Option<&DiskCache>,
    ) -> Result<Vec<(u64, Arc<[u8]>)>> {
        let mut out = Vec::new();
        for (frame_id, hash) in client.build().frame_record_hashes()? {
            let name = to_hex(&hash);
            let kind = BlobKind::FrameRecord;
            let cached = cache
                .and_then(|c| c.get(kind, &name))
                .filter(|b| check_blob(kind, &hash, b).is_ok());
            let bytes = match cached {
                Some(b) => b,
                None => {
                    let b = client
                        .get_blob(kind, &name)
                        .await?
                        .ok_or_else(|| anyhow!("frame record {name} is not stored"))?;
                    check_blob(kind, &hash, &b).map_err(|e| anyhow!("frame record {name}: {e}"))?;
                    if let Some(c) = cache {
                        if let Err(e) = c.put(kind, &name, &b) {
                            tracing::warn!(name, "could not cache the frame record: {e}");
                        }
                    }
                    b
                }
            };
            out.push((frame_id, Arc::from(bytes)));
        }
        Ok(out)
    }

    /// Decodes a cell's section on a blocking task: the one validation it
    /// gets (`rendering.md` 3.5). The registry key is passed through as
    /// bytes; [`crate::protocol::frame_system_from_registries`] decodes it.
    async fn decode_arrived(key: ChunkKey, bytes: Vec<u8>) -> Result<Arrived, String> {
        let bytes: Arc<[u8]> = Arc::from(bytes);
        let ChunkKey::Cell(cell) = key else {
            return Ok(Arrived::raw(bytes));
        };
        let raw = bytes.clone();
        tokio::task::spawn_blocking(move || {
            gx_core::matter::decode(&cell, &raw)
                .map(|s| Arrived {
                    bytes: raw.clone(),
                    decoded: Some(Arc::new(s)),
                })
                .map_err(|e| e.to_string())
        })
        .await
        .map_err(|e| format!("decoding task: {e}"))?
    }

    /// [`stream_section`] under the address's flight in the worker's
    /// shared input resolver (`client-compute.md` section 10.3): a `ready`
    /// section being streamed as an input is waited for and taken from
    /// memory, not downloaded a second time, and one streamed here is
    /// remembered for input work.
    async fn shared_stream(
        resolver: &gx_worker::InputResolver,
        client: &HubClient,
        cache: Option<Arc<DiskCache>>,
        key: ChunkKey,
        address: [u8; 32],
        expected: [u8; 32],
    ) -> StreamResult {
        let _flight = resolver.flights().acquire(&address, 0.0, |_| {}).await;
        if let Some(bytes) = resolver.remembered(&address) {
            if sha256(&bytes) == expected {
                if let Ok(a) = decode_arrived(key, bytes.to_vec()).await {
                    return StreamResult::Bytes(a, Origin::Cache);
                }
            }
        }
        let result = stream_section(client, cache, key, address, expected).await;
        if let StreamResult::Bytes(a, _) = &result {
            resolver.remember(&address, a.bytes.clone());
        }
        result
    }

    /// Gets a stored section: the local cache if it holds a valid copy,
    /// else the edge, caching what comes back once it has decoded. Each
    /// section is decoded, and so validated, once on its way in, off the
    /// frame thread.
    async fn stream_section(
        client: &HubClient,
        cache: Option<Arc<DiskCache>>,
        key: ChunkKey,
        address: [u8; 32],
        expected: [u8; 32],
    ) -> StreamResult {
        let name = to_hex(&address);
        if let Some(c) = &cache {
            let (c2, n2) = (c.clone(), name.clone());
            let cached = tokio::task::spawn_blocking(move || c2.get(BlobKind::Section, &n2))
                .await
                .ok()
                .flatten();
            if let Some(bytes) = cached {
                if sha256(&bytes) != expected {
                    // Another version: computed here, or replaced by the
                    // hub after a dispute. The stored one wins.
                    tracing::debug!(name, "discarding a cached section of another version");
                    c.remove(BlobKind::Section, &name);
                } else {
                    match decode_arrived(key, bytes).await {
                        Ok(a) => return StreamResult::Bytes(a, Origin::Cache),
                        Err(e) => {
                            tracing::warn!(name, "dropping a damaged cached section: {e}");
                            c.remove(BlobKind::Section, &name);
                        }
                    }
                }
            }
        }
        match client.get_section(&address, &expected).await {
            Ok(Some(bytes)) if sha256(&bytes) != expected => StreamResult::Failed(format!(
                "section {name} from the edge does not hash to {}",
                to_hex(&expected)
            )),
            Ok(Some(bytes)) => match decode_arrived(key, bytes).await {
                Ok(a) => {
                    if let Some(c) = cache {
                        let (n2, b2) = (name.clone(), a.bytes.clone());
                        let _ = tokio::task::spawn_blocking(move || {
                            if let Err(e) = c.put(BlobKind::Section, &n2, &b2) {
                                tracing::warn!(name = n2, "could not cache the section: {e}");
                            }
                        })
                        .await;
                    }
                    StreamResult::Bytes(a, Origin::Edge)
                }
                Err(e) => {
                    StreamResult::Failed(format!("section {name} from the edge is invalid: {e}"))
                }
            },
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
                    occupancy: Ratio::new(1.0),
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
        let keys: Vec<CellKey> = (0..n).map(|i| key(12, i, 0, 0)).collect();
        c.set_cells(sel(&keys));
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
    fn ancestors_of_selected_cells_are_kept() {
        let mut c = CellCache::new();
        let parent = key(3, 1, 1, 1);
        let child = key(4, 2, 2, 2);
        let other = key(3, 5, 5, 5);
        c.set_cells(sel(&[parent, other]));
        c.complete(parent, ready(), 0.0);
        c.complete(other, ready(), 0.0);
        // The parent is refined: its child is selected instead.
        c.set_cells(sel(&[child]));
        let gone = c.evict(1000.0);
        assert!(!gone.contains(&parent), "the parent was evicted");
        assert!(gone.contains(&other));
        assert!(c.entry(&parent).is_some());
    }

    #[test]
    fn invalid_cells_are_gone_and_not_asked_again() {
        let mut c = CellCache::new();
        let (a, b) = (key(1, 0, 0, 0), key(1, 1, 0, 0));
        c.set_cells(sel(&[a, b]));
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
    fn a_section_with_a_wrong_cell_edge_is_refused() {
        use gx_core::matter::{encode, Compression};
        let k = key(1, 1, 0, 0);
        let g = k.geometry(Meters::new(8.0));
        let wrong = Meters::new(g.edge.value() * (1.0 + f64::EPSILON));
        let section = Section::empty(k, g.origin, wrong).unwrap();
        let bytes = encode(&section, Compression::None);
        match decode_cell(&k, &[&bytes], Meters::new(8.0)) {
            FetchOutcome::Invalid(reason) => {
                assert!(reason.contains("geometry"), "{reason}")
            }
            other => panic!("drawn: {other:?}"),
        }
    }

    #[test]
    fn eviction_by_age_and_cap() {
        let mut c = CellCache::new();
        let a = key(1, 0, 0, 0);
        c.set_cells(sel(&[a]));
        c.take_requests(0.0);
        c.complete(a, ready(), 0.0);
        c.set_cells(Vec::new());
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
        c.set_cells(Vec::new());
        assert_eq!(c.take_requests(0.0), vec![k]);
        c.complete(k, ready(), 0.1);
        c.set_cells(Vec::new());
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
        c.set_cells(sel(&kids));
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
        c.set_cells(sel(&[parent]));
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
        c.set_cells(sel(&[k, key(2, 1, 1, 3), key(2, 0, 0, 3)]));
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

    /// A solid cell `key` of a 1 m root at resolution 4 with `occupancy`
    /// per sample `(x, y, z)`, vacuum where it is 0.
    fn solid_cell(key: CellKey, occupancy: impl Fn(u32, u32, u32) -> f64) -> ReadyCell {
        use gx_core::matter::{Sample, Samples};
        use gx_core::units::{Attenuation, Density, Kelvin, Ratio};
        let g = key.geometry(Meters::new(1.0));
        let samples = Samples::from_fn(4, |x, y, z| {
            let o = occupancy(x, y, z);
            if o <= 0.0 {
                Sample::VACUUM
            } else {
                Sample {
                    density: Density::new(1000.0 * o),
                    state: State::Solid,
                    temperature: Kelvin::new(300.0),
                    albedo: [Ratio::new(0.5); 3],
                    roughness: Ratio::new(0.5),
                    attenuation: Attenuation::new(0.0),
                    radiating_temperature: Kelvin::new(300.0),
                    occupancy: Ratio::new(o),
                }
            }
        });
        ReadyCell::new(Section::new(key, g.origin, g.edge, 4, samples).unwrap())
    }

    #[test]
    fn refinable_and_asking_ahead_follow_occupancy() {
        let k = key(0, 0, 0, 0);
        // Full everywhere: no surface, nothing below is refinable.
        let full = solid_cell(k, |_, _, _| 1.0);
        assert!(!full.refinable);
        assert_eq!(full.ahead, AheadBits::none());

        // A boundary at x = 2.5 samples: matter below, vacuum above.
        let half = solid_cell(k, |x, _, _| match x {
            0 | 1 => 1.0,
            2 => 0.5,
            _ => 0.0,
        });
        assert!(half.refinable);
        let child = |x, y, z| CellKey::new(5, 1, x, y, z).unwrap();
        // The children on either side of x = 0.5 of the cell hold the
        // crossing within one sample; the predictions are by region, the
        // same for every y and z.
        for (y, z) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
            assert!(half.ahead.predicts(&k, &child(0, y, z)));
            assert!(half.ahead.predicts(&k, &child(1, y, z)));
        }
        // Three depths down: the descendants in the first sample are more
        // than one sample from the crossing.
        let deep = |x| CellKey::new(5, 3, x, 0, 0).unwrap();
        assert!(!half.ahead.predicts(&k, &deep(0)));
        assert!(half.ahead.predicts(&k, &deep(4)));
        assert!(!half.ahead.predicts(&k, &key(4, 0, 0, 0)));

        // Full but for a partly filled sample on the +x face: the boundary
        // lies in that sample's sub-cube, maybe past its center, so the
        // cell and the descendants at that face count as crossing, and
        // those away from it do not.
        let face = solid_cell(k, |x, y, z| if (x, y, z) == (3, 1, 1) { 0.7 } else { 1.0 });
        assert!(face.refinable);
        assert!(face.ahead.predicts(&k, &deep(7)));
        assert!(!face.ahead.predicts(&k, &deep(0)));
        // The same partly filled sample inside the cell holds no crossing.
        let inside = solid_cell(k, |x, y, z| if (x, y, z) == (2, 1, 1) { 0.7 } else { 1.0 });
        assert!(!inside.refinable);
    }

    #[test]
    fn around_parents_are_the_parents_of_the_cells_around() {
        for k in [
            key(3, 0, 0, 0),
            key(3, 5, 2, 7),
            key(3, 4, 3, 3),
            key(1, 1, 0, 1),
        ] {
            let mut brute: Vec<CellKey> = AROUND
                .iter()
                .filter_map(|&o| offset_key(&k, o)?.parent())
                .filter(|p| Some(*p) != k.parent())
                .collect();
            brute.sort();
            brute.dedup();
            let mut fast: Vec<CellKey> = around_parents(&k).into_iter().flatten().collect();
            fast.sort();
            assert_eq!(fast, brute, "{k}");
        }
        assert_eq!(around_parents(&key(0, 0, 0, 0)), [None; 7]);
    }

    /// Every split cell has the 26 cells around it split above too, or
    /// outside the tree: no two leaves that touch differ by more than one
    /// depth.
    fn assert_balanced(tree: &CellTree) {
        for n in &tree.internal {
            for o in AROUND {
                let Some(p) = offset_key(n, o).and_then(|a| a.parent()) else {
                    continue;
                };
                assert!(
                    tree.internal.contains(&p)
                        || !tree.contains(&p) && {
                            // Not in the tree at all: no ancestor is a leaf.
                            let mut up = p.parent();
                            let mut leaf = false;
                            while let Some(a) = up {
                                leaf |= tree.leaves.contains(&a);
                                up = a.parent();
                            }
                            !leaf
                        },
                    "{n} is split next to the leaf holding {p}"
                );
            }
        }
    }

    #[test]
    fn balance_splits_only_what_the_rule_needs() {
        // Two chains of splits toward two points, each to depth 7: as
        // unbalanced as a tree can be.
        let mut leaves: CellSet = CellSet::default();
        let mut cells = vec![key(0, 0, 0, 0)];
        for target in [[5u32, 100, 77], [126, 3, 64]] {
            let mut n = key(0, 0, 0, 0);
            for d in 1..=7u8 {
                let shift = 7 - d;
                for c in n.children().unwrap() {
                    cells.push(c);
                }
                n = key(
                    d,
                    target[0] >> shift,
                    target[1] >> shift,
                    target[2] >> shift,
                );
            }
        }
        let tree0 = CellTree::from_leaves(cells.iter().copied());
        for k in &cells {
            if !tree0.internal.contains(k) {
                leaves.insert(*k);
            }
        }
        let mut tree = CellTree::from_leaves(leaves.iter().copied());
        let before = tree.leaves.len();
        balance_target(&mut tree);
        assert_balanced(&tree);
        assert!(tree.leaves.len() > before);
        // Balancing again changes nothing.
        let again = tree.clone();
        balance_target(&mut tree);
        assert_eq!(tree.leaves, again.leaves);
        // Every leaf of the balanced tree lies in a leaf of the original or
        // is one of its leaves: balancing only splits.
        for l in &tree.leaves {
            let mut up = Some(*l);
            let mut found = false;
            while let Some(a) = up {
                found |= leaves.contains(&a);
                up = a.parent();
            }
            assert!(found, "{l}");
        }
    }

    #[test]
    fn an_ancestor_stand_in_hides_the_cells_below_it() {
        let mut c = CellCache::new();
        let parent = key(1, 0, 0, 0);
        let mut kids = parent.children().unwrap();
        kids.sort();
        // One child is split further: its children are selected too.
        let mut grand = kids[0].children().unwrap();
        grand.sort();
        let lone = grand[0];
        let mut selected: Vec<CellKey> = kids[1..].to_vec();
        selected.extend(grand);
        selected.sort();
        c.set_cells(sel(&selected));
        let d = |c: &CellCache, s: &BTreeSet<CellKey>| c.draw_set(|k| s.contains(k));
        // Everything but one grandchild drawable, and the parent: the
        // grandchild's nearest drawable ancestor is the parent, which
        // hides every selected cell.
        let mut drawable: BTreeSet<CellKey> =
            selected.iter().copied().filter(|k| *k != lone).collect();
        drawable.insert(parent);
        assert_eq!(d(&c, &drawable), vec![parent]);
        // With the split child drawable too, it stands in for its own
        // children alone.
        drawable.insert(kids[0]);
        assert_eq!(d(&c, &drawable), kids.to_vec());
        // A grandchild that is not drawable but whose children all are is
        // covered by them.
        drawable.remove(&kids[0]);
        drawable.extend(lone.children().unwrap());
        let drawn = d(&c, &drawable);
        assert!(!drawn.contains(&parent));
        assert!(!drawn.contains(&lone));
        assert_eq!(drawn.len(), selected.len() - 1 + 8);
    }
}

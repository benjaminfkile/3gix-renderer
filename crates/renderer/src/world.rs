//! The matter pipeline from selection to drawable meshes and lights: the
//! frame thread's matter work (`rendering.md` section 4).
//!
//! Ties together the steps of `space-model.md` section 2 that turn chunks
//! into something to draw: cell selection and the cell cache
//! ([`crate::stream`]), surface extraction on a worker pool
//! ([`crate::extract`]), volumes of gas and plasma ([`crate::volume`]),
//! lights from hot matter ([`crate::light`]), and point sprites for far
//! frames ([`crate::farfield`]). It does no I/O itself: the caller hands it
//! fetch results (from the hub through [`crate::stream::Fetcher`] on the
//! desktop, or straight from bytes in tests) and gets back what to request
//! next. One [`World`] serves the window loop, the headless run, the budget
//! test, and the browser the same way, through [`World::step`].
//!
//! # One step
//!
//! [`World::step`] is everything the frame thread does for matter in one
//! frame: it applies the arrived results (already decoded off the frame
//! thread), selects when something changed ([`World::select`]), picks the
//! next requests, gathers extraction inputs for the cells whose inputs
//! changed and hands them to the extraction pool, evicts now and then, and
//! builds the draw list.
//!
//! Extraction inputs are gathered incrementally. An arrival marks the cells
//! whose inputs can read the arrived cell (itself, the cells around it at
//! its depth, the children of those, and its parent and the cells around
//! the parent); a change of the selection marks the cells near each cell
//! whose selected status changed. Only marked cells are looked
//! at, and only for what is left of the step's time slice once the draw
//! list is built ([`STEP_SLICE_SECONDS`]): what does not fit waits for the
//! next frame. A
//! few cells per step are also checked in turn ([`AUDIT_PER_STEP`]), so a
//! change no mark reached is still picked up.

use crate::camera::Camera;
use crate::clock::{now_seconds, StepTimes};
use crate::extract::{CellInputs, ExtractPool, Extraction, SurfaceMesh};
use crate::farfield::{depth_zero_key, far_frames_with_surfaces, frame_sprite, Sprite};
use crate::light::{lights_for, merge_frame_emitters, strongest_lights, PointLight};
use crate::seen::Seen;
use crate::selector::{SelectJob, Selected, SelectionLayout, Selector};
use crate::stream::{
    offset_key, CacheCounts, CellCache, CellMap, CellSet, CellState, FetchOutcome, AROUND,
    SELECT_INTERVAL_SECONDS,
};
use crate::volume::VolumeGrid;
use gx_core::frames::FrameSystem;
use gx_core::key::CellKey;
use gx_core::units::Vec3;
use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;

/// Seconds after the start of a step past which the step does no more
/// optional work: gathering extraction inputs natively, and compositing
/// arrived cells as well in the browser. The step's required work
/// (results, applying a finished selection, requests, the draw list) comes
/// first, and gathering takes what is left of this slice. Half of the 4 ms
/// of `rendering.md` section 4: measured with
/// `cargo test --release -p renderer --test budget`, the required work at
/// 4096 cells takes about 0.5 ms at the median and up to about 2 ms when
/// the worker threads load every core, and the rest of the 4 ms leaves room
/// for that and for one cell's gathering past the deadline.
pub const STEP_SLICE_SECONDS: f64 = 0.002;

/// Seconds after the start of a step past which the browser composites no
/// more arrived cells in that step (`crate::web`): half of
/// [`STEP_SLICE_SECONDS`], leaving the rest to the required work and to
/// gathering. At least one cell is composited per step.
pub const COMPOSITE_SLICE_SECONDS: f64 = 0.001;

/// Cells checked in turn per step whether or not anything marked them.
pub const AUDIT_PER_STEP: usize = 4;

/// Seconds between two evictions.
pub const EVICT_INTERVAL_SECONDS: f64 = 1.0;

/// What to draw this frame.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DrawList {
    /// Non-empty meshes of the draw set, in key order.
    pub meshes: Vec<Arc<SurfaceMesh>>,
    /// Surface tiles to draw (`crate::tiles`).
    pub tiles: Vec<crate::tiles::TileDrawInfo>,
    /// Volumes of the draw set, in key order.
    pub volumes: Vec<Arc<VolumeGrid>>,
    /// The strongest lights, strongest first, where the observer sees
    /// their emitters.
    pub lights: Vec<PointLight>,
    /// The same lights as the matter of each frame with meshes or a sprite
    /// receives them ([`lights_for`]), by frame id.
    pub frame_lights: BTreeMap<u64, Vec<PointLight>>,
    /// Point sprites of far frames, in ascending frame id order.
    pub sprites: Vec<Sprite>,
    /// Weight of the cells of frames in the far field transition, by
    /// frame id; frames not listed draw their cells at weight 1.
    pub cell_weights: BTreeMap<u64, f32>,
    /// Weight of the surface tiles of frames in the far field transition,
    /// by frame id ([`crate::farfield::FarFrame::tile_weight`]); frames not
    /// listed draw their tiles at weight 1.
    pub tile_weights: BTreeMap<u64, f32>,
}

impl DrawList {
    /// The weight the surface tiles of `frame_id` are drawn with.
    pub fn tile_weight(&self, frame_id: u64) -> f32 {
        self.tile_weights.get(&frame_id).copied().unwrap_or(1.0)
    }

    /// The weight the cells of `frame_id` are drawn with.
    pub fn cell_weight(&self, frame_id: u64) -> f32 {
        self.cell_weights.get(&frame_id).copied().unwrap_or(1.0)
    }

    /// The lights as the matter of `frame_id` receives them: its entry in
    /// [`DrawList::frame_lights`], or [`DrawList::lights`] for a frame
    /// without one.
    pub fn lights_of(&self, frame_id: u64) -> &[PointLight] {
        self.frame_lights
            .get(&frame_id)
            .map_or(&self.lights, Vec::as_slice)
    }
}

/// Counts for the overlay.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct WorldCounts {
    /// Cache counts.
    pub cache: CacheCounts,
    /// Meshes extracted and held.
    pub meshes: usize,
    /// Extractions queued or running.
    pub extracting: usize,
}

/// One consistent set of drawn cells and their meshes (see [`World`]).
struct Shown {
    /// The drawn selection.
    keys: Vec<CellKey>,
    /// Each selected cell's mesh, extracted for exactly this selection.
    meshes: Vec<Arc<SurfaceMesh>>,
}

/// A held mesh and the inputs it was extracted with.
struct Extracted {
    mesh: Arc<SurfaceMesh>,
    inputs: CellInputs,
}

/// Where the inputs of extraction may have changed, to be turned into
/// marked cells within the time slice.
#[derive(Copy, Clone, Debug, PartialEq)]
enum Mark {
    /// A cell arrived or left the cache.
    Arrival(CellKey),
    /// A cell's selected or above-selected status changed.
    Layout(CellKey),
}

/// The camera and view a selection was made for.
#[derive(Copy, Clone, Debug, PartialEq)]
struct Pose {
    frame_id: u64,
    values: [f64; 7],
    view: (u32, u32),
}

impl Pose {
    fn of(camera: &Camera, view: (u32, u32)) -> Pose {
        let (p, q) = (camera.position, camera.orientation);
        Pose {
            frame_id: camera.frame_id,
            values: [p.x, p.y, p.z, q.x, q.y, q.z, q.w],
            view,
        }
    }
}

/// One item eviction over the memory budget may drop, in the order of
/// [`crate::budget::sort_candidates`] on a full tie.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Held {
    Tile(gx_core::key::SurfaceKey),
    Cell(CellKey),
    Mesh(CellKey),
}

/// Longest a selection waits for the one before it to be shown
/// ([`World::select`]), seconds.
pub const PACE_LIMIT_SECONDS: f64 = 1.0;

/// The cell cache, the extraction pool, and the extracted meshes.
///
/// A drawn cell's mesh depends on the sections and the drawn selection in
/// its region ([`CellInputs`]). Whenever those change the cell is
/// extracted again.
///
/// What is drawn is always one consistent set ([`Shown`]): the drawn
/// selection at a moment when every selected cell's mesh was extracted for
/// exactly that selection. Transvoxel joins depths only between meshes
/// made for the same selection (`rendering.md` 2.3 and 3.3), so a mesh
/// made for an older selection next to one made for a newer one leaves a
/// crack. While the selection changes faster than its cells are
/// extracted, the last consistent set keeps drawing.
pub struct World {
    /// Surface tiles of frames that have them (`crate::tiles`).
    surfaces: crate::tiles::Surfaces,
    /// The set drawn now ([`Shown`]); `None` until the first selection
    /// whose cells are all extracted.
    shown: Option<Shown>,
    /// Times the shown set was replaced by a different selection.
    shown_switches: u64,
    /// The build's layers and the hidden ones (`rendering.md` section 7
    /// item 7).
    layer_view: crate::layers::LayerView,
    /// When the current selection was applied, while its cells are not
    /// all extracted for it yet; `None` once it is shown.
    building_since: Option<f64>,
    cache: CellCache,
    pool: ExtractPool,
    meshes: CellMap<Extracted>,
    extracting: CellMap<CellInputs>,
    /// Inputs whose extraction read an input neither present nor known
    /// empty: not extracted again until they change.
    waiting: CellMap<CellInputs>,
    layout: SelectionLayout,
    marks: VecDeque<Mark>,
    dirty: CellSet,
    dirty_queue: VecDeque<CellKey>,
    /// Marked while being extracted: looked at again once that is done.
    deferred: CellSet,
    audit: Vec<CellKey>,
    selector: Selector,
    /// Cell states changed since the latest selection job.
    changes: Vec<(CellKey, Option<CellState>)>,
    last_selection: Option<f64>,
    last_pose: Option<Pose>,
    arrived_since_selection: bool,
    rounds: u64,
    steps: StepTimes,
    extraction_times: StepTimes,
    last_evict: f64,
    /// The memory budget, bytes (`crate::budget`).
    budget: u64,
    /// The bytes held after the latest eviction pass.
    held: u64,
    /// The cells' and meshes' bytes as the latest count found them.
    cell_bytes: u64,
    /// Asking ahead is held: the budget was exceeded and held bytes have
    /// not yet fallen below [`crate::budget::RESUME_FRACTION`] of it.
    holding_ahead: bool,
    /// What the latest eviction pass over the budget did.
    last_evicted: crate::budget::Evicted,
    /// What every eviction pass over the budget did, summed.
    total_evicted: crate::budget::Evicted,
    /// The detail factor the surface selection scales split distances by
    /// over the budget ([`crate::budget::Detail`]).
    detail: crate::budget::Detail,
    /// The camera and view of the latest surface selection.
    surface_pose: Option<Pose>,
}

impl World {
    /// A world extracting on `threads` worker threads, and selecting on
    /// its own thread ([`crate::selector`]); with zero threads, both
    /// inline.
    pub fn new(threads: usize) -> World {
        World {
            surfaces: crate::tiles::Surfaces::new(),
            shown: None,
            shown_switches: 0,
            layer_view: crate::layers::LayerView::default(),
            building_since: None,
            cache: CellCache::new(),
            pool: ExtractPool::new(threads),
            meshes: CellMap::default(),
            extracting: CellMap::default(),
            waiting: CellMap::default(),
            layout: SelectionLayout::default(),
            marks: VecDeque::new(),
            dirty: CellSet::default(),
            dirty_queue: VecDeque::new(),
            deferred: CellSet::default(),
            audit: Vec::new(),
            selector: Selector::new(threads > 0),
            changes: Vec::new(),
            last_selection: None,
            last_pose: None,
            arrived_since_selection: false,
            rounds: 0,
            steps: StepTimes::new(),
            extraction_times: StepTimes::new(),
            last_evict: f64::NEG_INFINITY,
            budget: crate::budget::platform_budget(),
            held: 0,
            cell_bytes: 0,
            holding_ahead: false,
            last_evicted: crate::budget::Evicted::default(),
            total_evicted: crate::budget::Evicted::default(),
            detail: crate::budget::Detail::default(),
            surface_pose: None,
        }
    }

    /// The cell cache.
    pub fn cache(&self) -> &CellCache {
        &self.cache
    }

    /// The mesh extracted for a cell, if any.
    pub fn mesh(&self, key: &CellKey) -> Option<&Arc<SurfaceMesh>> {
        self.meshes.get(key).map(|e| &e.mesh)
    }

    /// The inputs the mesh of a cell was extracted with, if it has one.
    pub fn mesh_inputs(&self, key: &CellKey) -> Option<&CellInputs> {
        self.meshes.get(key).map(|e| &e.inputs)
    }

    /// Extraction time per cell so far, milliseconds, one entry per
    /// finished extraction.
    pub fn extraction_times(&self) -> &StepTimes {
        &self.extraction_times
    }

    /// Selections made this session ([`World::select`]).
    pub fn selection_rounds(&self) -> u64 {
        self.rounds
    }

    /// Frame-thread matter work per step so far ([`World::record_step`]).
    pub fn step_times(&self) -> &StepTimes {
        &self.steps
    }

    /// Records one step's frame-thread matter work, milliseconds: the
    /// caller measures [`World::step`] together with its own fetcher calls.
    pub fn record_step(&mut self, ms: f64) {
        self.steps.record(ms);
    }

    /// The deepest cell of the drawn selection, or `None` without one.
    pub fn deepest_selected(&self) -> Option<u8> {
        self.cache.selection().iter().map(|c| c.key.depth).max()
    }

    /// The extraction inputs of a drawn `Ready` cell now
    /// ([`CellInputs`]), or `None` for a cell that is not drawn or not
    /// `Ready`.
    fn inputs_of(&self, key: &CellKey) -> Option<CellInputs> {
        if !self.layout.selected.contains(key) {
            return None;
        }
        let Some(CellState::Ready(cell)) = self.cache.state(key) else {
            return None;
        };
        let mut drawn = std::collections::BTreeSet::new();
        let mut sections = BTreeMap::new();
        for k in CellInputs::region(key) {
            if self.layout.selected.contains(&k) {
                drawn.insert(k);
            }
            if k == *key {
                continue;
            }
            match self.cache.state(&k) {
                Some(CellState::Ready(c)) => {
                    sections.insert(k, Some(c.section.clone()));
                }
                _ if self.cache.known_empty(&k) => {
                    sections.insert(k, None);
                }
                _ => {}
            }
        }
        Some(CellInputs {
            key: *key,
            section: cell.section.clone(),
            drawn,
            sections,
        })
    }

    /// Selects cells for the camera ([`crate::stream::select_all`],
    /// through [`crate::selector`]) when something changed since the last
    /// selection (results arrived, the camera moved or turned, the view
    /// changed size), when [`SELECT_INTERVAL_SECONDS`] have passed, or when
    /// `force` is set; and pins the depth-0 cell of every frame with mass:
    /// the far field draws from it ([`crate::farfield`]) and every light
    /// comes from it ([`crate::light`]).
    ///
    /// A selection finished on the selection thread is applied here first.
    /// A new one starts only when none is in flight; on the thread it is
    /// applied by a later call, inline at once. With `force` the call waits
    /// for the selection it starts. Returns `true` if a selection was
    /// applied.
    pub fn select(
        &mut self,
        system: &FrameSystem,
        camera: &Camera,
        view_px: (u32, u32),
        now: f64,
        force: bool,
    ) -> bool {
        let mut applied = false;
        if force {
            if let Some(done) = self.selector.wait() {
                self.apply_selection(done);
                applied = true;
            }
        } else if let Some(done) = self.selector.try_finish() {
            self.apply_selection(done);
            applied = true;
        }
        let pose = Pose::of(camera, view_px);
        let due = force
            || self.arrived_since_selection
            || self.last_pose != Some(pose)
            || self
                .last_selection
                .is_none_or(|t| now - t >= SELECT_INTERVAL_SECONDS);
        if applied {
            self.building_since = Some(now);
        }
        // Paced by extraction: a new selection starts only once the current
        // one is shown (every cell extracted for it), or after
        // PACE_LIMIT_SECONDS, so a camera that keeps moving still sees each
        // finished selection drawn instead of none.
        let building = !force
            && self
                .building_since
                .is_some_and(|t| now - t < PACE_LIMIT_SECONDS && !self.consistent());
        if !due || self.selector.busy() || building {
            return applied;
        }
        self.last_selection = Some(now);
        self.last_pose = Some(pose);
        self.arrived_since_selection = false;
        for f in system
            .tree()
            .frames()
            .iter()
            .filter(|f| f.mass.value() > 0.0)
        {
            self.cache.pin(depth_zero_key(f.frame_id), now);
        }
        let job = SelectJob {
            system: system.clone(),
            camera: *camera,
            view_px: (f64::from(view_px.0.max(1)), f64::from(view_px.1.max(1))),
            changes: std::mem::take(&mut self.changes),
            skip_frames: self.surfaces.frames_with_surface().collect(),
        };
        let done = match self.selector.start(job) {
            Some(done) => Some(done),
            None if force => self.selector.wait(),
            None => None,
        };
        if let Some(done) = done {
            self.apply_selection(done);
            self.building_since = Some(now);
            applied = true;
        }
        applied
    }

    /// Waits for the selection in flight, if any, and applies it. Returns
    /// `true` if one was applied.
    pub fn wait_selection(&mut self) -> bool {
        match self.selector.wait() {
            Some(done) => {
                self.apply_selection(done);
                true
            }
            None => false,
        }
    }

    /// Makes a finished selection the current one, and marks the cells
    /// around those whose place in it changed.
    fn apply_selection(&mut self, done: Selected) {
        self.rounds += 1;
        self.cache.set_selection(done.selection);
        if let Some((layout, changes)) = done.layout {
            for k in changes {
                self.marks.push_back(Mark::Layout(k));
            }
            self.layout = layout;
        }
    }

    /// The cells wanted now, once per new selection (see
    /// [`CellCache::take_wanted`]).
    pub fn take_wanted(&mut self, now: f64) -> Option<Vec<(CellKey, f64)>> {
        self.cache.take_wanted(now)
    }

    /// The surface tiles wanted now, once per selection (see
    /// [`crate::tiles::Surfaces::take_wanted`]).
    pub fn take_wanted_surfaces(&mut self) -> Option<Vec<(gx_core::key::SurfaceKey, f64)>> {
        self.surfaces.take_wanted()
    }

    /// The cells to request now (see [`CellCache::take_requests`]).
    pub fn take_requests(&mut self, now: f64) -> Vec<CellKey> {
        self.cache.take_requests(now)
    }

    /// Selects surface tiles for every frame with mass in range
    /// ([`crate::tiles::Surfaces::select`]).
    /// `now`, seconds, is the clock overlay waits are measured on.
    pub fn select_surfaces(
        &mut self,
        system: &FrameSystem,
        camera: &Camera,
        view_px: (u32, u32),
        now: f64,
    ) {
        let aspect = f64::from(view_px.0.max(1)) / f64::from(view_px.1.max(1));
        let frames: Vec<u64> = system
            .tree()
            .frames()
            .iter()
            .filter(|f| {
                f.mass.value() > 0.0
                    && crate::stream::frame_in_range(system, camera, f.frame_id, aspect)
            })
            .map(|f| f.frame_id)
            .collect();
        let pose = Pose::of(camera, view_px);
        let moved = self.surface_pose != Some(pose);
        self.surface_pose = Some(pose);
        self.detail.observe(moved, now);
        self.surfaces.set_keep_built(self.detail.moving());
        self.surfaces.select(system, camera, view_px, &frames, now);
        self.enforce_budget(false);
        let settled = self.surfaces.settled();
        if self.detail.after_pass(self.held, self.budget, settled, now) {
            self.surfaces.set_detail_step(self.detail.step());
        }
    }

    /// The detail factor the surface selection scales split distances by
    /// ([`crate::budget::Detail`]): 1 unless the view's full detail did
    /// not fit the memory budget.
    pub fn detail_factor(&self) -> f64 {
        self.detail.factor()
    }

    /// The memory budget, bytes (`crate::budget`): the platform's default
    /// from [`crate::budget::platform_budget`] unless set.
    pub fn memory_budget(&self) -> u64 {
        self.budget
    }

    /// Sets the memory budget, bytes: for tests and tools that force a
    /// small one. No API option reaches it (`crate::budget`).
    pub fn set_memory_budget(&mut self, bytes: u64) {
        if bytes != self.budget {
            self.detail.forget();
        }
        self.budget = bytes;
    }

    /// The bytes held after the latest eviction pass: surface tiles,
    /// cells, and meshes, as the budget counts them.
    pub fn held_bytes(&self) -> u64 {
        self.held
    }

    /// Whether the camera counts as moving for the detail factor
    /// ([`crate::budget::Detail::moving`]).
    pub fn memory_moving(&self) -> bool {
        self.detail.moving()
    }

    /// Whether asking ahead is held by the budget (`crate::budget`).
    pub fn holding_ahead(&self) -> bool {
        self.holding_ahead
    }

    /// What the latest eviction pass over the budget did.
    pub fn last_evicted(&self) -> crate::budget::Evicted {
        self.last_evicted
    }

    /// What every eviction pass over the budget did, summed.
    pub fn total_evicted(&self) -> crate::budget::Evicted {
        self.total_evicted
    }

    fn mesh_bytes(&self) -> u64 {
        self.meshes.values().map(|e| e.mesh.heap_bytes()).sum()
    }

    /// Counts the held bytes. Cells and meshes are counted again only
    /// with `cells` (the cell eviction pass, every
    /// [`EVICT_INTERVAL_SECONDS`]); the surface pass after each selection
    /// takes their count from the latest cell pass, so it costs no walk
    /// over every cell.
    fn count_held(&mut self, cells: bool) -> u64 {
        if cells {
            self.cell_bytes = self.cache.held_bytes() + self.mesh_bytes();
        }
        self.surfaces.held_bytes() + self.cell_bytes
    }

    /// Evicts against the memory budget (`crate::budget`, `docs/architecture.md`,
    /// "Memory budget"). The limit is the budget while the camera moves
    /// and [`crate::budget::SETTLE_FRACTION`] of it while it is still
    /// ([`crate::budget::Detail::limit`]). While over it: built tiles not drawn and not
    /// wanted built go compact; then compact tiles, cells, and meshes not
    /// drawn and not in the current target go, least recently used first,
    /// the deepest first on a tie. The drawn tiles, the ancestors the
    /// selection walks to reach them, level-0 tiles, the tiles wanted
    /// built or asked for by the latest selection, the cell target, and
    /// pinned cells stay, even over the limit: after a surface pass that
    /// leaves the bytes over it, the detail factor goes down instead
    /// ([`crate::budget::Detail`]). Then holds asking ahead
    /// while held bytes exceed the limit, until they fall below
    /// [`crate::budget::RESUME_FRACTION`] of it.
    fn enforce_budget(&mut self, cells: bool) {
        let mut held = self.count_held(cells);
        let mut ev = crate::budget::Evicted::default();
        let limit = self.detail.limit(self.budget);
        if held > limit {
            let mut over = held - limit;
            ev.compacted = self.surfaces.compact_over(&mut over);
            if over > 0 {
                let mut items: Vec<crate::budget::Candidate<Held>> = Vec::new();
                for c in self.surfaces.eviction_candidates() {
                    items.push(crate::budget::Candidate {
                        last_used: c.last_used,
                        level: c.level,
                        key: Held::Tile(c.key),
                        bytes: c.bytes,
                    });
                }
                for c in self.cache.eviction_candidates() {
                    items.push(crate::budget::Candidate {
                        last_used: c.last_used,
                        level: c.level,
                        key: Held::Cell(c.key),
                        bytes: c.bytes,
                    });
                }
                let shown: CellSet = self.shown_keys().iter().copied().collect();
                for (k, e) in &self.meshes {
                    if shown.contains(k) || self.cache.in_target(k) {
                        continue;
                    }
                    items.push(crate::budget::Candidate {
                        last_used: self
                            .cache
                            .entry(k)
                            .map_or(f64::NEG_INFINITY, |e| e.last_used),
                        level: k.depth,
                        key: Held::Mesh(*k),
                        bytes: e.mesh.heap_bytes(),
                    });
                }
                crate::budget::sort_candidates(&mut items);
                let mut tiles = Vec::new();
                for c in items {
                    if over == 0 {
                        break;
                    }
                    let freed = match c.key {
                        Held::Tile(k) => {
                            tiles.push(k);
                            ev.tiles += 1;
                            c.bytes
                        }
                        Held::Cell(k) => {
                            self.cache.drop_cells(&[k]);
                            ev.cells += 1;
                            c.bytes + self.drop_mesh(k, false)
                        }
                        Held::Mesh(k) => {
                            let b = self.drop_mesh(k, true);
                            if b > 0 {
                                ev.meshes += 1;
                            }
                            b
                        }
                    };
                    over = over.saturating_sub(freed);
                }
                self.surfaces.drop_tiles(&tiles);
            }
            held = self.count_held(true);
        }
        self.holding_ahead = crate::budget::hold_ahead(held, limit, self.holding_ahead);
        self.surfaces.set_hold_ahead(self.holding_ahead);
        self.cache.set_hold_ahead(self.holding_ahead);
        self.held = held;
        self.last_evicted = ev;
        self.total_evicted.compacted += ev.compacted;
        self.total_evicted.tiles += ev.tiles;
        self.total_evicted.cells += ev.cells;
        self.total_evicted.meshes += ev.meshes;
    }

    /// Drops `k`'s mesh, if held, and returns its bytes. With `kept` the
    /// cell stays cached: its mesh is extracted again when it is next in
    /// the target. Without it the cell left the cache.
    fn drop_mesh(&mut self, k: CellKey, kept: bool) -> u64 {
        let bytes = self.meshes.get(&k).map_or(0, |e| e.mesh.heap_bytes());
        if bytes > 0 || !kept {
            self.changes.push((k, None));
            self.meshes.remove(&k);
            self.waiting.remove(&k);
            self.marks.push_back(Mark::Arrival(k));
        }
        bytes
    }

    /// The compact surface tiles wanted built since the last call
    /// ([`crate::tiles::Surfaces::take_builds`]), for the fetcher to build
    /// off the frame thread.
    pub fn take_surface_builds(&mut self) -> Vec<crate::tiles::BuildRequest> {
        self.surfaces.take_builds()
    }

    /// Applies a finished surface tile build
    /// ([`crate::tiles::Surfaces::complete_build`]).
    pub fn complete_surface_build(
        &mut self,
        key: gx_core::key::SurfaceKey,
        outcome: crate::tiles::BuildOutcome,
    ) {
        self.surfaces.complete_build(key, outcome);
    }

    /// Builds every surface tile wanted built on this thread
    /// ([`crate::tiles::Surfaces::build_now`]): for tests and tools that
    /// drive a world without a fetcher.
    pub fn build_surfaces_now(&mut self) {
        self.surfaces.build_now();
    }

    /// Applies a finished surface tile request.
    pub fn complete_surface(
        &mut self,
        key: gx_core::key::SurfaceKey,
        outcome: Result<crate::tiles::DecodedTile, String>,
        partial: bool,
    ) {
        self.surfaces.complete_layers(key, outcome, partial);
    }

    /// Hides the layers in `hidden` (indices in the build) and shows every
    /// other one (`rendering.md` section 7 item 7): a hidden overlay is not
    /// composited. A client starts with the build's
    /// [`crate::protocol::BuildDescription::initially_hidden_layers`].
    ///
    /// Every cached cell whose shown layers change is composited again
    /// from its kept layer sections, with nothing fetched again, and
    /// marked as arrived, so its mesh is extracted again off the frame
    /// thread and its far-field values follow it.
    pub fn set_hidden_layers(&mut self, hidden: std::collections::BTreeSet<usize>) {
        let old = self.surfaces.hidden_layers().clone();
        self.layer_view.hidden = hidden.clone();
        self.surfaces.set_hidden_layers(hidden.clone());
        if old == hidden {
            return;
        }
        let changed: Vec<(CellKey, Arc<crate::stream::ReadyCell>)> = (self.cache.ready_cells())
            .filter(|(_, c)| c.shown_layers(&old) != c.shown_layers(&hidden))
            .map(|(k, c)| (*k, Arc::clone(c)))
            .collect();
        for (key, cell) in changed {
            match cell.with_hidden(&hidden) {
                Ok(c) => {
                    self.cache.replace_ready(&key, Arc::new(c));
                    self.changes.push((key, self.cache.state(&key).cloned()));
                    self.marks.push_back(Mark::Arrival(key));
                }
                Err(reason) => tracing::warn!(key = %key, "cell composite failed: {reason}"),
            }
        }
    }

    /// Starts the layer view for `build` (`rendering.md` section 7 item
    /// 7): the `hide-layers` ids `hide` when given, else the build's
    /// `shown` defaults. Returns the ids of `hide` the build does not have,
    /// which are reported and ignored.
    pub fn start_layers(
        &mut self,
        build: &crate::protocol::BuildDescription,
        hide: Option<&[String]>,
    ) -> Vec<String> {
        let (view, unknown) = crate::layers::LayerView::for_build(build, hide);
        self.set_hidden_layers(view.hidden.clone());
        self.surfaces.set_required_layers(build.required_layers());
        self.layer_view = view;
        unknown
    }

    /// Applies the layer chooser's toggles
    /// ([`crate::controls::Controls::take_layer_toggles`]); they take
    /// effect on the next frame drawn. Returns `true` when the hidden set
    /// changed, so the caller tells the fetcher.
    pub fn toggle_layers(&mut self, toggles: &[usize]) -> bool {
        let mut view = self.layer_view.clone();
        if view.apply(toggles) {
            self.set_hidden_layers(view.hidden.clone());
            self.layer_view = view;
            return true;
        }
        false
    }

    /// The build's layers and which are hidden.
    pub fn layer_view(&self) -> &crate::layers::LayerView {
        &self.layer_view
    }

    /// The hidden layers, by index in the build.
    pub fn hidden_layers(&self) -> &std::collections::BTreeSet<usize> {
        self.surfaces.hidden_layers()
    }

    /// The surface tiles to request now.
    pub fn take_surface_requests(&mut self) -> Vec<gx_core::key::SurfaceKey> {
        self.surfaces.take_requests()
    }

    /// The surface tiles.
    pub fn surfaces(&self) -> &crate::tiles::Surfaces {
        &self.surfaces
    }

    /// Applies a fetch result: the cell's state, and marks for the cells
    /// whose extraction inputs it can change.
    /// A cell with a hidden layer is composited from its shown layers.
    pub fn complete(&mut self, key: CellKey, outcome: FetchOutcome, now: f64) {
        let hidden = self.surfaces.hidden_layers();
        let outcome = match outcome {
            FetchOutcome::Decoded(Some(cell))
                if cell.layers.iter().any(|(l, _)| hidden.contains(l)) =>
            {
                match cell.with_hidden(hidden) {
                    Ok(c) => FetchOutcome::Decoded(Some(Arc::new(c))),
                    Err(reason) => FetchOutcome::Invalid(reason),
                }
            }
            o => o,
        };
        self.cache.complete(key, outcome, now);
        self.changes.push((key, self.cache.state(&key).cloned()));
        self.arrived_since_selection = true;
        self.marks.push_back(Mark::Arrival(key));
    }

    /// One step of the frame thread's matter work (module docs): applies
    /// `results`, selects if due, takes the next requests, evicts every
    /// [`EVICT_INTERVAL_SECONDS`], collects finished meshes, builds the
    /// draw list for `seen`, and then gathers extraction inputs until
    /// [`STEP_SLICE_SECONDS`] after `started` (seconds of
    /// [`crate::clock::now_seconds`]). Returns the cells wanted now (once
    /// per new selection, see [`World::take_wanted`]) and the draw list;
    /// the surface tiles wanted come from [`World::take_wanted_surfaces`].
    pub fn step(
        &mut self,
        seen: &Seen,
        view_px: (u32, u32),
        now: f64,
        results: impl IntoIterator<Item = (CellKey, FetchOutcome)>,
        started: f64,
    ) -> (Option<Vec<(CellKey, f64)>>, DrawList) {
        for (key, outcome) in results {
            self.complete(key, outcome, now);
        }
        self.select(seen.system(), seen.camera(), view_px, now, false);
        self.select_surfaces(seen.system(), seen.camera(), view_px, now);
        let requests = self.take_wanted(now);
        if now - self.last_evict >= EVICT_INTERVAL_SECONDS {
            self.last_evict = now;
            self.evict(now);
        }
        for (key, done, seconds) in self.pool.try_results() {
            self.accept(key, done, seconds);
        }
        let draw = self.draw_list(seen, view_px, now);
        self.pump_until(Some(started + STEP_SLICE_SECONDS));
        (requests, draw)
    }

    /// Collects finished meshes and gathers extraction inputs for every
    /// marked cell, with no time limit.
    pub fn pump(&mut self) {
        self.pump_until(None);
    }

    /// Collects finished meshes, turns marks into marked cells, and looks
    /// at marked cells (then [`AUDIT_PER_STEP`] cells in turn), until the
    /// clock passes `deadline` (seconds of [`crate::clock::now_seconds`]),
    /// or until done without one. A marked `Ready` cell whose inputs differ
    /// from those its mesh was extracted with is queued for extraction; one
    /// already being extracted is looked at again once that is done.
    pub fn pump_until(&mut self, deadline: Option<f64>) {
        for (key, done, seconds) in self.pool.try_results() {
            self.accept(key, done, seconds);
        }
        let over = || deadline.is_some_and(|d| now_seconds() > d);
        while let Some(m) = self.marks.pop_front() {
            self.apply_mark(m);
            if over() {
                return;
            }
        }
        while let Some(k) = self.dirty_queue.pop_front() {
            self.dirty.remove(&k);
            self.look_at(k);
            if over() {
                return;
            }
        }
        for _ in 0..AUDIT_PER_STEP {
            if self.audit.is_empty() {
                self.audit = self
                    .cache
                    .entries()
                    .filter(|(_, e)| matches!(e.state, CellState::Ready(_)))
                    .map(|(k, _)| *k)
                    .collect();
                self.audit.sort_unstable_by(|a, b| b.cmp(a));
                if self.audit.is_empty() {
                    break;
                }
            }
            if let Some(k) = self.audit.pop() {
                self.look_at(k);
            }
        }
    }

    /// Marks `k` if it is `Ready`.
    fn mark(&mut self, k: CellKey) {
        if matches!(self.cache.state(&k), Some(CellState::Ready(_))) && self.dirty.insert(k) {
            self.dirty_queue.push_back(k);
        }
    }

    fn apply_mark(&mut self, m: Mark) {
        match m {
            Mark::Arrival(a) => {
                // A cell's region holds the cells around it at its depth,
                // one depth coarser, and one depth finer
                // ([`CellInputs::region`]).
                self.mark(a);
                if let Some(p) = a.parent() {
                    self.mark(p);
                    for o in AROUND {
                        if let Some(c) = offset_key(&p, o) {
                            self.mark(c);
                        }
                    }
                }
                for c in std::iter::once(Some(a)).chain(AROUND.iter().map(|&o| offset_key(&a, o))) {
                    let Some(c) = c else {
                        continue;
                    };
                    if c != a {
                        self.mark(c);
                    }
                    for ch in c.children().into_iter().flatten() {
                        self.mark(ch);
                    }
                }
            }
            Mark::Layout(c) => {
                // Cells up to two over at its depth, at the depth below up
                // to two over from its children, and at the depth above
                // around its parent.
                let block = |k: CellKey, lo: i64, hi: i64| -> Vec<CellKey> {
                    let mut out = Vec::new();
                    for dz in lo..=hi {
                        for dy in lo..=hi {
                            for dx in lo..=hi {
                                let at = |v: u32, d: i64| u32::try_from(i64::from(v) + d).ok();
                                if let (Some(x), Some(y), Some(z)) =
                                    (at(k.x, dx), at(k.y, dy), at(k.z, dz))
                                {
                                    if let Ok(n) = CellKey::new(k.frame_id, k.depth, x, y, z) {
                                        out.push(n);
                                    }
                                }
                            }
                        }
                    }
                    out
                };
                for k in block(c, -2, 2) {
                    self.mark(k);
                }
                if let Some(children) = c.children() {
                    for k in block(children[0], -2, 3) {
                        self.mark(k);
                    }
                }
                if let Some(p) = c.parent() {
                    for k in block(p, -1, 1) {
                        self.mark(k);
                    }
                }
            }
        }
    }

    /// Queues `k` for extraction if it is drawn and `Ready` and its
    /// inputs differ from those of its mesh.
    fn look_at(&mut self, k: CellKey) {
        if self.extracting.contains_key(&k) {
            self.deferred.insert(k);
            return;
        }
        let Some(inputs) = self.inputs_of(&k) else {
            return;
        };
        let held = self.meshes.get(&k).map(|e| &e.inputs);
        if held
            .into_iter()
            .chain(self.waiting.get(&k))
            .any(|h| h.same_as(&inputs))
        {
            return;
        }
        self.extracting.insert(k, inputs.clone());
        self.pool.submit(inputs);
    }

    fn accept(&mut self, key: CellKey, done: Extraction, seconds: f64) {
        let Some(inputs) = self.extracting.remove(&key) else {
            return;
        };
        self.extraction_times.record(seconds * 1e3);
        let ready = matches!(self.cache.state(&key), Some(CellState::Ready(_)));
        match done {
            Extraction::Done(mesh) => {
                self.waiting.remove(&key);
                if ready {
                    self.meshes.insert(key, Extracted { mesh, inputs });
                }
            }
            Extraction::Waiting => {
                if ready {
                    self.waiting.insert(key, inputs);
                }
            }
        }
        if self.deferred.remove(&key) {
            self.mark(key);
        }
    }

    /// Pumps until nothing is marked, queued, or running.
    pub fn finish_extraction(&mut self) {
        self.pump();
        while !self.extracting.is_empty() || !self.dirty_queue.is_empty() {
            if !self.extracting.is_empty() {
                match self.pool.wait_result() {
                    Some((key, done, seconds)) => self.accept(key, done, seconds),
                    None => break,
                }
            }
            self.pump();
        }
    }

    /// Evicts old cells and their meshes, then evicts against the memory
    /// budget (`crate::budget`).
    pub fn evict(&mut self, now: f64) {
        for k in self.cache.evict(now) {
            self.changes.push((k, None));
            self.meshes.remove(&k);
            self.waiting.remove(&k);
            self.marks.push_back(Mark::Arrival(k));
        }
        self.enforce_budget(true);
    }

    /// For every frame with a drawn mesh, the distance from `camera` to
    /// the nearest of their bounding boxes, meters: how far the light from
    /// that frame's nearest matter travels ([`crate::seen::Seen::retarded_near`]).
    /// A camera inside a box is at 0.
    pub fn matter_distances(&self, system: &FrameSystem, camera: &Camera) -> BTreeMap<u64, f64> {
        let mut out: BTreeMap<u64, f64> = BTreeMap::new();
        let mut cams: BTreeMap<u64, Vec3> = BTreeMap::new();
        for (key, e) in &self.meshes {
            let m = &e.mesh;
            if m.indices.is_empty() || system.tree().get(key.frame_id).is_none() {
                continue;
            }
            let cam = *cams
                .entry(key.frame_id)
                .or_insert_with(|| crate::stream::camera_in_frame(system, camera, key.frame_id));
            let gap = |c: f64, l: f64, h: f64| (l - c).max(c - h).max(0.0);
            let d = Vec3::new(
                gap(cam.x, m.bounds_min[0], m.bounds_max[0]),
                gap(cam.y, m.bounds_min[1], m.bounds_max[1]),
                gap(cam.z, m.bounds_min[2], m.bounds_max[2]),
            )
            .length();
            let v = out.entry(key.frame_id).or_insert(d);
            *v = v.min(d);
        }
        out
    }

    /// The distance from `frame_id`'s origin along the unit `direction`
    /// (frame axes) of the outermost drawn surface on that line, meters:
    /// the line against the triangles of the meshes in the draw set. `None`
    /// when no drawn mesh of the frame crosses it.
    pub fn surface_along(&self, frame_id: u64, direction: Vec3) -> Option<f64> {
        let d = [direction.x, direction.y, direction.z];
        // A frame drawn from surface tiles: the drawn tiles' triangles.
        if self.surfaces.has_surface(frame_id) {
            let indices = crate::tiles::TileGeometry::indices();
            let mut best: Option<f64> = None;
            for t in self.surfaces.drawn() {
                let g = &t.geometry;
                if g.key.frame_id != frame_id
                    || crate::spawn::ray_box(g.bounds.0, g.bounds.1, direction).is_none()
                {
                    continue;
                }
                for tri in indices[t.indices.start as usize..t.indices.end as usize]
                    .as_chunks::<3>()
                    .0
                {
                    let [a, b, c] = tri.map(|i| g.positions[i as usize]);
                    if let Some(t) = ray_triangle(d, a, b, c) {
                        if best.is_none_or(|r| t > r) {
                            best = Some(t);
                        }
                    }
                }
            }
            return best;
        }
        let mut best: Option<f64> = None;
        for key in self.cache.draw_set(|k| self.drawable(k)) {
            if key.frame_id != frame_id {
                continue;
            }
            let Some(m) = self.meshes.get(&key).map(|e| &e.mesh) else {
                continue;
            };
            if m.indices.is_empty()
                || crate::spawn::ray_box(m.bounds_min, m.bounds_max, direction).is_none()
            {
                continue;
            }
            for tri in m.indices.as_chunks::<3>().0 {
                let [a, b, c] = tri.map(|i| m.positions[i as usize]);
                if let Some(t) = ray_triangle(d, a, b, c) {
                    if best.is_none_or(|r| t > r) {
                        best = Some(t);
                    }
                }
            }
        }
        best
    }

    /// Whether a cell can be drawn now: it holds no matter, or it is
    /// `Ready` with its mesh.
    fn drawable(&self, key: &CellKey) -> bool {
        match self.cache.state(key) {
            Some(CellState::Empty) => true,
            Some(CellState::Ready(c)) => !c.has_matter || self.meshes.contains_key(key),
            _ => key.depth > 0 && self.cache.empty_above(key),
        }
    }

    /// Nothing is marked, queued for extraction, or being extracted, and
    /// no selection is in flight.
    fn extraction_idle(&self) -> bool {
        self.marks.is_empty()
            && self.dirty_queue.is_empty()
            && self.extracting.is_empty()
            && !self.selector.busy()
    }

    /// Returns `true` when nothing the selection needs is still on its way:
    /// every cell of the target has arrived or is `Gone`, every selected
    /// cell is drawable or `Gone`, every pinned cell has arrived or is
    /// `Gone`, and nothing is being extracted. A pinned cell is read for
    /// the far field and lights; it gets a mesh only when it is drawn.
    pub fn settled(&self) -> bool {
        let resolved =
            |k: &CellKey| self.drawable(k) || self.cache.state(k) == Some(&CellState::Gone);
        self.extraction_idle()
            && self.surfaces.settled()
            && self.cache.outstanding() == 0
            && self.cache.selection().iter().all(|s| resolved(&s.key))
            && self
                .cache
                .pinned()
                .all(|k| self.arrived(k) || self.cache.state(k) == Some(&CellState::Gone))
    }

    /// What keeps [`World::settled`] false, for logs: the first unmet
    /// condition, or `None` when settled.
    pub fn unsettled(&self) -> Option<&'static str> {
        let resolved =
            |k: &CellKey| self.drawable(k) || self.cache.state(k) == Some(&CellState::Gone);
        if !self.marks.is_empty() {
            Some("marks")
        } else if !self.dirty_queue.is_empty() {
            Some("dirty queue")
        } else if !self.extracting.is_empty() {
            Some("extracting")
        } else if !self.surfaces.settled() {
            Some("surface tiles")
        } else if self.cache.outstanding() != 0 {
            Some("cells outstanding")
        } else if !self.cache.selection().iter().all(|s| resolved(&s.key)) {
            Some("selection not drawable")
        } else if !self
            .cache
            .pinned()
            .all(|k| self.arrived(k) || self.cache.state(k) == Some(&CellState::Gone))
        {
            Some("pinned cells")
        } else if self.selector.busy() {
            Some("selecting")
        } else {
            None
        }
    }

    /// Whether anything is still on its way or being worked on, apart from
    /// a selection in flight (one runs every
    /// [`crate::stream::SELECT_INTERVAL_SECONDS`] anyway): the window loop
    /// ticks at the display rate while this holds, and slowly otherwise.
    pub fn busy(&self) -> bool {
        !matches!(self.unsettled(), None | Some("selecting"))
    }

    /// Returns `true` when every selected cell is drawn at its selected
    /// depth: the drawn selection is the target, every selected cell can
    /// be drawn, every pinned cell has arrived, and nothing is being
    /// extracted, and no selection is due for cells that arrived since the
    /// last one (a paced selection, [`World::select`], may still be
    /// waiting). Stricter than [`World::settled`]: a `Gone` cell is not
    /// ready.
    pub fn ready(&self) -> bool {
        self.extraction_idle()
            && self.surfaces.settled()
            && !self.arrived_since_selection
            && self.cache.outstanding() == 0
            && self.cache.at_target()
            && self.cache.selection().iter().all(|s| self.drawable(&s.key))
            && self.cache.pinned().all(|k| self.arrived(k))
    }

    /// Whether a cell is `Ready` or `Empty`.
    fn arrived(&self, key: &CellKey) -> bool {
        matches!(
            self.cache.state(key),
            Some(CellState::Ready(_) | CellState::Empty)
        )
    }

    /// The draw set (see [`CellCache::draw_set`]), its meshes and volumes,
    /// the lights its hot matter makes, and the sprites of far frames, for
    /// a view of `view_px` pixels. Marks the drawn cells as used.
    ///
    /// Lights come from the depth-0 cell of every frame: it holds all of the
    /// frame's matter, so the light escaping it is the light the frame
    /// sends into space ([`crate::light`]). Cold far frames reflect those
    /// lights.
    ///
    /// Everything is placed as `seen` sees it ([`crate::seen`]): every
    /// frame at its retarded state, and each frame with meshes or a sprite
    /// lit from where each emitter was one light-travel time before that
    /// frame is seen.
    pub fn draw_list(&mut self, seen: &Seen, view_px: (u32, u32), now: f64) -> DrawList {
        let (system, camera) = (seen.system(), seen.camera());
        let view_height = f64::from(view_px.1.max(1));
        let far = far_frames_with_surfaces(
            system,
            camera,
            view_height,
            |f| crate::stream::depth_zero_radius(&self.cache, f),
            |f| self.surfaces.has_surface(f),
        );
        let cell_weights: BTreeMap<u64, f32> = far
            .iter()
            .map(|f| (f.frame_id, f.cell_weight() as f32))
            .collect();
        let tile_weights: BTreeMap<u64, f32> = far
            .iter()
            .map(|f| (f.frame_id, f.tile_weight() as f32))
            .collect();
        let tile_weight = |frame: u64| tile_weights.get(&frame).copied().unwrap_or(1.0);
        let weight = |frame: u64| cell_weights.get(&frame).copied().unwrap_or(1.0);
        self.update_shown();
        let (keys, shown_meshes) = match &self.shown {
            Some(s) => (s.keys.clone(), s.meshes.clone()),
            None => (Vec::new(), Vec::new()),
        };
        self.cache.touch(&keys, now);
        let mut volumes = Vec::new();
        for k in keys.iter().filter(|k| weight(k.frame_id) > 0.0) {
            if let Some(CellState::Ready(cell)) = self.cache.state(k) {
                if let Some(v) = &cell.volume {
                    volumes.push(v.clone());
                }
            }
        }
        let meshes: Vec<Arc<SurfaceMesh>> = shown_meshes
            .into_iter()
            .filter(|m| weight(m.key.frame_id) > 0.0 && !m.is_empty())
            .collect();
        let mut emitters = Vec::new();
        for f in system.tree().frames() {
            if let Some(CellState::Ready(cell)) = self.cache.state(&depth_zero_key(f.frame_id)) {
                if let Some(e) = &cell.emitter {
                    emitters.push((f.frame_id, e));
                }
            }
        }
        let merged = merge_frame_emitters(emitters.iter().map(|(f, e)| (*f, *e)));
        let lights = strongest_lights(&merged, system, camera);
        let mut frame_lights = BTreeMap::new();
        let mut lit = |frame: u64| -> Vec<PointLight> {
            frame_lights
                .entry(frame)
                .or_insert_with(|| lights_for(&lights, frame, |l, e| seen.emitter_shift(l, e)))
                .clone()
        };
        for m in &meshes {
            lit(m.key.frame_id);
        }
        let tiles: Vec<crate::tiles::TileDrawInfo> = self
            .surfaces
            .drawn()
            .iter()
            .filter(|t| tile_weight(t.geometry.key.frame_id) > 0.0)
            .cloned()
            .collect();
        for t in &tiles {
            lit(t.geometry.key.frame_id);
        }
        let sprites = far
            .iter()
            .filter_map(|f| match self.cache.state(&depth_zero_key(f.frame_id)) {
                Some(CellState::Ready(cell)) => frame_sprite(
                    system,
                    camera,
                    f,
                    &cell.far_field(),
                    &lit(f.frame_id),
                    view_height,
                ),
                _ => None,
            })
            .collect();
        DrawList {
            meshes,
            tiles,
            volumes,
            lights,
            frame_lights,
            sprites,
            cell_weights,
            tile_weights,
        }
    }

    /// Whether the drawn selection's cells are all extracted for exactly
    /// that selection: nothing marked, queued, or being extracted (so every
    /// `Ready` cell's held mesh was looked at against its current inputs),
    /// and every selected cell is drawable without a stand-in and not
    /// waiting for an input.
    fn consistent(&self) -> bool {
        self.marks.is_empty()
            && self.dirty_queue.is_empty()
            && self.extracting.is_empty()
            && self
                .cache
                .selection()
                .iter()
                .all(|s| !self.waiting.contains_key(&s.key) && self.drawable(&s.key))
    }

    /// Takes the drawn selection as the shown set when it is consistent
    /// ([`World::consistent`]); otherwise the shown set stays.
    fn update_shown(&mut self) {
        if !self.consistent() {
            return;
        }
        let keys: Vec<CellKey> = self.cache.selection().iter().map(|s| s.key).collect();
        self.building_since = None;
        if self.shown.as_ref().is_none_or(|s| s.keys != keys) {
            self.shown_switches += 1;
        }
        let meshes = keys
            .iter()
            .filter_map(|k| self.meshes.get(k).map(|e| e.mesh.clone()))
            .collect();
        self.shown = Some(Shown { keys, meshes });
    }

    /// Open edges of the shown meshes of `frame_id`: directed edges whose
    /// reverse no shown mesh uses, with positions compared on a grid of
    /// `tolerance` meters (shared vertices of two cells agree only to
    /// rounding). Edges on the frame's root cube boundary are not counted.
    /// A watertight shown set has none (`rendering.md` 2.5); this is the
    /// check on real data, while the camera moves.
    pub fn open_edges(&self, frame_id: u64, root_extent: f64, tolerance: f64) -> usize {
        let Some(shown) = &self.shown else {
            return 0;
        };
        let q = |p: [f64; 3]| p.map(|v| (v / tolerance).round() as i64);
        let half = (0.5 * root_extent / tolerance).round() as i64;
        let on_root = |a: [i64; 3], b: [i64; 3]| {
            (0..3).any(|i| (a[i].abs() - half).abs() <= 1 && (b[i].abs() - half).abs() <= 1)
        };
        let mut edges: std::collections::HashMap<([i64; 3], [i64; 3]), i64> =
            std::collections::HashMap::new();
        for m in shown.meshes.iter().filter(|m| m.key.frame_id == frame_id) {
            for t in m.indices.as_chunks::<3>().0 {
                let p = t.map(|i| q(m.positions[i as usize]));
                if p[0] == p[1] || p[1] == p[2] || p[0] == p[2] {
                    continue;
                }
                for k in 0..3 {
                    let (a, b) = (p[k], p[(k + 1) % 3]);
                    *edges.entry((a, b)).or_default() += 1;
                    *edges.entry((b, a)).or_default() -= 1;
                }
            }
        }
        edges
            .iter()
            .filter(|((a, b), &n)| n > 0 && !on_root(*a, *b))
            .count()
    }

    /// The shown cells, empty before anything is shown.
    pub fn shown_keys(&self) -> &[CellKey] {
        self.shown.as_ref().map_or(&[], |s| &s.keys)
    }

    /// Times the shown set was replaced by a different selection.
    pub fn shown_switches(&self) -> u64 {
        self.shown_switches
    }

    /// What the renderer holds now (`docs/architecture.md`, "Memory
    /// report"): the allocator's counters, surface tiles, cells, meshes
    /// (each `Arc` counted once), and the decode timings.
    #[cfg(feature = "memory-report")]
    pub fn memory_report(&self) -> crate::memory::MemoryReport {
        let ptr = |m: &Arc<SurfaceMesh>| Arc::as_ptr(m) as usize;
        let mut all = BTreeMap::new();
        for e in self.meshes.values() {
            all.insert(ptr(&e.mesh), e.mesh.heap_bytes());
        }
        let mut shown = BTreeMap::new();
        for m in self.shown.iter().flat_map(|s| &s.meshes) {
            all.insert(ptr(m), m.heap_bytes());
            shown.insert(ptr(m), m.heap_bytes());
        }
        crate::memory::MemoryReport {
            allocator: crate::memory::AllocatorBytes::now(),
            surfaces: self.surfaces.memory(),
            cells: self.cache.memory(),
            meshes: crate::memory::MeshMemory {
                count: all.len(),
                bytes: all.values().sum(),
                shown: shown.len(),
                shown_bytes: shown.values().sum(),
            },
            budget: crate::memory::BudgetMemory {
                budget: self.budget,
                held: self.held,
                holding_ahead: self.holding_ahead,
                detail: self.detail.factor(),
                moving: self.detail.moving(),
                last: self.last_evicted,
                total: self.total_evicted,
            },
            timings: crate::memory::Timings::now(),
        }
    }

    /// Counts for the overlay.
    pub fn counts(&self) -> WorldCounts {
        WorldCounts {
            cache: self.cache.counts(),
            meshes: self.meshes.len(),
            extracting: self.extracting.len(),
        }
    }
}

/// Where the ray `t * d`, `t > 0`, from the origin crosses triangle
/// `a b c` (Moller-Trumbore), or `None`.
fn ray_triangle(d: [f64; 3], a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> Option<f64> {
    let sub = |x: [f64; 3], y: [f64; 3]| [x[0] - y[0], x[1] - y[1], x[2] - y[2]];
    let cross = |x: [f64; 3], y: [f64; 3]| {
        [
            x[1] * y[2] - x[2] * y[1],
            x[2] * y[0] - x[0] * y[2],
            x[0] * y[1] - x[1] * y[0],
        ]
    };
    let dot = |x: [f64; 3], y: [f64; 3]| x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
    let (e1, e2) = (sub(b, a), sub(c, a));
    let p = cross(d, e2);
    let det = dot(e1, p);
    if det.abs() < 1e-30 {
        return None;
    }
    let inv = 1.0 / det;
    let s = sub([0.0; 3], a);
    let u = dot(s, p) * inv;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = cross(s, e1);
    let v = dot(d, q) * inv;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = dot(e2, q) * inv;
    (t > 0.0).then_some(t)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stream::ReadyCell;
    use gx_core::matter::{Sample, Samples, Section, State};
    use gx_core::units::{Attenuation, Density, Kelvin, Meters, Ratio, Vec3};

    /// Cell `(x, y, z)` at depth 1 of a frame with a 16 m root, at
    /// resolution 4, solid where `fill` holds for the sample center.
    fn cell(x: u32, y: u32, z: u32, fill: impl Fn([f64; 3]) -> bool) -> (CellKey, FetchOutcome) {
        let key = CellKey::new(3, 1, x, y, z).unwrap();
        let solid = Sample {
            density: Density::new(1000.0),
            state: State::Solid,
            temperature: Kelvin::new(300.0),
            albedo: [Ratio::new(0.5); 3],
            roughness: Ratio::new(0.5),
            attenuation: Attenuation::new(0.0),
            radiating_temperature: Kelvin::new(300.0),
            occupancy: Ratio::new(1.0),
        };
        let o = [x, y, z].map(|v| -8.0 + 8.0 * f64::from(v));
        let section = Section::new(
            key,
            Vec3::new(o[0], o[1], o[2]),
            Meters::new(8.0),
            4,
            Samples::from_fn(4, |i, j, k| {
                let c = [i, j, k];
                let p: [f64; 3] = std::array::from_fn(|a| o[a] + 2.0 * f64::from(c[a]) + 1.0);
                if fill(p) {
                    solid
                } else {
                    Sample::VACUUM
                }
            }),
        )
        .unwrap();
        let ready = ReadyCell::new(section);
        (key, FetchOutcome::Decoded(Some(Arc::new(ready))))
    }

    #[test]
    fn a_hidden_plain_layer_is_composited_out_without_a_fetch() {
        let section = |o: FetchOutcome| match o {
            FetchOutcome::Decoded(Some(c)) => Arc::clone(&c.section),
            _ => panic!("a cell"),
        };
        let (key, low) = cell(0, 0, 0, |p| p[2] < -4.0);
        let (_, high) = cell(0, 0, 0, |p| p[2] > -4.0);
        let both = crate::stream::cell_from_sections(
            &key,
            &[section(low), section(high)],
            &[1, 2],
            Meters::new(16.0),
        );
        let mut w = World::new(1);
        w.complete(key, both, 0.0);
        let drawn = |w: &World| match w.cache().state(&key) {
            Some(CellState::Ready(c)) => Arc::clone(c),
            s => panic!("{s:?}"),
        };
        let full = drawn(&w);
        w.set_hidden_layers([2].into());
        let low_only = drawn(&w);
        assert_ne!(low_only.section, full.section);
        assert_eq!(low_only.layers, full.layers);
        assert_eq!(w.cache().counts().in_flight, 0, "nothing is fetched");
        w.set_hidden_layers([1, 2].into());
        assert!(!drawn(&w).has_matter);
        w.set_hidden_layers(Default::default());
        assert_eq!(drawn(&w).section, full.section);
        assert_eq!(drawn(&w).mean_albedo, full.mean_albedo);
    }

    #[test]
    fn a_drawn_cell_waits_for_its_inputs_and_is_extracted_once() {
        let mut world = World::new(0);
        // A solid ball of radius 5 m around the root's center, drawn as
        // the eight cells of depth 1, of which only one has arrived.
        let ball = |p: [f64; 3]| p.iter().map(|v| v * v).sum::<f64>() < 25.0;
        let all: Vec<[u32; 3]> = (0..8).map(|i| [i & 1, (i >> 1) & 1, i >> 2]).collect();
        for c in &all {
            let k = CellKey::new(3, 1, c[0], c[1], c[2]).unwrap();
            world.layout.selected.insert(k);
        }
        let (a, fetched) = cell(0, 0, 0, ball);
        world.complete(a, fetched, 0.0);
        world.finish_extraction();
        // Its neighbors are neither present nor known empty: no mesh.
        assert!(world.mesh(&a).is_none());

        for c in &all[1..] {
            let (k, fetched) = cell(c[0], c[1], c[2], ball);
            world.complete(k, fetched, 0.0);
        }
        world.finish_extraction();
        let mesh = world.mesh(&a).unwrap().clone();
        assert!(!mesh.is_empty());
        // The surface lies inside its own cell.
        assert!(mesh
            .positions
            .iter()
            .all(|p| p.iter().all(|v| (-8.0..=0.0).contains(v))));
        assert_eq!(world.mesh_inputs(&a).unwrap().drawn.len(), 8);
        assert!(world.extraction_times().count() >= 8);

        // Nothing changed: nothing is extracted again.
        world.pump();
        assert_eq!(world.counts().extracting, 0);
        assert!(Arc::ptr_eq(world.mesh(&a).unwrap(), &mesh));
    }

    /// One frame of a body with surface tiles to level 12 whose root is
    /// frame 4 (the tiles' frame), and a camera 20 km over face 4 looking
    /// 45 degrees down.
    struct SurfaceRig {
        system: FrameSystem,
        camera: Camera,
        cam: [f64; 3],
        forward: [f64; 3],
    }

    const SURFACE_VIEW: (u32, u32) = (1280, 720);

    fn surface_rig() -> SurfaceRig {
        use gx_core::registry::{Frame, FrameTree, Registry, ROOT_PARENT};
        use gx_core::units::{Kilograms, Quat, Seconds};
        let frame = Frame {
            frame_id: 4,
            parent_frame_id: ROOT_PARENT,
            root_extent: Meters::new(1.6e7),
            max_depth: 4,
            mass: Kilograms::new(6.0e24),
            position: Vec3::zero(),
            velocity: Vec3::zero(),
            orientation: Quat::identity(),
            angular_velocity: Vec3::zero(),
            j2: 0.0,
            reference_radius: Meters::new(0.0),
            principal_moments: Vec3::zero(),
        };
        let reg = Registry::new(Seconds::new(0.0), vec![frame]).unwrap();
        let system = FrameSystem::from_tree(FrameTree::from_registries(&[reg]).unwrap());
        let key = gx_core::key::SurfaceKey::new(4, 4, 0, 0, 0).unwrap();
        let d = gx_core::surface::vertex_direction(&key, 20, 9);
        let (p, n) = gx_core::surface::ellipsoid_point(d, 6_378_137.0, 6_356_752.3);
        let cam: [f64; 3] = std::array::from_fn(|k| p[k] + 20_000.0 * n[k]);
        let side = Vec3::new(0.0, 0.0, 1.0).cross(Vec3::new(n[0], n[1], n[2]));
        let side = side.scale(1.0 / side.length());
        let h = std::f64::consts::FRAC_1_SQRT_2;
        let forward: [f64; 3] = std::array::from_fn(|k| h * [side.x, side.y, side.z][k] - h * n[k]);
        let camera = Camera {
            frame_id: 4,
            position: Vec3::new(cam[0], cam[1], cam[2]),
            orientation: crate::camera::look_rotation(
                Vec3::new(forward[0], forward[1], forward[2]),
                Vec3::new(n[0], n[1], n[2]),
            ),
            speed: 1.0,
            cruise: 1.0,
            fov_y: crate::camera::FIELD_OF_VIEW_Y,
        };
        SurfaceRig {
            system,
            camera,
            cam,
            forward,
        }
    }

    /// One frame at `now`: the surface selection, every request answered
    /// and every build done at once. Returns how many tiles were built.
    fn surface_frame(world: &mut World, rig: &SurfaceRig, now: f64) -> usize {
        world.select_surfaces(&rig.system, &rig.camera, SURFACE_VIEW, now);
        for key in world.take_surface_requests() {
            let sections = if key.level <= 12 {
                vec![gx_core::surface::encode(
                    &crate::tiles::tests::tile(key),
                    gx_core::matter::Compression::None,
                )]
            } else {
                Vec::new()
            };
            let decoded = crate::tiles::decode_tile(&key, &sections, &[0], &Default::default());
            world.complete_surface(key, decoded, false);
        }
        let builds = world.take_surface_builds();
        let n = builds.len();
        for req in builds {
            let outcome = crate::tiles::build_tile(&req);
            world.complete_surface_build(req.key, outcome);
        }
        n
    }

    fn drawn_keys(world: &World) -> Vec<gx_core::key::SurfaceKey> {
        let mut k: Vec<_> = (world.surfaces().drawn().iter())
            .map(|t| t.geometry.key)
            .collect();
        k.sort();
        k
    }

    /// Frames at a still camera from `now` until the detail factor, the
    /// drawn tiles, and the builds have not changed for 300 frames.
    fn settle_surfaces(world: &mut World, rig: &SurfaceRig, now: &mut f64) {
        let mut last = (Vec::new(), f64::NAN);
        let mut still = 0;
        for _ in 0..20_000 {
            *now += 1.0 / 60.0;
            let built = surface_frame(world, rig, *now);
            let state = (drawn_keys(world), world.detail_factor());
            if built == 0 && state == last {
                still += 1;
                if still == 300 {
                    return;
                }
            } else {
                still = 0;
            }
            last = state;
        }
        panic!("the surface selection never settled");
    }

    #[test]
    fn over_the_budget_detail_yields_and_the_view_settles() {
        let rig = surface_rig();
        let mut world = World::new(0);
        let default = world.memory_budget();
        let mut now = 0.0;
        settle_surfaces(&mut world, &rig, &mut now);
        assert_eq!(world.detail_factor(), 1.0);
        let full = drawn_keys(&world);
        let reference = world.surfaces().drawn().to_vec();
        let needed = world.held_bytes();
        // Room for about 70 percent of what the view needs at full detail.
        let budget = needed * 7 / 10;
        world.set_memory_budget(budget);
        settle_surfaces(&mut world, &rig, &mut now);
        let factor = world.detail_factor();
        assert!(factor < 1.0, "detail did not yield");
        let drawn = drawn_keys(&world);
        let evicted = world.total_evicted();
        let mut built = 0;
        for i in 0..300 {
            now += 1.0 / 60.0;
            built += surface_frame(&mut world, &rig, now);
            assert_eq!(world.total_evicted(), evicted, "frame {i}: evicted again");
            assert_eq!(drawn_keys(&world), drawn, "frame {i}: drawn tiles changed");
            assert_eq!(world.detail_factor(), factor, "frame {i}: detail changed");
            let held = world.held_bytes();
            assert!(held <= budget, "frame {i}: {held} bytes over {budget}");
        }
        assert_eq!(built, 0, "tiles built again at a still camera");
        crate::tiles::tests::covered(world.surfaces().drawn(), &reference, rig.cam, rig.forward);
        println!(
            "detail: full detail {} tiles in {needed} bytes; budget {budget}: factor {factor}, {} tiles drawn, held {}, evicted in all {evicted:?}, built over 300 settled frames {built}",
            full.len(),
            drawn.len(),
            world.held_bytes()
        );
        // The default budget again: the unconstrained run's tiles.
        world.set_memory_budget(default);
        settle_surfaces(&mut world, &rig, &mut now);
        assert_eq!(world.detail_factor(), 1.0);
        assert_eq!(drawn_keys(&world), full);
    }

    /// `rig` with the camera moved `offset` meters along the surface
    /// ahead of its view (straight, so its height grows by under a
    /// kilometer over tens of kilometers).
    fn rig_moved(rig: &SurfaceRig, offset: f64) -> (Camera, [f64; 3]) {
        let c = rig.cam;
        let l = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
        let up: [f64; 3] = std::array::from_fn(|k| c[k] / l);
        let f = rig.forward;
        let along = f[0] * up[0] + f[1] * up[1] + f[2] * up[2];
        let ahead: [f64; 3] = std::array::from_fn(|k| f[k] - along * up[k]);
        let al = (ahead[0] * ahead[0] + ahead[1] * ahead[1] + ahead[2] * ahead[2]).sqrt();
        let cam: [f64; 3] = std::array::from_fn(|k| c[k] + offset * ahead[k] / al);
        let mut camera = rig.camera;
        camera.position = Vec3::new(cam[0], cam[1], cam[2]);
        (camera, cam)
    }

    /// Below the lowest point of the synthetic tiles: rays that meet it
    /// look at the surface.
    const BELOW_SURFACE: f64 = 6_356_752.3 - 1_000.0;

    /// A world settled at the rig's still camera with a budget of about 70
    /// percent of what the view needs at full detail. Returns the world,
    /// the budget, and the clock.
    fn settled_at_seventy_percent(rig: &SurfaceRig) -> (World, u64, f64) {
        let mut world = World::new(0);
        let mut now = 0.0;
        settle_surfaces(&mut world, rig, &mut now);
        let budget = world.held_bytes() * 7 / 10;
        world.set_memory_budget(budget);
        settle_surfaces(&mut world, rig, &mut now);
        (world, budget, now)
    }

    /// One frame as [`surface_frame`] from `camera`, returning the keys
    /// built.
    fn surface_frame_keys(
        world: &mut World,
        rig: &SurfaceRig,
        camera: &Camera,
        now: f64,
    ) -> Vec<gx_core::key::SurfaceKey> {
        world.select_surfaces(&rig.system, camera, SURFACE_VIEW, now);
        for key in world.take_surface_requests() {
            let sections = if key.level <= 12 {
                vec![gx_core::surface::encode(
                    &crate::tiles::tests::tile(key),
                    gx_core::matter::Compression::None,
                )]
            } else {
                Vec::new()
            };
            let decoded = crate::tiles::decode_tile(&key, &sections, &[0], &Default::default());
            world.complete_surface(key, decoded, false);
        }
        let builds = world.take_surface_builds();
        let keys = builds.iter().map(|r| r.key).collect();
        for req in builds {
            let outcome = crate::tiles::build_tile(&req);
            world.complete_surface_build(req.key, outcome);
        }
        keys
    }

    #[test]
    fn a_still_view_settles_with_room_to_move() {
        let rig = surface_rig();
        let (mut world, budget, mut now) = settled_at_seventy_percent(&rig);
        let settle = (crate::budget::SETTLE_FRACTION * budget as f64) as u64;
        let held = world.held_bytes();
        assert!(held <= settle, "{held} bytes over {settle}");
        assert!(!world.memory_moving());
        let factor = world.detail_factor();
        let drawn = drawn_keys(&world);
        let evicted = world.total_evicted();
        let mut built = 0;
        for i in 0..300 {
            now += 1.0 / 60.0;
            built += surface_frame(&mut world, &rig, now);
            assert_eq!(world.total_evicted(), evicted, "frame {i}: evicted again");
            assert_eq!(drawn_keys(&world), drawn, "frame {i}: drawn tiles changed");
            assert_eq!(world.detail_factor(), factor, "frame {i}: detail changed");
            let held = world.held_bytes();
            assert!(held <= settle, "frame {i}: {held} bytes over {settle}");
        }
        assert_eq!(built, 0, "tiles built again at a still camera");
        println!(
            "settle: budget {budget}, held {} ({:.3} of it), factor {factor}, {} tiles drawn",
            world.held_bytes(),
            world.held_bytes() as f64 / budget as f64,
            drawn.len()
        );
    }

    #[test]
    fn a_steady_pan_keeps_detail_and_the_view_covered() {
        let rig = surface_rig();
        let (mut world, budget, mut now) = settled_at_seventy_percent(&rig);
        let start = world.total_evicted();
        let mut factors = vec![world.detail_factor()];
        let (mut built, mut peak) = (0, 0);
        // 120 km at 3 km/s: several tiles of the levels drawn near the
        // camera, and far enough that held bytes reach the budget.
        for i in 1..=2400 {
            now += 1.0 / 60.0;
            let (camera, cam) = rig_moved(&rig, 50.0 * f64::from(i));
            built += surface_frame_keys(&mut world, &rig, &camera, now).len();
            assert!(
                world.memory_moving(),
                "frame {i}: the camera counts as still"
            );
            let f = world.detail_factor();
            if factors.last() != Some(&f) {
                factors.push(f);
            }
            let held = world.held_bytes();
            peak = peak.max(held);
            assert!(held <= budget, "frame {i}: {held} bytes over {budget}");
            crate::tiles::tests::view_covered(
                world.surfaces().drawn(),
                cam,
                rig.forward,
                BELOW_SURFACE,
            );
        }
        let end = world.total_evicted();
        println!(
            "pan: detail {factors:?}, peak held {peak} of {budget}, compacted {}, tiles dropped {}, geometry builds {built}",
            end.compacted - start.compacted,
            end.tiles - start.tiles
        );
        assert!(
            factors.len() <= 2,
            "the detail factor changed more than once: {factors:?}"
        );
    }

    #[test]
    fn a_pan_away_and_back_builds_nothing_drawn_before_again() {
        let rig = surface_rig();
        let (mut world, budget, mut now) = settled_at_seventy_percent(&rig);
        let before: std::collections::BTreeSet<_> = drawn_keys(&world).into_iter().collect();
        let start = world.total_evicted();
        let mut factors = vec![world.detail_factor()];
        let (mut built, mut peak, mut again) = (0, 0, Vec::new());
        // 15 km away at 3 km/s and straight back over the same places.
        let n = 300;
        for i in 1..=2 * n {
            now += 1.0 / 60.0;
            let step = if i <= n { i } else { 2 * n - i };
            let (camera, cam) = rig_moved(&rig, 50.0 * f64::from(step));
            let keys = surface_frame_keys(&mut world, &rig, &camera, now);
            built += keys.len();
            if i > n {
                again.extend(keys.into_iter().filter(|k| before.contains(k)));
            }
            let f = world.detail_factor();
            if factors.last() != Some(&f) {
                factors.push(f);
            }
            peak = peak.max(world.held_bytes());
            crate::tiles::tests::view_covered(
                world.surfaces().drawn(),
                cam,
                rig.forward,
                BELOW_SURFACE,
            );
        }
        let end = world.total_evicted();
        let reached = peak >= budget;
        println!(
            "pan back: detail {factors:?}, peak held {peak} of {budget} (reached the budget: {reached}), compacted {}, tiles dropped {}, geometry builds {built}, drawn before and built again on the way back {}",
            end.compacted - start.compacted,
            end.tiles - start.tiles,
            again.len()
        );
        if !reached {
            assert!(
                again.is_empty(),
                "under the budget, tiles drawn before the pan were built again: {again:?}"
            );
        }
    }

    /// `rig` turned `angle` radians about the local up at its place.
    fn rig_turned(rig: &SurfaceRig, angle: f64) -> (Camera, [f64; 3]) {
        let c = rig.cam;
        let l = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
        let k: [f64; 3] = std::array::from_fn(|n| c[n] / l);
        let v = rig.forward;
        let kv = k[0] * v[0] + k[1] * v[1] + k[2] * v[2];
        let kxv = [
            k[1] * v[2] - k[2] * v[1],
            k[2] * v[0] - k[0] * v[2],
            k[0] * v[1] - k[1] * v[0],
        ];
        let (sn, cs) = angle.sin_cos();
        let f: [f64; 3] = std::array::from_fn(|n| v[n] * cs + kxv[n] * sn + k[n] * kv * (1.0 - cs));
        let mut camera = rig.camera;
        camera.orientation =
            crate::camera::look_rotation(Vec3::new(f[0], f[1], f[2]), Vec3::new(k[0], k[1], k[2]));
        (camera, f)
    }

    /// The level drawn at each point of a grid over the middle of the
    /// view from `cam` along `fwd` (the grid of
    /// [`crate::tiles::tests::view_covered`]), by face direction; each
    /// point is drawn by exactly one tile or quarter.
    fn levels_in_view(
        drawn: &[crate::tiles::TileDrawInfo],
        cam: [f64; 3],
        fwd: [f64; 3],
    ) -> Vec<([f64; 3], u8)> {
        let norm = |a: [f64; 3]| {
            let l = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
            [a[0] / l, a[1] / l, a[2] / l]
        };
        let cross = |a: [f64; 3], b: [f64; 3]| {
            [
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ]
        };
        let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        let u = norm(cross(fwd, norm(cam)));
        let v = cross(u, fwd);
        let n = 16;
        let mut out = Vec::new();
        for i in 0..=n {
            for j in 0..=n {
                let (a, b) = (
                    0.4 * (2.0 * f64::from(i) / f64::from(n) - 1.0),
                    0.3 * (2.0 * f64::from(j) / f64::from(n) - 1.0),
                );
                let d = norm(std::array::from_fn(|k| fwd[k] + a * u[k] + b * v[k]));
                let (bh, c) = (dot(cam, d), dot(cam, cam) - BELOW_SURFACE * BELOW_SURFACE);
                let disc = bh * bh - c;
                if disc < 0.0 || -bh - disc.sqrt() <= 0.0 {
                    continue;
                }
                let t = -bh - disc.sqrt();
                let p = norm(std::array::from_fn(|k| cam[k] + t * d[k]));
                let at = crate::tiles::tests::drawn_at(drawn, p);
                assert_eq!(
                    at.len(),
                    1,
                    "the view at ({a}, {b}) is drawn {} times: {:?}",
                    at.len(),
                    at.iter()
                        .map(|&n| (drawn[n].geometry.key, drawn[n].indices.clone()))
                        .collect::<Vec<_>>()
                );
                out.push((p, drawn[at[0]].geometry.key.level));
            }
        }
        out
    }

    /// The turn of `world` settled at `rig`: 4 s at 90 degrees a second,
    /// 60 frames a second, every request answered and every build done
    /// each frame. Every frame covers the view once; returns the largest
    /// step coarser at a point between consecutive frames, the peak held
    /// bytes, and the drawn quarters seen.
    fn turn(world: &mut World, rig: &SurfaceRig, now: &mut f64) -> (u8, u64, usize) {
        let mut before: Vec<crate::tiles::TileDrawInfo> = world.surfaces().drawn().to_vec();
        let (mut jump, mut peak, mut quarters) = (0u8, 0u64, 0usize);
        for i in 1..=240 {
            *now += 1.0 / 60.0;
            let angle = std::f64::consts::FRAC_PI_2 * f64::from(i) / 60.0;
            let (camera, fwd) = rig_turned(rig, angle);
            surface_frame_keys(world, rig, &camera, *now);
            let drawn = world.surfaces().drawn().to_vec();
            quarters += drawn.iter().filter(|t| t.is_part()).count();
            for (p, level) in levels_in_view(&drawn, rig.cam, fwd) {
                let was = crate::tiles::tests::drawn_at(&before, p);
                if let Some(&w) = was.first() {
                    let w = before[w].geometry.key.level;
                    jump = jump.max(w.saturating_sub(level));
                }
            }
            peak = peak.max(world.held_bytes());
            before = drawn;
        }
        (jump, peak, quarters)
    }

    #[test]
    fn a_turn_covers_the_view_and_falls_back_one_level() {
        let rig = surface_rig();
        // The default budget.
        let mut world = World::new(0);
        let mut now = 0.0;
        settle_surfaces(&mut world, &rig, &mut now);
        let budget = world.memory_budget();
        let (jump, peak, quarters) = turn(&mut world, &rig, &mut now);
        println!(
            "turn at the default budget: largest jump {jump} levels, peak held {peak} of {budget}, drawn quarters over the turn {quarters}"
        );
        assert!(peak <= budget, "{peak} bytes over {budget}");
        assert!(jump <= 1, "a point fell back {jump} levels");
        // The M7 over-budget setting: 70 percent of what the view needs.
        let (mut world, budget, mut now) = settled_at_seventy_percent(&rig);
        let (jump, peak, quarters) = turn(&mut world, &rig, &mut now);
        println!(
            "turn at 70 percent: largest jump {jump} levels, peak held {peak} of {budget}, drawn quarters over the turn {quarters}, detail {}",
            world.detail_factor()
        );
        assert!(peak <= budget, "{peak} bytes over {budget}");
        assert!(jump <= 1, "a point fell back {jump} levels");
    }
}

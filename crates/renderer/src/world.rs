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
//! its depth, and the children of those); a change of the selection marks
//! the cells near each cell whose selected status changed; a change of a
//! frame's isovalue marks the frame's cells. Only marked cells are looked
//! at, and only for what is left of the step's time slice once the draw
//! list is built ([`STEP_SLICE_SECONDS`]): what does not fit waits for the
//! next frame. A
//! few cells per step are also checked in turn ([`AUDIT_PER_STEP`]), so a
//! change no mark reached is still picked up.

use crate::camera::Camera;
use crate::clock::{now_seconds, StepTimes};
use crate::extract::{ExtractPool, FaceNeighbor, FaceNeighbors, SurfaceMesh};
use crate::farfield::{depth_zero_key, far_frames, frame_sprite, Sprite};
use crate::light::{lights_for, merge_frame_emitters, strongest_lights, PointLight};
use crate::seen::Seen;
use crate::selector::{SelectJob, Selected, SelectionLayout, Selector};
use crate::stream::{
    face_neighbor_keys, offset_key, CacheCounts, CellCache, CellMap, CellSet, CellState,
    FetchOutcome, NeighborMatter, AROUND, SELECT_INTERVAL_SECONDS,
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
}

impl DrawList {
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

/// A held mesh and the face neighbors it was extracted with.
struct Extracted {
    mesh: Arc<SurfaceMesh>,
    neighbors: FaceNeighbors,
}

/// Where the inputs of extraction may have changed, to be turned into
/// marked cells within the time slice.
#[derive(Copy, Clone, Debug, PartialEq)]
enum Mark {
    /// A cell arrived or left the cache.
    Arrival(CellKey),
    /// A cell's selected or above-selected status changed.
    Layout(CellKey),
    /// A frame's isovalue changed.
    Frame(u64),
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

/// The cell cache, the extraction pool, and the extracted meshes.
///
/// A cell's mesh depends on its `Ready` and known empty neighbors, on the
/// selection around it, and on its frame's isovalue (the padding and
/// transition rules of [`crate::extract`]). Whenever those change the cell
/// is extracted again, and its previous mesh is drawn until the new one is
/// done.
pub struct World {
    cache: CellCache,
    pool: ExtractPool,
    meshes: CellMap<Extracted>,
    extracting: CellMap<FaceNeighbors>,
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
    last_evict: f64,
}

impl World {
    /// A world extracting on `threads` worker threads, and selecting on
    /// its own thread ([`crate::selector`]); with zero threads, both
    /// inline.
    pub fn new(threads: usize) -> World {
        World {
            cache: CellCache::new(),
            pool: ExtractPool::new(threads),
            meshes: CellMap::default(),
            extracting: CellMap::default(),
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
            last_evict: f64::NEG_INFINITY,
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

    /// The neighbors the mesh of a cell was extracted with, if it has one.
    pub fn mesh_neighbors(&self, key: &CellKey) -> Option<&FaceNeighbors> {
        self.meshes.get(key).map(|e| &e.neighbors)
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

    /// The `Ready` and known empty face neighbors of a cell now, as
    /// extraction takes them.
    fn neighbors_of(&self, key: &CellKey, isovalue: Option<f64>) -> FaceNeighbors {
        let layout = &self.layout;
        let across = face_neighbor_keys(key);
        let mut faces: [Option<FaceNeighbor>; 6] =
            self.cache.face_neighbors(key).map(|n| match n? {
                (_, NeighborMatter::Ready(cell)) => Some(FaceNeighbor::Ready(cell.section.clone())),
                (k, NeighborMatter::Empty) => Some(FaceNeighbor::Empty(k)),
            });
        for f in 0..6 {
            let Some(k) = across[f] else {
                continue;
            };
            if layout.selected.contains(&k) {
                continue;
            }
            // Finer cells are selected across this face: pad from what
            // is loaded there.
            if layout.above_selected.contains(&k) {
                continue;
            }
            // A coarser cell is selected across this face: pad from it,
            // so the surface meets what is drawn there.
            let mut up = k.parent();
            while let Some(a) = up {
                if layout.selected.contains(&a) {
                    if let Some(CellState::Ready(cell)) = self.cache.state(&a) {
                        faces[f] = Some(FaceNeighbor::Ready(cell.section.clone()));
                    }
                    break;
                }
                up = a.parent();
            }
        }
        let parent = key.parent().and_then(|p| match self.cache.state(&p) {
            Some(CellState::Ready(cell)) => Some(cell.section.clone()),
            _ => None,
        });
        // The loaded cells at the parent's depth around it, and which of
        // them hold no matter.
        let mut parent_neighbors = Vec::new();
        let mut parent_empty = Vec::new();
        if let (Some(p), Some(_)) = (key.parent(), &parent) {
            let order = crate::stream::diagonal_offsets().into_iter().chain([
                [-1, 0, 0],
                [1, 0, 0],
                [0, -1, 0],
                [0, 1, 0],
                [0, 0, -1],
                [0, 0, 1],
            ]);
            for o in order {
                let Some(k) = offset_key(&p, o) else {
                    continue;
                };
                match self.cache.state(&k) {
                    Some(CellState::Ready(cell)) => {
                        parent_neighbors.push(cell.section.clone());
                        if !cell.has_matter && self.cache.known_empty(&k) {
                            parent_empty.push(o);
                        }
                    }
                    _ => {
                        if self.cache.known_empty(&k) {
                            parent_empty.push(o);
                        }
                    }
                }
            }
        }
        // The cells at this depth up to two over in every direction whose
        // region a coarser cell one depth up draws: the transition toward
        // them (extract) reaches a cell's padding, which lies in the cells
        // next to it, so it must know theirs too.
        let mut coarser: Vec<[i8; 3]> = Vec::new();
        for dz in -2i8..=2 {
            for dy in -2i8..=2 {
                for dx in -2i8..=2 {
                    if dx == 0 && dy == 0 && dz == 0 {
                        continue;
                    }
                    let k = (|| {
                        CellKey::new(
                            key.frame_id,
                            key.depth,
                            key.x.checked_add_signed(i32::from(dx))?,
                            key.y.checked_add_signed(i32::from(dy))?,
                            key.z.checked_add_signed(i32::from(dz))?,
                        )
                        .ok()
                    })();
                    let drawn_coarser = k.is_some_and(|k| {
                        !layout.selected.contains(&k)
                            && !layout.above_selected.contains(&k)
                            && k.parent().is_some_and(|p| layout.selected.contains(&p))
                    });
                    if drawn_coarser {
                        coarser.push([dx, dy, dz]);
                    }
                }
            }
        }
        FaceNeighbors {
            faces,
            parent,
            parent_neighbors,
            parent_empty,
            coarser,
            isovalue,
            diagonal: self
                .cache
                .diagonal_neighbors(key)
                .into_iter()
                .map(|n| match n? {
                    (_, NeighborMatter::Ready(cell)) => {
                        Some(FaceNeighbor::Ready(cell.section.clone()))
                    }
                    (k, NeighborMatter::Empty) => Some(FaceNeighbor::Empty(k)),
                })
                .collect(),
        }
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
        if !due || self.selector.busy() {
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
        };
        let done = match self.selector.start(job) {
            Some(done) => Some(done),
            None if force => self.selector.wait(),
            None => None,
        };
        if let Some(done) = done {
            self.apply_selection(done);
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

    /// The cells to request now (see [`CellCache::take_requests`]).
    pub fn take_requests(&mut self, now: f64) -> Vec<CellKey> {
        self.cache.take_requests(now)
    }

    /// Applies a fetch result: the cell's state, and marks for the cells
    /// whose extraction inputs it can change.
    pub fn complete(&mut self, key: CellKey, outcome: FetchOutcome, now: f64) {
        let before = self.cache.frame_isovalue(key.frame_id);
        self.cache.complete(key, outcome, now);
        self.changes.push((key, self.cache.state(&key).cloned()));
        self.arrived_since_selection = true;
        self.marks.push_back(Mark::Arrival(key));
        if self.cache.frame_isovalue(key.frame_id).map(f64::to_bits) != before.map(f64::to_bits) {
            self.marks.push_back(Mark::Frame(key.frame_id));
        }
    }

    /// One step of the frame thread's matter work (module docs): applies
    /// `results`, selects if due, takes the next requests, evicts every
    /// [`EVICT_INTERVAL_SECONDS`], collects finished meshes, builds the
    /// draw list for `seen`, and then gathers extraction inputs until
    /// [`STEP_SLICE_SECONDS`] after `started` (seconds of
    /// [`crate::clock::now_seconds`]). Returns the cells to request and the
    /// draw list.
    pub fn step(
        &mut self,
        seen: &Seen,
        view_px: (u32, u32),
        now: f64,
        results: impl IntoIterator<Item = (CellKey, FetchOutcome)>,
        started: f64,
    ) -> (Vec<CellKey>, DrawList) {
        for (key, outcome) in results {
            self.complete(key, outcome, now);
        }
        self.select(seen.system(), seen.camera(), view_px, now, false);
        let requests = self.take_requests(now);
        if now - self.last_evict >= EVICT_INTERVAL_SECONDS {
            self.last_evict = now;
            self.evict(now);
        }
        for (key, mesh) in self.pool.try_results() {
            self.accept(key, mesh);
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
        for (key, mesh) in self.pool.try_results() {
            self.accept(key, mesh);
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
                // Faces, edges, and corners read the cells around at the
                // same depth; a cell's parent level reads its parent and
                // the cells around the parent.
                self.mark(a);
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
                // Cells up to two over at its depth, and at the depth below
                // up to two over from its children.
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
            }
            Mark::Frame(f) => {
                let keys: Vec<CellKey> = self
                    .cache
                    .entries()
                    .filter(|(k, e)| k.frame_id == f && matches!(e.state, CellState::Ready(_)))
                    .map(|(k, _)| *k)
                    .collect();
                for k in keys {
                    self.mark(k);
                }
            }
        }
    }

    /// Queues `k` for extraction if it is `Ready` and its inputs differ
    /// from those of its mesh.
    fn look_at(&mut self, k: CellKey) {
        let Some(CellState::Ready(cell)) = self.cache.state(&k) else {
            return;
        };
        if self.extracting.contains_key(&k) {
            self.deferred.insert(k);
            return;
        }
        let section = cell.section.clone();
        let neighbors = self.neighbors_of(&k, self.cache.frame_isovalue(k.frame_id));
        if let Some(held) = self.meshes.get(&k) {
            if held.neighbors.same_as(&neighbors) {
                return;
            }
        }
        self.extracting.insert(k, neighbors.clone());
        self.pool.submit(k, section, neighbors);
    }

    fn accept(&mut self, key: CellKey, mesh: Arc<SurfaceMesh>) {
        let Some(neighbors) = self.extracting.remove(&key) else {
            return;
        };
        if matches!(self.cache.state(&key), Some(CellState::Ready(_))) {
            self.meshes.insert(key, Extracted { mesh, neighbors });
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
                    Some((key, mesh)) => self.accept(key, mesh),
                    None => break,
                }
            }
            self.pump();
        }
    }

    /// Evicts old cells and their meshes.
    pub fn evict(&mut self, now: f64) {
        for k in self.cache.evict(now) {
            self.changes.push((k, None));
            self.meshes.remove(&k);
            self.marks.push_back(Mark::Arrival(k));
        }
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
    /// and pinned cell is drawable or `Gone`, and nothing is being
    /// extracted.
    pub fn settled(&self) -> bool {
        let resolved =
            |k: &CellKey| self.drawable(k) || self.cache.state(k) == Some(&CellState::Gone);
        self.extraction_idle()
            && self.cache.outstanding() == 0
            && self.cache.selection().iter().all(|s| resolved(&s.key))
            && self.cache.pinned().all(resolved)
    }

    /// Returns `true` when every selected cell is drawn at its selected
    /// depth: the drawn selection is the target, every selected and every
    /// pinned cell can be drawn, and nothing is being extracted. Stricter
    /// than [`World::settled`]: a `Gone` cell is not ready.
    pub fn ready(&self) -> bool {
        self.extraction_idle()
            && self.cache.outstanding() == 0
            && self.cache.at_target()
            && self.cache.selection().iter().all(|s| self.drawable(&s.key))
            && self.cache.pinned().all(|k| self.drawable(k))
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
        let far = far_frames(system, camera, view_height);
        let cell_weights: BTreeMap<u64, f32> = far
            .iter()
            .map(|f| (f.frame_id, f.cell_weight() as f32))
            .collect();
        let weight = |frame: u64| cell_weights.get(&frame).copied().unwrap_or(1.0);
        let keys = self.cache.draw_set(|k| self.drawable(k));
        self.cache.touch(&keys, now);
        let mut meshes = Vec::new();
        let mut volumes = Vec::new();
        for k in keys.iter().filter(|k| weight(k.frame_id) > 0.0) {
            if let Some(CellState::Ready(cell)) = self.cache.state(k) {
                if let Some(v) = &cell.volume {
                    volumes.push(v.clone());
                }
            }
            if let Some(e) = self.meshes.get(k) {
                if !e.mesh.is_empty() {
                    meshes.push(e.mesh.clone());
                }
            }
        }
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
            volumes,
            lights,
            frame_lights,
            sprites,
            cell_weights,
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

    /// Cell `(x, 0, 0)` at depth 2 of a frame with a 16 m root, at
    /// resolution 4, solid where `fill(k)` holds for the sample layer `k`
    /// along z.
    fn cell(x: u32, fill: impl Fn(u32) -> bool) -> (CellKey, FetchOutcome) {
        let key = CellKey::new(3, 2, x, 0, 0).unwrap();
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
        let section = Section::new(
            key,
            Vec3::new(-8.0 + 4.0 * f64::from(x), -8.0, -8.0),
            Meters::new(4.0),
            4,
            Samples::from_fn(4, |_, _, k| if fill(k) { solid } else { Sample::VACUUM }),
        )
        .unwrap();
        let ready = ReadyCell::new(section);
        (key, FetchOutcome::Decoded(Some(Arc::new(ready))))
    }

    #[test]
    fn a_cell_is_extracted_again_when_a_neighbor_arrives() {
        let mut world = World::new(0);
        let (a, fetched) = cell(1, |_| true);
        world.complete(a, fetched, 0.0);
        world.finish_extraction();
        // Filled and clamped on every face: nothing to draw.
        assert!(world.mesh(&a).unwrap().is_empty());
        assert_eq!(world.mesh_neighbors(&a).unwrap().count(), 0);

        // The +x neighbor holds matter in its lower half only, so the upper
        // half of the shared face is now a real surface of `a`.
        let (b, fetched) = cell(2, |k| k < 2);
        world.complete(b, fetched, 0.0);
        world.finish_extraction();
        let neighbors = world.mesh_neighbors(&a).unwrap();
        assert_eq!(neighbors.count(), 1);
        assert!(neighbors.faces[1].is_some());
        let mesh = world.mesh(&a).unwrap().clone();
        assert!(!mesh.is_empty());
        // The wall lies on the face, x = 0, and the top of the neighbor's
        // matter reaches half a sample (0.5 m) into the straddling cubes.
        let surface = &mesh.positions[..mesh.skirt_from];
        assert!(surface
            .iter()
            .all(|p| (0.0..=0.5).contains(&p[0]) && p[2] >= -6.0));
        assert!(surface.iter().any(|p| p[0] == 0.0 && p[2] > -6.0));
        assert_eq!(world.mesh_neighbors(&b).unwrap().count(), 1);

        // Nothing changed: nothing is extracted again.
        world.pump();
        assert_eq!(world.counts().extracting, 0);
        assert!(Arc::ptr_eq(world.mesh(&a).unwrap(), &mesh));

        // The -x neighbor turns out to hold no matter: its vacuum is data,
        // so `a` is extracted again with a wall on that face, x = -4.
        let c = CellKey::new(3, 2, 0, 0, 0).unwrap();
        world.complete(c, FetchOutcome::Decoded(None), 0.0);
        world.finish_extraction();
        assert_eq!(world.mesh_neighbors(&a).unwrap().count(), 2);
        let walled = world.mesh(&a).unwrap();
        assert!(walled.positions.iter().any(|p| p[0] == -4.0));
        assert!(walled
            .vertices
            .iter()
            .zip(&walled.positions)
            .filter(|(_, p)| p[0] == -4.0 && p[2] > -6.0)
            .all(|(v, _)| v.normal[0] < -0.999));
    }
}

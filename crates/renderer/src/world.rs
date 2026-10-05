//! The matter pipeline from selection to drawable meshes and lights.
//!
//! Ties together the steps of `space-model.md` section 2 that turn chunks
//! into something to draw: cell selection and the cell cache
//! ([`crate::stream`]), surface extraction on a worker pool
//! ([`crate::extract`]), volumes of gas and plasma ([`crate::volume`]),
//! lights from hot matter ([`crate::light`]), and point sprites for far
//! frames ([`crate::farfield`]). It
//! does no I/O itself: the caller hands it fetch results (from the hub
//! through [`crate::stream::Fetcher`] on the desktop, or straight from bytes
//! in tests) and asks it what to request next. One [`World`] serves the
//! window loop, the headless run, and the tests the same way.

use crate::camera::Camera;
use crate::extract::{ExtractPool, SurfaceMesh};
use crate::farfield::{depth_zero_key, far_frames, frame_sprite, Sprite};
use crate::light::{merge_frame_emitters, strongest_lights, PointLight};
use crate::stream::{
    select_all, CacheCounts, CellCache, CellState, FetchOutcome, SELECT_INTERVAL_SECONDS,
};
use crate::volume::VolumeGrid;
use gx_core::frames::FrameSystem;
use gx_core::key::CellKey;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

/// What to draw this frame.
#[derive(Clone, Debug, PartialEq)]
pub struct DrawList {
    /// Non-empty meshes of the draw set, in key order.
    pub meshes: Vec<Arc<SurfaceMesh>>,
    /// Volumes of the draw set, in key order.
    pub volumes: Vec<Arc<VolumeGrid>>,
    /// The strongest lights, strongest first.
    pub lights: Vec<PointLight>,
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

/// The cell cache, the extraction pool, and the extracted meshes.
pub struct World {
    cache: CellCache,
    pool: ExtractPool,
    meshes: BTreeMap<CellKey, Arc<SurfaceMesh>>,
    extracting: BTreeSet<CellKey>,
    last_selection: Option<f64>,
}

impl World {
    /// A world extracting on `threads` worker threads (zero: inline).
    pub fn new(threads: usize) -> World {
        World {
            cache: CellCache::new(),
            pool: ExtractPool::new(threads),
            meshes: BTreeMap::new(),
            extracting: BTreeSet::new(),
            last_selection: None,
        }
    }

    /// The cell cache.
    pub fn cache(&self) -> &CellCache {
        &self.cache
    }

    /// The mesh extracted for a cell, if any.
    pub fn mesh(&self, key: &CellKey) -> Option<&Arc<SurfaceMesh>> {
        self.meshes.get(key)
    }

    /// Selects cells for the camera if [`SELECT_INTERVAL_SECONDS`] have
    /// passed since the last selection (or `force`), and pins the depth-0
    /// cell of every frame in the far field ([`crate::farfield`]). Returns
    /// `true` if it selected.
    pub fn select(
        &mut self,
        system: &FrameSystem,
        camera: &Camera,
        view_px: (u32, u32),
        now: f64,
        force: bool,
    ) -> bool {
        let due = self
            .last_selection
            .is_none_or(|t| now - t >= SELECT_INTERVAL_SECONDS);
        if !(due || force) {
            return false;
        }
        self.last_selection = Some(now);
        let cells = select_all(
            system,
            camera,
            f64::from(view_px.0.max(1)),
            f64::from(view_px.1.max(1)),
        );
        self.cache.set_selection(cells, now);
        for far in far_frames(system, camera, f64::from(view_px.1.max(1))) {
            self.cache.pin(depth_zero_key(far.frame_id), now);
        }
        true
    }

    /// The cells to request now (see [`CellCache::take_requests`]).
    pub fn take_requests(&mut self, now: f64) -> Vec<CellKey> {
        self.cache.take_requests(now)
    }

    /// Applies a fetch result.
    pub fn complete(&mut self, key: CellKey, outcome: FetchOutcome, now: f64) {
        self.cache.complete(key, outcome, now);
    }

    /// Applies a `chunkReady` push.
    pub fn chunk_ready(&mut self, key: &CellKey) {
        self.cache.notify_ready(key);
    }

    /// Collects finished meshes and queues extraction for every `Ready`
    /// cell without one, in key order.
    pub fn pump(&mut self) {
        for (key, mesh) in self.pool.try_results() {
            self.accept(key, mesh);
        }
        let todo: Vec<(CellKey, Arc<gx_core::matter::Section>)> = self
            .cache
            .entries()
            .filter_map(|(k, e)| match &e.state {
                CellState::Ready(cell)
                    if !self.meshes.contains_key(k) && !self.extracting.contains(k) =>
                {
                    Some((*k, cell.section.clone()))
                }
                _ => None,
            })
            .collect();
        for (key, section) in todo {
            self.extracting.insert(key);
            self.pool.submit(key, section);
        }
    }

    fn accept(&mut self, key: CellKey, mesh: Arc<SurfaceMesh>) {
        self.extracting.remove(&key);
        if matches!(self.cache.state(&key), Some(CellState::Ready(_))) {
            self.meshes.insert(key, mesh);
        }
    }

    /// Pumps until no extraction is queued or running.
    pub fn finish_extraction(&mut self) {
        self.pump();
        while !self.extracting.is_empty() {
            match self.pool.wait_result() {
                Some((key, mesh)) => self.accept(key, mesh),
                None => break,
            }
            self.pump();
        }
    }

    /// Evicts old cells and their meshes.
    pub fn evict(&mut self, now: f64) {
        for k in self.cache.evict(now) {
            self.meshes.remove(&k);
        }
    }

    /// Whether a cell can be drawn now: `Empty`, or `Ready` with its mesh.
    fn drawable(&self, key: &CellKey) -> bool {
        match self.cache.state(key) {
            Some(CellState::Empty) => true,
            Some(CellState::Ready(_)) => self.meshes.contains_key(key),
            _ => false,
        }
    }

    /// Returns `true` when every selected and every pinned cell is
    /// resolved: drawable, or `Gone`, with nothing in flight or being
    /// extracted.
    pub fn settled(&self) -> bool {
        let resolved =
            |k: &CellKey| self.drawable(k) || self.cache.state(k) == Some(&CellState::Gone);
        self.extracting.is_empty()
            && self.cache.selection().iter().all(|s| resolved(&s.key))
            && self.cache.pinned().all(resolved)
    }

    /// The draw set (see [`CellCache::draw_set`]), its meshes and volumes,
    /// the lights its hot matter makes, and the sprites of far frames, for
    /// a view of `view_px` pixels. Marks the drawn cells as used.
    ///
    /// Lights come from the hot matter of the drawn cells and, for frames
    /// drawn as a sprite only, of their depth-0 cells, so a far hot frame
    /// still lights the rest. Cold far frames reflect those lights.
    pub fn draw_list(
        &mut self,
        system: &FrameSystem,
        camera: &Camera,
        view_px: (u32, u32),
        now: f64,
    ) -> DrawList {
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
        let mut emitters = Vec::new();
        for k in keys.iter().filter(|k| weight(k.frame_id) > 0.0) {
            if let Some(CellState::Ready(cell)) = self.cache.state(k) {
                if let Some(e) = &cell.emitter {
                    emitters.push((k.frame_id, e));
                }
                if let Some(v) = &cell.volume {
                    volumes.push(v.clone());
                }
            }
            if let Some(m) = self.meshes.get(k) {
                if !m.is_empty() {
                    meshes.push(m.clone());
                }
            }
        }
        for f in far.iter().filter(|f| f.sprite_only()) {
            if let Some(CellState::Ready(cell)) = self.cache.state(&depth_zero_key(f.frame_id)) {
                if let Some(e) = &cell.emitter {
                    emitters.push((f.frame_id, e));
                }
            }
        }
        let merged = merge_frame_emitters(emitters.iter().map(|(f, e)| (*f, *e)));
        let lights = strongest_lights(&merged, system, camera);
        let sprites = far
            .iter()
            .filter_map(|f| match self.cache.state(&depth_zero_key(f.frame_id)) {
                Some(CellState::Ready(cell)) => {
                    frame_sprite(system, camera, f, &cell.far_field(), &lights, view_height)
                }
                _ => None,
            })
            .collect();
        DrawList {
            meshes,
            volumes,
            lights,
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

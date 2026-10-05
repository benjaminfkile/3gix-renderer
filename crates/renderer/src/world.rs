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
use crate::extract::{ExtractPool, FaceNeighbor, FaceNeighbors, SurfaceMesh};
use crate::farfield::{depth_zero_key, far_frames, frame_sprite, Sprite};
use crate::light::{merge_frame_emitters, strongest_lights, PointLight};
use crate::stream::{
    select_all, CacheCounts, CellCache, CellState, FetchOutcome, NeighborMatter,
    SELECT_INTERVAL_SECONDS,
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

/// A held mesh and the face neighbors it was extracted with.
struct Extracted {
    mesh: Arc<SurfaceMesh>,
    neighbors: FaceNeighbors,
}

/// The cell cache, the extraction pool, and the extracted meshes.
///
/// A cell's mesh depends on its `Ready` and known empty face neighbors at
/// the same depth (the padding rule of [`crate::extract`]). Whenever that set
/// changes, as a neighbor arrives or is evicted, the cell is extracted again, and its
/// previous mesh is drawn until the new one is done, so seams close as
/// cells arrive.
pub struct World {
    cache: CellCache,
    pool: ExtractPool,
    meshes: BTreeMap<CellKey, Extracted>,
    extracting: BTreeMap<CellKey, FaceNeighbors>,
    last_selection: Option<f64>,
}

impl World {
    /// A world extracting on `threads` worker threads (zero: inline).
    pub fn new(threads: usize) -> World {
        World {
            cache: CellCache::new(),
            pool: ExtractPool::new(threads),
            meshes: BTreeMap::new(),
            extracting: BTreeMap::new(),
            last_selection: None,
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

    /// The `Ready` and known empty face neighbors of a cell now, as
    /// extraction takes them.
    fn neighbors_of(&self, key: &CellKey) -> FaceNeighbors {
        FaceNeighbors {
            faces: self.cache.face_neighbors(key).map(|n| match n? {
                (_, NeighborMatter::Ready(cell)) => Some(FaceNeighbor::Ready(cell.section.clone())),
                (k, NeighborMatter::Empty) => Some(FaceNeighbor::Empty(k)),
            }),
        }
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

    /// Collects finished meshes and queues extraction, in key order, for
    /// every `Ready` cell without a mesh or whose mesh was extracted with
    /// other face neighbors than it has now. A cell already being extracted
    /// waits for that result first.
    pub fn pump(&mut self) {
        for (key, mesh) in self.pool.try_results() {
            self.accept(key, mesh);
        }
        let todo: Vec<(CellKey, Arc<gx_core::matter::Section>, FaceNeighbors)> = self
            .cache
            .entries()
            .filter_map(|(k, e)| match &e.state {
                CellState::Ready(cell) if !self.extracting.contains_key(k) => {
                    let neighbors = self.neighbors_of(k);
                    match self.meshes.get(k) {
                        Some(held) if held.neighbors.same_as(&neighbors) => None,
                        _ => Some((*k, cell.section.clone(), neighbors)),
                    }
                }
                _ => None,
            })
            .collect();
        for (key, section, neighbors) in todo {
            self.extracting.insert(key, neighbors.clone());
            self.pool.submit(key, section, neighbors);
        }
    }

    fn accept(&mut self, key: CellKey, mesh: Arc<SurfaceMesh>) {
        let Some(neighbors) = self.extracting.remove(&key) else {
            return;
        };
        if matches!(self.cache.state(&key), Some(CellState::Ready(_))) {
            self.meshes.insert(key, Extracted { mesh, neighbors });
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

    /// Returns `true` when every selected and every pinned cell can be
    /// drawn (`Ready` with its mesh, or `Empty`) and nothing is being
    /// extracted. Stricter than [`World::settled`]: a `Gone` cell is not
    /// ready.
    pub fn ready(&self) -> bool {
        self.extracting.is_empty()
            && self.cache.selection().iter().all(|s| self.drawable(&s.key))
            && self.cache.pinned().all(|k| self.drawable(k))
    }

    /// The draw set (see [`CellCache::draw_set`]), its meshes and volumes,
    /// the lights its hot matter makes, and the sprites of far frames, for
    /// a view of `view_px` pixels. Marks the drawn cells as used.
    ///
    /// Lights come from the hot matter of the drawn cells and, for far
    /// frames none of whose drawn cells is hot (frames drawn as a sprite
    /// only, and frames in the transition whose cells are beyond the
    /// selection range), of their depth-0 cells, so a far hot frame still
    /// lights the rest. Cold far frames reflect those lights.
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
            if let Some(e) = self.meshes.get(k) {
                if !e.mesh.is_empty() {
                    meshes.push(e.mesh.clone());
                }
            }
        }
        // A far frame whose drawn cells carry no hot matter (a sprite only,
        // or in the transition with its cells out of the selection range)
        // still lights the rest through its depth-0 cell.
        let lit: BTreeSet<u64> = emitters.iter().map(|(f, _)| *f).collect();
        for f in far.iter().filter(|f| !lit.contains(&f.frame_id)) {
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
        assert!(mesh
            .positions
            .iter()
            .all(|p| (0.0..=0.5).contains(&p[0]) && p[2] >= -6.0));
        assert!(mesh.positions.iter().any(|p| p[0] == 0.0 && p[2] > -6.0));
        assert_eq!(world.mesh_neighbors(&b).unwrap().count(), 1);

        // Nothing changed: nothing is extracted again.
        world.pump();
        assert_eq!(world.counts().extracting, 0);
        assert!(Arc::ptr_eq(world.mesh(&a).unwrap(), &mesh));

        // The -x neighbor turns out to hold no matter: its vacuum is data,
        // so `a` is extracted again with a wall on that face, x = -4.
        let c = CellKey::new(3, 2, 0, 0, 0).unwrap();
        world.complete(c, FetchOutcome::NotFound, 0.0);
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

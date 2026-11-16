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
use crate::light::{lights_for, merge_frame_emitters, strongest_lights, PointLight};
use crate::seen::Seen;
use crate::stream::{
    select_all, CacheCounts, CellCache, CellState, FetchOutcome, NeighborMatter,
    SELECT_INTERVAL_SECONDS,
};
use crate::volume::VolumeGrid;
use gx_core::frames::FrameSystem;
use gx_core::key::CellKey;
use gx_core::units::Vec3;
use std::collections::BTreeMap;
use std::sync::Arc;

/// What to draw this frame.
#[derive(Clone, Debug, PartialEq)]
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

/// The latest selection as extraction needs it: the selected cells, and
/// every cell with a selected cell strictly below it.
struct SelectionLayout {
    selected: std::collections::BTreeSet<CellKey>,
    above_selected: std::collections::BTreeSet<CellKey>,
}

impl SelectionLayout {
    fn of(selection: &[crate::stream::SelectedCell]) -> SelectionLayout {
        let selected: std::collections::BTreeSet<CellKey> =
            selection.iter().map(|c| c.key).collect();
        let mut above_selected = std::collections::BTreeSet::new();
        for k in &selected {
            let mut up = k.parent();
            while let Some(p) = up {
                if !above_selected.insert(p) {
                    break;
                }
                up = p.parent();
            }
        }
        SelectionLayout {
            selected,
            above_selected,
        }
    }
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
    fn neighbors_of(
        &self,
        key: &CellKey,
        isovalue: Option<f64>,
        layout: &SelectionLayout,
    ) -> FaceNeighbors {
        let across = crate::stream::face_neighbor_keys(key);
        let mut clip = [false; 6];
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
            // Finer cells are selected across this face: they draw that
            // side, so this cell's mesh stops at the face.
            if layout.above_selected.contains(&k) {
                clip[f] = true;
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
        FaceNeighbors {
            faces,
            clip,
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

    /// Selects cells for the camera if [`SELECT_INTERVAL_SECONDS`] have
    /// passed since the last selection (or `force`), and pins the depth-0
    /// cell of every frame with mass: the far field draws from it
    /// ([`crate::farfield`]) and every light comes from it
    /// ([`crate::light`]). Returns `true` if it selected.
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
        let cache = &self.cache;
        let cells = select_all(
            system,
            camera,
            f64::from(view_px.0.max(1)),
            f64::from(view_px.1.max(1)),
            |k| cache.refinable(&k),
        );
        self.cache.set_selection(cells, now);
        for f in system
            .tree()
            .frames()
            .iter()
            .filter(|f| f.mass.value() > 0.0)
        {
            self.cache.pin(depth_zero_key(f.frame_id), now);
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

    /// Collects finished meshes and queues extraction, in key order, for
    /// every `Ready` cell without a mesh or whose mesh was extracted with
    /// other face neighbors than it has now. A cell already being extracted
    /// waits for that result first.
    pub fn pump(&mut self) {
        for (key, mesh) in self.pool.try_results() {
            self.accept(key, mesh);
        }
        // One isovalue per frame, so neighboring cells draw the surface at
        // the same density ([`crate::stream::CellCache::frame_isovalue`]).
        let mut isovalues: std::collections::BTreeMap<u64, f64> = std::collections::BTreeMap::new();
        for (k, e) in self.cache.entries() {
            if let CellState::Ready(cell) = &e.state {
                if let Some(iso) = cell.inner_isovalue {
                    let v = isovalues.entry(k.frame_id).or_insert(iso);
                    *v = v.min(iso);
                }
            }
        }
        let layout = SelectionLayout::of(self.cache.selection());
        let todo: Vec<(CellKey, Arc<gx_core::matter::Section>, FaceNeighbors)> = self
            .cache
            .entries()
            .filter_map(|(k, e)| match &e.state {
                CellState::Ready(cell) if !self.extracting.contains_key(k) => {
                    let neighbors =
                        self.neighbors_of(k, isovalues.get(&k.frame_id).copied(), &layout);
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

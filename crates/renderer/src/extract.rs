//! Surface extraction of drawn cells with Transvoxel (`rendering.md`
//! sections 2 and 3.3): what one cell's extraction reads, the mesh it
//! gives, and the worker pool that runs it.
//!
//! A drawn cell is extracted with [`crate::transvoxel::extract_mixed`] over
//! a snapshot of what it reads ([`CellInputs`]): the drawn selection around
//! it (which cells are drawn, and at which depth) and the sections of the
//! cells one depth coarser, at its depth, and one depth finer in the region
//! of the cell grown by one cell of its depth on every side. Same-depth
//! neighbors give the corners on shared faces; a face whose neighbor is
//! drawn one depth finer gets transition cells; a face whose neighbor is
//! drawn one depth coarser is owned by this (finer) side, which reads the
//! coarser field resampled (`rendering.md` section 2.3). The surface is the
//! level set `occupancy = 0.5`; nothing else is assumed about the matter.
//!
//! Only solid and fluid samples form surfaces ([`forms_surface`]). Gas and
//! plasma count as occupancy 0 here and are drawn as a volume instead
//! ([`crate::volume`]).
//!
//! # Inputs present or known empty
//!
//! A cell is extracted only from inputs that are present or known empty.
//! Every read of a section the snapshot does not hold is recorded; an
//! extraction that made one gives no mesh ([`Extraction::Waiting`]) and the
//! cell is looked at again when something in its region arrives.
//!
//! # Attributes
//!
//! Positions and normals come from the Transvoxel module. Albedo,
//! roughness, temperature, density, and state are those of the eight
//! samples of the cell's depth around the lattice point nearest the
//! vertex, density-weighted over solid and fluid samples, with the state
//! of the densest one (`matter-format.md` section 3.5).
//!
//! # Determinism
//!
//! The snapshot is held in ordered maps, extraction is plain `f64`
//! arithmetic in a fixed order, and each mesh depends only on its inputs,
//! so which worker thread ran it never shows in the output.

use crate::transvoxel::{extract_mixed, CellAt, MixedField};
use gx_core::key::CellKey;
use gx_core::matter::{Section, State};
use gx_core::units::Vec3;
use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};

/// Returns `true` for the states that form surfaces: solid and fluid.
/// Everything else (vacuum, gas, plasma) counts as vacuum for extraction.
pub fn forms_surface(state: State) -> bool {
    matches!(state, State::Solid | State::Fluid)
}

/// Per-vertex attributes other than the position.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SurfaceVertex {
    /// Unit normal, pointing from dense to thin matter, in the frame axes.
    pub normal: [f32; 3],
    /// Reflectance in the three bands, long to short wavelength.
    pub albedo: [f32; 3],
    /// Microfacet roughness, 0 to 1.
    pub roughness: f32,
    /// Radiating temperature of the surface's outermost matter, kelvin:
    /// what the surface glows at (`matter-format.md` section 3.3).
    pub temperature: f32,
    /// State as its format integer (`matter-format.md` section 3.3).
    pub state: u32,
    /// Density, kilograms per cubic meter.
    pub density: f32,
}

/// The surface of one cell.
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceMesh {
    /// The cell the surface belongs to.
    pub key: CellKey,
    /// Minimum corner of the cell, meters, frame coordinates.
    pub origin: Vec3,
    /// Vertex positions, meters, frame coordinates.
    pub positions: Vec<[f64; 3]>,
    /// Vertex attributes, one per position.
    pub vertices: Vec<SurfaceVertex>,
    /// Triangles, three indices each, counterclockwise from outside.
    pub indices: Vec<u32>,
    /// Smallest corner of the box around every position, frame coordinates.
    pub bounds_min: [f64; 3],
    /// Largest corner of the box around every position, frame coordinates.
    pub bounds_max: [f64; 3],
}

impl SurfaceMesh {
    /// Bytes this mesh holds: the struct and the capacity of its
    /// positions, vertices, and indices (the memory report).
    pub fn heap_bytes(&self) -> u64 {
        (std::mem::size_of::<SurfaceMesh>()
            + self.positions.capacity() * std::mem::size_of::<[f64; 3]>()
            + self.vertices.capacity() * std::mem::size_of::<SurfaceVertex>()
            + self.indices.capacity() * std::mem::size_of::<u32>()) as u64
    }

    /// Number of triangles.
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// Returns `true` when there is nothing to draw.
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }
}

/// Everything the extraction of one drawn cell reads (module docs).
#[derive(Clone, Debug)]
pub struct CellInputs {
    /// The cell, with the section it holds.
    pub key: CellKey,
    /// The cell's own section.
    pub section: Arc<Section>,
    /// Drawn cells of the region, at any depth.
    pub drawn: BTreeSet<CellKey>,
    /// Sections of the region one depth coarser, at the cell's depth, and
    /// one depth finer: `Some` when present, `None` when known empty.
    pub sections: BTreeMap<CellKey, Option<Arc<Section>>>,
}

impl CellInputs {
    /// Returns `true` when `other` holds the same drawn cells and the same
    /// sections (the same shared section, not merely equal bytes).
    pub fn same_as(&self, other: &CellInputs) -> bool {
        self.key == other.key
            && Arc::ptr_eq(&self.section, &other.section)
            && self.drawn == other.drawn
            && self.sections.len() == other.sections.len()
            && self
                .sections
                .iter()
                .zip(&other.sections)
                .all(|((ka, a), (kb, b))| {
                    ka == kb
                        && match (a, b) {
                            (Some(a), Some(b)) => Arc::ptr_eq(a, b),
                            (None, None) => true,
                            _ => false,
                        }
                })
    }

    /// The cells whose sections or drawn status a cell's extraction may
    /// read: at depths `depth - 1 ..= depth + 1`, those overlapping the cell
    /// grown by one cell of its depth on every side. Ascending order.
    pub fn region(key: &CellKey) -> Vec<CellKey> {
        let mut out = Vec::new();
        let d = key.depth;
        let lo = |v: u32| i64::from(v) - 1;
        let hi = |v: u32| i64::from(v) + 1;
        for dd in d.saturating_sub(1)..=d.saturating_add(1) {
            let side = 1i64 << dd;
            // Cell range at depth dd covering [lo, hi + 1) at depth d.
            let range = |v: u32| -> (i64, i64) {
                if dd >= d {
                    let s = 1i64 << (dd - d);
                    (lo(v) * s, (hi(v) + 1) * s - 1)
                } else {
                    let s = 1i64 << (d - dd);
                    (lo(v).div_euclid(s), hi(v).div_euclid(s))
                }
            };
            let (x, y, z) = (range(key.x), range(key.y), range(key.z));
            for cz in z.0.max(0)..=z.1.min(side - 1) {
                for cy in y.0.max(0)..=y.1.min(side - 1) {
                    for cx in x.0.max(0)..=x.1.min(side - 1) {
                        if let Ok(k) =
                            CellKey::new(key.frame_id, dd, cx as u32, cy as u32, cz as u32)
                        {
                            out.push(k);
                        }
                    }
                }
            }
        }
        out.sort_unstable();
        out
    }

    /// Inputs for a section drawn alone: every other cell of the region
    /// known empty and not drawn.
    pub fn alone(section: Arc<Section>) -> CellInputs {
        let key = section.key();
        let sections = CellInputs::region(&key)
            .into_iter()
            .filter(|k| *k != key)
            .map(|k| (k, None))
            .collect();
        CellInputs {
            key,
            section,
            drawn: BTreeSet::from([key]),
            sections,
        }
    }
}

/// The result of one extraction.
#[derive(Clone, Debug)]
pub enum Extraction {
    /// The mesh, from inputs that were all present or known empty.
    Done(Arc<SurfaceMesh>),
    /// A read found an input that is neither present nor known empty.
    Waiting,
}

/// The [`MixedField`] over a snapshot.
struct Field<'a> {
    inputs: &'a CellInputs,
    n: usize,
    root_origin: [f64; 3],
    root_edge: f64,
    missing: Cell<bool>,
    /// The region's cells at depths `depth - 1 ..= depth + 1` as flat
    /// tables ([`Level`]), so a lookup is an index, not a key built and
    /// searched in the ordered maps for every sample.
    levels: Vec<Level<'a>>,
}

/// What the snapshot holds for one cell of the region.
#[derive(Copy, Clone)]
enum Slot<'a> {
    /// The cell's section.
    Present(&'a Section),
    /// Known empty.
    Empty,
    /// Neither present nor known empty.
    Missing,
}

/// The region's cells at one depth, as a dense box of cell indices.
struct Level<'a> {
    depth: u8,
    /// Index of the box's minimum cell (may lie outside the root cell).
    base: [i64; 3],
    /// Cells per axis.
    dims: [i64; 3],
    /// Drawn, per cell of the box.
    drawn: Vec<bool>,
    /// The snapshot's slot, per cell of the box.
    slots: Vec<Slot<'a>>,
}

impl Level<'_> {
    /// The flat index of cell `idx`, if it lies in the box.
    fn index(&self, idx: [i64; 3]) -> Option<usize> {
        let l = [0, 1, 2].map(|a| idx[a] - self.base[a]);
        if (0..3).any(|a| !(0..self.dims[a]).contains(&l[a])) {
            return None;
        }
        Some((l[0] + self.dims[0] * (l[1] + self.dims[1] * l[2])) as usize)
    }
}

impl Field<'_> {
    fn new(inputs: &CellInputs) -> Field<'_> {
        let s = &inputs.section;
        let k = s.key();
        let e = s.edge().value();
        let o = s.origin();
        let idx = [k.x, k.y, k.z].map(f64::from);
        Field {
            inputs,
            n: usize::from(s.resolution()),
            root_origin: [o.x - idx[0] * e, o.y - idx[1] * e, o.z - idx[2] * e],
            root_edge: e * (1u64 << k.depth) as f64,
            missing: Cell::new(false),
            levels: Self::levels(inputs),
        }
    }

    /// The flat tables of [`Level`]: the same cell ranges as
    /// [`CellInputs::region`], filled from the snapshot.
    fn levels(inputs: &CellInputs) -> Vec<Level<'_>> {
        let key = inputs.key;
        let d = key.depth;
        let mut out = Vec::new();
        for dd in d.saturating_sub(1)..=d.saturating_add(1) {
            let range = |v: u32| -> (i64, i64) {
                let (lo, hi) = (i64::from(v) - 1, i64::from(v) + 1);
                if dd >= d {
                    let f = 1i64 << (dd - d);
                    (lo * f, (hi + 1) * f - 1)
                } else {
                    let f = 1i64 << (d - dd);
                    (lo.div_euclid(f), hi.div_euclid(f))
                }
            };
            let r = [range(key.x), range(key.y), range(key.z)];
            let base = r.map(|(lo, _)| lo);
            let dims = r.map(|(lo, hi)| hi - lo + 1);
            let count = (dims[0] * dims[1] * dims[2]) as usize;
            let mut level = Level {
                depth: dd,
                base,
                dims,
                drawn: vec![false; count],
                slots: vec![Slot::Missing; count],
            };
            let side = 1i64 << dd;
            for z in 0..dims[2] {
                for y in 0..dims[1] {
                    for x in 0..dims[0] {
                        let idx = [base[0] + x, base[1] + y, base[2] + z];
                        if idx.iter().any(|&v| !(0..side).contains(&v)) {
                            continue;
                        }
                        let Ok(k) = CellKey::new(
                            key.frame_id,
                            dd,
                            idx[0] as u32,
                            idx[1] as u32,
                            idx[2] as u32,
                        ) else {
                            continue;
                        };
                        let i = level.index(idx).expect("inside the box");
                        level.drawn[i] = inputs.drawn.contains(&k);
                        level.slots[i] = if k == key {
                            Slot::Present(&inputs.section)
                        } else {
                            match inputs.sections.get(&k) {
                                Some(Some(sec)) => Slot::Present(sec),
                                Some(None) => Slot::Empty,
                                None => Slot::Missing,
                            }
                        };
                    }
                }
            }
            out.push(level);
        }
        out
    }

    /// The cell index at depth `dd` holding the center of sample `c` of
    /// depth `d`, or `None` outside the root cell (the arithmetic of
    /// [`Field::holder`]).
    fn holder_index(&self, d: u8, c: [i64; 3], dd: u8) -> Option<[i64; 3]> {
        let n = self.n as i64;
        let idx = if dd <= d {
            c.map(|v| v.div_euclid(n << (d - dd)))
        } else {
            c.map(|v| ((2 * v + 1) << (dd - d)).div_euclid(2 * n))
        };
        let side = 1i64 << dd;
        if idx.iter().any(|&v| !(0..side).contains(&v)) {
            return None;
        }
        Some(idx)
    }

    /// The table of depth `dd`, if the region has one.
    fn level(&self, dd: u8) -> Option<&Level<'_>> {
        self.levels.iter().find(|l| l.depth == dd)
    }

    /// The cell of depth `dd` holding the center of sample `c` of depth `d`.
    fn holder(&self, d: u8, c: [i64; 3], dd: u8) -> Option<CellKey> {
        let n = self.n as i64;
        let idx = if dd <= d {
            c.map(|v| v.div_euclid(n << (d - dd)))
        } else {
            c.map(|v| ((2 * v + 1) << (dd - d)).div_euclid(2 * n))
        };
        let side = 1i64 << dd;
        if idx.iter().any(|&v| !(0..side).contains(&v)) {
            return None;
        }
        CellKey::new(
            self.inputs.key.frame_id,
            dd,
            idx[0] as u32,
            idx[1] as u32,
            idx[2] as u32,
        )
        .ok()
    }

    /// Sample `c` of depth `d`, or `None` for vacuum; records a missing
    /// input when `strict`.
    fn sample(&self, d: u8, c: [i64; 3], strict: bool) -> Option<gx_core::matter::Sample> {
        // Fast path: the region's tables.
        if let Some(level) = self.level(d) {
            let idx = self.holder_index(d, c, d)?;
            if let Some(i) = level.index(idx) {
                let section = match level.slots[i] {
                    Slot::Present(s) => s,
                    Slot::Empty => return None,
                    Slot::Missing => {
                        if strict {
                            self.missing.set(true);
                        }
                        return None;
                    }
                };
                return self.read(section, c, strict);
            }
        }
        let k = self.holder(d, c, d)?;
        let section = if k == self.inputs.key {
            Some(&self.inputs.section)
        } else {
            match self.inputs.sections.get(&k) {
                Some(Some(s)) => Some(s),
                Some(None) => None,
                None => {
                    if strict {
                        self.missing.set(true);
                    }
                    None
                }
            }
        }?;
        self.read(section, c, strict)
    }

    /// Sample `c` (in its depth's sample grid) of `section`, the cell
    /// holding it, or `None` for vacuum.
    fn read(
        &self,
        section: &Section,
        c: [i64; 3],
        strict: bool,
    ) -> Option<gx_core::matter::Sample> {
        let n = self.n as i64;
        if i64::from(section.resolution()) != n {
            if strict {
                self.missing.set(true);
            }
            return None;
        }
        let l = c.map(|v| v.rem_euclid(n) as usize);
        let s = section
            .samples()?
            .get(l[0] + self.n * (l[1] + self.n * l[2]))?;
        forms_surface(s.state).then_some(s)
    }
}

impl MixedField for Field<'_> {
    fn resolution(&self) -> usize {
        self.n
    }

    fn root_origin(&self) -> [f64; 3] {
        self.root_origin
    }

    fn root_edge(&self) -> f64 {
        self.root_edge
    }

    fn drawn_depth(&self, d: u8, c: [i64; 3]) -> Option<u8> {
        let home = self.inputs.key.depth;
        (home.saturating_sub(1)..=home.saturating_add(1)).find(|&dd| {
            let Some(idx) = self.holder_index(d, c, dd) else {
                return false;
            };
            match self.level(dd).and_then(|l| Some((l, l.index(idx)?))) {
                Some((l, i)) => l.drawn[i],
                // Outside the region's tables: the ordered set.
                None => self
                    .holder(d, c, dd)
                    .is_some_and(|k| self.inputs.drawn.contains(&k)),
            }
        })
    }

    fn interior_value(&self, d: u8, q: [i64; 3]) -> Option<f64> {
        // A point strictly inside this cell: its eight samples are the
        // cell's own, no finer cell touches it (the cell is drawn, so
        // nothing below it is), and the sum runs in lattice_value's order.
        let key = self.inputs.key;
        if d != key.depth {
            return None;
        }
        let n = self.n as i64;
        let base = [key.x, key.y, key.z].map(|v| i64::from(v) * n);
        let l = [0, 1, 2].map(|a| q[a] - base[a]);
        if l.iter().any(|&v| !(1..n).contains(&v)) {
            return None;
        }
        let section = &self.inputs.section;
        if i64::from(section.resolution()) != n {
            return None;
        }
        let samples = section.samples()?;
        let mut sum = 0.0;
        for o in 0..8usize {
            let c = [
                (l[0] - 1 + (o & 1) as i64) as usize,
                (l[1] - 1 + ((o >> 1) & 1) as i64) as usize,
                (l[2] - 1 + ((o >> 2) & 1) as i64) as usize,
            ];
            let j = c[0] + self.n * (c[1] + self.n * c[2]);
            sum += if forms_surface(samples.state(j)) {
                samples.occupancy(j).value()
            } else {
                0.0
            };
        }
        Some(sum / 8.0)
    }

    fn occupancy(&self, d: u8, c: [i64; 3]) -> f64 {
        // Only the occupancy and the state are read, straight from their
        // channels, not the whole sample.
        let Some(level) = self.level(d) else {
            return self.sample(d, c, true).map_or(0.0, |s| s.occupancy.value());
        };
        let Some(idx) = self.holder_index(d, c, d) else {
            return 0.0;
        };
        let Some(i) = level.index(idx) else {
            return self.sample(d, c, true).map_or(0.0, |s| s.occupancy.value());
        };
        let section = match level.slots[i] {
            Slot::Present(s) => s,
            Slot::Empty => return 0.0,
            Slot::Missing => {
                self.missing.set(true);
                return 0.0;
            }
        };
        let n = self.n as i64;
        if i64::from(section.resolution()) != n {
            self.missing.set(true);
            return 0.0;
        }
        let Some(samples) = section.samples() else {
            return 0.0;
        };
        let l = c.map(|v| v.rem_euclid(n) as usize);
        let j = l[0] + self.n * (l[1] + self.n * l[2]);
        if j >= samples.len() || !forms_surface(samples.state(j)) {
            return 0.0;
        }
        samples.occupancy(j).value()
    }
}

/// Extracts one drawn cell from its inputs (module docs).
pub fn extract(inputs: &CellInputs) -> Extraction {
    let field = Field::new(inputs);
    let key = inputs.key;
    let mixed = extract_mixed(
        &field,
        CellAt {
            depth: key.depth,
            index: [key.x, key.y, key.z].map(i64::from),
        },
    );
    if field.missing.get() {
        return Extraction::Waiting;
    }
    let h = field.root_edge / (field.n as f64 * (1u64 << key.depth) as f64);
    let vertices = mixed
        .positions
        .iter()
        .zip(&mixed.normals)
        .map(|(p, &normal)| {
            let q: [i64; 3] =
                std::array::from_fn(|a| ((p[a] - field.root_origin[a]) / h).round() as i64);
            let (mut mass, mut albedo, mut rough, mut temp) = (0.0, [0.0; 3], 0.0, 0.0);
            let (mut densest, mut state) = (0.0, 0u8);
            for o in 0..8 {
                let c = [
                    q[0] - 1 + (o & 1),
                    q[1] - 1 + ((o >> 1) & 1),
                    q[2] - 1 + ((o >> 2) & 1),
                ];
                let Some(s) = field.sample(key.depth, c, false) else {
                    continue;
                };
                let w = s.density.value();
                if w <= 0.0 {
                    continue;
                }
                mass += w;
                for (b, a) in albedo.iter_mut().enumerate() {
                    *a += w * s.albedo[b].value();
                }
                rough += w * s.roughness.value();
                temp += w * s.radiating_temperature.value();
                if w > densest {
                    densest = w;
                    state = s.state.as_u8();
                }
            }
            let m = if mass > 0.0 { mass } else { 1.0 };
            SurfaceVertex {
                normal,
                albedo: albedo.map(|a| (a / m) as f32),
                roughness: (rough / m) as f32,
                temperature: (temp / m) as f32,
                state: u32::from(state),
                density: (mass / 8.0) as f32,
            }
        })
        .collect();
    let mut bounds_min = [f64::INFINITY; 3];
    let mut bounds_max = [f64::NEG_INFINITY; 3];
    for p in &mixed.positions {
        for a in 0..3 {
            bounds_min[a] = bounds_min[a].min(p[a]);
            bounds_max[a] = bounds_max[a].max(p[a]);
        }
    }
    if mixed.positions.is_empty() {
        let o = inputs.section.origin();
        bounds_min = [o.x, o.y, o.z];
        bounds_max = bounds_min;
    }
    Extraction::Done(Arc::new(SurfaceMesh {
        key,
        origin: inputs.section.origin(),
        positions: mixed.positions,
        vertices,
        indices: mixed.indices,
        bounds_min,
        bounds_max,
    }))
}

/// A finished job: the cell, its result, and the extraction time,
/// seconds.
pub type Finished = (CellKey, Extraction, f64);

/// Extracts and times one job.
fn run(inputs: &CellInputs) -> Finished {
    let t0 = crate::clock::now_seconds();
    let out = extract(inputs);
    (inputs.key, out, crate::clock::now_seconds() - t0)
}

/// A pool of extraction workers.
///
/// Jobs go in over a channel and finished results come back over another,
/// in whatever order the workers finish; each carries its key, and its
/// contents depend only on its inputs. With zero threads (the browser
/// build, where `std::thread` is unavailable) extraction runs inline on
/// [`ExtractPool::submit`].
pub struct ExtractPool {
    jobs: Option<mpsc::Sender<CellInputs>>,
    results: mpsc::Receiver<Finished>,
    inline: VecDeque<Finished>,
    workers: Vec<std::thread::JoinHandle<()>>,
}

/// The background thread budget: every core but two (the frame thread's,
/// and one for the selection thread, the network, decoding, and the
/// system), at least two. The module workers and the extraction pool share
/// it instead of each taking all of it: two pools each sized to every core
/// but two starved the frame thread (choppy frames while a place was being
/// computed).
fn background_threads() -> usize {
    std::thread::available_parallelism()
        .map_or(1, |n| n.get())
        .saturating_sub(2)
        .max(2)
}

/// Module worker threads: three quarters of [`background_threads`], at
/// least one. Computing sections is what a first visit waits on.
pub fn worker_threads() -> usize {
    (background_threads() * 3 / 4).max(1)
}

/// The extraction pool size: the rest of [`background_threads`] after
/// [`worker_threads`], at least one; zero on the web, where there are no
/// threads. Surface tiles are built on it too.
pub fn default_threads() -> usize {
    if cfg!(target_arch = "wasm32") {
        0
    } else {
        background_threads().saturating_sub(worker_threads()).max(1)
    }
}

impl ExtractPool {
    /// Starts `threads` workers (zero for inline extraction).
    pub fn new(threads: usize) -> ExtractPool {
        let (result_tx, results) = mpsc::channel();
        if threads == 0 {
            return ExtractPool {
                jobs: None,
                results,
                inline: VecDeque::new(),
                workers: Vec::new(),
            };
        }
        let (jobs, job_rx) = mpsc::channel::<CellInputs>();
        let job_rx = Arc::new(Mutex::new(job_rx));
        let workers = (0..threads)
            .map(|i| {
                let rx = job_rx.clone();
                let tx = result_tx.clone();
                std::thread::Builder::new()
                    .name(format!("gx-extract-{i}"))
                    .spawn(move || loop {
                        let job = rx.lock().expect("job queue lock").recv();
                        let Ok(inputs) = job else {
                            break;
                        };
                        if tx.send(run(&inputs)).is_err() {
                            break;
                        }
                    })
                    .expect("spawning an extraction worker")
            })
            .collect();
        ExtractPool {
            jobs: Some(jobs),
            results,
            inline: VecDeque::new(),
            workers,
        }
    }

    /// Queues one cell for extraction.
    pub fn submit(&mut self, inputs: CellInputs) {
        match &self.jobs {
            Some(tx) => {
                tx.send(inputs).expect("extraction workers are running");
            }
            None => {
                let done = run(&inputs);
                self.inline.push_back(done);
            }
        }
    }

    /// The results finished so far, without waiting.
    pub fn try_results(&mut self) -> Vec<Finished> {
        let mut out: Vec<_> = self.inline.drain(..).collect();
        out.extend(self.results.try_iter());
        out
    }

    /// Waits for the next finished result. `None` if nothing can arrive.
    pub fn wait_result(&mut self) -> Option<Finished> {
        if let Some(r) = self.inline.pop_front() {
            return Some(r);
        }
        self.jobs.as_ref()?;
        self.results.recv().ok()
    }
}

impl Drop for ExtractPool {
    fn drop(&mut self) {
        self.jobs = None;
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transvoxel::MixedField;
    use gx_core::matter::{Sample, Samples};
    use gx_core::units::{Attenuation, Density, Kelvin, Meters, Ratio, Vec3};

    /// The snapshot's field with the interior shortcut turned off: the
    /// general search of `lattice_value` for every point.
    struct NoShortcut<'a>(Field<'a>);

    impl MixedField for NoShortcut<'_> {
        fn resolution(&self) -> usize {
            self.0.resolution()
        }
        fn root_origin(&self) -> [f64; 3] {
            self.0.root_origin()
        }
        fn root_edge(&self) -> f64 {
            self.0.root_edge()
        }
        fn drawn_depth(&self, d: u8, c: [i64; 3]) -> Option<u8> {
            self.0.drawn_depth(d, c)
        }
        fn occupancy(&self, d: u8, c: [i64; 3]) -> f64 {
            self.0.occupancy(d, c)
        }
    }

    /// A cell of resolution 16 cut by a tilted plane and holding a ball,
    /// occupancy ramping linearly over one sample, so the surface crosses
    /// interior and boundary points alike.
    fn section() -> Section {
        let key = CellKey::new(1, 2, 1, 2, 1).unwrap();
        let g = key.geometry(Meters::new(64.0));
        let res = 16u8;
        let h = g.edge.value() / f64::from(res);
        let center = g.origin + Vec3::new(8.0, 8.0, 8.0);
        let samples = Samples::from_fn(res, |x, y, z| {
            let p = g.origin
                + Vec3::new(
                    (f64::from(x) + 0.5) * h,
                    (f64::from(y) + 0.5) * h,
                    (f64::from(z) + 0.5) * h,
                );
            let plane =
                (p.z - g.origin.z - 5.0) + 0.3 * (p.x - g.origin.x) - 0.2 * (p.y - g.origin.y);
            let ball = (p - center).length() - 4.0;
            let dist = plane.min(ball);
            let occ = (0.5 - dist / h).clamp(0.0, 1.0);
            if occ <= 0.0 {
                return Sample::VACUUM;
            }
            Sample {
                density: Density::new(2700.0 * occ),
                state: State::Solid,
                temperature: Kelvin::new(305.0),
                albedo: [Ratio::new(0.3); 3],
                roughness: Ratio::new(0.9),
                attenuation: Attenuation::new(0.0),
                radiating_temperature: Kelvin::new(305.0),
                occupancy: Ratio::new(occ),
            }
        });
        Section::new(key, g.origin, g.edge, res, samples).expect("a valid section")
    }

    #[test]
    fn interior_shortcut_gives_identical_meshes() {
        let inputs = CellInputs::alone(Arc::new(section()));
        let key = inputs.key;
        let cell = CellAt {
            depth: key.depth,
            index: [key.x, key.y, key.z].map(i64::from),
        };
        let fast = extract_mixed(&Field::new(&inputs), cell);
        let slow = extract_mixed(&NoShortcut(Field::new(&inputs)), cell);
        assert!(!fast.indices.is_empty(), "the surface crosses the cell");
        assert_eq!(fast.indices, slow.indices);
        let bits = |m: &crate::transvoxel::MixedMesh| -> Vec<u64> {
            m.positions
                .iter()
                .flatten()
                .map(|v| v.to_bits())
                .chain(m.normals.iter().flatten().map(|v| u64::from(v.to_bits())))
                .collect()
        };
        assert_eq!(bits(&fast), bits(&slow));
    }
}

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
        }
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
            self.holder(d, c, dd)
                .is_some_and(|k| self.inputs.drawn.contains(&k))
        })
    }

    fn occupancy(&self, d: u8, c: [i64; 3]) -> f64 {
        self.sample(d, c, true).map_or(0.0, |s| s.occupancy.value())
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

/// The default worker count: the available parallelism, at most four, and
/// zero on the web where there are no threads.
pub fn default_threads() -> usize {
    if cfg!(target_arch = "wasm32") {
        0
    } else {
        std::thread::available_parallelism()
            .map_or(1, |n| n.get())
            .clamp(1, 4)
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

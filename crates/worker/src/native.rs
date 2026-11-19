//! The native worker: modules run with `gx-module-host` (wasmtime) on a
//! small pool of threads, so compiling never blocks drawing.
//!
//! Implements `client-compute.md` section 1 step 4 and section 4 for the
//! desktop client. For one [`ComputeJob`] the pool:
//!
//! 1. checks the job's address against its inputs ([`ComputeJob::check_address`]),
//! 2. gets the module by hash (the local cache, else [`Blobs`]), checks the
//!    hash, and admits it once per session ([`ModuleRunner`]),
//! 3. runs `gx_sources` (for a version 2 module, `gx_inputs` first, and
//!    both with the job's frame records, section 4.1), resolves the ids
//!    through the layer's manifest (fetched and checked the same way), and
//!    gets every tile, checking its length and hash,
//! 4. resolves every input section `gx_inputs` named (section 10.3, below),
//!    runs `gx_compile` (version 2: with the frame records and the input
//!    sections, in slot order) and validates the section by decoding it
//!    with the library decoder (the one validation the section gets on its
//!    way in, `rendering.md` section 3.5),
//! 5. hands the bytes and the decoded section to the caller at once, stores them in the local cache
//!    under their address, and, for a `missing` direction only, offers
//!    it to the upload lane ([`Uploader`]), which never holds up the next
//!    job. A `local` section is never offered.
//!
//! Jobs wait in one queue per pool (`client-compute.md` section 2.8,
//! [`crate::queue`]): a free thread takes the waiting compute of the
//! highest priority; only when no compute waits does it take work from a
//! lower lane ([`WorkerPool::set_lane`]). The native client registers only
//! the check lane: uploads are network I/O, sent by the client on its async
//! runtime as soon as they are offered (at most four in flight), and never
//! take a pool thread. A lower lane that has work it cannot start yet wakes
//! the pool when it can ([`PoolWaker`]). [`WorkerPool::reprioritize`] follows the
//! wanted set: waiting jobs take their key's new priority, and a job whose
//! key left it is removed before it starts and told
//! [`crate::FailureKind::Dropped`]. A running job finishes and its section
//! is cached.
//!
//! # Inputs (`client-compute.md` sections 2.8 and 10.3)
//!
//! An input section is got as any section, through the pool's
//! [`InputSource`]: this session's memory of resolved sections, else the
//! local cache, else directions for its key, then a stream (`ready`,
//! checked against the hash the directions carried) or a compute
//! (`missing`, `local`), which resolves its own inputs the same way. A
//! computed `missing` input is offered to the upload lane like any section.
//! Input work runs on the thread of the compute that needs it, so it has
//! that compute's priority and never waits in the queue behind other work,
//! and a compute dropped before it starts never resolves its inputs. Work
//! on one address is shared: whoever gets an address first fetches or
//! computes it while every other thread that needs it (another compute's
//! input, or the wanted set's own job for it) waits and takes the same
//! bytes. Locks are taken along the layers' input order, which has no
//! cycle, so the waits cannot deadlock. No input work ever runs on the
//! frame thread.
//!
//! The worker sees modules, requests, tile bytes, and section bytes, and
//! nothing else.

use crate::cache::DiskCache;
use crate::job::{
    check_blob, check_section, check_tile, decode_manifest, resolve_tiles, BlobKind, ComputeJob,
    Computed, WorkerError,
};
use crate::queue::{ComputeQueue, Lane};
use crate::stats::{WorkerSnapshot, WorkerStats};
use gx_core::address::SlotKind;
use gx_core::key::ChunkKey;
use gx_core::sha256::to_hex;
use gx_module_host::{HostConfig, HostError, LoadedModule, ModuleHost};
use std::collections::BTreeMap;
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

/// Where the worker gets content-addressed bytes it does not have cached:
/// `GET /{kind}/{hash}` through the edge, in the client. Called from the
/// pool's threads; may block.
pub trait Blobs: Send + Sync {
    /// The bytes named `hash` of `kind`. The worker checks them.
    fn fetch(&self, kind: BlobKind, hash: &[u8; 32]) -> Result<Vec<u8>, WorkerError>;
}

/// Takes a computed `missing` section into the upload lane
/// (`client-compute.md` sections 2.5 and 2.8, [`crate::upload`]). Must
/// not block: queue the offer and return. The section has already been
/// handed to the renderer.
pub trait Uploader: Send + Sync {
    /// Offers `section` for `job`, once.
    fn offer(&self, job: &ComputeJob, section: Arc<[u8]>);
}

/// What directions say about one input section (`client-compute.md`
/// section 2.4), as [`InputSource::direct`] reports it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InputDirection {
    /// The layer contributes nothing for this key: the input is passed as
    /// an empty section (section 4.1).
    Empty,
    /// Stored: stream it and check the bytes against `sha256`.
    Ready {
        /// SHA-256 of the stored bytes.
        sha256: [u8; 32],
    },
    /// Not stored: compute it, and offer it to the hub when `share` is
    /// `true` (`missing`); keep it here when `false` (`local`).
    Compute {
        /// `true` for `missing`, `false` for `local`.
        share: bool,
    },
}

/// Where a compute's input sections come from (`client-compute.md` section
/// 10.3), in the client. Called from the pool's threads; may block.
pub trait InputSource: Send + Sync {
    /// The job that makes the section `key` of `reader`'s layer slot
    /// `slot`, at the input span (section 10.3), with its address, slots,
    /// and frame records filled in from the build. Its `share` is ignored:
    /// [`InputSource::direct`] decides it. Fails for a slot that is not a
    /// layer slot of the reader's layer.
    fn input_job(
        &self,
        reader: &ComputeJob,
        slot: u32,
        key: ChunkKey,
    ) -> Result<ComputeJob, WorkerError>;

    /// Directions for `job`'s layer and key at `job`'s span. Fails when the
    /// answer's address is not `job`'s.
    fn direct(&self, job: &ComputeJob) -> Result<InputDirection, WorkerError>;

    /// The stored section at `address` (`GET /section/{address}?sha256=`).
    /// The worker checks the hash.
    fn stream(&self, address: &[u8; 32], sha256: &[u8; 32]) -> Result<Vec<u8>, WorkerError>;
}

/// Sections resolved as inputs, or computed, kept in memory
/// ([`ResolvedMemory`]).
pub const RESOLVED_MEMORY_SECTIONS: usize = 256;

/// Sections this session resolved or computed, by address, at most
/// [`RESOLVED_MEMORY_SECTIONS`]; the oldest go first. They hand one
/// address's bytes to every waiter without a cache directory.
#[derive(Default)]
struct ResolvedMemory {
    sections: BTreeMap<[u8; 32], Arc<[u8]>>,
    order: std::collections::VecDeque<[u8; 32]>,
}

impl ResolvedMemory {
    fn get(&self, address: &[u8; 32]) -> Option<Arc<[u8]>> {
        self.sections.get(address).cloned()
    }

    fn put(&mut self, address: [u8; 32], bytes: Arc<[u8]>) {
        if self.sections.insert(address, bytes).is_none() {
            self.order.push_back(address);
        }
        while self.order.len() > RESOLVED_MEMORY_SECTIONS {
            if let Some(old) = self.order.pop_front() {
                self.sections.remove(&old);
            }
        }
    }
}

/// The default pool size: the number of cores minus two, at least one.
pub fn default_threads() -> usize {
    std::thread::available_parallelism()
        .map_or(1, |n| n.get())
        .saturating_sub(2)
        .max(1)
}

fn host_error(e: HostError) -> WorkerError {
    WorkerError::permanent(e.to_string())
}

/// What a [`SingleFlight`] lock is keyed by: the blob kind (`None` for
/// admitting a module) and the hash.
type FlightKey = (Option<BlobKind>, [u8; 32]);

/// One lock per name being fetched, so threads that need the same module,
/// manifest, or tile at once fetch it once: the first fetches and caches,
/// the others wait and read the cache.
#[derive(Default)]
struct SingleFlight {
    /// Keyed by blob kind and name; `None` for admitting a module.
    locks: Mutex<BTreeMap<FlightKey, Arc<Mutex<()>>>>,
}

impl SingleFlight {
    fn run<T>(&self, kind: Option<BlobKind>, hash: &[u8; 32], f: impl FnOnce() -> T) -> T {
        let lock = {
            let mut locks = self.locks.lock().expect("fetch locks");
            Arc::clone(locks.entry((kind, *hash)).or_default())
        };
        let out = {
            let _held = lock.lock().unwrap_or_else(|p| p.into_inner());
            f()
        };
        let mut locks = self.locks.lock().expect("fetch locks");
        // Drop the entry once nobody else holds it.
        if Arc::strong_count(&lock) == 2 {
            locks.remove(&(kind, *hash));
        }
        out
    }
}

/// Gets bytes by name: the cache first (checked, dropped if damaged), then
/// [`Blobs`], whose bytes are checked and cached.
fn get_blob(
    kind: BlobKind,
    hash: &[u8; 32],
    blobs: &dyn Blobs,
    cache: Option<&DiskCache>,
    flights: &SingleFlight,
) -> Result<Vec<u8>, WorkerError> {
    flights.run(Some(kind), hash, || get_blob_now(kind, hash, blobs, cache))
}

fn get_blob_now(
    kind: BlobKind,
    hash: &[u8; 32],
    blobs: &dyn Blobs,
    cache: Option<&DiskCache>,
) -> Result<Vec<u8>, WorkerError> {
    let name = to_hex(hash);
    if let Some(c) = cache {
        if let Some(bytes) = c.get(kind, &name) {
            if check_blob(kind, hash, &bytes).is_ok() {
                return Ok(bytes);
            }
            tracing::warn!(%kind, name, "cached bytes are damaged, fetching again");
            c.remove(kind, &name);
        }
    }
    let bytes = blobs.fetch(kind, hash)?;
    check_blob(kind, hash, &bytes)?;
    if let Some(c) = cache {
        if let Err(e) = c.put(kind, &name, &bytes) {
            tracing::warn!(%kind, name, "could not cache: {e}");
        }
    }
    Ok(bytes)
}

/// Bytes of verified source tiles kept in memory ([`SourceMemory`]).
pub const SOURCE_MEMORY_BYTES: usize = 512 << 20;

/// Source tiles already fetched and checked this session, by hash, up to
/// [`SOURCE_MEMORY_BYTES`]; the oldest go first. Neighboring sections read
/// mostly the same tiles, so each is read from disk and hashed once.
#[derive(Default)]
struct SourceMemory {
    tiles: BTreeMap<[u8; 32], Arc<Vec<u8>>>,
    order: std::collections::VecDeque<[u8; 32]>,
    bytes: usize,
}

impl SourceMemory {
    fn get(&self, hash: &[u8; 32]) -> Option<Arc<Vec<u8>>> {
        self.tiles.get(hash).cloned()
    }

    fn put(&mut self, hash: [u8; 32], bytes: Arc<Vec<u8>>) {
        if self.tiles.contains_key(&hash) {
            return;
        }
        self.bytes += bytes.len();
        self.tiles.insert(hash, bytes);
        self.order.push_back(hash);
        while self.bytes > SOURCE_MEMORY_BYTES {
            let Some(old) = self.order.pop_front() else {
                break;
            };
            if let Some(b) = self.tiles.remove(&old) {
                self.bytes -= b.len();
            }
        }
    }
}

/// The module host and every module admitted this session, by hash; the
/// source manifests decoded this session, by hash; and recent source tiles.
pub struct ModuleRunner {
    host: ModuleHost,
    loaded: Mutex<BTreeMap<[u8; 32], Arc<LoadedModule>>>,
    manifests: Mutex<BTreeMap<[u8; 32], Arc<gx_core::manifest::Manifest>>>,
    sources: Mutex<SourceMemory>,
    stats: Arc<WorkerStats>,
    flights: SingleFlight,
    resolved: Mutex<ResolvedMemory>,
}

/// What [`ModuleRunner::run`] uses to resolve a job's input sections:
/// where they come from, the cache, and where computed `missing` inputs
/// are offered.
#[derive(Copy, Clone)]
pub struct Resolver<'a> {
    /// Directions, streams, and input jobs.
    pub source: &'a dyn InputSource,
    /// Takes computed `missing` inputs; `None` to offer nothing (checks).
    pub uploader: Option<&'a dyn Uploader>,
}

impl ModuleRunner {
    /// A runner with the host's limits (`client-compute.md` section 4: 1 GiB
    /// and a time limit per call by default).
    pub fn new(config: HostConfig, stats: Arc<WorkerStats>) -> Result<ModuleRunner, WorkerError> {
        Ok(ModuleRunner {
            host: ModuleHost::with_config(config).map_err(host_error)?,
            loaded: Mutex::new(BTreeMap::new()),
            manifests: Mutex::new(BTreeMap::new()),
            sources: Mutex::new(SourceMemory::default()),
            stats,
            flights: SingleFlight::default(),
            resolved: Mutex::new(ResolvedMemory::default()),
        })
    }

    /// The source manifest `hash`, fetched, checked, and decoded the first
    /// time and kept for the session: it lists every source tile (tens of
    /// megabytes), and every section of its layer needs it.
    fn manifest(
        &self,
        hash: &[u8; 32],
        blobs: &dyn Blobs,
        cache: Option<&DiskCache>,
    ) -> Result<Arc<gx_core::manifest::Manifest>, WorkerError> {
        let known = || self.manifests.lock().expect("manifests").get(hash).cloned();
        if let Some(m) = known() {
            return Ok(m);
        }
        self.flights.run(None, hash, || {
            if let Some(m) = known() {
                return Ok(m);
            }
            let m = Arc::new(decode_manifest(&get_blob_now(
                BlobKind::Manifest,
                hash,
                blobs,
                cache,
            )?)?);
            self.manifests
                .lock()
                .expect("manifests")
                .insert(*hash, Arc::clone(&m));
            Ok(m)
        })
    }

    /// The counters this runner updates.
    pub fn stats(&self) -> &Arc<WorkerStats> {
        &self.stats
    }

    /// The admitted module for `hash`, admitting `bytes()` the first time.
    /// Threads asking for the same module at once admit it once.
    pub fn module(
        &self,
        hash: &[u8; 32],
        bytes: impl FnOnce() -> Result<Vec<u8>, WorkerError>,
    ) -> Result<Arc<LoadedModule>, WorkerError> {
        let loaded = || self.loaded.lock().expect("module map").get(hash).cloned();
        if let Some(m) = loaded() {
            return Ok(m);
        }
        self.flights.run(None, hash, || {
            if let Some(m) = loaded() {
                return Ok(m);
            }
            let wasm = bytes()?;
            let module = Arc::new(self.host.load(&wasm, hash).map_err(host_error)?);
            self.loaded
                .lock()
                .expect("module map")
                .insert(*hash, Arc::clone(&module));
            self.stats.module_loaded();
            Ok(module)
        })
    }

    /// Runs one job to a validated section (steps 1 to 4 of the module
    /// docs), resolving its input sections through `inputs`. Does not
    /// upload the job's own section; computed `missing` inputs are offered
    /// to the resolver's uploader. A job whose module names input sections
    /// fails when `inputs` is `None`.
    pub fn run(
        &self,
        job: &ComputeJob,
        blobs: &dyn Blobs,
        cache: Option<&DiskCache>,
        inputs: Option<Resolver<'_>>,
    ) -> Result<Computed, WorkerError> {
        job.check_address()?;
        let module = self.module(&job.module, || {
            get_blob(BlobKind::Module, &job.module, blobs, cache, &self.flights)
        })?;
        self.stats.module_run();
        let frames: Vec<&[u8]> = job.frames.iter().map(|f| f.as_ref()).collect();
        let v2 = module.abi_version() >= 2;
        let list = if v2 {
            module.inputs(&job.request, &frames).map_err(host_error)?
        } else {
            Vec::new()
        };
        if !v2 && !frames.is_empty() {
            return Err(WorkerError::permanent(
                "frame records given to a version 1 module, which reads none",
            ));
        }
        let ids = if v2 {
            module.sources_v2(&job.request, &frames)
        } else {
            module.sources(&job.request)
        }
        .map_err(host_error)?;
        let manifest = match &job.manifest {
            Some(h) if !ids.is_empty() => Some(self.manifest(h, blobs, cache)?),
            _ => None,
        };
        let refs = resolve_tiles(manifest.as_deref(), &ids)?;
        let mut tiles = Vec::with_capacity(refs.len());
        for r in &refs {
            let remembered = self.sources.lock().expect("source memory").get(&r.sha256);
            let bytes = match remembered {
                Some(b) if b.len() as u64 == r.length => b,
                _ => {
                    let bytes = get_blob(BlobKind::Source, &r.sha256, blobs, cache, &self.flights)?;
                    check_tile(r, &bytes)?;
                    let bytes = Arc::new(bytes);
                    self.sources
                        .lock()
                        .expect("source memory")
                        .put(r.sha256, Arc::clone(&bytes));
                    bytes
                }
            };
            tiles.push((r.id, bytes.as_ref().clone()));
        }
        let passed = if v2 {
            self.gather_inputs(job, &list, blobs, cache, inputs)?
        } else {
            Vec::new()
        };
        let passed_refs: Vec<&[u8]> = passed.iter().map(|b| b.as_ref()).collect();
        self.stats.module_run();
        let bytes = if v2 {
            module.compile_v2(&job.request, &tiles, &passed_refs)
        } else {
            module.compile(&job.request, &tiles)
        }
        .map_err(host_error)?;
        let section = check_section(&job.request.key, &bytes)?;
        Ok(Computed {
            bytes: Arc::from(bytes),
            section: section.map(Arc::new),
            inputs_sha256: (!job.slots.is_empty())
                .then(|| gx_core::module_io::inputs_sha256(&passed_refs)),
        })
    }

    /// The inputs `gx_compile` receives (`client-compute.md` section 4.1),
    /// in slot order: a frames slot's records from the job, a layer slot's
    /// sections in the order `list` names them, each resolved through
    /// `inputs` (section 10.3).
    fn gather_inputs(
        &self,
        job: &ComputeJob,
        list: &[(u32, ChunkKey)],
        blobs: &dyn Blobs,
        cache: Option<&DiskCache>,
        inputs: Option<Resolver<'_>>,
    ) -> Result<Vec<Arc<[u8]>>, WorkerError> {
        let slots = job.slots.len();
        let per_slot = if job.frames_per_slot.len() == slots {
            job.frames_per_slot.clone()
        } else if slots == 0 {
            Vec::new()
        } else {
            return Err(WorkerError::permanent(format!(
                "the job has {slots} slots and {} frame counts",
                job.frames_per_slot.len()
            )));
        };
        let mut out = Vec::with_capacity(job.frames.len() + list.len());
        let mut records = job.frames.iter();
        for (slot, (value, &count)) in job.slots.iter().zip(&per_slot).enumerate() {
            match value.kind {
                SlotKind::Frames => {
                    for _ in 0..count {
                        let r = records.next().ok_or_else(|| {
                            WorkerError::permanent("the job has fewer frame records than slots")
                        })?;
                        out.push(Arc::clone(r));
                    }
                }
                SlotKind::Layer => {
                    for (_, key) in list.iter().filter(|(s, _)| *s as usize == slot) {
                        let Some(resolver) = inputs else {
                            return Err(WorkerError::permanent(format!(
                                "the module reads input {key} of slot {slot}, and nothing resolves inputs here"
                            )));
                        };
                        out.push(self.input(job, slot as u32, *key, blobs, cache, resolver)?);
                    }
                }
            }
        }
        if let Some((slot, key)) = list.iter().find(|(s, _)| {
            job.slots
                .get(*s as usize)
                .is_none_or(|v| v.kind != SlotKind::Layer)
        }) {
            return Err(WorkerError::permanent(format!(
                "the module names input {key} of slot {slot}, which is not a layer slot of the layer"
            )));
        }
        if slots == 0 && !job.frames.is_empty() {
            // A job without slots passes its frame records as given.
            out.extend(job.frames.iter().cloned());
        }
        Ok(out)
    }

    /// One input section of `reader` (`client-compute.md` section 10.3):
    /// this session's memory, else the cache, else directions and a stream
    /// or a compute. One thread per address does the work; the others wait
    /// for it and take its bytes.
    fn input(
        &self,
        reader: &ComputeJob,
        slot: u32,
        key: ChunkKey,
        blobs: &dyn Blobs,
        cache: Option<&DiskCache>,
        resolver: Resolver<'_>,
    ) -> Result<Arc<[u8]>, WorkerError> {
        let mut job = resolver.source.input_job(reader, slot, key)?;
        let address = job.address;
        if let Some(b) = self.known_section(&key, &address, cache) {
            return Ok(b);
        }
        self.flights.run(Some(BlobKind::Section), &address, || {
            if let Some(b) = self.known_section(&key, &address, cache) {
                return Ok(b);
            }
            let bytes: Arc<[u8]> = match resolver.source.direct(&job)? {
                InputDirection::Empty => Arc::from(Vec::new()),
                InputDirection::Ready { sha256 } => {
                    let bytes = resolver.source.stream(&address, &sha256)?;
                    if gx_core::sha256::sha256(&bytes) != sha256 {
                        return Err(WorkerError::transient(format!(
                            "input section {} does not hash to {}",
                            to_hex(&address),
                            to_hex(&sha256)
                        )));
                    }
                    check_section(&key, &bytes)?;
                    let bytes: Arc<[u8]> = Arc::from(bytes);
                    self.remember(&address, &bytes, cache);
                    bytes
                }
                InputDirection::Compute { share } => {
                    job.share = share;
                    let computed = self.run(&job, blobs, cache, Some(resolver))?;
                    self.stats.job_finished(true);
                    self.remember(&address, &computed.bytes, cache);
                    job.inputs_sha256 = computed.inputs_sha256;
                    if share {
                        if let Some(u) = resolver.uploader {
                            u.offer(&job, Arc::clone(&computed.bytes));
                        }
                    }
                    computed.bytes
                }
            };
            Ok(bytes)
        })
    }

    /// The section at `address` if this session resolved it or the cache
    /// holds a valid copy.
    fn known_section(
        &self,
        key: &ChunkKey,
        address: &[u8; 32],
        cache: Option<&DiskCache>,
    ) -> Option<Arc<[u8]>> {
        if let Some(b) = self.resolved.lock().expect("resolved").get(address) {
            return Some(b);
        }
        let name = to_hex(address);
        let bytes = cache?.get(BlobKind::Section, &name)?;
        if check_section(key, &bytes).is_err() {
            return None;
        }
        let bytes: Arc<[u8]> = Arc::from(bytes);
        self.resolved
            .lock()
            .expect("resolved")
            .put(*address, Arc::clone(&bytes));
        Some(bytes)
    }

    /// Keeps a resolved or computed section in memory and in the cache.
    fn remember(&self, address: &[u8; 32], bytes: &Arc<[u8]>, cache: Option<&DiskCache>) {
        self.resolved
            .lock()
            .expect("resolved")
            .put(*address, Arc::clone(bytes));
        if let Some(c) = cache {
            let name = to_hex(address);
            if let Err(e) = c.put(BlobKind::Section, &name, bytes) {
                tracing::warn!(address = name, "could not cache the section: {e}");
            }
        }
    }
}

/// What the caller is told when a job ends.
pub type JobResult = Result<Computed, WorkerError>;

struct Task {
    job: ComputeJob,
    done: Box<dyn FnOnce(JobResult) + Send>,
}

/// A lower lane's work (uploads, checks): a pool thread asks it for the
/// next piece of work only when no higher lane has any waiting.
pub trait LaneSource: Send + Sync {
    /// The next piece of work, or `None` when nothing waits or nothing
    /// may start yet. Called with the pool's queue locked: must not block.
    /// The work runs on the pool thread; an upload's work only hands the
    /// request to an async runtime and returns, so no network wait ever
    /// holds a compute thread. A source that returned `None` while work
    /// waits calls [`PoolWaker::wake`] once that work may start.
    fn next(&self) -> Option<Box<dyn FnOnce() + Send>>;
}

/// What the pool's threads wait on.
struct State {
    computes: ComputeQueue<Task>,
    /// The upload and check lanes, by [`Lane`].
    lanes: Vec<(Lane, Arc<dyn LaneSource>)>,
    stopping: bool,
}

/// The queue and the signal that something was queued.
struct Queue {
    state: Mutex<State>,
    ready: Condvar,
}

/// One piece of work taken by a thread.
enum Work {
    Compute(Box<Task>),
    Lower(Box<dyn FnOnce() + Send>),
}

impl Queue {
    /// Blocks until there is work, highest lane first, or the pool stops
    /// with nothing left (`None`).
    fn take(&self) -> Option<Work> {
        let mut state = self.state.lock().expect("job queue");
        loop {
            if let Some((_, task)) = state.computes.pop() {
                return Some(Work::Compute(Box::new(task)));
            }
            for (_, source) in &state.lanes {
                if let Some(work) = source.next() {
                    return Some(Work::Lower(work));
                }
            }
            if state.stopping {
                return None;
            }
            state = self.ready.wait(state).expect("job queue");
        }
    }
}

/// Wakes a pool's idle threads so they ask the lane sources again
/// ([`WorkerPool::waker`]). Safe to call from any thread, after the pool
/// is gone too.
#[derive(Clone)]
pub struct PoolWaker {
    queue: Arc<Queue>,
}

impl PoolWaker {
    /// Wakes every idle thread. Takes the queue's lock first, so a thread
    /// that has just found nothing and is about to wait cannot miss it.
    pub fn wake(&self) {
        drop(self.queue.state.lock().expect("job queue"));
        self.queue.ready.notify_all();
    }
}

/// The pool of worker threads.
pub struct WorkerPool {
    queue: Arc<Queue>,
    shared: Arc<Shared>,
    threads: Vec<JoinHandle<()>>,
    stats: Arc<WorkerStats>,
    size: usize,
}

/// Recomputes sections for the check lane ([`WorkerPool::check_runner`]).
#[derive(Clone)]
pub struct CheckRunner {
    shared: Arc<Shared>,
}

impl CheckRunner {
    /// Runs `job` to a validated section on this thread. Counts nothing
    /// as computed and neither caches nor offers the section.
    /// A layer with inputs resolves them the same way a compute does
    /// (`client-compute.md` section 10.3); computed `missing` inputs are
    /// offered like any.
    pub fn run(&self, job: &ComputeJob) -> JobResult {
        let source = self.shared.inputs.lock().expect("input source").clone();
        self.shared.runner.run(
            job,
            self.shared.blobs.as_ref(),
            self.shared.cache.as_deref(),
            source.as_deref().map(|s| Resolver {
                source: s,
                uploader: Some(self.shared.uploader.as_ref()),
            }),
        )
    }
}

/// Everything the pool's threads share.
struct Shared {
    runner: ModuleRunner,
    blobs: Arc<dyn Blobs>,
    cache: Option<Arc<DiskCache>>,
    uploader: Arc<dyn Uploader>,
    inputs: Mutex<Option<Arc<dyn InputSource>>>,
}

impl WorkerPool {
    /// Starts `threads` workers (at least one) with the host limits in
    /// `config`, fetching through `blobs`, caching in `cache`, and
    /// offering `missing` sections to `uploader`.
    pub fn new(
        threads: usize,
        config: HostConfig,
        blobs: Arc<dyn Blobs>,
        cache: Option<Arc<DiskCache>>,
        uploader: Arc<dyn Uploader>,
    ) -> Result<WorkerPool, WorkerError> {
        WorkerPool::with_stats(threads, config, blobs, cache, uploader, WorkerStats::new())
    }

    /// [`WorkerPool::new`], counting into `stats`, which an upload lane
    /// ([`crate::UploadLane`]) may share.
    pub fn with_stats(
        threads: usize,
        config: HostConfig,
        blobs: Arc<dyn Blobs>,
        cache: Option<Arc<DiskCache>>,
        uploader: Arc<dyn Uploader>,
        stats: Arc<WorkerStats>,
    ) -> Result<WorkerPool, WorkerError> {
        let shared = Arc::new(Shared {
            runner: ModuleRunner::new(config, Arc::clone(&stats))?,
            blobs,
            cache,
            uploader,
            inputs: Mutex::new(None),
        });
        let queue = Arc::new(Queue {
            state: Mutex::new(State {
                computes: ComputeQueue::new(),
                lanes: Vec::new(),
                stopping: false,
            }),
            ready: Condvar::new(),
        });
        let size = threads.max(1);
        let threads = (0..size)
            .map(|i| {
                let queue = Arc::clone(&queue);
                let shared = Arc::clone(&shared);
                std::thread::Builder::new()
                    .name(format!("gx-worker-{i}"))
                    .spawn(move || {
                        while let Some(work) = queue.take() {
                            match work {
                                Work::Compute(task) => run_task(&shared, *task),
                                Work::Lower(work) => work(),
                            }
                        }
                    })
                    .expect("spawning a worker thread")
            })
            .collect();
        Ok(WorkerPool {
            queue,
            shared,
            threads,
            stats,
            size,
        })
    }

    /// Number of worker threads.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Takes `lane`'s work ([`Lane::Upload`] or [`Lane::Check`]) from
    /// `source`, below every higher lane, replacing an earlier source for
    /// it. [`Lane::Compute`] is the pool's own queue and is ignored here.
    pub fn set_lane(&self, lane: Lane, source: Arc<dyn LaneSource>) {
        if lane == Lane::Compute {
            return;
        }
        let mut state = self.queue.state.lock().expect("job queue");
        state.lanes.retain(|(l, _)| *l != lane);
        state.lanes.push((lane, source));
        state.lanes.sort_by_key(|(l, _)| *l);
        drop(state);
        self.queue.ready.notify_all();
    }

    /// Resolves the input sections of layers with inputs through `source`
    /// (`client-compute.md` section 10.3), for computes and checks. Until
    /// it is set, a module that names input sections fails.
    pub fn set_inputs(&self, source: Arc<dyn InputSource>) {
        *self.shared.inputs.lock().expect("input source") = Some(source);
    }

    /// Tells idle threads that a lane source has new work. Work offered
    /// from a pool thread (a finished job's upload) needs no call: the
    /// thread looks again before it waits.
    pub fn wake(&self) {
        self.waker().wake();
    }

    /// A handle that wakes the pool's idle threads from anywhere, for a
    /// lane source whose work becomes ready off the pool (an upload slot
    /// freed on the async runtime).
    pub fn waker(&self) -> PoolWaker {
        PoolWaker {
            queue: Arc::clone(&self.queue),
        }
    }

    /// Queues `job` at `priority` (larger starts sooner; the key's
    /// projected size on screen, pixels); `done` is called on a worker
    /// thread with the section (before it is offered for upload) or the
    /// failure, [`crate::FailureKind::Dropped`] if the job was removed by
    /// [`WorkerPool::reprioritize`] before it started.
    pub fn submit(
        &self,
        job: ComputeJob,
        priority: f64,
        done: impl FnOnce(JobResult) + Send + 'static,
    ) {
        let task = Task {
            job,
            done: Box::new(done),
        };
        let mut state = self.queue.state.lock().expect("job queue");
        if state.stopping {
            drop(state);
            (task.done)(Err(WorkerError::transient("the worker pool has stopped")));
            return;
        }
        state.computes.push(task.job.key(), priority, task);
        drop(state);
        self.queue.ready.notify_one();
    }

    /// Follows a new wanted set: every waiting job takes its key's
    /// priority from `priority`, and every waiting job whose key gets
    /// `None` is removed and its `done` told
    /// [`crate::FailureKind::Dropped`] (on this thread). Running jobs are
    /// not affected. Returns how many were dropped.
    pub fn reprioritize(&self, priority: impl Fn(&ChunkKey) -> Option<f64>) -> usize {
        let dropped = {
            let mut state = self.queue.state.lock().expect("job queue");
            state.computes.reprioritize(priority)
        };
        let n = dropped.len();
        for task in dropped {
            (task.done)(Err(WorkerError::dropped()));
        }
        n
    }

    /// A handle that recomputes a section on the calling thread with the
    /// pool's module host, fetcher, and cache, without caching or offering
    /// the result: the check lane's work, run on a pool thread.
    pub fn check_runner(&self) -> CheckRunner {
        CheckRunner {
            shared: Arc::clone(&self.shared),
        }
    }

    /// Compute jobs waiting to start.
    pub fn waiting(&self) -> usize {
        self.queue.state.lock().expect("job queue").computes.len()
    }

    /// The counters now.
    pub fn stats(&self) -> WorkerSnapshot {
        self.stats.snapshot()
    }

    /// The live counters, which stay readable after the pool is dropped.
    pub fn stats_handle(&self) -> Arc<WorkerStats> {
        Arc::clone(&self.stats)
    }
}

impl Drop for WorkerPool {
    /// Lets the threads finish what is queued in every lane, then joins
    /// them.
    fn drop(&mut self) {
        self.queue.state.lock().expect("job queue").stopping = true;
        self.queue.ready.notify_all();
        for t in self.threads.drain(..) {
            let _ = t.join();
        }
    }
}

fn run_task(shared: &Shared, task: Task) {
    let Task { mut job, done } = task;
    let runner = &shared.runner;
    let stats = runner.stats();
    let source = shared.inputs.lock().expect("input source").clone();
    let resolver = source.as_deref().map(|s| Resolver {
        source: s,
        uploader: Some(shared.uploader.as_ref()),
    });
    let key = job.key();
    let address = job.address;
    let cache = shared.cache.as_deref();
    // One address is worked on once: a section already resolved as
    // another compute's input is handed over, and one being resolved now
    // is waited for (section 10.3).
    let result = runner.flights.run(Some(BlobKind::Section), &address, || {
        let remembered = runner.resolved.lock().expect("resolved").get(&address);
        if let Some(bytes) = remembered {
            return check_section(&key, &bytes).map(|section| {
                (
                    Computed {
                        bytes,
                        section: section.map(Arc::new),
                        inputs_sha256: None,
                    },
                    false,
                )
            });
        }
        let r = runner.run(&job, shared.blobs.as_ref(), cache, resolver)?;
        runner
            .resolved
            .lock()
            .expect("resolved")
            .put(address, Arc::clone(&r.bytes));
        Ok((r, true))
    });
    let result = match result {
        // Already computed here: handed over, neither cached nor offered
        // again.
        Ok((computed, false)) => {
            done(Ok(computed));
            return;
        }
        Ok((computed, true)) => Ok(computed),
        Err(e) => Err(e),
    };
    stats.job_finished(result.is_ok());
    match result {
        Ok(computed) => {
            let section = Arc::clone(&computed.bytes);
            job.inputs_sha256 = computed.inputs_sha256;
            done(Ok(computed));
            let address = job.address_hex();
            if let Some(c) = &shared.cache {
                if let Err(e) = c.put(BlobKind::Section, &address, &section) {
                    tracing::warn!(address, "could not cache the section: {e}");
                }
            }
            if job.share {
                shared.uploader.offer(&job, section);
            }
        }
        Err(e) => {
            tracing::warn!(key = %job.key(), address = job.address_hex(), "module job failed: {e}");
            done(Err(e));
        }
    }
}

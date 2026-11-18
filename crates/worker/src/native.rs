//! The native worker: modules run with `gx-module-host` (wasmtime) on a
//! small pool of threads, so compiling never blocks drawing.
//!
//! Implements `client-compute.md` section 1 step 4 and section 4 for the
//! desktop client. For one [`ComputeJob`] the pool:
//!
//! 1. checks the job's address against its inputs ([`ComputeJob::check_address`]),
//! 2. gets the module by hash (the local cache, else [`Blobs`]), checks the
//!    hash, and admits it once per session ([`ModuleRunner`]),
//! 3. runs `gx_sources`, resolves the ids through the layer's manifest
//!    (fetched and checked the same way), and gets every tile, checking
//!    its length and hash,
//! 4. runs `gx_compile` and validates the section by decoding it with the
//!    library decoder (the one validation the section gets on its way in,
//!    `rendering.md` section 3.5),
//! 5. hands the bytes and the decoded section to the caller at once, stores them in the local cache
//!    under their address, and, for a `missing` direction only, offers
//!    it to the upload lane ([`Uploader`]), which never holds up the next
//!    job. A `local` section is never offered.
//!
//! Jobs wait in one queue per pool (`client-compute.md` section 2.8,
//! [`crate::queue`]): a free thread takes the waiting compute of the
//! highest priority; only when no compute waits does it take work from the
//! upload lane, and only when no upload waits from the check lane
//! ([`WorkerPool::set_lane`]). Lower-lane work never holds a thread on the
//! network: the client's upload source only hands each request to its
//! async runtime and returns, with at most four in flight; while four are
//! in flight it gives the pool nothing and the offers stay in the bounded
//! lane, and a finished request wakes the pool ([`PoolWaker`]). [`WorkerPool::reprioritize`] follows the
//! wanted set: waiting jobs take their key's new priority, and a job whose
//! key left it is removed before it starts and told
//! [`crate::FailureKind::Dropped`]. A running job finishes and its section
//! is cached.
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
    /// docs). Does not upload.
    pub fn run(
        &self,
        job: &ComputeJob,
        blobs: &dyn Blobs,
        cache: Option<&DiskCache>,
    ) -> Result<Computed, WorkerError> {
        job.check_address()?;
        let module = self.module(&job.module, || {
            get_blob(BlobKind::Module, &job.module, blobs, cache, &self.flights)
        })?;
        self.stats.module_run();
        let ids = module.sources(&job.request).map_err(host_error)?;
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
        self.stats.module_run();
        let bytes = module.compile(&job.request, &tiles).map_err(host_error)?;
        let section = check_section(&job.request.key, &bytes)?;
        Ok(Computed {
            bytes: Arc::from(bytes),
            section: section.map(Arc::new),
        })
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
    Compute(Task),
    Lower(Box<dyn FnOnce() + Send>),
}

impl Queue {
    /// Blocks until there is work, highest lane first, or the pool stops
    /// with nothing left (`None`).
    fn take(&self) -> Option<Work> {
        let mut state = self.state.lock().expect("job queue");
        loop {
            if let Some((_, task)) = state.computes.pop() {
                return Some(Work::Compute(task));
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
    threads: Vec<JoinHandle<()>>,
    stats: Arc<WorkerStats>,
    size: usize,
}

/// Everything the pool's threads share.
struct Shared {
    runner: ModuleRunner,
    blobs: Arc<dyn Blobs>,
    cache: Option<Arc<DiskCache>>,
    uploader: Arc<dyn Uploader>,
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
                                Work::Compute(task) => run_task(&shared, task),
                                Work::Lower(work) => work(),
                            }
                        }
                    })
                    .expect("spawning a worker thread")
            })
            .collect();
        Ok(WorkerPool {
            queue,
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
    let Task { job, done } = task;
    let stats = shared.runner.stats();
    let result = shared
        .runner
        .run(&job, shared.blobs.as_ref(), shared.cache.as_deref());
    stats.job_finished(result.is_ok());
    match result {
        Ok(computed) => {
            let section = Arc::clone(&computed.bytes);
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

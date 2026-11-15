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
//! 4. runs `gx_compile` and validates the section with the library
//!    validator,
//! 5. hands the bytes to the caller at once, stores them in the local cache
//!    under their address, and starts the upload with the claim
//!    ([`Uploader`]), which never holds up the next job.
//!
//! The worker sees modules, requests, tile bytes, and section bytes, and
//! nothing else.

use crate::cache::DiskCache;
use crate::job::{
    check_blob, check_section, check_tile, decode_manifest, resolve_tiles, BlobKind, ComputeJob,
    WorkerError,
};
use crate::stats::{UploadTicket, WorkerSnapshot, WorkerStats};
use gx_core::sha256::to_hex;
use gx_module_host::{HostConfig, HostError, LoadedModule, ModuleHost};
use std::collections::BTreeMap;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

/// Where the worker gets content-addressed bytes it does not have cached:
/// `GET /{kind}/{hash}` through the edge, in the client. Called from the
/// pool's threads; may block.
pub trait Blobs: Send + Sync {
    /// The bytes named `hash` of `kind`. The worker checks them.
    fn fetch(&self, kind: BlobKind, hash: &[u8; 32]) -> Result<Vec<u8>, WorkerError>;
}

/// Uploads a computed section with its claim (`PUT
/// /space/{s}/build/{b}/section/{address}` with `X-Claim`). Must not block:
/// start the upload and return. Finish the ticket with the outcome; log
/// failures, which never block drawing.
pub trait Uploader: Send + Sync {
    /// Starts the upload of `section` for `job`.
    fn upload(&self, job: &ComputeJob, section: Arc<[u8]>, ticket: UploadTicket);
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

/// The module host and every module admitted this session, by hash.
pub struct ModuleRunner {
    host: ModuleHost,
    loaded: Mutex<BTreeMap<[u8; 32], Arc<LoadedModule>>>,
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
            stats,
            flights: SingleFlight::default(),
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
    ) -> Result<Vec<u8>, WorkerError> {
        job.check_address()?;
        let module = self.module(&job.module, || {
            get_blob(BlobKind::Module, &job.module, blobs, cache, &self.flights)
        })?;
        self.stats.module_run();
        let ids = module.sources(&job.request).map_err(host_error)?;
        let manifest = match &job.manifest {
            Some(h) if !ids.is_empty() => Some(decode_manifest(&get_blob(
                BlobKind::Manifest,
                h,
                blobs,
                cache,
                &self.flights,
            )?)?),
            _ => None,
        };
        let refs = resolve_tiles(manifest.as_ref(), &ids)?;
        let mut tiles = Vec::with_capacity(refs.len());
        for r in &refs {
            let bytes = get_blob(BlobKind::Source, &r.sha256, blobs, cache, &self.flights)?;
            check_tile(r, &bytes)?;
            tiles.push((r.id, bytes));
        }
        self.stats.module_run();
        let section = module.compile(&job.request, &tiles).map_err(host_error)?;
        check_section(&job.request.key, &section)?;
        Ok(section)
    }
}

/// What the caller is told when a job ends.
pub type JobResult = Result<Arc<[u8]>, WorkerError>;

struct Task {
    job: ComputeJob,
    done: Box<dyn FnOnce(JobResult) + Send>,
}

/// The pool of worker threads.
pub struct WorkerPool {
    tx: Option<mpsc::Sender<Task>>,
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
    /// uploading through `uploader`.
    pub fn new(
        threads: usize,
        config: HostConfig,
        blobs: Arc<dyn Blobs>,
        cache: Option<Arc<DiskCache>>,
        uploader: Arc<dyn Uploader>,
    ) -> Result<WorkerPool, WorkerError> {
        let stats = WorkerStats::new();
        let shared = Arc::new(Shared {
            runner: ModuleRunner::new(config, Arc::clone(&stats))?,
            blobs,
            cache,
            uploader,
        });
        let (tx, rx) = mpsc::channel::<Task>();
        let rx = Arc::new(Mutex::new(rx));
        let size = threads.max(1);
        let threads = (0..size)
            .map(|i| {
                let rx = Arc::clone(&rx);
                let shared = Arc::clone(&shared);
                std::thread::Builder::new()
                    .name(format!("gx-worker-{i}"))
                    .spawn(move || loop {
                        let task = { rx.lock().expect("job queue").recv() };
                        match task {
                            Ok(task) => run_task(&shared, task),
                            Err(_) => break,
                        }
                    })
                    .expect("spawning a worker thread")
            })
            .collect();
        Ok(WorkerPool {
            tx: Some(tx),
            threads,
            stats,
            size,
        })
    }

    /// Number of worker threads.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Queues `job`; `done` is called on a worker thread with the section
    /// (before its upload starts) or the failure.
    pub fn submit(&self, job: ComputeJob, done: impl FnOnce(JobResult) + Send + 'static) {
        let task = Task {
            job,
            done: Box::new(done),
        };
        if let Some(tx) = &self.tx {
            if let Err(mpsc::SendError(task)) = tx.send(task) {
                (task.done)(Err(WorkerError::transient("the worker pool has stopped")));
            }
        }
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
    fn drop(&mut self) {
        self.tx = None;
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
        Ok(section) => {
            let section: Arc<[u8]> = Arc::from(section);
            done(Ok(Arc::clone(&section)));
            let address = job.address_hex();
            if let Some(c) = &shared.cache {
                if let Err(e) = c.put(BlobKind::Section, &address, &section) {
                    tracing::warn!(address, "could not cache the section: {e}");
                }
            }
            let ticket = stats.upload_started();
            shared.uploader.upload(&job, section, ticket);
        }
        Err(e) => {
            tracing::warn!(key = %job.key(), address = job.address_hex(), "module job failed: {e}");
            done(Err(e));
        }
    }
}

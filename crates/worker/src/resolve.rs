//! Input resolution, written once for both builds (`client-compute.md`
//! sections 2.8, 10.3, and 10.4; `rendering.md` section 6 rule 5).
//!
//! For one job [`InputResolver::compute`] runs `gx_inputs` with the job's
//! frame records, then gets every input section it names: this session's
//! memory, else the cache, else directions for the input's key, then a
//! stream (`ready`: the bytes are checked against the hash the directions
//! carried and validated as a section) or a compute (`missing`, `local`),
//! which resolves its own inputs the same way. The inputs are passed to
//! `gx_compile` in `gx_inputs` order, the section carries their
//! `X-Inputs-Sha256` (section 10.4), and a computed `missing` input is
//! offered for upload.
//!
//! Work on one address runs once ([`Flights`]): whoever takes an address
//! first streams or computes it while everyone else that needs it (another
//! compute's input, the wanted set's own compute, or the renderer's own
//! stream of a `ready` section) waits and then takes the same bytes from
//! memory. A waiter raises the priority of the work it waits on to the
//! highest of its waiters where the platform has a queue
//! ([`Platform::raise`]). Flights are taken along the layers' input order,
//! which has no cycle, so the waits cannot deadlock.
//!
//! The code is async over a small [`Platform`] trait: the native worker
//! drives it on its pool threads with [`block_on`] (its platform calls
//! block), the browser on its event loop with `fetch` and its Web Worker
//! module host.

use crate::job::{check_section, ComputeJob, Computed, WorkerError};
use gx_core::address::SlotKind;
use gx_core::key::ChunkKey;
use gx_core::sha256::to_hex;
use std::collections::{BTreeMap, VecDeque};
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};

/// What directions say about one input section (`client-compute.md`
/// section 2.4), as [`Platform::direct`] reports it.
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

/// What a module job's section is made from and where input sections come
/// from, on one platform. Everything that differs between the native and
/// the browser worker is here; the algorithm is [`InputResolver`].
#[allow(async_fn_in_trait)]
pub trait Platform {
    /// The job that makes the section `key` of `reader`'s layer slot
    /// `slot`, at the input span (section 10.3), with its address, slots,
    /// and frame records filled in from the build. Its `share` is ignored:
    /// [`Platform::direct`] decides it. Fails for a slot that is not a
    /// layer slot of the reader's layer, or where nothing resolves inputs.
    fn input_job(
        &self,
        reader: &ComputeJob,
        slot: u32,
        key: ChunkKey,
    ) -> Result<ComputeJob, WorkerError>;

    /// Directions for `job`'s layer and key at `job`'s span. Fails when the
    /// answer's address is not `job`'s.
    async fn direct(&self, job: &ComputeJob) -> Result<InputDirection, WorkerError>;

    /// The stored section at `address` (`GET /section/{address}?sha256=`).
    /// The resolver checks the hash and validates the section.
    async fn stream(&self, address: &[u8; 32], sha256: &[u8; 32]) -> Result<Vec<u8>, WorkerError>;

    /// Gets `job`'s module and runs `gx_inputs` with the job's frame
    /// records (section 4.1): the `(slot, key)` list, empty for a version
    /// 1 module.
    async fn input_list(&self, job: &ComputeJob) -> Result<Vec<(u32, ChunkKey)>, WorkerError>;

    /// Runs `gx_sources`, gets the tiles, runs `gx_compile` with `inputs`
    /// (in slot order, already resolved), and validates the section. The
    /// result's `inputs_sha256` is set by the resolver.
    async fn compile(
        &self,
        job: &ComputeJob,
        inputs: &[Arc<[u8]>],
    ) -> Result<Computed, WorkerError>;

    /// The cached section at `address`, if any. The resolver validates it.
    async fn cache_get(&self, address: &[u8; 32]) -> Option<Vec<u8>>;

    /// Caches the section at `address`.
    async fn cache_put(&self, address: &[u8; 32], bytes: &Arc<[u8]>);

    /// Offers a computed `missing` input for upload, once. Must not wait
    /// for the upload.
    fn offer(&self, job: &ComputeJob, bytes: Arc<[u8]>);

    /// An input was computed (counted like any compute).
    fn input_computed(&self) {}

    /// Someone of `priority` waits on work for `key` at `address`: where
    /// the platform has a queue, queued work for it starts no later than
    /// work of that priority. Called with the highest waiter's priority.
    fn raise(&self, key: &ChunkKey, address: &[u8; 32], priority: f64) {
        let _ = (key, address, priority);
    }
}

/// Sections resolved as inputs, or computed, kept in memory
/// ([`InputResolver`]).
pub const RESOLVED_MEMORY_SECTIONS: usize = 256;

/// Sections this session resolved or computed, by address, at most
/// [`RESOLVED_MEMORY_SECTIONS`]; the oldest go first.
#[derive(Default)]
struct ResolvedMemory {
    sections: BTreeMap<[u8; 32], Arc<[u8]>>,
    order: VecDeque<[u8; 32]>,
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

#[derive(Default)]
struct Flight {
    waiters: Vec<Waker>,
    top: Option<f64>,
}

/// One flight per address: an async lock per address being streamed or
/// computed. Works on one thread (the browser) and across threads (the
/// native pool, with [`block_on`]'s waker).
#[derive(Default)]
pub struct Flights {
    held: Mutex<BTreeMap<[u8; 32], Flight>>,
}

/// Holds one address's flight until dropped.
pub struct FlightGuard<'a> {
    flights: &'a Flights,
    address: [u8; 32],
}

impl Drop for FlightGuard<'_> {
    fn drop(&mut self) {
        let flight = self
            .flights
            .held
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .remove(&self.address);
        for w in flight.into_iter().flat_map(|f| f.waiters) {
            w.wake();
        }
    }
}

struct Acquire<'a, F> {
    flights: &'a Flights,
    address: [u8; 32],
    priority: f64,
    raise: F,
}

impl<'a, F: Fn(f64) + Unpin> Future for Acquire<'a, F> {
    type Output = FlightGuard<'a>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<FlightGuard<'a>> {
        let mut held = self.flights.held.lock().unwrap_or_else(|p| p.into_inner());
        let Some(flight) = held.get_mut(&self.address) else {
            held.insert(self.address, Flight::default());
            return Poll::Ready(FlightGuard {
                flights: self.flights,
                address: self.address,
            });
        };
        if !flight.waiters.iter().any(|w| w.will_wake(cx.waker())) {
            flight.waiters.push(cx.waker().clone());
        }
        let raised = flight.top.is_none_or(|t| self.priority > t);
        if raised {
            flight.top = Some(self.priority);
        }
        drop(held);
        if raised {
            (self.raise)(self.priority);
        }
        Poll::Pending
    }
}

impl Flights {
    /// Takes `address`'s flight, waiting while someone else holds it.
    /// `raise` is told the waiter's priority when it is the highest yet.
    pub async fn acquire<'a>(
        &'a self,
        address: &[u8; 32],
        priority: f64,
        raise: impl Fn(f64) + Unpin,
    ) -> FlightGuard<'a> {
        Acquire {
            flights: self,
            address: *address,
            priority,
            raise,
        }
        .await
    }
}

/// The input resolver: this session's memory of resolved sections and the
/// flights, shared by every job on a platform.
#[derive(Default)]
pub struct InputResolver {
    memory: Mutex<ResolvedMemory>,
    flights: Flights,
}

/// What [`InputResolver::run_wanted`] hands back: the section and whether
/// it was computed now (`true`) or taken from this session's memory
/// (`false`: already cached and offered by whoever computed it).
pub type Wanted = (Computed, bool);

impl InputResolver {
    /// An empty resolver.
    pub fn new() -> InputResolver {
        InputResolver::default()
    }

    /// The flights, for a platform's own work on an address (the
    /// renderer's stream of a `ready` section), so it shares the flight
    /// with input work on the same address.
    pub fn flights(&self) -> &Flights {
        &self.flights
    }

    /// The section at `address` if this session resolved or computed it.
    pub fn remembered(&self, address: &[u8; 32]) -> Option<Arc<[u8]>> {
        self.memory
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .get(address)
    }

    /// Keeps `bytes`, already checked, as the section at `address`.
    pub fn remember(&self, address: &[u8; 32], bytes: Arc<[u8]>) {
        self.memory
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .put(*address, bytes);
    }

    /// Runs one job to a validated section: `gx_inputs`, every input
    /// resolved (section 10.3), then the platform's compile. `priority` is
    /// the job's, passed to the input work it waits on.
    pub async fn compute<P: Platform>(
        &self,
        p: &P,
        job: &ComputeJob,
        priority: f64,
    ) -> Result<Computed, WorkerError> {
        job.check_address()?;
        let list = p.input_list(job).await?;
        let passed = self.gather(p, job, &list, priority).await?;
        let mut computed = p.compile(job, &passed).await?;
        let refs: Vec<&[u8]> = passed.iter().map(|b| b.as_ref()).collect();
        computed.inputs_sha256 =
            (!job.slots.is_empty()).then(|| gx_core::module_io::inputs_sha256(&refs));
        Ok(computed)
    }

    /// [`InputResolver::compute`] for a wanted compute: one flight per
    /// address, so a section already resolved as another job's input is
    /// handed over and one being resolved now is waited for.
    pub async fn run_wanted<P: Platform>(
        &self,
        p: &P,
        job: &ComputeJob,
        priority: f64,
    ) -> Result<Wanted, WorkerError> {
        let key = job.key();
        let address = job.address;
        let raise = |w: f64| p.raise(&key, &address, w);
        let _flight = self.flights.acquire(&address, priority, &raise).await;
        if let Some(bytes) = self.remembered(&address) {
            let section = check_section(&key, &bytes)?;
            return Ok((
                Computed {
                    bytes,
                    section: section.map(Arc::new),
                    inputs_sha256: None,
                },
                false,
            ));
        }
        let computed = self.compute(p, job, priority).await?;
        self.remember(&address, Arc::clone(&computed.bytes));
        Ok((computed, true))
    }

    /// The inputs `gx_compile` receives (section 4.1), in slot order: a
    /// frames slot's records from the job, a layer slot's sections in the
    /// order `list` names them, each resolved (section 10.3).
    async fn gather<P: Platform>(
        &self,
        p: &P,
        job: &ComputeJob,
        list: &[(u32, ChunkKey)],
        priority: f64,
    ) -> Result<Vec<Arc<[u8]>>, WorkerError> {
        let slots = job.slots.len();
        if let Some((slot, key)) = list.iter().find(|(s, _)| {
            job.slots
                .get(*s as usize)
                .is_none_or(|v| v.kind != SlotKind::Layer)
        }) {
            return Err(WorkerError::permanent(format!(
                "the module names input {key} of slot {slot}, which is not a layer slot of the layer"
            )));
        }
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
                        out.push(self.input(p, job, slot as u32, *key, priority).await?);
                    }
                }
            }
        }
        if slots == 0 && !job.frames.is_empty() {
            // A job without slots passes its frame records as given.
            out.extend(job.frames.iter().cloned());
        }
        Ok(out)
    }

    /// One input section of `reader` (section 10.3): this session's
    /// memory, else the cache, else directions and a stream or a compute.
    async fn input<P: Platform>(
        &self,
        p: &P,
        reader: &ComputeJob,
        slot: u32,
        key: ChunkKey,
        priority: f64,
    ) -> Result<Arc<[u8]>, WorkerError> {
        let mut job = p.input_job(reader, slot, key)?;
        let address = job.address;
        if let Some(b) = self.known(p, &key, &address).await {
            return Ok(b);
        }
        let raise = |w: f64| p.raise(&key, &address, w);
        let _flight = self.flights.acquire(&address, priority, &raise).await;
        if let Some(b) = self.known(p, &key, &address).await {
            return Ok(b);
        }
        let bytes: Arc<[u8]> = match p.direct(&job).await? {
            InputDirection::Empty => Arc::from(Vec::new()),
            InputDirection::Ready { sha256 } => {
                let bytes = p.stream(&address, &sha256).await?;
                if gx_core::sha256::sha256(&bytes) != sha256 {
                    return Err(WorkerError::transient(format!(
                        "input section {} does not hash to {}",
                        to_hex(&address),
                        to_hex(&sha256)
                    )));
                }
                check_section(&key, &bytes)?;
                let bytes: Arc<[u8]> = Arc::from(bytes);
                self.keep(p, &address, &bytes).await;
                bytes
            }
            InputDirection::Compute { share } => {
                job.share = share;
                let computed = Box::pin(self.compute(p, &job, priority)).await?;
                p.input_computed();
                self.keep(p, &address, &computed.bytes).await;
                job.inputs_sha256 = computed.inputs_sha256;
                if share {
                    p.offer(&job, Arc::clone(&computed.bytes));
                }
                computed.bytes
            }
        };
        Ok(bytes)
    }

    /// The section at `address` if this session resolved it or the cache
    /// holds a valid copy.
    async fn known<P: Platform>(
        &self,
        p: &P,
        key: &ChunkKey,
        address: &[u8; 32],
    ) -> Option<Arc<[u8]>> {
        if let Some(b) = self.remembered(address) {
            return Some(b);
        }
        let bytes = p.cache_get(address).await?;
        check_section(key, &bytes).ok()?;
        let bytes: Arc<[u8]> = Arc::from(bytes);
        self.remember(address, Arc::clone(&bytes));
        Some(bytes)
    }

    /// Keeps a resolved or computed section in memory and in the cache.
    async fn keep<P: Platform>(&self, p: &P, address: &[u8; 32], bytes: &Arc<[u8]>) {
        self.remember(address, Arc::clone(bytes));
        p.cache_put(address, bytes).await;
    }
}

/// Runs `f` to completion on this thread, parking while it waits: the
/// native pool's executor. Its platform calls block, so the only waits are
/// on another thread's flight, which wakes this one.
#[cfg(not(target_arch = "wasm32"))]
pub fn block_on<F: Future>(f: F) -> F::Output {
    struct Unpark(std::thread::Thread);
    impl std::task::Wake for Unpark {
        fn wake(self: Arc<Self>) {
            self.0.unpark();
        }
    }
    let waker = Waker::from(Arc::new(Unpark(std::thread::current())));
    let mut cx = Context::from_waker(&waker);
    let mut f = std::pin::pin!(f);
    loop {
        if let Poll::Ready(v) = f.as_mut().poll(&mut cx) {
            return v;
        }
        std::thread::park();
    }
}

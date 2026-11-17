//! The browser's fetcher: the client loop of `client-compute.md` section 1
//! on the page's event loop.
//!
//! Drives one [`crate::directions::Coordinator`], exactly as the native
//! [`crate::stream::Fetcher`] does, and carries out its actions with
//! futures on the page: directions through [`WebHub`], sections from the
//! Cache API ([`WebCache`]) or the edge, `compute` jobs on the Web Worker
//! ([`ModuleWorker`], one at a time so a queued call never runs into
//! another's time limit), and readiness notices from the socket, opened
//! again every [`SOCKET_RETRY_SECONDS`] when it closes. A computed section
//! is handed over at once; its upload runs in the background, and a failed
//! upload is logged and never blocks drawing.
//!
//! Every section is decoded once on its way in (`rendering.md` 3.5): a
//! streamed one when it arrives, a computed one by the worker's check.
//! Compositing a finished cell and deriving what the renderer keeps of it
//! is time-sliced: [`WebFetcher::drain_until`] does it only until the
//! frame's deadline and leaves the rest for the next frame
//! (`rendering.md` section 4).

use super::cache::WebCache;
use super::hub::{now_seconds, sleep, ReadySocket, WebHub};
use super::worker::{run_job, ModuleWorker};
use crate::directions::{Action, Arrived, Completion, Coordinator, Origin, StreamResult};
use crate::protocol::DirectionsResponse;
use crate::stream::{
    cell_from_arrived, directions_time, FetchEvent, FetchOutcome, FetchTally, FetcherStats,
};
use anyhow::{anyhow, Result};
use gx_core::frames::FrameSystem;
use gx_core::key::{CellKey, ChunkKey};
use gx_core::sha256::to_hex;
use gx_core::units::Meters;
use gx_worker::{BlobKind, ComputeJob, Computed, WorkerError, WorkerStats};
use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, VecDeque};
use std::rc::Rc;
use std::sync::Arc;
use std::time::Duration;

/// Seconds between two attempts to open a closed readiness socket.
pub const SOCKET_RETRY_SECONDS: f64 = 5.0;

/// Seconds the registry may take to arrive.
pub const REGISTRY_DEADLINE_SECONDS: f64 = 120.0;

enum Internal {
    Directions(Vec<ChunkKey>, Result<DirectionsResponse, String>),
    Stream {
        key: ChunkKey,
        layer: usize,
        address: [u8; 32],
        result: StreamResult,
    },
    Compute {
        key: ChunkKey,
        layer: usize,
        address: [u8; 32],
        result: Result<Computed, WorkerError>,
    },
}

/// Fetches cells in the browser.
pub struct WebFetcher {
    hub: Rc<WebHub>,
    cache: Option<Rc<WebCache>>,
    worker: ModuleWorker,
    stats: Arc<WorkerStats>,
    coordinator: Coordinator,
    socket: Option<ReadySocket>,
    socket_tried_at: f64,
    inbox: Rc<RefCell<VecDeque<Internal>>>,
    jobs: VecDeque<(ChunkKey, usize, ComputeJob)>,
    job_running: Rc<Cell<bool>>,
    extents: BTreeMap<CellKey, Meters>,
    completed: VecDeque<(CellKey, Completion, Meters, Duration)>,
    out: Vec<FetchEvent>,
    started: f64,
    wanted_utc_ms: Option<i64>,
    tally: FetchTally,
}

impl WebFetcher {
    /// A fetcher over `hub`, caching in `cache` when the page has one.
    /// Keeps the hub's clock measured ([`WebHub::keep_clock`]).
    pub fn new(hub: Rc<WebHub>, cache: Option<WebCache>) -> WebFetcher {
        let socket = ReadySocket::open(&hub.routes().ready(hub.build_id()));
        hub.keep_clock();
        WebFetcher {
            hub,
            cache: cache.map(Rc::new),
            worker: ModuleWorker::new(),
            stats: WorkerStats::new(),
            coordinator: Coordinator::new(),
            socket,
            socket_tried_at: now_seconds(),
            inbox: Rc::default(),
            jobs: VecDeque::new(),
            job_running: Rc::new(Cell::new(false)),
            extents: BTreeMap::new(),
            completed: VecDeque::new(),
            out: Vec::new(),
            started: now_seconds(),
            wanted_utc_ms: None,
            tally: FetchTally::default(),
        }
    }

    fn now(&self) -> f64 {
        now_seconds() - self.started
    }

    /// The hub connection.
    pub fn hub(&self) -> &Rc<WebHub> {
        &self.hub
    }

    /// The hub's now, UTC milliseconds ([`WebHub::now_utc_ms`]).
    pub fn now_utc_ms(&self) -> i64 {
        self.hub.now_utc_ms()
    }

    /// Asks for directions at `t` (UTC milliseconds) from now on, clamped
    /// to the hub's now and the space's origin.
    pub fn set_time_utc_ms(&mut self, t: i64) {
        self.wanted_utc_ms = Some(t);
    }

    /// Starts a request for `key`, a cell of a frame of edge `root_extent`.
    pub fn request(&mut self, key: CellKey, root_extent: Meters) {
        self.extents.insert(key, root_extent);
        let now = self.now();
        self.coordinator.request(ChunkKey::Cell(key), now);
    }

    /// Starts the request for the registry key.
    pub fn request_registry(&mut self) {
        let now = self.now();
        self.coordinator.request(ChunkKey::Registry, now);
        self.pump();
    }

    fn keep_socket(&mut self) {
        let closed = self.socket.as_ref().is_none_or(ReadySocket::is_closed);
        let now = now_seconds();
        if !closed || now - self.socket_tried_at < SOCKET_RETRY_SECONDS {
            return;
        }
        self.socket_tried_at = now;
        self.socket = ReadySocket::open(&self.hub.routes().ready(self.hub.build_id()));
        if let Some(s) = &self.socket {
            for a in self.coordinator.waiting_addresses() {
                s.subscribe(&a);
            }
        }
    }

    fn apply(&mut self, msg: Internal) -> Vec<Action> {
        let now = self.now();
        match msg {
            Internal::Directions(keys, r) => self.coordinator.directions_done(&keys, r, now),
            Internal::Stream {
                key,
                layer,
                address,
                result,
            } => self
                .coordinator
                .stream_done(&key, layer, address, result, now),
            Internal::Compute {
                key,
                layer,
                address,
                result,
            } => self
                .coordinator
                .compute_done(&key, layer, address, result, now),
        }
    }

    fn pump(&mut self) {
        self.keep_socket();
        let mut actions = Vec::new();
        let inbox: Vec<Internal> = self.inbox.borrow_mut().drain(..).collect();
        for msg in inbox {
            actions.extend(self.apply(msg));
        }
        if let Some(s) = &self.socket {
            for a in s.take_ready() {
                actions.extend(self.coordinator.ready_notice(&a));
            }
        }
        actions.extend(self.coordinator.poll(self.now()));
        while !actions.is_empty() {
            for a in std::mem::take(&mut actions) {
                self.execute(a);
            }
            actions = self.coordinator.poll(self.now());
        }
        self.start_job();
        if let Some(c) = &self.cache {
            let c = c.clone();
            wasm_bindgen_futures::spawn_local(async move { c.flush_index().await });
        }
    }

    /// Starts the next queued job if none is running.
    fn start_job(&mut self) {
        if self.job_running.get() {
            return;
        }
        let Some((key, layer, job)) = self.jobs.pop_front() else {
            return;
        };
        self.job_running.set(true);
        let (hub, cache, worker, stats) = (
            self.hub.clone(),
            self.cache.clone(),
            self.worker.clone(),
            self.stats.clone(),
        );
        let (inbox, running) = (self.inbox.clone(), self.job_running.clone());
        wasm_bindgen_futures::spawn_local(async move {
            let address = job.address;
            let result = run_job(&job, &hub, cache.as_deref(), &worker, &stats).await;
            stats.job_finished(result.is_ok());
            if let Err(e) = &result {
                web_sys::console::warn_1(
                    &format!("gx-renderer: module job for {key} failed: {e}").into(),
                );
            }
            inbox.borrow_mut().push_back(Internal::Compute {
                key,
                layer,
                address,
                result: result.clone(),
            });
            running.set(false);
            let Ok(Computed { bytes: section, .. }) = result else {
                return;
            };
            if let Some(c) = &cache {
                let _ = c.put(BlobKind::Section, &job.address_hex(), &section).await;
            }
            let ticket = stats.upload_started();
            match hub.upload(&job.address, &job.claim, section.to_vec()).await {
                Ok(_) => ticket.finish(true),
                Err(e) => {
                    web_sys::console::warn_1(
                        &format!("gx-renderer: upload for {key} failed: {e:#}").into(),
                    );
                    ticket.finish(false);
                }
            }
        });
    }

    fn execute(&mut self, action: Action) {
        match action {
            Action::Directions(keys) => {
                let t = directions_time(
                    self.wanted_utc_ms,
                    self.coordinator.origin_utc_ms(),
                    self.hub.now_utc_ms(),
                );
                let (hub, inbox) = (self.hub.clone(), self.inbox.clone());
                wasm_bindgen_futures::spawn_local(async move {
                    let r = hub.directions(t, &keys).await.map_err(|e| format!("{e:#}"));
                    inbox.borrow_mut().push_back(Internal::Directions(keys, r));
                });
            }
            Action::Stream {
                key,
                layer,
                address,
            } => {
                let (hub, cache, inbox) =
                    (self.hub.clone(), self.cache.clone(), self.inbox.clone());
                wasm_bindgen_futures::spawn_local(async move {
                    let result = stream_section(&hub, cache.as_deref(), key, address).await;
                    inbox.borrow_mut().push_back(Internal::Stream {
                        key,
                        layer,
                        address,
                        result,
                    });
                });
            }
            Action::Compute { key, layer, job } => self.jobs.push_back((key, layer, job)),
            Action::Subscribe(address) => {
                if let Some(s) = &self.socket {
                    s.subscribe(&address);
                }
            }
            Action::Complete(done) => match done.key {
                ChunkKey::Registry => self.out.push(FetchEvent::Registry(Ok(done
                    .sections
                    .into_iter()
                    .map(|a| a.bytes)
                    .collect()))),
                ChunkKey::Cell(key) => {
                    let Some(extent) = self.extents.remove(&key) else {
                        return;
                    };
                    let round_trip =
                        Duration::from_secs_f64((self.now() - done.requested_at).max(0.0));
                    self.completed.push_back((key, done, extent, round_trip));
                }
            },
            Action::Failed { key, reason } => match key {
                ChunkKey::Registry => self.out.push(FetchEvent::Registry(Err(reason))),
                ChunkKey::Cell(key) => {
                    self.extents.remove(&key);
                    self.out.push(FetchEvent::Completed {
                        key,
                        outcome: FetchOutcome::Invalid(reason),
                        bytes: 0,
                        round_trip: Duration::ZERO,
                    });
                }
            },
        }
    }

    /// Carries out everything due and returns every event since the last
    /// call, compositing every finished cell.
    pub fn drain(&mut self) -> Vec<FetchEvent> {
        self.drain_until(f64::INFINITY)
    }

    /// [`WebFetcher::drain`], compositing finished cells only until the
    /// clock ([`crate::clock::now_seconds`]) passes `deadline`; the rest
    /// wait for a later call.
    pub fn drain_until(&mut self, deadline: f64) -> Vec<FetchEvent> {
        self.pump();
        while let Some((key, done, extent, round_trip)) = self.completed.pop_front() {
            let bytes = done.sections.iter().map(|s| s.len() as u64).sum();
            let outcome = cell_from_arrived(&key, &done.sections, extent);
            self.out.push(FetchEvent::Completed {
                key,
                outcome,
                bytes,
                round_trip,
            });
            if crate::clock::now_seconds() > deadline {
                break;
            }
        }
        let events = std::mem::take(&mut self.out);
        self.tally.record(&events);
        events
    }

    /// The counters for the overlay.
    pub fn stats(&self) -> FetcherStats {
        FetcherStats {
            tally: self.tally,
            directions: self.coordinator.stats(),
            worker: self.stats.snapshot(),
        }
    }

    /// Asks for the registry key and waits for every layer's registry,
    /// then builds the frame system from their union.
    pub async fn load_registry(&mut self) -> Result<FrameSystem> {
        self.request_registry();
        let start = now_seconds();
        loop {
            for e in self.drain() {
                if let FetchEvent::Registry(r) = e {
                    let sections = r.map_err(|e| anyhow!("registry: {e}"))?;
                    return crate::protocol::frame_system_from_registries(
                        &sections,
                        self.hub.build(),
                    );
                }
            }
            if now_seconds() - start > REGISTRY_DEADLINE_SECONDS {
                return Err(anyhow!(
                    "the registry did not arrive within {REGISTRY_DEADLINE_SECONDS} s"
                ));
            }
            sleep(0.05).await;
        }
    }
}

/// Decodes a cell's section, the one validation it gets on its way in
/// (`rendering.md` 3.5); the registry key passes through as bytes.
fn decode_arrived(key: ChunkKey, bytes: Vec<u8>) -> Result<Arrived, String> {
    let bytes: Arc<[u8]> = Arc::from(bytes);
    match key {
        ChunkKey::Cell(cell) => gx_core::matter::decode(&cell, &bytes)
            .map(|s| Arrived {
                bytes: bytes.clone(),
                decoded: Some(Arc::new(s)),
            })
            .map_err(|e| e.to_string()),
        ChunkKey::Registry => Ok(Arrived::raw(bytes)),
    }
}

/// Streams the section at `address`: the cache first (decoded, dropped if
/// damaged), then the edge (decoded, then cached).
async fn stream_section(
    hub: &WebHub,
    cache: Option<&WebCache>,
    key: ChunkKey,
    address: [u8; 32],
) -> StreamResult {
    let name = to_hex(&address);
    if let Some(c) = cache {
        if let Some(bytes) = c.get(BlobKind::Section, &name).await {
            match decode_arrived(key, bytes) {
                Ok(a) => return StreamResult::Bytes(a, Origin::Cache),
                Err(_) => c.remove(BlobKind::Section, &name).await,
            }
        }
    }
    match hub.get_blob(BlobKind::Section, &name).await {
        Ok(Some(bytes)) => {
            let arrived = match decode_arrived(key, bytes) {
                Ok(a) => a,
                Err(e) => {
                    return StreamResult::Failed(format!(
                        "section {name} from the edge is invalid: {e}"
                    ))
                }
            };
            if let Some(c) = cache {
                if let Err(e) = c.put(BlobKind::Section, &name, &arrived.bytes).await {
                    web_sys::console::warn_1(
                        &format!("gx-renderer: caching section {name}: {e}").into(),
                    );
                }
            }
            StreamResult::Bytes(arrived, Origin::Edge)
        }
        Ok(None) => StreamResult::NotFound,
        Err(e) => StreamResult::Failed(format!("{e:#}")),
    }
}

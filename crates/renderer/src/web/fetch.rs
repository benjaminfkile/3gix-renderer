//! The browser's fetcher: the client loop of `client-compute.md` section 1
//! on the page's event loop.
//!
//! Drives one [`crate::directions::Coordinator`], exactly as the native
//! [`crate::stream::Fetcher`] does, and carries out its actions with
//! futures on the page: directions through [`WebHub`], sections from the
//! Cache API ([`WebCache`]) or the edge (`GET
//! /section/{address}?sha256={hex}`), checked against the hash the
//! directions carried (a cached copy that differs is discarded and fetched
//! again), and `missing` and `local` jobs on the Web Worker
//! ([`ModuleWorker`], one at a time so a queued call never runs into
//! another's time limit, the highest priority waiting job first, see
//! [`gx_worker::queue`]; a waiting job whose key left the wanted set is
//! dropped before it starts). A computed section is handed over at once. A
//! `missing` one then enters the upload lane ([`UploadLane`],
//! `client-compute.md` sections 2.5 and 2.8): bounded at 64, oldest
//! dropped when full, sent one at a time and once each, and only while no
//! compute waits, a failure counted
//! and never retried. A `local` section is never offered. Nothing waits
//! on another client, so there is no readiness socket.
//!
//! Every section is decoded once on its way in (`rendering.md` 3.5): a
//! streamed one when it arrives, a computed one by the worker's check.
//! Compositing a finished cell and deriving what the renderer keeps of it
//! is time-sliced: [`WebFetcher::drain_until`] does it only until the
//! frame's deadline and leaves the rest for the next frame
//! (`rendering.md` section 4).

use super::cache::WebCache;
use super::hub::{now_seconds, sleep, WebHub};
use super::worker::{run_job, ModuleWorker};
use crate::directions::{Action, Arrived, Completion, Coordinator, Origin, StreamResult};
use crate::protocol::DirectionsResponse;
use crate::stream::{
    cell_from_arrived, directions_time, FetchEvent, FetchOutcome, FetchTally, FetcherStats,
};
use anyhow::{anyhow, Result};
use gx_core::frames::FrameSystem;
use gx_core::key::{CellKey, ChunkKey};
use gx_core::sha256::{sha256, to_hex};
use gx_core::units::Meters;
use gx_worker::{
    BlobKind, CheckJob, CheckLane, CheckOutcome, ComputeJob, ComputeQueue, Computed, UploadLane,
    UploadOffer, UploadOutcome, WorkerError, WorkerStats,
};
use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, VecDeque};
use std::rc::Rc;
use std::sync::Arc;
use std::time::Duration;

/// Seconds the registry may take to arrive.
pub const REGISTRY_DEADLINE_SECONDS: f64 = 120.0;

enum Internal {
    Replace {
        key: ChunkKey,
        layer: usize,
        address: [u8; 32],
        sha256: [u8; 32],
        computed: Computed,
    },
    Directions(Vec<ChunkKey>, Result<DirectionsResponse, String>),
    Stream {
        key: ChunkKey,
        layer: usize,
        address: [u8; 32],
        sha256: [u8; 32],
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
    lane: Rc<UploadLane>,
    uploading: Rc<Cell<bool>>,
    checks: CheckLane,
    asked_at: BTreeMap<ChunkKey, i64>,
    checked_extents: BTreeMap<CellKey, Meters>,
    inbox: Rc<RefCell<VecDeque<Internal>>>,
    jobs: ComputeQueue<(usize, ComputeJob)>,
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
        hub.keep_clock();
        let stats = WorkerStats::new();
        WebFetcher {
            hub,
            cache: cache.map(Rc::new),
            worker: ModuleWorker::new(),
            lane: Rc::new(UploadLane::new(stats.clone())),
            uploading: Rc::new(Cell::new(false)),
            checks: CheckLane::new(stats.clone()),
            asked_at: BTreeMap::new(),
            checked_extents: BTreeMap::new(),
            stats,
            coordinator: {
                // Seeded once per page load, never from an address: it
                // only chooses which downloads to check.
                let mut c = Coordinator::new();
                c.set_check_seed((js_sys::Math::random() * (u64::MAX as f64)) as u64);
                c
            },
            inbox: Rc::default(),
            jobs: ComputeQueue::new(),
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

    /// Replaces the wanted cells (`client-compute.md` section 2.8) with
    /// `cells`, each a cell of a frame of edge `root_extent` with its
    /// priority (projected size, pixels). Directions and streams follow
    /// the wanted set, highest priority first; a cell left out that has
    /// nothing in flight is dropped.
    pub fn set_wanted_cells(&mut self, cells: Vec<(CellKey, Meters, f64)>) {
        let now = self.now();
        let mut wanted = Vec::with_capacity(cells.len());
        for (key, root_extent, priority) in cells {
            self.extents.insert(key, root_extent);
            wanted.push((key, priority));
        }
        self.coordinator.set_wanted_cells(wanted, now);
        self.follow_wanted();
    }

    /// Gives the waiting jobs their keys' new priorities and drops those
    /// whose key left the wanted set, reporting each as dropped.
    fn follow_wanted(&mut self) {
        let coordinator = &self.coordinator;
        let dropped = self.jobs.reprioritize(|k| coordinator.compute_priority(k));
        let mut inbox = self.inbox.borrow_mut();
        for (layer, job) in dropped {
            inbox.push_back(Internal::Compute {
                key: job.key(),
                layer,
                address: job.address,
                result: Err(WorkerError::dropped()),
            });
        }
    }

    /// Replaces the wanted surface tiles with `tiles`, each with its
    /// priority (projected size, pixels), like
    /// [`Self::set_wanted_cells`].
    pub fn set_wanted_surfaces(&mut self, tiles: Vec<(gx_core::key::SurfaceKey, f64)>) {
        let now = self.now();
        self.coordinator.set_wanted_surfaces(tiles, now);
        self.follow_wanted();
    }

    /// Queues a request for a surface tile; the result arrives as
    /// [`FetchEvent::Surface`] through [`WebFetcher::drain`].
    pub fn request_surface(&mut self, key: gx_core::key::SurfaceKey) {
        let now = self.now();
        self.coordinator.request(ChunkKey::Surface(key), now);
    }

    /// Starts a request for the registry key.
    pub fn request_registry(&mut self) {
        let now = self.now();
        self.coordinator.request(ChunkKey::Registry, now);
        self.pump();
    }

    fn apply(&mut self, msg: Internal) -> Vec<Action> {
        let now = self.now();
        match msg {
            Internal::Replace {
                key,
                layer,
                address,
                sha256,
                computed,
            } => {
                if let ChunkKey::Cell(cell) = key {
                    if let Some(e) = self.checked_extents.remove(&cell) {
                        self.extents.entry(cell).or_insert(e);
                    }
                }
                self.coordinator
                    .replace(key, layer, address, sha256, computed.into(), now)
            }
            Internal::Directions(keys, r) => self.coordinator.directions_done(&keys, r, now),
            Internal::Stream {
                key,
                layer,
                address,
                sha256,
                result,
            } => self
                .coordinator
                .stream_done(&key, layer, address, sha256, result, now),
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
        let mut actions = Vec::new();
        let inbox: Vec<Internal> = self.inbox.borrow_mut().drain(..).collect();
        for msg in inbox {
            actions.extend(self.apply(msg));
        }
        actions.extend(self.coordinator.poll(self.now()));
        while !actions.is_empty() {
            for a in std::mem::take(&mut actions) {
                self.execute(a);
            }
            actions = self.coordinator.poll(self.now());
        }
        self.start_job();
        self.start_upload();
        self.start_check();
        if let Some(c) = &self.cache {
            let c = c.clone();
            wasm_bindgen_futures::spawn_local(async move { c.flush_index().await });
        }
    }

    /// Starts the queued job of the highest priority if none is running.
    fn start_job(&mut self) {
        if self.job_running.get() {
            return;
        }
        let Some((key, (layer, job))) = self.jobs.pop() else {
            return;
        };
        self.job_running.set(true);
        let (hub, cache, worker, stats, lane) = (
            self.hub.clone(),
            self.cache.clone(),
            self.worker.clone(),
            self.stats.clone(),
            self.lane.clone(),
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
            if job.share {
                lane.offer(UploadOffer {
                    job,
                    bytes: section,
                });
            }
        });
    }

    /// Sends the upload lane's oldest offer if none is being sent and no
    /// wanted compute waits (`client-compute.md` section 2.8: uploads are
    /// below computes). Each offer is sent once; a failure is counted and
    /// dropped.
    fn start_upload(&mut self) {
        if self.uploading.get() || !self.jobs.is_empty() {
            return;
        }
        let Some((offer, ticket)) = self.lane.take() else {
            return;
        };
        self.uploading.set(true);
        let (hub, uploading) = (self.hub.clone(), self.uploading.clone());
        wasm_bindgen_futures::spawn_local(async move {
            let outcome = match hub.upload(&offer.job, offer.bytes.to_vec()).await {
                Ok(status) => UploadOutcome::from_status(status),
                Err(_) => UploadOutcome::Failed,
            };
            ticket.finish(outcome);
            uploading.set(false);
        });
    }

    /// Runs the check lane's oldest check on the Web Worker if it is idle
    /// and no wanted compute and no upload waits (`client-compute.md`
    /// section 2.8: checks are below uploads), then reports its hash. A
    /// recompute that differs from the drawn bytes takes their place,
    /// whatever the hub answers.
    fn start_check(&mut self) {
        if self.job_running.get() || !self.jobs.is_empty() || !self.lane.is_empty() {
            return;
        }
        let Some(check) = self.checks.take() else {
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
            let result = run_job(&check.job, &hub, cache.as_deref(), &worker, &stats).await;
            running.set(false);
            let Ok(computed) = result else {
                stats.check_skipped();
                return;
            };
            stats.check_run();
            let hash = sha256(&computed.bytes);
            if hash != check.drawn {
                inbox.borrow_mut().push_back(Internal::Replace {
                    key: check.job.key(),
                    layer: check.layer_index,
                    address: check.job.address,
                    sha256: check.drawn,
                    computed,
                });
            }
            let outcome = match hub.check(&check.job, &hash).await {
                Ok(status) => CheckOutcome::from_status(status),
                Err(_) => CheckOutcome::Failed,
            };
            stats.check_finished(outcome);
        });
    }

    fn execute(&mut self, action: Action) {
        if let Action::Complete(Completion { key, .. }) | Action::Failed { key, .. } = &action {
            self.asked_at.remove(key);
        }
        match action {
            Action::Directions(keys) => {
                let t = directions_time(
                    self.wanted_utc_ms,
                    self.coordinator.origin_utc_ms(),
                    self.hub.now_utc_ms(),
                );
                for k in &keys {
                    self.asked_at.insert(*k, t);
                }
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
                sha256,
            } => {
                let (hub, cache, inbox) =
                    (self.hub.clone(), self.cache.clone(), self.inbox.clone());
                wasm_bindgen_futures::spawn_local(async move {
                    let result = stream_section(&hub, cache.as_deref(), key, address, sha256).await;
                    inbox.borrow_mut().push_back(Internal::Stream {
                        key,
                        layer,
                        address,
                        sha256,
                        result,
                    });
                });
            }
            Action::Compute {
                key,
                layer,
                job,
                priority,
            } => self.jobs.push(key, priority, (layer, job)),
            Action::Check {
                key,
                layer,
                address,
                sha256,
            } => {
                let job = match (self.asked_at.get(&key), self.coordinator.origin_utc_ms()) {
                    (Some(&t), Some(origin)) => {
                        crate::protocol::check_job(self.hub.build(), layer, key, address, t, origin)
                    }
                    _ => None,
                };
                if let ChunkKey::Cell(cell) = key {
                    if let Some(e) = self.extents.get(&cell) {
                        self.checked_extents.insert(cell, *e);
                    }
                }
                match job {
                    Some(job) => {
                        self.checks.push(CheckJob {
                            job,
                            layer_index: layer,
                            drawn: sha256,
                        });
                    }
                    None => self.stats.check_skipped(),
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
                ChunkKey::Surface(key) => {
                    let bytes = done.sections.iter().map(|s| s.bytes.len() as u64).sum();
                    let sections: Vec<Arc<[u8]>> =
                        done.sections.iter().map(|a| a.bytes.clone()).collect();
                    self.out.push(FetchEvent::Surface {
                        key,
                        outcome: crate::tiles::decode_tile(&key, &sections),
                        bytes,
                    });
                }
            },
            Action::Failed { key, reason } => match key {
                ChunkKey::Registry => self.out.push(FetchEvent::Registry(Err(reason))),
                ChunkKey::Surface(key) => self.out.push(FetchEvent::Surface {
                    key,
                    outcome: Err(reason),
                    bytes: 0,
                }),
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
        ChunkKey::Registry | ChunkKey::Surface(_) => Ok(Arrived::raw(bytes)),
    }
}

/// Streams the version of the section at `address` whose bytes hash to
/// `expected`: the cache first (dropped if it is another version or
/// damaged), then the edge (checked, decoded, then cached).
async fn stream_section(
    hub: &WebHub,
    cache: Option<&WebCache>,
    key: ChunkKey,
    address: [u8; 32],
    expected: [u8; 32],
) -> StreamResult {
    let name = to_hex(&address);
    if let Some(c) = cache {
        if let Some(bytes) = c.get(BlobKind::Section, &name).await {
            if sha256(&bytes) != expected {
                c.remove(BlobKind::Section, &name).await;
            } else {
                match decode_arrived(key, bytes) {
                    Ok(a) => return StreamResult::Bytes(a, Origin::Cache),
                    Err(_) => c.remove(BlobKind::Section, &name).await,
                }
            }
        }
    }
    match hub.get_section(&address, &expected).await {
        Ok(Some(bytes)) if sha256(&bytes) != expected => StreamResult::Failed(format!(
            "section {name} from the edge does not hash to {}",
            to_hex(&expected)
        )),
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

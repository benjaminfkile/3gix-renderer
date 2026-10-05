//! Cell streaming: which cells to fetch, the cache of fetched cells, and
//! which cells to draw.
//!
//! Implements the renderer's steps "fetches matter cells near the camera at
//! a depth chosen by distance" and "composites overlapping sections from
//! different layers" of `space-model.md` section 2, over the octree cells of
//! section 5 and the chunk keys of section 7, with the compositing rule of
//! section 8 and `matter-format.md` section 3.5. The hub side of a fetch is
//! the `200` / `202` / `chunkReady` cycle of `compiler-pipeline.md`
//! section 4.
//!
//! # Selection
//!
//! Every 100 ms ([`SELECT_INTERVAL_SECONDS`]), every frame whose
//! `root_extent / 8` region is in the view or within `4 * root_extent` of
//! the camera is passed to [`gx_core::lod::select_cells`] with a pixel error
//! of 4, the viewport height, the vertical field of view, and at most 512
//! cells ([`select_all`]).
//!
//! # The cache
//!
//! [`CellCache`] holds one [`CellState`] per cell key. Selected cells start
//! `Missing`; at most 16 are `Requested` at a time, largest projected size
//! first; a `202` makes a cell `Pending` and the renderer subscribes to its
//! readiness; a `chunkReady` push (or, after 10 s, a poll every 2 s) asks
//! again; a `200` is decoded with [`gx_core::container::decode_chunk`] and
//! composited with [`gx_core::matter::composite`] into `Ready` or, when the
//! composite holds no matter, `Empty`. `410` and sections that fail
//! validation are `Gone`; `404` (the build has no layers for the key) is
//! `Empty`. Cells not selected for 60 s are evicted, and the cache never
//! holds more than 4096 cells it could drop.
//!
//! # Depth transitions
//!
//! [`CellCache::draw_set`] never leaves a hole. A selected cell that cannot
//! be drawn yet is covered by its loaded children if all of them can be
//! drawn, or else by its nearest drawable ancestor. A drawn ancestor hides
//! every drawn cell below it, so a parent keeps drawing while any of its
//! children is not ready, and stops once all of them are.

use crate::camera::{Camera, FIELD_OF_VIEW_Y};
use crate::light::cell_emitter;
use gx_core::container::decode_chunk;
use gx_core::emission::Emitter;
use gx_core::frames::FrameSystem;
use gx_core::key::CellKey;
use gx_core::lod::{select_cells, SelectionParams};
use gx_core::matter::{composite, Section};
use gx_core::units::{Meters, Vec3};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

/// Seconds between two selections.
pub const SELECT_INTERVAL_SECONDS: f64 = 0.1;

/// Largest projected cell edge, pixels, that is not refined further.
pub const PIXEL_ERROR: f64 = 4.0;

/// Most cells selected per frame.
pub const MAX_CELLS_PER_FRAME: usize = 512;

/// Most requests in flight at once.
pub const MAX_IN_FLIGHT: usize = 16;

/// Seconds a cell may go unselected before it is evicted.
pub const EVICT_AFTER_SECONDS: f64 = 60.0;

/// Most cells the cache keeps once eviction has dropped what it can.
pub const MAX_CACHED_CELLS: usize = 4096;

/// Seconds a cell stays `Pending` on the readiness push alone before the
/// cache also polls for it.
pub const PENDING_POLL_AFTER_SECONDS: f64 = 10.0;

/// Seconds between polls of a long `Pending` cell.
pub const PENDING_POLL_INTERVAL_SECONDS: f64 = 2.0;

/// Seconds before a failed request is tried again.
pub const RETRY_AFTER_SECONDS: f64 = 2.0;

/// How far below a selected cell [`CellCache::draw_set`] looks for loaded
/// descendants to cover it.
pub const COVER_LEVELS: u8 = 2;

/// One selected cell and its projected size.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SelectedCell {
    /// The cell.
    pub key: CellKey,
    /// Projected edge in pixels, the request priority (largest first).
    pub projected_px: f64,
}

/// A fetched and composited cell.
#[derive(Clone, Debug, PartialEq)]
pub struct ReadyCell {
    /// The composite of every layer's section for the cell.
    pub section: Arc<Section>,
    /// The cell's hot matter as one emitter, if any
    /// ([`crate::light::cell_emitter`]).
    pub emitter: Option<Emitter>,
}

/// Where a cell is in its fetch.
#[derive(Clone, Debug, PartialEq)]
pub enum CellState {
    /// Selected, not yet requested.
    Missing,
    /// A request is in flight.
    Requested,
    /// The hub answered `202`; waiting for readiness since this session
    /// time, seconds.
    Pending(f64),
    /// Fetched and composited.
    Ready(Arc<ReadyCell>),
    /// No matter: an empty composite or `404`. Draws nothing.
    Empty,
    /// `410`, or bytes that fail validation. Never drawn.
    Gone,
}

/// The result of one fetch, decoded.
#[derive(Clone, Debug, PartialEq)]
pub enum FetchOutcome {
    /// `200`, decoded and composited; `None` when there is no matter.
    Decoded(Option<Arc<ReadyCell>>),
    /// `202`: compilation is under way.
    Pending,
    /// `404`: the build has no layers for the key.
    NotFound,
    /// `410`: the key can never be produced.
    Gone,
    /// The bytes failed validation, with the reason.
    Invalid(String),
    /// Transport failure or an unexpected status; tried again later.
    Failed(String),
}

/// Decodes a chunk container for `key` (`matter-format.md` section 6),
/// checks each section's geometry against the registry's `root_extent`
/// (section 3.1), and composites the sections (section 3.5).
pub fn decode_cell(key: &CellKey, bytes: &[u8], root_extent: Meters) -> FetchOutcome {
    let sections = match decode_chunk(key, bytes) {
        Ok(s) => s,
        Err(e) => return FetchOutcome::Invalid(format!("code {}: {}", e.code, e.reason)),
    };
    let g = key.geometry(root_extent);
    if let Some(i) = sections
        .iter()
        .position(|s| s.origin() != g.origin || s.edge() != g.edge)
    {
        return FetchOutcome::Invalid(format!(
            "section {i}: cell geometry does not match the registry"
        ));
    }
    if sections.is_empty() {
        return FetchOutcome::Decoded(None);
    }
    let refs: Vec<&Section> = sections.iter().collect();
    match composite(&refs) {
        Ok(c) if c.is_empty() => FetchOutcome::Decoded(None),
        Ok(c) => {
            let emitter = cell_emitter(&c);
            FetchOutcome::Decoded(Some(Arc::new(ReadyCell {
                section: Arc::new(c),
                emitter,
            })))
        }
        Err(e) => FetchOutcome::Invalid(format!("code {}: {}", e.code, e.reason)),
    }
}

/// Pixels per unit of `edge / distance` for a view `view_height_px` tall.
fn focal_px(view_height_px: f64) -> f64 {
    view_height_px / (2.0 * (0.5 * FIELD_OF_VIEW_Y).tan())
}

/// The camera position in `frame_id`'s coordinates (relative to its
/// origin, in its axes).
pub fn camera_in_frame(system: &FrameSystem, camera: &Camera, frame_id: u64) -> Vec3 {
    let v = system.relative(camera.position, camera.frame_id, Vec3::zero(), frame_id);
    system.root_orientation(frame_id).conjugate().rotate(v)
}

/// Returns `true` when `frame_id`'s `root_extent / 8` region is within
/// `4 * root_extent` of the camera or inside the view cone.
pub fn frame_in_range(system: &FrameSystem, camera: &Camera, frame_id: u64, aspect: f64) -> bool {
    let Some(frame) = system.tree().get(frame_id) else {
        return false;
    };
    let extent = frame.root_extent.value();
    let region = extent / 8.0;
    let rel = camera.relative(system, Vec3::zero(), frame_id);
    let d = rel.length();
    if d - region <= 4.0 * extent {
        return true;
    }
    // Camera axes: looking along -z.
    let c = camera.root_orientation(system).conjugate().rotate(rel);
    let half_y = 0.5 * FIELD_OF_VIEW_Y;
    let half_x = (half_y.tan() * aspect.max(1.0)).atan();
    let half_diag = (half_x.tan().hypot(half_y.tan())).atan();
    let off_axis = (-c.z / d).clamp(-1.0, 1.0).acos();
    off_axis <= half_diag + (region / d).min(1.0).asin()
}

/// Selects the cells of every frame in range ([`frame_in_range`]), in
/// ascending key order, with their projected sizes.
pub fn select_all(
    system: &FrameSystem,
    camera: &Camera,
    view_width_px: f64,
    view_height_px: f64,
) -> Vec<SelectedCell> {
    let aspect = view_width_px / view_height_px.max(1.0);
    let params = SelectionParams {
        pixel_error: PIXEL_ERROR,
        view_height_px,
        vertical_fov_rad: FIELD_OF_VIEW_Y,
        max_cells: MAX_CELLS_PER_FRAME,
    };
    let focal = focal_px(view_height_px);
    let mut out = Vec::new();
    for frame in system.tree().frames() {
        if !frame_in_range(system, camera, frame.frame_id, aspect) {
            continue;
        }
        let cam = camera_in_frame(system, camera, frame.frame_id);
        for key in select_cells(frame, cam, &params).cells {
            let g = key.geometry(frame.root_extent);
            let e = g.edge.value();
            let gap = |c: f64, o: f64| (o - c).max(c - (o + e)).max(0.0);
            let d = Vec3::new(
                gap(cam.x, g.origin.x),
                gap(cam.y, g.origin.y),
                gap(cam.z, g.origin.z),
            )
            .length();
            out.push(SelectedCell {
                key,
                projected_px: e / d.max(e) * focal,
            });
        }
    }
    out
}

/// One cache entry.
#[derive(Clone, Debug, PartialEq)]
pub struct CellEntry {
    /// Fetch state.
    pub state: CellState,
    /// In the latest selection.
    pub selected: bool,
    /// Session time the cell was last selected or drawn, seconds.
    pub last_used: f64,
    /// Projected size at the latest selection, pixels.
    pub projected_px: f64,
    /// Session time of the latest request, seconds.
    pub requested_at: f64,
    /// Session time of the first `202`, seconds.
    pub pending_since: Option<f64>,
    /// A readiness push arrived while a request was in flight.
    pub ready_hint: bool,
    /// Do not request before this session time, seconds.
    pub retry_at: f64,
}

impl CellEntry {
    fn new(now: f64) -> CellEntry {
        CellEntry {
            state: CellState::Missing,
            selected: false,
            last_used: now,
            projected_px: 0.0,
            requested_at: f64::NEG_INFINITY,
            pending_since: None,
            ready_hint: false,
            retry_at: f64::NEG_INFINITY,
        }
    }
}

/// Counts for the overlay.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct CacheCounts {
    /// Cells in the latest selection.
    pub selected: usize,
    /// Selected cells that are `Ready` or `Empty`.
    pub ready: usize,
    /// Selected cells that are `Requested` or `Pending`.
    pub pending: usize,
    /// Requests in flight.
    pub in_flight: usize,
    /// Every cell in the cache.
    pub cached: usize,
}

/// The cell cache: one entry per key, in key order.
#[derive(Clone, Debug, Default)]
pub struct CellCache {
    entries: BTreeMap<CellKey, CellEntry>,
    selection: Vec<SelectedCell>,
}

impl CellCache {
    /// An empty cache.
    pub fn new() -> CellCache {
        CellCache::default()
    }

    /// The entry for a key.
    pub fn entry(&self, key: &CellKey) -> Option<&CellEntry> {
        self.entries.get(key)
    }

    /// The state of a key, `None` if it is not cached.
    pub fn state(&self, key: &CellKey) -> Option<&CellState> {
        self.entries.get(key).map(|e| &e.state)
    }

    /// Every entry, in key order.
    pub fn entries(&self) -> impl Iterator<Item = (&CellKey, &CellEntry)> {
        self.entries.iter()
    }

    /// The latest selection.
    pub fn selection(&self) -> &[SelectedCell] {
        &self.selection
    }

    /// Replaces the selection. New keys start `Missing`; `Missing` keys no
    /// longer selected are dropped, since there is nothing to keep.
    pub fn set_selection(&mut self, cells: Vec<SelectedCell>, now: f64) {
        for e in self.entries.values_mut() {
            e.selected = false;
        }
        for c in &cells {
            let e = self
                .entries
                .entry(c.key)
                .or_insert_with(|| CellEntry::new(now));
            e.selected = true;
            e.last_used = now;
            e.projected_px = c.projected_px;
        }
        self.entries
            .retain(|_, e| e.selected || e.state != CellState::Missing);
        self.selection = cells;
    }

    /// Picks the next requests: selected `Missing` cells, and `Pending`
    /// cells due for a poll, largest projected size first (ties in key
    /// order), up to [`MAX_IN_FLIGHT`] in flight. Marks them `Requested`.
    pub fn take_requests(&mut self, now: f64) -> Vec<CellKey> {
        let in_flight = self
            .entries
            .values()
            .filter(|e| e.state == CellState::Requested)
            .count();
        let room = MAX_IN_FLIGHT.saturating_sub(in_flight);
        if room == 0 {
            return Vec::new();
        }
        let mut due: Vec<(f64, CellKey)> = self
            .entries
            .iter()
            .filter(|(_, e)| {
                e.selected
                    && match e.state {
                        CellState::Missing => now >= e.retry_at,
                        CellState::Pending(since) => {
                            now - since >= PENDING_POLL_AFTER_SECONDS
                                && now - e.requested_at >= PENDING_POLL_INTERVAL_SECONDS
                        }
                        _ => false,
                    }
            })
            .map(|(k, e)| (e.projected_px, *k))
            .collect();
        due.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        due.truncate(room);
        due.into_iter()
            .map(|(_, k)| {
                let e = self.entries.get_mut(&k).expect("listed above");
                e.state = CellState::Requested;
                e.requested_at = now;
                e.ready_hint = false;
                k
            })
            .collect()
    }

    /// Applies the result of a request for `key`.
    pub fn complete(&mut self, key: CellKey, outcome: FetchOutcome, now: f64) {
        let e = self
            .entries
            .entry(key)
            .or_insert_with(|| CellEntry::new(now));
        e.state = match outcome {
            FetchOutcome::Decoded(Some(cell)) => CellState::Ready(cell),
            FetchOutcome::Decoded(None) | FetchOutcome::NotFound => CellState::Empty,
            FetchOutcome::Gone => CellState::Gone,
            FetchOutcome::Invalid(reason) => {
                tracing::warn!(key = %key, "cell is invalid: {reason}");
                CellState::Gone
            }
            FetchOutcome::Pending => {
                let since = *e.pending_since.get_or_insert(now);
                if e.ready_hint {
                    // The push beat the 202 here; ask again right away.
                    e.ready_hint = false;
                    e.retry_at = now;
                    CellState::Missing
                } else {
                    CellState::Pending(since)
                }
            }
            FetchOutcome::Failed(reason) => {
                tracing::debug!(key = %key, "cell request failed: {reason}");
                e.retry_at = now + RETRY_AFTER_SECONDS;
                CellState::Missing
            }
        };
        if !matches!(e.state, CellState::Pending(_) | CellState::Missing) {
            e.pending_since = None;
        }
    }

    /// Handles a `chunkReady` push: a `Pending` cell is asked for again.
    /// Returns `true` if the push changed anything.
    pub fn notify_ready(&mut self, key: &CellKey) -> bool {
        let Some(e) = self.entries.get_mut(key) else {
            return false;
        };
        match e.state {
            CellState::Pending(_) => {
                e.state = CellState::Missing;
                e.retry_at = f64::NEG_INFINITY;
                true
            }
            CellState::Requested => {
                e.ready_hint = true;
                true
            }
            _ => false,
        }
    }

    /// Marks cells as used now (drawn as stand-ins), so eviction keeps them.
    pub fn touch(&mut self, keys: &[CellKey], now: f64) {
        for k in keys {
            if let Some(e) = self.entries.get_mut(k) {
                e.last_used = now;
            }
        }
    }

    /// Evicts cells not used for [`EVICT_AFTER_SECONDS`], then, while more
    /// than [`MAX_CACHED_CELLS`] remain, the least recently used (ties in key
    /// order). Selected cells and requests in flight are never evicted.
    /// Returns the evicted keys in key order.
    pub fn evict(&mut self, now: f64) -> Vec<CellKey> {
        let droppable = |e: &CellEntry| !e.selected && e.state != CellState::Requested;
        let mut gone: BTreeSet<CellKey> = self
            .entries
            .iter()
            .filter(|(_, e)| droppable(e) && now - e.last_used >= EVICT_AFTER_SECONDS)
            .map(|(k, _)| *k)
            .collect();
        let remaining = self.entries.len() - gone.len();
        if remaining > MAX_CACHED_CELLS {
            let mut lru: Vec<(f64, CellKey)> = self
                .entries
                .iter()
                .filter(|(k, e)| droppable(e) && !gone.contains(k))
                .map(|(k, e)| (e.last_used, *k))
                .collect();
            lru.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            gone.extend(
                lru.into_iter()
                    .take(remaining - MAX_CACHED_CELLS)
                    .map(|(_, k)| k),
            );
        }
        for k in &gone {
            self.entries.remove(k);
        }
        gone.into_iter().collect()
    }

    /// Counts for the overlay.
    pub fn counts(&self) -> CacheCounts {
        let mut c = CacheCounts {
            selected: self.selection.len(),
            cached: self.entries.len(),
            ..CacheCounts::default()
        };
        for e in self.entries.values() {
            if e.state == CellState::Requested {
                c.in_flight += 1;
            }
            if !e.selected {
                continue;
            }
            match e.state {
                CellState::Ready(_) | CellState::Empty => c.ready += 1,
                CellState::Requested | CellState::Pending(_) => c.pending += 1,
                _ => {}
            }
        }
        c
    }

    /// The cells to draw for the latest selection, in key order, applying
    /// the depth transition rule of the module docs. `drawable` says whether
    /// a cell can be drawn now (its mesh is ready, or it is `Empty`).
    pub fn draw_set(&self, drawable: impl Fn(&CellKey) -> bool) -> Vec<CellKey> {
        let mut set: BTreeSet<CellKey> = BTreeSet::new();
        for s in &self.selection {
            if drawable(&s.key) {
                set.insert(s.key);
                continue;
            }
            if let Some(cover) = cover_below(&s.key, COVER_LEVELS, &drawable) {
                set.extend(cover);
                continue;
            }
            let mut up = s.key.parent();
            while let Some(p) = up {
                if drawable(&p) {
                    set.insert(p);
                    break;
                }
                up = p.parent();
            }
        }
        // A drawn ancestor hides everything below it.
        let hidden: Vec<CellKey> = set
            .iter()
            .filter(|k| {
                let mut up = k.parent();
                while let Some(p) = up {
                    if set.contains(&p) {
                        return true;
                    }
                    up = p.parent();
                }
                false
            })
            .copied()
            .collect();
        for k in hidden {
            set.remove(&k);
        }
        set.into_iter().collect()
    }
}

/// Drawable descendants within `levels` that cover all of `key`, or `None`.
fn cover_below(
    key: &CellKey,
    levels: u8,
    drawable: &impl Fn(&CellKey) -> bool,
) -> Option<Vec<CellKey>> {
    if levels == 0 {
        return None;
    }
    let mut out = Vec::new();
    for child in key.children()? {
        if drawable(&child) {
            out.push(child);
        } else {
            out.extend(cover_below(&child, levels - 1, drawable)?);
        }
    }
    Some(out)
}

#[cfg(not(target_arch = "wasm32"))]
pub use native::{FetchEvent, Fetcher};

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use super::{decode_cell, FetchOutcome};
    use crate::hub::{ChunkFetch, HubClient};
    use gx_core::key::{CellKey, ChunkKey};
    use gx_core::units::Meters;
    use std::sync::mpsc;
    use std::sync::Arc;
    use std::time::{Duration, Instant};
    use tokio::sync::mpsc as tokio_mpsc;

    /// Something the fetcher reports back to the frame loop.
    #[derive(Clone, Debug, PartialEq)]
    pub enum FetchEvent {
        /// A request finished.
        Completed {
            /// The cell.
            key: CellKey,
            /// What came back, decoded.
            outcome: FetchOutcome,
            /// Body bytes received.
            bytes: u64,
            /// Time from sending the request to having the body.
            round_trip: Duration,
        },
        /// The hub pushed `chunkReady` for a cell.
        ChunkReady(CellKey),
    }

    /// Fetches cells through the hub client on a tokio runtime and reports
    /// back over a channel the frame loop drains without blocking.
    ///
    /// The readiness WebSocket is opened once; a cell answering `202` is
    /// subscribed to on it. If the socket cannot be opened the cache's
    /// polling fallback still brings every cell in.
    pub struct Fetcher {
        handle: tokio::runtime::Handle,
        client: Arc<HubClient>,
        events_tx: mpsc::Sender<FetchEvent>,
        events: mpsc::Receiver<FetchEvent>,
        subscribe: tokio_mpsc::UnboundedSender<String>,
        bytes_fetched: u64,
        last_round_trip: Option<Duration>,
    }

    impl Fetcher {
        /// Starts a fetcher on `handle` and opens the readiness socket.
        pub fn new(handle: tokio::runtime::Handle, client: Arc<HubClient>) -> Fetcher {
            let (events_tx, events) = mpsc::channel();
            let (subscribe, mut keys) = tokio_mpsc::unbounded_channel::<String>();
            let socket_client = client.clone();
            let ready_tx = events_tx.clone();
            handle.spawn(async move {
                let build = socket_client.build_id().to_string();
                let mut sub = match socket_client.subscribe_ready(&build).await {
                    Ok(s) => s,
                    Err(e) => {
                        tracing::warn!("readiness socket unavailable ({e:#}), polling instead");
                        while keys.recv().await.is_some() {}
                        return;
                    }
                };
                loop {
                    tokio::select! {
                        key = keys.recv() => match key {
                            Some(k) => {
                                if sub.subscribe(&k).is_err() {
                                    break;
                                }
                            }
                            None => break,
                        },
                        ready = sub.next() => match ready {
                            Some(k) => {
                                if let Ok(ChunkKey::Cell(cell)) = k.parse::<ChunkKey>() {
                                    if ready_tx.send(FetchEvent::ChunkReady(cell)).is_err() {
                                        break;
                                    }
                                }
                            }
                            None => {
                                tracing::warn!("readiness socket closed, polling instead");
                                while keys.recv().await.is_some() {}
                                break;
                            }
                        },
                    }
                }
            });
            Fetcher {
                handle,
                client,
                events_tx,
                events,
                subscribe,
                bytes_fetched: 0,
                last_round_trip: None,
            }
        }

        /// Starts a request for `key`, a cell of a frame of edge
        /// `root_extent`. The result arrives through [`Fetcher::drain`].
        pub fn request(&self, key: CellKey, root_extent: Meters) {
            let client = self.client.clone();
            let tx = self.events_tx.clone();
            let subscribe = self.subscribe.clone();
            self.handle.spawn(async move {
                let name = key.to_string();
                let start = Instant::now();
                let fetched = client.fetch_chunk_uncached(&name).await;
                let round_trip = start.elapsed();
                let (outcome, bytes) = match fetched {
                    ChunkFetch::Ready(body) => {
                        let len = body.len() as u64;
                        let decoded = tokio::task::spawn_blocking(move || {
                            decode_cell(&key, &body, root_extent)
                        })
                        .await
                        .unwrap_or_else(|e| FetchOutcome::Failed(format!("decode task: {e}")));
                        (decoded, len)
                    }
                    ChunkFetch::Pending => {
                        let _ = subscribe.send(name);
                        (FetchOutcome::Pending, 0)
                    }
                    ChunkFetch::NotFound => (FetchOutcome::NotFound, 0),
                    ChunkFetch::Gone => (FetchOutcome::Gone, 0),
                    ChunkFetch::Error(e) => (FetchOutcome::Failed(e), 0),
                };
                let _ = tx.send(FetchEvent::Completed {
                    key,
                    outcome,
                    bytes,
                    round_trip,
                });
            });
        }

        /// Every event so far, without waiting. Updates the byte count and
        /// the last round-trip time.
        pub fn drain(&mut self) -> Vec<FetchEvent> {
            let events: Vec<FetchEvent> = self.events.try_iter().collect();
            self.account(&events);
            events
        }

        /// Waits up to `timeout` for at least one event, then drains.
        pub fn wait(&mut self, timeout: Duration) -> Vec<FetchEvent> {
            let mut events = Vec::new();
            if let Ok(e) = self.events.recv_timeout(timeout) {
                events.push(e);
            }
            events.extend(self.events.try_iter());
            self.account(&events);
            events
        }

        fn account(&mut self, events: &[FetchEvent]) {
            for e in events {
                if let FetchEvent::Completed {
                    bytes, round_trip, ..
                } = e
                {
                    self.bytes_fetched += bytes;
                    self.last_round_trip = Some(*round_trip);
                }
            }
        }

        /// Body bytes received this session.
        pub fn bytes_fetched(&self) -> u64 {
            self.bytes_fetched
        }

        /// Round-trip time of the latest finished request.
        pub fn last_round_trip(&self) -> Option<Duration> {
            self.last_round_trip
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(d: u8, x: u32, y: u32, z: u32) -> CellKey {
        CellKey::new(5, d, x, y, z).unwrap()
    }

    fn sel(keys: &[CellKey]) -> Vec<SelectedCell> {
        keys.iter()
            .enumerate()
            .map(|(i, &key)| SelectedCell {
                key,
                projected_px: i as f64,
            })
            .collect()
    }

    fn ready() -> FetchOutcome {
        use gx_core::matter::{Sample, Samples, State};
        use gx_core::units::{Attenuation, Density, Kelvin, Ratio};
        let s = Section::new(
            key(0, 0, 0, 0),
            Vec3::zero(),
            Meters::new(1.0),
            1,
            Samples::filled(
                1,
                Sample {
                    density: Density::new(1.0),
                    state: State::Solid,
                    temperature: Kelvin::new(1.0),
                    albedo: [Ratio::new(0.0); 3],
                    roughness: Ratio::new(0.0),
                    attenuation: Attenuation::new(0.0),
                },
            ),
        )
        .unwrap();
        FetchOutcome::Decoded(Some(Arc::new(ReadyCell {
            section: Arc::new(s),
            emitter: None,
        })))
    }

    #[test]
    fn requests_are_capped_and_prioritized() {
        let mut c = CellCache::new();
        let keys: Vec<CellKey> = (0..20).map(|i| key(5, i, 0, 0)).collect();
        c.set_selection(sel(&keys), 0.0);
        let first = c.take_requests(0.0);
        assert_eq!(first.len(), MAX_IN_FLIGHT);
        // Largest projected size first: the last keys had the largest.
        assert_eq!(first[0], keys[19]);
        assert!(c.take_requests(0.0).is_empty());
        c.complete(first[0], FetchOutcome::NotFound, 0.1);
        assert_eq!(c.take_requests(0.1).len(), 1);
        assert_eq!(c.counts().in_flight, MAX_IN_FLIGHT);
    }

    #[test]
    fn pending_waits_for_push_then_polls() {
        let mut c = CellCache::new();
        let k = key(1, 0, 0, 0);
        c.set_selection(sel(&[k]), 0.0);
        assert_eq!(c.take_requests(0.0), vec![k]);
        c.complete(k, FetchOutcome::Pending, 0.2);
        assert_eq!(c.state(&k), Some(&CellState::Pending(0.2)));
        assert!(c.take_requests(5.0).is_empty());
        assert!(c.notify_ready(&k));
        assert_eq!(c.take_requests(5.0), vec![k]);
        c.complete(k, FetchOutcome::Pending, 5.1);
        // Still pending since the first 202; polled after 10 s, every 2 s.
        assert_eq!(c.state(&k), Some(&CellState::Pending(0.2)));
        assert!(c.take_requests(6.0).is_empty());
        assert_eq!(c.take_requests(10.3), vec![k]);
        c.complete(k, FetchOutcome::Pending, 10.4);
        assert!(c.take_requests(11.0).is_empty());
        assert_eq!(c.take_requests(12.4), vec![k]);
        c.complete(k, ready(), 12.5);
        assert!(matches!(c.state(&k), Some(CellState::Ready(_))));
        assert_eq!(c.counts().ready, 1);
    }

    #[test]
    fn push_during_request_asks_again() {
        let mut c = CellCache::new();
        let k = key(1, 1, 0, 0);
        c.set_selection(sel(&[k]), 0.0);
        c.take_requests(0.0);
        assert!(c.notify_ready(&k));
        c.complete(k, FetchOutcome::Pending, 0.1);
        assert_eq!(c.take_requests(0.1), vec![k]);
    }

    #[test]
    fn failures_retry_and_gone_stays() {
        let mut c = CellCache::new();
        let (a, b) = (key(1, 0, 0, 0), key(1, 1, 0, 0));
        c.set_selection(sel(&[a, b]), 0.0);
        c.take_requests(0.0);
        c.complete(a, FetchOutcome::Failed("x".into()), 0.0);
        c.complete(b, FetchOutcome::Gone, 0.0);
        assert!(c.take_requests(1.0).is_empty());
        assert_eq!(c.take_requests(2.0), vec![a]);
        assert_eq!(c.state(&b), Some(&CellState::Gone));
    }

    #[test]
    fn eviction_by_age_and_cap() {
        let mut c = CellCache::new();
        let a = key(1, 0, 0, 0);
        c.set_selection(sel(&[a]), 0.0);
        c.take_requests(0.0);
        c.complete(a, ready(), 0.0);
        c.set_selection(Vec::new(), 1.0);
        assert!(c.evict(59.0).is_empty());
        assert_eq!(c.evict(60.0), vec![a]);

        let many: Vec<CellKey> = (0..(MAX_CACHED_CELLS as u32 + 10))
            .map(|i| CellKey::new(5, 13, i % 8192, i / 8192, 0).unwrap())
            .collect();
        for (i, k) in many.iter().enumerate() {
            c.complete(*k, FetchOutcome::NotFound, i as f64 * 1e-3);
        }
        let gone = c.evict(10.0);
        assert_eq!(gone.len(), 10);
        assert_eq!(gone, many[..10].to_vec());
        assert_eq!(c.counts().cached, MAX_CACHED_CELLS);
    }

    #[test]
    fn parent_draws_until_every_child_is_ready() {
        let mut c = CellCache::new();
        let parent = key(1, 0, 0, 0);
        let mut kids = parent.children().unwrap();
        kids.sort();
        c.set_selection(sel(&kids), 0.0);
        // Only the parent is drawable: it stands in for all children.
        let mut drawable: BTreeSet<CellKey> = [parent].into();
        let d = |c: &CellCache, s: &BTreeSet<CellKey>| c.draw_set(|k| s.contains(k));
        assert_eq!(d(&c, &drawable), vec![parent]);
        // Seven of eight children ready: still the parent alone.
        drawable.extend(kids[..7].iter().copied());
        assert_eq!(d(&c, &drawable), vec![parent]);
        // All eight: the children, without the parent.
        drawable.insert(kids[7]);
        assert_eq!(d(&c, &drawable), kids.to_vec());

        // Moving away: the parent is selected but not ready, its children
        // cover it.
        c.set_selection(sel(&[parent]), 1.0);
        drawable.remove(&parent);
        assert_eq!(d(&c, &drawable), kids.to_vec());
        // Nothing drawable anywhere: nothing drawn.
        assert!(d(&c, &BTreeSet::new()).is_empty());
    }

    #[test]
    fn decode_rejects_geometry_that_disagrees_with_the_registry() {
        use gx_core::container::encode_chunk;
        use gx_core::matter::{encode, Compression};
        let k = key(1, 1, 0, 0);
        let g = k.geometry(Meters::new(8.0));
        let empty = Section::empty(k, g.origin, g.edge).unwrap();
        let bytes = encode(&empty, Compression::None);
        let chunk = encode_chunk(&[&bytes], &["a"]);
        assert_eq!(
            decode_cell(&k, &chunk, Meters::new(8.0)),
            FetchOutcome::Decoded(None)
        );
        assert!(matches!(
            decode_cell(&k, &chunk, Meters::new(16.0)),
            FetchOutcome::Invalid(_)
        ));
        assert!(matches!(
            decode_cell(&k, &[1, 2], Meters::new(8.0)),
            FetchOutcome::Invalid(_)
        ));
        assert_eq!(
            decode_cell(&k, &encode_chunk(&[], &[]), Meters::new(8.0)),
            FetchOutcome::Decoded(None)
        );
    }
}

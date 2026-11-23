//! The client loop of `client-compute.md` section 1 as a state machine
//! without I/O: which keys to ask the hub about, what to do with each
//! layer's answer, and when a key has every layer's section.
//!
//! The native fetcher ([`crate::stream::Fetcher`]) and the browser fetcher
//! (`crate::web`) drive one [`Coordinator`] each: they feed it requests and
//! answers, and carry out the [`Action`]s it
//! returns. Everything here is deterministic in its inputs, so the whole
//! protocol is tested without a network.
//!
//! # The work list (`client-compute.md` section 2.8)
//!
//! The coordinator follows a wanted set: on every selection the cells
//! ([`Coordinator::set_wanted_cells`]) and the surface tiles
//! ([`Coordinator::set_wanted_surfaces`]) hand it every key they want now,
//! each with its priority (projected size on screen, pixels), and it
//! replaces its wanted set of that kind. [`Coordinator::request`] wants a
//! key (the registry, a pinned cell) at the highest priority until it
//! completes, whatever the selections say.
//!
//! - Directions batches are filled from wanted keys whose answer is
//!   unknown, highest priority first (ties in key order). Streams start
//!   highest priority first. Both orders are taken afresh on every
//!   [`Coordinator::poll`], so they follow the latest wanted set.
//! - A key that leaves the wanted set and has nothing in flight is
//!   dropped: no directions are asked and no stream is started for it.
//!   A directions answer or a stream already in flight completes. The
//!   answer is kept by key ([`Coordinator::cached_answers`]) and a streamed
//!   section lands in the driver's section cache, but neither starts
//!   anything more for an unwanted key.
//! - A key that comes back into the wanted set reuses its kept answer
//!   instead of asking the hub again.
//!
//! - Compute jobs carry their key's priority ([`Action::Compute`]). The
//!   driver's worker starts the highest first and, after every change of
//!   the wanted set, gives each waiting job its key's priority
//!   ([`Coordinator::compute_priority`]), removing a job whose key left
//!   the set before it starts. A removed job is reported as
//!   [`FailureKind::Dropped`]: its layer goes back to unknown and starts
//!   again from the kept answer if the key comes back. A running job
//!   finishes and its section is cached.
//! - With computing off ([`Coordinator::set_compute`], a compute budget
//!   of 0, `docs/api.md` section 7) a `missing` or `local` layer starts no
//!   job and stays missing, so its key does not complete; `ready` layers
//!   still stream. Turned on again, each such layer starts from the kept
//!   answer.
//!
//! For each wanted key (a cell, a surface tile, or the registry, which is
//! just another key):
//!
//! 1. The key is asked about in a directions batch of at most
//!    [`MAX_DIRECTION_KEYS`] keys, at most [`MAX_DIRECTIONS_IN_FLIGHT`]
//!    batches at a time, highest priority first. A key already in flight
//!    is not asked about twice.
//! 2. Per layer (`client-compute.md` section 2.4): `empty` contributes
//!    nothing; `ready` is streamed by its address and the hash of the
//!    stored bytes, `GET /section/{address}?sha256={hex}` (from the local
//!    cache or the edge, at most [`MAX_STREAMS_IN_FLIGHT`] at a time), and
//!    the driver checks the bytes against that hash, discarding a cached
//!    copy that differs and fetching again; `missing` and `local` go to the
//!    worker, whose job says whether the section is offered to the hub
//!    afterwards (only `missing`). Nothing waits on another client: there
//!    are no readiness subscriptions and no waiting addresses.
//! 3. A failed directions request, a failed stream (including bytes that
//!    do not match their hash), a `404` for a stored address, or a worker
//!    failure that may go away is asked about again after a backoff
//!    ([`backoff_seconds`]). A permanent worker failure fails the key.
//! 4. Once every layer has its section (or is empty) the key completes with
//!    the sections in the build's layer order, for compositing.
//! 5. A surface key never waits on an overlay layer
//!    ([`Coordinator::set_overlay_layers`], `rendering.md` section 7 item
//!    6): it completes as soon as every other layer has its section, with
//!    the overlays that have arrived, and completes again, with every
//!    section so far, each time another overlay arrives, listing the
//!    overlays still to come ([`Completion::waiting`]). A permanent worker
//!    failure of an overlay of a surface key is logged and the overlay is
//!    not waited for: the key completes without it.
//! 6. From its first partial completion until it completes, a surface
//!    key's compute jobs go before every job of a key with nothing drawn
//!    ([`drawn_first`], `rendering.md` section 7 item 6); the drivers give
//!    the waiting jobs their new priorities on each partial completion.
//!
//! # Checks (`client-compute.md` sections 2.6, "On the client", and 2.8)
//!
//! When a `ready` section of a cell or a surface tile arrives (from the
//! edge or the local cache), the coordinator draws a number from its own
//! random source, seeded once per client run ([`Coordinator::set_check_seed`])
//! and never from the address, and with probability `checkRate` returns an
//! [`Action::Check`]. The driver builds the job from the build config and
//! queues it in its check lane. When the recomputed hash differs from the
//! drawn one, the driver calls [`Coordinator::replace`]: the computed
//! section takes the place of the drawn one for that address and hash
//! (the key completes again with it, as a computed section would),
//! whatever the hub answers. The random source only chooses what to check;
//! it never touches bytes.

use crate::protocol::{Direction, DirectionsResponse, MAX_DIRECTION_KEYS};
use gx_core::key::{CellKey, ChunkKey, SurfaceKey};
use gx_core::matter::Section;
use gx_worker::{ComputeJob, Computed, FailureKind, WorkerError};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

/// Most directions requests in flight at once.
pub const MAX_DIRECTIONS_IN_FLIGHT: usize = 2;

/// Most section streams in flight at once.
pub const MAX_STREAMS_IN_FLIGHT: usize = 16;

/// Longest backoff, seconds.
pub const MAX_BACKOFF_SECONDS: f64 = 8.0;

/// The backoff after `attempt` consecutive failures (1 for the first):
/// 0.25 s doubled per attempt, at most [`MAX_BACKOFF_SECONDS`].
pub fn backoff_seconds(attempt: u32) -> f64 {
    (0.25 * 2f64.powi(attempt.saturating_sub(1).min(16) as i32)).min(MAX_BACKOFF_SECONDS)
}

/// Where a section's bytes came from.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Origin {
    /// Streamed from the edge.
    Edge,
    /// Read from the local cache.
    Cache,
    /// Computed here by the worker.
    Computed,
}

/// One layer's section as it arrived: its bytes, and for a cell key the
/// section decoded from them. Decoding is the validation
/// (`rendering.md` section 3.5): the stream or the worker that brought the
/// section in decoded it once, off the frame thread, and compositing takes
/// the decoded section instead of decoding again. `decoded` is `None` for
/// the registry key, which [`crate::protocol::frame_system_from_registries`]
/// decodes, and for bytes nobody has decoded yet.
#[derive(Clone, Debug, PartialEq)]
pub struct Arrived {
    /// The section bytes.
    pub bytes: Arc<[u8]>,
    /// The decoded section, if the bytes were decoded on the way in.
    pub decoded: Option<Arc<Section>>,
}

impl Arrived {
    /// Bytes nobody has decoded yet.
    pub fn raw(bytes: Arc<[u8]>) -> Arrived {
        Arrived {
            bytes,
            decoded: None,
        }
    }

    /// Byte length of the section.
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    /// Returns `true` for zero bytes.
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }
}

impl From<Computed> for Arrived {
    fn from(c: Computed) -> Arrived {
        Arrived {
            bytes: c.bytes,
            decoded: c.section,
        }
    }
}

/// The outcome of a stream.
#[derive(Clone, Debug, PartialEq)]
pub enum StreamResult {
    /// The section, from the edge or the local cache.
    Bytes(Arrived, Origin),
    /// `404`: the address is not stored (any more).
    NotFound,
    /// A network or transport failure.
    Failed(String),
}

/// A key with every layer's section.
#[derive(Clone, Debug, PartialEq)]
pub struct Completion {
    /// The key.
    pub key: ChunkKey,
    /// The non-empty layers' sections, in the build's layer order. For a
    /// surface key completed before its overlays (module docs, item 5),
    /// the overlays that have arrived so far.
    pub sections: Vec<Arrived>,
    /// Each section's layer index in the build, parallel to `sections`.
    pub layers: Vec<usize>,
    /// `true` while overlay layers of a surface key are still to come: the
    /// key stays in flight and completes again.
    pub partial: bool,
    /// The overlay layers still to come, in layer order: empty unless
    /// `partial`. A tile waits on these before it replaces its parent
    /// (`rendering.md` section 7 item 6).
    pub waiting: Vec<usize>,
    /// Session time, seconds, the key was requested.
    pub requested_at: f64,
}

/// Something the driver must do.
#[derive(Clone, Debug, PartialEq)]
pub enum Action {
    /// Ask for directions for these keys, then call
    /// [`Coordinator::directions_done`] with the same keys.
    Directions(Vec<ChunkKey>),
    /// Get the section at `address` (local cache, else the edge), check
    /// that its bytes hash to `sha256` (a cached copy that does not is
    /// discarded and fetched again), then call [`Coordinator::stream_done`].
    Stream {
        /// The key.
        key: ChunkKey,
        /// The layer's index in the build.
        layer: usize,
        /// The address.
        address: [u8; 32],
        /// SHA-256 of the stored bytes.
        sha256: [u8; 32],
    },
    /// Run the job on the worker, then call [`Coordinator::compute_done`].
    /// A job with `share` set (`missing`) is offered to the hub once the
    /// renderer has its section; one without (`local`) never is.
    Compute {
        /// The key.
        key: ChunkKey,
        /// The layer's index in the build.
        layer: usize,
        /// The job, boxed: it is much larger than the other actions.
        job: Box<ComputeJob>,
        /// The key's priority now (projected size on screen, pixels); the
        /// worker starts the highest first and follows later changes
        /// through [`Coordinator::compute_priority`].
        priority: f64,
    },
    /// Recompute the `ready` section at `address`, whose drawn bytes hash
    /// to `sha256`, in the check lane, report the hash to the hub, and if
    /// it differs call [`Coordinator::replace`].
    Check {
        /// The key.
        key: ChunkKey,
        /// The layer's index in the build.
        layer: usize,
        /// The address.
        address: [u8; 32],
        /// SHA-256 of the bytes drawn.
        sha256: [u8; 32],
    },
    /// The key is complete.
    Complete(Completion),
    /// The key can never complete.
    Failed {
        /// The key.
        key: ChunkKey,
        /// Why.
        reason: String,
    },
}

#[derive(Clone, Debug, PartialEq)]
enum Slot {
    /// Not answered yet, or to be asked about again.
    Unknown,
    Empty,
    StreamQueued([u8; 32], [u8; 32]),
    Streaming([u8; 32], [u8; 32]),
    Computing([u8; 32]),
    /// `missing` or `local` while computing is off
    /// ([`Coordinator::set_compute`]): stays missing until computing is
    /// on again, then is taken up from the kept answer.
    Held,
    /// A stream or a job failed in a way that may go away: ask again.
    Retry {
        retry_at: f64,
    },
    Done(Arrived),
}

impl Slot {
    fn finished(&self) -> bool {
        matches!(self, Slot::Empty | Slot::Done(_))
    }

    /// The slot's answer is stale and the key must be asked about again
    /// once `now` passes the returned time.
    fn ask_at(&self) -> Option<f64> {
        match self {
            Slot::Unknown => Some(f64::NEG_INFINITY),
            Slot::Retry { retry_at } => Some(*retry_at),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
struct Entry {
    requested_at: f64,
    /// One slot per layer; empty until the first answer.
    slots: Vec<Slot>,
    asking: bool,
    /// Not before this time, after a failed directions request.
    hold_until: f64,
    failures: u32,
    /// Finished slots when the key last completed early (module docs,
    /// item 5); 0 before.
    emitted: usize,
}

impl Entry {
    /// `true` while a directions request, a stream, or a job for the key
    /// is running.
    fn busy(&self) -> bool {
        self.asking
            || self
                .slots
                .iter()
                .any(|s| matches!(s, Slot::Streaming(..) | Slot::Computing(_)))
    }

    /// When the key is due for directions, if at all.
    fn due_at(&self) -> Option<f64> {
        if self.asking {
            return None;
        }
        let slot = if self.slots.is_empty() {
            Some(f64::NEG_INFINITY)
        } else {
            self.slots
                .iter()
                .filter_map(Slot::ask_at)
                .min_by(f64::total_cmp)
        };
        slot.map(|t| t.max(self.hold_until))
    }
}

/// Running counts, for the overlay and the statistics.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct DirectionStats {
    /// Directions requests sent.
    pub directions_requests: u64,
    /// Keys asked about, over all requests.
    pub keys_asked: u64,
    /// The most keys one request carried.
    pub largest_batch: u64,
    /// `missing` answers.
    pub missing_answers: u64,
    /// `local` answers.
    pub local_answers: u64,
    /// Failed directions requests.
    pub directions_failures: u64,
    /// Sections streamed from the edge.
    pub sections_streamed: u64,
    /// Sections read from the local cache.
    pub sections_cached: u64,
    /// Sections computed here.
    pub sections_computed: u64,
    /// Keys completed.
    pub keys_completed: u64,
    /// Keys failed.
    pub keys_failed: u64,
}

/// Priority of a key wanted through [`Coordinator::request`]: above every
/// projected size.
pub const REQUESTED_PRIORITY: f64 = f64::INFINITY;

/// The state of every key in flight, and the wanted set.
#[derive(Debug, Default)]
pub struct Coordinator {
    keys: BTreeMap<ChunkKey, Entry>,
    stream_queue: Vec<(ChunkKey, usize, [u8; 32], [u8; 32])>,
    streams_in_flight: usize,
    directions_in_flight: usize,
    /// Every wanted key and its priority.
    wanted: BTreeMap<ChunkKey, f64>,
    /// Keys wanted through [`Coordinator::request`].
    requested: BTreeSet<ChunkKey>,
    /// Keys that completed or failed and are still in the wanted set: the
    /// selection has not seen their result yet, so they are not started
    /// again.
    finished: BTreeSet<ChunkKey>,
    /// The latest directions answer per key.
    answers: BTreeMap<ChunkKey, Vec<Direction>>,
    origin_utc_ms: Option<i64>,
    stats: DirectionStats,
    /// `checkRate` of each `ready` slot, by key and layer.
    check_rates: BTreeMap<(ChunkKey, usize), f64>,
    /// The check source's state (SplitMix64).
    check_state: u64,
    /// Sections computed here in place of drawn ones that differed, by
    /// address and the hash of the drawn bytes.
    replaced: BTreeMap<([u8; 32], [u8; 32]), Arrived>,
    /// The build's overlay layers, by index.
    overlay_layers: BTreeSet<usize>,
    /// The hidden layers, by index.
    hidden_layers: BTreeSet<usize>,
    /// `true` while `missing` and `local` answers are held instead of
    /// computed (a compute budget of 0, `docs/api.md` section 7).
    compute_off: bool,
}

impl Coordinator {
    /// No keys in flight.
    pub fn new() -> Coordinator {
        Coordinator::default()
    }

    /// The build's overlay layers, by index (`client-compute.md` section
    /// 2.2): a surface key does not wait on them (module docs, item 5).
    pub fn set_overlay_layers(&mut self, layers: BTreeSet<usize>) {
        self.overlay_layers = layers;
    }

    /// The hidden layers, by index (`rendering.md` section 7 item 7): a
    /// hidden overlay is not fetched for surface keys answered from now on.
    /// An overlay shown again is wanted at once for every wanted surface
    /// key, at that key's priority: a finished key starts again (its kept
    /// answer is reused, so no directions are asked again), and a key in
    /// flight takes up the overlay's slot again.
    pub fn set_hidden_layers(&mut self, layers: BTreeSet<usize>, now: f64) {
        let shown: Vec<usize> = (self.hidden_layers.difference(&layers))
            .filter(|l| self.overlay_layers.contains(l))
            .copied()
            .collect();
        self.hidden_layers = layers;
        if shown.is_empty() {
            return;
        }
        let surfaces: Vec<ChunkKey> = (self.wanted.keys())
            .filter(|k| matches!(k, ChunkKey::Surface(_)))
            .copied()
            .collect();
        for k in surfaces {
            if self.finished.remove(&k) {
                self.start(k, now);
            } else if let Some(entry) = self.keys.get_mut(&k) {
                for &l in &shown {
                    if let Some(slot) = entry.slots.get_mut(l) {
                        if *slot == Slot::Empty {
                            *slot = Slot::Unknown;
                        }
                    }
                }
            }
        }
    }

    /// Seeds the random source that chooses which downloads to check.
    /// Called once per client run; the seed never affects any bytes.
    pub fn set_check_seed(&mut self, seed: u64) {
        self.check_state = seed;
    }

    /// A number in `[0, 1)` from the check source (SplitMix64).
    fn check_draw(&mut self) -> f64 {
        self.check_state = self.check_state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.check_state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Puts `computed`, recomputed by a check, in place of the drawn
    /// section of `key`'s `layer` at `address` whose bytes hash to
    /// `sha256`. The key completes again with it: at once if it is still
    /// in flight, else through a new request whose `ready` answer for
    /// that address and hash takes the computed section instead of
    /// streaming.
    pub fn replace(
        &mut self,
        key: ChunkKey,
        layer: usize,
        address: [u8; 32],
        sha256: [u8; 32],
        computed: Arrived,
        now: f64,
    ) -> Vec<Action> {
        self.replaced.insert((address, sha256), computed.clone());
        let mut out = Vec::new();
        if let Some(slot) = self.slot_mut(&key, layer) {
            if let Slot::Done(_) = slot {
                *slot = Slot::Done(computed);
            }
            return out;
        }
        self.request(key, now);
        self.finish_if_done(&key, &mut out);
        out
    }

    /// The universe's origin, once a directions response has carried it.
    pub fn origin_utc_ms(&self) -> Option<i64> {
        self.origin_utc_ms
    }

    /// The running counts.
    pub fn stats(&self) -> DirectionStats {
        self.stats
    }

    /// Keys in flight.
    pub fn in_flight(&self) -> usize {
        self.keys.len()
    }

    /// `true` when `key` is in flight.
    pub fn contains(&self, key: &ChunkKey) -> bool {
        self.keys.contains_key(key)
    }

    /// Wants `key` at [`REQUESTED_PRIORITY`] until it completes or fails,
    /// whatever the selections say. A key already in flight is not added
    /// twice.
    pub fn request(&mut self, key: ChunkKey, now: f64) {
        self.requested.insert(key);
        self.finished.remove(&key);
        self.wanted.insert(key, REQUESTED_PRIORITY);
        self.start(key, now);
    }

    /// Replaces the wanted cells with `cells`, each with its priority
    /// (projected size, pixels). See the module docs.
    pub fn set_wanted_cells(&mut self, cells: impl IntoIterator<Item = (CellKey, f64)>, now: f64) {
        self.set_wanted(
            |k| matches!(k, ChunkKey::Cell(_)),
            cells.into_iter().map(|(k, p)| (ChunkKey::Cell(k), p)),
            now,
        );
    }

    /// Replaces the wanted surface tiles with `tiles`, each with its
    /// priority (projected size, pixels). See the module docs.
    pub fn set_wanted_surfaces(
        &mut self,
        tiles: impl IntoIterator<Item = (SurfaceKey, f64)>,
        now: f64,
    ) {
        self.set_wanted(
            |k| matches!(k, ChunkKey::Surface(_)),
            tiles.into_iter().map(|(k, p)| (ChunkKey::Surface(k), p)),
            now,
        );
    }

    /// Replaces the wanted keys of one kind (`kind`) with `keys`, then
    /// drops every key no longer wanted that has nothing in flight.
    fn set_wanted(
        &mut self,
        kind: impl Fn(&ChunkKey) -> bool,
        keys: impl IntoIterator<Item = (ChunkKey, f64)>,
        now: f64,
    ) {
        let requested = &self.requested;
        self.wanted.retain(|k, _| !kind(k) || requested.contains(k));
        for (k, p) in keys {
            let p = if self.requested.contains(&k) {
                REQUESTED_PRIORITY
            } else {
                p
            };
            self.wanted.insert(k, p);
        }
        let wanted = &self.wanted;
        self.finished.retain(|k| wanted.contains_key(k));
        let fresh: Vec<ChunkKey> = self
            .wanted
            .keys()
            .filter(|k| kind(k) && !self.finished.contains(k))
            .copied()
            .collect();
        for k in fresh {
            self.start(k, now);
        }
        let idle: Vec<ChunkKey> = self
            .keys
            .iter()
            .filter(|(k, e)| !self.wanted.contains_key(k) && !e.busy())
            .map(|(k, _)| *k)
            .collect();
        for k in idle {
            self.keys.remove(&k);
        }
    }

    /// The wanted set: every wanted key and its priority, in key order.
    pub fn wanted(&self) -> &BTreeMap<ChunkKey, f64> {
        &self.wanted
    }

    /// The priority a waiting compute job for `key` should have now, or
    /// `None` when the key is no longer wanted and the job is to be dropped
    /// before it starts (`client-compute.md` section 2.8).
    /// A key whose surface tile is drawn and waits only on overlays (it
    /// has had a partial completion) is raised above every key with
    /// nothing drawn ([`drawn_first`], `rendering.md` section 7 item 6).
    pub fn compute_priority(&self, key: &ChunkKey) -> Option<f64> {
        let p = self.wanted.get(key).copied()?;
        Some(drawn_first(p, self.partly_done(key)))
    }

    /// Whether `key` has had a partial completion and is still in flight.
    fn partly_done(&self, key: &ChunkKey) -> bool {
        self.keys.get(key).is_some_and(|e| e.emitted > 0)
    }

    /// Turns computing on or off (`setComputeBudget`, `docs/api.md`
    /// section 7). Off, a `missing` or `local` answer starts no job and
    /// its layer stays missing; `ready` layers still stream. On again,
    /// every held layer is taken up from its kept answer at the next
    /// [`Coordinator::poll`].
    pub fn set_compute(&mut self, on: bool) {
        self.compute_off = !on;
        if on {
            for e in self.keys.values_mut() {
                for s in &mut e.slots {
                    if *s == Slot::Held {
                        *s = Slot::Unknown;
                    }
                }
            }
        }
    }

    /// Whether computing is on ([`Coordinator::set_compute`]).
    pub fn computes(&self) -> bool {
        !self.compute_off
    }

    /// The kept directions answers, by key.
    pub fn cached_answers(&self) -> &BTreeMap<ChunkKey, Vec<Direction>> {
        &self.answers
    }

    fn priority(&self, key: &ChunkKey) -> f64 {
        self.wanted.get(key).copied().unwrap_or(f64::NEG_INFINITY)
    }

    /// Adds an entry for a wanted key not in flight.
    fn start(&mut self, key: ChunkKey, now: f64) {
        if self.keys.contains_key(&key) {
            return;
        }
        self.keys.insert(
            key,
            Entry {
                requested_at: now,
                slots: Vec::new(),
                asking: false,
                hold_until: f64::NEG_INFINITY,
                failures: 0,
                emitted: 0,
            },
        );
    }

    /// Drops `key` if it is no longer wanted and has nothing in flight.
    fn drop_if_unwanted(&mut self, key: &ChunkKey) {
        if !self.wanted.contains_key(key) && self.keys.get(key).is_some_and(|e| !e.busy()) {
            self.keys.remove(key);
        }
    }

    /// The actions due at `now`: directions batches and streams.
    pub fn poll(&mut self, now: f64) -> Vec<Action> {
        let mut out = Vec::new();
        // A wanted key with a kept answer and no slots yet takes it
        // instead of asking again.
        let reuse: Vec<ChunkKey> = self
            .keys
            .iter()
            .filter(|(k, e)| {
                (e.slots.is_empty() || e.slots.contains(&Slot::Unknown))
                    && !e.asking
                    && self.wanted.contains_key(k)
                    && self.answers.contains_key(k)
            })
            .map(|(k, _)| *k)
            .collect();
        for k in reuse {
            let directions = self.answers[&k].clone();
            self.apply(&k, directions, &mut out);
        }
        while self.directions_in_flight < MAX_DIRECTIONS_IN_FLIGHT {
            let mut due: Vec<(f64, ChunkKey)> = self
                .keys
                .iter()
                .filter(|(k, e)| {
                    self.wanted.contains_key(k) && e.due_at().is_some_and(|t| t <= now)
                })
                .map(|(k, _)| (self.priority(k), *k))
                .collect();
            if due.is_empty() {
                break;
            }
            due.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
            due.truncate(MAX_DIRECTION_KEYS);
            let keys: Vec<ChunkKey> = due.into_iter().map(|(_, k)| k).collect();
            for k in &keys {
                self.keys.get_mut(k).expect("listed above").asking = true;
            }
            self.directions_in_flight += 1;
            self.stats.directions_requests += 1;
            self.stats.keys_asked += keys.len() as u64;
            self.stats.largest_batch = self.stats.largest_batch.max(keys.len() as u64);
            out.push(Action::Directions(keys));
        }
        // Queued streams of keys still wanted, highest priority first.
        let mut queue = std::mem::take(&mut self.stream_queue);
        queue.retain(|(key, layer, address, sha256)| {
            self.keys
                .get(key)
                .and_then(|e| e.slots.get(*layer))
                .is_some_and(|s| *s == Slot::StreamQueued(*address, *sha256))
        });
        queue.sort_by(|a, b| {
            self.priority(&b.0)
                .total_cmp(&self.priority(&a.0))
                .then(a.0.cmp(&b.0))
                .then(a.1.cmp(&b.1))
        });
        let mut queue = queue.into_iter();
        while self.streams_in_flight < MAX_STREAMS_IN_FLIGHT {
            let Some((key, layer, address, sha256)) = queue.next() else {
                break;
            };
            if !self.wanted.contains_key(&key) {
                self.stream_queue.push((key, layer, address, sha256));
                continue;
            }
            let slot = self.slot_mut(&key, layer).expect("kept above");
            *slot = Slot::Streaming(address, sha256);
            self.streams_in_flight += 1;
            out.push(Action::Stream {
                key,
                layer,
                address,
                sha256,
            });
        }
        self.stream_queue.extend(queue);
        out
    }

    /// The earliest time something becomes due, if anything is waiting on
    /// time alone.
    pub fn next_wake(&self) -> Option<f64> {
        if self.directions_in_flight >= MAX_DIRECTIONS_IN_FLIGHT {
            return None;
        }
        self.keys
            .iter()
            .filter(|(k, _)| self.wanted.contains_key(k))
            .filter_map(|(_, e)| e.due_at())
            .min_by(f64::total_cmp)
    }

    fn slot_mut(&mut self, key: &ChunkKey, layer: usize) -> Option<&mut Slot> {
        self.keys.get_mut(key)?.slots.get_mut(layer)
    }

    /// Applies the answer to a directions request for `keys`.
    pub fn directions_done(
        &mut self,
        keys: &[ChunkKey],
        response: Result<DirectionsResponse, String>,
        now: f64,
    ) -> Vec<Action> {
        self.directions_in_flight = self.directions_in_flight.saturating_sub(1);
        let mut out = Vec::new();
        let response = match response {
            Ok(r) => r,
            Err(reason) => {
                self.stats.directions_failures += 1;
                tracing::debug!(keys = keys.len(), "directions failed: {reason}");
                for k in keys {
                    self.hold(k, now);
                    self.drop_if_unwanted(k);
                }
                return out;
            }
        };
        self.origin_utc_ms = Some(response.origin_utc_ms);
        let by_key: BTreeMap<&str, &crate::protocol::KeyDirections> = response
            .results
            .iter()
            .map(|r| (r.key.as_str(), r))
            .collect();
        for key in keys {
            let name = key.to_string();
            let Some(result) = by_key.get(name.as_str()) else {
                tracing::debug!(key = %key, "directions left the key out");
                self.hold(key, now);
                self.drop_if_unwanted(key);
                continue;
            };
            let parsed: Result<Vec<Direction>, String> = result
                .layers
                .iter()
                .map(|l| l.parse(key).map_err(|e| format!("{e:#}")))
                .collect();
            let directions = match parsed {
                Ok(d) => d,
                Err(e) => {
                    tracing::warn!(key = %key, "directions are malformed: {e}");
                    self.hold(key, now);
                    self.drop_if_unwanted(key);
                    continue;
                }
            };
            self.answers.insert(*key, directions.clone());
            if self.wanted.contains_key(key) {
                self.apply(key, directions, &mut out);
            } else if let Some(e) = self.keys.get_mut(key) {
                // Kept, and starts nothing.
                e.asking = false;
                self.drop_if_unwanted(key);
            }
        }
        out
    }

    /// Asks about `key` again after a backoff.
    fn hold(&mut self, key: &ChunkKey, now: f64) {
        if let Some(e) = self.keys.get_mut(key) {
            e.asking = false;
            e.failures += 1;
            e.hold_until = now + backoff_seconds(e.failures);
        }
    }

    fn apply(&mut self, key: &ChunkKey, directions: Vec<Direction>, out: &mut Vec<Action>) {
        let Some(entry) = self.keys.get_mut(key) else {
            return;
        };
        entry.asking = false;
        entry.failures = 0;
        entry.hold_until = f64::NEG_INFINITY;
        if entry.slots.len() != directions.len() {
            entry.slots = vec![Slot::Unknown; directions.len()];
        }
        let mut streams = Vec::new();
        let priority = drawn_first(
            self.wanted.get(key).copied().unwrap_or(f64::NEG_INFINITY),
            entry.emitted > 0,
        );
        for (layer, d) in directions.into_iter().enumerate() {
            let slot = &mut entry.slots[layer];
            // An answer already being acted on, or finished, stands.
            if !matches!(slot, Slot::Unknown | Slot::Retry { .. }) {
                continue;
            }
            // A hidden overlay is not fetched for drawing (`rendering.md`
            // section 7 item 7); inputs are resolved elsewhere.
            let hidden = matches!(key, ChunkKey::Surface(_))
                && self.overlay_layers.contains(&layer)
                && self.hidden_layers.contains(&layer);
            *slot = match d {
                _ if hidden => Slot::Empty,
                Direction::Empty => Slot::Empty,
                Direction::Ready {
                    address,
                    sha256,
                    check_rate,
                } => {
                    if let Some(computed) = self.replaced.get(&(address, sha256)) {
                        self.stats.sections_computed += 1;
                        Slot::Done(computed.clone())
                    } else {
                        self.check_rates.insert((*key, layer), check_rate);
                        streams.push((*key, layer, address, sha256));
                        Slot::StreamQueued(address, sha256)
                    }
                }
                Direction::Missing(_) | Direction::Local(_) if self.compute_off => Slot::Held,
                Direction::Missing(job) | Direction::Local(job) => {
                    if job.share {
                        self.stats.missing_answers += 1;
                    } else {
                        self.stats.local_answers += 1;
                    }
                    let address = job.address;
                    out.push(Action::Compute {
                        key: *key,
                        layer,
                        job: Box::new(job),
                        priority,
                    });
                    Slot::Computing(address)
                }
            };
        }
        self.stream_queue.extend(streams);
        self.finish_if_done(key, out);
        self.drop_if_unwanted(key);
    }

    /// Applies the outcome of a stream of `address` checked against
    /// `sha256`.
    pub fn stream_done(
        &mut self,
        key: &ChunkKey,
        layer: usize,
        address: [u8; 32],
        sha256: [u8; 32],
        result: StreamResult,
        now: f64,
    ) -> Vec<Action> {
        self.streams_in_flight = self.streams_in_flight.saturating_sub(1);
        let mut out = Vec::new();
        let Some(entry) = self.keys.get_mut(key) else {
            return out;
        };
        let Some(slot) = entry.slots.get_mut(layer) else {
            return out;
        };
        if *slot != Slot::Streaming(address, sha256) {
            return out;
        }
        match result {
            StreamResult::Bytes(bytes, origin) => {
                *slot = Slot::Done(bytes);
                match origin {
                    Origin::Edge => self.stats.sections_streamed += 1,
                    Origin::Cache => self.stats.sections_cached += 1,
                    Origin::Computed => self.stats.sections_computed += 1,
                }
                let rate = self.check_rates.remove(&(*key, layer)).unwrap_or(0.0);
                let checkable = !matches!(key, ChunkKey::Registry) && origin != Origin::Computed;
                if checkable && rate > 0.0 && self.check_draw() < rate {
                    out.push(Action::Check {
                        key: *key,
                        layer,
                        address,
                        sha256,
                    });
                }
            }
            StreamResult::NotFound | StreamResult::Failed(_) => {
                if let StreamResult::Failed(reason) = &result {
                    tracing::debug!(key = %key, "section stream failed: {reason}");
                }
                entry.failures += 1;
                *slot = Slot::Retry {
                    retry_at: now + backoff_seconds(entry.failures),
                };
            }
        }
        self.finish_if_done(key, &mut out);
        self.drop_if_unwanted(key);
        out
    }

    /// Applies the outcome of a worker job.
    pub fn compute_done(
        &mut self,
        key: &ChunkKey,
        layer: usize,
        address: [u8; 32],
        result: Result<Computed, WorkerError>,
        now: f64,
    ) -> Vec<Action> {
        let mut out = Vec::new();
        let Some(entry) = self.keys.get_mut(key) else {
            return out;
        };
        let Some(slot) = entry.slots.get_mut(layer) else {
            return out;
        };
        if *slot != Slot::Computing(address) {
            return out;
        }
        match result {
            Ok(computed) => {
                *slot = Slot::Done(computed.into());
                self.stats.sections_computed += 1;
                self.finish_if_done(key, &mut out);
            }
            Err(e) if e.kind == FailureKind::Dropped => {
                // Never started; taken up again from the kept answer if
                // the key comes back.
                *slot = Slot::Unknown;
            }
            Err(e) if e.kind == FailureKind::Transient => {
                entry.failures += 1;
                *slot = Slot::Retry {
                    retry_at: now + backoff_seconds(entry.failures),
                };
            }
            Err(e)
                if matches!(key, ChunkKey::Surface(_)) && self.overlay_layers.contains(&layer) =>
            {
                // A failed overlay is not waited for (`rendering.md`
                // section 7 item 6): logged once, and the tile completes
                // without it.
                tracing::warn!(key = %key, layer, "overlay failed: {}", e.message);
                *slot = Slot::Empty;
                self.finish_if_done(key, &mut out);
            }
            Err(e) => {
                self.fail(key, e.message, &mut out);
            }
        }
        self.drop_if_unwanted(key);
        out
    }

    fn fail(&mut self, key: &ChunkKey, reason: String, out: &mut Vec<Action>) {
        if self.keys.remove(key).is_some() {
            self.settle(key);
            self.stats.keys_failed += 1;
            out.push(Action::Failed { key: *key, reason });
        }
    }

    /// A key completed or failed: a requested key is no longer wanted, a
    /// selected one is not started again until a selection leaves it out.
    fn settle(&mut self, key: &ChunkKey) {
        if self.requested.remove(key) {
            self.wanted.remove(key);
        } else if self.wanted.contains_key(key) {
            self.finished.insert(*key);
        }
    }

    fn finish_if_done(&mut self, key: &ChunkKey, out: &mut Vec<Action>) {
        let surface = matches!(key, ChunkKey::Surface(_));
        let overlays = &self.overlay_layers;
        let Some(e) = self.keys.get_mut(key) else {
            return;
        };
        if e.slots.is_empty() {
            return;
        }
        let all = e.slots.iter().all(Slot::finished);
        if !all {
            // A surface key completes early when only overlays are left,
            // and again whenever another overlay has arrived.
            let base_done = surface
                && (e.slots.iter().enumerate()).all(|(i, s)| s.finished() || overlays.contains(&i));
            let finished = e.slots.iter().filter(|s| s.finished()).count();
            if base_done && finished > e.emitted {
                e.emitted = finished;
                let (layers, sections) = done_sections(e.slots.iter().cloned());
                let waiting = (e.slots.iter().enumerate())
                    .filter(|(_, s)| !s.finished())
                    .map(|(i, _)| i)
                    .collect();
                out.push(Action::Complete(Completion {
                    key: *key,
                    sections,
                    layers,
                    partial: true,
                    waiting,
                    requested_at: e.requested_at,
                }));
            }
            return;
        }
        let e = self.keys.remove(key).expect("checked above");
        self.settle(key);
        let (layers, sections) = done_sections(e.slots.into_iter());
        self.stats.keys_completed += 1;
        out.push(Action::Complete(Completion {
            key: *key,
            sections,
            layers,
            partial: false,
            waiting: Vec::new(),
            requested_at: e.requested_at,
        }));
    }
}

/// Projected sizes above this, pixels, are taken as this one when compute
/// jobs are ordered ([`drawn_first`]): a tile this large on screen is as
/// urgent as any larger one.
pub const PRIORITY_CEILING: f64 = 1e12;

/// The compute priority of a key at wanted priority `priority`
/// (`rendering.md` section 7 item 6, `client-compute.md` section 2.8):
/// a two-part order in one number. A key whose tile is drawn and waits
/// only on overlays (`drawn`) goes before every key with nothing drawn;
/// within each group the order is the projected size, capped at
/// [`PRIORITY_CEILING`]. A requested key ([`REQUESTED_PRIORITY`]) and a
/// key no longer wanted keep their infinite priorities. Only the order of
/// work changes, never what is computed.
pub fn drawn_first(priority: f64, drawn: bool) -> f64 {
    if !priority.is_finite() {
        return priority;
    }
    let p = priority.min(PRIORITY_CEILING);
    if drawn {
        p + 2.0 * PRIORITY_CEILING
    } else {
        p
    }
}

/// The finished sections among `slots` with their layer indices.
fn done_sections(slots: impl Iterator<Item = Slot>) -> (Vec<usize>, Vec<Arrived>) {
    slots
        .enumerate()
        .filter_map(|(i, s)| match s {
            Slot::Done(b) => Some((i, b)),
            _ => None,
        })
        .unzip()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{KeyDirections, RawDirection};

    fn cell(i: u32) -> ChunkKey {
        format!("5-12-{i}-0-0").parse().unwrap()
    }

    fn hex(b: u8) -> String {
        format!("{b:02x}").repeat(32)
    }

    fn empty(layer: &str) -> RawDirection {
        RawDirection {
            layer: layer.into(),
            state: "empty".into(),
            ..RawDirection::default()
        }
    }

    /// `ready` at address `a`, stored bytes hashing to `[a + 100; 32]`.
    fn ready(layer: &str, a: u8) -> RawDirection {
        RawDirection {
            layer: layer.into(),
            state: "ready".into(),
            address: Some(hex(a)),
            sha256: Some(hex(a + 100)),
            check_rate: Some(0.25),
            ..RawDirection::default()
        }
    }

    fn compute(layer: &str, a: u8) -> RawDirection {
        compute_as("missing", layer, a)
    }

    fn compute_as(state: &str, layer: &str, a: u8) -> RawDirection {
        RawDirection {
            layer: layer.into(),
            state: state.into(),
            address: Some(hex(a)),
            module: Some(hex(9)),
            origin_utc_ms: Some(0),
            span_index: Some(0),
            span_seconds: Some(0),
            ..RawDirection::default()
        }
    }

    fn answer(rows: Vec<(ChunkKey, Vec<RawDirection>)>) -> Result<DirectionsResponse, String> {
        Ok(DirectionsResponse {
            origin_utc_ms: 1000,
            results: rows
                .into_iter()
                .map(|(k, layers)| KeyDirections {
                    key: k.to_string(),
                    layers,
                })
                .collect(),
        })
    }

    fn bytes(b: u8) -> Arrived {
        Arrived::raw(Arc::from(vec![b; 4]))
    }

    fn computed(b: u8) -> Computed {
        Computed {
            bytes: Arc::from(vec![b; 4]),
            section: None,
            inputs_sha256: None,
        }
    }

    #[test]
    fn surface_key_never_waits_on_an_overlay() {
        let mut c = Coordinator::new();
        c.set_overlay_layers([1, 2].into());
        let k = ChunkKey::Surface(SurfaceKey::new(1, 0, 0, 0, 0).unwrap());
        c.request(k, 0.0);
        c.poll(0.0);
        c.directions_done(
            &[k],
            answer(vec![(
                k,
                vec![ready("s", 1), compute("o", 2), compute("p", 3)],
            )]),
            0.1,
        );
        c.poll(0.1);
        let a = c.stream_done(
            &k,
            0,
            [1; 32],
            [101; 32],
            StreamResult::Bytes(bytes(1), Origin::Cache),
            0.2,
        );
        let a: Vec<Action> = a
            .into_iter()
            .filter(|x| matches!(x, Action::Complete(_)))
            .collect();
        let [Action::Complete(early)] = &a[..] else {
            panic!("{a:?}")
        };
        assert!(early.partial);
        assert_eq!(early.layers, vec![0]);
        assert_eq!(early.waiting, vec![1, 2]);
        assert_eq!(c.in_flight(), 1);
        let a = c.compute_done(&k, 2, [3; 32], Ok(computed(3)), 0.3);
        let [Action::Complete(more)] = &a[..] else {
            panic!("{a:?}")
        };
        assert!(more.partial);
        assert_eq!(more.layers, vec![0, 2]);
        assert_eq!(more.waiting, vec![1]);
        let a = c.compute_done(&k, 1, [2; 32], Ok(computed(2)), 0.4);
        let [Action::Complete(all)] = &a[..] else {
            panic!("{a:?}")
        };
        assert!(!all.partial);
        assert_eq!(all.layers, vec![0, 1, 2]);
        assert_eq!(c.in_flight(), 0);
    }

    #[test]
    fn a_drawn_partial_tile_computes_before_a_larger_undrawn_one() {
        let mut c = Coordinator::new();
        c.set_overlay_layers([1].into());
        let a = SurfaceKey::new(1, 0, 0, 0, 0).unwrap();
        let b = SurfaceKey::new(1, 1, 0, 0, 0).unwrap();
        let (drawn, undrawn) = (ChunkKey::Surface(a), ChunkKey::Surface(b));
        c.set_wanted_surfaces([(a, 1.0), (b, 100.0)], 0.0);
        c.poll(0.0);
        c.directions_done(
            &[drawn, undrawn],
            answer(vec![
                (drawn, vec![ready("s", 1), compute("o", 2)]),
                (undrawn, vec![compute("s", 3), compute("o", 4)]),
            ]),
            0.1,
        );
        c.poll(0.1);
        assert!(c.compute_priority(&undrawn) > c.compute_priority(&drawn));
        let done = c.stream_done(
            &drawn,
            0,
            [1; 32],
            [101; 32],
            StreamResult::Bytes(bytes(1), Origin::Cache),
            0.2,
        );
        assert!(done
            .iter()
            .any(|x| matches!(x, Action::Complete(d) if d.partial)));
        let (p_drawn, p_undrawn) = (c.compute_priority(&drawn), c.compute_priority(&undrawn));
        assert!(p_drawn > p_undrawn, "{p_drawn:?} {p_undrawn:?}");
        // Within each group the order stays the projected size.
        assert!(drawn_first(2.0, true) > drawn_first(1.0, true));
        assert!(drawn_first(2.0, false) > drawn_first(1.0, false));
        assert_eq!(drawn_first(REQUESTED_PRIORITY, false), REQUESTED_PRIORITY);
        // Kept through a later change of the wanted set.
        c.set_wanted_surfaces([(a, 1.0), (b, 1000.0)], 0.3);
        assert!(c.compute_priority(&drawn) > c.compute_priority(&undrawn));
    }

    #[test]
    fn a_failed_overlay_is_not_waited_for() {
        let mut c = Coordinator::new();
        c.set_overlay_layers([1].into());
        let k = ChunkKey::Surface(SurfaceKey::new(1, 0, 0, 0, 0).unwrap());
        c.request(k, 0.0);
        c.poll(0.0);
        c.directions_done(
            &[k],
            answer(vec![(k, vec![ready("s", 1), compute("o", 2)])]),
            0.1,
        );
        c.poll(0.1);
        c.stream_done(
            &k,
            0,
            [1; 32],
            [101; 32],
            StreamResult::Bytes(bytes(1), Origin::Cache),
            0.2,
        );
        let a = c.compute_done(&k, 1, [2; 32], Err(WorkerError::permanent("trap")), 0.3);
        let [Action::Complete(done)] = &a[..] else {
            panic!("{a:?}")
        };
        assert!(!done.partial);
        assert_eq!(done.layers, vec![0]);
    }

    #[test]
    fn batches_dedup_and_order() {
        let mut c = Coordinator::new();
        for i in 0..1100 {
            c.request(cell(i), 0.0);
        }
        c.request(cell(3), 0.0);
        assert_eq!(c.in_flight(), 1100);
        let a = c.poll(0.0);
        assert_eq!(a.len(), MAX_DIRECTIONS_IN_FLIGHT);
        let Action::Directions(first) = &a[0] else {
            panic!()
        };
        assert_eq!(first.len(), MAX_DIRECTION_KEYS);
        assert_eq!(first[0], cell(0), "request order");
        // Nothing more until a batch returns.
        assert!(c.poll(0.0).is_empty());
        let keys = first.clone();
        c.directions_done(&keys, Err("down".into()), 0.0);
        // The failed batch waits out its backoff, then the rest goes first.
        let a = c.poll(0.0);
        let Action::Directions(third) = &a[0] else {
            panic!()
        };
        assert_eq!(third.len(), 1100 - 2 * MAX_DIRECTION_KEYS);
        assert_eq!(c.next_wake(), None, "both requests in flight");
        assert_eq!(c.stats().largest_batch, MAX_DIRECTION_KEYS as u64);
    }

    #[test]
    fn network_failures_back_off() {
        let mut c = Coordinator::new();
        c.request(cell(1), 0.0);
        let keys = vec![cell(1)];
        assert_eq!(c.poll(0.0), vec![Action::Directions(keys.clone())]);
        c.directions_done(&keys, Err("x".into()), 1.0);
        assert!(c.poll(1.1).is_empty());
        assert_eq!(c.next_wake(), Some(1.25));
        assert_eq!(c.poll(1.25), vec![Action::Directions(keys.clone())]);
        c.directions_done(&keys, Err("x".into()), 2.0);
        assert_eq!(c.next_wake(), Some(2.5));
        assert_eq!(backoff_seconds(30), MAX_BACKOFF_SECONDS);
    }

    #[test]
    fn ready_empty_and_compute_complete_in_layer_order() {
        let mut c = Coordinator::new();
        let k = cell(1);
        c.request(k, 0.0);
        c.poll(0.0);
        let a = c.directions_done(
            &[k],
            answer(vec![(k, vec![compute("a", 1), empty("b"), ready("c", 3)])]),
            0.1,
        );
        assert_eq!(c.origin_utc_ms(), Some(1000));
        let Action::Compute { layer: 0, job, .. } = &a[0] else {
            panic!("{a:?}")
        };
        assert_eq!(job.layer, "a");
        assert!(job.share);
        let s = c.poll(0.1);
        assert_eq!(
            s,
            vec![Action::Stream {
                key: k,
                layer: 2,
                address: [3; 32],
                sha256: [103; 32]
            }]
        );
        let none = c.stream_done(
            &k,
            2,
            [3; 32],
            [103; 32],
            StreamResult::Bytes(bytes(3), Origin::Cache),
            0.2,
        );
        assert!(none.is_empty());
        let done = c.compute_done(&k, 0, [1; 32], Ok(computed(1)), 0.3);
        assert_eq!(
            done,
            vec![Action::Complete(Completion {
                key: k,
                sections: vec![bytes(1), bytes(3)],
                layers: vec![0, 2],
                partial: false,
                waiting: Vec::new(),
                requested_at: 0.0
            })]
        );
        assert_eq!(c.in_flight(), 0);
        let s = c.stats();
        assert_eq!((s.sections_cached, s.sections_computed), (1, 1));
        // A key whose layers are all empty completes at once, with nothing.
        let r = ChunkKey::Registry;
        c.request(r, 1.0);
        c.poll(1.0);
        let a = c.directions_done(&[r], answer(vec![(r, vec![empty("a")])]), 1.0);
        assert!(matches!(&a[0], Action::Complete(x) if x.sections.is_empty()));
    }

    #[test]
    fn missing_and_local_both_compute_and_only_missing_is_shared() {
        let mut c = Coordinator::new();
        let k = cell(1);
        c.request(k, 0.0);
        c.poll(0.0);
        let a = c.directions_done(
            &[k],
            answer(vec![(
                k,
                vec![compute_as("missing", "a", 1), compute_as("local", "b", 2)],
            )]),
            1.0,
        );
        let shares: Vec<(usize, bool)> = a
            .iter()
            .map(|x| match x {
                Action::Compute { layer, job, .. } => (*layer, job.share),
                other => panic!("{other:?}"),
            })
            .collect();
        assert_eq!(shares, vec![(0, true), (1, false)]);
        assert_eq!(c.next_wake(), None, "nothing waits on time");
        c.compute_done(&k, 1, [2; 32], Ok(computed(2)), 1.1);
        let a = c.compute_done(&k, 0, [1; 32], Ok(computed(1)), 1.2);
        assert!(matches!(&a[0], Action::Complete(x) if x.sections.len() == 2));
        let s = c.stats();
        assert_eq!((s.missing_answers, s.local_answers), (1, 1));
    }

    #[test]
    fn a_changed_hash_streams_the_new_version() {
        let mut c = Coordinator::new();
        let k = cell(1);
        c.request(k, 0.0);
        c.poll(0.0);
        c.directions_done(&[k], answer(vec![(k, vec![ready("a", 1)])]), 0.0);
        c.poll(0.0);
        // The bytes did not match the hash: asked about again.
        c.stream_done(
            &k,
            0,
            [1; 32],
            [101; 32],
            StreamResult::Failed("hash".into()),
            1.0,
        );
        assert_eq!(c.poll(1.25), vec![Action::Directions(vec![k])]);
        // The hub replaced the section after a dispute: same address, new
        // hash, so the stream names the new version.
        let mut r = ready("a", 1);
        r.sha256 = Some(hex(7));
        c.directions_done(&[k], answer(vec![(k, vec![r])]), 1.3);
        assert_eq!(
            c.poll(1.3),
            vec![Action::Stream {
                key: k,
                layer: 0,
                address: [1; 32],
                sha256: [7; 32]
            }]
        );
        // A late result for the old version is ignored.
        assert!(c
            .stream_done(
                &k,
                0,
                [1; 32],
                [101; 32],
                StreamResult::Bytes(bytes(1), Origin::Edge),
                1.4
            )
            .is_empty());
        let a = c.stream_done(
            &k,
            0,
            [1; 32],
            [7; 32],
            StreamResult::Bytes(bytes(1), Origin::Edge),
            1.4,
        );
        assert!(matches!(a[0], Action::Complete(_)));
    }

    #[test]
    fn stream_and_worker_failures() {
        let mut c = Coordinator::new();
        let k = cell(1);
        c.request(k, 0.0);
        c.poll(0.0);
        c.directions_done(
            &k_slice(k),
            answer(vec![(k, vec![ready("a", 1), compute("b", 2)])]),
            0.0,
        );
        c.poll(0.0);
        // A 404 for a stored address: asked about again after a backoff.
        assert!(c
            .stream_done(&k, 0, [1; 32], [101; 32], StreamResult::NotFound, 1.0)
            .is_empty());
        assert_eq!(c.next_wake(), Some(1.25));
        // A transient worker failure: the same.
        c.compute_done(&k, 1, [2; 32], Err(WorkerError::transient("net")), 1.0);
        assert_eq!(c.poll(1.6), vec![Action::Directions(vec![k])]);
        let a = c.directions_done(
            &[k],
            answer(vec![(k, vec![ready("a", 1), compute("b", 2)])]),
            1.6,
        );
        assert!(matches!(a[0], Action::Compute { layer: 1, .. }));
        // A permanent failure fails the key.
        let a = c.compute_done(&k, 1, [2; 32], Err(WorkerError::permanent("trap")), 1.7);
        assert_eq!(
            a,
            vec![Action::Failed {
                key: k,
                reason: "trap".into()
            }]
        );
        assert_eq!(c.in_flight(), 0);
        // A late stream result for the failed key is ignored.
        assert!(c
            .stream_done(
                &k,
                0,
                [1; 32],
                [101; 32],
                StreamResult::Bytes(bytes(1), Origin::Edge),
                1.8
            )
            .is_empty());
    }

    fn k_slice(k: ChunkKey) -> Vec<ChunkKey> {
        vec![k]
    }

    #[test]
    fn missing_or_malformed_answers_are_asked_again() {
        let mut c = Coordinator::new();
        let (k, j) = (cell(1), cell(2));
        c.request(k, 0.0);
        c.request(j, 0.0);
        c.poll(0.0);
        let mut bad = ready("a", 1);
        bad.address = Some("nope".into());
        c.directions_done(&[k, j], answer(vec![(j, vec![bad])]), 0.0);
        assert_eq!(c.poll(0.25), vec![Action::Directions(vec![k, j])]);
    }

    fn cell_key(i: u32) -> CellKey {
        CellKey::new(12, 12, i, 0, 0).unwrap()
    }

    fn tile_key(i: u32) -> SurfaceKey {
        SurfaceKey::new(4, 0, 10, i, 0).unwrap()
    }

    /// The keys `[i, i + 24)` along the line, nearest (`i + 12`) largest.
    fn view_along(i: u32) -> Vec<(u32, f64)> {
        (i..i + 24)
            .map(|j| (j, 1000.0 / (1.0 + f64::from(j.abs_diff(i + 12)))))
            .collect()
    }

    /// Drives a coordinator through a flight along a line: 50 selections,
    /// each answered in part, with the rest still in flight when the next
    /// selection lands. Checks that nothing starts for an unwanted key,
    /// that no key is asked about twice, and that after the last selection
    /// its keys go first, highest priority first.
    fn fly(chunk: impl Fn(u32) -> ChunkKey, set: impl Fn(&mut Coordinator, Vec<(u32, f64)>)) {
        let mut c = Coordinator::new();
        let mut asked: BTreeMap<ChunkKey, u32> = BTreeMap::new();
        let mut directions: Vec<Vec<ChunkKey>> = Vec::new();
        let mut streams: Vec<(ChunkKey, usize, [u8; 32], [u8; 32])> = Vec::new();
        let mut completed = Vec::new();
        let check = |c: &Coordinator,
                     a: &[Action],
                     asked: &mut BTreeMap<ChunkKey, u32>,
                     directions: &mut Vec<Vec<ChunkKey>>,
                     streams: &mut Vec<(ChunkKey, usize, [u8; 32], [u8; 32])>,
                     completed: &mut Vec<ChunkKey>| {
            for a in a {
                match a {
                    Action::Directions(keys) => {
                        for k in keys {
                            assert!(c.wanted().contains_key(k), "asked about {k}");
                            *asked.entry(*k).or_default() += 1;
                            assert_eq!(asked[k], 1, "{k} asked twice");
                        }
                        directions.push(keys.clone());
                    }
                    Action::Stream {
                        key,
                        layer,
                        address,
                        sha256,
                    } => {
                        assert!(c.wanted().contains_key(key), "streamed {key}");
                        streams.push((*key, *layer, *address, *sha256));
                    }
                    Action::Complete(done) => completed.push(done.key),
                    // Chosen at the answer's `checkRate`; runs below
                    // everything wanted and starts nothing here.
                    Action::Check { .. } => {}
                    other => panic!("unexpected {other:?}"),
                }
            }
        };
        for step in 0..50u32 {
            let now = f64::from(step) * 0.1;
            set(&mut c, view_along(step));
            // Half of what is in flight answers after the view moved.
            let n = directions.len().div_ceil(2);
            for keys in directions.drain(..n).collect::<Vec<_>>() {
                let rows = keys
                    .iter()
                    .map(|k| (*k, vec![ready("a", 1), empty("b")]))
                    .collect();
                let a = c.directions_done(&keys, answer(rows), now);
                check(
                    &c,
                    &a,
                    &mut asked,
                    &mut directions,
                    &mut streams,
                    &mut completed,
                );
            }
            let n = streams.len().div_ceil(2);
            for (k, layer, address, sha256) in streams.drain(..n).collect::<Vec<_>>() {
                let r = StreamResult::Bytes(bytes(1), Origin::Edge);
                let a = c.stream_done(&k, layer, address, sha256, r, now);
                check(
                    &c,
                    &a,
                    &mut asked,
                    &mut directions,
                    &mut streams,
                    &mut completed,
                );
            }
            let a = c.poll(now);
            check(
                &c,
                &a,
                &mut asked,
                &mut directions,
                &mut streams,
                &mut completed,
            );
        }
        // Everything still in flight finishes; only the last view's keys
        // start anything, highest priority first.
        let last: BTreeMap<ChunkKey, f64> = view_along(49)
            .into_iter()
            .map(|(i, p)| (chunk(i), p))
            .collect();
        let before = completed.len();
        let mut started = Vec::new();
        while !directions.is_empty() || !streams.is_empty() {
            for keys in std::mem::take(&mut directions) {
                let rows = keys
                    .iter()
                    .map(|k| (*k, vec![ready("a", 1), empty("b")]))
                    .collect();
                let a = c.directions_done(&keys, answer(rows), 10.0);
                check(
                    &c,
                    &a,
                    &mut asked,
                    &mut directions,
                    &mut streams,
                    &mut completed,
                );
            }
            let a = c.poll(10.0);
            check(
                &c,
                &a,
                &mut asked,
                &mut directions,
                &mut streams,
                &mut completed,
            );
            for (k, layer, address, sha256) in std::mem::take(&mut streams) {
                started.push(k);
                let r = StreamResult::Bytes(bytes(1), Origin::Edge);
                let a = c.stream_done(&k, layer, address, sha256, r, 10.0);
                check(
                    &c,
                    &a,
                    &mut asked,
                    &mut directions,
                    &mut streams,
                    &mut completed,
                );
            }
            let a = c.poll(10.0);
            check(
                &c,
                &a,
                &mut asked,
                &mut directions,
                &mut streams,
                &mut completed,
            );
        }
        assert!(started.iter().all(|k| last.contains_key(k)));
        // Streams started after a wanted-set change follow its priority.
        let firsts: Vec<f64> = started
            .iter()
            .take(MAX_STREAMS_IN_FLIGHT)
            .map(|k| last[k])
            .collect();
        assert!(firsts.windows(2).all(|w| w[0] >= w[1]), "{firsts:?}");
        // Every key of the last view finished; none of the others started
        // anything after the last selection.
        assert!(!started.is_empty());
        // Every key of the last view finished, each exactly once.
        for k in last.keys() {
            assert_eq!(completed.iter().filter(|d| *d == k).count(), 1, "{k}");
        }
        // The keys completing after the last selection are the last view's.
        assert!(completed[before..].iter().all(|k| last.contains_key(k)));
        assert_eq!(c.in_flight(), 0);
    }

    #[test]
    fn cells_follow_the_view() {
        fly(
            |i| ChunkKey::Cell(cell_key(i)),
            |c, v| c.set_wanted_cells(v.into_iter().map(|(i, p)| (cell_key(i), p)), 0.0),
        );
    }

    #[test]
    fn tiles_follow_the_view() {
        fly(
            |i| ChunkKey::Surface(tile_key(i)),
            |c, v| c.set_wanted_surfaces(v.into_iter().map(|(i, p)| (tile_key(i), p)), 0.0),
        );
    }

    #[test]
    fn highest_priority_is_asked_first() {
        let mut c = Coordinator::new();
        c.set_wanted_cells((0..600).map(|i| (cell_key(i), f64::from(i))), 0.0);
        let a = c.poll(0.0);
        let Action::Directions(first) = &a[0] else {
            panic!()
        };
        assert_eq!(first.len(), MAX_DIRECTION_KEYS);
        assert_eq!(first[0], ChunkKey::Cell(cell_key(599)));
        assert_eq!(first[MAX_DIRECTION_KEYS - 1], ChunkKey::Cell(cell_key(88)));
        let Action::Directions(second) = &a[1] else {
            panic!()
        };
        assert_eq!(second.len(), 88);
    }

    #[test]
    fn a_key_that_leaves_and_returns_is_not_asked_twice() {
        for surface in [false, true] {
            let mut c = Coordinator::new();
            let (k, other) = if surface {
                (
                    ChunkKey::Surface(tile_key(1)),
                    ChunkKey::Surface(tile_key(2)),
                )
            } else {
                (ChunkKey::Cell(cell_key(1)), ChunkKey::Cell(cell_key(2)))
            };
            let set = |c: &mut Coordinator, keys: &[ChunkKey]| {
                if surface {
                    c.set_wanted_surfaces(
                        keys.iter().map(|k| match k {
                            ChunkKey::Surface(t) => (*t, 1.0),
                            _ => unreachable!(),
                        }),
                        0.0,
                    )
                } else {
                    c.set_wanted_cells(
                        keys.iter().map(|k| match k {
                            ChunkKey::Cell(t) => (*t, 1.0),
                            _ => unreachable!(),
                        }),
                        0.0,
                    )
                }
            };
            set(&mut c, &[k]);
            assert_eq!(c.poll(0.0), vec![Action::Directions(vec![k])]);
            // The view moves on while the answer is in flight.
            set(&mut c, &[other]);
            assert_eq!(c.poll(0.0), vec![Action::Directions(vec![other])]);
            let a = c.directions_done(&[k], answer(vec![(k, vec![ready("a", 1)])]), 0.0);
            assert!(a.is_empty(), "an unwanted key starts nothing: {a:?}");
            assert!(c.poll(0.0).is_empty());
            assert!(!c.contains(&k), "dropped once idle");
            assert!(c.cached_answers().contains_key(&k), "the answer is kept");
            // Back in view: the kept answer streams, no directions.
            set(&mut c, &[k]);
            let a = c.poll(0.0);
            assert_eq!(
                a,
                vec![Action::Stream {
                    key: k,
                    layer: 0,
                    address: [1; 32],
                    sha256: [101; 32],
                }]
            );
            // A stream in flight when the key leaves finishes and is kept.
            set(&mut c, &[]);
            let a = c.stream_done(
                &k,
                0,
                [1; 32],
                [101; 32],
                StreamResult::Bytes(bytes(1), Origin::Edge),
                0.0,
            );
            assert!(matches!(&a[..], [Action::Complete(done)] if done.key == k));
            assert!(!c.contains(&k));
            assert!(c.poll(0.0).is_empty());
        }
    }

    #[test]
    fn cells_and_tiles_are_wanted_apart() {
        let mut c = Coordinator::new();
        c.request(ChunkKey::Registry, 0.0);
        c.set_wanted_surfaces([(tile_key(1), 5.0)], 0.0);
        c.set_wanted_cells([(cell_key(1), 9.0)], 0.0);
        c.set_wanted_cells([], 0.0);
        assert_eq!(
            c.poll(0.0),
            vec![Action::Directions(vec![
                ChunkKey::Registry,
                ChunkKey::Surface(tile_key(1))
            ])]
        );
    }

    /// Streams `n` cells whose one layer is `ready` at `rate` and returns
    /// how many were chosen for a check.
    fn checks_chosen(rate: f64, n: u32, seed: u64) -> usize {
        let mut c = Coordinator::new();
        c.set_check_seed(seed);
        let mut chosen = 0;
        for i in 0..n {
            let k = cell(i);
            c.request(k, 0.0);
            c.poll(0.0);
            let mut r = ready("a", 1);
            r.check_rate = Some(rate);
            c.directions_done(&[k], answer(vec![(k, vec![r])]), 0.0);
            c.poll(0.0);
            let a = c.stream_done(
                &k,
                0,
                [1; 32],
                [101; 32],
                StreamResult::Bytes(bytes(1), Origin::Edge),
                0.0,
            );
            chosen += a
                .iter()
                .filter(|a| matches!(a, Action::Check { .. }))
                .count();
            assert!(a.iter().any(|a| matches!(a, Action::Complete(_))));
        }
        chosen
    }

    #[test]
    fn rate_zero_never_checks_and_rate_one_always_does() {
        assert_eq!(checks_chosen(0.0, 200, 1), 0);
        assert_eq!(checks_chosen(1.0, 200, 1), 200);
        let half = checks_chosen(0.5, 400, 7);
        assert!((150..250).contains(&half), "{half}");
        // The choice is the client's own, not the address's.
        assert_ne!(checks_chosen(0.5, 64, 1), checks_chosen(0.5, 64, 2));
    }

    #[test]
    fn a_differing_check_replaces_the_drawn_section() {
        let mut c = Coordinator::new();
        let k = cell(1);
        c.request(k, 0.0);
        c.poll(0.0);
        let mut r = ready("a", 1);
        r.check_rate = Some(1.0);
        c.directions_done(&[k], answer(vec![(k, vec![r])]), 0.0);
        c.poll(0.0);
        let a = c.stream_done(
            &k,
            0,
            [1; 32],
            [101; 32],
            StreamResult::Bytes(bytes(1), Origin::Edge),
            0.0,
        );
        assert_eq!(
            a[0],
            Action::Check {
                key: k,
                layer: 0,
                address: [1; 32],
                sha256: [101; 32]
            }
        );
        let Action::Complete(done) = &a[1] else {
            panic!()
        };
        assert_eq!(done.sections, vec![bytes(1)]);
        // The recompute differs: the key completes again with it, from
        // the kept answer, without streaming.
        assert!(c
            .replace(k, 0, [1; 32], [101; 32], bytes(2), 1.0)
            .is_empty());
        let a = c.poll(1.0);
        let Some(Action::Complete(done)) = a.last() else {
            panic!("{a:?}")
        };
        assert_eq!(done.sections, vec![bytes(2)]);
        assert!(!a.iter().any(|a| matches!(a, Action::Stream { .. })));
    }
}

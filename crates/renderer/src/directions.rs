//! The client loop of `client-compute.md` section 1 as a state machine
//! without I/O: which keys to ask the hub about, what to do with each
//! layer's answer, and when a key has every layer's section.
//!
//! The native fetcher ([`crate::stream::Fetcher`]) and the browser fetcher
//! (`crate::web`) drive one [`Coordinator`] each: they feed it requests,
//! answers, and readiness notices, and carry out the [`Action`]s it
//! returns. Everything here is deterministic in its inputs, so the whole
//! protocol is tested without a network.
//!
//! For each requested key (a cell or the registry, which is just another
//! key):
//!
//! 1. The key is asked about in a directions batch of at most
//!    [`MAX_DIRECTION_KEYS`] keys, at most [`MAX_DIRECTIONS_IN_FLIGHT`]
//!    batches at a time, in request order. A key already in flight is not
//!    asked about twice.
//! 2. Per layer: `empty` contributes nothing; `ready` is streamed by its
//!    address (from the local cache or the edge, at most
//!    [`MAX_STREAMS_IN_FLIGHT`] at a time); `compute` goes to the worker;
//!    `pending` is subscribed to on the readiness socket and asked about
//!    again after the hub's `retryMs`, or streamed as soon as its notice
//!    arrives.
//! 3. A failed directions request, a failed stream, a `404` for a stored
//!    address, or a worker failure that may go away is asked about again
//!    after a backoff ([`backoff_seconds`]). A permanent worker failure
//!    fails the key.
//! 4. Once every layer has its section (or is empty) the key completes with
//!    the sections in the build's layer order, for compositing.

use crate::protocol::{Direction, DirectionsResponse, MAX_DIRECTION_KEYS};
use gx_core::key::ChunkKey;
use gx_worker::{ComputeJob, FailureKind, WorkerError};
use std::collections::{BTreeMap, VecDeque};
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

/// The outcome of a stream.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StreamResult {
    /// The bytes, from the edge or the local cache.
    Bytes(Arc<[u8]>, Origin),
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
    /// The non-empty layers' sections, in the build's layer order.
    pub sections: Vec<Arc<[u8]>>,
    /// Session time, seconds, the key was requested.
    pub requested_at: f64,
}

/// Something the driver must do.
#[derive(Clone, Debug, PartialEq)]
pub enum Action {
    /// Ask for directions for these keys, then call
    /// [`Coordinator::directions_done`] with the same keys.
    Directions(Vec<ChunkKey>),
    /// Get the section at `address` (local cache, else the edge), then call
    /// [`Coordinator::stream_done`].
    Stream {
        /// The key.
        key: ChunkKey,
        /// The layer's index in the build.
        layer: usize,
        /// The address.
        address: [u8; 32],
    },
    /// Run the job on the worker, then call [`Coordinator::compute_done`].
    Compute {
        /// The key.
        key: ChunkKey,
        /// The layer's index in the build.
        layer: usize,
        /// The job.
        job: ComputeJob,
    },
    /// Subscribe to the address on the readiness socket.
    Subscribe([u8; 32]),
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
    StreamQueued([u8; 32]),
    Streaming([u8; 32]),
    Computing([u8; 32]),
    Pending {
        address: [u8; 32],
        retry_at: f64,
    },
    Done(Arc<[u8]>),
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
            Slot::Pending { retry_at, .. } => Some(*retry_at),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
struct Entry {
    seq: u64,
    requested_at: f64,
    /// One slot per layer; empty until the first answer.
    slots: Vec<Slot>,
    asking: bool,
    /// Not before this time, after a failed directions request.
    hold_until: f64,
    failures: u32,
}

impl Entry {
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
    /// `pending` answers.
    pub pending_answers: u64,
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

/// The state of every key in flight.
#[derive(Debug, Default)]
pub struct Coordinator {
    keys: BTreeMap<ChunkKey, Entry>,
    /// Addresses waiting on a readiness notice: key and layer.
    waiting: BTreeMap<[u8; 32], (ChunkKey, usize)>,
    stream_queue: VecDeque<(ChunkKey, usize, [u8; 32])>,
    streams_in_flight: usize,
    directions_in_flight: usize,
    next_seq: u64,
    origin_utc_ms: Option<i64>,
    stats: DirectionStats,
}

impl Coordinator {
    /// No keys in flight.
    pub fn new() -> Coordinator {
        Coordinator::default()
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

    /// Adds a key. A key already in flight is not added twice.
    pub fn request(&mut self, key: ChunkKey, now: f64) {
        if self.keys.contains_key(&key) {
            return;
        }
        let seq = self.next_seq;
        self.next_seq += 1;
        self.keys.insert(
            key,
            Entry {
                seq,
                requested_at: now,
                slots: Vec::new(),
                asking: false,
                hold_until: f64::NEG_INFINITY,
                failures: 0,
            },
        );
    }

    /// The actions due at `now`: directions batches and streams.
    pub fn poll(&mut self, now: f64) -> Vec<Action> {
        let mut out = Vec::new();
        while self.directions_in_flight < MAX_DIRECTIONS_IN_FLIGHT {
            let mut due: Vec<(u64, ChunkKey)> = self
                .keys
                .iter()
                .filter(|(_, e)| e.due_at().is_some_and(|t| t <= now))
                .map(|(k, e)| (e.seq, *k))
                .collect();
            if due.is_empty() {
                break;
            }
            due.sort_unstable();
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
        while self.streams_in_flight < MAX_STREAMS_IN_FLIGHT {
            let Some((key, layer, address)) = self.stream_queue.pop_front() else {
                break;
            };
            let Some(slot) = self.slot_mut(&key, layer) else {
                continue;
            };
            if *slot != Slot::StreamQueued(address) {
                continue;
            }
            *slot = Slot::Streaming(address);
            self.streams_in_flight += 1;
            out.push(Action::Stream {
                key,
                layer,
                address,
            });
        }
        out
    }

    /// The earliest time something becomes due, if anything is waiting on
    /// time alone.
    pub fn next_wake(&self) -> Option<f64> {
        if self.directions_in_flight >= MAX_DIRECTIONS_IN_FLIGHT {
            return None;
        }
        self.keys
            .values()
            .filter_map(Entry::due_at)
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
                    continue;
                }
            };
            self.apply(key, directions, now, &mut out);
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

    fn apply(
        &mut self,
        key: &ChunkKey,
        directions: Vec<Direction>,
        now: f64,
        out: &mut Vec<Action>,
    ) {
        let Some(entry) = self.keys.get_mut(key) else {
            return;
        };
        entry.asking = false;
        entry.failures = 0;
        entry.hold_until = f64::NEG_INFINITY;
        if entry.slots.len() != directions.len() {
            entry.slots = vec![Slot::Unknown; directions.len()];
        }
        let mut subscribe = Vec::new();
        let mut unwait = Vec::new();
        let mut streams = Vec::new();
        for (layer, d) in directions.into_iter().enumerate() {
            let slot = &mut entry.slots[layer];
            // An answer already being acted on, or finished, stands.
            if !matches!(slot, Slot::Unknown | Slot::Pending { .. }) {
                continue;
            }
            if let Slot::Pending { address, .. } = slot {
                unwait.push(*address);
            }
            *slot = match d {
                Direction::Empty => Slot::Empty,
                Direction::Ready { address } => {
                    streams.push((*key, layer, address));
                    Slot::StreamQueued(address)
                }
                Direction::Compute(job) => {
                    let address = job.address;
                    out.push(Action::Compute {
                        key: *key,
                        layer,
                        job,
                    });
                    Slot::Computing(address)
                }
                Direction::Pending { address, retry_ms } => {
                    subscribe.push((address, layer));
                    Slot::Pending {
                        address,
                        retry_at: now + retry_ms as f64 / 1000.0,
                    }
                }
            };
        }
        self.stream_queue.extend(streams);
        self.stats.pending_answers += subscribe.len() as u64;
        for a in unwait {
            self.waiting.remove(&a);
        }
        for (address, layer) in subscribe {
            if self.waiting.insert(address, (*key, layer)).is_none() {
                out.push(Action::Subscribe(address));
            }
        }
        self.finish_if_done(key, out);
    }

    /// Applies the outcome of a stream.
    pub fn stream_done(
        &mut self,
        key: &ChunkKey,
        layer: usize,
        address: [u8; 32],
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
        if *slot != Slot::Streaming(address) {
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
            }
            StreamResult::NotFound | StreamResult::Failed(_) => {
                if let StreamResult::Failed(reason) = &result {
                    tracing::debug!(key = %key, "section stream failed: {reason}");
                }
                entry.failures += 1;
                *slot = Slot::Pending {
                    address,
                    retry_at: now + backoff_seconds(entry.failures),
                };
            }
        }
        self.finish_if_done(key, &mut out);
        out
    }

    /// Applies the outcome of a worker job.
    pub fn compute_done(
        &mut self,
        key: &ChunkKey,
        layer: usize,
        address: [u8; 32],
        result: Result<Arc<[u8]>, WorkerError>,
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
            Ok(bytes) => {
                *slot = Slot::Done(bytes);
                self.stats.sections_computed += 1;
                self.finish_if_done(key, &mut out);
            }
            Err(e) if e.kind == FailureKind::Transient => {
                entry.failures += 1;
                *slot = Slot::Pending {
                    address,
                    retry_at: now + backoff_seconds(entry.failures),
                };
            }
            Err(e) => {
                self.fail(key, e.message, &mut out);
            }
        }
        out
    }

    /// Applies a readiness notice: a layer waiting on `address` is streamed
    /// at once.
    pub fn ready_notice(&mut self, address: &[u8; 32]) -> Vec<Action> {
        if let Some((key, layer)) = self.waiting.remove(address) {
            if let Some(slot) = self.slot_mut(&key, layer) {
                if matches!(slot, Slot::Pending { address: a, .. } if a == address) {
                    *slot = Slot::StreamQueued(*address);
                    self.stream_queue.push_back((key, layer, *address));
                }
            }
        }
        Vec::new()
    }

    /// The addresses waiting on a readiness notice, for a socket that has
    /// to subscribe again after reconnecting.
    pub fn waiting_addresses(&self) -> Vec<[u8; 32]> {
        self.waiting.keys().copied().collect()
    }

    fn fail(&mut self, key: &ChunkKey, reason: String, out: &mut Vec<Action>) {
        if let Some(e) = self.keys.remove(key) {
            self.forget_waits(key, &e);
            self.stats.keys_failed += 1;
            out.push(Action::Failed { key: *key, reason });
        }
    }

    fn forget_waits(&mut self, key: &ChunkKey, e: &Entry) {
        for s in &e.slots {
            if let Slot::Pending { address, .. } = s {
                if self.waiting.get(address).is_some_and(|(k, _)| k == key) {
                    self.waiting.remove(address);
                }
            }
        }
    }

    fn finish_if_done(&mut self, key: &ChunkKey, out: &mut Vec<Action>) {
        let done = self
            .keys
            .get(key)
            .is_some_and(|e| !e.slots.is_empty() && e.slots.iter().all(Slot::finished));
        if !done {
            return;
        }
        let e = self.keys.remove(key).expect("checked above");
        let sections = e
            .slots
            .into_iter()
            .filter_map(|s| match s {
                Slot::Done(b) => Some(b),
                _ => None,
            })
            .collect();
        self.stats.keys_completed += 1;
        out.push(Action::Complete(Completion {
            key: *key,
            sections,
            requested_at: e.requested_at,
        }));
    }
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

    fn ready(layer: &str, a: u8) -> RawDirection {
        RawDirection {
            layer: layer.into(),
            state: "ready".into(),
            address: Some(hex(a)),
            ..RawDirection::default()
        }
    }

    fn pending(layer: &str, a: u8, retry_ms: u64) -> RawDirection {
        RawDirection {
            layer: layer.into(),
            state: "pending".into(),
            address: Some(hex(a)),
            retry_ms: Some(retry_ms),
            ..RawDirection::default()
        }
    }

    fn compute(layer: &str, a: u8) -> RawDirection {
        RawDirection {
            layer: layer.into(),
            state: "compute".into(),
            address: Some(hex(a)),
            claim: Some("c".into()),
            lease_ms: Some(120_000),
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

    fn bytes(b: u8) -> Arc<[u8]> {
        Arc::from(vec![b; 4])
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
        assert_eq!(job.claim, "c");
        let s = c.poll(0.1);
        assert_eq!(
            s,
            vec![Action::Stream {
                key: k,
                layer: 2,
                address: [3; 32]
            }]
        );
        let none = c.stream_done(
            &k,
            2,
            [3; 32],
            StreamResult::Bytes(bytes(3), Origin::Cache),
            0.2,
        );
        assert!(none.is_empty());
        let done = c.compute_done(&k, 0, [1; 32], Ok(bytes(1)), 0.3);
        assert_eq!(
            done,
            vec![Action::Complete(Completion {
                key: k,
                sections: vec![bytes(1), bytes(3)],
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
    fn pending_subscribes_honors_retry_and_streams_on_notice() {
        let mut c = Coordinator::new();
        let (k, j) = (cell(1), cell(2));
        c.request(k, 0.0);
        c.request(j, 0.0);
        c.poll(0.0);
        let a = c.directions_done(
            &[k, j],
            answer(vec![
                (k, vec![pending("a", 1, 500)]),
                (j, vec![pending("a", 2, 250)]),
            ]),
            1.0,
        );
        assert_eq!(
            a,
            vec![Action::Subscribe([1; 32]), Action::Subscribe([2; 32])]
        );
        assert_eq!(c.waiting_addresses(), vec![[1; 32], [2; 32]]);
        assert_eq!(c.next_wake(), Some(1.25));
        // The notice for the first arrives: streamed without asking again.
        c.ready_notice(&[1; 32]);
        assert_eq!(
            c.poll(1.1),
            vec![Action::Stream {
                key: k,
                layer: 0,
                address: [1; 32]
            }]
        );
        // The second is asked about again after its retryMs; the claim has
        // lapsed, so this client computes it.
        assert_eq!(c.poll(1.25), vec![Action::Directions(vec![j])]);
        let a = c.directions_done(&[j], answer(vec![(j, vec![compute("a", 2)])]), 1.3);
        assert!(matches!(a[0], Action::Compute { .. }));
        assert!(c.waiting_addresses().is_empty());
        let a = c.stream_done(
            &k,
            0,
            [1; 32],
            StreamResult::Bytes(bytes(1), Origin::Edge),
            1.4,
        );
        assert!(matches!(a[0], Action::Complete(_)));
        assert_eq!(c.stats().pending_answers, 2);
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
            .stream_done(&k, 0, [1; 32], StreamResult::NotFound, 1.0)
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
}

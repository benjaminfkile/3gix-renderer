//! The check lane (`client-compute.md` sections 2.6, "On the client", and
//! 2.8, checks): downloaded sections chosen to be recomputed and reported,
//! without any I/O.
//!
//! The lane holds at most [`CHECK_LANE_CAPACITY`] checks; a new check into
//! a full lane is skipped and counted, never queued in place of an older
//! one. A check runs below every compute and every upload: the native pool
//! asks the lane for work only when neither waits. Each check is run once;
//! its recompute is hashed and compared with the hash of the bytes drawn
//! ([`CheckJob::drawn`]), and the hub's answer is counted with
//! [`crate::WorkerStats::check_finished`].

use crate::job::ComputeJob;
use crate::stats::WorkerStats;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

/// Most checks the lane holds (`client-compute.md` section 2.8: bounded).
pub const CHECK_LANE_CAPACITY: usize = 32;

/// One downloaded section to recompute and report.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CheckJob {
    /// The job that recomputes it: address, layer, key, span index,
    /// module, and manifest.
    pub job: ComputeJob,
    /// The layer's index in the build.
    pub layer_index: usize,
    /// SHA-256 of the bytes the renderer drew.
    pub drawn: [u8; 32],
}

/// What the hub made of a check (`client-compute.md` section 2.6).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CheckOutcome {
    /// `200 {"result": "match"}`.
    Matched,
    /// `202 {"result": "disputed"}`.
    Disputed,
    /// `409`: the hub does not hold the inputs this section was
    /// computed from as true (`client-compute.md` section 10.4). Logged
    /// and counted, never retried; nobody is blamed.
    Conflict,
    /// Anything else (`403`, `404`, `429`, no answer), or a failed
    /// recompute.
    Failed,
}

impl CheckOutcome {
    /// The outcome of an HTTP status.
    pub fn from_status(status: u16) -> CheckOutcome {
        match status {
            200 => CheckOutcome::Matched,
            202 => CheckOutcome::Disputed,
            409 => CheckOutcome::Conflict,
            _ => CheckOutcome::Failed,
        }
    }
}

/// The bounded queue of checks, shared by whoever chooses them and
/// whoever runs them.
#[derive(Debug)]
pub struct CheckLane {
    queue: Mutex<VecDeque<CheckJob>>,
    capacity: usize,
    stats: Arc<WorkerStats>,
}

impl CheckLane {
    /// An empty lane of [`CHECK_LANE_CAPACITY`] counting into `stats`.
    pub fn new(stats: Arc<WorkerStats>) -> CheckLane {
        CheckLane::with_capacity(CHECK_LANE_CAPACITY, stats)
    }

    /// An empty lane holding at most `capacity` checks (at least one).
    pub fn with_capacity(capacity: usize, stats: Arc<WorkerStats>) -> CheckLane {
        CheckLane {
            queue: Mutex::new(VecDeque::new()),
            capacity: capacity.max(1),
            stats,
        }
    }

    /// Queues a check. A job that must not be shared, or a full lane,
    /// skips it. Returns `true` when it was queued.
    pub fn push(&self, check: CheckJob) -> bool {
        let mut q = self.queue.lock().expect("check lane");
        if !check.job.share || q.len() >= self.capacity {
            drop(q);
            self.stats.check_skipped();
            return false;
        }
        q.push_back(check);
        self.stats.check_queued();
        true
    }

    /// The oldest check, or `None` when the lane is empty.
    pub fn take(&self) -> Option<CheckJob> {
        self.queue.lock().expect("check lane").pop_front()
    }

    /// Checks waiting.
    pub fn len(&self) -> usize {
        self.queue.lock().expect("check lane").len()
    }

    /// `true` when nothing waits.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The counters this lane counts into.
    pub fn stats(&self) -> &Arc<WorkerStats> {
        &self.stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gx_core::module_io::ModuleRequest;

    fn check(i: u8) -> CheckJob {
        CheckJob {
            job: ComputeJob {
                address: [i; 32],
                layer: "l".into(),
                share: true,
                module: [0; 32],
                manifest: None,
                request: ModuleRequest {
                    origin_utc_ms: 0,
                    span_seconds: 0,
                    span_index: 0,
                    key: "1-0-0-0-0".parse().unwrap(),
                },
                slots: Vec::new(),
                frames: Vec::new(),
                frames_per_slot: Vec::new(),
                inputs_sha256: None,
            },
            layer_index: 0,
            drawn: [i; 32],
        }
    }

    #[test]
    fn a_full_lane_skips_new_checks() {
        let stats = WorkerStats::new();
        let lane = CheckLane::with_capacity(2, Arc::clone(&stats));
        assert!(lane.push(check(1)));
        assert!(lane.push(check(2)));
        assert!(!lane.push(check(3)));
        assert_eq!(lane.take().unwrap().drawn, [1; 32]);
        assert_eq!(lane.take().unwrap().drawn, [2; 32]);
        assert!(lane.take().is_none());
        let n = stats.snapshot();
        assert_eq!((n.checks_queued, n.checks_skipped), (2, 1));
        assert_eq!(CHECK_LANE_CAPACITY, CheckLane::new(stats).capacity);
    }

    #[test]
    fn statuses_map_to_outcomes() {
        assert_eq!(CheckOutcome::from_status(200), CheckOutcome::Matched);
        assert_eq!(CheckOutcome::from_status(202), CheckOutcome::Disputed);
        assert_eq!(CheckOutcome::from_status(429), CheckOutcome::Failed);
    }
}

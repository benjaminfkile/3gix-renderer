//! The upload lane (`client-compute.md` sections 2.5 and 2.8, uploads):
//! computed `missing` sections waiting to be offered to the hub, without
//! any I/O.
//!
//! A section enters the lane after it has been handed to the renderer.
//! The lane holds at most [`UPLOAD_LANE_CAPACITY`] offers; a new offer into
//! a full lane drops the oldest. Each offer is sent once: a failure is
//! counted and dropped, never retried, because another client produces the
//! section when it needs it. `local` sections never enter the lane. The
//! native fetcher and the browser fetcher both drain one lane each through
//! [`UploadLane::take`] and record the answer with
//! [`crate::UploadTicket::finish`]. The native fetcher takes an offer only
//! while fewer than [`MAX_UPLOADS_IN_FLIGHT`] are being sent and sends it
//! on its async runtime, never on a pool thread; the browser sends one at
//! a time from the page's event loop.

use crate::job::ComputeJob;
use crate::stats::{UploadTicket, WorkerStats};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

/// Most uploads a native client sends at once (`client-compute.md`
/// section 2.8). While this many are in flight the rest wait in the lane.
pub const MAX_UPLOADS_IN_FLIGHT: usize = 4;

/// Most offers the lane holds (`client-compute.md` section 2.8: bounded).
pub const UPLOAD_LANE_CAPACITY: usize = 64;

/// One computed section waiting to be offered.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UploadOffer {
    /// The job that produced it: address, layer, key, and span index.
    pub job: ComputeJob,
    /// The section bytes, as handed to the renderer.
    pub bytes: Arc<[u8]>,
}

/// What the hub made of an offer (`client-compute.md` section 2.5).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum UploadOutcome {
    /// `201`: the hub stored this copy.
    Stored,
    /// `200`: a copy was already stored and this one matches it.
    Matched,
    /// `202`: a copy was already stored and this one differs from it.
    Disputed,
    /// Anything else, or no answer at all.
    Failed,
}

impl UploadOutcome {
    /// The outcome of an HTTP status.
    pub fn from_status(status: u16) -> UploadOutcome {
        match status {
            201 => UploadOutcome::Stored,
            200 => UploadOutcome::Matched,
            202 => UploadOutcome::Disputed,
            _ => UploadOutcome::Failed,
        }
    }
}

/// The bounded queue of offers, shared by whoever computes sections and
/// whoever sends them.
#[derive(Debug)]
pub struct UploadLane {
    queue: Mutex<VecDeque<UploadOffer>>,
    capacity: usize,
    stats: Arc<WorkerStats>,
}

impl UploadLane {
    /// An empty lane of [`UPLOAD_LANE_CAPACITY`] counting into `stats`.
    pub fn new(stats: Arc<WorkerStats>) -> UploadLane {
        UploadLane::with_capacity(UPLOAD_LANE_CAPACITY, stats)
    }

    /// An empty lane holding at most `capacity` offers (at least one).
    pub fn with_capacity(capacity: usize, stats: Arc<WorkerStats>) -> UploadLane {
        UploadLane {
            queue: Mutex::new(VecDeque::new()),
            capacity: capacity.max(1),
            stats,
        }
    }

    /// Queues an offer. A job that must not be shared is ignored. When the
    /// lane is full the oldest offer is dropped and counted.
    pub fn offer(&self, offer: UploadOffer) {
        if !offer.job.share {
            return;
        }
        let mut q = self.queue.lock().expect("upload lane");
        if q.len() >= self.capacity {
            q.pop_front();
            self.stats.upload_dropped();
        }
        q.push_back(offer);
        self.stats.upload_offered();
    }

    /// The oldest offer and the ticket that records its answer, or `None`
    /// when the lane is empty.
    pub fn take(&self) -> Option<(UploadOffer, UploadTicket)> {
        let offer = self.queue.lock().expect("upload lane").pop_front()?;
        Some((offer, self.stats.upload_started()))
    }

    /// Offers waiting.
    pub fn len(&self) -> usize {
        self.queue.lock().expect("upload lane").len()
    }

    /// `true` when nothing waits.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gx_core::module_io::ModuleRequest;

    fn offer(i: u8, share: bool) -> UploadOffer {
        UploadOffer {
            job: ComputeJob {
                address: [i; 32],
                layer: "l".into(),
                share,
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
            },
            bytes: Arc::from(vec![i]),
        }
    }

    #[test]
    fn a_full_lane_drops_its_oldest() {
        let stats = WorkerStats::new();
        let lane = UploadLane::with_capacity(2, Arc::clone(&stats));
        lane.offer(offer(1, true));
        lane.offer(offer(2, true));
        lane.offer(offer(3, true));
        lane.offer(offer(4, false));
        assert_eq!(lane.len(), 2);
        let (a, t) = lane.take().unwrap();
        assert_eq!(a.job.address, [2; 32]);
        t.finish(UploadOutcome::Stored);
        let (b, t) = lane.take().unwrap();
        assert_eq!(b.job.address, [3; 32]);
        t.finish(UploadOutcome::from_status(500));
        assert!(lane.take().is_none());
        let n = stats.snapshot();
        assert_eq!(
            (
                n.uploads_offered,
                n.uploads_dropped,
                n.uploads_stored,
                n.uploads_failed,
                n.uploads_pending
            ),
            (3, 1, 1, 1, 0)
        );
        assert_eq!(UPLOAD_LANE_CAPACITY, UploadLane::new(stats).capacity);
    }

    #[test]
    fn statuses_map_to_outcomes() {
        assert_eq!(UploadOutcome::from_status(201), UploadOutcome::Stored);
        assert_eq!(UploadOutcome::from_status(200), UploadOutcome::Matched);
        assert_eq!(UploadOutcome::from_status(202), UploadOutcome::Disputed);
        assert_eq!(UploadOutcome::from_status(403), UploadOutcome::Failed);
    }
}

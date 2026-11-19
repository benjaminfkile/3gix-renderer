//! Running counts of what the worker did, for the overlay
//! (`client-compute.md` section 1: sections computed here, modules
//! loaded; section 2.8: uploads offered, stored, matched, disputed,
//! dropped, and failed; section 2.6: checks queued, run, skipped, matched,
//! disputed, and failed).

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crate::upload::UploadOutcome;

/// Counters shared by the worker's threads and whoever reports them.
#[derive(Debug, Default)]
pub struct WorkerStats {
    modules_loaded: AtomicU64,
    module_runs: AtomicU64,
    sections_computed: AtomicU64,
    jobs_failed: AtomicU64,
    uploads_pending: AtomicU64,
    uploads_offered: AtomicU64,
    uploads_stored: AtomicU64,
    uploads_matched: AtomicU64,
    uploads_disputed: AtomicU64,
    uploads_dropped: AtomicU64,
    uploads_failed: AtomicU64,
    checks_queued: AtomicU64,
    checks_run: AtomicU64,
    checks_skipped: AtomicU64,
    checks_matched: AtomicU64,
    checks_disputed: AtomicU64,
    checks_failed: AtomicU64,
}

/// A copy of the counters at one moment.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct WorkerSnapshot {
    /// Distinct modules compiled and admitted this session.
    pub modules_loaded: u64,
    /// Module calls made (`gx_sources` and `gx_compile` each count once).
    pub module_runs: u64,
    /// Sections computed here and validated.
    pub sections_computed: u64,
    /// Jobs that failed.
    pub jobs_failed: u64,
    /// Offers in the upload lane or being sent.
    pub uploads_pending: u64,
    /// Sections put into the upload lane.
    pub uploads_offered: u64,
    /// Offers the hub stored (`201`).
    pub uploads_stored: u64,
    /// Offers that matched a stored copy (`200`).
    pub uploads_matched: u64,
    /// Offers that differed from a stored copy (`202`).
    pub uploads_disputed: u64,
    /// Offers dropped from a full lane, never sent.
    pub uploads_dropped: u64,
    /// Offers sent that failed. Counted, never retried.
    pub uploads_failed: u64,
    /// Checks put into the check lane.
    pub checks_queued: u64,
    /// Checks whose recompute ran.
    pub checks_run: u64,
    /// Checks not queued because the check lane was full, or not runnable here.
    pub checks_skipped: u64,
    /// Checks the hub answered `match`.
    pub checks_matched: u64,
    /// Checks the hub answered `disputed`.
    pub checks_disputed: u64,
    /// Checks whose recompute or report failed.
    pub checks_failed: u64,
}

impl WorkerStats {
    /// Fresh counters.
    pub fn new() -> Arc<WorkerStats> {
        Arc::new(WorkerStats::default())
    }

    /// Records a module admitted for the first time.
    pub fn module_loaded(&self) {
        self.modules_loaded.fetch_add(1, Ordering::Relaxed);
    }

    /// Records one module call.
    pub fn module_run(&self) {
        self.module_runs.fetch_add(1, Ordering::Relaxed);
    }

    /// Records one finished job.
    pub fn job_finished(&self, ok: bool) {
        if ok {
            self.sections_computed.fetch_add(1, Ordering::Relaxed);
        } else {
            self.jobs_failed.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Records a section put into the upload lane.
    pub fn upload_offered(&self) {
        self.uploads_offered.fetch_add(1, Ordering::Relaxed);
        self.uploads_pending.fetch_add(1, Ordering::Relaxed);
    }

    /// Records an offer dropped from a full lane.
    pub fn upload_dropped(&self) {
        self.uploads_dropped.fetch_add(1, Ordering::Relaxed);
        self.uploads_pending.fetch_sub(1, Ordering::Relaxed);
    }

    /// Starts sending an offer already counted by
    /// [`WorkerStats::upload_offered`]; the ticket ends it.
    pub fn upload_started(self: &Arc<Self>) -> UploadTicket {
        UploadTicket {
            stats: Some(Arc::clone(self)),
        }
    }

    /// Records a check put into the check lane.
    pub fn check_queued(&self) {
        self.checks_queued.fetch_add(1, Ordering::Relaxed);
    }

    /// Records a check that was not queued or could not run.
    pub fn check_skipped(&self) {
        self.checks_skipped.fetch_add(1, Ordering::Relaxed);
    }

    /// Records a check whose recompute ran.
    pub fn check_run(&self) {
        self.checks_run.fetch_add(1, Ordering::Relaxed);
    }

    /// Records the hub's answer to a check.
    pub fn check_finished(&self, outcome: crate::check::CheckOutcome) {
        use crate::check::CheckOutcome;
        let counter = match outcome {
            CheckOutcome::Matched => &self.checks_matched,
            CheckOutcome::Disputed => &self.checks_disputed,
            CheckOutcome::Failed => &self.checks_failed,
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }

    /// The counters now.
    pub fn snapshot(&self) -> WorkerSnapshot {
        let l = |a: &AtomicU64| a.load(Ordering::Relaxed);
        WorkerSnapshot {
            modules_loaded: l(&self.modules_loaded),
            module_runs: l(&self.module_runs),
            sections_computed: l(&self.sections_computed),
            jobs_failed: l(&self.jobs_failed),
            uploads_pending: l(&self.uploads_pending),
            uploads_offered: l(&self.uploads_offered),
            uploads_stored: l(&self.uploads_stored),
            uploads_matched: l(&self.uploads_matched),
            uploads_disputed: l(&self.uploads_disputed),
            uploads_dropped: l(&self.uploads_dropped),
            uploads_failed: l(&self.uploads_failed),
            checks_queued: l(&self.checks_queued),
            checks_run: l(&self.checks_run),
            checks_skipped: l(&self.checks_skipped),
            checks_matched: l(&self.checks_matched),
            checks_disputed: l(&self.checks_disputed),
            checks_failed: l(&self.checks_failed),
        }
    }

    fn upload_finished(&self, outcome: UploadOutcome) {
        self.uploads_pending.fetch_sub(1, Ordering::Relaxed);
        let counter = match outcome {
            UploadOutcome::Stored => &self.uploads_stored,
            UploadOutcome::Matched => &self.uploads_matched,
            UploadOutcome::Disputed => &self.uploads_disputed,
            UploadOutcome::Failed => &self.uploads_failed,
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }
}

/// One upload being sent. [`UploadTicket::finish`] records the outcome;
/// a ticket dropped unfinished counts as a failed upload.
#[derive(Debug)]
pub struct UploadTicket {
    stats: Option<Arc<WorkerStats>>,
}

impl UploadTicket {
    /// Records the outcome of the upload.
    pub fn finish(mut self, outcome: UploadOutcome) {
        if let Some(s) = self.stats.take() {
            s.upload_finished(outcome);
        }
    }
}

impl Drop for UploadTicket {
    fn drop(&mut self) {
        if let Some(s) = self.stats.take() {
            s.upload_finished(UploadOutcome::Failed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tickets_count_pending_and_outcomes() {
        let s = WorkerStats::new();
        for _ in 0..4 {
            s.upload_offered();
        }
        let a = s.upload_started();
        let b = s.upload_started();
        let c = s.upload_started();
        assert_eq!(s.snapshot().uploads_pending, 4);
        a.finish(UploadOutcome::Matched);
        b.finish(UploadOutcome::Disputed);
        drop(c);
        s.upload_dropped();
        let n = s.snapshot();
        assert_eq!(
            (
                n.uploads_pending,
                n.uploads_matched,
                n.uploads_disputed,
                n.uploads_failed,
                n.uploads_dropped
            ),
            (0, 1, 1, 1, 1)
        );
    }
}

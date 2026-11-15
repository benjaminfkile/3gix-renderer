//! Running counts of what the worker did, for the overlay
//! (`client-compute.md` section 1: sections computed here, uploads pending,
//! modules loaded).

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Counters shared by the worker's threads and whoever reports them.
#[derive(Debug, Default)]
pub struct WorkerStats {
    modules_loaded: AtomicU64,
    module_runs: AtomicU64,
    sections_computed: AtomicU64,
    jobs_failed: AtomicU64,
    uploads_pending: AtomicU64,
    uploads_done: AtomicU64,
    uploads_failed: AtomicU64,
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
    /// Uploads started and not yet answered.
    pub uploads_pending: u64,
    /// Uploads the hub accepted (`201`, or `200` for a section already
    /// stored).
    pub uploads_done: u64,
    /// Uploads that failed. Logged; drawing never waits for them.
    pub uploads_failed: u64,
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

    /// Starts an upload; the ticket ends it.
    pub fn upload_started(self: &Arc<Self>) -> UploadTicket {
        self.uploads_pending.fetch_add(1, Ordering::Relaxed);
        UploadTicket {
            stats: Some(Arc::clone(self)),
        }
    }

    /// The counters now.
    pub fn snapshot(&self) -> WorkerSnapshot {
        WorkerSnapshot {
            modules_loaded: self.modules_loaded.load(Ordering::Relaxed),
            module_runs: self.module_runs.load(Ordering::Relaxed),
            sections_computed: self.sections_computed.load(Ordering::Relaxed),
            jobs_failed: self.jobs_failed.load(Ordering::Relaxed),
            uploads_pending: self.uploads_pending.load(Ordering::Relaxed),
            uploads_done: self.uploads_done.load(Ordering::Relaxed),
            uploads_failed: self.uploads_failed.load(Ordering::Relaxed),
        }
    }
}

/// One upload in progress. [`UploadTicket::finish`] records the outcome;
/// a ticket dropped unfinished counts as a failed upload.
#[derive(Debug)]
pub struct UploadTicket {
    stats: Option<Arc<WorkerStats>>,
}

impl UploadTicket {
    /// Records the outcome of the upload.
    pub fn finish(mut self, ok: bool) {
        if let Some(s) = self.stats.take() {
            s.uploads_pending.fetch_sub(1, Ordering::Relaxed);
            if ok {
                s.uploads_done.fetch_add(1, Ordering::Relaxed);
            } else {
                s.uploads_failed.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

impl Drop for UploadTicket {
    fn drop(&mut self) {
        if let Some(s) = self.stats.take() {
            s.uploads_pending.fetch_sub(1, Ordering::Relaxed);
            s.uploads_failed.fetch_add(1, Ordering::Relaxed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tickets_count_pending_and_outcomes() {
        let s = WorkerStats::new();
        let a = s.upload_started();
        let b = s.upload_started();
        let c = s.upload_started();
        assert_eq!(s.snapshot().uploads_pending, 3);
        a.finish(true);
        b.finish(false);
        drop(c);
        let n = s.snapshot();
        assert_eq!(
            (n.uploads_pending, n.uploads_done, n.uploads_failed),
            (0, 1, 2)
        );
    }
}

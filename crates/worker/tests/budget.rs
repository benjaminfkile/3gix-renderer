//! A pool running below its size (`docs/api.md` section 7,
//! `setComputeBudget`): every submitted job still runs, whichever thread
//! a new job wakes.

use gx_core::key::ChunkKey;
use gx_core::module_io::ModuleRequest;
use gx_module_host::HostConfig;
use gx_worker::{BlobKind, Blobs, ComputeJob, Uploader, WorkerError, WorkerPool};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::Duration;

struct NoBlobs;

impl Blobs for NoBlobs {
    fn fetch(&self, _: BlobKind, _: &[u8; 32]) -> Result<Vec<u8>, WorkerError> {
        Err(WorkerError::transient("none"))
    }
}

struct NoUploads;

impl Uploader for NoUploads {
    fn offer(&self, _: &ComputeJob, _: Arc<[u8]>) {}
}

/// A job that fails at once (its address does not match its inputs), so
/// `done` is called as soon as a thread takes it.
fn job(i: u32) -> ComputeJob {
    ComputeJob {
        address: [0xee; 32],
        layer: "l".into(),
        share: true,
        module: [0; 32],
        manifest: None,
        request: ModuleRequest {
            origin_utc_ms: 0,
            span_seconds: 0,
            span_index: u64::from(i),
            key: format!("1-6-{}-0-0", i % 64).parse::<ChunkKey>().unwrap(),
        },
        slots: Vec::new(),
        frames: Vec::new(),
        frames_per_slot: Vec::new(),
        inputs_sha256: None,
    }
}

#[test]
fn every_job_runs_with_a_budget_below_the_pool_size() {
    let pool = WorkerPool::new(
        8,
        HostConfig::default(),
        Arc::new(NoBlobs),
        None,
        Arc::new(NoUploads),
    )
    .unwrap();
    assert_eq!(pool.set_budget(1), 1);
    for i in 0..200 {
        let (done, is_done) = mpsc::channel();
        pool.submit(job(i), 1.0, move |_| {
            let _ = done.send(());
        });
        assert!(
            is_done.recv_timeout(Duration::from_secs(10)).is_ok(),
            "job {i} never ran with one of eight threads allowed"
        );
    }
}

//! The pool's work order (`client-compute.md` section 2.8): the highest
//! priority waiting compute starts next, a waiting compute whose key left
//! the wanted set never starts, and lower lanes run only when no compute
//! waits; checks run only after every compute and every upload, and a
//! check chosen while the check lane is full is skipped.
//!
//! The jobs here carry an address that does not match their inputs, so
//! each one fails its first check at once; the order in which their
//! `done` is called on a one-thread pool is the order in which they
//! started.

use gx_core::key::ChunkKey;
use gx_core::module_io::ModuleRequest;
use gx_module_host::HostConfig;
use gx_worker::{
    BlobKind, Blobs, CheckJob, CheckLane, ComputeJob, FailureKind, Lane, LaneSource, Uploader,
    WorkerError, WorkerPool, WorkerStats, CHECK_LANE_CAPACITY,
};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};

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

fn key(i: u32) -> ChunkKey {
    format!("1-6-{}-0-0", i % 64).parse().unwrap()
}

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
            span_index: i as u64,
            key: key(i),
        },
    }
}

type Log = Arc<Mutex<Vec<String>>>;

/// Holds the pool's only thread in the check lane until released.
struct Gate {
    work: Mutex<Option<Box<dyn FnOnce() + Send>>>,
}

impl LaneSource for Gate {
    fn next(&self) -> Option<Box<dyn FnOnce() + Send>> {
        self.work.lock().unwrap().take()
    }
}

/// Uploads that are always waiting, each logged when it runs.
struct Uploads {
    log: Log,
    left: Mutex<u32>,
}

impl LaneSource for Uploads {
    fn next(&self) -> Option<Box<dyn FnOnce() + Send>> {
        let mut left = self.left.lock().unwrap();
        if *left == 0 {
            return None;
        }
        *left -= 1;
        let log = Arc::clone(&self.log);
        Some(Box::new(move || log.lock().unwrap().push("upload".into())))
    }
}

/// A one-thread pool whose thread is held; dropping the returned sender
/// releases it.
fn held_pool() -> (WorkerPool, mpsc::Sender<()>) {
    let pool = WorkerPool::new(
        1,
        HostConfig::default(),
        Arc::new(NoBlobs),
        None,
        Arc::new(NoUploads),
    )
    .unwrap();
    let (release, wait) = mpsc::channel::<()>();
    let (started, is_started) = mpsc::channel::<()>();
    pool.set_lane(
        Lane::Check,
        Arc::new(Gate {
            work: Mutex::new(Some(Box::new(move || {
                started.send(()).unwrap();
                let _ = wait.recv();
            }))),
        }),
    );
    is_started.recv().unwrap();
    (pool, release)
}

fn submit(pool: &WorkerPool, log: &Log, i: u32, priority: f64) {
    let log = Arc::clone(log);
    pool.submit(job(i), priority, move |r| {
        let e = r.unwrap_err();
        let entry = if e.kind == FailureKind::Dropped {
            format!("dropped {i}")
        } else {
            format!("ran {i}")
        };
        log.lock().unwrap().push(entry);
    });
}

#[test]
fn waiting_jobs_for_keys_that_leave_the_view_never_start() {
    let (pool, release) = held_pool();
    let log = Log::default();
    for i in 0..6 {
        submit(&pool, &log, i, i as f64);
    }
    // The view moves: only keys 1, 3, and 4 are still wanted, 1 now first.
    let dropped = pool.reprioritize(|k| match k {
        k if *k == key(1) => Some(100.0),
        k if *k == key(3) => Some(3.0),
        k if *k == key(4) => Some(4.0),
        _ => None,
    });
    assert_eq!(dropped, 3);
    assert_eq!(pool.waiting(), 3);
    drop(release);
    drop(pool);
    let log = log.lock().unwrap().clone();
    assert_eq!(
        log,
        [
            "dropped 0",
            "dropped 2",
            "dropped 5",
            "ran 1",
            "ran 4",
            "ran 3"
        ]
    );
}

#[test]
fn the_highest_priority_waiting_job_always_starts_next() {
    let pool = WorkerPool::new(
        1,
        HostConfig::default(),
        Arc::new(NoBlobs),
        None,
        Arc::new(NoUploads),
    )
    .unwrap();
    let log = Log::default();
    let (release, wait) = mpsc::channel::<()>();
    let (started, is_started) = mpsc::channel::<()>();
    // A stream of new wanted keys, arriving while the thread is busy: a
    // deterministic shuffle of priorities.
    let priorities: Vec<f64> = (0..40u32).map(|i| ((i * 17 + 5) % 41) as f64).collect();
    let mut expected: Vec<(f64, u32)> = Vec::new();
    {
        let log = Arc::clone(&log);
        pool.submit(job(1000), f64::INFINITY, move |_| {
            started.send(()).unwrap();
            let _ = wait.recv();
            log.lock().unwrap().push("ran first".into());
        });
    }
    is_started.recv().unwrap();
    for (i, p) in priorities.iter().enumerate() {
        submit(&pool, &log, i as u32, *p);
        expected.push((*p, i as u32));
    }
    drop(release);
    drop(pool);
    expected.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    let mut want = vec!["ran first".to_string()];
    want.extend(expected.iter().map(|(_, i)| format!("ran {i}")));
    assert_eq!(*log.lock().unwrap(), want);
}

#[test]
fn a_long_run_of_wanted_work_starves_uploads_and_nothing_else() {
    let (pool, release) = held_pool();
    let log = Log::default();
    pool.set_lane(
        Lane::Upload,
        Arc::new(Uploads {
            log: Arc::clone(&log),
            left: Mutex::new(3),
        }),
    );
    for i in 0..50 {
        submit(&pool, &log, i, 1.0);
    }
    drop(release);
    drop(pool);
    let log = log.lock().unwrap().clone();
    assert_eq!(log.len(), 53);
    // Every compute ran, in the order queued, before any upload.
    for (i, entry) in log.iter().take(50).enumerate() {
        assert_eq!(*entry, format!("ran {i}"));
    }
    assert!(log[50..].iter().all(|e| e == "upload"));
}

/// The check lane as the pool sees it: each check logged when it runs.
struct Checks {
    log: Log,
    lane: Arc<CheckLane>,
}

impl LaneSource for Checks {
    fn next(&self) -> Option<Box<dyn FnOnce() + Send>> {
        let check = self.lane.take()?;
        let log = Arc::clone(&self.log);
        Some(Box::new(move || {
            log.lock()
                .unwrap()
                .push(format!("check {}", check.job.request.span_index))
        }))
    }
}

#[test]
fn a_busy_worker_runs_no_check_and_a_full_lane_skips_new_ones() {
    let (pool, release) = held_pool();
    let log = Log::default();
    let stats = WorkerStats::new();
    let lane = Arc::new(CheckLane::new(Arc::clone(&stats)));
    pool.set_lane(
        Lane::Check,
        Arc::new(Checks {
            log: Arc::clone(&log),
            lane: Arc::clone(&lane),
        }),
    );
    pool.set_lane(
        Lane::Upload,
        Arc::new(Uploads {
            log: Arc::clone(&log),
            left: Mutex::new(2),
        }),
    );
    for i in 0..5 {
        submit(&pool, &log, i, 1.0);
    }
    let chosen = CHECK_LANE_CAPACITY as u32 + 8;
    for i in 0..chosen {
        lane.push(CheckJob {
            job: job(100 + i),
            layer_index: 0,
            drawn: [0; 32],
        });
    }
    let n = stats.snapshot();
    assert_eq!(n.checks_queued, CHECK_LANE_CAPACITY as u64);
    assert_eq!(n.checks_skipped, 8);
    drop(release);
    drop(pool);
    let log = log.lock().unwrap().clone();
    let mut want: Vec<String> = (0..5).map(|i| format!("ran {i}")).collect();
    want.extend(["upload".to_string(), "upload".to_string()]);
    want.extend((0..CHECK_LANE_CAPACITY as u32).map(|i| format!("check {}", 100 + i)));
    assert_eq!(log, want);
}

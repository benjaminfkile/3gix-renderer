//! Worker determinism (`space-model.md` section 1 rule 7, section 4 item
//! 6): the native worker reproduces the core example module's conformance
//! hashes, both on the raw vectors and through the whole job pipeline
//! (address check, module, manifest, and tiles by hash, validation, the
//! local cache, and the upload hand-off).
//!
//! The module and its vectors are vendored in `testdata/example-module/`
//! by `scripts/update-example-module.sh` from the pinned core revision.

use gx_core::address::section_address;
use gx_core::manifest::{self, Entry, Manifest};
use gx_core::module_io::{self, ModuleRequest};
use gx_core::sha256::{sha256, to_hex};
use gx_module_host::HostConfig;
use gx_worker::{
    BlobKind, Blobs, ComputeJob, DiskCache, FailureKind, ModuleRunner, UploadTicket, Uploader,
    WorkerError, WorkerPool, WorkerStats,
};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::Duration;

fn data() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/example-module")
}

fn module_bytes() -> Vec<u8> {
    std::fs::read(data().join("module.wasm")).expect("the vendored example module")
}

/// Every vector: name, request bytes, tile bytes, expected SHA-256.
fn vectors() -> Vec<(String, Vec<u8>, Vec<u8>, String)> {
    let dir = data().join("vectors");
    let mut names: Vec<String> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| {
            e.unwrap()
                .file_name()
                .to_str()
                .and_then(|n| n.strip_suffix(".request.bin"))
                .map(str::to_string)
        })
        .collect();
    names.sort();
    assert!(names.len() >= 10, "vectors are missing");
    names
        .into_iter()
        .map(|n| {
            let read = |ext: &str| std::fs::read(dir.join(format!("{n}.{ext}"))).unwrap();
            let sha = String::from_utf8(read("sha256"))
                .unwrap()
                .trim()
                .to_string();
            (n.clone(), read("request.bin"), read("tiles.bin"), sha)
        })
        .collect()
}

/// An in-memory edge that counts what it serves.
#[derive(Default)]
struct Store {
    blobs: BTreeMap<(BlobKind, [u8; 32]), Vec<u8>>,
    fetches: AtomicU32,
}

impl Store {
    fn add(&mut self, kind: BlobKind, bytes: Vec<u8>) -> [u8; 32] {
        let h = sha256(&bytes);
        self.blobs.insert((kind, h), bytes);
        h
    }
}

impl Blobs for Store {
    fn fetch(&self, kind: BlobKind, hash: &[u8; 32]) -> Result<Vec<u8>, WorkerError> {
        self.fetches.fetch_add(1, Ordering::SeqCst);
        self.blobs
            .get(&(kind, *hash))
            .cloned()
            .ok_or_else(|| WorkerError::transient(format!("{kind} {} not found", to_hex(hash))))
    }
}

/// Records uploads and finishes their tickets.
#[derive(Default)]
struct Uploads(Mutex<Vec<(String, String, Vec<u8>)>>);

impl Uploader for Uploads {
    fn upload(&self, job: &ComputeJob, section: Arc<[u8]>, ticket: UploadTicket) {
        self.0
            .lock()
            .unwrap()
            .push((job.address_hex(), job.claim.clone(), section.to_vec()));
        ticket.finish(true);
    }
}

#[test]
fn native_worker_reproduces_the_conformance_hashes() {
    let wasm = module_bytes();
    let runner = ModuleRunner::new(HostConfig::default(), WorkerStats::new()).unwrap();
    let module = runner.module(&sha256(&wasm), || Ok(wasm.clone())).unwrap();
    for (name, request, tiles, expected) in vectors() {
        let out = module.compile_raw(&request, &tiles).unwrap();
        assert_eq!(to_hex(&sha256(&out)), expected, "{name}");
    }
    assert_eq!(runner.stats().snapshot().modules_loaded, 1);
}

/// Builds the job for a vector, with a manifest over its tiles in `store`.
fn job_for(store: &mut Store, module: [u8; 32], request: &[u8], tiles: &[u8]) -> ComputeJob {
    let request: ModuleRequest = module_io::decode_request(request).unwrap();
    let tiles = module_io::decode_tiles(tiles).unwrap();
    let manifest = if tiles.is_empty() {
        None
    } else {
        let entries = tiles
            .iter()
            .map(|t| Entry {
                id: t.id,
                sha256: store.add(BlobKind::Source, t.bytes.to_vec()),
                length: t.bytes.len() as u64,
            })
            .collect();
        let m = manifest::encode(&Manifest::new(entries).unwrap());
        Some(store.add(BlobKind::Manifest, m))
    };
    let r = &request;
    ComputeJob {
        address: section_address(
            &module,
            manifest.as_ref(),
            r.origin_utc_ms,
            r.span_seconds,
            r.span_index,
            &r.key,
        ),
        claim: format!("claim-{}", r.key),
        module,
        manifest,
        request,
    }
}

#[test]
fn pool_runs_jobs_end_to_end_and_caches_everything() {
    let mut store = Store::default();
    let module = store.add(BlobKind::Module, module_bytes());
    let cases = vectors();
    let jobs: Vec<(ComputeJob, String)> = cases
        .iter()
        .map(|(_, req, tiles, sha)| (job_for(&mut store, module, req, tiles), sha.clone()))
        .collect();
    let store = Arc::new(store);
    let cache_dir = std::env::temp_dir().join(format!("gx-worker-pool-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&cache_dir);
    let cache = Arc::new(DiskCache::open(&cache_dir, 1 << 30).unwrap());
    let uploads = Arc::new(Uploads::default());

    let run = |cache: Option<Arc<DiskCache>>| {
        let pool = WorkerPool::new(
            3,
            HostConfig::default(),
            store.clone(),
            cache,
            uploads.clone(),
        )
        .unwrap();
        let (tx, rx) = mpsc::channel();
        for (job, sha) in &jobs {
            let tx = tx.clone();
            let sha = sha.clone();
            pool.submit(job.clone(), move |r| tx.send((r, sha)).unwrap());
        }
        for _ in 0..jobs.len() {
            let (r, sha) = rx.recv_timeout(Duration::from_secs(120)).unwrap();
            assert_eq!(to_hex(&sha256(&r.unwrap().bytes)), sha);
        }
        // Dropping the pool joins its threads, so every upload has started.
        let stats = pool.stats_handle();
        drop(pool);
        stats.snapshot()
    };

    let first = run(Some(cache.clone()));
    assert_eq!(first.sections_computed, jobs.len() as u64);
    assert_eq!(first.modules_loaded, 1);
    assert_eq!(first.module_runs, 2 * jobs.len() as u64);
    assert_eq!(first.uploads_done, jobs.len() as u64);
    let fetched = store.fetches.load(Ordering::SeqCst);
    // A second pool with the same cache fetches nothing from the edge.
    run(Some(cache.clone()));
    assert_eq!(store.fetches.load(Ordering::SeqCst), fetched);
    // Every section is cached under its address, and every upload carried
    // its claim.
    for (job, _) in &jobs {
        assert!(cache.contains(BlobKind::Section, &job.address_hex()));
    }
    let up = uploads.0.lock().unwrap();
    assert_eq!(up.len(), 2 * jobs.len());
    for (address, claim, bytes) in up.iter() {
        let (job, sha) = jobs
            .iter()
            .find(|(j, _)| &j.address_hex() == address)
            .unwrap();
        assert_eq!(claim, &job.claim);
        assert_eq!(&to_hex(&sha256(bytes)), sha);
    }
    let _ = std::fs::remove_dir_all(&cache_dir);
}

#[test]
fn bad_inputs_fail_the_job() {
    let mut store = Store::default();
    let module = store.add(BlobKind::Module, module_bytes());
    let (_, req, tiles, _) = vectors().into_iter().nth(1).unwrap();
    let job = job_for(&mut store, module, &req, &tiles);
    let runner = ModuleRunner::new(HostConfig::default(), WorkerStats::new()).unwrap();

    // A wrong address is refused before anything runs.
    let mut wrong = job.clone();
    wrong.request.span_index += 1;
    assert_eq!(
        runner.run(&wrong, &store, None).unwrap_err().kind,
        FailureKind::Permanent
    );

    // Without its manifest, the module's tile request cannot be met.
    let mut no_manifest = job.clone();
    no_manifest.manifest = None;
    no_manifest.address = section_address(
        &module,
        None,
        job.request.origin_utc_ms,
        job.request.span_seconds,
        job.request.span_index,
        &job.request.key,
    );
    let e = runner.run(&no_manifest, &store, None).unwrap_err();
    assert_eq!(e.kind, FailureKind::Permanent, "{e}");

    // A damaged tile on the edge fails its hash check, and may be retried.
    let mut damaged = Store {
        blobs: store.blobs.clone(),
        ..Store::default()
    };
    for ((kind, _), bytes) in damaged.blobs.iter_mut() {
        if *kind == BlobKind::Source {
            bytes[0] ^= 1;
        }
    }
    let e = runner.run(&job, &damaged, None).unwrap_err();
    assert_eq!(e.kind, FailureKind::Transient, "{e}");

    // The good inputs work, and the module was admitted once.
    assert!(runner.run(&job, &store, None).is_ok());
    assert_eq!(runner.stats().snapshot().modules_loaded, 1);
}

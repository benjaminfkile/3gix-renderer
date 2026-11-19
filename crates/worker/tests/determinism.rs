//! Worker determinism (`space-model.md` section 1 rule 7, section 4 item
//! 6): the native worker reproduces the core example module's conformance
//! hashes, both on the raw vectors and through the whole job pipeline
//! (address check, module, manifest, and tiles by hash, validation, the
//! local cache, and the upload hand-off).
//!
//! The module and its vectors are vendored in `testdata/example-module/`
//! by `scripts/update-example-module.sh` from the pinned core revision.

use gx_core::address::{section_address, section_address_v2, SlotKind, SlotValue};
use gx_core::manifest::{self, Entry, Manifest};
use gx_core::module_io::{self, ModuleRequest};
use gx_core::sha256::{sha256, to_hex};
use gx_module_host::HostConfig;
use gx_worker::{
    BlobKind, Blobs, ComputeJob, DiskCache, FailureKind, InputDirection, InputSource, ModuleRunner,
    Resolver, Uploader, WorkerError, WorkerPool, WorkerStats,
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

/// Records offers to the upload lane.
#[derive(Default)]
struct Uploads(Mutex<Vec<(String, String, Vec<u8>)>>);

impl Uploader for Uploads {
    fn offer(&self, job: &ComputeJob, section: Arc<[u8]>) {
        self.0
            .lock()
            .unwrap()
            .push((job.address_hex(), job.layer.clone(), section.to_vec()));
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
        layer: format!("layer-{}", r.key),
        share: true,
        module,
        manifest,
        request,
        slots: Vec::new(),
        frames: Vec::new(),
        frames_per_slot: Vec::new(),
        inputs_sha256: None,
    }
}

#[test]
fn pool_runs_jobs_end_to_end_and_caches_everything() {
    let mut store = Store::default();
    let module = store.add(BlobKind::Module, module_bytes());
    let cases = vectors();
    // Every other job is `local`: computed and cached, never offered.
    let jobs: Vec<(ComputeJob, String)> = cases
        .iter()
        .enumerate()
        .map(|(i, (_, req, tiles, sha))| {
            let mut job = job_for(&mut store, module, req, tiles);
            job.share = i % 2 == 0;
            (job, sha.clone())
        })
        .collect();
    let shared = jobs.iter().filter(|(j, _)| j.share).count();
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
            pool.submit(job.clone(), 0.0, move |r| tx.send((r, sha)).unwrap());
        }
        for _ in 0..jobs.len() {
            let (r, sha) = rx.recv_timeout(Duration::from_secs(120)).unwrap();
            assert_eq!(to_hex(&sha256(&r.unwrap().bytes)), sha);
        }
        // Dropping the pool joins its threads, so every offer has been made.
        let stats = pool.stats_handle();
        drop(pool);
        stats.snapshot()
    };

    let first = run(Some(cache.clone()));
    assert_eq!(first.sections_computed, jobs.len() as u64);
    assert_eq!(first.modules_loaded, 1);
    assert_eq!(first.module_runs, 2 * jobs.len() as u64);
    let fetched = store.fetches.load(Ordering::SeqCst);
    // A second pool with the same cache fetches nothing from the edge.
    run(Some(cache.clone()));
    assert_eq!(store.fetches.load(Ordering::SeqCst), fetched);
    // Every section is cached under its address, every `missing` section
    // was offered with its layer, and no `local` one was.
    for (job, _) in &jobs {
        assert!(cache.contains(BlobKind::Section, &job.address_hex()));
    }
    let up = uploads.0.lock().unwrap();
    assert_eq!(up.len(), 2 * shared);
    for (address, layer, bytes) in up.iter() {
        let (job, sha) = jobs
            .iter()
            .find(|(j, _)| &j.address_hex() == address)
            .unwrap();
        assert!(job.share);
        assert_eq!(layer, &job.layer);
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
        runner.run(&wrong, &store, None, None).unwrap_err().kind,
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
    let e = runner.run(&no_manifest, &store, None, None).unwrap_err();
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
    let e = runner.run(&job, &damaged, None, None).unwrap_err();
    assert_eq!(e.kind, FailureKind::Transient, "{e}");

    // The good inputs work, and the module was admitted once.
    assert!(runner.run(&job, &store, None, None).is_ok());
    assert_eq!(runner.stats().snapshot().modules_loaded, 1);
}

fn data_v2() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/example-module-v2")
}

/// One version 2 vector (`client-compute.md` section 4.1): its name and
/// the request, frame, tile, input list, and input bytes, and the
/// expected SHA-256 of the output.
struct VectorV2 {
    name: String,
    request: Vec<u8>,
    frames: Vec<u8>,
    tiles: Vec<u8>,
    input_list: Vec<u8>,
    inputs: Vec<u8>,
    sha256: String,
}

fn vectors_v2() -> Vec<VectorV2> {
    let dir = data_v2().join("vectors");
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
    assert!(names.len() >= 5, "version 2 vectors are missing");
    names
        .into_iter()
        .map(|n| {
            let read = |ext: &str| std::fs::read(dir.join(format!("{n}.{ext}"))).unwrap();
            VectorV2 {
                request: read("request.bin"),
                frames: read("frames.bin"),
                tiles: read("tiles.bin"),
                input_list: read("input-list.bin"),
                inputs: read("inputs.bin"),
                sha256: String::from_utf8(read("sha256"))
                    .unwrap()
                    .trim()
                    .to_string(),
                name: n,
            }
        })
        .collect()
}

#[test]
fn native_worker_reproduces_the_version_2_conformance_hashes() {
    let wasm = std::fs::read(data_v2().join("module.wasm")).expect("the vendored v2 module");
    let runner = ModuleRunner::new(HostConfig::default(), WorkerStats::new()).unwrap();
    let module = runner.module(&sha256(&wasm), || Ok(wasm.clone())).unwrap();
    assert_eq!(module.abi_version(), 2);
    for v in vectors_v2() {
        let list = module.inputs_raw(&v.request, &v.frames).unwrap();
        assert_eq!(list, v.input_list, "{}: input list", v.name);
        module.sources_v2_raw(&v.request, &v.frames).unwrap();
        let out = module
            .compile_v2_raw(&v.request, &v.tiles, &v.inputs)
            .unwrap();
        assert_eq!(to_hex(&sha256(&out)), v.sha256, "{}", v.name);
    }
}

/// Serves one vector's input sections as stored (`ready`) sections, by a
/// made-up address per key, and counts the streams.
struct VectorInputs {
    sections: BTreeMap<String, Vec<u8>>,
    streams: AtomicU32,
}

impl VectorInputs {
    fn address(key: &gx_core::key::ChunkKey) -> [u8; 32] {
        sha256(key.to_string().as_bytes())
    }
}

impl InputSource for VectorInputs {
    fn input_job(
        &self,
        reader: &ComputeJob,
        slot: u32,
        key: gx_core::key::ChunkKey,
    ) -> Result<ComputeJob, WorkerError> {
        assert_eq!(slot, 1, "the example reader's layer slot");
        let mut job = reader.clone();
        job.address = VectorInputs::address(&key);
        job.request.key = key;
        Ok(job)
    }

    fn direct(&self, job: &ComputeJob) -> Result<InputDirection, WorkerError> {
        let bytes = &self.sections[&job.key().to_string()];
        Ok(InputDirection::Ready {
            sha256: sha256(bytes),
        })
    }

    fn stream(&self, address: &[u8; 32], _: &[u8; 32]) -> Result<Vec<u8>, WorkerError> {
        self.streams.fetch_add(1, Ordering::SeqCst);
        Ok(self
            .sections
            .iter()
            .find(|(k, _)| VectorInputs::address(&k.parse().unwrap()) == *address)
            .map(|(_, b)| b.clone())
            .expect("a vector input"))
    }
}

/// A version 2 job through the pool's runner (`client-compute.md`
/// sections 4.1 and 10.3): the frame records go to `gx_inputs`,
/// `gx_sources`, and `gx_compile`, and the input sections `gx_inputs`
/// names are resolved through the input source and passed after the
/// frames slot's record. Every vector is computed to its hash, and the
/// inputs hash is the one over exactly the inputs passed.
#[test]
fn runner_runs_version_2_jobs_with_their_inputs() {
    let wasm = std::fs::read(data_v2().join("module.wasm")).unwrap();
    let mut store = Store::default();
    let module = store.add(BlobKind::Module, wasm);
    let runner = ModuleRunner::new(HostConfig::default(), WorkerStats::new()).unwrap();
    let mut with_inputs = 0;
    let mut seen = std::collections::BTreeSet::new();
    for v in vectors_v2() {
        let mut job = job_for(&mut store, module, &v.request, &v.tiles);
        let frames: Vec<Arc<[u8]>> = module_io::decode_inputs(&v.frames)
            .unwrap()
            .into_iter()
            .map(Arc::from)
            .collect();
        let list = module_io::decode_input_list(&v.input_list).unwrap();
        let passed = module_io::decode_inputs(&v.inputs).unwrap();
        let source = VectorInputs {
            sections: list
                .iter()
                .zip(&passed[frames.len()..])
                .map(|(e, b)| (e.key.to_string(), b.to_vec()))
                .collect(),
            streams: AtomicU32::new(0),
        };
        if !frames.is_empty() {
            let r = &job.request;
            job.slots = vec![
                SlotValue {
                    kind: SlotKind::Frames,
                    value: [1; 32],
                },
                SlotValue {
                    kind: SlotKind::Layer,
                    value: [2; 32],
                },
            ];
            job.frames_per_slot = vec![frames.len(), 0];
            job.address = section_address_v2(
                &module,
                job.manifest.as_ref(),
                r.origin_utc_ms,
                r.span_seconds,
                r.span_index,
                &job.slots,
                &r.key,
            )
            .unwrap();
        }
        job.frames = frames;
        let resolver = Resolver {
            source: &source,
            uploader: None,
        };
        let out = runner
            .run(&job, &store, None, Some(resolver))
            .unwrap_or_else(|e| panic!("{}: {e}", v.name));
        assert_eq!(to_hex(&sha256(&out.bytes)), v.sha256, "{}", v.name);
        // An input resolved for an earlier vector is remembered by address
        // and not streamed again.
        let new = list
            .iter()
            .filter(|e| seen.insert(e.key.to_string()))
            .count();
        assert_eq!(
            source.streams.load(Ordering::SeqCst) as usize,
            new,
            "{}",
            v.name
        );
        if !job.slots.is_empty() {
            assert_eq!(
                out.inputs_sha256,
                Some(module_io::inputs_sha256(&passed)),
                "{}",
                v.name
            );
        }
        if !list.is_empty() {
            with_inputs += 1;
            // Without an input source the same job fails, never guesses.
            let e = runner.run(&job, &store, None, None).expect_err(&v.name);
            assert_eq!(e.kind, FailureKind::Permanent, "{}", v.name);
        }
    }
    assert!(with_inputs >= 3, "{with_inputs} vectors read inputs");
}

/// A version 1 module given frame records fails before it runs.
#[test]
fn runner_refuses_frame_records_for_a_version_1_module() {
    let mut store = Store::default();
    let module = store.add(BlobKind::Module, module_bytes());
    let (name, request, tiles, _) = vectors().remove(0);
    let mut job = job_for(&mut store, module, &request, &tiles);
    job.frames = vec![Arc::from(&[0u8; 88][..])];
    let runner = ModuleRunner::new(HostConfig::default(), WorkerStats::new()).unwrap();
    let e = runner.run(&job, &store, None, None).expect_err(&name);
    assert_eq!(e.kind, FailureKind::Permanent);
}

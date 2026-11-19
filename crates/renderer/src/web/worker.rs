//! The browser's worker (`space-model.md` section 2, "The worker";
//! `client-compute.md` sections 1 step 4, 4, and 5.2): modules run with the
//! browser's own `WebAssembly` in a dedicated Web Worker
//! (`web/module-worker.js`, instantiated with no imports), so computing
//! never blocks drawing.
//!
//! The page side here does everything else, through the same checks as the
//! native worker ([`gx_worker::job`]): it checks the address, fetches the
//! module, the manifest, and the tiles by hash (the local cache first,
//! [`super::cache::WebCache`]), encodes the request and the tiles with
//! `gx-core` built for WebAssembly, validates the section, and hands it
//! over before the upload starts. A version 2 module (section 4.1) gets
//! the job's frame records in `gx_inputs`, `gx_sources`, and `gx_compile`,
//! and the input sections `gx_inputs` names are resolved by the shared
//! resolver ([`gx_worker::resolve`], section 10.3) through
//! [`WebPlatform`]; a version 1 module gets empty buffers and runs as
//! before.
//!
//! The memory limit is enforced inside the Web Worker. The time limit
//! ([`TIME_LIMIT_MS`], as natively) is enforced here: a call that runs over
//! it terminates the Web Worker, which is started again for the next call.

use super::cache::WebCache;
use super::hub::WebHub;
use gx_core::key::ChunkKey;
use gx_core::module_io::{self, Tile};
use gx_core::sha256::to_hex;
use gx_worker::job::{check_blob, check_section, check_tile, decode_manifest, resolve_tiles};
use gx_worker::{
    BlobKind, ComputeJob, Computed, InputDirection, Platform, UploadLane, UploadOffer, WorkerError,
    WorkerStats,
};
use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, BTreeSet};
use std::rc::{Rc, Weak};
use std::sync::Arc;
use wasm_bindgen::closure::Closure;
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::JsFuture;

/// The script the Web Worker runs, relative to the page.
pub const WORKER_SCRIPT: &str = "./module-worker.js";

/// The time limit of one module call, milliseconds: 60 s, the native
/// host's default.
pub const TIME_LIMIT_MS: i32 = 60_000;

type Callbacks = (js_sys::Function, js_sys::Function);

/// The Web Worker's event handlers, kept alive as long as it runs.
struct Handlers {
    _on_message: Closure<dyn FnMut(web_sys::MessageEvent)>,
    _on_error: Closure<dyn FnMut(web_sys::Event)>,
}

struct Inner {
    worker: RefCell<Option<web_sys::Worker>>,
    handlers: RefCell<Option<Handlers>>,
    calls: RefCell<BTreeMap<u32, Callbacks>>,
    next_id: Cell<u32>,
    loaded: RefCell<BTreeSet<[u8; 32]>>,
}

/// Rejected when the Web Worker stopped under a call (a crash, or another
/// call's time limit): the call may succeed on a later try.
const RESTARTED: &str = "the module worker restarted";

impl Inner {
    /// Terminates the Web Worker and fails every call in flight.
    fn stop(&self, reason: &str) {
        if let Some(w) = self.worker.borrow_mut().take() {
            w.set_onmessage(None);
            w.set_onerror(None);
            w.terminate();
        }
        self.loaded.borrow_mut().clear();
        let calls = std::mem::take(&mut *self.calls.borrow_mut());
        for (_, (_, reject)) in calls {
            let _ = reject.call1(&JsValue::NULL, &JsValue::from_str(reason));
        }
    }
}

/// The dedicated Web Worker that runs modules, one call at a time.
#[derive(Clone)]
pub struct ModuleWorker {
    inner: Rc<Inner>,
}

impl Default for ModuleWorker {
    fn default() -> ModuleWorker {
        ModuleWorker::new()
    }
}

fn get(obj: &JsValue, name: &str) -> JsValue {
    js_sys::Reflect::get(obj, &JsValue::from_str(name)).unwrap_or(JsValue::UNDEFINED)
}

fn set(obj: &js_sys::Object, name: &str, value: &JsValue) {
    let _ = js_sys::Reflect::set(obj, &JsValue::from_str(name), value);
}

impl ModuleWorker {
    /// A worker; the Web Worker itself starts on the first call.
    pub fn new() -> ModuleWorker {
        ModuleWorker {
            inner: Rc::new(Inner {
                worker: RefCell::new(None),
                handlers: RefCell::new(None),
                calls: RefCell::default(),
                next_id: Cell::new(0),
                loaded: RefCell::default(),
            }),
        }
    }

    /// `true` when the Web Worker holds the module `hash`.
    pub fn has(&self, hash: &[u8; 32]) -> bool {
        self.inner.loaded.borrow().contains(hash)
    }

    fn worker(&self) -> Result<web_sys::Worker, WorkerError> {
        if let Some(w) = self.inner.worker.borrow().as_ref() {
            return Ok(w.clone());
        }
        let options = web_sys::WorkerOptions::new();
        options.set_type(web_sys::WorkerType::Module);
        let w = web_sys::Worker::new_with_options(WORKER_SCRIPT, &options)
            .map_err(|e| WorkerError::transient(format!("starting the module worker: {e:?}")))?;
        let weak: Weak<Inner> = Rc::downgrade(&self.inner);
        let on_message =
            Closure::<dyn FnMut(web_sys::MessageEvent)>::new(move |e: web_sys::MessageEvent| {
                let Some(inner) = weak.upgrade() else {
                    return;
                };
                let data = e.data();
                let Some(id) = get(&data, "id").as_f64() else {
                    return;
                };
                let Some((resolve, _)) = inner.calls.borrow_mut().remove(&(id as u32)) else {
                    return;
                };
                let _ = resolve.call1(&JsValue::NULL, &data);
            });
        let weak: Weak<Inner> = Rc::downgrade(&self.inner);
        let on_error = Closure::<dyn FnMut(web_sys::Event)>::new(move |e: web_sys::Event| {
            e.prevent_default();
            if let Some(inner) = weak.upgrade() {
                web_sys::console::warn_1(&"gx-renderer: the module worker failed".into());
                inner.stop(RESTARTED);
            }
        });
        w.set_onmessage(Some(on_message.as_ref().unchecked_ref()));
        w.set_onerror(Some(on_error.as_ref().unchecked_ref()));
        *self.inner.handlers.borrow_mut() = Some(Handlers {
            _on_message: on_message,
            _on_error: on_error,
        });
        *self.inner.worker.borrow_mut() = Some(w.clone());
        Ok(w)
    }

    /// Posts one message and waits for its answer: the bytes, or the
    /// reason the call failed.
    async fn call(&self, msg: js_sys::Object) -> Result<Vec<u8>, WorkerError> {
        let worker = self.worker()?;
        let id = self.inner.next_id.get();
        self.inner.next_id.set(id.wrapping_add(1));
        set(&msg, "id", &JsValue::from(id));
        let calls = Rc::downgrade(&self.inner);
        let promise = js_sys::Promise::new(&mut |resolve, reject| {
            if let Some(inner) = calls.upgrade() {
                inner.calls.borrow_mut().insert(id, (resolve, reject));
            }
        });
        if let Err(e) = worker.post_message(&msg) {
            self.inner.calls.borrow_mut().remove(&id);
            return Err(WorkerError::transient(format!(
                "posting to the module worker: {e:?}"
            )));
        }
        let weak = Rc::downgrade(&self.inner);
        let timeout = Closure::once_into_js(move || {
            let Some(inner) = weak.upgrade() else {
                return;
            };
            let Some((_, reject)) = inner.calls.borrow_mut().remove(&id) else {
                return;
            };
            let _ = reject.call1(&JsValue::NULL, &JsValue::from_str("time limit"));
            inner.stop(RESTARTED);
        });
        if let Some(w) = web_sys::window() {
            let _ = w.set_timeout_with_callback_and_timeout_and_arguments_0(
                timeout.unchecked_ref(),
                TIME_LIMIT_MS,
            );
        }
        match JsFuture::from(promise).await {
            Ok(answer) => {
                if get(&answer, "ok").as_bool() == Some(true) {
                    let bytes: js_sys::Uint8Array = get(&answer, "bytes").unchecked_into();
                    Ok(bytes.to_vec())
                } else {
                    let reason = get(&answer, "error")
                        .as_string()
                        .unwrap_or_else(|| "the module failed".into());
                    Err(WorkerError::permanent(reason))
                }
            }
            Err(e) => {
                let reason = e.as_string().unwrap_or_else(|| RESTARTED.into());
                if reason == "time limit" {
                    Err(WorkerError::permanent(format!(
                        "the module exceeded the time limit of {TIME_LIMIT_MS} ms"
                    )))
                } else {
                    Err(WorkerError::transient(reason))
                }
            }
        }
    }

    fn message(op: &str, hash: &[u8; 32]) -> js_sys::Object {
        let msg = js_sys::Object::new();
        set(&msg, "op", &JsValue::from_str(op));
        set(&msg, "hash", &JsValue::from_str(&to_hex(hash)));
        msg
    }

    /// Compiles the module `hash` from `wasm` in the Web Worker. A module
    /// that imports anything is refused there.
    pub async fn load(&self, hash: &[u8; 32], wasm: &[u8]) -> Result<(), WorkerError> {
        let msg = Self::message("load", hash);
        set(&msg, "wasm", &js_sys::Uint8Array::from(wasm));
        self.call(msg).await?;
        self.inner.loaded.borrow_mut().insert(*hash);
        Ok(())
    }

    /// Runs `gx_inputs` (`client-compute.md` section 4.1): the input list
    /// bytes for `request` and the frame bytes `frames`. A version 1 module
    /// reads no inputs: the empty list.
    pub async fn inputs(
        &self,
        hash: &[u8; 32],
        request: &[u8],
        frames: &[u8],
    ) -> Result<Vec<u8>, WorkerError> {
        let msg = Self::message("inputs", hash);
        set(&msg, "request", &js_sys::Uint8Array::from(request));
        set(&msg, "frames", &js_sys::Uint8Array::from(frames));
        self.call(msg).await
    }

    /// Runs `gx_sources`: the source id list bytes for `request`, with the
    /// frame bytes `frames` for a version 2 module (empty for version 1).
    pub async fn sources(
        &self,
        hash: &[u8; 32],
        request: &[u8],
        frames: &[u8],
    ) -> Result<Vec<u8>, WorkerError> {
        let msg = Self::message("sources", hash);
        set(&msg, "request", &js_sys::Uint8Array::from(request));
        set(&msg, "frames", &js_sys::Uint8Array::from(frames));
        self.call(msg).await
    }

    /// Runs `gx_compile`: the section bytes for `request` and `tiles`, with
    /// the input bytes `inputs` for a version 2 module (empty for version
    /// 1).
    pub async fn compile(
        &self,
        hash: &[u8; 32],
        request: &[u8],
        tiles: &[u8],
        inputs: &[u8],
    ) -> Result<Vec<u8>, WorkerError> {
        let msg = Self::message("compile", hash);
        set(&msg, "request", &js_sys::Uint8Array::from(request));
        set(&msg, "tiles", &js_sys::Uint8Array::from(tiles));
        set(&msg, "inputs", &js_sys::Uint8Array::from(inputs));
        self.call(msg).await
    }
}

/// Gets content-addressed bytes: the cache first (checked, dropped if
/// damaged), then the edge, whose bytes are checked and cached.
pub async fn get_blob(
    hub: &WebHub,
    cache: Option<&WebCache>,
    kind: BlobKind,
    hash: &[u8; 32],
) -> Result<Vec<u8>, WorkerError> {
    let name = to_hex(hash);
    if let Some(c) = cache {
        if let Some(bytes) = c.get(kind, &name).await {
            if check_blob(kind, hash, &bytes).is_ok() {
                return Ok(bytes);
            }
            c.remove(kind, &name).await;
        }
    }
    let bytes = match hub.get_blob(kind, &name).await {
        Ok(Some(b)) => b,
        Ok(None) => return Err(WorkerError::transient(format!("{kind} {name}: not found"))),
        Err(e) => return Err(WorkerError::transient(format!("{e:#}"))),
    };
    check_blob(kind, hash, &bytes)?;
    if let Some(c) = cache {
        if let Err(e) = c.put(kind, &name, &bytes).await {
            web_sys::console::warn_1(&format!("gx-renderer: caching {kind} {name}: {e}").into());
        }
    }
    Ok(bytes)
}

thread_local! {
    /// Source manifests decoded this session, by hash: one lists every
    /// source tile (tens of megabytes) and every section of its layer needs
    /// it, so it is fetched, checked, and decoded once.
    static MANIFESTS: RefCell<BTreeMap<[u8; 32], Rc<gx_core::manifest::Manifest>>> =
        const { RefCell::new(BTreeMap::new()) };
}

/// The source manifest `hash`, decoded once per session ([`MANIFESTS`]).
async fn manifest(
    hub: &WebHub,
    cache: Option<&WebCache>,
    hash: &[u8; 32],
) -> Result<Rc<gx_core::manifest::Manifest>, WorkerError> {
    if let Some(m) = MANIFESTS.with(|m| m.borrow().get(hash).cloned()) {
        return Ok(m);
    }
    let m = Rc::new(decode_manifest(
        &get_blob(hub, cache, BlobKind::Manifest, hash).await?,
    )?);
    MANIFESTS.with(|ms| ms.borrow_mut().insert(*hash, Rc::clone(&m)));
    Ok(m)
}

/// The browser platform of the shared input resolver
/// ([`gx_worker::resolve`], `client-compute.md` section 10.3): the hub over
/// `fetch` for directions and streams, the Cache API, the Web Worker
/// module host, the build and frame records for input jobs, and the upload
/// lane for computed `missing` inputs. Everything runs on the page's event
/// loop; there is one module call at a time, so there is no queue whose
/// priority a waiter could raise.
pub struct WebPlatform<'a> {
    /// The hub.
    pub hub: &'a WebHub,
    /// The page's cache, if it has one.
    pub cache: Option<&'a WebCache>,
    /// The Web Worker module host.
    pub worker: &'a ModuleWorker,
    /// The counters.
    pub stats: &'a WorkerStats,
    /// The frame records the build lists, by hash.
    pub records: &'a crate::protocol::FrameRecords,
    /// The space's origin, once directions have told it.
    pub origin: Option<i64>,
    /// Takes computed `missing` inputs.
    pub lane: &'a UploadLane,
}

impl Platform for WebPlatform<'_> {
    fn input_job(
        &self,
        reader: &ComputeJob,
        slot: u32,
        key: ChunkKey,
    ) -> Result<ComputeJob, WorkerError> {
        let origin = self
            .origin
            .ok_or_else(|| WorkerError::transient("the origin is not known yet"))?;
        self.hub
            .build()
            .input_job(reader, slot, key, origin, self.records)
            .map_err(|e| WorkerError::permanent(format!("input {key} of slot {slot}: {e:#}")))
    }

    async fn direct(&self, job: &ComputeJob) -> Result<InputDirection, WorkerError> {
        let origin = self
            .origin
            .ok_or_else(|| WorkerError::transient("the origin is not known yet"))?;
        let t = crate::protocol::input_directions_time(job, origin, self.hub.now_utc_ms());
        let key = job.key();
        let response =
            self.hub.directions(t, &[key]).await.map_err(|e| {
                WorkerError::transient(format!("directions for input {key}: {e:#}"))
            })?;
        crate::protocol::input_direction(&response, job)
    }

    async fn stream(&self, address: &[u8; 32], sha256: &[u8; 32]) -> Result<Vec<u8>, WorkerError> {
        match self.hub.get_section(address, sha256).await {
            Ok(Some(b)) => Ok(b),
            Ok(None) => Err(WorkerError::transient(format!(
                "input section {}: not found",
                to_hex(address)
            ))),
            Err(e) => Err(WorkerError::transient(format!("{e:#}"))),
        }
    }

    async fn input_list(&self, job: &ComputeJob) -> Result<Vec<(u32, ChunkKey)>, WorkerError> {
        if !self.worker.has(&job.module) {
            let wasm = get_blob(self.hub, self.cache, BlobKind::Module, &job.module).await?;
            self.worker.load(&job.module, &wasm).await?;
            self.stats.module_loaded();
        }
        let request = module_io::encode_request(&job.request);
        self.stats.module_run();
        let list = self
            .worker
            .inputs(&job.module, &request, &frames(job)?)
            .await?;
        let list = module_io::decode_input_list(&list)
            .map_err(|e| WorkerError::permanent(format!("gx_inputs output is invalid: {e}")))?;
        Ok(list.into_iter().map(|e| (e.slot, e.key)).collect())
    }

    async fn compile(
        &self,
        job: &ComputeJob,
        inputs: &[Arc<[u8]>],
    ) -> Result<Computed, WorkerError> {
        let request = module_io::encode_request(&job.request);
        let ids = self
            .worker
            .sources(&job.module, &request, &frames(job)?)
            .await?;
        let ids = module_io::decode_source_ids(&ids)
            .map_err(|e| WorkerError::permanent(format!("gx_sources output is invalid: {e}")))?;
        let manifest = match &job.manifest {
            Some(h) if !ids.is_empty() => Some(manifest(self.hub, self.cache, h).await?),
            _ => None,
        };
        let refs = resolve_tiles(manifest.as_deref(), &ids)?;
        let mut tiles = Vec::with_capacity(refs.len());
        for r in &refs {
            let bytes = get_blob(self.hub, self.cache, BlobKind::Source, &r.sha256).await?;
            check_tile(r, &bytes)?;
            tiles.push((r.id, bytes));
        }
        let tile_list: Vec<Tile<'_>> = tiles
            .iter()
            .map(|(id, bytes)| Tile { id: *id, bytes })
            .collect();
        let tiles = module_io::encode_tiles(&tile_list)
            .map_err(|e| WorkerError::permanent(format!("encoding the tiles: {e}")))?;
        let passed: Vec<&[u8]> = inputs.iter().map(|b| b.as_ref()).collect();
        let passed = module_io::encode_inputs(&passed)
            .map_err(|e| WorkerError::permanent(format!("encoding the inputs: {e}")))?;
        self.stats.module_run();
        let bytes = self
            .worker
            .compile(&job.module, &request, &tiles, &passed)
            .await?;
        let section = check_section(&job.request.key, &bytes)?;
        Ok(Computed {
            bytes: Arc::from(bytes),
            section: section.map(Arc::new),
            inputs_sha256: None,
        })
    }

    async fn cache_get(&self, address: &[u8; 32]) -> Option<Vec<u8>> {
        self.cache?.get(BlobKind::Section, &to_hex(address)).await
    }

    async fn cache_put(&self, address: &[u8; 32], bytes: &Arc<[u8]>) {
        if let Some(c) = self.cache {
            let _ = c.put(BlobKind::Section, &to_hex(address), bytes).await;
        }
    }

    fn offer(&self, job: &ComputeJob, bytes: Arc<[u8]>) {
        self.lane.offer(UploadOffer {
            job: job.clone(),
            bytes,
        });
    }

    fn input_computed(&self) {
        self.stats.job_finished(true);
    }
}

/// The job's frame records as one input buffer (section 4.1), what
/// `gx_inputs` and `gx_sources` read.
fn frames(job: &ComputeJob) -> Result<Vec<u8>, WorkerError> {
    let frames: Vec<&[u8]> = job.frames.iter().map(|f| f.as_ref()).collect();
    module_io::encode_inputs(&frames)
        .map_err(|e| WorkerError::permanent(format!("encoding the frame records: {e}")))
}

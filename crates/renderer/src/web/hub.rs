//! The hub client in the browser: chunks over `fetch`, readiness over a
//! `WebSocket`, the registry, and the cell fetcher of the frame loop.
//!
//! The same protocol as the native client ([`crate::hub`], built on
//! [`crate::protocol`]): `GET /space/{s}/build/{b}/chunk/{key}` with the
//! `X-API-Key` header, `200` / `202` / `404` / `410`, and `chunkReady` pushes
//! on `/space/{s}/build/{b}/chunks/ready` (`compiler-pipeline.md` section
//! 4). Requests go through `reqwest`'s wasm client (the browser's `fetch`);
//! the socket is a `web_sys::WebSocket`; futures run on the page's event
//! loop through `wasm-bindgen-futures`.
//!
//! A browser cannot set headers on a WebSocket upgrade, so the socket opens
//! without the API key. A hub that refuses it leaves the client on the
//! polling fallback the protocol allows: the cell cache polls a pending
//! cell every 2 s once it has waited 10 s, and the registry is polled every
//! 2 s. The hub must also allow the page's origin (CORS) for the `X-API-Key`
//! header.

use crate::config::ApiKey;
use crate::protocol::{
    builds_url, chunk_url, classify_status, frame_system_from_chunk, parse_ready_frame,
    pick_active_build, ready_url, subscribe_frame, BuildSummary, ChunkFetch, API_KEY_HEADER,
};
use crate::stream::{decode_cell, FetchEvent, FetchOutcome, FetchTally};
use anyhow::{anyhow, bail, Context, Result};
use gx_core::frames::FrameSystem;
use gx_core::key::{CellKey, ChunkKey, REGISTRY_KEY};
use gx_core::units::Meters;
use std::cell::{Cell, RefCell};
use std::collections::VecDeque;
use std::rc::Rc;
use std::sync::Arc;
use std::time::Duration;
use wasm_bindgen::closure::Closure;
use wasm_bindgen::JsCast;

/// Seconds between polls of the registry while the hub answers `202`.
pub const REGISTRY_POLL_SECONDS: f64 = 2.0;

/// Longest wait for the registry, seconds.
pub const REGISTRY_DEADLINE_SECONDS: f64 = 60.0;

/// Seconds since the page loaded, from `performance.now()`.
pub fn now_seconds() -> f64 {
    web_sys::window()
        .and_then(|w| w.performance())
        .map_or(0.0, |p| p.now() / 1000.0)
}

/// Waits `seconds` on the page's event loop.
pub async fn sleep(seconds: f64) {
    let promise = js_sys::Promise::new(&mut |resolve, _| {
        if let Some(w) = web_sys::window() {
            let _ = w.set_timeout_with_callback_and_timeout_and_arguments_0(
                &resolve,
                (seconds * 1000.0).round() as i32,
            );
        }
    });
    let _ = wasm_bindgen_futures::JsFuture::from(promise).await;
}

/// A connection to one build of one space on the hub.
pub struct WebHub {
    http: reqwest::Client,
    base_url: String,
    api_key: ApiKey,
    space_id: String,
    build_id: String,
}

impl WebHub {
    /// A client for `build_id`, or for the active build of the space
    /// (`GET /space/{spaceId}/builds`) when `build_id` is `None`.
    pub async fn connect(
        base_url: &str,
        api_key: ApiKey,
        space_id: &str,
        build_id: Option<&str>,
    ) -> Result<WebHub> {
        let mut hub = WebHub {
            http: reqwest::Client::new(),
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key,
            space_id: space_id.to_string(),
            build_id: String::new(),
        };
        hub.build_id = match build_id {
            Some(b) => b.to_string(),
            None => hub.active_build().await?,
        };
        Ok(hub)
    }

    /// The build this client reads.
    pub fn build_id(&self) -> &str {
        &self.build_id
    }

    /// The readiness WebSocket URL of the build.
    pub fn ready_url(&self) -> String {
        ready_url(&self.base_url, &self.space_id, &self.build_id)
    }

    async fn active_build(&self) -> Result<String> {
        let resp = self
            .http
            .get(builds_url(&self.base_url, &self.space_id))
            .header(API_KEY_HEADER, self.api_key.expose())
            .send()
            .await
            .map_err(|e| anyhow!("listing builds: {}", e.without_url()))?;
        let status = resp.status().as_u16();
        if status != 200 {
            bail!("listing builds: hub answered {status}");
        }
        let body = resp
            .bytes()
            .await
            .map_err(|e| anyhow!("reading the builds listing: {}", e.without_url()))?;
        let builds: Vec<BuildSummary> =
            serde_json::from_slice(&body).context("decoding the builds listing")?;
        pick_active_build(&builds)
            .map(str::to_string)
            .ok_or_else(|| anyhow!("the space has no active build; enter a build id"))
    }

    /// Fetches one chunk.
    pub async fn fetch_chunk(&self, key: &str) -> ChunkFetch {
        let resp = match self
            .http
            .get(chunk_url(
                &self.base_url,
                &self.space_id,
                &self.build_id,
                key,
            ))
            .header(API_KEY_HEADER, self.api_key.expose())
            .send()
            .await
        {
            Ok(r) => r,
            Err(e) => return ChunkFetch::Error(format!("request failed: {}", e.without_url())),
        };
        if let Some(outcome) = classify_status(resp.status().as_u16()) {
            return outcome;
        }
        match resp.bytes().await {
            Ok(body) => ChunkFetch::Ready(Arc::from(body.as_ref())),
            Err(e) => ChunkFetch::Error(format!("reading body failed: {}", e.without_url())),
        }
    }
}

/// The readiness socket: subscriptions out, ready keys in. Subscriptions
/// made before the socket opens are sent when it does.
pub struct ReadySocket {
    socket: web_sys::WebSocket,
    open: Rc<Cell<bool>>,
    queued: Rc<RefCell<Vec<String>>>,
    ready: Rc<RefCell<VecDeque<String>>>,
    _on_open: Closure<dyn FnMut()>,
    _on_message: Closure<dyn FnMut(web_sys::MessageEvent)>,
    _on_close: Closure<dyn FnMut()>,
}

impl ReadySocket {
    /// Opens the socket at `url`, or `None` if the browser refuses the URL.
    pub fn open(url: &str) -> Option<ReadySocket> {
        let socket = web_sys::WebSocket::new(url).ok()?;
        let open = Rc::new(Cell::new(false));
        let queued: Rc<RefCell<Vec<String>>> = Rc::default();
        let ready: Rc<RefCell<VecDeque<String>>> = Rc::default();
        let on_open = {
            let (open, queued, socket) = (open.clone(), queued.clone(), socket.clone());
            Closure::<dyn FnMut()>::new(move || {
                open.set(true);
                for text in queued.borrow_mut().drain(..) {
                    let _ = socket.send_with_str(&text);
                }
            })
        };
        let on_message = {
            let ready = ready.clone();
            Closure::<dyn FnMut(web_sys::MessageEvent)>::new(move |e: web_sys::MessageEvent| {
                if let Some(key) = e.data().as_string().and_then(|t| parse_ready_frame(&t)) {
                    ready.borrow_mut().push_back(key);
                }
            })
        };
        let on_close = {
            let open = open.clone();
            Closure::<dyn FnMut()>::new(move || open.set(false))
        };
        socket.set_onopen(Some(on_open.as_ref().unchecked_ref()));
        socket.set_onmessage(Some(on_message.as_ref().unchecked_ref()));
        socket.set_onclose(Some(on_close.as_ref().unchecked_ref()));
        Some(ReadySocket {
            socket,
            open,
            queued,
            ready,
            _on_open: on_open,
            _on_message: on_message,
            _on_close: on_close,
        })
    }

    /// Registers interest in `key`.
    pub fn subscribe(&self, key: &str) {
        let text = subscribe_frame(key);
        if self.open.get() {
            let _ = self.socket.send_with_str(&text);
        } else {
            self.queued.borrow_mut().push(text);
        }
    }

    /// The keys the hub has reported ready since the last call.
    pub fn take_ready(&self) -> Vec<String> {
        self.ready.borrow_mut().drain(..).collect()
    }
}

impl Drop for ReadySocket {
    fn drop(&mut self) {
        self.socket.set_onopen(None);
        self.socket.set_onmessage(None);
        self.socket.set_onclose(None);
        let _ = self.socket.close();
    }
}

/// Fetches the registry, waiting while the hub answers `202` for a
/// readiness push or the next poll every [`REGISTRY_POLL_SECONDS`], up to
/// [`REGISTRY_DEADLINE_SECONDS`], then decodes and validates it.
pub async fn load_registry(hub: &WebHub, socket: Option<&ReadySocket>) -> Result<FrameSystem> {
    if let Some(s) = socket {
        s.subscribe(REGISTRY_KEY);
    }
    let start = now_seconds();
    loop {
        match hub.fetch_chunk(REGISTRY_KEY).await {
            ChunkFetch::Ready(bytes) => return frame_system_from_chunk(&bytes),
            ChunkFetch::Pending => {}
            ChunkFetch::NotFound => bail!("registry: not found (404), the build has no layers"),
            ChunkFetch::Gone => bail!("registry: gone (410), it can never be produced"),
            ChunkFetch::Error(e) => bail!("registry: {e}"),
        }
        let asked = now_seconds();
        loop {
            if now_seconds() - start > REGISTRY_DEADLINE_SECONDS {
                bail!("registry did not become ready within {REGISTRY_DEADLINE_SECONDS} s");
            }
            let pushed = socket.is_some_and(|s| s.take_ready().iter().any(|k| k == REGISTRY_KEY));
            if pushed || now_seconds() - asked >= REGISTRY_POLL_SECONDS {
                break;
            }
            sleep(0.1).await;
        }
    }
}

/// Fetches cells for the frame loop. Each request runs as a future on the
/// page's event loop and leaves its result, decoded, in a queue the frame
/// loop drains; readiness pushes arrive the same way.
pub struct WebFetcher {
    hub: Rc<WebHub>,
    socket: Option<Rc<ReadySocket>>,
    events: Rc<RefCell<Vec<FetchEvent>>>,
    tally: FetchTally,
}

impl WebFetcher {
    /// A fetcher over `hub`, with the readiness socket if it opened.
    pub fn new(hub: Rc<WebHub>, socket: Option<ReadySocket>) -> WebFetcher {
        WebFetcher {
            hub,
            socket: socket.map(Rc::new),
            events: Rc::default(),
            tally: FetchTally::default(),
        }
    }

    /// Starts a request for `key`, a cell of a frame of edge `root_extent`.
    pub fn request(&self, key: CellKey, root_extent: Meters) {
        let hub = self.hub.clone();
        let socket = self.socket.clone();
        let events = self.events.clone();
        wasm_bindgen_futures::spawn_local(async move {
            let name = key.to_string();
            let start = now_seconds();
            let fetched = hub.fetch_chunk(&name).await;
            let round_trip = Duration::from_secs_f64((now_seconds() - start).max(0.0));
            let (outcome, bytes) = match fetched {
                ChunkFetch::Ready(body) => {
                    (decode_cell(&key, &body, root_extent), body.len() as u64)
                }
                ChunkFetch::Pending => {
                    if let Some(s) = &socket {
                        s.subscribe(&name);
                    }
                    (FetchOutcome::Pending, 0)
                }
                ChunkFetch::NotFound => (FetchOutcome::NotFound, 0),
                ChunkFetch::Gone => (FetchOutcome::Gone, 0),
                ChunkFetch::Error(e) => (FetchOutcome::Failed(e), 0),
            };
            events.borrow_mut().push(FetchEvent::Completed {
                key,
                outcome,
                bytes,
                round_trip,
            });
        });
    }

    /// Every event so far: finished requests, then readiness pushes.
    pub fn drain(&mut self) -> Vec<FetchEvent> {
        let mut events: Vec<FetchEvent> = self.events.borrow_mut().drain(..).collect();
        if let Some(s) = &self.socket {
            for k in s.take_ready() {
                if let Ok(ChunkKey::Cell(cell)) = k.parse::<ChunkKey>() {
                    events.push(FetchEvent::ChunkReady(cell));
                }
            }
        }
        self.tally.record(&events);
        events
    }

    /// The totals over every finished request this session.
    pub fn tally(&self) -> FetchTally {
        self.tally
    }
}

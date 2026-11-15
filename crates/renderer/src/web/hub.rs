//! The hub client in the browser, version 2: the clock, the build, and
//! directions over `fetch`, content-addressed reads through the edge,
//! uploads with a claim, and the readiness socket keyed by address.
//!
//! The same routes as the native client ([`crate::hub`], both built on
//! [`crate::protocol`]; `client-compute.md` sections 2.3 to 2.5). Requests
//! go through `reqwest`'s wasm client (the browser's `fetch`); the socket
//! is a `web_sys::WebSocket`; futures run on the page's event loop through
//! `wasm-bindgen-futures`.
//!
//! A browser cannot set headers on a WebSocket upgrade, so the socket opens
//! without the API key. A hub that refuses it leaves the client on the
//! fallback the protocol allows: a `pending` key is asked about again after
//! the hub's `retryMs`. The hub must also allow the page's origin (CORS)
//! for the `X-API-Key` and `X-Claim` headers.

use crate::config::ApiKey;
use crate::protocol::{
    parse_ready_frame, status_problem, subscribe_frame, ActiveBuild, BuildDescription, Clock,
    DirectionsRequest, DirectionsResponse, Routes, API_KEY_HEADER, CLAIM_HEADER,
    MAX_DIRECTION_KEYS,
};
use anyhow::{anyhow, bail, Context, Result};
use gx_core::key::ChunkKey;
use gx_core::sha256::to_hex;
use gx_worker::BlobKind;
use serde::de::DeserializeOwned;
use std::cell::{Cell, RefCell};
use std::collections::VecDeque;
use std::rc::Rc;
use wasm_bindgen::closure::Closure;
use wasm_bindgen::JsCast;

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
    routes: Routes,
    api_key: ApiKey,
    build: BuildDescription,
    clock: Clock,
    clock_read_at: f64,
}

impl WebHub {
    /// Connects through `routes`: reads the clock, resolves the active
    /// build when `build_id` is `None`, and reads the build's layers.
    pub async fn open(routes: Routes, api_key: ApiKey, build_id: Option<&str>) -> Result<WebHub> {
        let mut hub = WebHub {
            http: reqwest::Client::new(),
            routes,
            api_key,
            build: BuildDescription {
                build_id: String::new(),
                layers: Vec::new(),
            },
            clock: Clock {
                origin_utc_ms: None,
                now_utc_ms: 0,
            },
            clock_read_at: now_seconds(),
        };
        hub.clock = hub
            .get_json(&hub.routes.clock(), "reading the clock")
            .await?;
        hub.clock_read_at = now_seconds();
        let build_id = match build_id {
            Some(b) => b.to_string(),
            None => {
                let a: ActiveBuild = hub
                    .get_json(&hub.routes.active_build(), "reading the active build")
                    .await?;
                a.build_id
            }
        };
        hub.build = hub
            .get_json(&hub.routes.build(&build_id), "reading the build")
            .await?;
        if hub.build.build_id.is_empty() {
            hub.build.build_id = build_id;
        }
        Ok(hub)
    }

    async fn get_json<T: DeserializeOwned>(&self, url: &str, what: &str) -> Result<T> {
        let resp = self
            .http
            .get(url)
            .header(API_KEY_HEADER, self.api_key.expose())
            .send()
            .await
            .map_err(|e| anyhow!("{what}: {}", e.without_url()))?;
        if let Some(p) = status_problem(resp.status().as_u16()) {
            bail!("{what}: {p}");
        }
        let body = resp
            .bytes()
            .await
            .map_err(|e| anyhow!("{what}: {}", e.without_url()))?;
        serde_json::from_slice(&body).with_context(|| format!("{what}: decoding the answer"))
    }

    /// The routes this client uses.
    pub fn routes(&self) -> &Routes {
        &self.routes
    }

    /// The build this client reads.
    pub fn build_id(&self) -> &str {
        &self.build.build_id
    }

    /// The build's layers.
    pub fn build(&self) -> &BuildDescription {
        &self.build
    }

    /// The hub's now, UTC milliseconds: the clock read at connect time
    /// advanced by the page's time since.
    pub fn now_utc_ms(&self) -> i64 {
        let elapsed_ms = ((now_seconds() - self.clock_read_at) * 1000.0).max(0.0);
        self.clock.now_utc_ms.saturating_add(elapsed_ms as i64)
    }

    /// Asks for directions for at most 512 keys at `t_utc_ms`.
    pub async fn directions(&self, t_utc_ms: i64, keys: &[ChunkKey]) -> Result<DirectionsResponse> {
        if keys.len() > MAX_DIRECTION_KEYS {
            bail!(
                "{} keys; at most {MAX_DIRECTION_KEYS} per request",
                keys.len()
            );
        }
        let body = DirectionsRequest {
            t_utc_ms,
            keys: keys.iter().map(ChunkKey::to_string).collect(),
        };
        let resp = self
            .http
            .post(self.routes.directions(self.build_id()))
            .header(API_KEY_HEADER, self.api_key.expose())
            .header("Content-Type", "application/json")
            .body(serde_json::to_vec(&body).context("directions: encoding the request")?)
            .send()
            .await
            .map_err(|e| anyhow!("directions: {}", e.without_url()))?;
        let status = resp.status().as_u16();
        let bytes = resp
            .bytes()
            .await
            .map_err(|e| anyhow!("directions: {}", e.without_url()))?;
        if let Some(p) = status_problem(status) {
            let detail = String::from_utf8_lossy(&bytes[..bytes.len().min(300)]).into_owned();
            bail!("directions: {p}: {detail}");
        }
        serde_json::from_slice(&bytes).context("directions: decoding the answer")
    }

    /// Reads `name` of `kind` from the edge: `Some` bytes for `200`, `None`
    /// for `404`.
    pub async fn get_blob(&self, kind: BlobKind, name: &str) -> Result<Option<Vec<u8>>> {
        let resp = self
            .http
            .get(self.routes.blob(kind, name))
            .header(API_KEY_HEADER, self.api_key.expose())
            .send()
            .await
            .map_err(|e| anyhow!("{kind} {name}: {}", e.without_url()))?;
        let status = resp.status().as_u16();
        if status == 404 {
            return Ok(None);
        }
        if let Some(p) = status_problem(status) {
            bail!("{kind} {name}: {p}");
        }
        let body = resp
            .bytes()
            .await
            .map_err(|e| anyhow!("{kind} {name}: {}", e.without_url()))?;
        Ok(Some(body.to_vec()))
    }

    /// Uploads a computed section with its claim: `201` stored, `200`
    /// already stored, anything else an error with the hub's reason.
    pub async fn upload(&self, address: &[u8; 32], claim: &str, bytes: Vec<u8>) -> Result<u16> {
        let address = to_hex(address);
        let resp = self
            .http
            .put(self.routes.upload(self.build_id(), &address))
            .header(API_KEY_HEADER, self.api_key.expose())
            .header(CLAIM_HEADER, claim)
            .header("Content-Type", "application/octet-stream")
            .body(bytes)
            .send()
            .await
            .map_err(|e| anyhow!("upload {address}: {}", e.without_url()))?;
        let status = resp.status().as_u16();
        if status == 200 || status == 201 {
            return Ok(status);
        }
        let body = resp.bytes().await.unwrap_or_default();
        let detail = String::from_utf8_lossy(&body[..body.len().min(300)]).into_owned();
        bail!("upload {address}: hub answered {status}: {detail}")
    }
}

/// The readiness socket: subscriptions out, stored addresses in.
/// Subscriptions made before the socket opens are sent when it does.
pub struct ReadySocket {
    socket: web_sys::WebSocket,
    open: Rc<Cell<bool>>,
    closed: Rc<Cell<bool>>,
    queued: Rc<RefCell<Vec<String>>>,
    ready: Rc<RefCell<VecDeque<[u8; 32]>>>,
    _on_open: Closure<dyn FnMut()>,
    _on_message: Closure<dyn FnMut(web_sys::MessageEvent)>,
    _on_close: Closure<dyn FnMut()>,
}

impl ReadySocket {
    /// Opens the socket at `url`, or `None` if the browser refuses the URL.
    pub fn open(url: &str) -> Option<ReadySocket> {
        let socket = web_sys::WebSocket::new(url).ok()?;
        let open = Rc::new(Cell::new(false));
        let closed = Rc::new(Cell::new(false));
        let queued: Rc<RefCell<Vec<String>>> = Rc::default();
        let ready: Rc<RefCell<VecDeque<[u8; 32]>>> = Rc::default();
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
                if let Some(a) = e.data().as_string().and_then(|t| parse_ready_frame(&t)) {
                    ready.borrow_mut().push_back(a);
                }
            })
        };
        let on_close = {
            let (open, closed) = (open.clone(), closed.clone());
            Closure::<dyn FnMut()>::new(move || {
                open.set(false);
                closed.set(true);
            })
        };
        socket.set_onopen(Some(on_open.as_ref().unchecked_ref()));
        socket.set_onmessage(Some(on_message.as_ref().unchecked_ref()));
        socket.set_onclose(Some(on_close.as_ref().unchecked_ref()));
        Some(ReadySocket {
            socket,
            open,
            closed,
            queued,
            ready,
            _on_open: on_open,
            _on_message: on_message,
            _on_close: on_close,
        })
    }

    /// Registers interest in `address`.
    pub fn subscribe(&self, address: &[u8; 32]) {
        let text = subscribe_frame(&to_hex(address));
        if self.open.get() {
            let _ = self.socket.send_with_str(&text);
        } else {
            self.queued.borrow_mut().push(text);
        }
    }

    /// `true` once the socket has closed (or failed to open).
    pub fn is_closed(&self) -> bool {
        self.closed.get()
    }

    /// The addresses the hub has reported stored since the last call.
    pub fn take_ready(&self) -> Vec<[u8; 32]> {
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

//! The hub client in the browser, version 2: the clock, the build, and
//! directions over `fetch`, content-addressed reads through the edge, and
//! uploads with the section 2.5 headers. There is no readiness socket.
//!
//! The same routes as the native client ([`crate::hub`], both built on
//! [`crate::protocol`]; `client-compute.md` sections 2.3 to 2.5). Requests
//! go through `reqwest`'s wasm client (the browser's `fetch`); futures run
//! on the page's event loop through `wasm-bindgen-futures`.
//!
//! A page cannot set `Expect: 100-continue` (fetch forbids it), so a
//! browser upload always sends its body. The hub must allow the page's
//! origin (CORS) for the `X-API-Key`, `X-Layer`, `X-Key`, `X-Span-Index`,
//! and `X-Content-Sha256` headers.

use crate::config::ApiKey;
use crate::protocol::{
    status_problem, upload_headers, ActiveBuild, BuildDescription, Clock, ClockSync,
    DirectionsRequest, DirectionsResponse, Routes, API_KEY_HEADER, MAX_DIRECTION_KEYS,
};
use anyhow::{anyhow, bail, Context, Result};
use gx_core::key::ChunkKey;
use gx_core::sha256::to_hex;
use gx_worker::{BlobKind, ComputeJob};
use serde::de::DeserializeOwned;
use std::cell::Cell;
use std::rc::Rc;

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
    clock: Cell<ClockSync>,
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
            clock: Cell::new(ClockSync::measure(
                Clock {
                    origin_utc_ms: None,
                    now_utc_ms: 0,
                },
                0.0,
                0.0,
            )),
        };
        hub.sync_clock().await?;
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

    /// The hub's now, UTC milliseconds: the latest measurement of the
    /// hub's clock advanced by `performance.now()` since ([`ClockSync`]).
    pub fn now_utc_ms(&self) -> i64 {
        self.clock.get().now_utc_ms(now_seconds())
    }

    /// Measures the offset from the hub's clock again with `GET
    /// /space/{s}/clock`. Done at connect time and, by
    /// [`WebHub::keep_clock`], every
    /// [`crate::protocol::CLOCK_SYNC_SECONDS`].
    pub async fn sync_clock(&self) -> Result<()> {
        let sent = now_seconds();
        let clock: Clock = self
            .get_json(&self.routes.clock(), "reading the clock")
            .await?;
        let mut sync = self.clock.get();
        sync.update(clock, sent, now_seconds());
        self.clock.set(sync);
        Ok(())
    }

    /// Keeps the clock measured: a task on the page's event loop that
    /// measures it again every [`crate::protocol::CLOCK_SYNC_SECONDS`] for
    /// as long as the hub lives. A failed measurement keeps the previous
    /// one and is logged.
    pub fn keep_clock(self: &Rc<Self>) {
        let weak = Rc::downgrade(self);
        wasm_bindgen_futures::spawn_local(async move {
            loop {
                sleep(crate::protocol::CLOCK_SYNC_SECONDS).await;
                let Some(hub) = weak.upgrade() else {
                    return;
                };
                if let Err(e) = hub.sync_clock().await {
                    web_sys::console::warn_1(&format!("gx-renderer: {e:#}").into());
                }
            }
        });
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
        self.get_url(&self.routes.blob(kind, name), &format!("{kind} {name}"))
            .await
    }

    /// Reads the version of the section at `address` whose bytes hash to
    /// `sha256`: `GET /section/{address}?sha256={hex}` on the edge. The
    /// caller checks the hash.
    pub async fn get_section(
        &self,
        address: &[u8; 32],
        sha256: &[u8; 32],
    ) -> Result<Option<Vec<u8>>> {
        let address = to_hex(address);
        self.get_url(
            &self.routes.section(&address, &to_hex(sha256)),
            &format!("section {address}"),
        )
        .await
    }

    async fn get_url(&self, url: &str, what: &str) -> Result<Option<Vec<u8>>> {
        let resp = self
            .http
            .get(url)
            .header(API_KEY_HEADER, self.api_key.expose())
            .send()
            .await
            .map_err(|e| anyhow!("{what}: {}", e.without_url()))?;
        let status = resp.status().as_u16();
        if status == 404 {
            return Ok(None);
        }
        if let Some(p) = status_problem(status) {
            bail!("{what}: {p}");
        }
        let body = resp
            .bytes()
            .await
            .map_err(|e| anyhow!("{what}: {}", e.without_url()))?;
        Ok(Some(body.to_vec()))
    }

    /// Offers a computed section for `job` once (`client-compute.md`
    /// section 2.5): `201` stored, `200` matched, `202` disputed, anything
    /// else an error with the hub's reason.
    pub async fn upload(&self, job: &ComputeJob, bytes: Vec<u8>) -> Result<u16> {
        let address = job.address_hex();
        let mut req = self
            .http
            .put(self.routes.upload(self.build_id(), &address))
            .header(API_KEY_HEADER, self.api_key.expose())
            .header("Content-Type", "application/octet-stream");
        for (name, value) in upload_headers(job, &bytes) {
            req = req.header(name, value);
        }
        let resp = req
            .body(bytes)
            .send()
            .await
            .map_err(|e| anyhow!("upload {address}: {}", e.without_url()))?;
        let status = resp.status().as_u16();
        if matches!(status, 200..=202) {
            return Ok(status);
        }
        let body = resp.bytes().await.unwrap_or_default();
        let detail = String::from_utf8_lossy(&body[..body.len().min(300)]).into_owned();
        bail!("upload {address}: hub answered {status}: {detail}")
    }
}

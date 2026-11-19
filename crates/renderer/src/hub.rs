//! The native hub client, version 2: the clock, the build and its layers,
//! directions, content-addressed reads through the edge, uploads with the
//! section 2.5 headers, and check reports ([`HubClient::check`], section
//! 2.6). There is no readiness socket: nothing waits on another client.
//!
//! Implements the client side of `client-compute.md` sections 2.3 to 2.6
//! over [`crate::protocol`]:
//!
//! - [`HubClient::connect`] reads `GET /space/{s}/clock`, `GET
//!   /space/{s}/build/active` (unless a build id is configured), and `GET
//!   /space/{s}/build/{b}`, whose layer configs drive the depth of each
//!   frame ([`crate::protocol::frame_system_from_registries`]).
//! - [`HubClient::sync_clock`] measures the offset from the hub's clock,
//!   at connect time and every minute after through
//!   [`HubClient::keep_clock`] ([`crate::protocol::ClockSync`],
//!   `space-model.md` section 7).
//! - [`HubClient::directions`] posts one batch of at most 512 keys.
//! - [`HubClient::get_blob`] reads `/module/{hash}`, `/source/{hash}`,
//!   and `/source-manifest/{hash}` from the edge, and
//!   [`HubClient::get_section`] reads `/section/{address}?sha256={hex}`,
//!   one version of a stored section.
//! - [`HubClient::upload`] offers a computed `missing` section once, with
//!   `X-Layer`, `X-Key`, `X-Span-Index`, `X-Content-Sha256`, and `Expect:
//!   100-continue`, and returns the hub's status (`201` stored, `200`
//!   matched, `202` disputed).
//!
//! The batching, retries, and per-layer flow live in
//! [`crate::directions`]; the fetcher that drives them is
//! [`crate::stream::Fetcher`]. Every request carries the API key in
//! `X-API-Key`; the key is never logged.

use crate::config::{ApiKey, Config};
pub use crate::protocol::API_KEY_HEADER;
use crate::protocol::{
    status_problem, upload_headers, ActiveBuild, BuildDescription, Clock, ClockSync,
    DirectionsRequest, DirectionsResponse, Routes, MAX_DIRECTION_KEYS,
};
use anyhow::{anyhow, bail, Context, Result};
use gx_core::key::ChunkKey;
use gx_core::sha256::to_hex;
use gx_worker::{BlobKind, ComputeJob};
use serde::de::DeserializeOwned;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// A connection to one build of one space.
pub struct HubClient {
    http: reqwest::Client,
    routes: Routes,
    api_key: ApiKey,
    build: BuildDescription,
    clock: Mutex<ClockSync>,
    /// The local monotonic clock's zero.
    base: Instant,
}

impl HubClient {
    /// Connects to the configured space and build.
    pub async fn connect(config: &Config) -> Result<HubClient> {
        HubClient::open(
            Routes::new(
                &config.hub_url,
                config.edge_url.as_deref(),
                &config.space_id,
            ),
            config.api_key.clone(),
            config.build_id.as_deref(),
        )
        .await
    }

    /// Connects through `routes` with `api_key`: reads the clock, resolves
    /// the active build when `build_id` is `None`, and reads the build's
    /// layers.
    pub async fn open(
        routes: Routes,
        api_key: ApiKey,
        build_id: Option<&str>,
    ) -> Result<HubClient> {
        let http = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_secs(60))
            .build()
            .context("building the HTTP client")?;
        let mut client = HubClient {
            http,
            routes,
            api_key,
            build: BuildDescription {
                build_id: String::new(),
                layers: Vec::new(),
            },
            clock: Mutex::new(ClockSync::measure(
                Clock {
                    origin_utc_ms: None,
                    now_utc_ms: 0,
                },
                0.0,
                0.0,
            )),
            base: Instant::now(),
        };
        client.sync_clock().await?;
        let build_id = match build_id {
            Some(b) => b.to_string(),
            None => {
                let active: ActiveBuild = client
                    .get_json(&client.routes.active_build(), "reading the active build")
                    .await?;
                active.build_id
            }
        };
        client.build = client
            .get_json(&client.routes.build(&build_id), "reading the build")
            .await?;
        if client.build.build_id.is_empty() {
            client.build.build_id = build_id;
        }
        tracing::info!(
            build = %client.build.build_id,
            layers = client.build.layers.len(),
            "using build"
        );
        Ok(client)
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

    /// The hub's clock as this client keeps it.
    pub fn clock(&self) -> ClockSync {
        *self.clock.lock().expect("clock")
    }

    /// Seconds on the local monotonic clock since this client was made.
    fn local_seconds(&self) -> f64 {
        self.base.elapsed().as_secs_f64()
    }

    /// The hub's now, UTC milliseconds: the latest measurement of the
    /// hub's clock advanced by the local monotonic clock since
    /// ([`ClockSync`]).
    pub fn now_utc_ms(&self) -> i64 {
        self.clock().now_utc_ms(self.local_seconds())
    }

    /// Measures the offset from the hub's clock again with `GET
    /// /space/{s}/clock`. Done at connect time and, by
    /// [`HubClient::keep_clock`], every
    /// [`crate::protocol::CLOCK_SYNC_SECONDS`].
    pub async fn sync_clock(&self) -> Result<()> {
        let sent = self.local_seconds();
        let clock: Clock = self
            .get_json(&self.routes.clock(), "reading the clock")
            .await?;
        let received = self.local_seconds();
        self.clock
            .lock()
            .expect("clock")
            .update(clock, sent, received);
        Ok(())
    }

    /// Keeps the clock measured on `handle`: a task that measures it again
    /// every [`crate::protocol::CLOCK_SYNC_SECONDS`] for as long as the
    /// client lives. A failed measurement keeps the previous one and is
    /// logged.
    pub fn keep_clock(self: &Arc<Self>, handle: &tokio::runtime::Handle) {
        let weak = Arc::downgrade(self);
        handle.spawn(async move {
            let period = Duration::from_secs_f64(crate::protocol::CLOCK_SYNC_SECONDS);
            loop {
                tokio::time::sleep(period).await;
                let Some(client) = weak.upgrade() else {
                    return;
                };
                if let Err(e) = client.sync_clock().await {
                    tracing::warn!("{e:#}");
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
    /// `sha256`: `GET /section/{address}?sha256={hex}` on the edge. `Some`
    /// bytes for `200`, `None` for `404`. The caller checks the hash.
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

    /// Reports the hash of a recomputed section for `job` once
    /// (`client-compute.md` section 2.6). Returns the status: `200` match,
    /// `202` disputed, or a refusal (`403`, `404`, `429`).
    pub async fn check(&self, job: &ComputeJob, recomputed: &[u8; 32]) -> Result<u16> {
        let address = job.address_hex();
        let mut req = self
            .http
            .post(self.routes.check(self.build_id(), &address))
            .header(API_KEY_HEADER, self.api_key.expose())
            .header("Content-Type", "application/json");
        for (name, value) in crate::protocol::check_headers(job) {
            req = req.header(name, value);
        }
        let body = serde_json::to_vec(&crate::protocol::CheckRequest {
            sha256: to_hex(recomputed),
        })?;
        let resp = req
            .body(body)
            .send()
            .await
            .map_err(|e| anyhow!("check {address}: {}", e.without_url()))?;
        Ok(resp.status().as_u16())
    }

    /// Offers a computed section for `job` once (`client-compute.md`
    /// section 2.5). Returns the status: `201` stored, `200` matched a
    /// stored copy, `202` differed from it. Anything else is an error with
    /// the hub's reason.
    pub async fn upload(&self, job: &ComputeJob, bytes: Vec<u8>) -> Result<u16> {
        let address = job.address_hex();
        let mut req = self
            .http
            .put(self.routes.upload(self.build_id(), &address))
            .header(API_KEY_HEADER, self.api_key.expose())
            .header("Content-Type", "application/octet-stream")
            .header("Expect", "100-continue");
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

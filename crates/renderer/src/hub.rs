//! The native hub client, version 2: the clock, the build and its layers,
//! directions, content-addressed reads through the edge, uploads with a
//! claim, and the readiness socket keyed by address.
//!
//! Implements the client side of `client-compute.md` sections 2.3 to 2.5
//! over [`crate::protocol`]:
//!
//! - [`HubClient::connect`] reads `GET /space/{s}/clock`, `GET
//!   /space/{s}/build/active` (unless a build id is configured), and `GET
//!   /space/{s}/build/{b}`, whose layer configs drive the depth of each
//!   frame ([`crate::protocol::frame_system_from_registries`]).
//! - [`HubClient::directions`] posts one batch of at most 512 keys.
//! - [`HubClient::get_blob`] reads `/section/{address}`, `/module/{hash}`,
//!   `/source/{hash}`, and `/source-manifest/{hash}` from the edge.
//! - [`HubClient::upload`] puts a computed section with `X-Claim`.
//! - [`HubClient::subscribe_ready`] keeps the readiness socket open,
//!   reconnecting with a backoff and subscribing again to every address
//!   still awaited.
//!
//! The batching, retries, and per-layer flow live in
//! [`crate::directions`]; the fetcher that drives them is
//! [`crate::stream::Fetcher`]. Every request carries the API key in
//! `X-API-Key`; the key is never logged.

use crate::config::{ApiKey, Config};
pub use crate::protocol::API_KEY_HEADER;
use crate::protocol::{
    parse_ready_frame, status_problem, subscribe_frame, ActiveBuild, BuildDescription, Clock,
    DirectionsRequest, DirectionsResponse, Routes, CLAIM_HEADER, MAX_DIRECTION_KEYS,
};
use anyhow::{anyhow, bail, Context, Result};
use futures_util::{SinkExt, StreamExt};
use gx_core::key::ChunkKey;
use gx_core::sha256::to_hex;
use gx_worker::BlobKind;
use serde::de::DeserializeOwned;
use std::collections::BTreeSet;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::http::HeaderValue;
use tokio_tungstenite::tungstenite::Message;

/// A connection to one build of one space.
pub struct HubClient {
    http: reqwest::Client,
    routes: Routes,
    api_key: ApiKey,
    build: BuildDescription,
    clock: Clock,
    clock_read_at: Instant,
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
            clock: Clock {
                origin_utc_ms: None,
                now_utc_ms: 0,
            },
            clock_read_at: Instant::now(),
        };
        client.clock = client
            .get_json(&client.routes.clock(), "reading the clock")
            .await?;
        client.clock_read_at = Instant::now();
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

    /// The clock as read at connect time.
    pub fn clock(&self) -> Clock {
        self.clock
    }

    /// The hub's now, UTC milliseconds: the clock read at connect time
    /// advanced by the time since.
    pub fn now_utc_ms(&self) -> i64 {
        let elapsed = self.clock_read_at.elapsed().as_millis();
        self.clock
            .now_utc_ms
            .saturating_add(i64::try_from(elapsed).unwrap_or(i64::MAX))
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

    /// Uploads a computed section with its claim. Returns the status:
    /// `201` stored, `200` already stored. Anything else is an error with
    /// the hub's reason.
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

    async fn open_socket(
        &self,
    ) -> Result<
        tokio_tungstenite::WebSocketStream<
            tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
        >,
    > {
        let mut request = self
            .routes
            .ready(self.build_id())
            .into_client_request()
            .context("readiness URL")?;
        let key = HeaderValue::from_str(self.api_key.expose())
            .context("API key is not a valid header value")?;
        request.headers_mut().insert(API_KEY_HEADER, key);
        let (socket, _) = tokio_tungstenite::connect_async(request)
            .await
            .context("opening the readiness socket")?;
        Ok(socket)
    }

    /// Starts the readiness socket on `runtime`: addresses sent to the
    /// returned subscription are subscribed to, and `on_ready` is called
    /// with every address the hub reports stored. The socket reconnects
    /// with a backoff after it closes, subscribing again to every address
    /// not yet reported; it stops once `on_ready` returns `false` or the
    /// subscription is dropped. A client that cannot reach it still completes
    /// every key, by asking for directions again after each `retryMs`.
    pub fn subscribe_ready(
        self: &std::sync::Arc<Self>,
        runtime: &tokio::runtime::Handle,
        on_ready: impl Fn([u8; 32]) -> bool + Send + 'static,
    ) -> ReadySubscription {
        let (out_tx, mut out_rx) = mpsc::unbounded_channel::<[u8; 32]>();
        let client = std::sync::Arc::clone(self);
        runtime.spawn(async move {
            let mut wanted: BTreeSet<[u8; 32]> = BTreeSet::new();
            let mut failures = 0u32;
            loop {
                let socket = match client.open_socket().await {
                    Ok(s) => s,
                    Err(e) => {
                        failures += 1;
                        if failures == 1 {
                            tracing::warn!("readiness socket unavailable ({e:#}); asking again after each retryMs");
                        }
                        let wait = crate::directions::backoff_seconds(failures).max(1.0);
                        let sleep = tokio::time::sleep(Duration::from_secs_f64(wait));
                        tokio::pin!(sleep);
                        loop {
                            tokio::select! {
                                _ = &mut sleep => break,
                                a = out_rx.recv() => match a {
                                    Some(a) => { wanted.insert(a); }
                                    None => return,
                                },
                            }
                        }
                        continue;
                    }
                };
                failures = 0;
                let (mut write, mut read) = socket.split();
                let mut ok = true;
                for a in &wanted {
                    if write.send(Message::text(subscribe_frame(&to_hex(a)))).await.is_err() {
                        ok = false;
                        break;
                    }
                }
                while ok {
                    tokio::select! {
                        a = out_rx.recv() => match a {
                            Some(a) => {
                                if wanted.insert(a)
                                    && write.send(Message::text(subscribe_frame(&to_hex(&a)))).await.is_err()
                                {
                                    ok = false;
                                }
                            }
                            None => {
                                let _ = write.send(Message::Close(None)).await;
                                return;
                            }
                        },
                        m = read.next() => match m {
                            Some(Ok(Message::Text(text))) => {
                                if let Some(a) = parse_ready_frame(text.as_str()) {
                                    wanted.remove(&a);
                                    if !on_ready(a) {
                                        return;
                                    }
                                }
                            }
                            Some(Ok(Message::Close(_))) | None | Some(Err(_)) => ok = false,
                            Some(Ok(_)) => {}
                        },
                    }
                }
                tracing::debug!("readiness socket closed, reconnecting");
            }
        });
        ReadySubscription { out: out_tx }
    }
}

/// The readiness socket's subscription side. Dropping it closes the
/// socket.
pub struct ReadySubscription {
    out: mpsc::UnboundedSender<[u8; 32]>,
}

impl ReadySubscription {
    /// Registers interest in `address`.
    pub fn subscribe(&self, address: [u8; 32]) {
        let _ = self.out.send(address);
    }
}

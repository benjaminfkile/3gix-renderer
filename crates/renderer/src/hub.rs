//! The hub client: chunks, the readiness WebSocket, and the frame registry.
//!
//! Implements the client side of the hub's chunk flow
//! (`compiler-pipeline.md` section 4, "The 202 / Poll Cycle") and the
//! renderer's first request, the registry (`space-model.md` section 7):
//!
//! - [`HubClient::fetch_chunk`] issues
//!   `GET /space/{spaceId}/build/{buildId}/chunk/{key}` and maps `200`,
//!   `202`, `404`, and `410` to [`ChunkFetch`]. A `200` body is immutable for
//!   its key (the hub sends `Cache-Control: public, max-age=31536000,
//!   immutable`), so the client keeps it for the session and never requests
//!   that key again.
//! - [`HubClient::subscribe_ready`] opens the readiness WebSocket
//!   `GET /space/{spaceId}/build/{buildId}/chunks/ready`, sends
//!   `{ "subscribe": "<key>" }` text frames, and yields the keys of the
//!   `{ "chunkReady": "<key>" }` frames the hub pushes.
//! - [`HubClient::wait_for_chunk`] combines the two: it fetches, and while
//!   the hub answers `202` it waits for a readiness push, falling back to
//!   polling every 2 s for a key still pending after 10 s, up to a deadline.
//! - [`load_registry`] fetches the `registry` chunk, decodes it with
//!   [`gx_core::container::decode_registry_chunk`], and builds the
//!   [`FrameTree`] and [`FrameSystem`] (`matter-format.md` sections 5 and 6).
//!
//! Every request carries the API key in the `X-API-Key` header. The key is
//! never logged.

use crate::config::{ApiKey, Config};
use anyhow::{anyhow, bail, Context, Result};
use futures_util::{SinkExt, StreamExt};
use gx_core::container::decode_registry_chunk;
use gx_core::frames::FrameSystem;
use gx_core::key::REGISTRY_KEY;
use gx_core::registry::FrameTree;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::time::Instant;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::http::HeaderValue;
use tokio_tungstenite::tungstenite::Message;

/// The header that carries the API key.
pub const API_KEY_HEADER: &str = "X-API-Key";

/// The outcome of one chunk request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ChunkFetch {
    /// `200 OK`: the assembled chunk container.
    Ready(Arc<[u8]>),
    /// `202 Accepted`: compilation was dispatched; ask again when ready.
    Pending,
    /// `404 Not Found`: the build has no layers to compile it.
    NotFound,
    /// `410 Gone`: a required compiler is retired; it can never be produced.
    Gone,
    /// Anything else: transport failure or an unexpected status.
    Error(String),
}

/// The outcome for an HTTP status of the chunk endpoint. A `200` needs its
/// body, so it maps to `None` here and is handled by the caller.
pub fn classify_status(status: u16) -> Option<ChunkFetch> {
    match status {
        200 => None,
        202 => Some(ChunkFetch::Pending),
        404 => Some(ChunkFetch::NotFound),
        410 => Some(ChunkFetch::Gone),
        401 => Some(ChunkFetch::Error(
            "hub refused the API key (401 Unauthorized)".into(),
        )),
        403 => Some(ChunkFetch::Error(
            "the API key lacks the fetch:chunks capability (403 Forbidden)".into(),
        )),
        other => Some(ChunkFetch::Error(format!("unexpected status {other}"))),
    }
}

/// How long to wait for a pending chunk, and how to poll for it.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ReadyPolicy {
    /// Rely on the readiness push alone for this long after the first `202`.
    pub fallback_after: Duration,
    /// After that, poll the chunk endpoint this often.
    pub poll_interval: Duration,
    /// Give up after this long.
    pub deadline: Duration,
}

impl Default for ReadyPolicy {
    /// Push only for 10 s, then poll every 2 s, give up after 60 s.
    fn default() -> Self {
        ReadyPolicy {
            fallback_after: Duration::from_secs(10),
            poll_interval: Duration::from_secs(2),
            deadline: Duration::from_secs(60),
        }
    }
}

/// One frame the client sends on the readiness socket.
#[derive(Serialize)]
struct SubscribeFrame<'a> {
    subscribe: &'a str,
}

/// One frame the hub pushes on the readiness socket.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ReadyFrame {
    chunk_ready: String,
}

/// Parses a readiness push, `{ "chunkReady": "<key>" }`, or returns `None`
/// for anything else.
pub fn parse_ready_frame(text: &str) -> Option<String> {
    serde_json::from_str::<ReadyFrame>(text)
        .ok()
        .map(|f| f.chunk_ready)
}

/// The text of a subscription frame, `{"subscribe":"<key>"}`.
pub fn subscribe_frame(key: &str) -> String {
    serde_json::to_string(&SubscribeFrame { subscribe: key }).expect("a string serializes")
}

/// One entry of `GET /space/{spaceId}/builds`.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct BuildSummary {
    /// The build id.
    pub build_id: String,
    /// `Active` for the build clients should draw, `Archived` for the rest.
    pub status: String,
}

/// Picks the active build from a builds listing: the first entry whose
/// status is `Active` (the hub lists newest first).
pub fn pick_active_build(builds: &[BuildSummary]) -> Option<&str> {
    builds
        .iter()
        .find(|b| b.status.eq_ignore_ascii_case("active"))
        .map(|b| b.build_id.as_str())
}

/// A connection to one build of one space on the hub.
pub struct HubClient {
    http: reqwest::Client,
    base_url: String,
    api_key: ApiKey,
    space_id: String,
    build_id: String,
    /// `200` bodies by key, kept for the session.
    ready: Mutex<BTreeMap<String, Arc<[u8]>>>,
}

impl HubClient {
    /// A client for a known build. `base_url` has no trailing slash.
    pub fn new(
        base_url: &str,
        api_key: ApiKey,
        space_id: &str,
        build_id: &str,
    ) -> Result<HubClient> {
        let http = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_secs(30))
            .build()
            .context("building the HTTP client")?;
        Ok(HubClient {
            http,
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key,
            space_id: space_id.to_string(),
            build_id: build_id.to_string(),
            ready: Mutex::new(BTreeMap::new()),
        })
    }

    /// A client for the configured build, resolving the active build through
    /// `GET /space/{spaceId}/builds` when the configuration names none.
    pub async fn connect(config: &Config) -> Result<HubClient> {
        let mut client = HubClient::new(
            &config.hub_url,
            config.api_key.clone(),
            &config.space_id,
            "",
        )?;
        client.build_id = match &config.build_id {
            Some(b) => b.clone(),
            None => client.active_build().await?,
        };
        tracing::info!(build = %client.build_id, "using build");
        Ok(client)
    }

    /// The build this client reads.
    pub fn build_id(&self) -> &str {
        &self.build_id
    }

    /// Asks the hub for the active build of the space.
    pub async fn active_build(&self) -> Result<String> {
        let url = format!("{}/space/{}/builds", self.base_url, self.space_id);
        let resp = self
            .http
            .get(&url)
            .header(API_KEY_HEADER, self.api_key.expose())
            .send()
            .await
            .context("listing builds")?;
        let status = resp.status().as_u16();
        if status != 200 {
            bail!("listing builds: hub answered {status}");
        }
        let body = resp.bytes().await.context("reading the builds listing")?;
        let builds: Vec<BuildSummary> =
            serde_json::from_slice(&body).context("decoding the builds listing")?;
        pick_active_build(&builds)
            .map(str::to_string)
            .ok_or_else(|| anyhow!("the space has no active build; set GX_BUILD_ID"))
    }

    fn chunk_url(&self, key: &str) -> String {
        format!(
            "{}/space/{}/build/{}/chunk/{}",
            self.base_url, self.space_id, self.build_id, key
        )
    }

    /// The readiness WebSocket URL for `build`: the hub URL with `http`
    /// replaced by `ws` (or `https` by `wss`).
    pub fn ready_url(&self, build: &str) -> String {
        let ws_base = if let Some(rest) = self.base_url.strip_prefix("https://") {
            format!("wss://{rest}")
        } else if let Some(rest) = self.base_url.strip_prefix("http://") {
            format!("ws://{rest}")
        } else {
            self.base_url.clone()
        };
        format!(
            "{ws_base}/space/{}/build/{build}/chunks/ready",
            self.space_id
        )
    }

    /// Fetches one chunk. A key that has answered `200` before is served
    /// from the session cache without a request.
    pub async fn fetch_chunk(&self, key: &str) -> ChunkFetch {
        if let Some(bytes) = self.ready.lock().expect("cache lock").get(key) {
            return ChunkFetch::Ready(bytes.clone());
        }
        let resp = match self
            .http
            .get(self.chunk_url(key))
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
            Ok(body) => {
                let bytes: Arc<[u8]> = Arc::from(body.as_ref());
                self.ready
                    .lock()
                    .expect("cache lock")
                    .insert(key.to_string(), bytes.clone());
                ChunkFetch::Ready(bytes)
            }
            Err(e) => ChunkFetch::Error(format!("reading body failed: {}", e.without_url())),
        }
    }

    /// Opens the readiness WebSocket for `build`.
    pub async fn subscribe_ready(&self, build: &str) -> Result<ReadySubscription> {
        let mut request = self
            .ready_url(build)
            .into_client_request()
            .context("readiness URL")?;
        let key = HeaderValue::from_str(self.api_key.expose())
            .context("API key is not a valid header value")?;
        request.headers_mut().insert(API_KEY_HEADER, key);
        let (socket, _) = tokio_tungstenite::connect_async(request)
            .await
            .context("opening the readiness socket")?;
        let (mut write, mut read) = socket.split();
        let (out_tx, mut out_rx) = mpsc::unbounded_channel::<String>();
        let (keys_tx, keys_rx) = mpsc::unbounded_channel::<String>();
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    outgoing = out_rx.recv() => match outgoing {
                        Some(text) => {
                            if write.send(Message::text(text)).await.is_err() {
                                break;
                            }
                        }
                        None => {
                            let _ = write.send(Message::Close(None)).await;
                            break;
                        }
                    },
                    incoming = read.next() => match incoming {
                        Some(Ok(Message::Text(text))) => {
                            if let Some(key) = parse_ready_frame(text.as_str()) {
                                if keys_tx.send(key).is_err() {
                                    break;
                                }
                            }
                        }
                        Some(Ok(Message::Close(_))) | None | Some(Err(_)) => break,
                        Some(Ok(_)) => {}
                    },
                }
            }
        });
        Ok(ReadySubscription {
            out: out_tx,
            keys: keys_rx,
        })
    }

    /// Fetches `key`, waiting while the hub answers `202`: for a readiness
    /// push on `subscription` (if any), and after
    /// [`ReadyPolicy::fallback_after`] also by polling every
    /// [`ReadyPolicy::poll_interval`]. Fails on `404`, `410`, an error, or
    /// when the key is not ready by [`ReadyPolicy::deadline`].
    pub async fn wait_for_chunk(
        &self,
        key: &str,
        policy: ReadyPolicy,
        mut subscription: Option<&mut ReadySubscription>,
    ) -> Result<Arc<[u8]>> {
        let start = Instant::now();
        if let Some(sub) = subscription.as_deref_mut() {
            sub.subscribe(key)?;
        }
        loop {
            match self.fetch_chunk(key).await {
                ChunkFetch::Ready(bytes) => return Ok(bytes),
                ChunkFetch::Pending => tracing::debug!(key, "chunk pending"),
                ChunkFetch::NotFound => {
                    bail!("chunk {key}: not found (404), the build has no layers")
                }
                ChunkFetch::Gone => bail!("chunk {key}: gone (410), it can never be produced"),
                ChunkFetch::Error(e) => bail!("chunk {key}: {e}"),
            }
            // Wait for the push or the next poll, whichever comes first.
            loop {
                let elapsed = start.elapsed();
                if elapsed >= policy.deadline {
                    bail!(
                        "chunk {key} did not become ready within {} s",
                        policy.deadline.as_secs_f64()
                    );
                }
                let poll_wait = if elapsed < policy.fallback_after {
                    policy.fallback_after - elapsed
                } else {
                    policy.poll_interval
                };
                let wait = poll_wait.min(policy.deadline - elapsed);
                let pushed = match subscription.as_deref_mut() {
                    Some(sub) => tokio::select! {
                        k = sub.next() => Some(k),
                        _ = tokio::time::sleep(wait) => None,
                    },
                    None => {
                        tokio::time::sleep(wait).await;
                        None
                    }
                };
                match pushed {
                    Some(Some(k)) if k == key => break,
                    Some(Some(_)) => continue,
                    Some(None) => {
                        tracing::warn!("readiness socket closed, polling instead");
                        subscription = None;
                        continue;
                    }
                    None if start.elapsed() >= policy.fallback_after => break,
                    None => continue,
                }
            }
        }
    }
}

/// An open readiness socket: send subscriptions, receive ready keys.
pub struct ReadySubscription {
    out: mpsc::UnboundedSender<String>,
    keys: mpsc::UnboundedReceiver<String>,
}

impl ReadySubscription {
    /// Registers interest in `key`.
    pub fn subscribe(&self, key: &str) -> Result<()> {
        self.out
            .send(subscribe_frame(key))
            .map_err(|_| anyhow!("readiness socket is closed"))
    }

    /// The next key the hub reports ready, or `None` once the socket closes.
    pub async fn next(&mut self) -> Option<String> {
        self.keys.recv().await
    }
}

/// Decodes a registry chunk container and builds the frame system at the
/// epoch. Fails with the validator's code and reason if any registry or the
/// union of them is invalid.
pub fn frame_system_from_chunk(bytes: &[u8]) -> Result<FrameSystem> {
    let registries = decode_registry_chunk(bytes)
        .map_err(|e| anyhow!("registry is invalid: code {}: {}", e.code, e.reason))?;
    let tree = FrameTree::from_registries(&registries)
        .map_err(|e| anyhow!("registry is invalid: code {}: {}", e.code, e.reason))?;
    Ok(FrameSystem::from_tree(tree))
}

/// Loads the frame registry of the client's build: opens the readiness
/// socket (polling instead if it does not open), fetches the `registry`
/// chunk with [`HubClient::wait_for_chunk`], then decodes and validates it.
pub async fn load_registry(client: &HubClient, policy: ReadyPolicy) -> Result<FrameSystem> {
    let mut sub = match client.subscribe_ready(client.build_id()).await {
        Ok(s) => Some(s),
        Err(e) => {
            tracing::warn!("readiness socket unavailable ({e:#}), polling instead");
            None
        }
    };
    let bytes = client
        .wait_for_chunk(REGISTRY_KEY, policy, sub.as_mut())
        .await?;
    let system = frame_system_from_chunk(&bytes)?;
    tracing::info!(frames = system.tree().frames().len(), "registry loaded");
    Ok(system)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn statuses_map_to_outcomes() {
        assert_eq!(classify_status(200), None);
        assert_eq!(classify_status(202), Some(ChunkFetch::Pending));
        assert_eq!(classify_status(404), Some(ChunkFetch::NotFound));
        assert_eq!(classify_status(410), Some(ChunkFetch::Gone));
        assert!(matches!(classify_status(500), Some(ChunkFetch::Error(_))));
        assert!(matches!(classify_status(401), Some(ChunkFetch::Error(_))));
    }

    #[test]
    fn socket_frames() {
        assert_eq!(subscribe_frame("registry"), r#"{"subscribe":"registry"}"#);
        assert_eq!(
            parse_ready_frame(r#"{"chunkReady":"4-1-0-1-0"}"#).as_deref(),
            Some("4-1-0-1-0")
        );
        assert_eq!(parse_ready_frame(r#"{"other":1}"#), None);
        assert_eq!(parse_ready_frame("not json"), None);
    }

    #[test]
    fn active_build_is_the_first_active() {
        let builds: Vec<BuildSummary> = serde_json::from_str(
            r#"[{"buildId":"b2","spaceId":"s","createdAt":"t","status":"Archived"},
                {"buildId":"b1","spaceId":"s","createdAt":"t","status":"Active"}]"#,
        )
        .unwrap();
        assert_eq!(pick_active_build(&builds), Some("b1"));
        assert_eq!(pick_active_build(&builds[..1]), None);
    }

    #[test]
    fn ready_url_swaps_the_scheme() {
        let c = HubClient::new("https://hub.invalid/", ApiKey::new("k"), "s", "b").unwrap();
        assert_eq!(
            c.ready_url("b"),
            "wss://hub.invalid/space/s/build/b/chunks/ready"
        );
        let c = HubClient::new("http://hub.invalid", ApiKey::new("k"), "s", "b").unwrap();
        assert_eq!(
            c.ready_url("x"),
            "ws://hub.invalid/space/s/build/x/chunks/ready"
        );
        assert_eq!(
            c.chunk_url("registry"),
            "http://hub.invalid/space/s/build/b/chunk/registry"
        );
    }

    #[test]
    fn invalid_registry_chunk_is_a_clear_error() {
        let e = frame_system_from_chunk(&[1, 2, 3]).unwrap_err().to_string();
        assert!(e.starts_with("registry is invalid: code"), "{e}");
    }
}

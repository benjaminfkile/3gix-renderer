//! A small in-process hub for tests and local runs, speaking the v2 client
//! routes (`client-compute.md` sections 2.3 to 2.5) with the shapes of the
//! real hub's coordinator, blob, and readiness controllers:
//!
//! - `GET /space/{s}/clock`, `GET /space/{s}/build/active`, `GET
//!   /space/{s}/build/{b}` (the layers, ordered by layer id).
//! - `POST /space/{s}/build/{b}/directions`: sets the origin on the first
//!   call, refuses times past now or before the origin (both with 2 s of
//!   tolerance) with `422`, and answers `empty`, `ready`, `compute` (with a
//!   claim of a 120 s lease), or `pending` per key and layer, applying
//!   attachment, exclusivity, and max depth. With
//!   [`MockHubOptions::foreign_claim_ms`], every address is first held by a
//!   claim of another client for that long, so a client sees `pending`
//!   before it may compute.
//! - `PUT /space/{s}/build/{b}/section/{address}` with `X-Claim`: `200` for
//!   a stored address, `409` without the live claim, `400` when the library
//!   validator rejects the bytes for the address's chunk key, else stores,
//!   drops the claim, notifies, and answers `201`.
//! - `GET /section/{address}`, `/module/{hash}`, `/source/{hash}`,
//!   `/source-manifest/{hash}`: `200` immutable, or `404`.
//! - `WS /space/{s}/build/{b}/sections/ready`: `{"subscribe": address}` in,
//!   `{"sectionReady": address}` out once stored, at once for an address
//!   already stored.
//!
//! Every request must carry the configured key in `X-API-Key`, or the mock
//! answers `401`. It speaks just enough HTTP/1.1 for one request per
//! connection. Its state outlives client sessions, so a second client sees
//! what the first uploaded.
//!
//! [`registry_module`] builds a module that only ever answers the registry
//! key with fixed bytes, for a layer attached to no frames; with the core's
//! example module ([`example_matter_layer`]) it makes a whole space without
//! any compiler.

use crate::protocol::{
    Attachment, BuildDescription, LayerConfig, RawDirection, API_KEY_HEADER, CLAIM_HEADER,
    DEEPEST_DEPTH,
};
use futures_util::{SinkExt, StreamExt};
use gx_core::address::{section_address, span_index};
use gx_core::key::ChunkKey;
use gx_core::manifest::{self, Entry, Manifest};
use gx_core::sha256::{sha256, to_hex};
use std::collections::{BTreeMap, BTreeSet};
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::watch;
use tokio::task::JoinHandle;
use tokio_tungstenite::tungstenite::Message;

/// The space id the mock serves by default.
pub const MOCK_SPACE_ID: &str = "00000000-0000-0000-0000-00000000000a";
/// The build id the mock serves by default.
pub const MOCK_BUILD_ID: &str = "00000000-0000-0000-0000-00000000000b";
/// The API key the mock accepts by default. Not a secret.
pub const MOCK_API_KEY: &str = "mock-key";

/// A claim's lease, milliseconds.
pub const LEASE_MS: u64 = 120_000;

/// Clock skew tolerance on directions times, milliseconds.
pub const TOLERANCE_MS: i64 = 2_000;

/// One layer of the mock's build.
#[derive(Clone, Debug)]
pub struct MockLayer {
    /// The layer id.
    pub id: String,
    /// The module's WebAssembly bytes.
    pub module: Vec<u8>,
    /// The source manifest's bytes, if the layer reads sources.
    pub manifest: Option<Vec<u8>>,
    /// Where it is attached.
    pub attachment: Attachment,
    /// Exclusive ownership of its frames.
    pub exclusive: bool,
    /// Max depth per frame id, or `"*"`.
    pub max_depth: BTreeMap<String, u8>,
    /// Span length, seconds.
    pub span_seconds: u64,
    /// The module depends on the origin.
    pub uses_origin: bool,
}

impl MockLayer {
    fn config(&self) -> LayerConfig {
        LayerConfig {
            layer: self.id.clone(),
            module: to_hex(&sha256(&self.module)),
            source_manifest: self.manifest.as_ref().map(|m| to_hex(&sha256(m))),
            attachment: self.attachment.clone(),
            exclusive: self.exclusive,
            max_depth: self.max_depth.clone(),
            span_seconds: self.span_seconds,
            uses_origin: self.uses_origin,
        }
    }

    /// The hub's rule: the frame's own entry, else `"*"`, else no limit.
    fn max_depth_for(&self, frame_id: u64) -> u8 {
        self.max_depth
            .get(&frame_id.to_string())
            .or_else(|| self.max_depth.get("*"))
            .copied()
            .unwrap_or(DEEPEST_DEPTH)
    }
}

/// What the mock serves.
#[derive(Clone, Debug)]
pub struct MockHubOptions {
    /// Space id in the routes.
    pub space_id: String,
    /// Build id in the routes.
    pub build_id: String,
    /// The key every request must carry.
    pub api_key: String,
    /// The build's layers.
    pub layers: Vec<MockLayer>,
    /// Source tiles, served by their SHA-256.
    pub sources: Vec<Vec<u8>>,
    /// The origin, if it was set before the mock started.
    pub origin_utc_ms: Option<i64>,
    /// Hold every address under another client's claim for this long the
    /// first time it is asked about.
    pub foreign_claim_ms: Option<u64>,
}

impl MockHubOptions {
    /// Default ids and key, no origin yet, and these layers and tiles.
    pub fn new(layers: Vec<MockLayer>, sources: Vec<Vec<u8>>) -> MockHubOptions {
        MockHubOptions {
            space_id: MOCK_SPACE_ID.into(),
            build_id: MOCK_BUILD_ID.into(),
            api_key: MOCK_API_KEY.into(),
            layers,
            sources,
            origin_utc_ms: None,
            foreign_claim_ms: None,
        }
    }
}

/// What the mock has served, for assertions.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct MockCounts {
    /// Directions requests answered `200`.
    pub directions_requests: u64,
    /// The most keys one directions request carried.
    pub largest_batch: u64,
    /// `compute` answers.
    pub compute_answers: u64,
    /// `pending` answers.
    pub pending_answers: u64,
    /// `ready` answers.
    pub ready_answers: u64,
    /// Uploads stored (`201`).
    pub uploads_stored: u64,
    /// Uploads of an address already stored (`200`).
    pub uploads_repeated: u64,
    /// Uploads refused (`400` or `409`).
    pub uploads_refused: u64,
    /// `GET /section/...` answered `200`.
    pub section_reads: u64,
    /// `GET /module/...` answered `200`.
    pub module_reads: u64,
    /// `GET /source/...` answered `200`.
    pub source_reads: u64,
    /// `GET /source-manifest/...` answered `200`.
    pub manifest_reads: u64,
    /// `sectionReady` notices pushed.
    pub notices: u64,
}

#[derive(Clone, Debug)]
struct Claim {
    token: String,
    key: String,
    expires: Instant,
}

#[derive(Default)]
struct Counters {
    directions_requests: AtomicU64,
    largest_batch: AtomicU64,
    compute_answers: AtomicU64,
    pending_answers: AtomicU64,
    ready_answers: AtomicU64,
    uploads_stored: AtomicU64,
    uploads_repeated: AtomicU64,
    uploads_refused: AtomicU64,
    section_reads: AtomicU64,
    module_reads: AtomicU64,
    source_reads: AtomicU64,
    manifest_reads: AtomicU64,
    notices: AtomicU64,
}

struct Space {
    origin: Option<i64>,
    sections: BTreeMap<String, Vec<u8>>,
    claims: BTreeMap<String, Claim>,
    asked: BTreeSet<String>,
    next_token: u64,
}

struct State {
    options: MockHubOptions,
    /// Layers ordered by id, as the hub lists them.
    layers: Vec<MockLayer>,
    modules: BTreeMap<String, Vec<u8>>,
    manifests: BTreeMap<String, Vec<u8>>,
    sources: BTreeMap<String, Vec<u8>>,
    space: Mutex<Space>,
    counters: Counters,
    /// Every stored address, for the readiness sockets.
    stored: watch::Sender<BTreeSet<String>>,
}

/// A running mock hub. Stops when dropped.
pub struct MockHub {
    addr: SocketAddr,
    state: Arc<State>,
    task: JoinHandle<()>,
}

impl Drop for MockHub {
    fn drop(&mut self) {
        self.task.abort();
    }
}

fn now_utc_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| i64::try_from(d.as_millis()).unwrap_or(i64::MAX))
}

impl MockHub {
    /// Binds a loopback port chosen by the system and starts serving.
    pub async fn start(options: MockHubOptions) -> std::io::Result<MockHub> {
        MockHub::bind(
            options,
            SocketAddr::from((std::net::Ipv4Addr::LOCALHOST, 0)),
        )
        .await
    }

    /// Binds `addr` and starts serving.
    pub async fn bind(options: MockHubOptions, addr: SocketAddr) -> std::io::Result<MockHub> {
        let listener = TcpListener::bind(addr).await?;
        let addr = listener.local_addr()?;
        let mut layers = options.layers.clone();
        layers.sort_by(|a, b| a.id.cmp(&b.id));
        let hex = |b: &[u8]| to_hex(&sha256(b));
        let state = Arc::new(State {
            modules: layers
                .iter()
                .map(|l| (hex(&l.module), l.module.clone()))
                .collect(),
            manifests: layers
                .iter()
                .filter_map(|l| l.manifest.as_ref().map(|m| (hex(m), m.clone())))
                .collect(),
            sources: options
                .sources
                .iter()
                .map(|s| (hex(s), s.clone()))
                .collect(),
            space: Mutex::new(Space {
                origin: options.origin_utc_ms,
                sections: BTreeMap::new(),
                claims: BTreeMap::new(),
                asked: BTreeSet::new(),
                next_token: 1,
            }),
            layers,
            options,
            counters: Counters::default(),
            stored: watch::channel(BTreeSet::new()).0,
        });
        let serving = state.clone();
        let task = tokio::spawn(async move {
            while let Ok((stream, _)) = listener.accept().await {
                let state = serving.clone();
                tokio::spawn(async move {
                    if let Err(e) = handle(stream, state).await {
                        tracing::debug!("mock hub connection ended: {e}");
                    }
                });
            }
        });
        Ok(MockHub { addr, state, task })
    }

    /// The base URL, `http://` plus the bound address.
    pub fn url(&self) -> String {
        format!("http://{}", self.addr)
    }

    /// The bound address.
    pub fn addr(&self) -> SocketAddr {
        self.addr
    }

    /// The origin, once set.
    pub fn origin_utc_ms(&self) -> Option<i64> {
        self.state.space.lock().expect("space").origin
    }

    /// Every stored section by address.
    pub fn sections(&self) -> BTreeMap<String, Vec<u8>> {
        self.state.space.lock().expect("space").sections.clone()
    }

    /// The build's description as served.
    pub fn build(&self) -> BuildDescription {
        build_description(&self.state)
    }

    /// What the mock has served so far.
    pub fn counts(&self) -> MockCounts {
        let c = &self.state.counters;
        let l = |a: &AtomicU64| a.load(Ordering::SeqCst);
        MockCounts {
            directions_requests: l(&c.directions_requests),
            largest_batch: l(&c.largest_batch),
            compute_answers: l(&c.compute_answers),
            pending_answers: l(&c.pending_answers),
            ready_answers: l(&c.ready_answers),
            uploads_stored: l(&c.uploads_stored),
            uploads_repeated: l(&c.uploads_repeated),
            uploads_refused: l(&c.uploads_refused),
            section_reads: l(&c.section_reads),
            module_reads: l(&c.module_reads),
            source_reads: l(&c.source_reads),
            manifest_reads: l(&c.manifest_reads),
            notices: l(&c.notices),
        }
    }
}

fn build_description(state: &State) -> BuildDescription {
    BuildDescription {
        build_id: state.options.build_id.clone(),
        layers: state.layers.iter().map(MockLayer::config).collect(),
    }
}

/// A parsed request head.
struct Head {
    method: String,
    path: String,
    headers: Vec<(String, String)>,
    len: usize,
}

impl Head {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }
}

/// Peeks at the request head without consuming it, so a WebSocket upgrade
/// can hand the untouched stream to the handshake.
async fn peek_head(stream: &TcpStream) -> std::io::Result<Option<Head>> {
    let mut buf = vec![0u8; 16 * 1024];
    loop {
        let n = stream.peek(&mut buf).await?;
        if n == 0 {
            return Ok(None);
        }
        if let Some(end) = buf[..n].windows(4).position(|w| w == b"\r\n\r\n") {
            let text = String::from_utf8_lossy(&buf[..end]).into_owned();
            let mut lines = text.split("\r\n");
            let mut first = lines.next().unwrap_or("").split(' ');
            let method = first.next().unwrap_or("").to_string();
            let path = first.next().unwrap_or("").to_string();
            let headers = lines
                .filter_map(|l| l.split_once(':'))
                .map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))
                .collect();
            return Ok(Some(Head {
                method,
                path,
                headers,
                len: end + 4,
            }));
        }
        if n == buf.len() {
            return Err(std::io::Error::other("request head too large"));
        }
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
}

async fn respond(
    stream: &mut TcpStream,
    status: &str,
    headers: &[(&str, &str)],
    body: &[u8],
) -> std::io::Result<()> {
    let mut out = format!(
        "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n",
        body.len()
    );
    for (k, v) in headers {
        out.push_str(&format!("{k}: {v}\r\n"));
    }
    out.push_str("\r\n");
    stream.write_all(out.as_bytes()).await?;
    stream.write_all(body).await?;
    stream.flush().await?;
    stream.shutdown().await
}

const IMMUTABLE: (&str, &str) = ("Cache-Control", "public, max-age=31536000, immutable");
const NO_STORE: (&str, &str) = ("Cache-Control", "no-store");
const JSON: (&str, &str) = ("Content-Type", "application/json");

async fn json(
    stream: &mut TcpStream,
    status: &str,
    body: &serde_json::Value,
) -> std::io::Result<()> {
    respond(
        stream,
        status,
        &[JSON, NO_STORE],
        body.to_string().as_bytes(),
    )
    .await
}

async fn handle(mut stream: TcpStream, state: Arc<State>) -> std::io::Result<()> {
    let Some(head) = peek_head(&stream).await? else {
        return Ok(());
    };
    let o = &state.options;
    let authorized = head.header(API_KEY_HEADER) == Some(o.api_key.as_str());
    let build = format!("/space/{}/build/{}", o.space_id, o.build_id);
    let upgrade = head
        .header("Upgrade")
        .is_some_and(|v| v.eq_ignore_ascii_case("websocket"));
    if upgrade && authorized && head.path == format!("{build}/sections/ready") {
        return serve_socket(stream, state).await;
    }
    let mut sink = vec![0u8; head.len];
    stream.read_exact(&mut sink).await?;
    let len: usize = head
        .header("Content-Length")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let mut body = vec![0u8; len];
    stream.read_exact(&mut body).await?;
    if !authorized {
        return respond(&mut stream, "401 Unauthorized", &[NO_STORE], b"").await;
    }
    let space = format!("/space/{}", o.space_id);
    let path = head.path.as_str();
    match head.method.as_str() {
        "GET" if path == format!("{space}/clock") => {
            let origin = state.space.lock().expect("space").origin;
            json(
                &mut stream,
                "200 OK",
                &serde_json::json!({"originUtcMs": origin, "nowUtcMs": now_utc_ms()}),
            )
            .await
        }
        "GET" if path == format!("{space}/build/active") => {
            json(
                &mut stream,
                "200 OK",
                &serde_json::json!({"buildId": o.build_id}),
            )
            .await
        }
        "GET" if path == build => {
            let desc = serde_json::to_vec(&build_description(&state)).expect("serializes");
            respond(&mut stream, "200 OK", &[JSON, IMMUTABLE], &desc).await
        }
        "POST" if path == format!("{build}/directions") => {
            let (status, answer) = directions(&state, &body);
            json(&mut stream, status, &answer).await
        }
        "PUT" if path.starts_with(&format!("{build}/section/")) => {
            let address = &path[build.len() + "/section/".len()..];
            let (status, reason) = upload(&state, address, head.header(CLAIM_HEADER), &body);
            respond(&mut stream, status, &[NO_STORE], reason.as_bytes()).await
        }
        "GET" => {
            let found = blob(&state, path);
            match found {
                Some(bytes) => {
                    respond(
                        &mut stream,
                        "200 OK",
                        &[("Content-Type", "application/octet-stream"), IMMUTABLE],
                        &bytes,
                    )
                    .await
                }
                None => respond(&mut stream, "404 Not Found", &[NO_STORE], b"").await,
            }
        }
        _ => respond(&mut stream, "404 Not Found", &[NO_STORE], b"").await,
    }
}

fn blob(state: &State, path: &str) -> Option<Vec<u8>> {
    let c = &state.counters;
    let (counter, bytes) = if let Some(a) = path.strip_prefix("/section/") {
        let sections = &state.space.lock().expect("space").sections;
        (&c.section_reads, sections.get(a).cloned())
    } else if let Some(h) = path.strip_prefix("/module/") {
        (&c.module_reads, state.modules.get(h).cloned())
    } else if let Some(h) = path.strip_prefix("/source-manifest/") {
        (&c.manifest_reads, state.manifests.get(h).cloned())
    } else {
        let h = path.strip_prefix("/source/")?;
        (&c.source_reads, state.sources.get(h).cloned())
    };
    if bytes.is_some() {
        counter.fetch_add(1, Ordering::SeqCst);
    }
    bytes
}

fn refusal(reason: &str) -> serde_json::Value {
    serde_json::json!({ "error": reason })
}

/// Answers a directions request (`client-compute.md` section 2.4).
fn directions(state: &State, body: &[u8]) -> (&'static str, serde_json::Value) {
    #[derive(serde::Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct Body {
        t_utc_ms: i64,
        keys: Vec<String>,
    }
    let Ok(req) = serde_json::from_slice::<Body>(body) else {
        return (
            "400 Bad Request",
            refusal("the body is not a directions request"),
        );
    };
    if req.keys.len() > crate::protocol::MAX_DIRECTION_KEYS {
        return ("400 Bad Request", refusal("at most 512 keys per request"));
    }
    let mut keys = Vec::with_capacity(req.keys.len());
    for k in &req.keys {
        match k.parse::<ChunkKey>() {
            Ok(key) => keys.push(key),
            Err(_) => return ("400 Bad Request", refusal("not a chunk key")),
        }
    }
    let now = now_utc_ms();
    let mut space = state.space.lock().expect("space");
    let origin = *space.origin.get_or_insert(now);
    if req.t_utc_ms < origin - TOLERANCE_MS {
        return (
            "422 Unprocessable Entity",
            refusal("'tUtcMs' is before the origin"),
        );
    }
    if req.t_utc_ms > now + TOLERANCE_MS {
        return ("422 Unprocessable Entity", refusal("'tUtcMs' is after now"));
    }
    let t = req.t_utc_ms.max(origin);
    let c = &state.counters;
    c.directions_requests.fetch_add(1, Ordering::SeqCst);
    c.largest_batch
        .fetch_max(keys.len() as u64, Ordering::SeqCst);
    let exclusive: BTreeMap<u64, &str> = state
        .layers
        .iter()
        .filter(|l| l.exclusive)
        .flat_map(|l| match &l.attachment {
            Attachment::Frames(f) => f.iter().map(|&id| (id, l.id.as_str())).collect(),
            Attachment::All => Vec::new(),
        })
        .collect();
    let mut results = Vec::with_capacity(keys.len());
    for (key, name) in keys.iter().zip(&req.keys) {
        let mut layers = Vec::with_capacity(state.layers.len());
        for l in &state.layers {
            let mut d = RawDirection {
                layer: l.id.clone(),
                state: "empty".into(),
                ..RawDirection::default()
            };
            if let ChunkKey::Cell(cell) = key {
                let owned_elsewhere = exclusive
                    .get(&cell.frame_id)
                    .is_some_and(|owner| *owner != l.id);
                if !l.attachment.contains(cell.frame_id)
                    || owned_elsewhere
                    || cell.depth > l.max_depth_for(cell.frame_id)
                {
                    layers.push(d);
                    continue;
                }
            }
            let address_origin = if l.uses_origin { origin } else { 0 };
            let span = span_index(t, address_origin, l.span_seconds).unwrap_or(0);
            let module = sha256(&l.module);
            let manifest = l.manifest.as_ref().map(|m| sha256(m));
            let address = to_hex(&section_address(
                &module,
                manifest.as_ref(),
                address_origin,
                l.span_seconds,
                span,
                key,
            ));
            d.address = Some(address.clone());
            let fresh = space.asked.insert(address.clone());
            let instant = Instant::now();
            if space.sections.contains_key(&address) {
                d.state = "ready".into();
                c.ready_answers.fetch_add(1, Ordering::SeqCst);
            } else {
                if fresh {
                    if let Some(ms) = state.options.foreign_claim_ms {
                        space.claims.insert(
                            address.clone(),
                            Claim {
                                token: "held-by-another-client".into(),
                                key: name.clone(),
                                expires: instant + Duration::from_millis(ms),
                            },
                        );
                    }
                }
                match space.claims.get(&address) {
                    Some(claim) if claim.expires > instant => {
                        let left = (claim.expires - instant).as_millis() as u64;
                        d.state = "pending".into();
                        d.retry_ms = Some(left.clamp(50, 2000));
                        c.pending_answers.fetch_add(1, Ordering::SeqCst);
                    }
                    _ => {
                        let token = format!("claim-{:08}", space.next_token);
                        space.next_token += 1;
                        space.claims.insert(
                            address.clone(),
                            Claim {
                                token: token.clone(),
                                key: name.clone(),
                                expires: instant + Duration::from_millis(LEASE_MS),
                            },
                        );
                        d.state = "compute".into();
                        d.claim = Some(token);
                        d.lease_ms = Some(LEASE_MS);
                        d.module = Some(to_hex(&module));
                        d.source_manifest = manifest.as_ref().map(|m| to_hex(m));
                        d.origin_utc_ms = Some(address_origin);
                        d.span_index = Some(span);
                        d.span_seconds = Some(l.span_seconds);
                        c.compute_answers.fetch_add(1, Ordering::SeqCst);
                    }
                }
            }
            layers.push(d);
        }
        results.push(serde_json::json!({ "key": name, "layers": layers }));
    }
    (
        "200 OK",
        serde_json::json!({ "originUtcMs": origin, "results": results }),
    )
}

/// Applies an upload (`client-compute.md` section 2.5).
fn upload(
    state: &State,
    address: &str,
    claim: Option<&str>,
    body: &[u8],
) -> (&'static str, String) {
    let c = &state.counters;
    let mut space = state.space.lock().expect("space");
    if space.sections.contains_key(address) {
        c.uploads_repeated.fetch_add(1, Ordering::SeqCst);
        return ("200 OK", String::new());
    }
    let live = space
        .claims
        .get(address)
        .filter(|cl| Some(cl.token.as_str()) == claim && cl.expires > Instant::now())
        .cloned();
    let Some(held) = live else {
        c.uploads_refused.fetch_add(1, Ordering::SeqCst);
        return (
            "409 Conflict",
            "No live claim on this address matches X-Claim.".into(),
        );
    };
    if let Err(e) = gx_core::validate(&held.key, body) {
        space.claims.remove(address);
        c.uploads_refused.fetch_add(1, Ordering::SeqCst);
        return ("400 Bad Request", e.to_string());
    }
    space.sections.insert(address.to_string(), body.to_vec());
    space.claims.remove(address);
    c.uploads_stored.fetch_add(1, Ordering::SeqCst);
    drop(space);
    state.stored.send_modify(|s| {
        s.insert(address.to_string());
    });
    ("201 Created", String::new())
}

async fn serve_socket(stream: TcpStream, state: Arc<State>) -> std::io::Result<()> {
    let socket = tokio_tungstenite::accept_async(stream)
        .await
        .map_err(std::io::Error::other)?;
    let (mut write, mut read) = socket.split();
    let mut stored = state.stored.subscribe();
    let mut subscribed: BTreeSet<String> = BTreeSet::new();
    loop {
        tokio::select! {
            msg = read.next() => match msg {
                Some(Ok(Message::Text(text))) => {
                    #[derive(serde::Deserialize)]
                    struct Subscribe { subscribe: Option<String> }
                    if let Ok(Subscribe { subscribe: Some(a) }) = serde_json::from_str(text.as_str()) {
                        subscribed.insert(a);
                    }
                }
                Some(Ok(Message::Close(_))) | None | Some(Err(_)) => break,
                Some(Ok(_)) => {}
            },
            changed = stored.changed() => {
                if changed.is_err() {
                    break;
                }
            }
        }
        let due: Vec<String> = {
            let ready = stored.borrow_and_update();
            subscribed
                .iter()
                .filter(|a| ready.contains(*a))
                .cloned()
                .collect()
        };
        for a in due {
            subscribed.remove(&a);
            let frame = format!(r#"{{"sectionReady":"{a}"}}"#);
            write
                .send(Message::text(frame))
                .await
                .map_err(std::io::Error::other)?;
            state.counters.notices.fetch_add(1, Ordering::SeqCst);
        }
    }
    Ok(())
}

/// A module that answers every request with `section` and reads no
/// sources: for a layer attached to no frames, which is only ever directed
/// for the registry key, `section` is that layer's registry. It has no
/// imports and only the exports of the module ABI (`client-compute.md`
/// section 4).
pub fn registry_module(section: &[u8]) -> Vec<u8> {
    const DATA_AT: usize = 1024;
    let heap = (DATA_AT + section.len()).div_ceil(8) * 8;
    let pages = (heap + (1 << 16)).div_ceil(1 << 16);
    let escaped: String = section.iter().map(|b| format!("\\{b:02x}")).collect();
    let packed = ((DATA_AT as u64) << 32) | section.len() as u64;
    let text = format!(
        r#"(module
  (memory (export "memory") {pages})
  (global $heap (mut i32) (i32.const {heap}))
  (data (i32.const {DATA_AT}) "{escaped}")
  (func (export "gx_abi_version") (result i32) (i32.const 1))
  (func (export "gx_alloc") (param $len i32) (result i32)
    (local $p i32)
    (local.set $p (global.get $heap))
    (global.set $heap
      (i32.and (i32.add (i32.add (local.get $p) (local.get $len)) (i32.const 7)) (i32.const -8)))
    (block $fits
      (br_if $fits (i32.le_u (global.get $heap) (i32.shl (memory.size) (i32.const 16))))
      (drop (memory.grow
        (i32.add
          (i32.shr_u (i32.sub (global.get $heap) (i32.shl (memory.size) (i32.const 16))) (i32.const 16))
          (i32.const 1)))))
    (local.get $p))
  (func (export "gx_free") (param i32 i32))
  (func (export "gx_sources") (param i32 i32) (result i64) (i64.const 0))
  (func (export "gx_compile") (param i32 i32 i32 i32) (result i64) (i64.const {packed}))
  (func (export "gx_last_error") (result i64) (i64.const 0)))"#
    );
    wat::parse_str(text).expect("the registry module assembles")
}

/// The layer that declares the frames: [`registry_module`] over
/// `registry`, attached to no frames.
pub fn registry_layer(id: &str, registry: &[u8]) -> MockLayer {
    MockLayer {
        id: id.into(),
        module: registry_module(registry),
        manifest: None,
        attachment: Attachment::Frames(Vec::new()),
        exclusive: false,
        max_depth: BTreeMap::new(),
        span_seconds: 0,
        uses_origin: false,
    }
}

/// The parameters of one of the core example module's tiles: the frame's
/// cube edge (meters), a base density (kilograms per cubic meter), and a
/// temperature (kelvin), three `f64` LE values. The tile's id is the frame
/// id shifted left by 5 bits with the depth in the low 5 bits. Both are the
/// example module's own conventions, private to it.
pub fn example_tile(root_extent: f64, base_density: f64, temperature: f64) -> Vec<u8> {
    let mut out = Vec::with_capacity(24);
    for v in [root_extent, base_density, temperature] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// A matter layer running the core example module `module` on every frame
/// in `frames` (id and cube edge, meters), filled to `max_depth`, with one
/// tile per frame and depth. Returns the layer and its tiles.
pub fn example_matter_layer(
    id: &str,
    module: &[u8],
    frames: &[(u64, f64)],
    max_depth: u8,
    base_density: f64,
    temperature: f64,
) -> (MockLayer, Vec<Vec<u8>>) {
    let mut entries = Vec::new();
    let mut tiles = Vec::new();
    for &(frame, extent) in frames {
        let tile = example_tile(extent, base_density, temperature);
        for depth in 0..=max_depth {
            entries.push(Entry {
                id: frame.wrapping_shl(5) | u64::from(depth),
                sha256: sha256(&tile),
                length: tile.len() as u64,
            });
        }
        tiles.push(tile);
    }
    let manifest = manifest::encode(&Manifest::new(entries).expect("tile ids are unique"));
    let layer = MockLayer {
        id: id.into(),
        module: module.to_vec(),
        manifest: Some(manifest),
        attachment: Attachment::All,
        exclusive: false,
        max_depth: [("*".to_string(), max_depth)].into(),
        span_seconds: 0,
        uses_origin: false,
    };
    (layer, tiles)
}

#[cfg(test)]
mod tests {
    use super::*;
    use gx_core::registry::{self, Registry};

    #[test]
    fn registry_module_is_admitted_and_answers_its_bytes() {
        let reg = registry::encode(&Registry::empty());
        let wasm = registry_module(&reg);
        let host = gx_module_host::ModuleHost::new().unwrap();
        let m = host.load(&wasm, &sha256(&wasm)).unwrap();
        let req = gx_core::module_io::ModuleRequest {
            origin_utc_ms: 0,
            span_seconds: 0,
            span_index: 0,
            key: ChunkKey::Registry,
        };
        assert!(m.sources(&req).unwrap().is_empty());
        assert_eq!(m.compile(&req, &[]).unwrap(), reg);
    }
}

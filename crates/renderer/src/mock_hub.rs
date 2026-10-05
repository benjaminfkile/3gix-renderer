//! A small in-process hub for tests and local runs.
//!
//! Serves the three hub endpoints the renderer uses, with the shapes of the
//! real hub (`compiler-pipeline.md` section 4, `third-party-api.md`, and the
//! hub's builds, chunk, and chunk subscription controllers):
//!
//! - `GET /space/{s}/builds`: one `Active` build.
//! - `GET /space/{s}/build/{b}/chunk/registry`: `202 Accepted` for the
//!   first [`MockHubOptions::pending_responses`] requests, then `200 OK` with
//!   the registry container.
//! - `GET /space/{s}/build/{b}/chunk/{cell}`: for a key in
//!   [`MockHubOptions::cells`], `202 Accepted` for the first
//!   [`MockHubOptions::cell_pending_responses`] requests, then `200 OK` with
//!   its container. Any other key answers `404`.
//! - `GET /space/{s}/build/{b}/chunks/ready` (WebSocket): accepts
//!   `{ "subscribe": "<key>" }` frames and, once a `202` for a key has been
//!   served, pushes `{ "chunkReady": "<key>" }` to subscribers of that key,
//!   so the push lands between the `202` and the `200`.
//!
//! Every request must carry the configured key in `X-API-Key`, or the mock
//! answers `401`. It speaks just enough HTTP/1.1 for one request per
//! connection.

use crate::hub::API_KEY_HEADER;
use gx_core::key::REGISTRY_KEY;
use std::collections::{BTreeMap, BTreeSet};
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::watch;
use tokio::task::JoinHandle;

use futures_util::{SinkExt, StreamExt};
use tokio_tungstenite::tungstenite::Message;

/// The space id the mock serves by default.
pub const MOCK_SPACE_ID: &str = "00000000-0000-0000-0000-00000000000a";
/// The build id the mock serves by default.
pub const MOCK_BUILD_ID: &str = "00000000-0000-0000-0000-00000000000b";
/// The API key the mock accepts by default. Not a secret.
pub const MOCK_API_KEY: &str = "mock-key";

/// What the mock serves.
#[derive(Clone, Debug)]
pub struct MockHubOptions {
    /// Space id in the routes.
    pub space_id: String,
    /// Build id in the routes and the builds listing.
    pub build_id: String,
    /// The key every request must carry.
    pub api_key: String,
    /// The bytes served for the `registry` key: a hub container.
    pub registry_chunk: Vec<u8>,
    /// How many registry requests answer `202` before the first `200`.
    pub pending_responses: u32,
    /// Matter chunk containers by cell key string.
    pub cells: BTreeMap<String, Vec<u8>>,
    /// How many requests of each cell answer `202` before the first `200`.
    pub cell_pending_responses: u32,
}

impl MockHubOptions {
    /// Default ids and key, one `202` before the `200`.
    pub fn new(registry_chunk: Vec<u8>) -> MockHubOptions {
        MockHubOptions {
            space_id: MOCK_SPACE_ID.into(),
            build_id: MOCK_BUILD_ID.into(),
            api_key: MOCK_API_KEY.into(),
            registry_chunk,
            pending_responses: 1,
            cells: BTreeMap::new(),
            cell_pending_responses: 1,
        }
    }

    /// Adds a cell's chunk container.
    pub fn with_cell(mut self, key: impl Into<String>, chunk: Vec<u8>) -> MockHubOptions {
        self.cells.insert(key.into(), chunk);
        self
    }
}

struct State {
    options: MockHubOptions,
    /// Chunk requests served per key.
    requests: Mutex<BTreeMap<String, u32>>,
    ready_pushes: AtomicU32,
    /// The keys for which a `202` has been served: ready to push.
    assembled: watch::Sender<BTreeSet<String>>,
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
        let (assembled, _) = watch::channel(BTreeSet::new());
        let state = Arc::new(State {
            options,
            requests: Mutex::new(BTreeMap::new()),
            ready_pushes: AtomicU32::new(0),
            assembled,
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

    /// Registry chunk requests served so far.
    pub fn registry_requests(&self) -> u32 {
        self.chunk_requests(REGISTRY_KEY)
    }

    /// Requests served so far for one chunk key.
    pub fn chunk_requests(&self, key: &str) -> u32 {
        self.state
            .requests
            .lock()
            .expect("request count lock")
            .get(key)
            .copied()
            .unwrap_or(0)
    }

    /// `chunkReady` frames pushed so far.
    pub fn ready_pushes(&self) -> u32 {
        self.state.ready_pushes.load(Ordering::SeqCst)
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
        tokio::time::sleep(std::time::Duration::from_millis(2)).await;
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

async fn handle(mut stream: TcpStream, state: Arc<State>) -> std::io::Result<()> {
    let Some(head) = peek_head(&stream).await? else {
        return Ok(());
    };
    let o = &state.options;
    let authorized = head.header(API_KEY_HEADER) == Some(o.api_key.as_str());
    let ready_path = format!("/space/{}/build/{}/chunks/ready", o.space_id, o.build_id);
    let upgrade = head
        .header("Upgrade")
        .is_some_and(|v| v.eq_ignore_ascii_case("websocket"));
    if upgrade && authorized && head.path == ready_path {
        return serve_socket(stream, state).await;
    }
    // Consume the head; requests carry no body.
    let mut sink = vec![0u8; head.len];
    stream.read_exact(&mut sink).await?;
    if !authorized {
        return respond(&mut stream, "401 Unauthorized", &[], b"").await;
    }
    if head.method != "GET" {
        return respond(&mut stream, "405 Method Not Allowed", &[], b"").await;
    }
    let builds_path = format!("/space/{}/builds", o.space_id);
    let chunk_prefix = format!("/space/{}/build/{}/chunk/", o.space_id, o.build_id);
    if head.path == builds_path {
        let body = format!(
            r#"[{{"buildId":"{}","spaceId":"{}","createdAt":"2026-01-01T00:00:00Z","status":"Active"}}]"#,
            o.build_id, o.space_id
        );
        return respond(
            &mut stream,
            "200 OK",
            &[("Content-Type", "application/json")],
            body.as_bytes(),
        )
        .await;
    }
    if let Some(key) = head.path.strip_prefix(&chunk_prefix) {
        let served = if key == REGISTRY_KEY {
            Some((&o.registry_chunk, o.pending_responses))
        } else {
            o.cells.get(key).map(|c| (c, o.cell_pending_responses))
        };
        if let Some((body, pending)) = served {
            let n = {
                let mut counts = state.requests.lock().expect("request count lock");
                let n = counts.entry(key.to_string()).or_insert(0);
                *n += 1;
                *n - 1
            };
            if n < pending {
                respond(
                    &mut stream,
                    "202 Accepted",
                    &[("Cache-Control", "no-store")],
                    b"",
                )
                .await?;
                state.assembled.send_modify(|keys| {
                    keys.insert(key.to_string());
                });
                return Ok(());
            }
            return respond(
                &mut stream,
                "200 OK",
                &[
                    ("Content-Type", "application/octet-stream"),
                    ("Cache-Control", "public, max-age=31536000, immutable"),
                ],
                body,
            )
            .await;
        }
        return respond(
            &mut stream,
            "404 Not Found",
            &[("Cache-Control", "no-store")],
            b"",
        )
        .await;
    }
    respond(&mut stream, "404 Not Found", &[], b"").await
}

async fn serve_socket(stream: TcpStream, state: Arc<State>) -> std::io::Result<()> {
    let socket = tokio_tungstenite::accept_async(stream)
        .await
        .map_err(std::io::Error::other)?;
    let (mut write, mut read) = socket.split();
    let mut assembled = state.assembled.subscribe();
    let mut subscribed: BTreeSet<String> = BTreeSet::new();
    let mut pushed: BTreeSet<String> = BTreeSet::new();
    loop {
        tokio::select! {
            msg = read.next() => match msg {
                Some(Ok(Message::Text(text))) => {
                    #[derive(serde::Deserialize)]
                    struct Subscribe { subscribe: Option<String> }
                    if let Ok(Subscribe { subscribe: Some(key) }) = serde_json::from_str(text.as_str()) {
                        subscribed.insert(key);
                    }
                }
                Some(Ok(Message::Close(_))) | None | Some(Err(_)) => break,
                Some(Ok(_)) => {}
            },
            changed = assembled.changed() => {
                if changed.is_err() {
                    break;
                }
            }
        }
        let due: Vec<String> = {
            let ready = assembled.borrow_and_update();
            subscribed
                .iter()
                .filter(|k| ready.contains(*k) && !pushed.contains(*k))
                .cloned()
                .collect()
        };
        for key in due {
            let frame = format!(r#"{{"chunkReady":"{key}"}}"#);
            write
                .send(Message::text(frame))
                .await
                .map_err(std::io::Error::other)?;
            state.ready_pushes.fetch_add(1, Ordering::SeqCst);
            pushed.insert(key);
        }
    }
    Ok(())
}

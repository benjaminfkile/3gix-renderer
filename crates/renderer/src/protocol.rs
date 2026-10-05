//! The hub's wire protocol for clients, without any I/O: chunk statuses,
//! readiness socket frames, URLs, the builds listing, and the registry
//! chunk.
//!
//! Shared by the native client ([`crate::hub`]) and the browser client
//! (`crate::web`), which differ only in how they send requests. The
//! protocol is the hub's chunk flow (`compiler-pipeline.md` section 4, "The
//! 202 / Poll Cycle") and the registry, the renderer's first request
//! (`space-model.md` section 7, `matter-format.md` sections 5 and 6).

use anyhow::{anyhow, Result};
use gx_core::container::decode_registry_chunk;
use gx_core::frames::FrameSystem;
use gx_core::registry::FrameTree;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

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

/// `GET /space/{spaceId}/builds` on the hub at `base_url` (no trailing
/// slash).
pub fn builds_url(base_url: &str, space_id: &str) -> String {
    format!("{base_url}/space/{space_id}/builds")
}

/// `GET /space/{spaceId}/build/{buildId}/chunk/{key}` on the hub at
/// `base_url` (no trailing slash).
pub fn chunk_url(base_url: &str, space_id: &str, build_id: &str, key: &str) -> String {
    format!("{base_url}/space/{space_id}/build/{build_id}/chunk/{key}")
}

/// The readiness WebSocket URL of `build`: the hub URL with `http` replaced
/// by `ws` (or `https` by `wss`), then
/// `/space/{spaceId}/build/{buildId}/chunks/ready`.
pub fn ready_url(base_url: &str, space_id: &str, build_id: &str) -> String {
    let ws_base = if let Some(rest) = base_url.strip_prefix("https://") {
        format!("wss://{rest}")
    } else if let Some(rest) = base_url.strip_prefix("http://") {
        format!("ws://{rest}")
    } else {
        base_url.to_string()
    };
    format!("{ws_base}/space/{space_id}/build/{build_id}/chunks/ready")
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
    fn invalid_registry_chunk_is_a_clear_error() {
        let e = frame_system_from_chunk(&[1, 2, 3]).unwrap_err().to_string();
        assert!(e.starts_with("registry is invalid: code"), "{e}");
    }

    #[test]
    fn urls() {
        assert_eq!(
            ready_url("https://hub.invalid", "s", "b"),
            "wss://hub.invalid/space/s/build/b/chunks/ready"
        );
        assert_eq!(
            chunk_url("http://hub.invalid", "s", "b", "registry"),
            "http://hub.invalid/space/s/build/b/chunk/registry"
        );
        assert_eq!(
            builds_url("http://hub.invalid", "s"),
            "http://hub.invalid/space/s/builds"
        );
    }
}

//! The hub's client routes, without any I/O: URLs, the clock, builds and
//! layer configs, directions, upload headers, and the frame registry built
//! from every layer's registry section.
//!
//! Implements the wire side of `client-compute.md` sections 2.2 (layer
//! config), 2.3 (client routes), 2.4 (directions: `ready`, `missing`,
//! `local`, `empty`), 2.5 (the upload headers), and 2.6 (the check body
//! and headers, and the job that recomputes a `ready` section from the
//! layer config, [`check_job`]). There is no readiness
//! socket: nothing waits on another client. Shared by the native client ([`crate::hub`])
//! and the browser client (`crate::web`), which differ only in how they
//! send requests.
//!
//! The registry is just another key (`matter-format.md` section 5): every
//! layer contributes a registry section, and [`frame_system_from_registries`]
//! validates each, takes their union, and raises every frame's depth to the
//! deepest any layer config declares for it (`space-model.md` section 8).

use anyhow::{anyhow, bail, Result};
use gx_core::frames::FrameSystem;
use gx_core::key::{ChunkKey, REGISTRY_KEY};
use gx_core::module_io::ModuleRequest;
use gx_core::registry::{self, FrameTree, Registry};
use gx_core::sha256::{sha256, to_hex};
use gx_worker::job::parse_hash;
use gx_worker::{BlobKind, ComputeJob};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::Arc;

/// The header that carries the API key.
pub const API_KEY_HEADER: &str = "X-API-Key";

/// Upload header: the layer id (`client-compute.md` section 2.5).
pub const LAYER_HEADER: &str = "X-Layer";

/// Upload header: the chunk key.
pub const KEY_HEADER: &str = "X-Key";

/// Upload header: the span index.
pub const SPAN_INDEX_HEADER: &str = "X-Span-Index";

/// Upload header: SHA-256 of the body, lowercase hex.
pub const CONTENT_SHA256_HEADER: &str = "X-Content-Sha256";

/// The section 2.5 headers of an upload of `bytes` for `job`, in a fixed
/// order: `X-Layer`, `X-Key`, `X-Span-Index`, `X-Content-Sha256`. The
/// `Expect: 100-continue` header is added by the native client; the
/// browser's fetch does not let a page set it.
pub fn upload_headers(job: &ComputeJob, bytes: &[u8]) -> [(&'static str, String); 4] {
    [
        (LAYER_HEADER, job.layer.clone()),
        (KEY_HEADER, job.key().to_string()),
        (SPAN_INDEX_HEADER, job.request.span_index.to_string()),
        (CONTENT_SHA256_HEADER, to_hex(&sha256(bytes))),
    ]
}

/// Most keys one directions request may carry (`client-compute.md`
/// section 2.3).
pub const MAX_DIRECTION_KEYS: usize = 512;

/// The deepest cell depth a chunk key can have (`matter-format.md`
/// section 2).
pub const DEEPEST_DEPTH: u8 = 31;

/// `GET /space/{spaceId}/clock`.
#[derive(Copy, Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct Clock {
    /// The universe's origin, UTC milliseconds, or `None` before the first
    /// directions request.
    pub origin_utc_ms: Option<i64>,
    /// The hub's current time, UTC milliseconds.
    pub now_utc_ms: i64,
}

/// How often the client measures its offset from the hub's clock again,
/// seconds (`space-model.md` section 7: now is the hub's now).
pub const CLOCK_SYNC_SECONDS: f64 = 60.0;

/// The hub's clock as the client keeps it: the offset of the hub's now from
/// the client's own monotonic clock, measured at connect time and again
/// every [`CLOCK_SYNC_SECONDS`].
///
/// A measurement takes the local time just before the clock request was
/// sent and just after the answer arrived, and assumes the hub read its
/// clock halfway between the two. Between measurements "now" advances with
/// the local monotonic clock only, so it never jumps with a change of the
/// local wall clock and drifts from the hub's by at most the local clock's
/// rate error over one interval.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ClockSync {
    /// The hub's now minus the local monotonic clock, milliseconds.
    offset_ms: f64,
    /// The local monotonic time of the latest measurement, seconds.
    measured_at: f64,
    /// The origin as the hub last reported it.
    origin_utc_ms: Option<i64>,
}

impl ClockSync {
    /// A clock from one answer, read between the local monotonic times
    /// `sent` and `received`, seconds.
    pub fn measure(clock: Clock, sent: f64, received: f64) -> ClockSync {
        let mut s = ClockSync {
            offset_ms: 0.0,
            measured_at: received,
            origin_utc_ms: None,
        };
        s.update(clock, sent, received);
        s
    }

    /// Takes a new measurement (see [`ClockSync::measure`]).
    pub fn update(&mut self, clock: Clock, sent: f64, received: f64) {
        let mid_ms = 500.0 * (sent + received.max(sent));
        self.offset_ms = clock.now_utc_ms as f64 - mid_ms;
        self.measured_at = received;
        if clock.origin_utc_ms.is_some() {
            self.origin_utc_ms = clock.origin_utc_ms;
        }
    }

    /// The hub's now at the local monotonic time `local`, seconds, in UTC
    /// milliseconds.
    pub fn now_utc_ms(&self, local: f64) -> i64 {
        (local * 1000.0 + self.offset_ms).floor() as i64
    }

    /// The origin, once the hub has reported one.
    pub fn origin_utc_ms(&self) -> Option<i64> {
        self.origin_utc_ms
    }

    /// `true` when the latest measurement is [`CLOCK_SYNC_SECONDS`] or
    /// more before the local monotonic time `local`.
    pub fn due(&self, local: f64) -> bool {
        local - self.measured_at >= CLOCK_SYNC_SECONDS
    }
}

/// `GET /space/{spaceId}/build/active`.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct ActiveBuild {
    /// The active build id.
    pub build_id: String,
}

/// Where a layer is attached (`space-model.md` section 8).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Attachment {
    /// Every frame.
    All,
    /// These frame ids, ascending.
    Frames(Vec<u64>),
}

impl Attachment {
    /// `true` when the layer is directed for matter cells of `frame_id`.
    pub fn contains(&self, frame_id: u64) -> bool {
        match self {
            Attachment::All => true,
            Attachment::Frames(f) => f.binary_search(&frame_id).is_ok(),
        }
    }
}

impl<'de> Deserialize<'de> for Attachment {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Attachment, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Raw {
            Word(String),
            Frames(Vec<u64>),
        }
        match Raw::deserialize(d)? {
            Raw::Word(w) if w == "all" => Ok(Attachment::All),
            Raw::Word(w) => Err(serde::de::Error::custom(format!(
                "attachment {w:?} is neither \"all\" nor a frame list"
            ))),
            Raw::Frames(mut f) => {
                f.sort_unstable();
                f.dedup();
                Ok(Attachment::Frames(f))
            }
        }
    }
}

impl Serialize for Attachment {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            Attachment::All => s.serialize_str("all"),
            Attachment::Frames(f) => f.serialize(s),
        }
    }
}

/// One layer of a build (`client-compute.md` section 2.2), as `GET
/// /space/{spaceId}/build/{buildId}` lists it.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct LayerConfig {
    /// The layer id. An opaque identifier, never shown.
    pub layer: String,
    /// SHA-256 of the module the layer runs, lowercase hex.
    pub module: String,
    /// SHA-256 of the layer's source manifest, or `None`.
    #[serde(default)]
    pub source_manifest: Option<String>,
    /// SHA-256 of the layer's source epoch index, or `None`
    /// (`client-compute.md` section 5.4).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_epochs: Option<String>,
    /// The frames the layer is directed for.
    pub attachment: Attachment,
    /// The layer owns its frames exclusively.
    #[serde(default)]
    pub exclusive: bool,
    /// Deepest cell per frame id (decimal), or `"*"` for the default of a
    /// layer attached to all frames.
    #[serde(default)]
    pub max_depth: BTreeMap<String, u8>,
    /// Span length in seconds; 0 for matter that never changes.
    #[serde(default)]
    pub span_seconds: u64,
    /// The module's output depends on the origin.
    #[serde(default)]
    pub uses_origin: bool,
    /// `false` for an unshared layer: directions answer `local`, and its
    /// sections are never uploaded (`client-compute.md` section 2.2).
    #[serde(default = "shared_by_default")]
    pub shared: bool,
}

fn shared_by_default() -> bool {
    true
}

impl LayerConfig {
    /// The depth this layer's config declares for `frame_id`: the frame's
    /// own entry, else the `"*"` default, else `None` (the config says
    /// nothing, and the registry's `max_depth` stands).
    pub fn declared_depth(&self, frame_id: u64) -> Option<u8> {
        self.max_depth
            .get(&frame_id.to_string())
            .or_else(|| self.max_depth.get("*"))
            .map(|&d| d.min(DEEPEST_DEPTH))
    }
}

/// `GET /space/{spaceId}/build/{buildId}`: the build's layers, ordered by
/// layer id.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct BuildDescription {
    /// The build id.
    pub build_id: String,
    /// The layers.
    pub layers: Vec<LayerConfig>,
}

impl BuildDescription {
    /// `true` when `layer` is directed for matter cells of `frame_id`:
    /// attached, and no exclusive layer of another id owns the frame
    /// (`space-model.md` section 8).
    pub fn directs(&self, layer: &LayerConfig, frame_id: u64) -> bool {
        layer.attachment.contains(frame_id)
            && !self.layers.iter().any(|other| {
                other.layer != layer.layer && other.exclusive && other.attachment.contains(frame_id)
            })
    }

    /// The deepest depth any layer directed for `frame_id` declares for it,
    /// or `None` when no config declares one.
    pub fn layer_depth(&self, frame_id: u64) -> Option<u8> {
        self.layers
            .iter()
            .filter(|l| self.directs(l, frame_id))
            .filter_map(|l| l.declared_depth(frame_id))
            .max()
    }
}

/// The job that recomputes the `ready` section at `address` of `key` in
/// the build's layer `layer`, asked about at `t_utc_ms` in a space of
/// origin `origin_utc_ms` (`client-compute.md` section 2.6, "On the
/// client"): the module and manifest from the layer's config and the span
/// index by the rule of section 2.4, so no extra request is needed. `None`
/// for an unknown layer, a malformed hash, or a layer with source epochs,
/// whose manifest per span needs the epoch index (section 5.4) and is not
/// checked here.
pub fn check_job(
    build: &BuildDescription,
    layer: usize,
    key: ChunkKey,
    address: [u8; 32],
    t_utc_ms: i64,
    origin_utc_ms: i64,
) -> Option<ComputeJob> {
    let config = build.layers.get(layer)?;
    if config.source_epochs.is_some() || !config.shared {
        return None;
    }
    let manifest = match &config.source_manifest {
        Some(h) => Some(parse_hash(h)?),
        None => None,
    };
    let span_index = if config.span_seconds > 0 {
        let start = if config.uses_origin { origin_utc_ms } else { 0 };
        let span_ms = config.span_seconds as i64 * 1000;
        u64::try_from((t_utc_ms - start).div_euclid(span_ms)).ok()?
    } else {
        0
    };
    Some(ComputeJob {
        address,
        layer: config.layer.clone(),
        share: true,
        module: parse_hash(&config.module)?,
        manifest,
        request: ModuleRequest {
            origin_utc_ms: if config.uses_origin { origin_utc_ms } else { 0 },
            span_seconds: config.span_seconds,
            span_index,
            key,
        },
    })
}

/// The section 2.6 headers of a check for `job`, in a fixed order:
/// `X-Layer`, `X-Key`, `X-Span-Index`.
pub fn check_headers(job: &ComputeJob) -> [(&'static str, String); 3] {
    [
        (LAYER_HEADER, job.layer.clone()),
        (KEY_HEADER, job.key().to_string()),
        (SPAN_INDEX_HEADER, job.request.span_index.to_string()),
    ]
}

/// `POST /space/{spaceId}/build/{buildId}/section/{address}/check` body.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct CheckRequest {
    /// SHA-256 of the recomputed section, lowercase hex.
    pub sha256: String,
}

/// `POST /space/{spaceId}/build/{buildId}/directions` body.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct DirectionsRequest {
    /// The moment the cells are drawn at, UTC milliseconds.
    pub t_utc_ms: i64,
    /// Chunk keys, at most [`MAX_DIRECTION_KEYS`].
    pub keys: Vec<String>,
}

/// One layer's answer for one key, as sent. See [`Direction`] for the
/// checked form.
#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct RawDirection {
    /// The layer id.
    pub layer: String,
    /// `ready`, `missing`, `local`, or `empty`.
    pub state: String,
    /// The section's address (all but `empty`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub address: Option<String>,
    /// SHA-256 of the stored bytes (`ready`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
    /// The probability this download should be checked (`ready`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub check_rate: Option<f64>,
    /// The module hash (`missing`, `local`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub module: Option<String>,
    /// The source manifest hash or `null` (`missing`, `local`).
    #[serde(default)]
    pub source_manifest: Option<String>,
    /// The origin the module receives, 0 when the layer does not use it
    /// (`missing`, `local`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub origin_utc_ms: Option<i64>,
    /// The span index (`missing`, `local`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub span_index: Option<u64>,
    /// The span length, seconds (`missing`, `local`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub span_seconds: Option<u64>,
}

/// One key's answers.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct KeyDirections {
    /// The chunk key.
    pub key: String,
    /// One answer per layer, in the build's layer order.
    pub layers: Vec<RawDirection>,
}

/// The directions response.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct DirectionsResponse {
    /// The universe's origin, UTC milliseconds.
    pub origin_utc_ms: i64,
    /// One entry per key asked.
    pub results: Vec<KeyDirections>,
}

/// One layer's answer for one key, checked.
#[derive(Clone, Debug, PartialEq)]
pub enum Direction {
    /// The layer contributes nothing to this cell.
    Empty,
    /// The section is stored: stream it by its address and check the
    /// bytes against `sha256`.
    Ready {
        /// The section's address.
        address: [u8; 32],
        /// SHA-256 of the stored bytes.
        sha256: [u8; 32],
        /// The probability this download should be checked (section 2.6),
        /// clamped to `[0, 1]`.
        check_rate: f64,
    },
    /// No section is stored: compute it, then offer it to the hub. The
    /// job's `share` is `true`.
    Missing(ComputeJob),
    /// The layer is unshared: compute it and keep it, never upload it. The
    /// job's `share` is `false`.
    Local(ComputeJob),
}

impl RawDirection {
    /// Checks the answer for `key` and turns it into a [`Direction`].
    pub fn parse(&self, key: &ChunkKey) -> Result<Direction> {
        let hash = |field: &str, v: &Option<String>| -> Result<[u8; 32]> {
            let s = v
                .as_deref()
                .ok_or_else(|| anyhow!("{} direction without {field}", self.state))?;
            parse_hash(s).ok_or_else(|| anyhow!("{field} {s:?} is not 64 lowercase hex"))
        };
        let job = |share: bool| -> Result<ComputeJob> {
            let manifest = match &self.source_manifest {
                Some(_) => Some(hash("sourceManifest", &self.source_manifest)?),
                None => None,
            };
            Ok(ComputeJob {
                address: hash("address", &self.address)?,
                layer: self.layer.clone(),
                share,
                module: hash("module", &self.module)?,
                manifest,
                request: ModuleRequest {
                    origin_utc_ms: self.origin_utc_ms.unwrap_or(0),
                    span_seconds: self.span_seconds.unwrap_or(0),
                    span_index: self.span_index.unwrap_or(0),
                    key: *key,
                },
            })
        };
        Ok(match self.state.as_str() {
            "empty" => Direction::Empty,
            "ready" => Direction::Ready {
                address: hash("address", &self.address)?,
                sha256: hash("sha256", &self.sha256)?,
                check_rate: self
                    .check_rate
                    .filter(|r| r.is_finite())
                    .unwrap_or(0.0)
                    .clamp(0.0, 1.0),
            },
            "missing" => Direction::Missing(job(true)?),
            "local" => Direction::Local(job(false)?),
            other => bail!("unknown direction state {other:?}"),
        })
    }
}

/// Builds the hub's URLs. Content-addressed reads (sections, modules,
/// sources, manifests) go to the edge, which is the hub itself unless a
/// separate edge URL is configured.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Routes {
    /// Hub base URL, no trailing slash.
    pub hub: String,
    /// Edge base URL for immutable reads, no trailing slash.
    pub edge: String,
    /// The space.
    pub space_id: String,
}

impl Routes {
    /// Routes for `space_id` on `hub`, reading immutable bytes through
    /// `edge` (or the hub when `None`).
    pub fn new(hub: &str, edge: Option<&str>, space_id: &str) -> Routes {
        let hub = hub.trim().trim_end_matches('/').to_string();
        let edge = edge
            .map(|e| e.trim().trim_end_matches('/').to_string())
            .filter(|e| !e.is_empty())
            .unwrap_or_else(|| hub.clone());
        Routes {
            hub,
            edge,
            space_id: space_id.trim().to_string(),
        }
    }

    fn space(&self) -> String {
        format!("{}/space/{}", self.hub, self.space_id)
    }

    /// `GET /space/{spaceId}/clock`.
    pub fn clock(&self) -> String {
        format!("{}/clock", self.space())
    }

    /// `GET /space/{spaceId}/build/active`.
    pub fn active_build(&self) -> String {
        format!("{}/build/active", self.space())
    }

    /// `GET /space/{spaceId}/build/{buildId}`.
    pub fn build(&self, build_id: &str) -> String {
        format!("{}/build/{build_id}", self.space())
    }

    /// `POST /space/{spaceId}/build/{buildId}/directions`.
    pub fn directions(&self, build_id: &str) -> String {
        format!("{}/build/{build_id}/directions", self.space())
    }

    /// `PUT /space/{spaceId}/build/{buildId}/section/{address}`.
    pub fn upload(&self, build_id: &str, address: &str) -> String {
        format!("{}/build/{build_id}/section/{address}", self.space())
    }

    /// `POST /space/{spaceId}/build/{buildId}/section/{address}/check`
    /// (`client-compute.md` section 2.6).
    pub fn check(&self, build_id: &str, address: &str) -> String {
        format!("{}/check", self.upload(build_id, address))
    }

    /// `GET /section/{address}?sha256={hex}` on the edge: one version of a
    /// stored section's bytes (`client-compute.md` sections 2.4 and 8).
    pub fn section(&self, address: &str, sha256: &str) -> String {
        format!("{}?sha256={sha256}", self.blob(BlobKind::Section, address))
    }

    /// `GET /{kind}/{name}` on the edge: `/section/{address}`,
    /// `/module/{hash}`, `/source/{hash}`, `/source-manifest/{hash}`.
    pub fn blob(&self, kind: BlobKind, name: &str) -> String {
        format!("{}/{}/{name}", self.edge, kind.segment())
    }
}

/// A message for an HTTP status the hub answered with, or `None` for
/// `200` and `201`.
pub fn status_problem(status: u16) -> Option<String> {
    match status {
        200 | 201 => None,
        401 => Some("hub refused the API key (401 Unauthorized)".into()),
        403 => Some("the API key lacks the fetch:chunks capability (403 Forbidden)".into()),
        other => Some(format!("hub answered {other}")),
    }
}

/// Builds the frame system from every layer's registry section: validates
/// each with the library validator, takes their union
/// ([`FrameTree::from_registries`]), and raises each frame's `max_depth` to
/// the deepest the layer configs declare for it ([`BuildDescription::layer_depth`]).
/// Fails with the validator's code and reason if any registry or the union
/// is invalid.
pub fn frame_system_from_registries(
    sections: &[Arc<[u8]>],
    build: &BuildDescription,
) -> Result<FrameSystem> {
    let mut registries = Vec::with_capacity(sections.len());
    for bytes in sections {
        gx_core::validate(REGISTRY_KEY, bytes)
            .map_err(|e| anyhow!("registry is invalid: code {}: {}", e.code, e.reason))?;
        let r = registry::decode(bytes)
            .map_err(|e| anyhow!("registry is invalid: code {}: {}", e.code, e.reason))?;
        if r.is_empty() {
            continue;
        }
        let frames = r
            .frames()
            .iter()
            .map(|f| {
                let mut f = *f;
                if let Some(d) = build.layer_depth(f.frame_id) {
                    f.max_depth = f.max_depth.max(d);
                }
                f
            })
            .collect();
        registries.push(
            Registry::new(r.epoch(), frames)
                .map_err(|e| anyhow!("registry is invalid: code {}: {}", e.code, e.reason))?,
        );
    }
    let tree = FrameTree::from_registries(&registries)
        .map_err(|e| anyhow!("registry is invalid: code {}: {}", e.code, e.reason))?;
    Ok(FrameSystem::from_tree(tree))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build() -> BuildDescription {
        serde_json::from_str(
            r#"{"buildId":"b","layers":[
                {"layer":"a","module":"00","sourceManifest":null,"attachment":"all",
                 "exclusive":false,"maxDepth":{"*":3},"spanSeconds":0,"usesOrigin":true},
                {"layer":"b","module":"00","sourceManifest":null,"attachment":[4],
                 "exclusive":true,"maxDepth":{"4":21},"spanSeconds":0,"usesOrigin":false},
                {"layer":"c","module":"00","sourceManifest":null,"attachment":[5,2],
                 "exclusive":false,"maxDepth":{"2":1,"5":9},"spanSeconds":60,"usesOrigin":false}]}"#,
        )
        .unwrap()
    }

    #[test]
    fn layer_depths_and_exclusivity() {
        let b = build();
        assert_eq!(b.layers[2].attachment, Attachment::Frames(vec![2, 5]));
        // Frame 4: owned by "b", so "a" is not directed there.
        assert!(!b.directs(&b.layers[0], 4));
        assert_eq!(b.layer_depth(4), Some(21));
        assert_eq!(b.layer_depth(5), Some(9));
        assert_eq!(b.layer_depth(2), Some(3));
        assert_eq!(b.layer_depth(7), Some(3));
        let mut none = b.clone();
        none.layers[0].max_depth.clear();
        assert_eq!(none.layer_depth(7), None);
        assert!(serde_json::from_str::<Attachment>(r#""some""#).is_err());
    }

    #[test]
    fn directions_parse_and_check() {
        let key: ChunkKey = "4-12-1024-2048-3071".parse().unwrap();
        let resp: DirectionsResponse = serde_json::from_str(&format!(
            r#"{{"originUtcMs":1763130000000,"results":[{{"key":"4-12-1024-2048-3071","layers":[
                {{"layer":"system","state":"empty"}},
                {{"layer":"matter","state":"missing","address":"{a}",
                  "module":"{m}","sourceManifest":"{s}","originUtcMs":0,"spanIndex":4,"spanSeconds":60}},
                {{"layer":"x","state":"local","address":"{a}","module":"{m}","sourceManifest":null,
                  "originUtcMs":0,"spanIndex":0,"spanSeconds":0}},
                {{"layer":"y","state":"ready","address":"{a}","sha256":"{s}","checkRate":0.5}}]}}]}}"#,
            a = "1".repeat(64),
            m = "2".repeat(64),
            s = "3".repeat(64)
        ))
        .unwrap();
        let l = &resp.results[0].layers;
        assert_eq!(l[0].parse(&key).unwrap(), Direction::Empty);
        match l[1].parse(&key).unwrap() {
            Direction::Missing(job) => {
                assert_eq!(job.layer, "matter");
                assert!(job.share);
                assert_eq!(job.module, [0x22; 32]);
                assert_eq!(job.manifest, Some([0x33; 32]));
                assert_eq!(job.request.key, key);
                let h = upload_headers(&job, b"abc");
                assert_eq!(h[0], (LAYER_HEADER, "matter".to_string()));
                assert_eq!(h[1], (KEY_HEADER, "4-12-1024-2048-3071".to_string()));
                assert_eq!(h[2], (SPAN_INDEX_HEADER, "4".to_string()));
                assert_eq!(
                    h[3],
                    (
                        CONTENT_SHA256_HEADER,
                        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad".into()
                    )
                );
            }
            other => panic!("{other:?}"),
        }
        match l[2].parse(&key).unwrap() {
            Direction::Local(job) => assert!(!job.share && job.manifest.is_none()),
            other => panic!("{other:?}"),
        }
        assert_eq!(
            l[3].parse(&key).unwrap(),
            Direction::Ready {
                address: [0x11; 32],
                sha256: [0x33; 32],
                check_rate: 0.5
            }
        );
        let no_hash = RawDirection {
            state: "ready".into(),
            address: Some("1".repeat(64)),
            ..RawDirection::default()
        };
        assert!(no_hash.parse(&key).is_err());
        let bad = RawDirection {
            state: "ready".into(),
            address: Some("XYZ".into()),
            ..RawDirection::default()
        };
        assert!(bad.parse(&key).is_err());
        for removed in ["compute", "pending", "later"] {
            let odd = RawDirection {
                state: removed.into(),
                address: Some("1".repeat(64)),
                ..RawDirection::default()
            };
            assert!(odd.parse(&key).is_err(), "{removed}");
        }
    }

    #[test]
    fn urls() {
        let r = Routes::new("https://hub.invalid/", None, "s");
        assert_eq!(r.clock(), "https://hub.invalid/space/s/clock");
        assert_eq!(
            r.directions("b"),
            "https://hub.invalid/space/s/build/b/directions"
        );
        assert_eq!(
            r.upload("b", "x"),
            "https://hub.invalid/space/s/build/b/section/x"
        );
        assert_eq!(
            r.check("b", "x"),
            "https://hub.invalid/space/s/build/b/section/x/check"
        );
        assert_eq!(
            r.blob(BlobKind::Manifest, "h"),
            "https://hub.invalid/source-manifest/h"
        );
        let r = Routes::new("http://hub.invalid", Some("http://edge.invalid/"), "s");
        assert_eq!(
            r.blob(BlobKind::Section, "a"),
            "http://edge.invalid/section/a"
        );
        assert_eq!(
            r.section("a", "f0"),
            "http://edge.invalid/section/a?sha256=f0"
        );
    }

    #[test]
    fn clock_sync_takes_the_midpoint_and_follows_the_local_clock() {
        let clock = Clock {
            origin_utc_ms: None,
            now_utc_ms: 1_000_000,
        };
        // Sent at 10 s, answered at 10.2 s: the hub read its clock at 10.1 s.
        let mut s = ClockSync::measure(clock, 10.0, 10.2);
        assert_eq!(s.now_utc_ms(10.1), 1_000_000);
        assert_eq!(s.now_utc_ms(12.1), 1_002_000);
        assert_eq!(s.origin_utc_ms(), None);
        assert!(!s.due(10.2 + CLOCK_SYNC_SECONDS - 0.5));
        assert!(s.due(10.2 + CLOCK_SYNC_SECONDS));
        // The hub's clock ran 30 ms ahead of the local one meanwhile.
        s.update(
            Clock {
                origin_utc_ms: Some(5),
                now_utc_ms: 1_060_030,
            },
            70.1,
            70.1,
        );
        assert_eq!(s.now_utc_ms(70.1), 1_060_030);
        assert_eq!(s.origin_utc_ms(), Some(5));
        assert!(!s.due(70.2));
    }

    #[test]
    fn invalid_registry_is_a_clear_error() {
        let b = BuildDescription {
            build_id: "b".into(),
            layers: Vec::new(),
        };
        let e = frame_system_from_registries(&[Arc::from(&[1u8, 2, 3][..])], &b)
            .unwrap_err()
            .to_string();
        assert!(e.starts_with("registry is invalid: code"), "{e}");
    }
}

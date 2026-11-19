//! What the worker is asked to do and the checks it makes on everything it
//! reads and returns, without any I/O.
//!
//! Implements the platform-independent part of `client-compute.md`
//! section 1 step 4, section 3 (addresses), section 4 (the module request),
//! and section 5.2 (resolving and checking source tiles). The native pool
//! ([`crate::native`]) and the browser worker (the renderer's `web`
//! module) both run their jobs through these functions, so the two
//! platforms check exactly the same things.

use gx_core::address::{section_address, section_address_v2, SlotValue};
use gx_core::key::ChunkKey;
use gx_core::manifest::{self, Manifest};
use gx_core::matter::Section;
use gx_core::module_io::ModuleRequest;
use gx_core::sha256::{sha256, to_hex};
use std::fmt;
use std::sync::Arc;

/// The kinds of content-addressed bytes the client reads and caches
/// (`client-compute.md` sections 2.3 and 6).
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BlobKind {
    /// A module's WebAssembly bytes, named by their SHA-256.
    Module,
    /// A source manifest, named by its SHA-256.
    Manifest,
    /// A source tile, named by its SHA-256.
    Source,
    /// A section, named by its address (not the hash of its bytes).
    Section,
    /// A frame record (`client-compute.md` section 10.2), named by its
    /// SHA-256.
    FrameRecord,
}

impl BlobKind {
    /// Every kind, in a fixed order.
    pub const ALL: [BlobKind; 5] = [
        BlobKind::Module,
        BlobKind::Manifest,
        BlobKind::Source,
        BlobKind::Section,
        BlobKind::FrameRecord,
    ];

    /// The route segment the hub serves this kind under: `GET
    /// /{segment}/{name}`. Also the cache's directory name for the kind.
    pub fn segment(self) -> &'static str {
        match self {
            BlobKind::Module => "module",
            BlobKind::Manifest => "source-manifest",
            BlobKind::Source => "source",
            BlobKind::Section => "section",
            BlobKind::FrameRecord => "frame-record",
        }
    }

    /// `true` for the kinds named by the SHA-256 of their own bytes, whose
    /// bytes are checked against their name.
    pub fn is_hash_named(self) -> bool {
        self != BlobKind::Section
    }
}

impl fmt::Display for BlobKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.segment())
    }
}

/// `true` when `name` is a SHA-256 or an address as the hub writes them: 64
/// lowercase hex characters.
pub fn is_hex_name(name: &str) -> bool {
    name.len() == 64 && name.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

/// Parses 64 lowercase hex characters into 32 bytes.
pub fn parse_hash(name: &str) -> Option<[u8; 32]> {
    if !is_hex_name(name) {
        return None;
    }
    gx_core::sha256::from_hex(name)
}

/// Whether a failure can go away by trying again.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum FailureKind {
    /// The network, the cache, or the hub failed; trying again may work.
    Transient,
    /// The module or the job itself is at fault. By the determinism rule
    /// the same job fails the same way again.
    Permanent,
    /// The job's key left the wanted set before the job started, so it
    /// never ran (`client-compute.md` section 2.8). Nothing went wrong.
    Dropped,
}

/// Why a job failed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorkerError {
    /// Whether to try again.
    pub kind: FailureKind,
    /// What went wrong, for the log.
    pub message: String,
}

impl WorkerError {
    /// A failure that may go away on a later try.
    pub fn transient(message: impl Into<String>) -> WorkerError {
        WorkerError {
            kind: FailureKind::Transient,
            message: message.into(),
        }
    }

    /// A job removed from the queue before it started.
    pub fn dropped() -> WorkerError {
        WorkerError {
            kind: FailureKind::Dropped,
            message: "dropped: the key is no longer wanted".into(),
        }
    }

    /// A failure that repeats on every try.
    pub fn permanent(message: impl Into<String>) -> WorkerError {
        WorkerError {
            kind: FailureKind::Permanent,
            message: message.into(),
        }
    }
}

impl fmt::Display for WorkerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for WorkerError {}

/// One `missing` or `local` direction (`client-compute.md` section 2.4):
/// the section to produce, whether to offer it to the hub afterwards, and
/// everything that goes into it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ComputeJob {
    /// The section's address.
    pub address: [u8; 32],
    /// The layer id, sent as `X-Layer` with an upload.
    pub layer: String,
    /// `true` for a `missing` direction: once computed, the section is
    /// offered to the hub (section 2.5). `false` for `local`: computed,
    /// kept, never uploaded.
    pub share: bool,
    /// SHA-256 of the module to run.
    pub module: [u8; 32],
    /// SHA-256 of the layer's source manifest, if the layer reads sources.
    pub manifest: Option<[u8; 32]>,
    /// The request the module receives: key, origin, and span.
    pub request: ModuleRequest,
    /// The layer's input slots in slot order (`client-compute.md` section
    /// 10.5), which make the address version 2. Empty for a layer without
    /// inputs, whose address is version 1.
    pub slots: Vec<SlotValue>,
    /// The records of the layer's frames slots, in slot order and, within a
    /// slot, in the order the config lists the frames (section 4.1). They
    /// come from the build. Empty for a layer without frames slots.
    pub frames: Vec<Arc<[u8]>>,
}

impl ComputeJob {
    /// The address as lowercase hex, the form routes and caches use.
    pub fn address_hex(&self) -> String {
        to_hex(&self.address)
    }

    /// The chunk key of the section.
    pub fn key(&self) -> ChunkKey {
        self.request.key
    }

    /// Checks that the address the hub sent is the address of everything
    /// that goes into the section (`client-compute.md` section 3). A
    /// mismatch means the hub and this client disagree on the inputs, and
    /// the section would be stored under the wrong name.
    pub fn check_address(&self) -> Result<(), WorkerError> {
        let r = &self.request;
        let expected = if self.slots.is_empty() {
            section_address(
                &self.module,
                self.manifest.as_ref(),
                r.origin_utc_ms,
                r.span_seconds,
                r.span_index,
                &r.key,
            )
        } else {
            section_address_v2(
                &self.module,
                self.manifest.as_ref(),
                r.origin_utc_ms,
                r.span_seconds,
                r.span_index,
                &self.slots,
                &r.key,
            )
            .map_err(|e| WorkerError::permanent(format!("the job's slots: {e}")))?
        };
        if expected == self.address {
            Ok(())
        } else {
            Err(WorkerError::permanent(format!(
                "address {} does not match the job's inputs, which address {}",
                self.address_hex(),
                to_hex(&expected)
            )))
        }
    }
}

/// Checks bytes fetched or read back under a hash-named kind: their SHA-256
/// must be `hash`. Sections are named by address and are checked by the
/// validator instead, so this accepts them unchanged.
pub fn check_blob(kind: BlobKind, hash: &[u8; 32], bytes: &[u8]) -> Result<(), WorkerError> {
    if !kind.is_hash_named() {
        return Ok(());
    }
    let actual = sha256(bytes);
    if actual == *hash {
        Ok(())
    } else {
        Err(WorkerError::transient(format!(
            "{kind} {}: the bytes hash to {}",
            to_hex(hash),
            to_hex(&actual)
        )))
    }
}

/// Decodes and validates a source manifest whose bytes already passed
/// [`check_blob`].
pub fn decode_manifest(bytes: &[u8]) -> Result<Manifest, WorkerError> {
    manifest::decode(bytes)
        .map_err(|e| WorkerError::permanent(format!("source manifest is invalid: {e}")))
}

/// One source tile to fetch: its id, the SHA-256 it is fetched by, and its
/// length, from the layer's manifest.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct TileRef {
    /// The tile id the module asked for.
    pub id: u64,
    /// SHA-256 of the tile's bytes.
    pub sha256: [u8; 32],
    /// Length of the tile's bytes.
    pub length: u64,
}

/// Resolves the tile ids `gx_sources` returned through the layer's
/// manifest (`client-compute.md` section 5.2). An id the manifest does not
/// list, or any id when the layer has no manifest, is a module failure.
pub fn resolve_tiles(
    manifest: Option<&Manifest>,
    ids: &[u64],
) -> Result<Vec<TileRef>, WorkerError> {
    if ids.is_empty() {
        return Ok(Vec::new());
    }
    let Some(m) = manifest else {
        return Err(WorkerError::permanent(format!(
            "the module asked for {} source tiles but the layer has no source manifest",
            ids.len()
        )));
    };
    ids.iter()
        .map(|&id| {
            m.lookup(id)
                .map(|(sha256, length)| TileRef { id, sha256, length })
                .ok_or_else(|| {
                    WorkerError::permanent(format!(
                        "the module asked for source tile {id}, which the manifest does not list"
                    ))
                })
        })
        .collect()
}

/// Checks a fetched tile against its manifest entry: length and SHA-256.
pub fn check_tile(tile: &TileRef, bytes: &[u8]) -> Result<(), WorkerError> {
    if bytes.len() as u64 != tile.length {
        return Err(WorkerError::transient(format!(
            "source tile {}: {} bytes, the manifest says {}",
            tile.id,
            bytes.len(),
            tile.length
        )));
    }
    check_blob(BlobKind::Source, &tile.sha256, bytes)
}

/// A section the worker computed: its bytes, as uploaded, and for a cell
/// key the section decoded from them.
#[derive(Clone, Debug, PartialEq)]
pub struct Computed {
    /// The section bytes the module returned.
    pub bytes: Arc<[u8]>,
    /// The decoded section for a cell key, `None` for the registry key.
    /// Decoding is the validation, so the renderer takes this instead of
    /// decoding the bytes a second time (`rendering.md` section 3.5).
    pub section: Option<Arc<Section>>,
}

/// Validates a section under its chunk key (`space-model.md` section 4
/// item 2: the worker validates before upload). For a cell key this is the
/// library decoder, which runs every validation rule and returns the
/// decoded section, so the bytes are validated once; for the registry key
/// it is the library validator and the result is `None`.
pub fn check_section(key: &ChunkKey, bytes: &[u8]) -> Result<Option<Section>, WorkerError> {
    let fail = |e: gx_core::error::ValidationError| {
        WorkerError::permanent(format!("section {key} fails validation: {e}"))
    };
    match key {
        ChunkKey::Cell(cell) => gx_core::matter::decode(cell, bytes).map(Some).map_err(fail),
        ChunkKey::Registry | ChunkKey::Surface(_) | ChunkKey::Frame(_) => {
            gx_core::validate(&key.to_string(), bytes)
                .map(|()| None)
                .map_err(fail)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gx_core::manifest::Entry;

    fn job() -> ComputeJob {
        let request = ModuleRequest {
            origin_utc_ms: 0,
            span_seconds: 0,
            span_index: 0,
            key: "1-0-0-0-0".parse().unwrap(),
        };
        let module = sha256(b"module");
        ComputeJob {
            address: section_address(&module, None, 0, 0, 0, &request.key),
            layer: "l".into(),
            share: true,
            module,
            manifest: None,
            request,
            slots: Vec::new(),
            frames: Vec::new(),
        }
    }

    #[test]
    fn names_and_kinds() {
        assert!(is_hex_name(&"a".repeat(64)));
        assert!(!is_hex_name(&"A".repeat(64)));
        assert!(!is_hex_name("../x"));
        assert_eq!(parse_hash(&"0".repeat(64)), Some([0; 32]));
        assert_eq!(BlobKind::Manifest.segment(), "source-manifest");
        assert!(!BlobKind::Section.is_hash_named());
    }

    #[test]
    fn address_is_checked() {
        let mut j = job();
        assert!(j.check_address().is_ok());
        j.request.span_index = 1;
        assert_eq!(j.check_address().unwrap_err().kind, FailureKind::Permanent);
    }

    #[test]
    fn blobs_and_tiles_are_checked() {
        let h = sha256(b"abc");
        assert!(check_blob(BlobKind::Module, &h, b"abc").is_ok());
        assert!(check_blob(BlobKind::Module, &h, b"abd").is_err());
        assert!(check_blob(BlobKind::Section, &h, b"anything").is_ok());
        let m = Manifest::new(vec![Entry {
            id: 7,
            sha256: h,
            length: 3,
        }])
        .unwrap();
        let tiles = resolve_tiles(Some(&m), &[7]).unwrap();
        assert_eq!(tiles[0].length, 3);
        assert!(check_tile(&tiles[0], b"abc").is_ok());
        assert!(check_tile(&tiles[0], b"abcd").is_err());
        assert!(resolve_tiles(Some(&m), &[8]).is_err());
        assert!(resolve_tiles(None, &[7]).is_err());
        assert!(resolve_tiles(None, &[]).unwrap().is_empty());
    }
}

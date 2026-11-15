//! What the worker is asked to do and the checks it makes on everything it
//! reads and returns, without any I/O.
//!
//! Implements the platform-independent part of `client-compute.md`
//! section 1 step 4, section 3 (addresses), section 4 (the module request),
//! and section 5.2 (resolving and checking source tiles). The native pool
//! ([`crate::native`]) and the browser worker (the renderer's `web`
//! module) both run their jobs through these functions, so the two
//! platforms check exactly the same things.

use gx_core::address::section_address;
use gx_core::key::ChunkKey;
use gx_core::manifest::{self, Manifest};
use gx_core::module_io::ModuleRequest;
use gx_core::sha256::{sha256, to_hex};
use std::fmt;

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
}

impl BlobKind {
    /// Every kind, in a fixed order.
    pub const ALL: [BlobKind; 4] = [
        BlobKind::Module,
        BlobKind::Manifest,
        BlobKind::Source,
        BlobKind::Section,
    ];

    /// The route segment the hub serves this kind under: `GET
    /// /{segment}/{name}`. Also the cache's directory name for the kind.
    pub fn segment(self) -> &'static str {
        match self {
            BlobKind::Module => "module",
            BlobKind::Manifest => "source-manifest",
            BlobKind::Source => "source",
            BlobKind::Section => "section",
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

/// One `compute` direction (`client-compute.md` section 2.4): the section
/// to produce, the claim to upload it with, and everything that goes into
/// it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ComputeJob {
    /// The section's address.
    pub address: [u8; 32],
    /// The claim token the hub gave this client, sent as `X-Claim`.
    pub claim: String,
    /// SHA-256 of the module to run.
    pub module: [u8; 32],
    /// SHA-256 of the layer's source manifest, if the layer reads sources.
    pub manifest: Option<[u8; 32]>,
    /// The request the module receives: key, origin, and span.
    pub request: ModuleRequest,
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
        let expected = section_address(
            &self.module,
            self.manifest.as_ref(),
            r.origin_utc_ms,
            r.span_seconds,
            r.span_index,
            &r.key,
        );
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

/// Validates a section with the library validator under its chunk key
/// (`space-model.md` section 4 item 2: the worker validates before upload).
pub fn check_section(key: &ChunkKey, bytes: &[u8]) -> Result<(), WorkerError> {
    gx_core::validate(&key.to_string(), bytes)
        .map_err(|e| WorkerError::permanent(format!("section {key} fails validation: {e}")))
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
            claim: "c".into(),
            module,
            manifest: None,
            request,
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

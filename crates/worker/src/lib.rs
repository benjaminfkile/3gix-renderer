//! The client's worker (`space-model.md` section 2, "The worker runs
//! compiler modules"; `client-compute.md` sections 1, 4, 5.2, and 6).
//!
//! The worker is generic: it runs any module and knows nothing about what a
//! module computes. It only ever sees modules, requests, tile bytes, and
//! section bytes. Given one `compute` direction it fetches the module and
//! the source tiles the module asks for, all by hash and all checked, runs
//! `gx_sources` then `gx_compile`, validates the section with the library
//! validator, hands it to the renderer at once, and uploads it with the
//! claim.
//!
//! - [`job`]: the job, the blob kinds, and the checks on every byte read
//!   and returned, shared by both platforms.
//! - [`stats`]: counters for the overlay.
//! - [`cache`]: the native local cache directory (native only).
//! - [`native`]: the thread pool running modules with `gx-module-host`
//!   (native only).
//!
//! The browser runs modules with its own `WebAssembly` in a dedicated Web
//! Worker (`web/module-worker.js` and the renderer's `web` module), through
//! the same [`job`] checks.

#![warn(missing_docs)]

pub mod job;
pub mod stats;

#[cfg(not(target_arch = "wasm32"))]
pub mod cache;
#[cfg(not(target_arch = "wasm32"))]
pub mod native;

pub use job::{BlobKind, ComputeJob, FailureKind, TileRef, WorkerError};
pub use stats::{UploadTicket, WorkerSnapshot, WorkerStats};

#[cfg(not(target_arch = "wasm32"))]
pub use cache::DiskCache;
#[cfg(not(target_arch = "wasm32"))]
pub use native::{default_threads, Blobs, ModuleRunner, Uploader, WorkerPool};

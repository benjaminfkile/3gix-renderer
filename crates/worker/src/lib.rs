//! The client's worker (`space-model.md` section 2, "The worker runs
//! compiler modules"; `client-compute.md` sections 1, 4, 5.2, and 6).
//!
//! The worker is generic: it runs any module and knows nothing about what a
//! module computes. It only ever sees modules, requests, tile bytes, and
//! section bytes. Given one `missing` or `local` direction it fetches the
//! module and the source tiles the module asks for, all by hash and all
//! checked, runs `gx_sources` then `gx_compile`, validates the section
//! with the library validator, and hands it to the renderer at once. A
//! `missing` section is then offered to the hub once from the bounded
//! upload lane; a `local` one never leaves the client.
//!
//! - [`job`]: the job, the blob kinds, and the checks on every byte read
//!   and returned, shared by both platforms.
//! - [`stats`]: counters for the overlay.
//! - [`queue`]: the work list's order: three lanes (computes, uploads,
//!   checks) and the compute queue, highest priority first, shared by
//!   both platforms.
//! - [`resolve`]: input resolution (section 10.3), async over a small
//!   platform trait, shared by both platforms.
//! - [`upload`]: the bounded upload lane and the outcome of each offer,
//!   shared by both platforms.
//! - [`check`]: the bounded check lane and the outcome of each check,
//!   shared by both platforms.
//! - [`cache`]: the native local cache directory (native only).
//! - [`native`]: the thread pool running modules with `gx-module-host`
//!   (native only).
//!
//! The browser runs modules with its own `WebAssembly` in a dedicated Web
//! Worker (`web/module-worker.js` and the renderer's `web` module), through
//! the same [`job`] checks.

#![warn(missing_docs)]

pub mod check;
pub mod job;
pub mod queue;
pub mod resolve;
pub mod stats;
pub mod upload;

#[cfg(not(target_arch = "wasm32"))]
pub mod cache;
#[cfg(not(target_arch = "wasm32"))]
pub mod native;

pub use check::{CheckJob, CheckLane, CheckOutcome, CHECK_LANE_CAPACITY};
pub use job::{BlobKind, ComputeJob, Computed, FailureKind, TileRef, WorkerError};
pub use queue::{ComputeQueue, Lane};
pub use resolve::{InputDirection, InputResolver, Platform};
pub use stats::{UploadTicket, WorkerSnapshot, WorkerStats};
pub use upload::{
    UploadLane, UploadOffer, UploadOutcome, MAX_UPLOADS_IN_FLIGHT, UPLOAD_LANE_CAPACITY,
};

#[cfg(not(target_arch = "wasm32"))]
pub use cache::DiskCache;
#[cfg(not(target_arch = "wasm32"))]
pub use native::{
    default_threads, Blobs, CheckRunner, InputSource, LaneSource, ModuleRunner, PoolWaker,
    Resolver, Uploader, WorkerPool,
};

//! The memory report: a developer tool for the memory wave, built only with
//! the `memory-report` cargo feature (`docs/architecture.md`, "Memory
//! report"). It holds no cache and changes nothing drawn (`rendering.md`
//! section 8 item 7); without the feature none of it is compiled.
//!
//! Three parts: a counting global allocator ([`Counting`]) wrapping
//! [`std::alloc::System`], timers for the decode steps ([`Timer`]), and the
//! report ([`MemoryReport`], from [`crate::world::World::memory_report`]).
//! Byte counts are `Vec` capacity times element size plus the struct
//! itself; each type counts its own bytes (`heap_bytes()` next to it).

use serde::Serialize;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// The system allocator, counting live and peak bytes.
pub struct Counting;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

fn grew(bytes: usize) {
    let live = LIVE.fetch_add(bytes, Ordering::Relaxed) + bytes;
    PEAK.fetch_max(live, Ordering::Relaxed);
}

fn shrank(bytes: usize) {
    LIVE.fetch_sub(bytes, Ordering::Relaxed);
}

// SAFETY: every call is forwarded to `System` unchanged; only counters are
// updated around it.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            grew(layout.size());
        }
        p
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc_zeroed(layout) };
        if !p.is_null() {
            grew(layout.size());
        }
        p
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        shrank(layout.size());
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let p = unsafe { System.realloc(ptr, layout, new_size) };
        if !p.is_null() {
            shrank(layout.size());
            grew(new_size);
        }
        p
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// Bytes the allocator holds for the program now and at most so far.
#[derive(Copy, Clone, Debug, Default, PartialEq, Serialize)]
pub struct AllocatorBytes {
    /// Bytes allocated and not freed.
    pub live: u64,
    /// The largest `live` has been.
    pub peak: u64,
}

impl AllocatorBytes {
    /// The counters now.
    pub fn now() -> AllocatorBytes {
        AllocatorBytes {
            live: LIVE.load(Ordering::Relaxed) as u64,
            peak: PEAK.load(Ordering::Relaxed) as u64,
        }
    }
}

/// Count and total time of one step, kept in atomics so any thread may
/// record.
pub struct Timer {
    count: AtomicU64,
    nanos: AtomicU64,
}

impl Timer {
    const fn new() -> Timer {
        Timer {
            count: AtomicU64::new(0),
            nanos: AtomicU64::new(0),
        }
    }

    /// The count and mean so far.
    pub fn timing(&self) -> Timing {
        let count = self.count.load(Ordering::Relaxed);
        let nanos = self.nanos.load(Ordering::Relaxed);
        Timing {
            count,
            mean_us: if count == 0 {
                0.0
            } else {
                nanos as f64 / count as f64 / 1e3
            },
        }
    }
}

/// [`crate::tiles::TileGeometry::new`].
pub static TILE_GEOMETRY: Timer = Timer::new();
/// [`crate::tiles::decode_tile`].
pub static DECODE_TILE: Timer = Timer::new();
/// A cell section's decode (`gx_core::matter::decode`).
pub static CELL_DECODE: Timer = Timer::new();

/// Runs `f`, adding its time to `timer` on the session clock
/// ([`crate::clock::now_seconds`], `performance.now()` in the browser).
pub fn timed<T>(timer: &Timer, f: impl FnOnce() -> T) -> T {
    let start = crate::clock::now_seconds();
    let out = f();
    let nanos = ((crate::clock::now_seconds() - start) * 1e9).max(0.0) as u64;
    timer.count.fetch_add(1, Ordering::Relaxed);
    timer.nanos.fetch_add(nanos, Ordering::Relaxed);
    out
}

/// How many times a step ran and its mean time.
#[derive(Copy, Clone, Debug, Default, PartialEq, Serialize)]
pub struct Timing {
    /// Times it ran.
    pub count: u64,
    /// Mean time, microseconds; 0 before the first run.
    pub mean_us: f64,
}

/// The decode timings.
#[derive(Copy, Clone, Debug, Default, PartialEq, Serialize)]
pub struct Timings {
    /// [`crate::tiles::TileGeometry::new`].
    pub tile_geometry_new: Timing,
    /// [`crate::tiles::decode_tile`], which includes its
    /// `TileGeometry::new` calls.
    pub decode_tile: Timing,
    /// A cell section's decode.
    pub cell_decode: Timing,
}

impl Timings {
    /// The timings so far.
    pub fn now() -> Timings {
        Timings {
            tile_geometry_new: TILE_GEOMETRY.timing(),
            decode_tile: DECODE_TILE.timing(),
            cell_decode: CELL_DECODE.timing(),
        }
    }
}

/// Surface tile memory ([`crate::tiles::Surfaces::memory`]).
#[derive(Copy, Clone, Debug, Default, PartialEq, Serialize)]
pub struct SurfaceMemory {
    /// Entries requested and not arrived.
    pub requested: usize,
    /// Entries arrived with a surface, built or compact.
    pub ready: usize,
    /// Arrived tiles held compact (`crate::tiles` module docs).
    pub compact: usize,
    /// Bytes of the compact tiles: section bytes, summary, and overlays.
    pub compact_bytes: u64,
    /// Arrived tiles held built.
    pub built: usize,
    /// Bytes of the built tiles: section bytes, summary, overlays, and
    /// geometry (material and shape).
    pub built_bytes: u64,
    /// Of the built tiles, those drawn.
    pub built_drawn: usize,
    /// Of the built tiles, children a drawn tile's split is waiting for.
    pub built_for_split: usize,
    /// Of the built tiles, split tiles about to join (drawn again).
    pub built_for_join: usize,
    /// Of the built tiles, level-0 tiles held for no other reason.
    pub built_level_0: usize,
    /// Of the built tiles, those wanted built for no reason now, held
    /// within [`crate::tiles::BUILT_GRACE_SECONDS`].
    pub built_in_grace: usize,
    /// Entries known empty (or with every base hidden).
    pub empty: usize,
    /// Entries whose sections were invalid.
    pub gone: usize,
    /// Tiles in the latest selection's `used` set, which eviction keeps.
    pub used: usize,
    /// Tiles drawn by the latest selection.
    pub drawn: usize,
    /// Distinct tile geometries held, each `Arc` once.
    pub geometries: usize,
    /// Distinct tile shapes held, each `Arc` once: one per tile, shared
    /// by its base and every composite of it.
    pub shapes: usize,
    /// Bytes of those shapes.
    pub shape_bytes: u64,
    /// Bytes of those geometries: each one's material, plus each shape
    /// once ([`SurfaceMemory::shape_bytes`]).
    pub geometry_bytes: u64,
    /// Of those, composited copies: geometries that are no tile's base.
    pub composited: usize,
    /// Bytes of the composited copies: their material only, since they
    /// share their base's shape.
    pub composited_bytes: u64,
    /// Bytes of the geometries the drawn tiles need: the drawn geometry
    /// and the tile's bases, each material and each shape once.
    pub drawn_geometry_bytes: u64,
    /// Bytes of every distinct overlay tile held.
    pub overlay_bytes: u64,
    /// Bytes of the drawn tiles' overlay tiles.
    pub drawn_overlay_bytes: u64,
}

/// Cell memory ([`crate::stream::CellCache::memory`]).
#[derive(Copy, Clone, Debug, Default, PartialEq, Serialize)]
pub struct CellMemory {
    /// Cells in the cache.
    pub cached: usize,
    /// Cached cells that are `Ready`.
    pub ready: usize,
    /// Cells in the drawn selection.
    pub selected: usize,
    /// Bytes of every distinct decoded section held (the composite and
    /// each layer's), each `Arc` once.
    pub sample_bytes: u64,
    /// Bytes of the selected cells' decoded sections.
    pub selected_sample_bytes: u64,
}

/// Extracted mesh memory.
#[derive(Copy, Clone, Debug, Default, PartialEq, Serialize)]
pub struct MeshMemory {
    /// Distinct meshes held, each `Arc` once.
    pub count: usize,
    /// Bytes of those meshes.
    pub bytes: u64,
    /// Meshes of the shown selection.
    pub shown: usize,
    /// Bytes of the shown selection's meshes.
    pub shown_bytes: u64,
}

/// The memory budget (`crate::budget`) and what eviction against it did.
#[derive(Copy, Clone, Debug, Default, PartialEq, Serialize)]
pub struct BudgetMemory {
    /// The budget, bytes.
    pub budget: u64,
    /// The bytes it counts, after the latest eviction pass.
    pub held: u64,
    /// Whether asking ahead is held.
    pub holding_ahead: bool,
    /// What the latest pass evicted.
    pub last: crate::budget::Evicted,
    /// What every pass evicted, summed.
    pub total: crate::budget::Evicted,
}

/// What the renderer holds, in bytes and counts
/// ([`crate::world::World::memory_report`]).
#[derive(Copy, Clone, Debug, Default, PartialEq, Serialize)]
pub struct MemoryReport {
    /// The allocator's counters.
    pub allocator: AllocatorBytes,
    /// Surface tiles.
    pub surfaces: SurfaceMemory,
    /// Cells.
    pub cells: CellMemory,
    /// Extracted meshes.
    pub meshes: MeshMemory,
    /// The memory budget.
    pub budget: BudgetMemory,
    /// Decode timings.
    pub timings: Timings,
}

/// Bytes as megabytes (10^6 bytes), for the overlay line.
pub fn megabytes(bytes: u64) -> f64 {
    bytes as f64 / 1e6
}

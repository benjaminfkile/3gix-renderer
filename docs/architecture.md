# Architecture

The client applies the laws to compiled matter and draws whatever is near
the camera. It is one program with two parts that share nothing but section
bytes: the worker, which runs compiler modules, and the renderer, which
draws. The specification is `space-model.md` (v2), `client-compute.md`, and
`matter-format.md` in the hub repository's `docs/architecture/`.

## The one rule

The renderer may know physics, never objects.

No module, identifier, string, comment, or file in this repository names
anything in the universe. A frame is an integer: "frame 7". The enforcement is
the core library's vocabulary lint, `scripts/vocab-lint.sh` with
`scripts/vocab-banned.txt` and `scripts/vocab-numbers.txt`, copied here
unchanged and run by `scripts/ci.sh` (with its own test,
`scripts/vocab-lint-test.sh`). A banned word anywhere in a tracked file or
file name, or a banned number (one body's size, mass, spin, orbit, day, or
temperature) in a tracked file, fails the build. Test fixtures use made-up
values. The lint's one exception, a file under `tests/` whose first comment
block carries a `measured-reference:` line, is for a test of a law against a
real measurement; no renderer test uses it.

## Module map

All engine code is in the library crate `renderer` (`crates/renderer`), so the
desktop binary, the headless tests, and the browser build share it. The
worker is its own crate, `gx-worker` (`crates/worker`), so its boundary is a
crate boundary: it only ever sees modules, requests, tile bytes, and section
bytes, and depends on `gx-core` (and natively `gx-module-host`) only.

| `gx-worker` module | What it does | Spec |
|---|---|---|
| `job` | The job (`missing` or `local` direction, with its layer and whether it is shared), the address check, blob hash checks, manifest and tile resolution, section validation; shared by both platforms | `client-compute.md` 1, 3, 4, 5.2 |
| `resolve` | Input resolution, written once as async code over a small platform trait: `gx_inputs`, then memory, cache, directions, stream or compute per input, one flight per address, `X-Inputs-Sha256`, computed `missing` inputs offered; a blocking executor for the native pool | `client-compute.md` 2.8, 10.3, 10.4 |
| `native` | The pool of worker threads running modules with `gx-module-host` (default: cores minus two, at least one), fetching by hash with single flight per name, caching, handing sections over, then offering `missing` sections to the upload lane | `client-compute.md` 1, 4 |
| `queue` | The work order: the three lanes (computes, uploads, checks) and the compute queue, highest priority first, reprioritized and pruned by the wanted set | `client-compute.md` 2.8 |
| `upload` | The upload lane: bounded at 64, oldest dropped when full, one try per offer, and the outcome of each status (`201` stored, `200` matched, `202` disputed) | `client-compute.md` 2.5, 2.8 |
| `check` | The check lane: bounded at 32, a new check skipped when full, and the outcome of each status (`200` match, `202` disputed) | `client-compute.md` 2.6, 2.8 |
| `cache` | The native local cache directory, keyed by kind and hash or address, evicted least recently used by size (default 20 GB) | `client-compute.md` 6 |
| `stats` | Counters for the overlay: modules loaded, module runs, sections computed, uploads pending, offered, stored, matched, disputed, dropped, failed; checks queued, run, skipped, matched, disputed, failed | |

`renderer` modules:

| Module | What it does | Spec |
|---|---|---|
| `config` | Environment, `.env`, and command line flags, flags winning | |
| `protocol` | The hub protocol without I/O: clock and its sync, builds and layer configs, directions, upload headers, statuses, URLs, the frame system from the union of the layers' registries | `client-compute.md` 2.2 to 2.5, `matter-format.md` 5 and 6 |
| `hub` | The native hub client v2: clock, active build, build, directions, edge reads (sections by address and `sha256`), uploads with the section 2.5 headers | `client-compute.md` 2.3 to 2.5 |
| `directions` | The client loop as a state machine without I/O: batches of at most 512 keys, deduplication in flight, `ready`/`missing`/`local`/`empty` per layer, retries with backoff | `client-compute.md` 1, 2.4 |
| `mock_hub` | A small v2 hub for tests and `gx-mock-hub` | `client-compute.md` 2 |
| `sim` | `SimClock` between the origin and now, the relativistic integration of every frame with checkpoints (a coarse history every 65 536 simulated seconds, a fine one around the simulation time for light time) | `space-model.md` 6.1, 6.2, 7 |
| `seen` | Every frame at its retarded state for the camera (position and orientation), the camera re-expressed among them, the emitter shift per lit frame, the frame nearest the crosshair | `space-model.md` 6.4 |
| `camera` | Free camera that floats in root coordinates (or follows a frame), parented to its nearest frame for precision, re-parenting | `space-model.md` 5 |
| `stream` | Cell selection (a target that asks up to three depths ahead from arrived occupancy, and the drawn selection pruned from it, `rendering.md` 3), the cell cache keyed by hash map, decode and composite, the depth transition rule, the native fetcher driving `directions`, the local cache, and the worker pool | `space-model.md` 2, 5, 7, 8; `matter-format.md` 3.5, 6 |
| `extract` | The inputs of one drawn cell's Transvoxel extraction, the surface mesh, the extraction worker pool | `rendering.md` 2, 3.3 |
| `volume` | Gas and plasma as 3D grids, the ray march and its CPU reference | `space-model.md` 2, `matter-format.md` 3.3 |
| `farfield` | Frames too small on screen drawn as point sprites, the transition | `space-model.md` 1 and 2, `matter-format.md` 3.3 |
| `light` | Emitters from the light escaping each frame's depth-0 cell, point lights, the emission lookup table | `matter-format.md` 3.3 |
| `world` | Ties stream, extract, volume, light, and farfield together for one frame | |
| `render::scene` | Camera-relative `f32` positions, surface placement, projection, depth strategy | `space-model.md` 5 |
| `render::overlay` | The overlay text | |
| `render::gpu` | wgpu pipelines: surface, volume, and sprite passes, exposure, display pass | |
| `render::headless` | Offscreen RGBA8 target, readback of it and of the float radiance, PNG | |
| `controls` | Keyboard and mouse bindings for the window and the canvas | |
| `api` | The renderer API (`docs/api.md`): command queue, state snapshot, events, the `Consumer` trait | |
| `app` | Desktop window loop and the headless run; the window is a consumer of `api`, driving each tick through it | |
| `memory` | The memory report, `memory-report` feature only: counting allocator, decode timings, the report types | |
| `redraw` | Whether a scene differs visibly from the last one drawn | `rendering.md` 4 |
| `web` | The browser build: canvas loop, `fetch` and `WebSocket` hub client, the Cache API cache, the Web Worker driver, the browser fetcher, the `Renderer` class (the web binding of `docs/api.md`) | `space-model.md` 2 and 10, `client-compute.md` 4 and 6 |

Shaders: `src/shaders/surface.wgsl` (lit surfaces), `src/shaders/volume.wgsl`
(the ray march), `src/shaders/sprite.wgsl` (far field sprites),
`src/shaders/exposure.wgsl` (luminance, adaptation, tone curve),
`src/render/markers.wgsl` (markers and lines).

Binaries: `gx-renderer` (`src/main.rs`) and `gx-mock-hub`
(`src/bin/gx-mock-hub.rs`).

The workspace also holds `tools/png-stats`, a small crate with no renderer
code that measures screenshots (bright blobs, a lit disc and its two sides,
the centroid of bright pixels, the steepest step between neighboring
pixels) for the end to end run, `scripts/e2e.sh` (`docs/e2e/README.md`).

## The headless run

`app::render_headless_view` draws the simulation at its time (entered at
now, or wherever `--start` and `--start-offset-seconds` put it, clamped
between the origin and now), places the camera for the run's view (`Home`, or the view of a number key, at a factor
of its distance), selects once, and streams until every selected and pinned
cell is resolved. With `--wait-ready-seconds` it waits for every one to be
drawable (`World::ready`: `Ready` with its mesh, or `Empty`; a `Gone` cell
is not ready) and the binary exits non-zero after writing the image when
the time runs out; without it the run renders whatever has settled within
60 s. `--stats-json` writes the overlay statistics of the rendered frame
(`app::HeadlessStats`), including the request totals of the fetcher
(`stream::FetchTally`): cells requested, cells decoded, bytes, and the
round trips of the first and the latest cell; and the counts of
`directions` and the worker: directions requests, sections streamed from
the edge and read from the local cache, sections computed here, module
runs, modules loaded, and uploads pending, offered, stored, matched,
disputed, dropped, and failed, and checks queued, run, skipped, matched,
and disputed. `--no-overlay`
leaves out the text, the frame markers, and the lines, so only matter is
in the image.

`hub`, `mock_hub`, `app`, and `render::headless` are native only. The library
builds for `wasm32-unknown-unknown` without them, with the `web` feature off
or on; with it on, `web` adds the canvas loop and the browser fetch path
(`docs/web.md`). The binaries are empty stubs on `wasm32`, so the whole
package builds for that target.

## The frame loop

The loop ticks: steps 1 to 5 below run about once per display frame while
the camera flies, matter streams, or the exposure settles, and ten times a
second otherwise. A tick builds the scene and draws it only when it differs
visibly from the last one drawn (`redraw`, `rendering.md` section 4).

1. The `SimClock` takes the hub's now (`protocol::ClockSync`, measured at
   start and every minute) as universe time, seconds since the origin
   (`gx_core::clock::universe_seconds` from the hub's `originUtcMs`, or
   from the hub's now at start while the hub has no origin yet; the
   registry epoch is 0, the origin), and advances by the wall-clock delta
   times the rate, clamped between the origin and now; at now it follows
   now. Instants leave the renderer (`getTime`, the HUD, directions) as
   wall-clock milliseconds through `gx_core::clock::wall_ms`.
2. Every frame is integrated to the clock's time with
   `gx_core::integrate::Scheme::Relativistic` (first post-Newtonian gravity,
   oblateness, rotation with torques) in steps of at most 600 s and at most
   1/256 of the shortest dynamical time `sqrt(d^3 / (G (m1 + m2)))` of two
   frames (`sim::max_step`), read from a
   fine `gx_core::history::History` that is rebuilt from a coarse one
   (checkpoints every 65 536 simulated seconds from the origin) whenever the time
   leaves it. A jump anywhere costs at most one coarse interval.
3. The camera moves (`camera::advance`): it keeps its place and view
   direction in root coordinates through the integration, free flight
   moves it, and if `FrameSystem::nearest_frame` of its position differs
   from its frame, it re-parents.
4. Light time: `seen::Seen` solves the light-time equation of every frame
   for the camera's position (for a frame with drawn meshes, over the
   distance to the nearest of them, so a camera at a large body's surface
   sees the surface with its own light time, not its center's) against the
   fine history
   (`gx_core::light::retarded_state`) and builds the frame system of
   retarded states, with the camera at the same place in space among them.
   Everything below reads it.
5. Matter streams: selection (on arrivals, camera moves, and every 100 ms; natively on its own thread, `selector`), requests, directions at the
   simulation time, streamed and computed sections,
   finished meshes, eviction. See "From selection to pixels" below.
6. The scene is built camera-relative with the drawn cells and their lights
   and drawn with the overlay. Each surface draw carries its own light
   positions: where each emitter was one light-travel time before that
   draw's frame is seen (`gx_core::light::retarded_emitter`).

## `f64` versus `f32`

Everything about the world is `f64` and lives in `gx-core` types
(`Vec3`, `Quat`, `Seconds`, `Meters`): frame states, the integrator, the
camera position and orientation, and every transform.

`f32` appears only in data handed to the GPU, built in `render::scene`:
marker and line positions after the camera position has been subtracted with
`FrameSystem::relative`, the rotation from root axes into camera axes, and the
projection. `glam` is used for that GPU-side math only.

## Where the floating origin lives

The camera is a frame id, a position in that frame (relative to its origin,
in its axes), and an orientation relative to the frame axes. Because the
camera lives in its frame's axes it turns with a frame that spins, like any
point addressed in that frame.

Every drawn position is `FrameSystem::relative(point, frame, camera.position,
camera.frame_id)`: the vector from the camera to the point, computed by
carrying both up the frame tree only to their nearest common ancestor, so the
large offsets above it never enter the arithmetic. The result is small near
the camera and is cast to `f32` only then. Nothing on the GPU has a
translation.

Re-parenting converts the position with `FrameSystem::relative` and the
orientation with the frame orientations, so the camera's place and view
direction in space are unchanged. The tests check that the root-space point
agrees to within 1e-6 m and that the camera-relative positions and view
rotation handed to the GPU do not change.

## Depth strategy

Reversed-z with an infinite far plane in `f32` depth (`Depth32Float`). A point
at view distance `z` gets depth `near / z`: 1 at the near plane, falling
toward 0 at infinity. The depth buffer clears to 0 and the test keeps the
greater value. Reversed-z puts the dense end of the float range at the far
distances, which is where a perspective depth needs it, and the infinite far
plane removes the far plane choice. The near plane is chosen per frame as half
the distance to the nearest marker in front of the camera or to the nearest
drawn mesh's bounds, whichever is closer (at least 1 mm).

## Memory budget

One budget in bytes (`src/budget.rs`) covers everything the renderer
caches: built surface tiles (section bytes, summary, geometry material and
shape, composited material), compact surface tiles with their overlays,
cached cells (each ready cell's decoded sections, counted by
`stream::section_heap_bytes`), and extracted meshes
(`SurfaceMesh::heap_bytes`). These byte counts are always compiled; only
the counting allocator is behind the `memory-report` feature.

The budget is chosen once at start by the platform layer
(`budget::platform_budget`); no API option sets it:

| Platform | Budget | Why |
|---|---|---|
| native | 2 GiB | memory to spare, and freed memory goes back to the system |
| browser | 512 MiB | WebAssembly memory never shrinks: the highest point while loading is what the page keeps |
| browser, `matchMedia("(pointer: coarse)")` matches and `navigator.deviceMemory` is 8 or more | 384 MiB | large phones and tablets keep a tab this size, with room to move |
| browser, `matchMedia("(pointer: coarse)")` matches and `navigator.deviceMemory` is under 8 or not reported (Safari), or `navigator.deviceMemory` is at most 4 | 192 MiB | other phones and small tablets end tabs that hold much more |

Eviction runs where it ran before: after each surface selection and every
`EVICT_INTERVAL_SECONDS` for cells (`World::enforce_budget`); the pass
after a surface selection takes the cells' and meshes' bytes from the
latest cell pass instead of counting every cell again. Eviction keeps
to a limit (`budget::Detail::limit`): the budget while the camera moves,
and 70 percent of it (`budget::SETTLE_FRACTION`) once the camera has
been still for 2 s (`budget::DETAIL_WAIT_SECONDS`), so a still view
leaves about 30 percent free for the tiles a pan or a zoom brings in.
Under the limit nothing is evicted for size (cells still age out after
60 s). While over it, in this order:

1. Built tiles not level-0 and not wanted built (drawn tiles are) go
   compact, least recently looked at first; the ancestors the selection
   walks need only their summaries.
2. Compact tiles, cells, and meshes not drawn and not in the current
   target go, least recently used first, the deepest level first on a tie
   (then by key, so the order is the same on every run).

What the current frame draws, the ancestors the selection walks to reach
it, level-0 tiles, the tiles the latest selection wants built (drawn,
children a split waits on, split tiles about to join) or asked for, the
cell target, and pinned cells are never evicted, even over the limit.
The rest of the selection's `used` set protects nothing past the budget.

When what the view wants at full detail does not fit, detail yields to
the budget instead (`budget::Detail`): after a surface pass that leaves
held bytes over the limit, once the previous step has taken effect (no
build or request outstanding) or 2 s have passed, every split distance
is scaled down one more step of 0.8 (`budget::DETAIL_STEP`, at most 16
steps), so the view draws fewer, coarser tiles; joins wait for the parent
to be built as always, so the view is never open. While the camera
moves the limit is the budget, so the factor steps down only when
eviction cannot get under the budget without dropping what the
selection wants. The factor goes up a step only at a camera still for
2 s, once the held bytes expected after the step (held bytes over
`DETAIL_STEP^2`) have stayed under `SETTLE_FRACTION` of the budget for
2 s, never while the camera moves, and never back to a step found over
the limit since the camera, the view size, or the budget last changed,
so the factor does not swing. Under `SETTLE_FRACTION` of the budget the
factor stays 1 and the drawn tiles do not change.

While the camera moves, built tiles that are no longer wanted built stay
built past `tiles::BUILT_GRACE_SECONDS` (`Surfaces::set_keep_built`): the
free room holds what the view just left, so a pan back draws them at
once. Over the budget eviction compacts them first; once the camera has
been still for 2 s they go compact as before.

While held bytes exceed the limit, the selection asks only for what it
needs at the current view: children of tiles within their split distance
(not out to `PREFETCH` times it) and cells of the drawn target (no cell
whose parent has not arrived, the `AHEAD_DEPTHS` prediction). Asking ahead
resumes once held bytes fall below 90 percent of the limit
(`budget::RESUME_FRACTION`). This keeps the highest point near the budget.

A compact tile is built off the frame thread (`tiles::build_tile`, on a
decode thread natively, in the fetcher's slice in the browser), and
composited there too for the hidden layers of the request, so
`Surfaces::complete_build` only stores the result; it composites again
only when the hidden layers or the tile's overlays changed meanwhile. A
tile that arrives not wanted built is not composited.

## Memory report

A developer tool for measuring what the renderer holds (`src/memory.rs`).
It is the cargo feature `memory-report`, off by default; without it none
of it is compiled, the allocator is the system's, and the release wasm is
unchanged. With it:

- A counting global allocator wraps `std::alloc::System`: live and peak
  bytes in atomics, updated in `alloc`, `alloc_zeroed`, `dealloc`, and
  `realloc`.
- `World::memory_report()` returns `MemoryReport` (serde), in bytes and
  counts. Bytes are `Vec` capacity times element size plus the struct
  itself, counted by each type (`TileGeometry::heap_bytes` for a geometry's
  material, `TileShape::heap_bytes` for its shape,
  `SurfaceMesh::heap_bytes`, `tiles::overlay_heap_bytes`,
  `stream::section_heap_bytes`, which counts samples by length since the
  core library keeps their arrays private). Data shared through an `Arc`
  is counted once, by pointer.
  - `allocator`: `live` and `peak`.
  - `budget`: the `budget`, the bytes it counts (`held`), whether asking
    ahead is held, the `detail` factor the surface selection scales split
    distances by (1 unless the view's full detail did not fit), whether
    the camera counts as `moving` for it (the limit is the budget while
    it moves, `SETTLE_FRACTION` of it while still), and what
    the latest pass and all passes evicted
    (`compacted`, `tiles`, `cells`, `meshes`).
  - `surfaces`: entries `requested`, `ready` (built or compact),
    `empty`, `gone`; `compact` and `compact_bytes` (section bytes,
    summaries, overlays), `built` and `built_bytes` (the same plus their
    geometries), and the built tiles by reason: `built_drawn`,
    `built_for_split` (children a drawn tile's split waits for),
    `built_for_join` (split tiles about to be drawn again),
    `built_fallback` (ancestors of drawn tiles, kept built so a child
    that cannot cover its part falls back to its nearest built ancestor
    for its own part),
    `built_level_0`, and `built_in_grace` (within the grace time); the
    selection's `used` set, `drawn` count, and `drawn_quarters` (drawn
    entries that are one quarter of a tile); `geometries` and
    `geometry_bytes`, every distinct `TileGeometry` held (tile states,
    bases, the drawn list): each one's material plus each distinct
    `TileShape` once (`shapes`, `shape_bytes`), since a base and its
    composites share one shape `Arc`; of which `composited` copies
    (geometries that are no tile's base) and their material bytes;
    `drawn_geometry_bytes`, what the drawn tiles need (their drawn
    geometry and their bases, each shape once);
    `overlay_bytes` and `drawn_overlay_bytes` for overlay tiles.
  - `cells`: `cached`, `ready`, `selected`; `sample_bytes` of every
    distinct decoded section held (composites and layers) and
    `selected_sample_bytes` for the drawn selection.
  - `meshes`: `count` and `bytes` (positions, vertices, indices by
    capacity) of every held mesh, `shown` and `shown_bytes` for the shown
    selection.
  - `timings`: `count` and `mean_us` of `TileGeometry::new`, of
    `decode_tile` (which includes its `TileGeometry::new` calls), and of a
    cell section's decode, on the session clock (`performance.now()` in
    the browser, which some browsers coarsen).
- Native: `--stats-json` gains a `memory` object, the report when the
  headless frame was drawn, and the overlay gains the line
  `memory live <MB> MB  tile geometry <MB> MB` (megabytes of 10^6 bytes).
  Build with `cargo build --release --features memory-report`.
- Browser: `sh web/build.sh --memory-report`; the loop logs
  `console.log("gx-memory", <json>)` every 5 s. `scripts/memory-pan.mjs`
  pans a page in headless Chrome and prints the WebAssembly memory sizes
  and the latest report every 30 s (`docs/web.md`, "Measuring memory").

`scripts/ci.sh` runs clippy and the renderer's tests with the feature, so
it builds on both platforms.

## Overlay text

The overlay uses `wgpu_text` (a thin wrapper over `glyph_brush`) with the
Hack typeface embedded through `epaint_default_fonts`. `wgpu_text` takes font
bytes directly and does no system font discovery, so every machine draws the
same glyphs and headless images are reproducible. `glyphon` would bring
`cosmic-text` with system font lookup and shaping, which this overlay does not
need.

## Hub client and the client loop

At start (`hub::HubClient::connect`, and `web::hub::WebHub::open` in the
browser) the client reads `GET /space/{s}/clock`, `GET
/space/{s}/build/active` unless `GX_BUILD_ID` names a build, and `GET
/space/{s}/build/{b}`. The layer configs drive the depth limit of each
frame: the deepest of the registry's `max_depth` and every attached
layer's `maxDepth` (`protocol::frame_system_from_registries`).

Every key, the registry included, goes through `directions::Coordinator`:

1. Keys are asked about in `POST .../directions` batches of at most 512
   keys, at most two batches in flight; a key already in flight is not
   asked about again. The time asked for is the drawn moment, clamped to
   the hub's now and the space's origin.
2. Per layer: `empty` contributes nothing. `ready` carries the stored
   bytes' `sha256`: it is streamed from `GET
   /section/{address}?sha256={hex}` (through the edge when `GX_EDGE_URL`
   is set), or read from the local cache; the bytes are checked against
   the hash (a cached copy of another version is discarded and fetched
   again), validated, then cached. `missing` and `local` go to the worker.
   There is no readiness socket: nothing waits on another client.
3. A failed directions request, a failed stream (bytes of another hash
   included), a `404` for a stored
   address, or a transient worker failure is retried after a backoff of
   0.25 s doubling to at most 8 s. A permanent worker failure (the module
   failed, or its output fails validation) fails the key, which becomes
   `Gone`.
4. Once every layer has its section, the key completes with the sections
   in the build's layer order and is composited (`matter-format.md` 3.5).

The registry is just another key. The renderer waits for the union of all
layers' registries before starting the integrator; the registry load gives
up after 300 s natively and 120 s in the browser with a clear error.

The worker (`gx-worker`), for each `missing` or `local` direction: checks that the
address is the address of the job's inputs (`client-compute.md` 3), fetches
the module by hash (local cache first, every byte checked against its
hash), runs `gx_sources`, resolves the ids through the layer's manifest
(fetched by hash), fetches and checks each tile, runs `gx_compile`,
validates the section with the library validator, hands it to the renderer
at once, and caches it. A `missing` section then enters the upload lane
(`client-compute.md` 2.5 and 2.8): at most 64 offers, the oldest dropped
when full, each sent once with `PUT .../section/{address}` and `X-Layer`,
`X-Key`, `X-Span-Index`, `X-Content-Sha256` (and natively `Expect:
100-continue`). A failure is counted and never retried, and uploads never
delay a result reaching the renderer. A `local` section is never offered.
Natively the modules run on a pool of threads with `gx-module-host`
(1 GiB and 60 s per call); in the browser in a dedicated Web Worker with
the browser's `WebAssembly` (`docs/web.md`).

Work order (`client-compute.md` 2.8, `gx_worker::queue`): three lanes,
strictly ordered: wanted computes, then uploads, then checks. A lower lane
starts work only when no higher lane has any waiting, and a free thread
never waits while a compute does. Computes start highest priority first
(the key's projected size on screen), with one rule before it
(`rendering.md` 7.6, `directions::drawn_first`): a surface tile already
drawn and waiting only on overlays (it has had a partial completion) goes
before every key with nothing drawn, from that completion on and through
every later change of priorities; within each group the order stays the
projected size. On every new wanted set, and on every partial
completion, the waiting
ones take their key's new priority, and those whose key left the set are
removed before they start and reported as dropped, so the coordinator
takes them up again from the kept answer if the key comes back. A
running job finishes and its section is cached. Uploads are network I/O
and never take a compute thread or wait for computes: natively the
fetcher sends each offer on its tokio runtime as soon as it enters the
lane, in the browser as an async `fetch` on the page's event loop (the
module work runs in a Web Worker, which an upload never occupies). At
most four are in flight (`gx_worker::MAX_UPLOADS_IN_FLIGHT`); while four
are in flight the lane keeps the rest (bounded at 64, oldest dropped)
and a finished request sends the next. Uploads used to wait until no
compute waited; a burst of computes then overflowed the lane and most
offers were dropped, so the hub never filled.

Checks (`client-compute.md` 2.6 and 2.8, `gx_worker::check`): when a
`ready` section of a cell or a surface tile arrives from the edge or the
local cache, the coordinator chooses it for a check with probability
`checkRate`, from its own random source seeded once per run (never from
the address; it never affects bytes). The job is built without another
request: the module and manifest from the build's layer config, the span
index by the rule of 2.4 from the time the key was asked about
(`protocol::check_job`). The check lane holds at most 32; a new check
into a full lane is skipped. Checks run only when no compute and no
upload waits: natively the recompute runs on a pool thread and the
`POST .../section/{address}/check` (body `{"sha256"}`, headers `X-Layer`,
`X-Key`, `X-Span-Index`) on the tokio runtime; in the browser the
recompute runs in the Web Worker when it is idle and nothing else waits.
When the recomputed hash differs from the drawn bytes, the computed
section takes their place for that address and hash
(`directions::Coordinator::replace`): the key completes again with it,
whatever the hub answers. Layers with source epochs are not checked: the
renderer has no epoch index reader, so each such check is counted as
skipped.

The local cache (`client-compute.md` 6) holds modules, manifests, tiles,
and sections keyed by kind and hash or address. Natively it is a
directory, by default `gx-renderer` under the platform cache directory,
moved with `--cache-dir`, limited with `--cache-max-bytes` (default 20 GB),
and turned off with `--no-cache`. In the browser it is the Cache API under
the same keys. Everything in it is immutable, so it is evicted by size
alone, least recently used first. A hit never touches the network.

## From selection to pixels

Each step knows physics and geometry only. A cell is a cell, a hot sample is
a light, a dense sample is a surface.

Surface tile selection (`tiles::Surfaces::select`, `rendering.md` 7.2
and 7.6) runs every frame on the frame's clock. A tile replaces its
parent only when each child has its surface tile and every shown overlay
layer has arrived for it, failed for it (an invalid section or a module
error, logged once), or is not `required` (`client-compute.md` 2.2) and
`tiles::OVERLAY_WAIT_SECONDS` (2 s) have passed since the child's surface
tile arrived. A hidden overlay is never waited for, and level-0 tiles draw
as soon as their surface tiles arrive.

An arrived tile is held built (its geometry) or compact (its surface
section bytes, its overlays, and a `tiles::TileSummary`). The selection
walks on summaries, so only drawn tiles need geometry. A tile is built
while it is drawn, while it is a level-0 tile, while it is a child (in
the view cone, not below the horizon) of a drawn tile within its split
distance, or while it is split, beyond its split distance, and about to
be drawn again; split ancestors the walk passes through are not; a
compact tile wanted built is built off the
frame thread (`tiles::Surfaces::take_builds`, on the native fetcher's
decode threads, in the browser fetcher's slice of at most
`web::fetch::BUILD_SLICE_SECONDS`, largest projected size first). A tile
splits only when the tiles its children will draw are built; a split
tile beyond its split distance and not built keeps drawing its children
until it is built, then joins. A built tile not
wanted built for `tiles::BUILT_GRACE_SECONDS` (2 s) goes back to compact.

1. **Selection** (`stream::select_all`, run by `selector` on its own thread natively and inline in the browser, when results arrive, when the camera or view changes, and every 100 ms). Pixel error, view cone, and refinability from occupancy only; no occlusion culling and no shape model (`rendering.md` 1.1 and 3.1). It builds the target (asking up to three depths ahead of what has arrived, balanced 2:1) and the drawn selection (only arrived cells, split only when every child and its same-depth neighbors have arrived). The `stream` module docs have the details; `rendering.md` is the contract.
2. **The cell cache** (`stream::CellCache`), keyed by
   `(frame_id, depth, x, y, z)`:

   | State | Meaning |
   |---|---|
   | `Missing` | Selected, not requested yet |
   | `Requested` | The client loop is resolving it (directions, streams, or a module run); at most 512 at once, largest projected size first |
   | `Ready(cell)` | Every layer's section, decoded and composited |
   | `Empty` | Every layer is `empty`, or the composite holds no matter |
   | `Gone` | Sections that fail validation, or a module that failed |

   A request stays `Requested` until every layer has its section; the
   client loop retries on its own. Cells not
   selected (or drawn as a stand-in) for 60 s are evicted; past that, the
   memory budget evicts (see "Memory budget"). Cells of the target,
   pinned cells, and requests in flight are never evicted.
3. **Decode and composite** (`stream::decode_cell`).
   `gx_core::matter::decode` validates every layer's section; each
   section's `cell_origin` and `cell_edge` must also match the registry's
   geometry for the key (`matter-format.md` 3.1); then
   `gx_core::matter::composite` combines the layers. The composited
   `Section` is kept together with its emitter if it is a depth-0 cell
   (step 5).
4. **Extraction** (`extract`) runs on a pool of `std::thread` workers fed by
   a channel: every drawn cell is extracted with the Transvoxel algorithm
   (`transvoxel`, `rendering.md` section 2) at the level set
   `occupancy = 0.5` of its solid and fluid samples. Gas and plasma samples
   count as occupancy 0 here. Its inputs (`extract::CellInputs`) are the
   drawn selection and the sections, one depth coarser, at its depth, and
   one depth finer, in the region of the cell grown by one cell of its
   depth on every side: same-depth neighbors give the values on shared
   faces, a face with a neighbor drawn one depth finer gets transition
   cells, and a face with a neighbor drawn one depth coarser is owned by
   this, the finer, side. A cell is extracted only from inputs that are
   present or known empty; an extraction that reads anything else gives no
   mesh and waits. `World` extracts a cell again only when its inputs
   change, found from arrivals and selection changes; the previous mesh is
   drawn until the new one is done. The browser build, with no threads,
   extracts inline. Extraction time per cell (median and 95th percentile)
   is reported in `--stats-json` (`extraction_ms_p50`,
   `extraction_ms_p95`) and by the budget test.
   **Volumes** (`volume`): the gas and plasma samples of the same section
   become a `VolumeGrid` at decode time, drawn by ray marching
   (`docs/shading.md`). A cell whose densest sample is gas or plasma is
   therefore a volume, and a cell with both kinds draws a mesh and a
   volume.
5. **Lights** (`light`). Every frame with mass has its depth-0 cell pinned
   (fetched once, never evicted). That cell holds all of the frame's
   matter, so its boundary is the frame's boundary with space, and the
   light escaping it is the light the frame sends out:
   `gx_core::emission::summarize(section, 0 K)`, radiative transfer
   through the section (opaque solids and fluids radiate from their exposed
   faces, gas and plasma by their extinction) at each sample's radiating
   temperature (`matter-format.md` 3.3). Finer cells give no lights: their
   boundaries face other matter of the same frame. There is no temperature
   cutoff: an emitter is a light when its light, at its own matter's
   radius, reaches the faintest radiance the exposure pass shows
   (`light::MIN_LIGHT_RADIANCE`). The eight strongest
   frame emitters become point lights of radiant intensity
   `band_power / (4 pi)` per band.
6. **Draw set** (`stream::CellCache::draw_set`), with the depth transition
   rule below, and the **far field** (`farfield`): a frame with mass whose
   matter (the ball of the volume its depth-0 cell fills; before that cell
   arrives, the ball around its cube) projects to under 2 px has no cells selected
   and is a point sprite instead; between 2 and 8 px the sprite fades out
   as the cells fade in. The sprite's brightness comes from the frame's
   pinned depth-0 cell (`docs/shading.md`).
7. **Placement** (`render::scene::add_matter`). A mesh is uploaded once
   with positions relative to its cell origin in `f32`. Per frame, the cell
   origin is made relative to the camera in `f64` with
   `FrameSystem::relative` and cast to `f32`; that offset and the frame's
   rotation are the draw's model transform. Volumes are placed the same way
   and sorted farthest first; light and sprite positions go through the
   same `f64` camera-relative step. The near plane also comes in to half the
   distance to the nearest mesh bounds or gas box.
8. **Shading and exposure** (`render::gpu`, `docs/shading.md`): the lit
   surface pass into a float target, the volume pass, the sprite pass,
   the exposure (automatic, or fixed with `--exposure-stops`), the tone
   curve, then markers, lines, and the overlay.

## The depth transition rule

No holes, ever. For every selected cell:

- if it can be drawn (its mesh is ready, or it is `Empty`), it is drawn;
- else, if its descendants down to two levels are loaded and cover it, they
  are drawn (the camera moved away and the coarse cell has not arrived);
- else its nearest drawable ancestor is drawn instead.

Then any drawn cell with a drawn ancestor is dropped. So while any child of
a cell is not ready, the parent keeps drawing and none of the children do;
the moment every child is ready, the children replace the parent. A `Gone`
cell never counts as ready, so its parent stays rather than leaving a hole.

## The extraction determinism contract

The same inputs always give bit-identical vertex and index buffers, whatever
thread extracts them and however many times:

- The inputs are held in ordered maps; the lattice is walked in index order
  and vertices are shared through an ordered table keyed by position.
- Every lattice point gets one value from one rule (`rendering.md` section
  2.3), so two cells that share a point compute the same bits.
- Every value is plain `f64` arithmetic with `sqrt`, both correctly rounded
  in IEEE 754; attributes are narrowed to `f32` last.
- A mesh depends only on its inputs, so the order the pool finishes jobs in
  changes only which frame a mesh first appears in.

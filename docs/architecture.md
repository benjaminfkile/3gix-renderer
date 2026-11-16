# Architecture

The client applies the laws to compiled matter and draws whatever is near
the camera. It is one program with two parts that share nothing but section
bytes: the worker, which runs compiler modules, and the renderer, which
draws. The specification is `space-model.md` (v2), `client-compute.md`, and
`matter-format.md` in the hub repository's `docs/architecture/`.

## The one rule

The renderer may know physics, never objects.

No module, identifier, string, comment, or file in this repository names
anything in the universe. A frame is an integer: "frame 4". The enforcement is
the core library's vocabulary lint, `scripts/vocab-lint.sh` with
`scripts/vocab-banned.txt`, copied here unchanged and run by `scripts/ci.sh`.
A banned word anywhere in a tracked file or file name fails the build.

## Module map

All engine code is in the library crate `renderer` (`crates/renderer`), so the
desktop binary, the headless tests, and the browser build share it. The
worker is its own crate, `gx-worker` (`crates/worker`), so its boundary is a
crate boundary: it only ever sees modules, requests, tile bytes, and section
bytes, and depends on `gx-core` (and natively `gx-module-host`) only.

| `gx-worker` module | What it does | Spec |
|---|---|---|
| `job` | The job (`compute` direction), the address check, blob hash checks, manifest and tile resolution, section validation; shared by both platforms | `client-compute.md` 1, 3, 4, 5.2 |
| `native` | The pool of worker threads running modules with `gx-module-host` (default: cores minus two, at least one), fetching by hash with single flight per name, caching, handing sections over, then uploading | `client-compute.md` 1, 4 |
| `cache` | The native local cache directory, keyed by kind and hash or address, evicted least recently used by size (default 20 GB) | `client-compute.md` 6 |
| `stats` | Counters for the overlay: modules loaded, module runs, sections computed, uploads pending, done, failed | |

`renderer` modules:

| Module | What it does | Spec |
|---|---|---|
| `config` | Environment, `.env`, and command line flags, flags winning | |
| `protocol` | The hub protocol without I/O: clock and its sync, builds and layer configs, directions, statuses, socket frames, URLs, the frame system from the union of the layers' registries | `client-compute.md` 2.2 to 2.5, `matter-format.md` 5 and 6 |
| `hub` | The native hub client v2: clock, active build, build, directions, edge reads, uploads with `X-Claim`, the readiness socket keyed by address | `client-compute.md` 2.3 to 2.5 |
| `directions` | The client loop as a state machine without I/O: batches of at most 512 keys, deduplication in flight, `ready`/`compute`/`pending`/`empty` per layer, retries with backoff, readiness notices | `client-compute.md` 1, 2.4 |
| `mock_hub` | A small v2 hub for tests and `gx-mock-hub` | `client-compute.md` 2 |
| `sim` | `SimClock` between the origin and now, the relativistic integration of every frame with checkpoints (a coarse history every simulated day, a fine one around the simulation time for light time) | `space-model.md` 6.1, 6.2, 7 |
| `seen` | Every frame at its retarded state for the camera (position and orientation), the camera re-expressed among them, the emitter shift per lit frame, the frame nearest the crosshair | `space-model.md` 6.4 |
| `camera` | Free camera that floats in root coordinates (or follows a frame), parented to its nearest frame for precision, re-parenting | `space-model.md` 5 |
| `stream` | Cell selection (refining only cells a surface crosses), the cell cache, decode and composite, the depth transition rule, the native fetcher driving `directions`, the local cache, and the worker pool | `space-model.md` 2, 5, 7, 8; `matter-format.md` 3.5, 6 |
| `extract` | Marching cubes over the solid and fluid samples of a composited section, the extraction worker pool | `space-model.md` 2 |
| `volume` | Gas and plasma as 3D grids, the ray march and its CPU reference | `space-model.md` 2, `matter-format.md` 3.3 |
| `farfield` | Frames too small on screen drawn as point sprites, the transition | `space-model.md` 1 and 2, `matter-format.md` 3.3 |
| `light` | Emitters from the light escaping each frame's depth-0 cell, point lights, the emission lookup table | `matter-format.md` 3.3 |
| `world` | Ties stream, extract, volume, light, and farfield together for one frame | |
| `render::scene` | Camera-relative `f32` positions, surface placement, projection, depth strategy | `space-model.md` 5 |
| `render::overlay` | The overlay text | |
| `render::gpu` | wgpu pipelines: surface, volume, and sprite passes, exposure, display pass | |
| `render::headless` | Offscreen RGBA8 target, readback of it and of the float radiance, PNG | |
| `controls` | Keyboard and mouse bindings for the window and the canvas | |
| `app` | Desktop window loop and the headless run | |
| `web` | The browser build: canvas loop, `fetch` and `WebSocket` hub client, the Cache API cache, the Web Worker driver, the browser fetcher, `start` | `space-model.md` 2 and 10, `client-compute.md` 4 and 6 |

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
runs, modules loaded, and uploads pending, done, and failed. `--no-overlay`
leaves out the text, the frame markers, and the lines, so only matter is
in the image.

`hub`, `mock_hub`, `app`, and `render::headless` are native only. The library
builds for `wasm32-unknown-unknown` without them, with the `web` feature off
or on; with it on, `web` adds the canvas loop and the browser fetch path
(`docs/web.md`). The binaries are empty stubs on `wasm32`, so the whole
package builds for that target.

## The frame loop

1. The `SimClock` takes the hub's now (`protocol::ClockSync`, measured at
   start and every minute) and advances by the wall-clock delta times the
   rate, clamped between the origin and now; at now it follows now.
2. Every frame is integrated to the clock's time with
   `gx_core::integrate::Scheme::Relativistic` (first post-Newtonian gravity,
   oblateness, rotation with torques) in steps of at most 600 s, read from a
   fine `gx_core::history::History` that is rebuilt from a coarse one
   (checkpoints every simulated day from the origin) whenever the time
   leaves it. A jump anywhere costs at most one coarse interval.
3. The camera moves (`camera::advance`): it keeps its place and view
   direction in root coordinates through the integration, free flight
   moves it, and if `FrameSystem::nearest_frame` of its position differs
   from its frame, it re-parents.
4. Light time: `seen::Seen` solves the light-time equation of every frame
   for the camera's position against the fine history
   (`gx_core::light::retarded_state`) and builds the frame system of
   retarded states, with the camera at the same place in space among them.
   Everything below reads it.
5. Matter streams: selection (every 100 ms), requests, directions at the
   simulation time, streamed and computed sections, readiness notices,
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
2. Per layer: `empty` contributes nothing. `ready` is streamed from `GET
   /section/{address}` (through the edge when `GX_EDGE_URL` is set), or
   read from the local cache; the bytes are validated, then cached.
   `compute` goes to the worker. `pending` is subscribed to on the
   readiness socket (`WS .../sections/ready`, `{"subscribe": address}`
   out, `{"sectionReady": address}` in) and asked about again after the
   hub's `retryMs`, or streamed as soon as its notice arrives.
3. A failed directions request, a failed stream, a `404` for a stored
   address, or a transient worker failure is retried after a backoff of
   0.25 s doubling to at most 8 s. A permanent worker failure (the module
   failed, or its output fails validation) fails the key, which becomes
   `Gone`.
4. Once every layer has its section, the key completes with the sections
   in the build's layer order and is composited (`matter-format.md` 3.5).

The registry is just another key. The renderer waits for the union of all
layers' registries before starting the integrator; the registry load gives
up after 300 s natively and 120 s in the browser with a clear error.

The worker (`gx-worker`), for each `compute` direction: checks that the
address is the address of the job's inputs (`client-compute.md` 3), fetches
the module by hash (local cache first, every byte checked against its
hash), runs `gx_sources`, resolves the ids through the layer's manifest
(fetched by hash), fetches and checks each tile, runs `gx_compile`,
validates the section with the library validator, hands it to the renderer
at once, caches it, and uploads it with `PUT .../section/{address}` and
`X-Claim`. Upload failures are logged and counted and never block drawing.
Natively the modules run on a pool of threads with `gx-module-host`
(1 GiB and 60 s per call); in the browser in a dedicated Web Worker with
the browser's `WebAssembly` (`docs/web.md`).

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

1. **Selection** (`stream::select_all`, every 100 ms). A frame takes part
   when its `root_extent / 8` region is inside the view cone or within
   `4 * root_extent` of the camera. For each such frame,
   `gx_core::lod::select_cells` runs with the camera in the frame's
   coordinates, a pixel error of 4, the viewport height, the vertical field
   of view, and at most 512 cells. Each selected cell keeps its projected
   size in pixels as its request priority.
2. **The cell cache** (`stream::CellCache`), keyed by
   `(frame_id, depth, x, y, z)`:

   | State | Meaning |
   |---|---|
   | `Missing` | Selected, not requested yet |
   | `Requested` | The client loop is resolving it (directions, streams, or a module run); at most 512 at once, largest projected size first |
   | `Ready(cell)` | Every layer's section, decoded and composited |
   | `Empty` | Every layer is `empty`, or the composite holds no matter |
   | `Gone` | Sections that fail validation, or a module that failed |

   A request stays `Requested` however long the hub keeps it `pending`;
   the client loop retries on its own. Cells not
   selected (or drawn as a stand-in) for 60 s are evicted, and beyond 4096
   cached cells the least recently used go first. Selected cells and
   requests in flight are never evicted.
3. **Decode and composite** (`stream::decode_cell`).
   `gx_core::matter::decode` validates every layer's section; each
   section's `cell_origin` and `cell_edge` must also match the registry's
   geometry for the key (`matter-format.md` 3.1); then
   `gx_core::matter::composite` combines the layers. The composited
   `Section` is kept together with its emitter if it is a depth-0 cell
   (step 5).
4. **Extraction** (`extract`) runs on a pool of `std::thread` workers fed by
   a channel: marching cubes over the composited density of the solid and
   fluid samples at the sample centers, at half the density of the matter
   just inside the boundary with vacuum: the smallest density of the solid
   and fluid samples at least two samples in from any boundary, halved (half
   the largest when no sample is that deep). A body denser toward its
   center draws only its outer surface. Gas and plasma samples count as vacuum here. The grid is
   padded on each face with the first layer of the face neighbor when that
   cell is `Ready` at the same depth (`CellCache::face_neighbors`), with
   vacuum when it is `Empty` at the same depth, and by clamping otherwise
   (see the padding rule below). `World` extracts a cell again whenever its
   set of `Ready` and `Empty` face neighbors changes, so
   seams close as neighbors arrive; the previous mesh is drawn until the
   new one is done. The browser build, with no threads, extracts inline.
   **Volumes** (`volume`): the gas and plasma samples of the same section
   become a `VolumeGrid` at decode time, drawn by ray marching
   (`docs/shading.md`). A cell whose densest sample is gas or plasma is
   therefore a volume, and a cell with both kinds draws a mesh and a
   volume.
5. **Lights** (`light`). Every frame with mass has its depth-0 cell pinned
   (fetched once, never evicted). That cell holds all of the frame's
   matter, so its boundary is the frame's boundary with space, and the
   light escaping it is the light the frame sends out:
   `gx_core::emission::summarize(section, 1000 K)`, radiative transfer
   through the section (opaque solids and fluids radiate from their exposed
   faces, gas and plasma by their extinction) at each sample's radiating
   temperature (`matter-format.md` 3.3). Finer cells give no lights: their
   boundaries face other matter of the same frame. The eight strongest
   frame emitters become point lights of radiant intensity
   `band_power / (4 pi)` per band.
6. **Draw set** (`stream::CellCache::draw_set`), with the depth transition
   rule below, and the **far field** (`farfield`): a frame with mass whose
   `root_extent / 8` region projects to under 2 px has no cells selected
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

The same composited section with the same face neighbors always gives
bit-identical vertex and index buffers, whatever thread extracts it and
however many times:

- The grid is walked in index order, x fastest, then y, then z. Vertices are
  created in that walk and shared through a table addressed by grid edge;
  nothing is sorted, hashed, or ordered by pointer.
- The triangle table is derived once, by a fixed procedure, from the cube
  faces (`extract::case_table`).
- Every value is plain `f64` arithmetic with `sqrt`, both correctly rounded
  in IEEE 754; attributes are narrowed to `f32` last.
- The isovalue (`extract::surface_isovalue`) comes from two fixed erosion
  passes over the section's solid and fluid samples, in index order.
- A mesh depends only on its own section and the sections of its `Ready`
  face neighbors at the same depth, so the order the pool finishes jobs in
  changes only which frame a mesh first appears in. Once every cell has
  arrived, every mesh has been extracted with its final neighbors.

## The padding rule

The section grid is padded with one layer on every face, so the cubes of
the outer layer straddle the cell faces. The padding is never a vacuum
default; it always comes from the hub's data:

- **Neighbor.** When the cell across the face is `Ready` at the same depth,
  the padding is that cell's first layer of samples on that side (resampled
  to the nearest sample center if its resolution differs). When it is
  `Empty` at the same depth (fetched, and the hub holds no matter for it),
  the padding is its samples, which are vacuum.
- **Clamp.** Otherwise (not fetched yet, `Gone`, or only at another depth)
  the cell's own boundary samples repeat outward.

A padding point beyond two or three faces at once is vacuum if any of those
faces has an `Empty` neighbor, else takes the first of those faces, in the
order x, y, z, with a `Ready` neighbor (its other coordinates clamped), else
clamps. Preferring the known vacuum makes two cells sharing one face agree
on the points that are also beyond a second, empty face.

Either way, a density that continues across a face has no crossing at the
face, so no wall is emitted there: two filled neighbors join without a seam
and a filled cell with no neighbor is simply open at its faces (its
neighbors, when they arrive, draw their side). A surface that crosses a
face continues half a sample into the straddling cubes; both cells draw
that half sample from the same samples, so the meshes overlap slightly
rather than leave a gap. Real surfaces at a face, matter in one cell
meeting a neighbor that holds vacuum (`Ready` with vacuum samples there,
or `Empty`), are drawn exactly at the face, because the neighbor's vacuum is
data. Clamping such a face instead would extrude the surface outward as
short walls lit edge-on: the first e2e run with this rule showed exactly
that as a dark notch where a body's poles cross into empty cells, which is
why `Empty` neighbors count.

Accepted limits: neighbors at another depth are not used (the face clamps),
so a depth transition shows a step of up to half a coarse sample; and the
isovalue is per section, so where two neighbors' isovalues differ the
surfaces meet with a small offset. Coarse cells give coarse meshes; the
renderer does not smooth beyond the gradient normals.

Before this rule the padding was vacuum, so matter filling a cell ended in
a wall at each face, and two filled neighbors met wall to wall. Those walls
were lit edge-on and showed as dark creases along every cell boundary
(`docs/e2e/README.md`).

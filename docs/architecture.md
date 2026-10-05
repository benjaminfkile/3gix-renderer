# Architecture

The renderer applies the laws to compiled matter and draws whatever is near
the camera. The specification is `space-model.md` and `matter-format.md` in
the hub repository's `docs/architecture/`.

## The one rule

The renderer may know physics, never objects.

No module, identifier, string, comment, or file in this repository names
anything in the universe. A frame is an integer: "frame 4". The enforcement is
the core library's vocabulary lint, `scripts/vocab-lint.sh` with
`scripts/vocab-banned.txt`, copied here unchanged and run by `scripts/ci.sh`.
A banned word anywhere in a tracked file or file name fails the build.

## Module map

All engine code is in the library crate `renderer` (`crates/renderer`), so the
desktop binary, the headless tests, and the browser build share it.

| Module | What it does | Spec |
|---|---|---|
| `config` | Environment, `.env`, and command line flags, flags winning | |
| `protocol` | The hub protocol without I/O: statuses, socket frames, URLs, registry decode | `compiler-pipeline.md` 4, `matter-format.md` 5 and 6 |
| `hub` | Chunk fetch (`200/202/404/410`), readiness WebSocket, registry load | `compiler-pipeline.md` 4, `matter-format.md` 5 and 6 |
| `mock_hub` | A small hub for tests and `gx-mock-hub` | same |
| `sim` | `SimClock` and integration of every frame to the simulation time | `space-model.md` 6 |
| `camera` | Free camera parented to its nearest frame, re-parenting | `space-model.md` 5 |
| `stream` | Cell selection, the cell cache, decode and composite, the depth transition rule, the fetcher | `space-model.md` 2, 5, 7, 8; `matter-format.md` 3.5, 6 |
| `extract` | Marching cubes over the solid and fluid samples of a composited section, the extraction worker pool | `space-model.md` 2 |
| `volume` | Gas and plasma as 3D grids, the ray march and its CPU reference | `space-model.md` 2, `matter-format.md` 3.3 |
| `farfield` | Frames too small on screen drawn as point sprites, the transition | `space-model.md` 1 and 2, `matter-format.md` 3.3 |
| `light` | Emitters from hot matter, point lights, the emission lookup table | `matter-format.md` 3.3 |
| `world` | Ties stream, extract, volume, light, and farfield together for one frame | |
| `render::scene` | Camera-relative `f32` positions, surface placement, projection, depth strategy | `space-model.md` 5 |
| `render::overlay` | The overlay text | |
| `render::gpu` | wgpu pipelines: surface, volume, and sprite passes, exposure, display pass | |
| `render::headless` | Offscreen RGBA8 target, readback of it and of the float radiance, PNG | |
| `controls` | Keyboard and mouse bindings for the window and the canvas | |
| `app` | Desktop window loop and the headless run | |
| `web` | The browser build: canvas loop, `fetch` and `WebSocket` hub client, `start` | `space-model.md` 10 |

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

`app::render_headless_view` integrates to the launch offset, places the
camera for the run's view (`Home`, or the view of a number key, at a factor
of its distance), selects once, and streams until every selected and pinned
cell is resolved. With `--wait-ready-seconds` it waits for every one to be
drawable (`World::ready`: `Ready` with its mesh, or `Empty`; a `Gone` cell
is not ready) and the binary exits non-zero after writing the image when
the time runs out; without it the run renders whatever has settled within
60 s. `--stats-json` writes the overlay statistics of the rendered frame
(`app::HeadlessStats`), including the request totals of the fetcher
(`stream::FetchTally`): requests, cells fetched (answered `200`), bytes,
and the round trips of the first and the latest request. `--no-overlay`
leaves out the text, the frame markers, and the lines, so only matter is
in the image.

`hub`, `mock_hub`, `app`, and `render::headless` are native only. The library
builds for `wasm32-unknown-unknown` without them, with the `web` feature off
or on; with it on, `web` adds the canvas loop and the browser fetch path
(`docs/web.md`). The binaries are empty stubs on `wasm32`, so the whole
package builds for that target.

## The frame loop

1. The `SimClock` advances by the wall-clock delta times the time scale.
2. `gx_core::integrate::advance` integrates every frame to the clock's time
   with Yoshida4 and steps of at most 600 s. More than 20000 steps in one
   rendered frame are clamped to 20000 and the overlay shows `sim lag`; the
   frames chase the clock over the next rendered frames. Integration is
   never skipped.
3. Free flight moves the camera; then, if `FrameSystem::nearest_frame` of the
   camera position differs from its frame, the camera re-parents.
4. Matter streams: selection (every 100 ms), requests, fetch results,
   readiness pushes, finished meshes, eviction. See "From selection to
   pixels" below.
5. The scene is built camera-relative with the drawn cells and their lights
   and drawn with the overlay.

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

## Hub client

- `fetch_chunk(key)` maps `200`, `202`, `404`, `410` to `Ready`, `Pending`,
  `NotFound`, `Gone`, and everything else to `Error`. A `200` body is kept for
  the session and the key is never requested again.
- `subscribe_ready(build)` opens `GET /space/{s}/build/{b}/chunks/ready` with
  the `X-API-Key` header, sends `{"subscribe":"<key>"}` text frames, and yields
  the keys of `{"chunkReady":"<key>"}` pushes.
- `wait_for_chunk` waits for the push and polls every 2 s for a key still
  pending after 10 s. The registry load gives up after 60 s with a clear
  error, and an invalid registry fails with the validator's code and reason.
- Without `GX_BUILD_ID` the active build comes from `GET /space/{s}/builds`.

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
   | `Missing` | Selected, not requested yet (or a failed request waiting 2 s to retry) |
   | `Requested` | A request is in flight; at most 16 at once, largest projected size first |
   | `Pending(since)` | The hub answered `202`; subscribed to `chunkReady` for the key |
   | `Ready(cell)` | `200`, decoded and composited |
   | `Empty` | The composite holds no matter, or `404` (no layers for the key) |
   | `Gone` | `410`, or bytes that fail validation |

   A `chunkReady` push sends a `Pending` cell back to `Missing`; a push that
   arrives while the request is still in flight makes the `202` ask again at
   once. A cell still pending after 10 s is also polled every 2 s. Cells not
   selected (or drawn as a stand-in) for 60 s are evicted, and beyond 4096
   cached cells the least recently used go first. Selected cells and
   requests in flight are never evicted.
3. **Decode and composite** (`stream::decode_cell`).
   `gx_core::container::decode_chunk` validates every section; each
   section's `cell_origin` and `cell_edge` must also match the registry's
   geometry for the key (`matter-format.md` 3.1); then
   `gx_core::matter::composite` combines the layers. The composited
   `Section` is kept together with its emitter (step 5).
4. **Extraction** (`extract`) runs on a pool of `std::thread` workers fed by
   a channel: marching cubes over the composited density of the solid and
   fluid samples at the sample centers, at half the largest of those
   densities. Gas and plasma samples count as vacuum here. The grid is
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
5. **Lights** (`light`). Each drawn `Ready` cell's emitter is
   `gx_core::emission::summarize(section, 1000 K)`. Emitters of one frame
   merge: band powers add, the position is the power-weighted centroid. The
   eight strongest merged emitters become point lights of radiant intensity
   `band_power / (4 pi)` per band. Only cells in the draw set count, so a
   parent and its children never both contribute.
6. **Draw set** (`stream::CellCache::draw_set`), with the depth transition
   rule below, and the **far field** (`farfield`): a frame with mass whose
   `root_extent / 8` region projects to under 2 px has no cells selected
   and is a point sprite instead; between 2 and 8 px the sprite fades out
   as the cells fade in. Its depth-0 cell is pinned in the cache (fetched
   once, never evicted) for the sprite's brightness, and when none of the
   frame's drawn cells is hot (a sprite-only frame, or one in the
   transition whose cells are beyond the selection range) its hot matter
   still makes a light (`docs/shading.md`).
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
- The isovalue is half the largest density of the section's solid and
  fluid samples, found in the same walk.
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

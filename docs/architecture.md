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
| `hub` | Chunk fetch (`200/202/404/410`), readiness WebSocket, registry load | `compiler-pipeline.md` 4, `matter-format.md` 5 and 6 |
| `mock_hub` | A small hub for tests and `gx-mock-hub` | same |
| `sim` | `SimClock` and integration of every frame to the simulation time | `space-model.md` 6 |
| `camera` | Free camera parented to its nearest frame, re-parenting | `space-model.md` 5 |
| `render::scene` | Camera-relative `f32` positions, projection, depth strategy | `space-model.md` 5 |
| `render::overlay` | The overlay text | |
| `render::gpu` | wgpu pipelines and the one render pass | |
| `render::headless` | Offscreen RGBA8 target, readback, PNG | |
| `app` | Desktop window loop and the headless run | |

Binaries: `gx-renderer` (`src/main.rs`) and `gx-mock-hub`
(`src/bin/gx-mock-hub.rs`).

`hub`, `mock_hub`, `app`, and `render::headless` are native only. The library
builds for `wasm32-unknown-unknown` without them and with the `web` feature
off; the browser build adds its own window and fetch path.

## The frame loop

1. The `SimClock` advances by the wall-clock delta times the time scale.
2. `gx_core::integrate::advance` integrates every frame to the clock's time
   with Yoshida4 and steps of at most 600 s. More than 20000 steps in one
   rendered frame are clamped to 20000 and the overlay shows `sim lag`; the
   frames chase the clock over the next rendered frames. Integration is
   never skipped.
3. Free flight moves the camera; then, if `FrameSystem::nearest_frame` of the
   camera position differs from its frame, the camera re-parents.
4. The scene is built camera-relative and drawn with the overlay.

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
the distance to the nearest marker in front of the camera (at least 1 mm).

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

# Renderer API

The renderer is a component other applications embed. It draws a universe and answers questions about it in runtime terms only: the camera, frames, surfaces over a frame's reference ellipsoid, layers, light, and time. Every convention of one body (coordinate names, place names, how an application likes to fly over a surface) belongs to the application that embeds the renderer, never to this API.

Approved 2026-11-22. This file holds the API's items; the rules they follow are `3GIXHub/docs/architecture/rendering.md` section 8, and that contract still binds everything drawn.

## 1. Shape

1. **One implementation.** The module `crates/renderer/src/api.rs` (and any submodules under `api/`) holds every item below. The desktop window, the browser page, and every future build are consumers of it, under `rendering.md` section 6 rule 5 (one renderer).
2. **Two bindings, same names, same behavior.**
   - Native: the `renderer` crate's public `api` module. A native consumer implements a `Consumer` trait whose `tick(&mut self, api: &mut Api)` runs once per loop tick on the render thread, before simulation and drawing, and receives events through `Api::events()`.
   - Web: the wasm-bindgen class `Renderer` (from `crates/renderer/src/web/`) and the JS package in `packages/3gix-renderer` (section 9). Calls are made from the page's thread between ticks.
   Names are camelCase in JS and snake_case in Rust; every item exists in both.
3. **When calls apply.** Setters queue a command applied at the start of the next tick, in call order. Getters return the state as of the last completed tick. Queries (section 6) run synchronously against what is loaded at the call and never fetch.
4. **Nothing moves by itself.** The API never moves the camera, never changes its speed, and never animates. An application that wants any of that calls `setCamera` or `setFlightInput` itself.

## 2. Values

- SI units, `f64`. Lengths in meters, angles in radians unless an item says degrees, speeds in m/s.
- A **pose** is `{ frameId, position: [x, y, z], orientation: [x, y, z, w], speed }`: position relative to the frame origin in the frame's own axes, orientation the camera axes relative to the frame axes (unit quaternion), speed the set flight speed. These are the fields of `camera::Camera`.
- Positions in a frame's axes stay put as the frame spins, so placing or reading a fixed point needs no time.
- Frame ids are integers. In JS they are numbers; an id at or above 2^53 is an error.
- Time in the API is wall-clock milliseconds (`utcMs`) for instants, as the hub's clock counts them; the renderer's universe time (seconds since the origin, `gx_core::clock`) stays internal.
- Errors: Rust items return `Result`; JS methods throw `Error` with the same message.

## 3. Lifecycle

| Item | Behavior |
|---|---|
| `create(target, connection, options)` | Connects to the hub (`hubUrl`, `apiKey`, `spaceId`, `buildId` may be empty), loads the registry, starts the loop. `options` are today's view and time options (`startOn`, `timeScale`, `exposureStops`, `hideLayers`, ...) parsed by `options.rs`, unchanged. Resolves (JS promise, Rust `Result`) once the loop runs. Native target: a window or a headless size; web target: a canvas element or its id. |
| `destroy()` | Stops the loop, the module workers, and uploads; frees GPU memory. Any later setter or query is an error; getters return the state as of the last tick. The page or process may create a new renderer afterward. |
| `pause()` / `resume()` | Paused: no drawing, no streaming, no computing; simulation time stays where it was and resumes from there. |
| `resize(width, height, pixelRatio)` | For hosts that size the surface themselves. The browser build also follows canvas size changes on its own, as now. |

## 4. Camera

| Item | Behavior |
|---|---|
| `getCamera()` | The pose. |
| `setCamera(pose, { ride })` | Puts the camera exactly at the pose, reparented as `Camera::reparent_to` does if the frame differs from the current one. Without `ride` (the default) the camera is free: it keeps its place in space while frames move and turn. With `ride: true` it moves and turns with its frame, keeping its place in the frame's axes, as a camera placed with `placeOver` does; flight input moves it relative to the frame. Rust: `set_camera(pose, ride)`. A pose with an unknown frame, a non-finite value, or a non-unit quaternion (beyond 1e-6) is an error. |
| `placeOver(frameId, direction, height, pitchDeg)` | Same placement as `--start-on <frame>:<x>,<y>,<z>@+<height>^<pitch>`, shared code. While it is pending, a change of time keeps the camera over the frame. |
| `setControls("builtin" \| "none")` | `builtin` (the default): keyboard and mouse flight and the key bindings in `docs/controls.md`. `none`: the renderer ignores keyboard and mouse input for the camera, choosers, clock, and exposure; window events (resize, focus, close) still apply. |
| `setFlightInput({ move: [x, y, z], look: [yaw, pitch, roll], boost })` | The free camera's input, the same `camera::FlightInput` the keys produce: `move` components in -1 to 1 along camera axes, `look` in radians to turn this tick, `boost` held or not. Held until replaced; `setFlightInput(null)` clears it. Works in both control modes. |
| `setSpeed(mps)` | The set flight speed, within `SPEED_LIMITS`. |
| `setFov(deg)` / `getFov()` | Vertical field of view. |

## 5. Time and frames

| Item | Behavior |
|---|---|
| `getTime()` | `{ utcMs, originUtcMs, nowUtcMs, timeScale }`: the instant drawn, the universe's origin, real now, the playback rate. |
| `setTime(utcMs)` | Jumps to that instant, clamped between origin and now as the start options are. |
| `setTimeScale(x)` | Playback rate; negative runs back. Same clamps as now. |
| `frames()` | Every frame of the registry: `{ id, parentId, mass, angularVelocity: [x, y, z], hasSurface, surface }`, `surface` null or `{ radiusXy, radiusZ, heightFloor }` from the frame's level-0 surface tiles once they have arrived (`matter-format.md` section 10). |
| `transform(point, fromFrame, toFrame, kind)` | A point (`kind: "point"`) or direction (`"direction"`) from one frame's axes into another's at the instant drawn. |

## 6. Queries

Each runs against what is loaded at the call. None fetches or waits.

| Item | Behavior |
|---|---|
| `surfaceHeight(frameId, direction)` | `{ height, level, finer }` or null: height of the drawn surface above the frame's reference ellipsoid along `direction` (from the frame origin, in its axes), the tile level it came from, and whether a finer tile there is still on its way. Null when the frame has no surface tile loaded there. Uses `gx_core::surface` geometry and the tiles' heights as drawn (morph included), so the value matches the picture. |
| `raycast(x, y)` | From a pixel (CSS pixels on the web, physical on native, top left origin): `{ frameId, position, distance, kind }` or null: the frame hit, the point in its axes, meters from the camera, and `"surface"` or `"cells"`. The nearest drawn thing wins, as the depth buffer would decide. |
| `project(frameId, point)` | `{ x, y, depth }` in the same pixels as `raycast`, or null when the point is behind the camera, outside the view, or hidden behind drawn matter or surface tiles. Lets an application place its own labels over the view. |

The renderer draws no text for consumers. Labels are the application's own elements, placed with `project()`.

## 7. Layers, display, compute

| Item | Behavior |
|---|---|
| `layers()` | `[{ id, overlay, exclusive, shown }]` for the build, `shown` as currently applied. |
| `setShown(layerId, bool)` | As the `L` chooser does now (`rendering.md` section 7 item 7). |
| `setExposure({ mode: "auto", biasStops } \| { mode: "fixed", stops })` | As the exposure keys and `--exposure-stops`. |
| `setHud(bool)` | The built-in text overlay on (default) or off. |
| `setMeshMode("surfaces" \| "vertices" \| "edges")`, `setMaxFps(n \| null)` | As the options of the same names. |
| `setComputeBudget(workers)` / `getComputeBudget()` | Module workers this client may run: 0 to the default (`cores - 2`, 3/4 workers, as now). 0 turns computing and uploading off: the client only streams what is already stored, and what is missing stays missing (drawn as the contract draws missing data). Takes effect without restarting; work in flight finishes or is dropped, never half uploaded. |

## 8. Events

Rust: `Api::events()` drains the queue each tick. JS: `on(name, fn)` returns an unsubscribe function.

| Event | Carries | When |
|---|---|---|
| `ready` | nothing | The loop runs. |
| `error` | message | Something stopped the loop or a call failed after it was queued. |
| `camera` | pose | At most once per tick, only when the pose changed (by input, `setCamera`, or a reparent). |
| `frameChanged` | `{ from, to }` | The camera's frame changed. |
| `settled` | `{ seconds }` | Everything the current view asked for has arrived and is drawn; the same condition the budget harness waits for. Fires again after each later settle. |
| `hit` | a `raycast` result or null, plus `{ x, y, button }` | A click (press and release within 4 px). Fired in both control modes. |
| `stats` | `{ fps, tilesDrawn, triangles, cellsDrawn, streaming, computing, uploading }`: `streaming` cells still waiting for sections, `computing` module runs so far this session, `uploading` offers waiting or being sent; `fps` 0 where nothing is drawn | Once a second. |

## 9. The JS package

`packages/3gix-renderer` (npm name `3gix-renderer`, marked `"private": true` until publishing is approved):

- A `Renderer` class over the wasm-bindgen class, with TypeScript types for every item above.
- A React component `GxRenderer` (React as a peer dependency) with props `connection`, `options`, `camera` (applied with `setCamera` when it changes), `controls`, `hud`, `computeBudget`, `shownLayers`, and callbacks `onReady`, `onError`, `onCamera`, `onFrameChanged`, `onSettled`, `onHit`, `onStats`; a ref gives the `Renderer`.
- The wasm, `module-worker.js`, and `module-runner.js` inside the package, resolved relative to the package so no consumer copies files by hand.
- A demo page that uses only the package's public surface.

## 10. Not exposed

Caches and eviction, the GPU copy budget, tile and cell selection (split distances, morph, the 8 px rule), cell layout and extraction, mesh formats, worker queues, section addresses, the upload and spot-check protocol, and the hub API. A consumer that needs one of these asks for an item to be added here first.

## 11. Proof

- Every item has a test in `crates/renderer/tests/api.rs` against the mock hub, on the native binding.
- The web binding builds in `scripts/ci.sh` and a Node check (`scripts/api-surface-check.mjs`) asserts the generated `renderer.d.ts` and the package's types export every item in this file under the names given here.
- The browser page `web/index.html` and the desktop window use only this API.

# 3gix-renderer

The renderer of the 3GIX space runtime. It applies the laws to compiled matter and draws whatever is near the camera. It runs natively on the desktop and in the browser from one Rust codebase through wgpu.

## The one rule

The renderer may know physics. It may never know objects.

Nothing in this repository knows what any piece of matter is. There is no code path, file, identifier, string, or comment named after any object in the universe. The vocabulary lint from the core library enforces the banned word list, and a failed lint fails the build.

## What it does

- Fetches the frame registry for a build and integrates every frame from the origin with first post-Newtonian gravity, oblateness, and rotation under torques, keeping checkpoints so any moment between the origin and now is one checkpoint interval away.
- Enters at the hub's now and can go back to the origin and replay forward at any rate, never past now.
- Draws every frame where it was when the light now reaching the camera left it, and lights every frame from where each emitter was one light-travel time earlier.
- Asks the hub for directions for the cells near the camera, at a depth chosen by distance, and for each layer streams the finished section from the edge or the local cache, or computes it here: the worker runs the layer's compiler module (WebAssembly, by hash) on its source tiles, hands the section to the renderer at once, and uploads it with the claim. The registry is just another key.
- Composites overlapping sections, relaxes fluid matter toward the local equipotential, derives emission from temperature.
- Extracts surfaces from solid and fluid matter, ray marches gas and plasma as volumes, and draws with camera-relative single precision over a floating origin.
- Draws frames too far away for cells as point sprites of the right brightness, so hot matter is visible from anywhere.
- Lets the camera go anywhere.

## What it depends on

- `gx-core`, the shared library: matter format decoder and validator, units, laws, chunk keys, section addresses, the module request and tile formats, source manifests, the frame registry.
- `gx-module-host` (native) to run modules; the browser uses its own `WebAssembly` in a Web Worker.
- `wgpu` and `winit` for graphics and windowing. Headless tests run on a software Vulkan driver with no window.

## Running

Configuration comes from the environment, or a `.env` file in the working
directory (see `.env.example`). The API key needs the `fetch:chunks`
capability and is never logged.

```sh
GX_HUB_URL=...      # hub base URL
GX_API_KEY=...      # API key with fetch:chunks
GX_SPACE_ID=...     # space id
GX_BUILD_ID=...     # optional, defaults to the active build
```

Desktop, with a window:

```sh
cargo run --release --bin gx-renderer
```

Headless screenshot, no window or display needed (a software Vulkan driver is
enough):

```sh
cargo run --release --bin gx-renderer -- --headless --screenshot out.png --width 1280 --height 720
```

Flags: `--start <now|origin>` (default now), `--start-offset-seconds <f64>`
(offset from that start, clamped between the origin and now),
`--time-scale <f64>` (playback rate of a start in the past), `--headless`,
`--screenshot <path.png>` (implies `--headless`), `--width`, `--height`, and
`--hub-url`, `--space-id`, `--build-id` to override the environment. See
`--help`.

The renderer asks for the high performance adapter, so a machine with an
integrated and a discrete GPU draws on the discrete one; `GX_ADAPTER` or
`--adapter <name>` picks an adapter by a part of its name instead (the log
names the one in use). The window presents one frame per display refresh;
`--max-fps <u32>` caps it lower.

Headless runs also take `--view <home|0-9>` (the `Home` view, the default,
or the view a number key jumps to), `--view-distance-scale <f64>` (a factor
on that view's distance from its frame origin), `--wait-ready-seconds <f64>`
(wait until every selected cell is ready and exit non-zero if that takes
longer), `--stats-json <path.json>` (the overlay statistics: cells selected
and fetched, bytes, the first and last round trip), and `--no-overlay`
(matter only: no overlay text, frame markers, or lines).
`GX_EDGE_URL` or `--edge-url` reads sections, modules, and tiles from an
edge instead of the hub. The local cache is a directory under the platform
cache directory by default; `--cache-dir <path>` moves it,
`--cache-max-bytes <u64>` limits it (default 20 GB), and `--no-cache` turns
it off. `--workers <u32>` sets the threads that run modules (default: cores
minus two, at least one).
`--exposure-stops <f64>`, in the window too, fixes the exposure instead of
adapting it, in stops relative to the automatic exposure of a scene whose
log-average luminance is 1 W m^-2 sr^-1 (`docs/shading.md`):

```sh
cargo run --release --bin gx-renderer -- --screenshot frame-3.png --view 3 \
    --wait-ready-seconds 300 --stats-json frame-3.json --no-overlay
```

Without a hub, `gx-mock-hub` serves a whole space the way the hub does (a
layer declaring the frames of a registry file, and the core library's
example module on every frame up to `--depth`) and prints the variables to
use. The first client computes and uploads what it sees; the next streams
it:

```sh
cargo run --bin gx-mock-hub &          # prints GX_HUB_URL, GX_API_KEY, GX_SPACE_ID
export GX_HUB_URL=... GX_API_KEY=... GX_SPACE_ID=...
cargo run --bin gx-renderer -- --headless --screenshot out.png
```

A headless run streams the cells the `Home` view selects (for up to 60 s)
before it renders.

Browser, with WebGPU (see `docs/web.md`):

```sh
sh web/build.sh
cd web && python3 -m http.server 8000
```

The page asks for the hub URL, API key, space id, and build id
and keeps them in `localStorage`.

Controls are in `docs/controls.md`; the module map, the floating origin, the
depth strategy, and the matter pipeline are in `docs/architecture.md`; the
lighting model, volumes, the far field, exposure, and tone curve are in
`docs/shading.md`; the browser build is in `docs/web.md`.

## Checks

```sh
sh scripts/ci.sh
```

Runs format, clippy with `-D warnings`, the tests (including the headless
renders, the GPU volume against its CPU reference, the client against the
mock hub with the example module, and the worker against the example
module's conformance hashes), the vocabulary lint, the browser worker's
module runner under Node against the same hashes,
the wasm32 builds with the `web` feature off and on, and `web/build.sh`. The
matter render tests save their images as `target/tmp/matter-render.png` and
`target/tmp/full-scene.png` with pixel statistics beside them in `.txt`
files. The browser build is not run in CI; it is verified by hand
(`docs/web.md`). CI also runs the budget test in release
(`cargo test --release -p renderer --test budget`): frame-thread step at
4096 cells, refinement rounds, and no shape model, on synthetic worlds
(`3GIXHub/docs/architecture/rendering.md` section 5.3).

### Performance budget on a real stack

```sh
GX_HUB_URL=... GX_API_KEY=... GX_SPACE_ID=... sh scripts/perf-budget.sh [--keep DIR]
```

Runs the benchmark views B1 to B4 of `rendering.md` section 5.1 headless,
each with a fresh cache, and checks them against section 5.2 and
`perf/baseline.json` (`tools/perf-budget`). Exits non-zero on any breach.
Every change that can affect what is drawn or how fast puts its overall line
in the commit message (`rendering.md` section 6).

### Push hook

`scripts/hooks/pre-push` runs format, clippy with `-D warnings`, the
vocabulary lint, and the library unit tests before every push and refuses
the push if any fails. Install it once per clone with
`sh scripts/install-hooks.sh` (sets `core.hooksPath` locally).

## End to end

```sh
sh scripts/e2e.sh
```

Starts the hub with its own container harness, clones and serves the system
compiler, takes four headless screenshots against them, and checks the
images with `tools/png-stats`. The recorded run, the screenshots, and their
statistics are in `docs/e2e/`.

## Specification

- Architecture: `3GIXHub/docs/architecture/space-model.md` (v2)
- Hub, client, and module contract: `3GIXHub/docs/architecture/client-compute.md`
- Matter format 3: `3GIXHub/docs/architecture/matter-format.md`
- Rendering contract and performance budget: `3GIXHub/docs/architecture/rendering.md`

## Contributing rules

- No word from the core library's `scripts/vocab-banned.txt`, and no proper noun for any body, anywhere in this repository.
- No infrastructure identifiers, secrets, hostnames, or LAN addresses in code, history, docs, or CI output. This repository is public.
- No em or en dashes in any text.
- `sh scripts/ci.sh` passes on every change, and the push hook is installed.
- The rendering contract (`rendering.md`) holds: no shape model, no rule fitted to one compiler, surfaces only at occupancy 0.5, one fix attempt per visual problem.

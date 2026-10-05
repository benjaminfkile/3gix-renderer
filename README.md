# 3gix-renderer

The renderer of the 3GIX space runtime. It applies the laws to compiled matter and draws whatever is near the camera. It runs natively on the desktop and in the browser from one Rust codebase through wgpu.

## The one rule

The renderer may know physics. It may never know objects.

Nothing in this repository knows what any piece of matter is. There is no code path, file, identifier, string, or comment named after any object in the universe. The vocabulary lint from the core library enforces the banned word list, and a failed lint fails the build.

## What it does

- Fetches the frame registry for a build and integrates every frame forward from the epoch under Newtonian gravity.
- Fetches matter cells near the camera at a depth chosen by distance, through the hub's chunk endpoint and readiness WebSocket.
- Composites overlapping sections, relaxes fluid matter toward the local equipotential, derives emission from temperature.
- Extracts surfaces from solid and fluid matter, ray marches gas and plasma as volumes, and draws with camera-relative single precision over a floating origin.
- Draws frames too far away for cells as point sprites of the right brightness, so hot matter is visible from anywhere.
- Lets the camera go anywhere, at any time scale.

## What it depends on

- `gx-core`, the shared library: matter format decoder and validator, units, laws, chunk keys, the frame registry.
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
cargo run --release --bin gx-renderer -- --time-scale 86400
```

Headless screenshot, no window or display needed (a software Vulkan driver is
enough):

```sh
cargo run --release --bin gx-renderer -- --headless --screenshot out.png --width 1280 --height 720
```

Flags: `--time-scale <f64>` (default 1), `--start-offset-seconds <f64>`
(simulation time offset from the epoch at launch), `--headless`,
`--screenshot <path.png>` (implies `--headless`), `--width`, `--height`, and
`--hub-url`, `--space-id`, `--build-id` to override the environment. See
`--help`.

Headless runs also take `--view <home|0-9>` (the `Home` view, the default,
or the view a number key jumps to), `--view-distance-scale <f64>` (a factor
on that view's distance from its frame origin), `--wait-ready-seconds <f64>`
(wait until every selected cell is ready and exit non-zero if that takes
longer), `--stats-json <path.json>` (the overlay statistics: cells selected
and fetched, bytes, the first and last round trip), and `--no-overlay`
(matter only: no overlay text, frame markers, or lines):

```sh
cargo run --release --bin gx-renderer -- --screenshot frame-3.png --view 3 \
    --wait-ready-seconds 300 --stats-json frame-3.json --no-overlay
```

Without a hub, `gx-mock-hub` serves a registry the way the hub does and prints
the variables to use. `--cell <key>=<file>` adds a matter cell from a hub
chunk container file:

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

The page asks for the hub URL, API key, space id, build id, and time scale
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
renders and the GPU volume against its CPU reference), the vocabulary lint,
the wasm32 builds with the `web` feature off and on, and `web/build.sh`. The
matter render tests save their images as `target/tmp/matter-render.png` and
`target/tmp/full-scene.png` with pixel statistics beside them in `.txt`
files. The browser build is not run in CI; it is verified by hand
(`docs/web.md`).

## End to end

```sh
sh scripts/e2e.sh
```

Starts the hub with its own container harness, clones and serves the system
compiler, takes four headless screenshots against them, and checks the
images with `tools/png-stats`. The recorded run, the screenshots, and their
statistics are in `docs/e2e/`.

## Specification

- Architecture: `3GIXHub/docs/architecture/space-model.md`
- Matter format v1: `3GIXHub/docs/architecture/matter-format.md`

## Contributing rules

- No word from the core library's `scripts/vocab-banned.txt`, and no proper noun for any body, anywhere in this repository.
- No infrastructure identifiers, secrets, hostnames, or LAN addresses in code, history, docs, or CI output. This repository is public.
- No em or en dashes in any text.
- `cargo fmt --check`, `cargo clippy -- -D warnings`, `cargo test`, and the vocabulary lint pass on every change.

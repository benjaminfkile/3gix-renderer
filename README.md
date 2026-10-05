# 3gix-renderer

The renderer of the 3GIX space runtime. It applies the laws to compiled matter and draws whatever is near the camera. It runs natively on the desktop and in the browser from one Rust codebase through wgpu.

## The one rule

The renderer may know physics. It may never know objects.

Nothing in this repository knows what any piece of matter is. There is no code path, file, identifier, string, or comment named after any object in the universe. The vocabulary lint from the core library enforces the banned word list, and a failed lint fails the build.

## What it does

- Fetches the frame registry for a build and integrates every frame forward from the epoch under Newtonian gravity.
- Fetches matter cells near the camera at a depth chosen by distance, through the hub's chunk endpoint and readiness WebSocket.
- Composites overlapping sections, relaxes fluid matter toward the local equipotential, derives emission from temperature.
- Extracts surfaces or ray marches the density field and draws with camera-relative single precision over a floating origin.
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

Controls are in `docs/controls.md`; the module map, the floating origin, the
depth strategy, and the matter pipeline are in `docs/architecture.md`; the
lighting model, exposure, and tone curve are in `docs/shading.md`.

## Checks

```sh
sh scripts/ci.sh
```

Runs format, clippy with `-D warnings`, the tests (including the headless
renders), the vocabulary lint, and the wasm32 build of the library crate. The
matter render test saves its image as `target/tmp/matter-render.png` and its
pixel statistics as `target/tmp/matter-render.txt`.

## Specification

- Architecture: `3GIXHub/docs/architecture/space-model.md`
- Matter format v1: `3GIXHub/docs/architecture/matter-format.md`

## Contributing rules

- No word from the core library's `scripts/vocab-banned.txt`, and no proper noun for any body, anywhere in this repository.
- No infrastructure identifiers, secrets, hostnames, or LAN addresses in code, history, docs, or CI output. This repository is public.
- No em or en dashes in any text.
- `cargo fmt --check`, `cargo clippy -- -D warnings`, `cargo test`, and the vocabulary lint pass on every change.

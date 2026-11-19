# The browser build

The renderer runs in the browser from the same library crate as the
desktop binary (`space-model.md` section 10: native desktop and WebAssembly
from one codebase). The browser-only code is the `web` module
(`crates/renderer/src/web/`), compiled for `wasm32-unknown-unknown` with the
`web` feature:

- `winit` draws on a canvas element instead of a window, and wgpu runs on
  the browser's WebGPU.
- The hub client (`web::hub`) uses `reqwest`'s wasm client (the browser's
  `fetch`) for the clock, the build, directions, edge reads, and uploads.
  Futures run on the page's event loop through `wasm-bindgen-futures`.
  There is no readiness socket.
- The client loop (`web::fetch`) is the same `directions::Coordinator` the
  desktop drives, so batching, deduplication, retries, and compositing are
  identical.
- The worker (`web::worker`) runs modules with the browser's own
  `WebAssembly` in a dedicated Web Worker, `web/module-worker.js`, which
  uses `web/module-runner.js`. Modules are instantiated with an empty
  import object, so a module that imports anything fails; the 1 GiB memory
  limit is checked there. The page side does every other step through the
  same `gx-worker::job` checks as the desktop: the address, the module,
  manifest, and tiles by hash, request and tile bytes from `gx-core` built
  for WebAssembly, and the library validator on the output. A call over the
  60 s time limit terminates the Web Worker, which starts again for the
  next call. Jobs run one at a time. Input sections of layers with inputs
  (`client-compute.md` 10.3) are resolved by the shared resolver
  (`gx-worker::resolve`, the same code the desktop runs) through
  `web::worker::WebPlatform`: `fetch` for directions and streams, the
  Cache API, and the Web Worker. Uploads and check reports carry
  `X-Inputs-Sha256`. The fetcher's own streams of `ready` sections share
  the resolver's flight per address, so a section is downloaded once.
- The local cache (`web::cache`) is the Cache API (cache `gx-renderer`,
  keys `/gx-cache/{kind}/{hash or address}`), with the same keys as the
  desktop's cache directory, evicted least recently used by size (20 GB,
  or less if the browser's quota is smaller). It needs a secure context;
  without one nothing is cached and everything still works.
- `console_error_panic_hook` sends panics to the browser console.
- The page has no threads, so surface extraction runs inline in the frame
  loop. Everything else (selection, the cell cache, volumes, the far field,
  lights, exposure, controls) is the desktop code.

## Building

Needs the `wasm32-unknown-unknown` target (pinned in `rust-toolchain.toml`),
`wasm-pack`, and `wasm-opt` from binaryen on the `PATH`.

```sh
sh web/build.sh
```

This runs `wasm-pack build crates/renderer --target web --out-dir ../../web/pkg
--no-opt --release -- --features web`, then `wasm-opt -O2` on
`web/pkg/renderer_bg.wasm` in place. The output is `web/pkg/renderer.js` and
`web/pkg/renderer_bg.wasm` (ignored by git). `wasm-pack` options go before
the `--`, and the cargo `--features web` after it; with the feature first,
`wasm-pack` hands `--out-dir` to cargo as well, which fails.
`scripts/ci.sh` runs the build and checks that both files exist.

To check that the code compiles without packaging:

```sh
cargo build --target wasm32-unknown-unknown --features web -p renderer
```

## Serving

Any static file server works; the page needs no server code:

```sh
cd web
python3 -m http.server 8000
```

Then open `http://localhost:8000/` in a browser with WebGPU. WebGPU needs a
secure context, which `localhost` is; from another machine, serve over
HTTPS.

The page is one canvas filling the window and a small settings panel: hub
URL, API key, space id, build id
(empty for the active build of the space). The values are stored in the browser's
`localStorage`, the API key included, so use a key meant for this purpose.
`Start` loads the module, checks the view options, connects to the hub,
loads the registry, and starts the frame loop; an error shows in the panel.
Controls are those of the desktop (`docs/controls.md`); click the canvas
first so it has the keyboard focus. `Escape` stops the frame loop.

## View and time options

Both builds start the same way from the same settings (`rendering.md`
section 6 rule 5). The browser takes the native view and time flags under
the same names in camelCase, from `web/config.js` (see
`web/config.example.js`) and then from the page's query string, which
overrides `config.js`, so a link opens a view:

```
http://localhost:8000/?startOn=3:0,0,1@%2B10&timeScale=60
```

(`+` in a query string is a space, so write `@+` as `@%2B`.)

| Option | Native flag | Meaning |
|---|---|---|
| `start` | `--start` | `now` (default) or `origin` |
| `startOffsetSeconds` | `--start-offset-seconds` | simulation time offset from the start, seconds, clamped between the origin and now (default 0) |
| `timeScale` | `--time-scale` | playback rate of a start in the past, simulation seconds per wall second, negative backwards (default 1) |
| `startOn` | `--start-on` | `<frame>:<x>,<y>,<z>[@[+]<m>[^<deg>]]`: spawn onto the frame along that direction, or with `@<m>` place the camera that far from its origin |
| `view` | `--view` | `home` or `0` to `9`; only headless runs use it |
| `viewDistanceScale` | `--view-distance-scale` | factor on the headless view's distance from its frame origin (default 1) |
| `exposureStops` | `--exposure-stops` | fixed exposure, stops; leave out for automatic |
| `meshMode` | `--mesh-mode` | `surfaces` (default), `vertices`, or `edges` |

Without any the client enters at the hub's now at the `Home` view. The
values go through the same parser as the native flags
(`crates/renderer/src/options.rs`), and the same function places the
camera and sets the clock at start. An invalid value shows the parser's
message in the panel and nothing starts. `web/view-options.js` gathers the
values; `scripts/web-options-check.mjs` checks it under Node in CI.

## Keys

The keys are those of `docs/controls.md`. The number keys, on the top row
or the keypad (`Numpad0` to `Numpad9`), jump to a frame's view, or pick a
frame while the spawn chooser (`G`) is open. While the speed field (`V`) is
open both type digits instead.

## What the hub must allow

- **CORS.** The page sends `GET`, `POST` (directions), and `PUT`
  (uploads) requests with the `X-API-Key`, `X-Layer`, `X-Key`,
  `X-Span-Index`, and `X-Content-Sha256` headers from its own origin, so
  the hub (and the edge) must answer the preflight and allow that origin,
  those methods, and those headers. A page cannot send `Expect:
  100-continue`, so a browser upload always carries its body.

## Browsers

WebGPU is needed; there is no WebGL fallback. At the time of writing:

| Browser | WebGPU |
|---|---|
| Chrome and Edge | 113 and later on Windows, macOS, and ChromeOS; later versions on Android; on Linux behind a flag in most versions |
| Firefox | 141 and later on Windows; other platforms in later versions or behind a flag |
| Safari | 26 and later on macOS, iOS, and iPadOS |

Volumes and sprites blend into a half float target, which every WebGPU
implementation supports (`docs/shading.md`, "Float target"). Check the
browser's own WebGPU status page for the current state.

## Verification

There is no browser in CI. CI verifies that the crate builds for
`wasm32-unknown-unknown` with the `web` feature, that clippy passes on that
build, and that `web/build.sh` produces the package. Node (V8) stands in
for the browser's `WebAssembly`: `node scripts/worker-node-check.mjs` runs
`web/module-runner.js`, then `web/module-worker.js` in a worker thread
through its message protocol, against the core example modules'
conformance vectors (`testdata/example-module/` for ABI version 1,
`testdata/example-module-v2/` for version 2, whose `gx_inputs`,
`gx_sources`, and `gx_compile` take the frame and input buffers), and
fails unless every input list and section's SHA-256 matches. A third pass
resolves the version 2 module's layer inputs through the module worker the
way `WebPlatform` does (each input computed with the version 1 module once
per key) and checks the resolved input bytes and sections against the
vectors. The native worker checks the same vectors
(`crates/worker/tests/determinism.rs`), so the two runtimes give
byte-identical sections. `scripts/ci.sh` runs it. Running it is verified
by hand on a desktop browser, against a hub:

1. `sh web/build.sh`, serve `web/`, open the page.
2. Fill in the panel and press `Start`; the overlay appears and the frame
   markers of the registry are drawn.
3. Matter streams in or is computed here (the overlay's cell counts,
   sections streamed, and sections computed here rise), surfaces are lit,
   hot gas glows as a volume, and far frames show as dots.
4. Fly with the desktop controls, go back in time with `,` or `Backspace`
   and return with `N`,
   reload the page and see the panel come back filled in.

No automated test pretends to cover these steps.

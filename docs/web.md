# The browser build

The renderer runs in the browser from the same library crate as the
desktop binary (`space-model.md` section 10: native desktop and WebAssembly
from one codebase). The browser-only code is the `web` module
(`crates/renderer/src/web/`), compiled for `wasm32-unknown-unknown` with the
`web` feature:

- `winit` draws on a canvas element instead of a window, and wgpu runs on
  the browser's WebGPU.
- The hub client (`web::hub`) uses `reqwest`'s wasm client (the browser's
  `fetch`) for the clock, the build, directions, edge reads, and uploads,
  and a `web_sys::WebSocket` for readiness notices. Futures run on the
  page's event loop through `wasm-bindgen-futures`.
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
  next call. Jobs run one at a time.
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
(empty for the active build of the space). The client enters at the hub's
now, like the desktop one. The values are stored in the browser's
`localStorage`, the API key included, so use a key meant for this purpose.
`Start` loads the module, connects to the hub, loads the registry, and
starts the frame loop; an error shows in the panel. Controls are those of
the desktop (`docs/controls.md`); click the canvas first so it has the
keyboard focus. `Escape` stops the frame loop.

## What the hub must allow

- **CORS.** The page sends `GET`, `POST` (directions), and `PUT`
  (uploads) requests with the `X-API-Key` and `X-Claim` headers from its
  own origin, so the hub (and the edge) must answer the preflight and allow
  that origin, those methods, and those headers.
- **The readiness socket.** A browser cannot set headers on a WebSocket
  upgrade, so the socket opens without the API key. If the hub refuses it,
  a `pending` key is still asked about again after the hub's `retryMs`, so
  everything still arrives, more slowly. A closed socket is opened again
  every 5 s and subscribes again to every address still awaited.

## Browsers

WebGPU is needed; there is no WebGL fallback. At the time of writing:

| Browser | WebGPU |
|---|---|
| Chrome and Edge | 113 and later on Windows, macOS, and ChromeOS; later versions on Android; on Linux behind a flag in most versions |
| Firefox | 141 and later on Windows; other platforms in later versions or behind a flag |
| Safari | 26 and later on macOS, iOS, and iPadOS |

Volumes and sprites blend into a 32-bit float target, which needs the
WebGPU `float32-blendable` feature (Chrome 132 and later, and current
Firefox and Safari releases). Without it they are drawn unblended
(`docs/shading.md`, "Float blending"). Check the browser's own WebGPU status
page for the current state.

## Verification

There is no browser in CI. CI verifies that the crate builds for
`wasm32-unknown-unknown` with the `web` feature, that clippy passes on that
build, and that `web/build.sh` produces the package. Node (V8) stands in
for the browser's `WebAssembly`: `node scripts/worker-node-check.mjs` runs
`web/module-runner.js`, then `web/module-worker.js` in a worker thread
through its message protocol, against the core example module's
conformance vectors (`testdata/example-module/`), and fails unless every
section's SHA-256 matches. The native worker checks the same vectors
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

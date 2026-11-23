# 3gix-renderer

The renderer as a JS package (`docs/api.md` section 9): a `Renderer` class
with TypeScript types for every item of the API, and a React component,
`GxRenderer`. The wasm, `module-worker.js`, and `module-runner.js` are files
of the package under `dist/wasm/`, found with `new URL(..., import.meta.url)`,
so a bundler (checked with Vite) and a plain static server both serve them
with no copying by hand. The package is private and not published.

## Install

From a local checkout, after building the browser package and this one:

```sh
sh web/build.sh
cd packages/3gix-renderer && npm ci && npm run build
cd /path/to/app && npm install /path/to/3gix-renderer/packages/3gix-renderer
```

React (18 or newer) is a peer dependency, needed only for `GxRenderer`.
With Vite, keep the package out of dependency prebundling
(`optimizeDeps: { exclude: ["3gix-renderer"] }`) so its file lookups stay
relative to its own files.

## The Renderer class

`Renderer.create(target, connection, options)` resolves once the loop runs
on `target` (a canvas or its id). `connection` is
`{ hubUrl, apiKey, spaceId, buildId? }`; `options` are the view and time
options by camelCase name. Every other method is the item of the same name
in `docs/api.md` sections 3 to 7. `on(name, fn)` subscribes to an event of
section 8 and returns the unsubscribe function.

## GxRenderer

| Prop | Applied |
|---|---|
| `connection`, `options` | At creation; a change by value creates the renderer again. |
| `camera` | `setCamera`, when it changes by value. |
| `controls` | `setControls`. |
| `hud` | `setHud`. |
| `computeBudget` | `setComputeBudget`. |
| `shownLayers` | `{ [layerId]: shown }`, `setShown` for each entry that changed. |
| `className`, `style` | On the canvas. |

| Callback | Called with |
|---|---|
| `onReady` | The `Renderer`, once the loop runs. |
| `onError` | The message, when creating fails, a setter throws, or the `error` event fires. |
| `onCamera`, `onFrameChanged`, `onSettled`, `onHit`, `onStats` | The event's value (section 8). |

A ref gives the `Renderer` (null until it runs). The renderer is destroyed
on unmount. Under React strict mode's double mount a creation that finishes
after its effect was cleaned up is destroyed at once, so one loop runs.

## Example

```tsx
import { useRef } from "react";
import { GxRenderer, type Renderer } from "3gix-renderer";

export function View({ connection }) {
    const renderer = useRef<Renderer | null>(null);
    return (
        <GxRenderer
            ref={renderer}
            connection={connection}
            options={{ timeScale: 60 }}
            hud={false}
            onCamera={(pose) => console.log(pose.frameId, pose.position)}
            onHit={(h) => h.hit && console.log(h.hit.kind, h.hit.distance)}
        />
    );
}
```

## Demo

`demo/` is a Vite page using only the package's exports: the view, the
`camera` event, a pause button, the layers with toggles, and the compute
budget. Copy `demo/config.example.js` to `demo/config.js` (gitignored), fill
in the connection and options, then `npm run build && npm run demo`.

## Checks

`scripts/ci.sh` runs `npm ci`, the build, `tsc --noEmit` over the package
and the demo, the component's unit tests (vitest under jsdom with the wasm
class mocked), `npm pack --dry-run` (which must list the wasm and both
worker files), and `scripts/api-surface-check.mjs` over the package's types.

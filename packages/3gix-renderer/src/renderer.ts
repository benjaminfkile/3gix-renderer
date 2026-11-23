// The `Renderer` class (`docs/api.md` sections 3 to 9): typed methods and
// events over the wasm-bindgen class of the web binding. The wasm and the
// module worker are files of this package, found with
// `new URL(..., import.meta.url)`, so a bundler and a plain static server
// both serve them from where the package is.

import init, { Renderer as WasmRenderer, setWorkerScript } from "./wasm/renderer.js";
import type {
    Connection,
    Controls,
    Exposure,
    FlightInput,
    FrameId,
    FrameInfo,
    LayerInfo,
    MeshMode,
    Pose,
    Projection,
    RaycastHit,
    RendererEvent,
    RendererEvents,
    SurfaceHeight,
    TimeState,
    TransformKind,
    Vec3,
    ViewOptions,
} from "./types.js";

let loading: Promise<void> | null = null;

/**
 * Loads the wasm module once per page and points the module worker at the
 * package's copy. A bundler may move or inline each file it finds in a
 * `new URL` literal, which breaks the worker's relative import of its
 * runner, so the worker starts from a blob of its own source with that
 * import pointing at wherever the runner ended up.
 */
function load(): Promise<void> {
    loading ??= (async () => {
        await init({ module_or_path: new URL("./wasm/renderer_bg.wasm", import.meta.url) });
        const worker = new URL("./wasm/module-worker.js", import.meta.url);
        const runner = new URL("./wasm/module-runner.js", import.meta.url);
        const source = await (await fetch(worker)).text();
        // A dev server may already have rewritten the specifier to its own
        // path (and a ?t= query), so match any specifier naming
        // module-runner.js.
        const linked = source.replace(/(["'])[^"']*module-runner\.js(\?[^"']*)?\1/, JSON.stringify(runner.href));
        if (linked === source) {
            throw new Error("module-worker.js: no import of ./module-runner.js");
        }
        setWorkerScript(URL.createObjectURL(new Blob([linked], { type: "text/javascript" })));
    })();
    return loading;
}

/** The renderer on a canvas: created with `Renderer.create`, driven by its methods, observed with `on`. */
export class Renderer {
    private readonly inner: WasmRenderer;

    private constructor(inner: WasmRenderer) {
        this.inner = inner;
    }

    /** `create(target, connection, options)`: connects, loads the registry, and resolves once the loop runs on `target`, a canvas or its id. */
    static async create(
        target: HTMLCanvasElement | string,
        connection: Connection,
        options: ViewOptions = {},
    ): Promise<Renderer> {
        await load();
        const c = { buildId: "", ...connection };
        return new Renderer(await WasmRenderer.create(target, c, options));
    }

    /** `destroy()`: stops the loop, the module workers, and uploads; any later setter or query throws. */
    destroy(): void {
        this.inner.destroy();
    }

    /** `pause()`: no drawing, streaming, or computing; the time holds. */
    pause(): void {
        this.inner.pause();
    }

    /** `resume()`: runs again from the time it was paused at. */
    resume(): void {
        this.inner.resume();
    }

    /** `resize(width, height, pixelRatio)`: the surface size, for hosts that size the canvas themselves. */
    resize(width: number, height: number, pixelRatio: number): void {
        this.inner.resize(width, height, pixelRatio);
    }

    /** `getCamera()`: the pose as of the last tick. */
    getCamera(): Pose {
        return this.inner.getCamera() as Pose;
    }

    /** `setCamera(pose)`: puts the camera exactly at the pose next tick. */
    setCamera(pose: Pose): void {
        this.inner.setCamera(pose);
    }

    /** `placeOver(frameId, direction, height, pitchDeg)`: the placement of `--start-on`. */
    placeOver(frameId: FrameId, direction: Vec3, height: number, pitchDeg: number): void {
        this.inner.placeOver(frameId, direction, height, pitchDeg);
    }

    /** `setControls("builtin" | "none")`: who flies the camera from the keyboard and mouse. */
    setControls(controls: Controls): void {
        this.inner.setControls(controls);
    }

    /** `setFlightInput(input | null)`: the free camera's held input; null clears it. */
    setFlightInput(input: FlightInput | null): void {
        this.inner.setFlightInput(input);
    }

    /** `setSpeed(mps)`: the set flight speed. */
    setSpeed(mps: number): void {
        this.inner.setSpeed(mps);
    }

    /** `setFov(deg)`: the vertical field of view, degrees. */
    setFov(deg: number): void {
        this.inner.setFov(deg);
    }

    /** `getFov()`: the vertical field of view, degrees. */
    getFov(): number {
        return this.inner.getFov();
    }

    /** `getTime()`: the instant drawn, the origin, real now, and the playback rate. */
    getTime(): TimeState {
        return this.inner.getTime() as TimeState;
    }

    /** `setTime(utcMs)`: jumps to that instant next tick. */
    setTime(utcMs: number): void {
        this.inner.setTime(utcMs);
    }

    /** `setTimeScale(x)`: the playback rate; negative runs back. */
    setTimeScale(x: number): void {
        this.inner.setTimeScale(x);
    }

    /** `frames()`: every frame of the registry. */
    frames(): FrameInfo[] {
        return this.inner.frames() as FrameInfo[];
    }

    /** `transform(point, fromFrame, toFrame, kind)`: a point or direction between frames' axes at the instant drawn. */
    transform(point: Vec3, fromFrame: FrameId, toFrame: FrameId, kind: TransformKind): Vec3 {
        return this.inner.transform(point, fromFrame, toFrame, kind) as Vec3;
    }

    /** `surfaceHeight(frameId, direction)`: the drawn surface height along a direction, or null. */
    surfaceHeight(frameId: FrameId, direction: Vec3): SurfaceHeight | null {
        return this.inner.surfaceHeight(frameId, direction) as SurfaceHeight | null;
    }

    /** `raycast(x, y)`: what a CSS pixel hits, or null. */
    raycast(x: number, y: number): RaycastHit | null {
        return this.inner.raycast(x, y) as RaycastHit | null;
    }

    /** `project(frameId, point)`: where a point is in the view, CSS pixels, or null. */
    project(frameId: FrameId, point: Vec3): Projection | null {
        return this.inner.project(frameId, point) as Projection | null;
    }

    /** `layers()`: the build's layers, `shown` as applied. */
    layers(): LayerInfo[] {
        return this.inner.layers() as LayerInfo[];
    }

    /** `setShown(layerId, shown)`: shows or hides a layer. */
    setShown(layerId: string, shown: boolean): void {
        this.inner.setShown(layerId, shown);
    }

    /** `setExposure(exposure)`: automatic with a bias, or fixed. */
    setExposure(exposure: Exposure): void {
        this.inner.setExposure(exposure);
    }

    /** `setHud(on)`: the built-in text overlay on or off. */
    setHud(on: boolean): void {
        this.inner.setHud(on);
    }

    /** `setMeshMode(mode)`: surfaces, vertices, or edges. */
    setMeshMode(mode: MeshMode): void {
        this.inner.setMeshMode(mode);
    }

    /** `setMaxFps(n | null)`: the frame cap, null for none. */
    setMaxFps(fps: number | null): void {
        this.inner.setMaxFps(fps);
    }

    /** `setComputeBudget(workers)`: module workers this client may run, 0 for none. */
    setComputeBudget(workers: number): void {
        this.inner.setComputeBudget(workers);
    }

    /** `getComputeBudget()`: module workers this client may run. */
    getComputeBudget(): number {
        return this.inner.getComputeBudget();
    }

    /** `on(name, fn)`: calls `fn` with each event `name` (section 8); returns the unsubscribe function. */
    on<E extends RendererEvent>(name: E, fn: (value: RendererEvents[E]) => void): () => void {
        const off = this.inner.on(name, fn);
        return () => {
            off.call(null);
        };
    }
}

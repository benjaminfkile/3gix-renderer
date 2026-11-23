// The `GxRenderer` React component (`docs/api.md` section 9): a canvas
// with a `Renderer` on it, created on mount and destroyed on unmount.
// Props are applied through the `Renderer`'s setters when they change;
// `camera` only when it changes by value, so a parent that rebuilds an
// equal pose each render does not hold the camera in place. A ref gives
// the `Renderer` (null until it runs). Under strict mode's double mount a
// creation that finishes after its effect was cleaned up is destroyed at
// once, so one loop runs.

import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";
import type { CSSProperties } from "react";
import { Renderer } from "./renderer.js";
import type {
    Connection,
    Controls,
    FrameChange,
    Hit,
    Pose,
    Settled,
    Stats,
    ViewOptions,
} from "./types.js";

/** The props of `GxRenderer` (section 9). */
export interface GxRendererProps {
    /** The hub to connect to; a change by value creates the renderer again. */
    connection: Connection;
    /** The view and time options at creation; a change by value creates the renderer again. */
    options?: ViewOptions;
    /** A pose applied with `setCamera` when it changes by value. */
    camera?: Pose;
    /** Applied with `setControls`. */
    controls?: Controls;
    /** Applied with `setHud`. */
    hud?: boolean;
    /** Applied with `setComputeBudget`. */
    computeBudget?: number;
    /** Layer id to shown, each applied with `setShown` when it changes. */
    shownLayers?: Record<string, boolean>;
    /** Called with the `Renderer` once the loop runs. */
    onReady?: (renderer: Renderer) => void;
    /** Called with the message when creating fails or the loop reports an error. */
    onError?: (message: string) => void;
    /** The `camera` event. */
    onCamera?: (pose: Pose) => void;
    /** The `frameChanged` event. */
    onFrameChanged?: (change: FrameChange) => void;
    /** The `settled` event. */
    onSettled?: (settled: Settled) => void;
    /** The `hit` event. */
    onHit?: (hit: Hit) => void;
    /** The `stats` event. */
    onStats?: (stats: Stats) => void;
    /** The canvas's class. */
    className?: string;
    /** The canvas's style. */
    style?: CSSProperties;
}

function message(e: unknown): string {
    return e instanceof Error ? e.message : String(e);
}

/** A canvas with the renderer on it; the ref gives the `Renderer`. */
export const GxRenderer = forwardRef<Renderer | null, GxRendererProps>(function GxRenderer(props, ref) {
    const { connection, options, camera, controls, hud, computeBudget, shownLayers } = props;
    const canvas = useRef<HTMLCanvasElement>(null);
    const latest = useRef(props);
    latest.current = props;
    const [renderer, setRenderer] = useState<Renderer | null>(null);
    useImperativeHandle(ref, () => renderer as Renderer, [renderer]);

    const createKey = JSON.stringify([connection, options ?? {}]);
    useEffect(() => {
        const target = canvas.current;
        if (!target) {
            return;
        }
        let cancelled = false;
        let created: Renderer | null = null;
        const offs: Array<() => void> = [];
        Renderer.create(target, connection, options ?? {}).then(
            (r) => {
                if (cancelled) {
                    r.destroy();
                    return;
                }
                created = r;
                offs.push(
                    r.on("error", (m) => latest.current.onError?.(m)),
                    r.on("camera", (p) => latest.current.onCamera?.(p)),
                    r.on("frameChanged", (c) => latest.current.onFrameChanged?.(c)),
                    r.on("settled", (s) => latest.current.onSettled?.(s)),
                    r.on("hit", (h) => latest.current.onHit?.(h)),
                    r.on("stats", (s) => latest.current.onStats?.(s)),
                );
                setRenderer(r);
                latest.current.onReady?.(r);
            },
            (e) => {
                if (!cancelled) {
                    latest.current.onError?.(message(e));
                }
            },
        );
        return () => {
            cancelled = true;
            for (const off of offs) {
                off();
            }
            created?.destroy();
            setRenderer(null);
        };
        // connection and options are compared by value through createKey.
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [createKey]);

    // Each setter call can throw (a bad value); the message goes to onError.
    const apply = (f: () => void) => {
        try {
            f();
        } catch (e) {
            latest.current.onError?.(message(e));
        }
    };

    const appliedCamera = useRef<string | null>(null);
    const cameraKey = camera === undefined ? null : JSON.stringify(camera);
    useEffect(() => {
        if (!renderer) {
            appliedCamera.current = null;
            return;
        }
        if (cameraKey !== null && cameraKey !== appliedCamera.current) {
            appliedCamera.current = cameraKey;
            apply(() => renderer.setCamera(JSON.parse(cameraKey) as Pose));
        }
    }, [renderer, cameraKey]);

    useEffect(() => {
        if (renderer && controls !== undefined) {
            apply(() => renderer.setControls(controls));
        }
    }, [renderer, controls]);

    useEffect(() => {
        if (renderer && hud !== undefined) {
            apply(() => renderer.setHud(hud));
        }
    }, [renderer, hud]);

    useEffect(() => {
        if (renderer && computeBudget !== undefined) {
            apply(() => renderer.setComputeBudget(computeBudget));
        }
    }, [renderer, computeBudget]);

    const appliedLayers = useRef<Record<string, boolean>>({});
    const layersKey = JSON.stringify(shownLayers ?? {});
    useEffect(() => {
        if (!renderer) {
            appliedLayers.current = {};
            return;
        }
        const wanted = JSON.parse(layersKey) as Record<string, boolean>;
        for (const id of Object.keys(wanted).sort()) {
            if (appliedLayers.current[id] !== wanted[id]) {
                apply(() => renderer.setShown(id, wanted[id]));
            }
        }
        appliedLayers.current = wanted;
    }, [renderer, layersKey]);

    return <canvas ref={canvas} className={props.className} style={props.style} />;
});

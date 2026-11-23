// The GxRenderer component (docs/api.md section 9) against a mocked wasm
// class: prop changes call the right methods, unmount destroys, and strict
// mode's double mount leaves one renderer running.

import { StrictMode, createRef } from "react";
import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const instances: Array<Record<string, ReturnType<typeof vi.fn>>> = [];

vi.mock("../src/wasm/renderer.js", () => {
    class Renderer {
        static create = vi.fn(async () => {
            const r = {
                destroy: vi.fn(),
                setCamera: vi.fn(),
                setControls: vi.fn(),
                setHud: vi.fn(),
                setComputeBudget: vi.fn(),
                setShown: vi.fn(),
                on: vi.fn(() => () => {}),
            };
            instances.push(r);
            return r;
        });
    }
    return { default: vi.fn(async () => {}), setWorkerScript: vi.fn(), Renderer };
});

// The package fetches module-worker.js to link it to its runner.
vi.stubGlobal("fetch", vi.fn(async () => ({ text: async () => 'import { a } from "./module-runner.js";' })));
URL.createObjectURL = vi.fn(() => "blob:worker");

const { GxRenderer } = await import("../src/index.js");
type Props = Parameters<typeof GxRenderer>[0];

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const connection = { hubUrl: "hub-url", apiKey: "k", spaceId: "s" };
const pose = (x: number) => ({ frameId: 1, position: [x, 0, 0] as [number, number, number], orientation: [0, 0, 0, 1] as [number, number, number, number], speed: 1 });

let container: HTMLDivElement;
let root: ReturnType<typeof createRoot>;

beforeEach(() => {
    instances.length = 0;
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
});

afterEach(() => {
    act(() => root.unmount());
    container.remove();
});

async function render(el: React.ReactElement) {
    await act(async () => {
        root.render(el);
    });
}

describe("GxRenderer", () => {
    it("applies props through the setters when they change", async () => {
        const props: Props = { connection, camera: pose(1), controls: "none", hud: false, computeBudget: 2, shownLayers: { a: true } };
        await render(<GxRenderer {...props} />);
        expect(instances).toHaveLength(1);
        const r = instances[0];
        expect(r.setCamera).toHaveBeenCalledTimes(1);
        expect(r.setCamera).toHaveBeenLastCalledWith(pose(1));
        expect(r.setControls).toHaveBeenLastCalledWith("none");
        expect(r.setHud).toHaveBeenLastCalledWith(false);
        expect(r.setComputeBudget).toHaveBeenLastCalledWith(2);
        expect(r.setShown).toHaveBeenLastCalledWith("a", true);

        // An equal pose in a new object is not applied again.
        await render(<GxRenderer {...props} camera={pose(1)} />);
        expect(r.setCamera).toHaveBeenCalledTimes(1);

        await render(<GxRenderer {...props} camera={pose(2)} hud={true} computeBudget={0} shownLayers={{ a: true, b: false }} />);
        expect(r.setCamera).toHaveBeenCalledTimes(2);
        expect(r.setCamera).toHaveBeenLastCalledWith(pose(2));
        expect(r.setHud).toHaveBeenLastCalledWith(true);
        expect(r.setComputeBudget).toHaveBeenLastCalledWith(0);
        expect(r.setShown).toHaveBeenCalledTimes(2);
        expect(r.setShown).toHaveBeenLastCalledWith("b", false);
        expect(r.setControls).toHaveBeenCalledTimes(1);
    });

    it("gives the Renderer through a ref and calls onReady", async () => {
        const ref = createRef<import("../src/index.js").Renderer | null>();
        const onReady = vi.fn();
        await render(<GxRenderer ref={ref} connection={connection} onReady={onReady} />);
        expect(ref.current).not.toBeNull();
        expect(onReady).toHaveBeenCalledWith(ref.current);
    });

    it("destroys the renderer on unmount", async () => {
        await render(<GxRenderer connection={connection} />);
        expect(instances[0].destroy).not.toHaveBeenCalled();
        act(() => root.unmount());
        expect(instances[0].destroy).toHaveBeenCalledTimes(1);
        root = createRoot(container);
    });

    it("leaves one renderer running under strict mode", async () => {
        await render(
            <StrictMode>
                <GxRenderer connection={connection} />
            </StrictMode>,
        );
        const live = instances.filter((r) => r.destroy.mock.calls.length === 0);
        expect(instances.length).toBeGreaterThanOrEqual(1);
        expect(live).toHaveLength(1);
    });
});

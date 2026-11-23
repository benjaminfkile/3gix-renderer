// The package demo (docs/api.md section 9): the view, the camera event,
// pause, layers, and the compute budget, through the package's public
// exports only. The connection and start come from demo/config.js.

import { StrictMode, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { GxRenderer } from "3gix-renderer";
import type { Connection, LayerInfo, Pose, Renderer, ViewOptions } from "3gix-renderer";

interface DemoConfig extends Connection {
    options?: ViewOptions;
}

async function loadConfig(): Promise<DemoConfig | null> {
    try {
        // Next to the page, not this module: a built demo keeps its
        // modules under assets/.
        const path = new URL("config.js", document.baseURI).href;
        return ((await import(/* @vite-ignore */ path)) as { default: DemoConfig }).default;
    } catch {
        return null;
    }
}

function Demo({ config }: { config: DemoConfig }) {
    const renderer = useRef<Renderer | null>(null);
    const [pose, setPose] = useState<Pose | null>(null);
    const [paused, setPaused] = useState(false);
    const [layers, setLayers] = useState<LayerInfo[]>([]);
    const [shown, setShown] = useState<Record<string, boolean>>({});
    const [budget, setBudget] = useState<number | undefined>(undefined);
    const [error, setError] = useState("");

    const togglePause = () => {
        const r = renderer.current;
        if (!r) return;
        if (paused) r.resume();
        else r.pause();
        setPaused(!paused);
    };

    return (
        <>
            <GxRenderer
                ref={renderer}
                className="view"
                connection={config}
                options={config.options}
                computeBudget={budget}
                shownLayers={shown}
                onReady={(r) => {
                    setLayers(r.layers());
                    setBudget(r.getComputeBudget());
                }}
                onError={setError}
                onCamera={setPose}
            />
            <div className="panel">
                <button onClick={togglePause}>{paused ? "resume" : "pause"}</button>
                <label>
                    {" "}compute budget{" "}
                    <input
                        type="number"
                        min={0}
                        value={budget ?? ""}
                        onChange={(e) => setBudget(Number(e.target.value))}
                    />
                </label>
                <pre>{pose ? JSON.stringify(pose, null, 1) : "no camera event yet"}</pre>
                {layers.map((l) => (
                    <div key={l.id}>
                        <label>
                            <input
                                type="checkbox"
                                checked={shown[l.id] ?? l.shown}
                                onChange={(e) => setShown({ ...shown, [l.id]: e.target.checked })}
                            />
                            {" "}{l.id}{l.overlay ? " (overlay)" : ""}
                        </label>
                    </div>
                ))}
                {error && <pre>{error}</pre>}
            </div>
        </>
    );
}

const config = await loadConfig();
const root = createRoot(document.getElementById("root")!);
root.render(
    config ? (
        <StrictMode>
            <Demo config={config} />
        </StrictMode>
    ) : (
        <p>Copy demo/config.example.js to demo/config.js and fill it in.</p>
    ),
);

// The client's Web Worker: runs compiler modules off the page's thread so
// computing never blocks drawing (space-model.md section 2, "The worker";
// client-compute.md section 4). Started by the renderer's web build
// (crates/renderer/src/web/worker.rs) as a module worker.
//
// Messages in, each with an integer id:
//   { id, op: "load", hash, wasm }        compile and keep a module by hash
//   { id, op: "sources", hash, request }  run gx_sources
//   { id, op: "compile", hash, request, tiles }  run gx_compile
// Messages out:
//   { id, ok: true, bytes }   (bytes: Uint8Array; empty for "load")
//   { id, ok: false, error }  (error: a reason for the log)
//
// The page checks hashes, fetches tiles, validates sections, and enforces
// the time limit by terminating this worker.

import { compileModule, runSources, runCompile } from "./module-runner.js";

const modules = new Map();

async function handle(msg) {
    switch (msg.op) {
        case "load":
            if (!modules.has(msg.hash)) modules.set(msg.hash, await compileModule(msg.wasm));
            return new Uint8Array(0);
        case "sources":
        case "compile": {
            const module = modules.get(msg.hash);
            if (!module) throw new Error(`module ${msg.hash} is not loaded`);
            return msg.op === "sources"
                ? await runSources(module, msg.request)
                : await runCompile(module, msg.request, msg.tiles);
        }
        default:
            throw new Error(`unknown operation ${msg.op}`);
    }
}

self.onmessage = async (event) => {
    const msg = event.data;
    try {
        const bytes = await handle(msg);
        self.postMessage({ id: msg.id, ok: true, bytes }, [bytes.buffer]);
    } catch (e) {
        self.postMessage({ id: msg.id, ok: false, error: String(e && e.message ? e.message : e) });
    }
};

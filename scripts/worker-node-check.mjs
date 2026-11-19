// Runs the browser worker's module code under Node (V8, standing in for
// the browser) against the core example module's conformance vectors, and
// fails unless every output matches its expected SHA-256 (client-compute.md
// section 4, space-model.md section 4 item 6). The native worker checks
// the same vectors in crates/worker/tests/determinism.rs, so the two
// runtimes give byte-identical sections.
//
// Usage: node scripts/worker-node-check.mjs [MODULE.wasm VECTORS_DIR]
// (default: testdata/example-module/module.wasm and its vectors/).
//
// Three passes: web/module-runner.js called directly, then web/module-worker.js
// itself in a worker thread, driven through its message protocol exactly as
// the page drives the Web Worker. Both passes also run the core's version 2
// example module (testdata/example-module-v2, client-compute.md section
// 4.1) through gx_inputs, gx_sources, and gx_compile with its vectors'
// frame and input bytes. The third resolves the version 2 module's layer
// inputs through the module worker, computing each with the version 1
// module, as the browser's platform layer does.

import { readFileSync, readdirSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { createHash } from "node:crypto";
import { Worker } from "node:worker_threads";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const [wasmPath, dir] = process.argv.length > 3
    ? process.argv.slice(2, 4)
    : [join(root, "testdata/example-module/module.wasm"), join(root, "testdata/example-module/vectors")];

const runner = await import(pathToFileURL(join(root, "web/module-runner.js")).href);

const sha = (b) => createHash("sha256").update(b).digest("hex");
const wasm = new Uint8Array(readFileSync(wasmPath));
const hash = sha(wasm);

const names = readdirSync(dir)
    .filter((f) => f.endsWith(".request.bin"))
    .map((f) => f.slice(0, -".request.bin".length))
    .sort();
if (names.length === 0) {
    console.error(`${dir}: no vectors`);
    process.exit(1);
}
const vectors = names.map((name) => ({
    name,
    request: new Uint8Array(readFileSync(join(dir, `${name}.request.bin`))),
    tiles: new Uint8Array(readFileSync(join(dir, `${name}.tiles.bin`))),
    expected: readFileSync(join(dir, `${name}.sha256`), "utf8").trim(),
}));

function tileIds(tiles) {
    const v = new DataView(tiles.buffer, tiles.byteOffset, tiles.byteLength);
    const ids = [];
    let at = 4;
    for (let i = 0; i < v.getUint32(0, true); i++) {
        ids.push(v.getBigUint64(at, true));
        at += 12 + v.getUint32(at + 8, true);
    }
    return ids.join(",");
}

function sourceIds(bytes) {
    const v = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const ids = [];
    for (let at = 0; at + 8 <= bytes.length; at += 8) ids.push(v.getBigUint64(at, true));
    return ids.join(",");
}

let failed = 0;
function check(pass, name, sources, tiles, section, expected) {
    const got = sha(section);
    if (sources !== tileIds(tiles)) {
        console.log(`${pass} FAIL ${name}: sources [${sources}], tiles carry [${tileIds(tiles)}]`);
        failed++;
    } else if (got !== expected) {
        console.log(`${pass} FAIL ${name}: output ${got}, expected ${expected}`);
        failed++;
    } else {
        console.log(`${pass} ok   ${name} ${got}`);
    }
}

// Pass 1: the runner directly.
const module = await runner.compileModule(wasm);
for (const v of vectors) {
    try {
        const sources = sourceIds(await runner.runSources(module, v.request));
        const section = await runner.runCompile(module, v.request, v.tiles);
        check("runner", v.name, sources, v.tiles, section, v.expected);
    } catch (e) {
        console.log(`runner FAIL ${v.name}: ${e.message}`);
        failed++;
    }
}

// The version 2 example module and its vectors.
const v2Dir = join(root, "testdata/example-module-v2");
const wasmV2 = new Uint8Array(readFileSync(join(v2Dir, "module.wasm")));
const hashV2 = sha(wasmV2);
const vectorsV2 = readdirSync(join(v2Dir, "vectors"))
    .filter((f) => f.endsWith(".request.bin"))
    .map((f) => f.slice(0, -".request.bin".length))
    .sort()
    .map((name) => {
        const read = (ext) => new Uint8Array(readFileSync(join(v2Dir, "vectors", `${name}.${ext}`)));
        return {
            name: `v2 ${name}`,
            request: read("request.bin"),
            frames: read("frames.bin"),
            tiles: read("tiles.bin"),
            inputList: read("input-list.bin"),
            inputs: read("inputs.bin"),
            expected: readFileSync(join(v2Dir, "vectors", `${name}.sha256`), "utf8").trim(),
        };
    });
if (vectorsV2.length === 0) {
    console.error(`${v2Dir}: no vectors`);
    process.exit(1);
}
const same = (a, b) => a.length === b.length && a.every((x, i) => x === b[i]);

async function checkV2(pass, run) {
    for (const v of vectorsV2) {
        try {
            const list = await run("inputs", v);
            if (!same(list, v.inputList)) {
                console.log(`${pass} FAIL ${v.name}: the input list differs`);
                failed++;
                continue;
            }
            const sources = sourceIds(await run("sources", v));
            check(pass, v.name, sources, v.tiles, await run("compile", v), v.expected);
        } catch (e) {
            console.log(`${pass} FAIL ${v.name}: ${e.message}`);
            failed++;
        }
    }
}

const moduleV2 = await runner.compileModule(wasmV2);
await checkV2("runner", (op, v) =>
    op === "inputs" ? runner.runInputs(moduleV2, v.request, v.frames)
        : op === "sources" ? runner.runSources(moduleV2, v.request, v.frames)
            : runner.runCompile(moduleV2, v.request, v.tiles, v.inputs));

// A version 1 module given frame records fails.
try {
    await runner.runSources(module, vectors[0].request, vectorsV2[0].frames);
    console.log("runner FAIL a version 1 module took frame records");
    failed++;
} catch (e) {
    console.log(`runner ok   frame records refused for version 1 (${e.message})`);
}

// A module with an import is refused.
const importing = new Uint8Array([
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, // magic, version
    0x01, 0x04, 0x01, 0x60, 0x00, 0x00, // type section: () -> ()
    0x02, 0x07, 0x01, 0x01, 0x6d, 0x01, 0x66, 0x00, 0x00, // import m.f
]);
try {
    await runner.compileModule(importing);
    console.log("runner FAIL a module with an import was accepted");
    failed++;
} catch (e) {
    console.log(`runner ok   imports refused (${e.message})`);
}

// Pass 2: module-worker.js in a worker thread, through its messages. The
// shim gives it the Web Worker globals it uses: self.onmessage and
// self.postMessage.
const shim = `
import { parentPort } from "node:worker_threads";
globalThis.self = globalThis;
globalThis.postMessage = (m, t) => parentPort.postMessage(m, t);
await import(${JSON.stringify(pathToFileURL(join(root, "web/module-worker.js")).href)});
parentPort.on("message", (data) => self.onmessage({ data }));
`;
const worker = new Worker(new URL("data:text/javascript," + encodeURIComponent(shim)));
let nextId = 1;
const waiting = new Map();
worker.on("message", (m) => {
    const w = waiting.get(m.id);
    waiting.delete(m.id);
    if (m.ok) w.resolve(m.bytes);
    else w.reject(new Error(m.error));
});
worker.on("error", (e) => {
    console.error(`worker error: ${e.message}`);
    process.exit(1);
});
const call = (msg) => new Promise((resolve, reject) => {
    const id = nextId++;
    waiting.set(id, { resolve, reject });
    worker.postMessage({ id, ...msg });
});

try {
    await call({ op: "load", hash, wasm });
    for (const v of vectors) {
        const sources = sourceIds(await call({ op: "sources", hash, request: v.request }));
        const section = await call({ op: "compile", hash, request: v.request, tiles: v.tiles });
        check("worker", v.name, sources, v.tiles, section, v.expected);
    }
    await call({ op: "load", hash: hashV2, wasm: wasmV2 });
    await checkV2("worker", (op, v) => call({ op, hash: hashV2, ...v }));
    try {
        await call({ op: "sources", hash: "0".repeat(64), request: vectors[0].request });
        console.log("worker FAIL a module never loaded ran");
        failed++;
    } catch (e) {
        console.log(`worker ok   unknown module refused (${e.message})`);
    }
} catch (e) {
    console.log(`worker FAIL ${e.message}`);
    failed++;
}

// Pass 3: input resolution through the module worker, the calls the
// browser's platform layer (crates/renderer/src/web/worker.rs,
// WebPlatform) makes for the shared resolver (client-compute.md section
// 10.3): gx_inputs of the version 2 module with the frame records, every
// listed layer slot input computed with the version 1 module (the layer
// the example reader's slot 1 reads) once per key, then gx_compile with
// the frame records and the inputs in gx_inputs order. Every input buffer
// and section must match the vectors.
function decodeList(bytes) {
    const v = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const out = [];
    let at = 4;
    for (let i = 0; i < v.getUint32(0, true); i++) {
        const slot = v.getUint32(at, true);
        const len = v.getUint32(at + 4, true);
        out.push({ slot, key: new TextDecoder().decode(bytes.subarray(at + 8, at + 8 + len)) });
        at += 8 + len;
    }
    return out;
}
function decodeInputs(bytes) {
    const v = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const out = [];
    let at = 4;
    for (let i = 0; i < v.getUint32(0, true); i++) {
        const len = v.getUint32(at, true);
        out.push(bytes.subarray(at + 4, at + 4 + len));
        at += 4 + len;
    }
    return out;
}
function encodeInputs(list) {
    const out = new Uint8Array(4 + list.reduce((n, b) => n + 4 + b.length, 0));
    const v = new DataView(out.buffer);
    v.setUint32(0, list.length, true);
    let at = 4;
    for (const b of list) {
        v.setUint32(at, b.length, true);
        out.set(b, at + 4);
        at += 4 + b.length;
    }
    return out;
}
// The input layer's source tiles as the core's generator gives them: every
// id the same parameters (root extent 2e7 m, 2.5 kg/m^3, 300 K).
function layerTiles(ids) {
    const out = new Uint8Array(4 + ids.length * 36);
    const v = new DataView(out.buffer);
    v.setUint32(0, ids.length, true);
    let at = 4;
    for (const id of ids) {
        v.setBigUint64(at, id, true);
        v.setUint32(at + 8, 24, true);
        v.setFloat64(at + 12, 2.0e7, true);
        v.setFloat64(at + 20, 2.5, true);
        v.setFloat64(at + 28, 300.0, true);
        at += 36;
    }
    return out;
}
// A request (encode_request) for another key with the reader's origin and
// span, which is the input span for these vectors' layers.
function withKey(request, key) {
    const k = new TextEncoder().encode(key);
    const out = new Uint8Array(36 + k.length);
    out.set(request.subarray(0, 32));
    new DataView(out.buffer).setUint32(32, k.length, true);
    out.set(k, 36);
    return out;
}
try {
    await call({ op: "load", hash, wasm });
    let resolved = 0;
    for (const v of vectorsV2) {
        const list = decodeList(await call({ op: "inputs", hash: hashV2, request: v.request, frames: v.frames }));
        const memory = new Map();
        const inputs = [...decodeInputs(v.frames)];
        for (const e of list) {
            if (e.slot !== 1) throw new Error(`${v.name}: input of slot ${e.slot}`);
            if (!memory.has(e.key)) {
                const request = withKey(v.request, e.key);
                const ids = sourceIds(await call({ op: "sources", hash, request }));
                const tiles = layerTiles(ids === "" ? [] : ids.split(",").map(BigInt));
                memory.set(e.key, await call({ op: "compile", hash, request, tiles }));
            }
            inputs.push(memory.get(e.key));
        }
        const passed = encodeInputs(inputs);
        if (!same(passed, v.inputs)) {
            console.log(`resolve FAIL ${v.name}: the resolved inputs differ from the vector's`);
            failed++;
            continue;
        }
        resolved += list.length;
        const section = await call({ op: "compile", hash: hashV2, request: v.request, tiles: v.tiles, inputs: passed });
        check("resolve", v.name, sourceIds(await call({ op: "sources", hash: hashV2, request: v.request, frames: v.frames })), v.tiles, section, v.expected);
    }
    if (resolved === 0) {
        console.log("resolve FAIL no vector read a layer input");
        failed++;
    }
} catch (e) {
    console.log(`resolve FAIL ${e.message}`);
    failed++;
}
await worker.terminate();

if (failed > 0) {
    console.error(`worker-node-check: ${failed} failed`);
    process.exit(1);
}
console.log(`worker-node-check: ${vectors.length} version 1 and ${vectorsV2.length} version 2 vectors ok in every pass`);

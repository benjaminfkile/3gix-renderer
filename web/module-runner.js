// Runs compiler modules with the browser's own WebAssembly, under the module
// ABI v1 and v2 (client-compute.md sections 4 and 4.1). Used by the client's Web Worker
// (module-worker.js) and, unchanged, by scripts/worker-node-check.mjs, which
// runs it under Node (V8, standing in for the browser) against the core
// example module's conformance vectors.
//
// The worker only ever sees modules, request bytes, tile bytes, and section
// bytes. Request and tile bytes are made by gx-core built for WebAssembly
// (the renderer's web build), the same bytes the native worker passes to
// wasmtime.
//
// Rules enforced here:
// - a module is compiled once and instantiated with an empty import object
//   for every call, so a module that imports anything fails;
// - gx_abi_version must return 1 or 2; a version 2 module also exports
//   gx_inputs, and its gx_sources and gx_compile take the frame and input
//   buffers. A version 1 module runs exactly as before and may be given no
//   frame records or inputs;
// - a call that leaves the module's memory above the limit (1 GiB) fails.
// The time limit is enforced by the page, which terminates the worker.

/** The memory limit of one call, bytes: 1 GiB. */
export const MEMORY_LIMIT = 1 << 30;

/** Compiles module bytes and checks they declare no imports. */
export async function compileModule(bytes) {
    const module = await WebAssembly.compile(bytes);
    const imports = WebAssembly.Module.imports(module);
    if (imports.length !== 0) {
        const i = imports[0];
        throw new Error(`module imports ${i.module}.${i.name}; modules are instantiated with no imports`);
    }
    return module;
}

function unpack(v) {
    const u = BigInt.asUintN(64, v);
    return [Number(u >> 32n), Number(u & 0xffffffffn)];
}

/** One fresh instance of a module, for one call. */
class Call {
    constructor(instance) {
        this.e = instance.exports;
        for (const name of ["memory", "gx_abi_version", "gx_alloc", "gx_free", "gx_sources", "gx_compile", "gx_last_error"]) {
            if (!(name in this.e)) throw new Error(`module export ${name} is missing`);
        }
        const version = this.e.gx_abi_version();
        if (version !== 1 && version !== 2) {
            throw new Error(`gx_abi_version returned ${version}, expected 1 or 2`);
        }
        if (version === 2 && !("gx_inputs" in this.e)) throw new Error("module export gx_inputs is missing");
        this.version = version;
    }

    checkMemory() {
        const size = this.e.memory.buffer.byteLength;
        if (size > MEMORY_LIMIT) throw new Error(`module exceeded the memory limit of ${MEMORY_LIMIT} bytes`);
    }

    bytes(ptr, len) {
        if (ptr + len > this.e.memory.buffer.byteLength) {
            throw new Error(`block at ${ptr} of ${len} bytes is outside memory`);
        }
        return new Uint8Array(this.e.memory.buffer, ptr, len).slice();
    }

    write(data) {
        const ptr = this.e.gx_alloc(data.length) >>> 0;
        this.checkMemory();
        if (ptr === 0 && data.length > 0) throw new Error(`gx_alloc(${data.length}) returned 0`);
        if (ptr + data.length > this.e.memory.buffer.byteLength) {
            throw new Error(`gx_alloc(${data.length}) returned ${ptr} outside memory`);
        }
        new Uint8Array(this.e.memory.buffer, ptr, data.length).set(data);
        return [ptr, data.length];
    }

    take(packed) {
        this.checkMemory();
        if (packed === -1n) {
            const [p, l] = unpack(this.e.gx_last_error());
            throw new Error("module failed: " + new TextDecoder().decode(this.bytes(p, l)));
        }
        const [p, l] = unpack(packed);
        const out = this.bytes(p, l);
        this.e.gx_free(p | 0, l | 0);
        return out;
    }
}

async function fresh(module) {
    const instance = await WebAssembly.instantiate(module, {});
    return new Call(instance);
}

/** The count of a frame or input buffer (u32 LE count first); 0 for none. */
function count(bytes) {
    if (!bytes || bytes.length === 0) return 0;
    if (bytes.length < 4) throw new Error(`a list of ${bytes.length} bytes has no count`);
    return new DataView(bytes.buffer, bytes.byteOffset, 4).getUint32(0, true);
}

function noneForV1(bytes, what) {
    const n = count(bytes);
    if (n !== 0) throw new Error(`${n} ${what} given to a version 1 module, which reads none`);
}

/**
 * Runs gx_inputs (version 2): the raw input list bytes for the request and
 * frame bytes. A version 1 module reads no inputs: the empty list.
 */
export async function runInputs(module, request, frames) {
    const call = await fresh(module);
    if (call.version === 1) {
        noneForV1(frames, "frame records");
        return new Uint8Array(4);
    }
    const [rp, rl] = call.write(request);
    const [fp, fl] = call.write(frames ?? new Uint8Array(4));
    return call.take(call.e.gx_inputs(rp, rl, fp, fl));
}

/**
 * Runs gx_sources: the raw source id list bytes (u64 LE values) for the
 * request bytes, and for a version 2 module the frame bytes.
 */
export async function runSources(module, request, frames) {
    const call = await fresh(module);
    const [rp, rl] = call.write(request);
    if (call.version === 1) {
        noneForV1(frames, "frame records");
        return call.take(call.e.gx_sources(rp, rl));
    }
    const [fp, fl] = call.write(frames ?? new Uint8Array(4));
    return call.take(call.e.gx_sources(rp, rl, fp, fl));
}

/**
 * Runs gx_compile: the section bytes for the request and tile bytes, and
 * for a version 2 module the input bytes.
 */
export async function runCompile(module, request, tiles, inputs) {
    const call = await fresh(module);
    const [rp, rl] = call.write(request);
    const [tp, tl] = call.write(tiles);
    if (call.version === 1) {
        noneForV1(inputs, "inputs");
        return call.take(call.e.gx_compile(rp, rl, tp, tl));
    }
    const [ip, il] = call.write(inputs ?? new Uint8Array(4));
    return call.take(call.e.gx_compile(rp, rl, tp, tl, ip, il));
}

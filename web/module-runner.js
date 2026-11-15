// Runs compiler modules with the browser's own WebAssembly, under the module
// ABI v1 (client-compute.md section 4). Used by the client's Web Worker
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
// - gx_abi_version must return 1;
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
        if (version !== 1) throw new Error(`gx_abi_version returned ${version}, expected 1`);
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

/**
 * Runs gx_sources: the raw source id list bytes (u64 LE values) for the
 * request bytes.
 */
export async function runSources(module, request) {
    const call = await fresh(module);
    const [rp, rl] = call.write(request);
    return call.take(call.e.gx_sources(rp, rl));
}

/** Runs gx_compile: the section bytes for the request and tile bytes. */
export async function runCompile(module, request, tiles) {
    const call = await fresh(module);
    const [rp, rl] = call.write(request);
    const [tp, tl] = call.write(tiles);
    return call.take(call.e.gx_compile(rp, rl, tp, tl));
}

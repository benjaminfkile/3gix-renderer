#!/usr/bin/env node
// Measures the browser build's memory while the camera pans (the memory
// report, docs/architecture.md; usage in docs/web.md). Node 22 or later,
// no dependencies.
//
//   node scripts/memory-pan.mjs <url> [--seconds 300] [--wait 20] [--phone]
//                                     [--chrome <path>]
//
// Starts headless Chrome with WebGPU and drives it over the DevTools
// protocol on a pipe (--remote-debugging-pipe, so no port is opened). It
// opens <url>, emulates a phone (390 x 844 CSS pixels, scale 3, touch) with
// --phone or a 1280 x 720 window otherwise, waits --wait seconds, then
// drags 300 pixels across the view every 10 s for --seconds seconds. Every
// 30 s it prints one line: elapsed seconds, the byte length of every
// WebAssembly.Memory alive on the page and in each of its workers (found
// with Runtime.queryObjects on WebAssembly.Memory.prototype), the resident
// size of Chrome's renderer processes (from /proc, Linux only), and the
// latest `gx-memory` object the page logged (a build made with
// `sh web/build.sh --memory-report`). It ends with a forced garbage
// collection in every target and a final line.
//
// The Chrome binary is --chrome, else $CHROME, else google-chrome.

import { spawn } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const USAGE =
    'usage: node scripts/memory-pan.mjs <url> [--seconds 300] [--wait 20] [--phone] [--chrome <path>]';

function parseArgs(argv) {
    const o = { url: null, seconds: 300, wait: 20, phone: false, chrome: process.env.CHROME || 'google-chrome' };
    for (let i = 0; i < argv.length; i++) {
        const a = argv[i];
        const value = () => {
            if (i + 1 >= argv.length) throw new Error(`${a} needs a value\n${USAGE}`);
            return argv[++i];
        };
        if (a === '--seconds') o.seconds = Number(value());
        else if (a === '--wait') o.wait = Number(value());
        else if (a === '--phone') o.phone = true;
        else if (a === '--chrome') o.chrome = value();
        else if (a.startsWith('--')) throw new Error(`unknown option ${a}\n${USAGE}`);
        else if (o.url === null) o.url = a;
        else throw new Error(`one URL only\n${USAGE}`);
    }
    if (o.url === null) throw new Error(USAGE);
    if (!(o.seconds >= 0) || !(o.wait >= 0)) throw new Error(`--seconds and --wait take seconds\n${USAGE}`);
    return o;
}

const PAN_PX = 300;
const PAN_EVERY_S = 10;
const REPORT_EVERY_S = 30;
const VIEW = { width: 1280, height: 720, deviceScaleFactor: 1, mobile: false };
const PHONE = { width: 390, height: 844, deviceScaleFactor: 3, mobile: true };

const sleep = (s) => new Promise((r) => setTimeout(r, s * 1000));

/** The DevTools protocol over Chrome's pipe: JSON messages ending in NUL. */
class Pipe {
    constructor(write, read) {
        this.write = write;
        this.next = 1;
        this.waiting = new Map();
        this.listeners = [];
        let buffer = '';
        read.on('data', (chunk) => {
            buffer += chunk.toString('utf8');
            let end;
            while ((end = buffer.indexOf('\0')) >= 0) {
                const msg = JSON.parse(buffer.slice(0, end));
                buffer = buffer.slice(end + 1);
                this.dispatch(msg);
            }
        });
    }

    dispatch(msg) {
        if (msg.id !== undefined && this.waiting.has(msg.id)) {
            const { resolve, reject } = this.waiting.get(msg.id);
            this.waiting.delete(msg.id);
            if (msg.error) reject(new Error(`${msg.error.message} (${msg.error.code})`));
            else resolve(msg.result);
            return;
        }
        for (const l of this.listeners) l(msg);
    }

    send(method, params = {}, sessionId = undefined) {
        const id = this.next++;
        const msg = { id, method, params };
        if (sessionId) msg.sessionId = sessionId;
        this.write.write(JSON.stringify(msg) + '\0');
        return new Promise((resolve, reject) => this.waiting.set(id, { resolve, reject }));
    }

    on(listener) {
        this.listeners.push(listener);
    }
}

/** Byte lengths of every WebAssembly.Memory alive in the session's target. */
async function wasmMemories(pipe, sessionId) {
    const group = 'memory-pan';
    try {
        const proto = await pipe.send(
            'Runtime.evaluate',
            { expression: 'WebAssembly.Memory.prototype', objectGroup: group },
            sessionId,
        );
        const found = await pipe.send(
            'Runtime.queryObjects',
            { prototypeObjectId: proto.result.objectId, objectGroup: group },
            sessionId,
        );
        const sizes = await pipe.send(
            'Runtime.callFunctionOn',
            {
                objectId: found.objects.objectId,
                functionDeclaration: 'function () { return this.map((m) => m.buffer.byteLength); }',
                returnByValue: true,
            },
            sessionId,
        );
        return sizes.result.value;
    } catch {
        return null;
    } finally {
        pipe.send('Runtime.releaseObjectGroup', { objectGroup: group }, sessionId).catch(() => {});
    }
}

/** Resident bytes of each of Chrome's renderer processes (Linux /proc). */
async function rendererResident(pipe) {
    try {
        const info = await pipe.send('SystemInfo.getProcessInfo');
        return info.processInfo
            .filter((p) => p.type === 'renderer')
            .map((p) => {
                const status = readFileSync(`/proc/${p.id}/status`, 'utf8');
                const kb = /VmRSS:\s+(\d+) kB/.exec(status);
                return kb ? Number(kb[1]) * 1024 : null;
            })
            .filter((v) => v !== null);
    } catch {
        return null;
    }
}

const mb = (b) => (b / 1e6).toFixed(1);

async function main() {
    const o = parseArgs(process.argv.slice(2));
    const profile = mkdtempSync(join(tmpdir(), 'memory-pan-'));
    const chrome = spawn(
        o.chrome,
        [
            '--headless=new',
            '--remote-debugging-pipe',
            `--user-data-dir=${profile}`,
            '--enable-unsafe-webgpu',
            '--enable-features=Vulkan,WebGPU',
            '--no-first-run',
            '--no-default-browser-check',
            '--js-flags=--expose-gc',
            'about:blank',
        ],
        { stdio: ['ignore', 'ignore', 'inherit', 'pipe', 'pipe'] },
    );
    const pipe = new Pipe(chrome.stdio[3], chrome.stdio[4]);
    const done = () => {
        chrome.kill();
        rmSync(profile, { recursive: true, force: true });
    };
    process.on('SIGINT', () => {
        done();
        process.exit(130);
    });

    const workers = new Map();
    let latest = null;
    pipe.on((msg) => {
        if (msg.method === 'Target.attachedToTarget') {
            const { sessionId, targetInfo } = msg.params;
            if (targetInfo.type === 'worker' || targetInfo.type === 'dedicated_worker') {
                workers.set(sessionId, targetInfo.url);
                pipe.send('Runtime.runIfWaitingForDebugger', {}, sessionId).catch(() => {});
            }
        } else if (msg.method === 'Target.detachedFromTarget') {
            workers.delete(msg.params.sessionId);
        } else if (msg.method === 'Runtime.consoleAPICalled') {
            const args = msg.params.args || [];
            if (args[0]?.value === 'gx-memory' && typeof args[1]?.value === 'string') {
                try {
                    latest = JSON.parse(args[1].value);
                } catch {
                    // Not one JSON object: ignored.
                }
            }
        }
    });

    const { targetInfos } = await pipe.send('Target.getTargets');
    const page = targetInfos.find((t) => t.type === 'page');
    const { sessionId } = await pipe.send('Target.attachToTarget', { targetId: page.targetId, flatten: true });
    await pipe.send('Runtime.enable', {}, sessionId);
    await pipe.send('Page.enable', {}, sessionId);
    await pipe.send(
        'Target.setAutoAttach',
        { autoAttach: true, waitForDebuggerOnStart: false, flatten: true },
        sessionId,
    );
    const view = o.phone ? PHONE : VIEW;
    await pipe.send('Emulation.setDeviceMetricsOverride', view, sessionId);
    if (o.phone) {
        await pipe.send('Emulation.setTouchEmulationEnabled', { enabled: true, maxTouchPoints: 5 }, sessionId);
    }
    await pipe.send('Page.navigate', { url: o.url }, sessionId);

    const started = Date.now();
    const elapsed = () => (Date.now() - started) / 1000;
    const line = async (label) => {
        const pageSizes = await wasmMemories(pipe, sessionId);
        const workerSizes = [];
        for (const w of workers.keys()) {
            const sizes = await wasmMemories(pipe, w);
            if (sizes) workerSizes.push(...sizes);
        }
        const resident = await rendererResident(pipe);
        const list = (v) => (v ? `[${v.map(mb).join(', ')}]` : '-');
        console.log(
            `${label} t ${elapsed().toFixed(0)} s` +
                `  wasm page MB ${list(pageSizes)}` +
                `  wasm workers MB ${list(workerSizes)}` +
                `  renderer resident MB ${list(resident)}` +
                `  gx-memory ${latest ? JSON.stringify(latest) : '-'}`,
        );
    };

    const { width, height } = view;
    const drag = async (step) => {
        const y = height / 2;
        const sign = step % 2 === 0 ? 1 : -1;
        const x0 = width / 2 - (sign * PAN_PX) / 2;
        const points = Array.from({ length: 11 }, (_, n) => x0 + (sign * PAN_PX * n) / 10);
        if (o.phone) {
            const touch = (type, x) =>
                pipe.send('Input.dispatchTouchEvent', { type, touchPoints: type === 'touchEnd' ? [] : [{ x, y }] }, sessionId);
            await touch('touchStart', points[0]);
            for (const x of points.slice(1)) await touch('touchMove', x);
            await touch('touchEnd', points[points.length - 1]);
        } else {
            const mouse = (type, x) =>
                pipe.send(
                    'Input.dispatchMouseEvent',
                    { type, x, y, button: 'left', buttons: type === 'mouseReleased' ? 0 : 1, clickCount: 1 },
                    sessionId,
                );
            await mouse('mousePressed', points[0]);
            for (const x of points.slice(1)) await mouse('mouseMoved', x);
            await mouse('mouseReleased', points[points.length - 1]);
        }
    };

    await sleep(o.wait);
    await line('wait');
    const panStart = elapsed();
    let step = 0;
    let lastReport = panStart;
    while (elapsed() - panStart < o.seconds) {
        await drag(step++);
        const until = panStart + step * PAN_EVERY_S;
        while (elapsed() < until && elapsed() - panStart < o.seconds) {
            await sleep(Math.min(1, until - elapsed()));
            if (elapsed() - lastReport >= REPORT_EVERY_S) {
                lastReport = elapsed();
                await line('pan');
            }
        }
    }

    for (const s of [sessionId, ...workers.keys()]) {
        await pipe.send('HeapProfiler.collectGarbage', {}, s).catch(() => {});
    }
    await sleep(2);
    await line('final');
    done();
}

main().catch((e) => {
    console.error(`memory-pan: ${e.message}`);
    process.exit(1);
});

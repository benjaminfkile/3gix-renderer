// Checks that the web binding exports the renderer API under its contract
// names (docs/api.md section 11): every item of sections 3 to 8 is a
// method of the Renderer class in web/pkg/renderer.d.ts (create a static
// one), and every event of section 8 is in its RendererEvent type.
// Fails with the missing names. Run after web/build.sh.
//
// Usage: node scripts/api-surface-check.mjs [renderer.d.ts]

import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const dtsPath = process.argv[2] ?? join(root, "web/pkg/renderer.d.ts");
const doc = readFileSync(join(root, "docs/api.md"), "utf8");
const dts = readFileSync(dtsPath, "utf8");

// The table rows of sections 3 to 8, by section number.
const sections = new Map();
let section = 0;
for (const line of doc.split("\n")) {
    const heading = line.match(/^## (\d+)\./);
    if (heading) {
        section = Number(heading[1]);
        continue;
    }
    if (section < 3 || section > 8 || !line.startsWith("| `")) {
        continue;
    }
    const first = line.split(/(?<!\\)\|/)[1];
    if (!sections.has(section)) {
        sections.set(section, []);
    }
    sections.get(section).push(first);
}

const items = [];
const events = [];
for (const [n, cells] of sections) {
    for (const cell of cells) {
        const codes = [...cell.matchAll(/`([^`]+)`/g)].map((m) => m[1]);
        for (const code of codes) {
            if (n === 8) {
                events.push(code);
            } else {
                for (const m of code.matchAll(/([A-Za-z_][A-Za-z0-9_]*)\(/g)) {
                    items.push(m[1]);
                }
            }
        }
    }
}
// `on` is the events' item (section 8).
items.push("on");

const start = dts.indexOf("export class Renderer {");
if (start < 0) {
    console.error(`${dtsPath}: no exported class Renderer`);
    process.exit(1);
}
const body = dts.slice(start, dts.indexOf("\n}", start));
const methods = new Set(
    [...body.matchAll(/^\s+(?:static\s+)?([A-Za-z_][A-Za-z0-9_]*)\(/gm)].map((m) => m[1]),
);
const eventType = dts.match(/export type RendererEvent\s*=([^;]*);/);
const exportedEvents = new Set(
    eventType ? [...eventType[1].matchAll(/"([^"]+)"/g)].map((m) => m[1]) : [],
);

const missing = [
    ...items.filter((i) => !methods.has(i)).map((i) => `item ${i}`),
    ...events.filter((e) => !exportedEvents.has(e)).map((e) => `event ${e}`),
];
if (!/static create\(/.test(body)) {
    missing.push("item create (static)");
}
if (items.length < 30 || events.length !== 7) {
    console.error(`docs/api.md: read ${items.length} items and ${events.length} events; the tables changed shape`);
    process.exit(1);
}
if (missing.length > 0) {
    console.error(`${dtsPath} does not export: ${missing.join(", ")}`);
    process.exit(1);
}
console.log(`api surface ok: ${items.length} items, ${events.length} events`);

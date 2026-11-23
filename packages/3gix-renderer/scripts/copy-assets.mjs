// Copies the browser build into the package (docs/api.md section 9): the
// wasm-bindgen output of web/build.sh from web/pkg, and the module worker
// with its runner from web/. They go to src/wasm (so tsc sees the
// binding's types) and dist/wasm (shipped, resolved from dist/renderer.js
// with new URL(..., import.meta.url)). Run web/build.sh first.

import { copyFileSync, mkdirSync, existsSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const pkg = join(dirname(fileURLToPath(import.meta.url)), "..");
const web = join(pkg, "../../web");
const files = [
    [join(web, "pkg/renderer.js"), "renderer.js"],
    [join(web, "pkg/renderer.d.ts"), "renderer.d.ts"],
    [join(web, "pkg/renderer_bg.wasm"), "renderer_bg.wasm"],
    [join(web, "module-worker.js"), "module-worker.js"],
    [join(web, "module-runner.js"), "module-runner.js"],
];
for (const dir of ["src/wasm", "dist/wasm"]) {
    mkdirSync(join(pkg, dir), { recursive: true });
    for (const [from, name] of files) {
        if (!existsSync(from)) {
            console.error(`missing ${from}: run sh web/build.sh first`);
            process.exit(1);
        }
        copyFileSync(from, join(pkg, dir, name));
    }
}
console.log(`copied ${files.length} files into src/wasm and dist/wasm`);

// The demo's Vite setup: "3gix-renderer" is the built package (dist), the
// same files `npm pack` ships. The package is not prebundled, so its
// new URL(..., import.meta.url) lookups stay relative to its own files.
import { defineConfig } from "vite";
import { fileURLToPath } from "node:url";

export default defineConfig({
    root: fileURLToPath(new URL(".", import.meta.url)),
    resolve: {
        alias: { "3gix-renderer": fileURLToPath(new URL("../dist/index.js", import.meta.url)) },
    },
    optimizeDeps: { exclude: ["3gix-renderer"] },
    build: { target: "es2022" },
    esbuild: { jsx: "automatic" },
});

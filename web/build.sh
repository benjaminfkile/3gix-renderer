#!/bin/sh
# Builds the browser version into web/pkg: wasm-pack compiles the renderer
# crate for wasm32-unknown-unknown with the web feature and generates the
# JavaScript bindings.
#
# Needs wasm-pack on the PATH. Serve web/ with any
# static file server afterwards, see docs/web.md.
set -eu

cd "$(dirname "$0")/.."

# wasm-pack takes its own options before the crate path's extra cargo
# arguments; --no-opt so wasm-pack does not download and run its own
# wasm-opt (see below).
wasm-pack build crates/renderer --target web --out-dir ../../web/pkg --no-opt --release -- --features web

# No wasm-opt pass: binaryen 108 (the installed one) rewrites the export
# of wasm-bindgen's externref table to point at the function table, so
# initSync fails with "WebAssembly.Table.grow(): failed to grow table"
# (even a plain read and write with no passes does it). The module stays
# as wasm-pack built it, in release; scripts/api-values-check.mjs loads it.

ls -l web/pkg/renderer_bg.wasm web/pkg/renderer.js

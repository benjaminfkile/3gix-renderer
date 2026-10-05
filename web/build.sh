#!/bin/sh
# Builds the browser version into web/pkg: wasm-pack compiles the renderer
# crate for wasm32-unknown-unknown with the web feature and generates the
# JavaScript bindings, then wasm-opt -O2 shrinks the module in place.
#
# Needs wasm-pack and wasm-opt (binaryen) on the PATH. Serve web/ with any
# static file server afterwards, see docs/web.md.
set -eu

cd "$(dirname "$0")/.."

# wasm-pack takes its own options before the crate path's extra cargo
# arguments; --no-opt because wasm-opt runs below with the installed
# binary instead of one wasm-pack would download.
wasm-pack build crates/renderer --target web --out-dir ../../web/pkg --no-opt --release -- --features web

wasm=web/pkg/renderer_bg.wasm
# The wasm features rustc enables by default for wasm32-unknown-unknown,
# so wasm-opt accepts the module and keeps to them.
wasm-opt -O2 \
    --enable-bulk-memory \
    --enable-mutable-globals \
    --enable-nontrapping-float-to-int \
    --enable-sign-ext \
    --enable-reference-types \
    --enable-multivalue \
    "$wasm" -o "$wasm.opt"
mv "$wasm.opt" "$wasm"

ls -l web/pkg/renderer_bg.wasm web/pkg/renderer.js

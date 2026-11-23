#!/bin/sh
# Builds the browser version into web/pkg: wasm-pack compiles the renderer
# crate for wasm32-unknown-unknown with the web feature and generates the
# JavaScript bindings, then wasm-opt -O2 shrinks the module in place when
# the installed binaryen is version 133 or newer.
#
# Needs wasm-pack on the PATH, and wasm-opt (binaryen) for the size pass.
# Serve web/ with any static file server afterwards, see docs/web.md.
set -eu

cd "$(dirname "$0")/.."

# wasm-pack takes its own options before the crate path's extra cargo
# arguments; --no-opt because wasm-opt runs below with the installed
# binary instead of one wasm-pack would download.
wasm-pack build crates/renderer --target web --out-dir ../../web/pkg --no-opt --release -- --features web

# binaryen 108 rewrites the export of wasm-bindgen's externref table to
# point at the function table, so initSync fails with
# "WebAssembly.Table.grow(): failed to grow table" (even a plain read and
# write with no passes does it). Version 133 keeps the module loadable
# (checked by loading it under Node), so older versions skip the pass and
# leave the module as wasm-pack built it, in release, about 1.1 MB larger.
# scripts/api-values-check.mjs loads whichever module this leaves.
MIN_BINARYEN=133
wasm=web/pkg/renderer_bg.wasm
version=$(wasm-opt --version 2>/dev/null | sed -n 's/^wasm-opt version \([0-9]*\).*/\1/p' || true)
if [ -n "$version" ] && [ "$version" -ge "$MIN_BINARYEN" ]; then
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
else
    echo "web/build.sh: wasm-opt ${version:-not found} is older than $MIN_BINARYEN; the module is left unoptimized"
fi

ls -l web/pkg/renderer_bg.wasm web/pkg/renderer.js

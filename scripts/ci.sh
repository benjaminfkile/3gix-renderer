#!/bin/sh
# Full local CI: toolchain versions, format, lint, tests (including the
# headless renders and the volume comparison on whatever wgpu adapter is
# present, a software Vulkan driver is enough), the vocabulary lint, the
# wasm32 builds with the web feature off and on, and the browser package
# built by web/build.sh (wasm-pack and wasm-opt). There is no browser here:
# running the browser build is verified by hand, see docs/web.md.
#
# Set GX_CORE_DIR to a checkout of the core library to also check that the
# vocabulary lint files here are identical to the core library's.
set -eu

cd "$(dirname "$0")/.."

step() {
    printf '\n==> %s\n' "$*"
}

step "1. toolchain versions"
rustc --version
cargo --version

step "2. cargo fmt"
cargo fmt --all -- --check

step "3. cargo clippy"
cargo clippy --workspace --all-targets -- -D warnings

step "4. cargo test"
cargo test --workspace

step "5. vocab-lint"
sh scripts/vocab-lint.sh
if [ -n "${GX_CORE_DIR:-}" ]; then
    diff "$GX_CORE_DIR/scripts/vocab-lint.sh" scripts/vocab-lint.sh
    diff "$GX_CORE_DIR/scripts/vocab-banned.txt" scripts/vocab-banned.txt
    echo "vocab-lint files match the core library"
fi

step "6. cargo build wasm32 (library, web feature off)"
cargo build -p renderer --lib --target wasm32-unknown-unknown

step "7. cargo build and clippy wasm32 (web feature on)"
cargo build --target wasm32-unknown-unknown --features web -p renderer
cargo clippy -p renderer --target wasm32-unknown-unknown --features web -- -D warnings

step "8. web/build.sh"
rm -f web/pkg/renderer_bg.wasm web/pkg/renderer.js
sh web/build.sh
for f in web/pkg/renderer_bg.wasm web/pkg/renderer.js; do
    if [ ! -s "$f" ]; then
        echo "missing $f" >&2
        exit 1
    fi
done
echo "browser package ok"

printf '\nci ok\n'

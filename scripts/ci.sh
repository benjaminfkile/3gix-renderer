#!/bin/sh
# Full local CI: toolchain versions, format, lint, tests (including the
# headless render on whatever wgpu adapter is present, a software Vulkan
# driver is enough), the vocabulary lint, and the wasm32 build of the
# library crate with the web feature off.
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

printf '\nci ok\n'

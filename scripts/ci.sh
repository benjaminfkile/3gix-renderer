#!/bin/sh
# Full local CI: toolchain versions, format, lint, tests (including the
# headless renders and the volume comparison on whatever wgpu adapter is
# present, a software Vulkan driver is enough), the vocabulary lint
# (banned words and banned numbers) and its own test, the
# browser worker's module runner under Node (V8, standing in for the
# browser) against the example module's conformance hashes, the browser's
# view options from config.js and the query string, the
# wasm32 builds with the web feature off and on, the memory-report
# feature's clippy and tests (natively and for the browser), and the browser package
# built by web/build.sh (wasm-pack), then the JS package
# (packages/3gix-renderer: npm ci, its build, tsc over the package and the
# demo, the React component's unit tests, and the npm pack file list), then
# the web binding's and the package's surface against docs/api.md and the
# binding's value conversions under Node. There is no browser here:
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

step "4a. memory-report feature: clippy and tests (docs/architecture.md)"
cargo clippy -p renderer --all-targets --features memory-report -- -D warnings
cargo test -p renderer --features memory-report
cargo clippy -p renderer --target wasm32-unknown-unknown --features web,memory-report -- -D warnings

step "4b. budget test (release, rendering.md section 5.3)"
cargo test --release -p renderer --test budget

step "5. vocab-lint, banned words and banned numbers"
sh scripts/vocab-lint.sh

step "5a. vocab-lint finds a banned number and honors only a measured reference in tests/"
sh scripts/vocab-lint-test.sh
if [ -n "${GX_CORE_DIR:-}" ]; then
    diff "$GX_CORE_DIR/scripts/vocab-lint.sh" scripts/vocab-lint.sh
    diff "$GX_CORE_DIR/scripts/vocab-banned.txt" scripts/vocab-banned.txt
    diff "$GX_CORE_DIR/scripts/vocab-numbers.txt" scripts/vocab-numbers.txt
    echo "vocab-lint files match the core library"
fi

step "6. browser worker under Node"
node --version
node scripts/worker-node-check.mjs
node scripts/web-options-check.mjs

step "7. cargo build wasm32 (library, web feature off)"
cargo build -p renderer --lib --target wasm32-unknown-unknown

step "8. cargo build and clippy wasm32 (web feature on)"
cargo build --target wasm32-unknown-unknown --features web -p renderer
cargo clippy -p renderer --target wasm32-unknown-unknown --features web -- -D warnings

step "9. web/build.sh"
rm -f web/pkg/renderer_bg.wasm web/pkg/renderer.js
sh web/build.sh
for f in web/pkg/renderer_bg.wasm web/pkg/renderer.js; do
    if [ ! -s "$f" ]; then
        echo "missing $f" >&2
        exit 1
    fi
done
echo "browser package ok"

step "10. packages/3gix-renderer"
(
    cd packages/3gix-renderer
    npm ci --no-audit --no-fund
    npm run build
    npm run typecheck
    npm test
    list=$(npm pack --dry-run 2>&1)
    printf '%s\n' "$list"
    for f in dist/wasm/renderer_bg.wasm dist/wasm/module-worker.js dist/wasm/module-runner.js; do
        if ! printf '%s\n' "$list" | grep -q " $f\$"; then
            echo "npm pack does not list $f" >&2
            exit 1
        fi
    done
    echo "package ok"
)

step "11. web binding and package: API surface and value conversions"
node scripts/api-surface-check.mjs
node scripts/api-values-check.mjs

printf '\nci ok\n'

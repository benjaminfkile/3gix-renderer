#!/bin/sh
# Rebuilds the core library's example modules at the core revision pinned in
# Cargo.lock and copies each, with its conformance vectors, into testdata:
# gx-module-test (ABI v1, conformance/module/) into testdata/example-module/,
# gx-module-reader (ABI v2, conformance/module-v2/) into
# testdata/example-module-v2/, and gx-module-overlay (an overlay layer's
# module, conformance/module-overlay/) into testdata/example-module-overlay/.
#
# The worker tests, the mock hub, and scripts/worker-node-check.mjs run this
# module. Rerun this script after moving the gx-core pin; the worker's
# determinism test fails until the vendored vectors match the module.
set -eu

cd "$(dirname "$0")/.."

core=$(cargo metadata --format-version 1 --locked | node -e '
let s = "";
process.stdin.on("data", (d) => (s += d));
process.stdin.on("end", () => {
  const p = JSON.parse(s).packages.find((p) => p.name === "gx-core");
  process.stdout.write(p.manifest_path.replace(/\/crates\/gx-core\/Cargo\.toml$/, ""));
});')
echo "core checkout: $core"

target="$(pwd)/target/example-module"
# The core checkout is read-only; build into our own target directory.
(cd "$core" && cargo build -q -p gx-module-test -p gx-module-reader -p gx-module-overlay --release \
    --target wasm32-unknown-unknown --target-dir "$target")

out=testdata/example-module
rm -rf "$out"
mkdir -p "$out/vectors"
cp "$target/wasm32-unknown-unknown/release/gx_module_test.wasm" "$out/module.wasm"
cp "$core"/conformance/module/* "$out/vectors/"
ls -l "$out/module.wasm"

out=testdata/example-module-v2
rm -rf "$out"
mkdir -p "$out/vectors"
cp "$target/wasm32-unknown-unknown/release/gx_module_reader.wasm" "$out/module.wasm"
cp "$core"/conformance/module-v2/* "$out/vectors/"
ls -l "$out/module.wasm"

out=testdata/example-module-overlay
rm -rf "$out"
mkdir -p "$out/vectors"
cp "$target/wasm32-unknown-unknown/release/gx_module_overlay.wasm" "$out/module.wasm"
cp "$core"/conformance/module-overlay/* "$out/vectors/"
ls -l "$out/module.wasm"

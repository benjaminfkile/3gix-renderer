#!/bin/sh
# Rebuilds the core library's example module (gx-module-test) at the core
# revision pinned in Cargo.lock and copies it, with its conformance vectors
# (conformance/module/), into testdata/example-module/.
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
(cd "$core" && cargo build -q -p gx-module-test --release \
    --target wasm32-unknown-unknown --target-dir "$target")

out=testdata/example-module
rm -rf "$out"
mkdir -p "$out/vectors"
cp "$target/wasm32-unknown-unknown/release/gx_module_test.wasm" "$out/module.wasm"
cp "$core"/conformance/module/* "$out/vectors/"
ls -l "$out/module.wasm"

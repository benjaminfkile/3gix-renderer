#!/bin/sh
# Tests scripts/vocab-lint.sh on a throwaway git work tree under target/:
# a file with a banned number fails; the same file under tests/ with the
# measured-reference marker in its first comment block passes and is
# printed; the marker in a file under src/, or later in a file under tests/,
# fails.
set -eu

cd "$(dirname "$0")/.."
here=$(pwd)
dir=target/vocab-lint-test
rm -rf "$dir"
mkdir -p "$dir/scripts"
cp scripts/vocab-banned.txt scripts/vocab-numbers.txt "$dir/scripts/"
git -C "$dir" init -q

lint() {
    sh "$here/scripts/vocab-lint.sh" "$dir" > "$dir.out" 2>&1
}

fail() {
    echo "vocab-lint-test: $*" >&2
    cat "$dir.out" >&2
    exit 1
}

number=$(head -n 1 scripts/vocab-numbers.txt)
body="const R: f64 = $number;"
marker="//! measured-reference: a test of the lint, not a law."

# 1. A banned number in a plain file fails.
mkdir -p "$dir/src"
printf '//! A file.\n%s\n' "$body" > "$dir/src/a.rs"
if lint; then fail "a banned number in src/ passed"; fi
grep -q '^src/a.rs:2$' "$dir.out" || fail "the finding is not reported"
rm "$dir/src/a.rs"

# 2. The same file under tests/ with the marker passes and is printed.
mkdir -p "$dir/tests"
printf '//! A file.\n%s\n%s\n' "$marker" "$body" > "$dir/tests/a.rs"
lint || fail "a measured reference under tests/ failed"
grep -q '^vocab-lint: measured reference: tests/a.rs$' "$dir.out" ||
    fail "the measured reference is not printed"

# 3. The marker in a file under src/ fails, even with nothing banned.
printf '//! A file.\n%s\nconst R: f64 = 1.0;\n' "$marker" > "$dir/src/b.rs"
if lint; then fail "the marker in src/ passed"; fi
rm "$dir/src/b.rs"

# 4. The marker after the first comment block fails, and does not exempt.
printf '//! A file.\n\n%s\n%s\n' "$marker" "$body" > "$dir/tests/a.rs"
if lint; then fail "the marker after the first comment block passed"; fi
grep -q '^tests/a.rs:4$' "$dir.out" || fail "the late marker exempted the file"

rm -rf "$dir" "$dir.out"
echo "vocab-lint-test ok"

#!/bin/sh
# The performance budget on a real stack (rendering.md section 5.3): runs the
# benchmark views B1 to B4 headless, each with a fresh cache directory, in
# real time, then checks their statistics against rendering.md section 5.2
# and the committed baseline perf/baseline.json (tools/perf-budget). Prints
# one line per view and one overall line for the commit message, and exits
# non-zero on any breach.
#
# The hub comes from the environment like the renderer's own settings:
# GX_HUB_URL, GX_API_KEY, GX_SPACE_ID (and GX_BUILD_ID if set). The views'
# frame and direction are picked by rule from whatever universe the hub
# serves (rendering.md section 5.1): `gx-renderer --print-benchmark-start`
# runs first, untimed, with its own fresh cache, and prints
# "<frame>:<x>,<y>,<z>"; the first line of the output names it so the run
# can be reproduced. The script names no frame or place.
#
# Usage: sh scripts/perf-budget.sh [--keep <dir>] [--baseline <file>]
#   --keep <dir>      write the statistics and screenshots there and keep them
#   --baseline <file> compare with this baseline instead of perf/baseline.json
set -eu

cd "$(dirname "$0")/.."

keep=
baseline=perf/baseline.json
while [ $# -gt 0 ]; do
    case "$1" in
        --keep) keep=$2; shift ;;
        --baseline) baseline=$2; shift ;;
        -h|--help) sed -n '2,19p' "$0"; exit 0 ;;
        *) echo "perf-budget: unknown argument $1" >&2; exit 2 ;;
    esac
    shift
done

: "${GX_HUB_URL:?set GX_HUB_URL}" "${GX_API_KEY:?set GX_API_KEY}" "${GX_SPACE_ID:?set GX_SPACE_ID}"

cargo build --release --bin gx-renderer -p renderer
cargo build --release -p perf-budget

if [ -n "$keep" ]; then
    out=$keep
    mkdir -p "$out"
else
    out=$(mktemp -d)
    trap 'rm -rf "$out"' EXIT HUP INT TERM
fi

# rendering.md section 5.1: the start picked by rule, not timed, from its
# own fresh cache. It is a frame and a unit vector in its axes; @+<m> is a
# height above whatever surface is drawn on that line.
rm -rf "$out/cache-pick"
mkdir -p "$out/cache-pick"
if ! dir=$(./target/release/gx-renderer --print-benchmark-start \
    --cache-dir "$out/cache-pick" 2>"$out/pick.log"); then
    cat "$out/pick.log" >&2
    echo "perf-budget: picking the benchmark start failed" >&2
    exit 1
fi
rm -rf "$out/cache-pick"
echo "perf-budget: start $dir"
view_args() {
    case "$1" in
        B1) echo "--start-on $dir@+10000^20" ;;
        B2) echo "--start-on $dir@+10000^90" ;;
        B3) echo "--start-on $dir@+400000^30" ;;
        B4) echo "--view home" ;;
    esac
}

for v in B1 B2 B3 B4; do
    rm -rf "$out/cache-$v"
    mkdir -p "$out/cache-$v"
    # A run that does not settle in time still writes its statistics, and the
    # check below reports it; its exit status is not the verdict.
    # shellcheck disable=SC2046
    ./target/release/gx-renderer --headless --width 1280 --height 720 --time-scale 1 \
        $(view_args "$v") --cache-dir "$out/cache-$v" --wait-ready-seconds 60 \
        --screenshot "$out/$v.png" --stats-json "$out/$v.json" \
        >"$out/$v.log" 2>&1 || true
    rm -rf "$out/cache-$v"
done

./target/release/perf-budget "$out" --baseline "$baseline"

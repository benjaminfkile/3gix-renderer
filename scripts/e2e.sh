#!/bin/sh
# End to end run of the renderer against the real hub and the real system
# compiler, inside one container, with assertions on the screenshots.
#
#   1. Copies the hub repository to a writable place and starts it with its
#      own container harness (scripts/e2e/hub-up.sh in the hub: Redis, a
#      SeaweedFS S3 gateway, Postgres, the hub, and its seed), then sources
#      the connection settings it writes.
#   2. Clones the system compiler, builds it in release mode, and runs its
#      `serve` daemon in the background with the compiler credentials from
#      the settings.
#   3. Builds this renderer and tools/png-stats in release mode.
#   4. Takes four headless screenshots at 640 x 360, matter only (no overlay
#      text, markers, or lines), each waiting until every cell it selected
#      is ready (--wait-ready-seconds) and writing its overlay statistics
#      (--stats-json):
#        home.png               the Home view at launch time
#        frame-3.png            the number key 3 view at launch time
#        frame-3-plus-90d.png   the same view 90 days after the epoch
#        frame-1-close.png      the number key 1 view at a quarter of its
#                               distance
#   5. Asserts on the images with png-stats (see tools/png-stats).
#   6. Prints the statistics of every screenshot.
#   7. Stops the compiler and the hub.
#
# Prints one line per assertion and exits 0 only if every one passed. The
# run is recorded in docs/e2e/README.md.
#
# Usage: sh scripts/e2e.sh
#
# Environment (all optional):
#   GX_E2E_HUB_SRC       hub repository to copy  (default /context/3GIXHub)
#   GX_E2E_HUB_DIR       writable copy of the hub (default /tmp/hub)
#   GX_E2E_COMPILER_DIR  compiler clone           (default /tmp/compiler)
#   GX_E2E_OUT           where the screenshots and statistics go
#                        (default docs/e2e in this repository)
#   GX_E2E_KEEP_STATE    1 keeps the hub database and object store from a
#                        previous run; by default both are reset, so every
#                        chunk is compiled during this run
#   GX_E2E_WAIT          seconds each screenshot waits for its cells
#                        (default 300)
set -u

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
HUB_SRC=${GX_E2E_HUB_SRC:-/context/3GIXHub}
HUB_DIR=${GX_E2E_HUB_DIR:-/tmp/hub}
COMPILER_DIR=${GX_E2E_COMPILER_DIR:-/tmp/compiler}
COMPILER_REPO=https://github.com/benjaminfkile/3gix-system-compiler
COMPILER_BRANCH=grunt
OUT=${GX_E2E_OUT:-$ROOT_DIR/docs/e2e}
WAIT=${GX_E2E_WAIT:-300}
ENV_FILE=/tmp/gx-e2e.env
DB=space_runtime
WEED_DIR=/tmp/weed
WORK=$(mktemp -d /tmp/gx-renderer-e2e.XXXXXX)
COMPILER_LOG=$WORK/compiler.log
RENDERER=$ROOT_DIR/target/release/gx-renderer
PNG_STATS=$ROOT_DIR/target/release/png-stats

# Screenshot size, and the simulation time offset of the second view of
# frame 3: 90 days of 86400 s.
WIDTH=640
HEIGHT=360
NINETY_DAYS=7776000

PGHOST=${PGHOST:-127.0.0.1}
PGPORT=${PGPORT:-5432}
PGUSER=${PGUSER:-postgres}
export PGHOST PGPORT PGUSER

passed=0
failed=0
compiler_pid=

pass() {
    passed=$((passed + 1))
    echo "PASS $*"
}
fail() {
    failed=$((failed + 1))
    echo "FAIL $*"
}

# Stops on a setup failure: no assertion can run after it.
die() {
    echo "error: $*" >&2
    fail "setup: $*"
    finish
}

stop_compiler() {
    if [ -n "$compiler_pid" ] && kill -0 "$compiler_pid" 2>/dev/null; then
        kill "$compiler_pid" 2>/dev/null
        i=0
        while kill -0 "$compiler_pid" 2>/dev/null && [ "$i" -lt 20 ]; do
            i=$((i + 1))
            sleep 1
        done
        kill -9 "$compiler_pid" 2>/dev/null || true
        wait "$compiler_pid" 2>/dev/null
        echo "compiler: stopped (pid $compiler_pid)"
    fi
    compiler_pid=
}

# Stops the compiler and the hub, prints the totals, and exits.
finish() {
    trap - EXIT INT TERM
    stop_compiler
    if [ -f "$COMPILER_LOG" ]; then
        echo "compiler log: $(grep -c 'section stored' "$COMPILER_LOG") sections stored, last lines:"
        tail -n 5 "$COMPILER_LOG"
    fi
    if [ -f "$HUB_DIR/scripts/e2e/hub-down.sh" ]; then
        sh "$HUB_DIR/scripts/e2e/hub-down.sh"
    fi
    rm -rf "$WORK"
    echo "e2e: $passed passed, $failed failed"
    if [ "$failed" -eq 0 ] && [ "$passed" -gt 0 ]; then
        exit 0
    fi
    exit 1
}
trap finish EXIT
trap 'echo "interrupted" >&2; failed=$((failed + 1)); finish' INT TERM

for cmd in curl jq psql git cargo awk; do
    command -v "$cmd" >/dev/null 2>&1 || die "required command not found: $cmd"
done

# 1. The hub: a fresh writable copy, then its harness.
rm -rf "$HUB_DIR"
cp -r "$HUB_SRC" "$HUB_DIR" || die "could not copy $HUB_SRC to $HUB_DIR"
echo "hub: copied $HUB_SRC to $HUB_DIR"
hub_commit=$(git -C "$HUB_SRC" rev-parse HEAD 2>/dev/null || echo unknown)

# A fresh database and object store, so every chunk is compiled during this
# run. Only when nothing is running: services the harness reuses keep their
# state.
hub_up=$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 "http://127.0.0.1:5297/api/health")
s3_up=$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 "http://127.0.0.1:8333/")
if [ "${GX_E2E_KEEP_STATE:-0}" = "1" ]; then
    echo "state: kept (GX_E2E_KEEP_STATE=1)"
elif [ "$hub_up" != "000" ] || [ "$s3_up" != "000" ]; then
    echo "state: hub or object store already running, state kept"
else
    psql -X -q -d postgres -c "DROP DATABASE IF EXISTS $DB" >/dev/null || die "could not drop $DB"
    rm -rf "$WEED_DIR"
    echo "state: reset database $DB and $WEED_DIR"
fi

sh "$HUB_DIR/scripts/e2e/hub-up.sh" || die "hub-up.sh failed"
[ -f "$ENV_FILE" ] || die "$ENV_FILE not written by hub-up.sh"
set -a
. "$ENV_FILE"
set +a

# 2. The compiler: a fresh clone, a release build, serve in the background.
rm -rf "$COMPILER_DIR"
git clone --quiet --depth 1 --branch "$COMPILER_BRANCH" "$COMPILER_REPO" "$COMPILER_DIR" ||
    die "could not clone the compiler"
compiler_commit=$(git -C "$COMPILER_DIR" rev-parse HEAD)
echo "compiler: cloned $COMPILER_BRANCH at $compiler_commit, building (release)"
(cd "$COMPILER_DIR" && cargo build --release --quiet -p system-compiler) || die "compiler build failed"
"$COMPILER_DIR/target/release/system-compiler" serve >"$COMPILER_LOG" 2>&1 &
compiler_pid=$!
# The compiler's own log is the only signal that its socket is up.
i=0
until grep -q "connected to hub" "$COMPILER_LOG" 2>/dev/null; do
    i=$((i + 1))
    if [ "$i" -gt 60 ] || ! kill -0 "$compiler_pid" 2>/dev/null; then
        cat "$COMPILER_LOG" >&2
        die "compiler did not connect within 60 s"
    fi
    sleep 1
done
echo "compiler: connected to the hub (pid $compiler_pid)"

# 3. The renderer and the image statistics tool.
echo "renderer: building (release)"
(cd "$ROOT_DIR" && cargo build --release --quiet -p renderer --bin gx-renderer -p png-stats) ||
    die "renderer build failed"
core_rev=$(sed -n 's|^source = "git+https://github.com/benjaminfkile/3gix-core?branch=grunt#\(.*\)"|\1|p' \
    "$ROOT_DIR/Cargo.lock" | head -n 1)
renderer_rev=$(git -C "$ROOT_DIR" rev-parse HEAD 2>/dev/null || echo unknown)

# 4. The screenshots. shoot NAME FLAGS...: one headless screenshot into
#    OUT/NAME.png with its statistics in OUT/NAME.json.
mkdir -p "$OUT"
shoot() {
    name=$1
    shift
    rm -f "$OUT/$name.png" "$OUT/$name.json"
    echo "screenshot: $name ($*)"
    if "$RENDERER" --screenshot "$OUT/$name.png" --stats-json "$OUT/$name.json" \
        --width "$WIDTH" --height "$HEIGHT" --no-overlay --wait-ready-seconds "$WAIT" \
        "$@" >"$WORK/$name.log" 2>&1; then
        pass "screenshot $name.png: every selected cell ready"
    else
        tail -n 5 "$WORK/$name.log"
        fail "screenshot $name.png: gx-renderer exited non-zero"
    fi
}
shoot home --view home
shoot frame-3 --view 3
shoot frame-3-plus-90d --view 3 --start-offset-seconds "$NINETY_DAYS"
shoot frame-1-close --view 1 --view-distance-scale 0.25

# 5. Assertions on the images. check COMMAND...: runs png-stats, which
#    prints its own PASS or FAIL line.
check() {
    if "$PNG_STATS" "$@"; then
        passed=$((passed + 1))
    else
        failed=$((failed + 1))
    fi
}
cd "$OUT" || die "cannot enter $OUT"
# At least 9 separate bright blobs.
check blobs home.png --threshold 32 --min-count 9
# One large lit disc, 5 to 60 percent of the image, leaning to one side:
# its brighter half at least 1.2 times as bright as its darker half (a disc
# lit face on has no lean and a ratio of 1).
check disc frame-3.png --threshold 32 --min-percent 5 --max-percent 60 --min-side-ratio 1.2
check disc frame-3-plus-90d.png --threshold 32 --min-percent 5 --max-percent 60 --min-side-ratio 1.2
# Time advanced the frames: the lit pixels moved by at least 16 px.
check moved frame-3.png frame-3-plus-90d.png --threshold 32 --min-shift 16
# A smooth falloff: at least 32 levels of range, no neighbor step above a
# quarter of it (a hard edge steps the whole range at once).
check smooth frame-1-close.png --min-range 32 --max-step-fraction 0.25
# Each screenshot stays under 300 KB.
for f in home frame-3 frame-3-plus-90d frame-1-close; do
    [ -f "$f.png" ] || continue
    size=$(wc -c <"$f.png" | tr -d ' ')
    if [ "$size" -lt 307200 ]; then
        pass "size $f.png: $size bytes (limit 307200)"
    else
        fail "size $f.png: $size bytes (limit 307200)"
    fi
done
cd "$ROOT_DIR" || die "cannot enter $ROOT_DIR"

# 6. Statistics and versions for docs/e2e/README.md.
echo "versions: hub $hub_commit, compiler $compiler_commit, core $core_rev, renderer $renderer_rev"
echo "stats: name | selected | pinned | requests | fetched | bytes | first rtt s | last rtt s | streamed s"
for f in home frame-3 frame-3-plus-90d frame-1-close; do
    [ -f "$OUT/$f.json" ] || continue
    jq -r --arg n "$f" '[$n, .cells_selected, .cells_pinned, .requests, .cells_fetched,
        .bytes_fetched, .first_round_trip_seconds, .last_round_trip_seconds, .stream_seconds]
        | map(tostring) | join(" | ")' "$OUT/$f.json" | sed 's/^/stats: /'
done

# Where every frame of the Home view projects to, from the registry the hub
# served (positions at the epoch, a child's added to its parent's): the
# Home camera sits on the root +z axis at h = 1.2 R / tan(30 deg), with R
# the largest |origin| + root_extent / 8 over the non-root frames, looking
# down -z with +y up, so a frame at (x, y, z) projects to
# (x, -y) / (h - z) * f pixels from the image center, with
# f = HEIGHT / (2 tan(30 deg)). Printed as evidence for the blob count.
curl -s -H "X-API-Key: $GX_API_KEY" -o "$WORK/registry.bin" \
    "$GX_HUB_URL/space/$GX_SPACE_ID/build/$GX_BUILD_ID/chunk/registry"
if "$COMPILER_DIR/target/release/system-compiler" describe --key registry \
    --from-file "$WORK/registry.bin" --container >"$WORK/registry.json" 2>/dev/null; then
    jq -r '.sections[0].frames[] | [.frame_id, (.parent_frame_id // "none"), .position_m[0],
        .position_m[1], .position_m[2], .root_extent_m] | map(tostring) | join(" ")' \
        "$WORK/registry.json" >"$WORK/frames.txt"
    awk -v height="$HEIGHT" '
        { n++; id[n] = $1; parent[n] = $2; x[n] = $3; y[n] = $4; z[n] = $5; e[n] = $6; row[$1] = n }
        END {
            # Registry order lists parents first, so one pass adds them up.
            for (i = 1; i <= n; i++) if (parent[i] != "none") {
                p = row[parent[i]]; x[i] += x[p]; y[i] += y[p]; z[i] += z[p]
            }
            for (i = 1; i <= n; i++) if (parent[i] != "none") {
                r = sqrt(x[i] * x[i] + y[i] * y[i] + z[i] * z[i]) + e[i] / 8; if (r > R) R = r
            }
            t = sin(atan2(1, 1) * 4 / 6) / cos(atan2(1, 1) * 4 / 6)
            h = 1.2 * R / t; f = height / (2 * t)
            for (i = 1; i <= n; i++) if (parent[i] != "none") {
                sx = x[i] / (h - z[i]) * f; sy = -y[i] / (h - z[i]) * f
                printf "home: frame %s projects to (%+.2f, %+.2f) px from the image center, %.2f px away\n",
                    id[i], sx, sy, sqrt(sx * sx + sy * sy)
            }
        }' "$WORK/frames.txt"
fi

finish

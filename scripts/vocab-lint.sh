#!/bin/sh
# Fails if any tracked file (or untracked file not ignored by git) contains a
# banned word from scripts/vocab-banned.txt, in its content or its name, or a
# banned number from scripts/vocab-numbers.txt in its content.
#
# Words match case-insensitively on whole words. Numbers match as fixed
# strings on whole tokens (grep -F -w). Cargo.lock is skipped: it holds
# third-party package names, not this repository's words. The two list files
# are skipped too.
#
# The one exception (the purity rule's measured references): a file under a
# tests/ directory whose first comment block has a line whose text starts
# with the marker below is skipped by both checks and printed as
# "vocab-lint: measured reference: <path>". A marker line anywhere else, in
# a file outside tests/ or after the first comment block, is a finding.
#
# With an argument, the lint runs in that directory instead of the
# repository root (scripts/ci.sh uses it to test the lint itself); the
# directory must be a git work tree with the two list files under scripts/.
set -eu
# Byte-wise matching: binary vectors must not trip a locale's multibyte rules.
export LC_ALL=C

if [ $# -gt 0 ]; then
    cd "$1"
fi
root=$(git rev-parse --show-toplevel)
cd "$root"

words=scripts/vocab-banned.txt
numbers=scripts/vocab-numbers.txt
marker='measured-reference:'
pattern=$(grep -v '^[[:space:]]*$' "$words" | paste -sd '|' -)

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT HUP INT TERM

# Files deleted from the work tree but still in the index are not checked.
git ls-files --cached --others --exclude-standard -- . ':!Cargo.lock' ':!**/Cargo.lock' |
    grep -v -x -F -e "$words" -e "$numbers" |
    while IFS= read -r f; do [ -f "$f" ] && echo "$f"; done > "$tmp/files" || true
grep -v '^[[:space:]]*$' "$numbers" > "$tmp/numbers"

# A comment line: optional space, then //, //!, ///, #, /*, *, or <!--.
# awk reads it from the environment, not -v, so its backslashes stay as
# written (-v would turn \* into *, which macOS awk rejects).
comment='^[[:space:]]*(//[/!]?|#|/\*+|\*+|<!--)'

# Files whose first comment block carries the marker. The block is the run
# of comment lines at the top of the file (a #! line counts as one).
: > "$tmp/exempt"
while IFS= read -r f; do
    if C="$comment" M="$marker" awk '
        BEGIN { c = ENVIRON["C"]; m = ENVIRON["M"] }
        $0 !~ c { exit }
        { t = $0; sub(c "[[:space:]]*", "", t); if (index(t, m) == 1) { found = 1; exit } }
        END { exit found ? 0 : 1 }' "$f"; then
        case "/$f" in
        */tests/*) echo "$f" >> "$tmp/exempt" ;;
        *) echo "$f: measured-reference marker outside a tests/ directory" >> "$tmp/found" ;;
        esac
    fi
done < "$tmp/files"

grep -v -x -F -f "$tmp/exempt" "$tmp/files" > "$tmp/checked" || true
if [ ! -s "$tmp/exempt" ]; then
    cp "$tmp/files" "$tmp/checked"
fi

tr '\n' '\0' < "$tmp/checked" | xargs -0 -r grep -I -n -i -w -E -H -- "$pattern" >> "$tmp/found" || true
tr '\n' '\0' < "$tmp/checked" | xargs -0 -r grep -I -n -F -w -H -f "$tmp/numbers" >> "$tmp/found" || true

# A marker line anywhere but the first comment block of an exempt file.
tr '\n' '\0' < "$tmp/files" | C="$comment" M="$marker" EX="$tmp/exempt" xargs -0 -r awk '
    BEGIN { c = ENVIRON["C"]; m = ENVIRON["M"]; ex = ENVIRON["EX"]; while ((getline l < ex) > 0) exempt[l] = 1 }
    FNR == 1 { top = 1 }
    {
        t = $0
        if (top && $0 !~ c) top = 0
        if (t ~ c) sub(c "[[:space:]]*", "", t); else sub("^[[:space:]]*", "", t)
        if (index(t, m) == 1 && !(top && (FILENAME in exempt)))
            printf "%s:%d: measured-reference marker out of place\n", FILENAME, FNR
    }' >> "$tmp/found" 2>/dev/null || true

# File names count too.
grep -i -w -E -- "$pattern" "$tmp/files" | sed 's/$/: banned word in file name/' >> "$tmp/found" || true

while IFS= read -r f; do
    echo "vocab-lint: measured reference: $f"
done < "$tmp/exempt"

if [ -s "$tmp/found" ]; then
    sort -u "$tmp/found" | cut -d: -f1,2
    echo "vocab-lint: banned words or numbers found" >&2
    exit 1
fi
echo "vocab-lint ok"

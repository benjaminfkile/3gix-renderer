#!/bin/sh
# Fails if any tracked file (or untracked file not ignored by git) contains a
# banned word from scripts/vocab-banned.txt, in its content or its name.
# Matching is case-insensitive and on whole words only.
set -eu

root=$(git rev-parse --show-toplevel)
cd "$root"

list=scripts/vocab-banned.txt
pattern=$(grep -v '^[[:space:]]*$' "$list" | paste -sd '|' -)

tmp=$(mktemp)
trap 'rm -f "$tmp"' EXIT HUP INT TERM

git ls-files -z --cached --others --exclude-standard | xargs -0 -r grep -I -n -i -w -E -H -- "$pattern" | grep -v "^$list:" > "$tmp" || true

# File names count too.
git ls-files --cached --others --exclude-standard | grep -i -w -E -- "$pattern" | grep -v "^$list\$" | sed 's/$/: banned word in file name/' >> "$tmp" || true

if [ -s "$tmp" ]; then
    cut -d: -f1,2 "$tmp"
    echo "vocab-lint: banned words found" >&2
    exit 1
fi
echo "vocab-lint ok"

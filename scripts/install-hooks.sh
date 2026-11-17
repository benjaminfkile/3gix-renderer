#!/bin/sh
# Installs the repository's git hooks for the clone this script runs in, by
# pointing core.hooksPath at scripts/hooks (rendering.md section 6 item 1).
# Local to this clone: it changes no global or system git configuration.
# Run it once after cloning: sh scripts/install-hooks.sh
set -eu

root=$(git rev-parse --show-toplevel)
cd "$root"

chmod +x scripts/hooks/*
git config --local core.hooksPath scripts/hooks
echo "install-hooks: core.hooksPath is $(git config --local core.hooksPath)"

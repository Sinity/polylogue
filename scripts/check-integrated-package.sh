#!/usr/bin/env bash
# Called after CircleCI serial-group admission: obsolete candidates do no build work.
set -euo pipefail
result=${1:?supply a qualification-result path}
revision=$(git rev-parse HEAD)
printf 'revision=%s\noutcome=pending\n' "$revision" > "$result"
git fetch --no-tags origin master
latest=$(git rev-parse FETCH_HEAD)
if [[ "$revision" != "$latest" ]]; then
    printf 'revision=%s\noutcome=superseded\nlatest=%s\n' "$revision" "$latest" > "$result"
    echo "Superseded by $latest; no package qualification executed."
    exit 0
fi
if nix flake check --print-build-logs; then
    printf 'revision=%s\noutcome=passed\n' "$revision" > "$result"
else
    code=$?
    printf 'revision=%s\noutcome=failed\nexit_code=%s\n' "$revision" "$code" > "$result"
    exit "$code"
fi

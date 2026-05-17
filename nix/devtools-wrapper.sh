#!/usr/bin/env bash
# devtools CLI wrapper.
#
# Resolves the polylogue checkout from the caller's cwd (via `git
# rev-parse --show-toplevel`) so the wrapper is worktree-aware: running
# `devtools` from inside a worktree operates on that worktree, not on
# whichever checkout's devshell happened to export POLYLOGUE_REPO_ROOT
# first.
#
# Resolution order:
#   1. git rev-parse --show-toplevel from $PWD (if that root has
#      devtools/__main__.py)
#   2. $POLYLOGUE_REPO_ROOT (if set and pointing at a real checkout) —
#      fallback for callers outside any git checkout (e.g. tarball
#      extraction) that still set the env var explicitly
#   3. error out with a clear message
#
# Owned by issue #1209; tested in tests/unit/devtools/test_cli_wrapper.py.

set -euo pipefail

resolved=""
if git_root=$(git rev-parse --show-toplevel 2>/dev/null); then
  if [ -f "$git_root/devtools/__main__.py" ]; then
    resolved="$git_root"
  fi
fi
if [ -z "$resolved" ] && [ -n "${POLYLOGUE_REPO_ROOT:-}" ] \
    && [ -f "$POLYLOGUE_REPO_ROOT/devtools/__main__.py" ]; then
  resolved="$POLYLOGUE_REPO_ROOT"
fi
if [ -z "$resolved" ]; then
  echo "devtools: cannot locate a polylogue checkout (no git root with devtools/__main__.py, and POLYLOGUE_REPO_ROOT is unset or invalid)" >&2
  exit 1
fi

exec python "$resolved/devtools/__main__.py" "$@"

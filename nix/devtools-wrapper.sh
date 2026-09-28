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

# Use the RESOLVED checkout's own interpreter when it has one.
#
# The wrapper already resolves which checkout's CODE to run, but exec'ing a bare
# `python` took whichever interpreter happened to be active -- in a linked
# worktree that is the main checkout's venv, whose editable install points at the
# main checkout. devtools/checkout_guard.py then correctly refuses (exit 125):
# code from one checkout, environment from another.
#
# The practical effect was that plain `devtools test` did not work inside a
# worktree at all. Every invocation needed an explicit
# `env VIRTUAL_ENV=... PATH=...` prefix, which is precisely the friction that
# pushes a caller back to bare `pytest` and silently out of the checkout guard,
# containment and receipts the wrapper exists to provide.
#
# Falls through to `python` when the checkout has no venv, preserving the
# previous behaviour for a fresh clone or a Nix-only environment.
# A checkout with a flake but no venv yet (every fresh worktree) is provisioned
# on first use: the devshell's own shellHook creates .venv and runs the frozen,
# fingerprinted `uv sync`, so the lockfile and environment guards are exactly
# the devshell's. Set POLYLOGUE_DEVTOOLS_NO_PROVISION=1 to skip it.
# Complete means the shellHook's sync fingerprint exists: it is written only
# after a proven-complete `uv sync`, so a venv whose sync failed is retried.
provisioned() { [ -x "$resolved/.venv/bin/python" ] && [ -f "$resolved/.venv/.uv-sync-fingerprint" ]; }
if ! provisioned && [ -f "$resolved/flake.nix" ] \
    && [ -z "${POLYLOGUE_DEVTOOLS_NO_PROVISION:-}" ] && command -v nix >/dev/null 2>&1; then
  echo "devtools: provisioning $resolved/.venv through its devshell" >&2
  # PYTHONPATH is unset here too: the shellHook starts Python while
  # provisioning, and an inherited entry could run a foreign sitecustomize.
  (cd "$resolved" && env -u PYTHONPATH nix develop --accept-flake-config "$resolved" --command true) >&2 \
    || echo "devtools: devshell provisioning failed" >&2
fi

# PYTHONPATH is dropped for the checkout's own interpreter: an inherited value
# can name another checkout, and a sitecustomize there would run during
# interpreter start-up, before devtools can rebind anything. The venv needs no
# PYTHONPATH; the agentctl job environment already unsets it.
if [ -x "$resolved/.venv/bin/python" ] && { provisioned || [ ! -f "$resolved/flake.nix" ]; }; then
  exec env -u PYTHONPATH "$resolved/.venv/bin/python" "$resolved/devtools/__main__.py" "$@"
fi
# A flake checkout that still has no venv would otherwise run whatever
# `python` PATH names first -- possibly another checkout's environment, even
# through a symlink that resolves into the Nix store -- so it is refused. The
# bare-`python` fall-through remains for a checkout without a devshell.
if [ -f "$resolved/flake.nix" ]; then
  echo "devtools: $resolved has no completely provisioned .venv and could not be provisioned; enter its devshell once, then rerun" >&2
  exit 1
fi
exec env -u PYTHONPATH python "$resolved/devtools/__main__.py" "$@"

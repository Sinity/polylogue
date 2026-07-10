#!/usr/bin/env bash
# Bootstrap script for Claude Code Web / Codex Cloud sandboxes.
# Idempotent: safe to re-run.
set -euo pipefail

# uv installer (cloud sandboxes typically do not ship uv).
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# Sync dev deps from the frozen lockfile.
uv sync --extra dev --frozen

# Pre-warm: render checks so the docs surface compiles cleanly.
# Non-fatal so a render mismatch does not block the sandbox starting, but the
# failure must stay visible — a silent pre-warm is worse than none.
if ! uv run devtools render all --check; then
  echo "WARNING: 'devtools render all --check' exited nonzero — inspect the output" >&2
  echo "WARNING: above; branch verification will classify the failure." >&2
fi

# Workspace dirs (matches POLYLOGUE_ARCHIVE_ROOT and
# POLYLOGUE_PYTEST_BASETEMP_ROOT in .claude/settings.json).
mkdir -p /tmp/polylogue-archive/inbox /tmp/polylogue-archive/blob
mkdir -p /tmp/polylogue-pytest

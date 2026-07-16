#!/usr/bin/env bash
set -euo pipefail

ROOT=${1:-.}

# Binary patches are validated by clean application. Generated logs are summarized
# in the validation report. Operational fork prompts intentionally use /mnt/data as
# the artifact handoff location for parallel ChatGPT sessions.
BASE_EXCLUDES=(
  --hidden
  --glob '!**/.git/**'
  --glob '!**/.venv/**'
  --glob '!**/target/**'
  --glob '!**/_deck_render/**'
  --glob '!**/validation/**'
  --glob '!**/*.patch'
  --glob '!**/*.log'
  --glob '!**/scripts/check-public-artifacts.sh'
)
PATH_EXCLUDES=(
  --glob '!**/fork-prompts/**'
  --glob '!**/alternate-prompts/**'
  --glob '!**/08-fork-prompts.md'
  --glob '!**/14-alternate-worktree-prompts.md'
)

# `/home/user/` is an intentional documentation placeholder. Reject actual
# execution-environment paths and named macOS home directories.
PATH_PATTERN='(/mnt/data/|/home/oai/|/home/runner/|/Users/[^/<[:space:]]+/)'
SECRET_PATTERN='(AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{24,}|sk-[A-Za-z0-9_-]{24,}|BEGIN (RSA|OPENSSH|EC) PRIVATE KEY|Bearer [A-Za-z0-9._-]{24,})'

found=0
if rg -n "${BASE_EXCLUDES[@]}" "${PATH_EXCLUDES[@]}" "$PATH_PATTERN" "$ROOT"; then
  echo 'possible execution-environment path found' >&2
  found=1
fi
if rg -n "${BASE_EXCLUDES[@]}" "$SECRET_PATTERN" "$ROOT"; then
  echo 'possible credential or private key found' >&2
  found=1
fi

if (( found )); then
  exit 1
fi

echo 'no obvious execution-environment paths or credential patterns found'

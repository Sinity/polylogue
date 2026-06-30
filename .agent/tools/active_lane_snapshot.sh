#!/usr/bin/env bash
set -euo pipefail
coord=${1:-/realm/inbox/agent-coordination.md}
echo '=== worktrees ==='
for repo in /realm/project/polylogue /realm/project/sinex /realm/project/sinnix /realm/project/sinity-lynchpin; do
  [[ -d "$repo/.git" || -f "$repo/.git" ]] || continue
  echo "--- $repo"; git -C "$repo" worktree list || true
done
echo '=== recent coordination ==='
[[ -f "$coord" ]] && tail -80 "$coord" || echo "missing $coord"
echo '=== pss top ==='
"$(dirname "$0")/pss_top.sh" 15 || true

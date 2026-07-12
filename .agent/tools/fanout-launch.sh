#!/usr/bin/env bash
# fanout-launch.sh — polylogue defaults for the canonical fanout launcher.
#
# The real implementation is permanent in sinnix:
#   dots/_ai/skills/agent-orchestration/scripts/launch_agent_tabs.sh
#   (sinnix commit 1a26aff: --persist-windows, --per-task-workdir-base,
#    --job-prefix, --status)
# The deployed skill copy under ~/.claude/skills resolves into the nix store
# and only refreshes on `switch`; this wrapper prefers the sinnix working-tree
# copy so improvements are usable immediately.
#
# Usage:
#   .agent/tools/fanout-launch.sh <lane...>              # launch on workspace 6
#   .agent/tools/fanout-launch.sh --status               # completion table
# Naming convention (lane name = worktree dir name = prompt basename):
#   lane "polylogue-hermes-wedge" ->
#     worktree /realm/worktrees/polylogue-hermes-wedge
#     prompt   .agent/scratch/fanout-prompts/polylogue-hermes-wedge.prompt
set -euo pipefail

CANONICAL="/realm/project/sinnix/dots/_ai/skills/agent-orchestration/scripts/launch_agent_tabs.sh"
DEPLOYED="$HOME/.claude/skills/agent-orchestration/scripts/launch_agent_tabs.sh"
LAUNCHER="$CANONICAL"
[[ -x "$LAUNCHER" ]] || LAUNCHER="$DEPLOYED"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
PROMPT_DIR="$REPO_ROOT/.agent/scratch/fanout-prompts"
OUTPUT_DIR="/realm/tmp/fanout-$(date +%Y-%m)/out"

if [[ "${1:-}" == "--status" ]]; then
  exec bash "$LAUNCHER" --status --output-dir "$OUTPUT_DIR"
fi

mkdir -p "$OUTPUT_DIR"
exec bash "$LAUNCHER" \
  --agent codex \
  --mode kitty \
  --launch-type os-window \
  --workspace 6 \
  --persist-windows \
  --per-task-workdir-base /realm/worktrees \
  --job-prefix fanout- \
  --model "${FANOUT_MODEL:-gpt-5.6-terra}" \
  --reasoning-effort "${FANOUT_EFFORT:-high}" \
  --prompt-dir "$PROMPT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  "$@"

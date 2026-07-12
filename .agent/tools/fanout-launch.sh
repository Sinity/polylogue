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

# resume <lane...> [--workspace N]: continue a dead lane's PERSISTED codex
# session (context intact) instead of paying fresh re-orientation. Extracts
# the exact session id from the lane's launch receipt in the log — never
# `resume --last` (25+ concurrent sessions make it ambiguous).
if [[ "${1:-}" == "resume" ]]; then
  shift
  ws=66
  lanes=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --workspace) ws="$2"; shift 2 ;;
      *) lanes+=("$1"); shift ;;
    esac
  done
  for lane in "${lanes[@]}"; do
    log="$OUTPUT_DIR/$lane.log"
    sid="$(grep -m1 -oE 'session id: [0-9a-f-]+' "$log" 2>/dev/null | awk '{print $3}')"
    wt="/realm/worktrees/$lane"
    if [[ -z "$sid" || ! -d "$wt" ]]; then
      echo "cannot resume $lane: sid='$sid' wt='$wt'" >&2
      continue
    fi
    rm -f "$OUTPUT_DIR/$lane.exit"
    n=1; while [[ -e "$OUTPUT_DIR/$lane.resume-$n.log" ]]; do n=$((n+1)); done
    rlog="$OUTPUT_DIR/$lane.resume-$n.log"
    cont="Your previous run was interrupted (process died; RAM or quota). First run: git status && git log --oneline -5 to see what you already committed in this worktree. Then continue your mission exactly where you left off — the original brief is at $PROMPT_DIR/$lane.prompt (re-read it). All the same rules apply: commit every chunk, report AC status, open a PR, do not merge."
    scope=(); command -v sinnix-scope >/dev/null && scope=(sinnix-scope background --)
    # NOTE: the `codex` command on this host is a wrapper that already injects
    # --profile; do NOT pass --profile here (double-flag is a hard error).
    # zsh pipestatus is 1-based: printf=[1], codex=[2], tee=[3].
    inner="cd $(printf '%q' "$wt") && { command -v direnv >/dev/null && direnv allow . 2>/dev/null; }; printf '%s' $(printf '%q' "$cont") | ${scope[*]} codex exec resume $(printf '%q' "$sid") --model ${FANOUT_MODEL:-gpt-5.6-terra} -c 'model_reasoning_effort=\"${FANOUT_EFFORT:-high}\"' --output-last-message $(printf '%q' "$OUTPUT_DIR/$lane.last.md") - 2>&1 | tee $(printf '%q' "$rlog"); ec=\${pipestatus[2]:-\$?}; echo \$ec > $(printf '%q' "$OUTPUT_DIR/$lane.exit"); print \"[resume $lane exit=\$ec]\"; exec zsh -i"
    kitty @ launch --keep-focus --type=os-window --title "agent-$lane" --cwd "$wt" -- zsh -lc "$inner" >/dev/null
    for _ in {1..40}; do
      sinnix-hypr-control dispatch movetoworkspacesilent "$ws,title:^(agent-$lane)\$" >/dev/null 2>&1 && break
      sleep 0.05
    done
    echo "resumed $lane (session $sid) -> workspace $ws, log $rlog"
  done
  exit 0
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

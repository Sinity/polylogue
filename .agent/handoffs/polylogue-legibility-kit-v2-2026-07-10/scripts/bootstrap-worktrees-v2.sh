#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bootstrap-worktrees-v2.sh --polylogue /path/to/polylogue --sinex /path/to/sinex --root /path/to/worktrees [--horizon 72h|7d|30d]

Creates one worktree per mission for the selected horizon. It does not launch agents.
Branches are deliberately disposable: legibility/<mission-id-lowercase>.
EOF
}

POLYLOGUE=""
SINEX=""
ROOT=""
HORIZON="72h"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --polylogue) POLYLOGUE=$2; shift 2 ;;
    --sinex) SINEX=$2; shift 2 ;;
    --root) ROOT=$2; shift 2 ;;
    --horizon) HORIZON=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done
[[ -n "$POLYLOGUE" && -n "$SINEX" && -n "$ROOT" ]] || { usage; exit 2; }

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
KIT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
PLAN="$KIT_ROOT/control/mission-plan.yaml"
mkdir -p "$ROOT"

python - "$PLAN" "$POLYLOGUE" "$SINEX" "$ROOT" "$HORIZON" <<'PY'
from __future__ import annotations
import subprocess
import sys
from pathlib import Path
import yaml

plan_path = Path(sys.argv[1])
poly = Path(sys.argv[2])
sinex = Path(sys.argv[3])
root = Path(sys.argv[4])
horizon = sys.argv[5]
plan = yaml.safe_load(plan_path.read_text())
for m in plan['missions']:
    if m['horizon'] != horizon or m['merge_mode'] == 'coordinator-owned':
        continue
    repos = ['polylogue', 'sinex'] if m['repo'] == 'joint' else [m['repo']]
    for repo_name in repos:
        repo = poly if repo_name == 'polylogue' else sinex
        suffix = f"-{repo_name}" if m['repo'] == 'joint' else ''
        target = root / f"{m['id'].lower()}{suffix}"
        branch = f"legibility/{m['id'].lower()}{suffix}"
        if target.exists():
            print(f"skip existing {target}")
            continue
        subprocess.run(['git', '-C', str(repo), 'worktree', 'add', '-b', branch, str(target), 'HEAD'], check=True)
        handoff_dir = target / '.agent-handoff'
        handoff_dir.mkdir(exist_ok=True)
        (handoff_dir / 'MISSION.md').write_text(
            f"# {m['id']}: {m['title']}\n\nPrompt: {m['prompt_file']}\nStop condition: {m['stop_condition']}\n",
            encoding='utf-8',
        )
        print(target)
PY

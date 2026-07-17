#!/usr/bin/env bash
# Build final launchable prompts: mission + shared contract, one file per job.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p prompts
declare -A CONTRACT=(
  [implementation]="../contracts/chatgpt-pro-implementation.md"
  [analysis]="../contracts/chatgpt-pro-analysis.md"
  [deep-research]="../contracts/chatgpt-deep-research.md"
)
python3 - <<'EOF'
import json, pathlib
c = json.load(open('campaign.json'))
contracts = {
    'implementation': '../contracts/chatgpt-pro-implementation.md',
    'analysis': '../contracts/chatgpt-pro-analysis.md',
    'deep-research': '../contracts/chatgpt-deep-research.md',
}
missing = []
for job in c['jobs']:
    mission = pathlib.Path(job['mission'])
    if not mission.exists():
        missing.append(job['id']); continue
    contract = pathlib.Path(contracts[job['contract']])
    out = pathlib.Path('prompts') / f"{job['id']}-{job['slug']}.md"
    out.write_text(mission.read_text() + "\n\n---\n\n" + contract.read_text())
    print(f"built {out}")
if missing:
    print(f"MISSING missions (not built): {', '.join(missing)}")
EOF

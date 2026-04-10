# Mutation Testing Baseline

This document is the stable operator guide for mutation testing in the live
repo. Keep campaign results in ignored local artifact storage under
`.local/mutation-campaigns/`.

## Canonical Workflow

### Verification Baseline

Commands below assume the devshell is already active. Outside it, prefix them
with `nix develop -c`.

```bash
ruff check polylogue tests devtools
pytest -q -n 0
```

### Broad Mutation Surface

Mutmut configuration lives in [`pyproject.toml`](../pyproject.toml):

- `paths_to_mutate = ["polylogue"]`
- package markers and thin entrypoints are excluded
- pytest is forced to run with `-n 0` for stable mutation bookkeeping

Keep the committed config broad. Use focused campaigns for narrower fronts.

### Focused Campaign Runner

Use [`devtools/mutmut_campaign.py`](../devtools/mutmut_campaign.py) for isolated
campaign runs. It copies the current worktree into a temporary workspace,
patches only that workspace's mutmut scope, runs the campaign there, then
writes durable JSON and Markdown artifacts.

```bash
# List campaigns
python -m devtools mutmut-campaign list

# Run a focused campaign and write durable artifacts
python -m devtools mutmut-campaign run filters \
  --json-out .local/mutation-campaigns/$(date +%F)-filters.json \
  --markdown-out .local/mutation-campaigns/$(date +%F)-filters.md

# Rebuild the campaign index
python -m devtools mutmut-campaign index
```

## Baseline Policy

- Treat the latest local artifact set under `.local/mutation-campaigns/` as
  the durable scoreboard.
- Refresh your local artifact index after any meaningful mutmut wave.
- When a baseline changes because production code moved, rerun the affected
  campaign and regenerate the artifact set.
- Keep this file focused on workflow and policy, not session-era historical
  tables that immediately drift.

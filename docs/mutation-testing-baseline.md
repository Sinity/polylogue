# Mutation Testing Baseline

This document is the stable operator guide for mutation testing in the live
repo. Keep campaign results in ignored local artifact storage under
`artifacts/mutation-campaigns/` rather than embedding long historical tables
here or checking artifact directories into version history.

## Canonical Workflow

### Verification Baseline

```bash
nix develop -c ruff check polylogue tests
nix develop -c pytest -q -n 0
```

### Broad Mutation Surface

Mutmut configuration lives in [`pyproject.toml`](../pyproject.toml):

- `paths_to_mutate = ["polylogue"]`
- package markers and thin entrypoints are excluded
- pytest is forced to run with `-n 0` for stable mutation bookkeeping

Do not narrow the committed config to a hand-picked subset. Use focused
campaigns for narrower fronts.

### Focused Campaign Runner

Use [`devtools/mutmut_campaign.py`](../devtools/mutmut_campaign.py) for isolated
campaign runs. It copies the current worktree into a temporary workspace,
patches only that workspace's mutmut scope, runs the campaign there, then
writes durable JSON and Markdown artifacts.

```bash
# List campaigns
nix develop -c python -m devtools.mutmut_campaign list

# Run a focused campaign and write durable artifacts
nix develop -c python -m devtools.mutmut_campaign run filters \
  --json-out artifacts/mutation-campaigns/$(date +%F)-filters.json \
  --markdown-out artifacts/mutation-campaigns/$(date +%F)-filters.md

# Rebuild the campaign index
nix develop -c python -m devtools.mutmut_campaign index
```

## Baseline Policy

- Treat the latest local artifact set under `artifacts/mutation-campaigns/` as
  the durable scoreboard.
- Refresh your local artifact index after any meaningful mutmut wave.
- When a baseline changes because production code moved, rerun the affected
  campaign instead of hand-editing old summaries.
- Keep this file focused on workflow and policy, not session-era historical
  tables that immediately drift.

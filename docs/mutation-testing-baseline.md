# Mutation Testing Baseline

This file is the durable ledger for mutation-testability work. Refresh it after
meaningful mutmut, law, or property-testing waves so later passes compare
against versioned evidence instead of session memory.

## Canonical Workflow

### Verification Baseline

```bash
nix develop -c ruff check .
nix develop -c pytest -q -n 0
```

### Broad Mutation Surface

Mutmut configuration lives in [`pyproject.toml`](../pyproject.toml):

- `paths_to_mutate = ["polylogue"]`
- excludes only thin package markers and entrypoints
- forces stable pytest runtime flags with xdist disabled

Do not narrow the committed config back to a tiny hand-picked module list. Run
focused campaigns with the isolated runner instead.

### Isolated Campaign Runner

Use [`devtools/mutmut_campaign.py`](../devtools/mutmut_campaign.py) for focused
campaigns. It copies the worktree into an isolated temporary workspace, patches
only that workspace's `[tool.mutmut]` scope/test selection, runs `mutmut` there,
then writes durable JSON/Markdown artifacts under
[`docs/mutation-campaigns/`](mutation-campaigns/README.md).

```bash
# List available campaigns
nix develop -c python -m devtools.mutmut_campaign list

# Run one isolated campaign and write durable artifacts
nix develop -c python -m devtools.mutmut_campaign run filters \
  --json-out docs/mutation-campaigns/$(date +%F)-filters.json \
  --markdown-out docs/mutation-campaigns/$(date +%F)-filters.md

# Rebuild the artifact index
nix develop -c python -m devtools.mutmut_campaign index
```

## Latest Non-Mutation Baseline

Recorded on `2026-03-11`.

### Repo-Wide Lint

- Command: `nix develop -c ruff check .`
- Result: `All checks passed!`

### Full Test Suite

- Command: `nix develop -c pytest -q -n 0`
- Result: `4304 passed in 63.35s`
- Note: repo-wide pytest defaults still enable `-n auto`; use `-n 0` here for
  stable mutation-comparison timing.

## Broad Campaign Wave: `2026-03-11`

The latest recorded artifact for each campaign is indexed in
[`docs/mutation-campaigns/README.md`](mutation-campaigns/README.md). These
numbers are from a clean rerun on commit `147e689d15ca`, not from the earlier
dirty-tree exploratory pass.

The wave was run to answer one question: are we ready to execute the next
law-based test generalization wave in
[`004-law-test-wave-iteration-plan-2026-03-11.md`](../.claude/scratch/004-law-test-wave-iteration-plan-2026-03-11.md)?

### High-Signal Campaigns

These are already mutation-usable. They still have residue, but not blindness.

| Campaign | Killed | Survived | Timeout | Not checked | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| `filters` | 475 | 5 | 117 | 0 | Mutation-usable; remaining work is mostly timeout-heavy filter pipeline paths rather than blind spots. |
| `json` | 24 | 2 | 0 | 0 | Nearly saturated after the latest exact contract additions. |
| `fts5` | 41 | 7 | 0 | 0 | Good signal; some ranking/search semantics still survive. |
| `hybrid` | 112 | 21 | 0 | 3 | Good enough to use as feedback during later law work. |
| `models` | 129 | 20 | 3 | 14 | Reasonable signal with bounded semantic weak spots. |

### Not Ready / Major Remediation Targets

These are the domains the next law-wave should attack first.

| Campaign | Killed | Survived | Timeout | Not checked | Primary issue |
| --- | ---: | ---: | ---: | ---: | --- |
| `schema-validation` | 229 | 167 | 0 | 0 | Validator/verification behavior is still too weakly constrained. |
| `schema-inference` | 536 | 759 | 3 | 0 | Large survivor surface across inference/privacy heuristics. |
| `schema-core` | 765 | 900 | 29 | 0 | Schema domain overall is still heavily under-specified. |
| `repository` | 343 | 250 | 6 | 81 | Read/query/projection contracts are not strong enough. |
| `pipeline-services` | 725 | 687 | 3 | 246 | Planning/validation/parse orchestration still has large blind spots. |
| `source-detection` | 41 | 197 | 0 | 910 | Detection/dispatch is still mostly unexercised under mutation. |
| `providers-semantics` | 162 | 588 | 0 | 432 | Harmonization/provider semantic extraction is not ready for saturation claims. |
| `sources-parse` | 1353 | 2307 | 0 | 2094 | Broad parse/harmonization surface confirms the next law-wave should start in sources. |

### Readiness Call

- We are **ready to start** the next law-wave.
- We are **not ready to skip it**.
- The mutation results say the next wave should begin with:
  1. source detection + parser dispatch,
  2. provider semantics + harmonization,
  3. broad sources/parse contracts,
  4. schema validator/inference contracts,
  5. pipeline service contracts,
  6. repository query/projection laws.

### Comparison Rule For Future Waves

When the next law/property wave lands, compare against this ledger and the
per-campaign artifacts. Expect at least one of these to improve:

1. Fewer total survivors in the targeted campaign.
2. Fewer total `not_checked` mutants in source/pipeline/repository domains.
3. No regression in killed counts for already-healthy campaigns.
4. No new timeout mass introduced by broader law/property generation.

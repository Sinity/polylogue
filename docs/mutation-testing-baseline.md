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
[`docs/mutation-campaigns/README.md`](mutation-campaigns/README.md). This
ledger now mixes three clean baselines:

- the broad core baseline on commit `147e689d15ca`
- the schema/pipeline rerun wave on commit `d1e704d7a2ba`
- the post-`004` law-wave reruns on commit `2bdb267e93b7`

The `004` law/property wave has now been executed. Its main runtime evidence is
captured in the fresh reruns for CLI, repository, site, and sources.

### Already-Healthy Campaigns

These are mutation-usable enough that they can now serve mainly as regression
guards during future waves rather than first-line remediation targets.

| Campaign | Commit | Killed | Survived | Timeout | Not checked | Interpretation |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `filters` | `147e689d15ca` | 475 | 5 | 117 | 0 | Very strong kill rate; remaining debt is timeout-heavy pipeline execution, not semantic blindness. |
| `json` | `147e689d15ca` | 24 | 2 | 0 | 0 | Effectively saturated for its size. |
| `fts5` | `147e689d15ca` | 41 | 7 | 0 | 0 | Good search-law signal with a small survivor tail. |
| `hybrid` | `147e689d15ca` | 112 | 21 | 0 | 3 | Strong enough to guide later search refactors. |
| `models` | `147e689d15ca` | 129 | 20 | 3 | 14 | Good semantic signal; not a blocking blind spot. |
| `cli-run` | `2bdb267e93b7` | 183 | 92 | 0 | 8 | `004` improved this from `121/99/0/63` to `183/92/0/8`; it is now a maintenance surface, not the next law-wave priority. |

### Schema / Pipeline Follow-Up Baselines

These were already improved substantially before the `004` execution and remain
the reference point for those domains.

| Campaign | Commit | Killed | Survived | Timeout | Not checked | Primary issue |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `schema-validation` | `d1e704d7a2ba` | 235 | 161 | 0 | 0 | Validator/verification behavior is still under-constrained. |
| `schema-inference` | `d1e704d7a2ba` | 534 | 317 | 447 | 0 | Timeouts still dominate inference/privacy heuristics. |
| `schema-core` | `d1e704d7a2ba` | 792 | 895 | 7 | 0 | Core schema behavior still has heavy survivor mass. |
| `pipeline-services` | `d1e704d7a2ba` | 736 | 595 | 84 | 246 | Acquisition/streaming/state-machine helpers still leave large timeout and not-checked clusters. |

### Post-`004` Rerun Results

This is the current evidence for the domains that the executed law-wave
targeted directly.

| Campaign | Killed | Survived | Timeout | Not checked | Delta vs previous baseline | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `cli-query` | 580 | 844 | 21 | 532 | `+112 killed`, `-15 survived`, `+13 timeout`, `-110 not_checked` | Better coverage of query/output contracts, but `_output_stats_by_summaries` and mutation-heavy query execution still need another pass. |
| `repository` | 370 | 181 | 124 | 5 | `+45 killed`, `+41 survived`, `-21 timeout`, `-65 not_checked` | Big reduction in blind spots; remaining issue is now timeout/survivor quality in read/query paths, not lack of reach. |
| `site-builder` | 245 | 228 | 1 | 0 | `+23 killed`, `-22 survived`, `-1 timeout` | Streaming/site generation laws materially improved this area. |
| `source-detection` | 563 | 455 | 3 | 127 | `+88 killed`, `-20 survived`, `-68 not_checked` | Detection/dispatch is no longer mostly blind, but still has meaningful survivor mass. |
| `providers-semantics` | 415 | 652 | 3 | 112 | `+80 killed`, `0 survived`, `+3 timeout`, `-83 not_checked` | Law-wave improved reach, but `schemas.unified` semantic extraction remains the dominant weak spot. |
| `sources-parse` | 1651 | 2390 | 7 | 1706 | `+158 killed`, `-11 survived`, `+4 timeout`, `-151 not_checked` | Broadest source surface improved meaningfully, but still confirms sources/parsing as the next major law-wave frontier. |

### Readiness Call

- `004` is complete.
- We are ready for another targeted law/property wave.
- The current highest-yield next fronts are:
  1. `sources-parse`
  2. `providers-semantics`
  3. `repository`
  4. `cli-query`
- We are not ready to claim source/harmonization semantics are exhaustively
  specified. The reruns made those domains mutation-usable, but not saturated.
- The dominant remaining structural issue is still survivor concentration in
  `polylogue.schemas.unified` and adjacent source/provider semantic extraction.

### Comparison Rule For Future Waves

When the next law/property wave lands, compare against this ledger and the
per-campaign artifacts. Expect at least one of these to improve:

1. Fewer total survivors in the targeted campaign.
2. Fewer total `not_checked` mutants in source/pipeline/repository domains.
3. No regression in killed counts for already-healthy campaigns.
4. No new timeout mass introduced by broader law/property generation unless it
   buys a larger reduction in survivors or not-checked mutants.

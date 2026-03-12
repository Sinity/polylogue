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

Recorded on `2026-03-12`.

### Repo-Wide Lint

- Command: `nix develop -c ruff check .`
- Result: `All checks passed!`

### Full Test Suite

- Command: `nix develop -c pytest -q -n 0`
- Result: `3917 passed, 1 warning in 244.35s`
- Note: repo-wide pytest defaults still enable `-n auto`; use `-n 0` here for
  stable mutation-comparison timing.

## Broad Campaign Wave: `2026-03-11`

The latest recorded artifact for each campaign is indexed in
[`docs/mutation-campaigns/README.md`](mutation-campaigns/README.md). This
ledger now mixes four clean baselines:

- the broad core baseline on commit `147e689d15ca`
- the schema/pipeline rerun wave on commit `d1e704d7a2ba`
- the post-`004` law-wave baselines on commit `2bdb267e93b7`
- the follow-up source/query/repository reruns on commits `7e7c310037f9`,
  `a27de694650d`, and `3bdd3f02dc87`
- the clean post-`005` rerun wave on commit `e759af23458d`
- the clean Phase-5 CLI/site concentration reruns on commit `58264c2c47be`

`004`, its follow-up reruns, and the first focused post-`005` source/provider
concentration reruns are now complete. The tables below are the current durable
mutation baselines for the next law/property wave.

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
| `cli-run` | `58264c2c47be` | 167 | 21 | 87 | 8 | Phase 5 concentrated the owner files sharply, but the clean rerun exposed heavy timeout mass in observer/display paths. It remains usable, but no longer belongs in the "healthy" bucket without follow-up runtime-contract work. |

### Schema / Pipeline Follow-Up Baselines

These are the clean post-Phase-4 baselines after splitting schema and pipeline
owner files, refreshing the campaign definitions, and rerunning all four
campaigns on commit `856caf495bab`.

| Campaign | Commit | Killed | Survived | Timeout | Not checked | Primary issue |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `schema-validation` | `856caf495bab` | 235 | 161 | 0 | 0 | Stable after the Phase 4 split: reach is complete, but drift detection and validation sample behavior are still under-constrained. |
| `schema-inference` | `856caf495bab` | 561 | 707 | 30 | 0 | The split removed the old timeout wall (`447 -> 30`) and exposed the real survivor mass in dynamic-key/pathlike/privacy heuristics. |
| `schema-core` | `856caf495bab` | 795 | 883 | 16 | 0 | Aggregate schema reach remains complete; the remaining debt is concentrated in schema annotation and inference heuristics rather than blind spots. |
| `pipeline-services` | `856caf495bab` | 841 | 648 | 84 | 35 | Phase 4 materially improved reach (`246 -> 35 not_checked`) and kill count, but Drive raw-stream acquisition still sits outside the current service-law surface. |

### Current Campaign Baselines

These are the current durable baselines for the domains touched by `004` and
the follow-up source/helper/query pass.

| Campaign | Commit | Killed | Survived | Timeout | Not checked | Interpretation |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `cli-query` | `58264c2c47be` | 954 | 985 | 23 | 0 | Clean Phase-5 rerun after replacing the mixed CLI owner files: concentration held, survivor density stayed essentially flat, and timeout mass shifted into `query_actions` modifier/delete flows instead of broad routing blindness. |
| `drive-client` | `37e26aba2d3d` | 581 | 299 | 3 | 0 | Clean post-tightening rerun improved kill rate again; the remaining residue is now mostly concentrated in credential loading, folder resolution, and metadata/download helpers rather than broad reach gaps. |
| `repository` | `3bdd3f02dc87` | 538 | 94 | 77 | 0 | Clean rerun after fixing mutation-artifact path anchoring and re-running the concentrated repository surface from a clean worktree: reach stayed complete, survivor count improved modestly, and the remaining debt is concentrated in archive stats, conversation-record conversion, save/search similarity, and render/session-tree hydration seams. Timeout mass remains real, especially around session-tree/render-projection paths. |
| `site-builder` | `58264c2c47be` | 240 | 224 | 10 | 0 | Clean Phase-5 rerun after owner concentration: survivor counts improved slightly, but timeout mass grew around archive scanning and root/provider index generation. The area is still concentrated, but not "set and forget." |
| `source-detection` | `844d52ee925d` | 825 | 324 | 2 | 0 | Clean post-concentration rerun after collapsing scattered source-iteration examples into `test_source_laws.py`; reach stayed complete, kill count improved, and timeout noise dropped while survivor mass remained concentrated in ZIP filtering, emit paths, and provider sniffing. |
| `providers-semantics` | `315beb0f19f1` | 819 | 455 | 2 | 0 | Clean rerun after consolidating semantic-law ownership and refactoring `schemas.unified` dispatch into explicit adapter/fallback maps. Reach stayed complete, but kill count regressed, which means the suite is more concentrated yet still underspecified around `extract_content_blocks`, `to_meta`, fallback Claude Code extraction, and harmonization edge cases. |
| `sources-parse` | `47a9b1cff33f` | 3597 | 2319 | 31 | 0 | Clean post-drive-parser concentration rerun improved kill and survivor counts again while keeping reach complete; the remaining debt is now more sharply concentrated in `schemas.unified`, Drive auth/filter helpers, and timeout-heavy `content_blocks_from_segments` coverage. |

### Readiness Call

- `004` is complete, and the immediate follow-up reruns are complete.
- The first post-`005` focused concentration rerun wave is complete on clean SHAs `c0596770631e`, `47a9b1cff33f`, `a3440a0f1a4b`, and `027519a11118`.
- We are ready for the next targeted law/property wave.
- The current highest-yield next fronts are:
  1. `sources-parse`
  2. `providers-semantics`
  3. `source-detection`
  4. `drive-client`
  5. `repository`
  6. `cli-query`
- We are not ready to claim source/provider/harmonization semantics are exhaustively specified.
  The reruns removed reach failures, but they did not saturate the semantic space.
- The dominant structural issues are now:
  - survivor concentration in `polylogue.schemas.unified` and provider viewport shaping,
  - `providers-semantics` concentration reduced duplication but also revealed lost mutation signal around adapter content-block/meta extraction,
  - high survivor density in source parsing and query orchestration,
  - source parsing now carrying a clearer timeout cluster in `content_blocks_from_segments`,
  - source detection still carrying a narrower but still meaningful survivor cluster around ZIP filtering, emit paths, and provider sniffing,
  - drive-client transport/auth behavior still carrying a narrower but meaningful survivor cluster in auth/load/download helpers.
- Additional mutmut infrastructure work is not the bottleneck now. The next
  gains come from stronger laws, better generators/oracles, and code
  refactors that collapse duplicated semantic authority.
- Phase 5 succeeded at structural concentration, but its reruns also showed
  that CLI/site areas still carry meaningful timeout-heavy blind spots. Those
  are now execution-contract problems, not file-ownership problems.
- The three-front concentration wave (`source-detection`, `providers-semantics`,
  `repository`) is now fully closed. The next execution wave should target
  survivor density, not reach restoration.

### Comparison Rule For Future Waves

When the next law/property wave lands, compare against this ledger and the
per-campaign artifacts. Expect at least one of these to improve:

1. Fewer total survivors in the targeted campaign.
2. Fewer total `not_checked` mutants in source/pipeline/repository domains.
3. No regression in killed counts for already-healthy campaigns.
4. No new timeout mass introduced by broader law/property generation unless it
   buys a larger reduction in survivors or not-checked mutants.

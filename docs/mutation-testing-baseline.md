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
- Result: `4273 passed, 1 warning in 298.07s`
- Note: repo-wide pytest defaults still enable `-n auto`; use `-n 0` here for
  stable mutation-comparison timing.

## Broad Campaign Wave: `2026-03-11`

The latest recorded artifact for each campaign is indexed in
[`docs/mutation-campaigns/README.md`](mutation-campaigns/README.md). This
ledger now mixes four clean baselines:

- the broad core baseline on commit `147e689d15ca`
- the schema/pipeline rerun wave on commit `d1e704d7a2ba`
- the post-`004` law-wave baselines on commit `2bdb267e93b7`
- the follow-up source/query/repository reruns on commits `7e7c310037f9` and
  `a27de694650d`
- the clean post-`005` rerun wave on commit `e759af23458d`

`004`, its follow-up reruns, and the first post-`005` source/provider
consolidation reruns are now complete. The tables below are the current durable
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

### Current Campaign Baselines

These are the current durable baselines for the domains touched by `004` and
the follow-up source/helper/query pass.

| Campaign | Commit | Killed | Survived | Timeout | Not checked | Interpretation |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `cli-query` | `e759af23458d` | 963 | 1018 | 16 | 0 | Query execution/output still has very high survivor density; the next wave needs stronger route/error/mutation laws, not more infrastructure. |
| `drive-client` | `7e7c310037f9` | 553 | 327 | 4 | 0 | Drive transport/auth seams are mutation-usable now; remaining residue clusters around credential loading, folder resolution, and retry/download paths. |
| `repository` | `e759af23458d` | 569 | 104 | 35 | 0 | Repository laws improved materially; the main remaining issue is timeout-heavy hydration/read-path coverage, not blind spots. |
| `site-builder` | `2bdb267e93b7` | 245 | 228 | 1 | 0 | Streaming/site generation laws materially improved this area and it is now mainly a regression guard. |
| `source-detection` | `e759af23458d` | 713 | 435 | 3 | 0 | Detection/dispatch now has zero `not_checked`; remaining survivors cluster in ZIP filtering, emit paths, and provider sniffing. |
| `providers-semantics` | `e759af23458d` | 847 | 487 | 2 | 0 | Provider/viewpoint coverage is now cleanly reachable; `extract_content_blocks`, `to_meta`, and fallback Claude Code harmonization still dominate. |
| `sources-parse` | `e759af23458d` | 3494 | 2467 | 11 | 0 | The broad source surface is fully reachable now; remaining work is concentrated in drive-client seams and provider semantic survivor mass. |

### Readiness Call

- `004` is complete, and the immediate follow-up reruns are complete.
- The first post-`005` rerun wave is also complete on a clean SHA.
- We are ready for the next targeted law/property wave.
- The current highest-yield next fronts are:
  1. `sources-parse`
  2. `cli-query`
  3. `providers-semantics`
  4. `drive-client` plus the adjacent `source-detection` seam
  5. `repository`
- We are not ready to claim source/provider/harmonization semantics are exhaustively specified.
  The reruns removed reach failures, but they did not saturate the semantic space.
- The dominant structural issues are now:
  - survivor concentration in `polylogue.schemas.unified` and provider viewport shaping,
  - high survivor/timeout mass in query orchestration,
  - drive-client transport/auth behavior still carrying a large untouched surface.
- Additional mutmut infrastructure work is not the bottleneck now. The next
  gains come from stronger laws, better generators/oracles, and code
  refactors that collapse duplicated semantic authority.

### Comparison Rule For Future Waves

When the next law/property wave lands, compare against this ledger and the
per-campaign artifacts. Expect at least one of these to improve:

1. Fewer total survivors in the targeted campaign.
2. Fewer total `not_checked` mutants in source/pipeline/repository domains.
3. No regression in killed counts for already-healthy campaigns.
4. No new timeout mass introduced by broader law/property generation unless it
   buys a larger reduction in survivors or not-checked mutants.

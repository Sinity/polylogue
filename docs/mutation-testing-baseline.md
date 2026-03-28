# Quality Baselines

This file is the durable baseline ledger for test-quality work. Refresh it after meaningful law/property/mutation waves so later passes can compare against a concrete, versioned reference instead of memory.

## Commands

```bash
# Repo-wide lint
nix develop -c ruff check .

# Full test suite baseline
nix develop -c pytest -q -n 0

# Mutation testing baseline
nix develop -c mutmut run
nix develop -c mutmut results
```

## Latest Baseline

Recorded on `2026-03-11`.

### Repo-Wide Lint

- Command: `nix develop -c ruff check .`
- Result: `All checks passed!`

### Full Test Suite

- Command: `nix develop -c pytest -q -n 0`
<<<<<<< ours
- Result: `4235 passed in 258.02s (0:04:18)`
- Note: the repo-wide pytest default still includes `-n auto --benchmark-disable` in `pyproject.toml`; this ledger uses `-n 0` as the stable comparison baseline.

### Mutation Testing Harness

- Command: `nix develop -c mutmut run`
- Scope source of truth: [`pyproject.toml`](../pyproject.toml)
- Configured mutation target root: `polylogue`
- Configured default exclusions:
  - `polylogue/**/__init__.py`
  - `polylogue/**/__main__.py`
- Harness note: mutmut now forces `pytest_add_cli_args = ["-n", "0", "-p", "no:randomly", "-p", "no:random-order", "--benchmark-disable", "-m", "not benchmark"]` so it does not inherit repo-wide `pytest -n auto` and so benchmark-only tests do not distort the campaign.
- Execution rule: chunk broad mutation campaigns with CLI globs, for example:
  - `nix develop -c mutmut run "polylogue.lib.*" "polylogue.storage.*"`
  - `nix develop -c mutmut run "polylogue.pipeline.*" "polylogue.facade.*" "polylogue.rendering.*" "polylogue.site.*"`
  - `nix develop -c mutmut run "polylogue.cli.*" "polylogue.sources.*"`

## Scoped Historical Mutation Baseline

- Command: `nix develop -c mutmut run`
- Scope note: this was the earlier narrow baseline before the switch to broad chunked mutation campaigns.
- Result: `1062` mutants checked at `18.83 mutations/second`
- Totals:

| Status | Count |
| --- | ---: |
| Killed | 339 |
| Survived | 106 |
| No tests | 613 |
| Timeout | 4 |

### Module Breakdown

| Module | Killed | Survived | No tests | Timeout |
| --- | ---: | ---: | ---: | ---: |
| `polylogue.lib.models` | 117 | 34 | 13 | 2 |
| `polylogue.lib.filters` | 0 | 0 | 597 | 0 |
| `polylogue.lib.roles` | 1 | 0 | 0 | 0 |
| `polylogue.lib.timestamps` | 44 | 2 | 0 | 0 |
| `polylogue.lib.hashing` | 31 | 9 | 0 | 2 |
| `polylogue.lib.json` | 11 | 15 | 0 | 0 |
| `polylogue.storage.search_providers.fts5` | 43 | 5 | 0 | 0 |
| `polylogue.storage.search_providers.hybrid` | 92 | 41 | 3 | 0 |
| **Total** | **339** | **106** | **613** | **4** |

## Dominant Survivor Clusters

These are the biggest current signals where tests execute code but do not constrain behavior strongly enough.

| Function cluster | Surviving mutants |
| --- | ---: |
| `polylogue.lib.models.Message.extract_thinking` | 20 |
| `polylogue.storage.search_providers.hybrid.HybridSearchProvider.search_scored` | 16 |
| `polylogue.lib.json.dumps` | 14 |
| `polylogue.lib.models.Message._is_chatgpt_thinking` | 10 |
| `polylogue.storage.search_providers.hybrid._resolve_ranked_conversation_ids` | 10 |
| `polylogue.storage.search_providers.hybrid.create_hybrid_provider` | 6 |
| `polylogue.storage.search_providers.hybrid.HybridSearchProvider.search_conversations` | 6 |
| `polylogue.lib.hashing.hash_payload` | 5 |

## Dominant No-Test Clusters

Almost all current `no tests` results are concentrated in `polylogue.lib.filters`, which means the configured mutmut scope still does not drive that module through enough executable paths.

| Function cluster | No-test mutants |
| --- | ---: |
| `polylogue.lib.filters.ConversationFilter._sql_pushdown_params` | 51 |
| `polylogue.lib.filters.ConversationFilter._describe_active_filters` | 50 |
| `polylogue.lib.filters.ConversationFilter.pick` | 40 |
| `polylogue.lib.filters.ConversationFilter._apply_common_filters` | 38 |
| `polylogue.lib.filters.ConversationFilter._fetch_generic` | 37 |
| `polylogue.lib.filters.ConversationFilter.__init__` | 35 |
| `polylogue.lib.filters.ConversationFilter._apply_filters` | 35 |
| `polylogue.lib.filters.ConversationFilter._apply_sort` | 32 |

## Timeout Clusters

| Function cluster | Timeout mutants |
| --- | ---: |
| `polylogue.lib.models.Conversation.iter_pairs` | 2 |
| `polylogue.lib.hashing.hash_file` | 2 |

## Interpretation

- The `filters` module is the largest blind spot by far. The current mutmut test selection avoids the slower DB-backed filter suites, but the remaining law/property tests are not enough to exercise the core filter pipeline.
- `models`, `json`, and `hybrid` are the strongest next candidates for law-strengthening. They already execute under mutmut, but too many semantic mutations still survive.
- `roles` is effectively saturated under the current scope.
- `timestamps` is close, but not complete.

## Comparison Rule For Future Waves

When the next law/property wave lands, compare against this file and expect at least one of these to improve:

1. Fewer total survivors.
2. Fewer total `no tests`, especially in `polylogue.lib.filters`.
3. No regression in killed count for already-strong modules.
4. Fewer timeout mutants in `hash_file` and `iter_pairs`.
||||||| base
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
- the follow-up source/query/repository reruns on commits `7e7c310037f9`,
  `a27de694650d`, and `b1f1d35bee28`
- the clean post-`005` rerun wave on commit `e759af23458d`

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
| `cli-query` | `e058c8240959` | 935 | 1007 | 20 | 0 | Clean post-concentration rerun: helper ownership is cleaner and survivors are still concentrated in `_async_execute_query` routing plus mutation-heavy modifier/delete paths. |
| `drive-client` | `37e26aba2d3d` | 581 | 299 | 3 | 0 | Clean post-tightening rerun improved kill rate again; the remaining residue is now mostly concentrated in credential loading, folder resolution, and metadata/download helpers rather than broad reach gaps. |
| `repository` | `b1f1d35bee28` | 568 | 74 | 66 | 0 | The concentrated read-model laws fixed the helper blind spot and cut survivor density sharply; the main remaining issue is timeout-heavy hydration/archive-stat coverage, not reach. |
| `site-builder` | `2bdb267e93b7` | 245 | 228 | 1 | 0 | Streaming/site generation laws materially improved this area and it is now mainly a regression guard. |
| `source-detection` | `38d65c004a2a` | 816 | 330 | 5 | 0 | Clean post-helper-contract rerun sharply improved kill rate while keeping reach complete; survivor mass is now much narrower and still concentrated in ZIP filtering, emit paths, and provider sniffing. |
| `providers-semantics` | `c0596770631e` | 805 | 504 | 2 | 0 | First focused concentration rerun after source semantic test collapse; reach is still clean and survivor mass remains concentrated in `extract_content_blocks`, `to_meta`, and fallback Claude Code harmonization. |
| `sources-parse` | `c0596770631e` | 3455 | 2482 | 10 | 0 | First focused concentration rerun after source semantic test collapse; the broad source surface remains fully reachable, with remaining work concentrated in drive-client seams and provider semantic survivor mass. |

### Readiness Call

- `004` is complete, and the immediate follow-up reruns are complete.
- The first post-`005` focused concentration rerun wave is also complete on clean SHA `c0596770631e`.
- We are ready for the next targeted law/property wave.
- The current highest-yield next fronts are:
  1. `sources-parse`
  2. `providers-semantics`
  3. `cli-query`
  4. `source-detection`
  5. `drive-client`
  6. `repository`
- We are not ready to claim source/provider/harmonization semantics are exhaustively specified.
  The reruns removed reach failures, but they did not saturate the semantic space.
- The dominant structural issues are now:
  - survivor concentration in `polylogue.schemas.unified` and provider viewport shaping,
  - high survivor density in source parsing/detection and query orchestration,
  - source detection still carrying a narrower but still meaningful survivor cluster around ZIP filtering, emit paths, and provider sniffing,
  - drive-client transport/auth behavior still carrying a narrower but meaningful survivor cluster in auth/load/download helpers.
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
=======
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
- the follow-up source/query/repository reruns on commits `7e7c310037f9`,
  `a27de694650d`, and `b1f1d35bee28`
- the clean post-`005` rerun wave on commit `e759af23458d`

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
| `cli-query` | `e058c8240959` | 935 | 1007 | 20 | 0 | Clean post-concentration rerun: helper ownership is cleaner and survivors are still concentrated in `_async_execute_query` routing plus mutation-heavy modifier/delete paths. |
| `drive-client` | `37e26aba2d3d` | 581 | 299 | 3 | 0 | Clean post-tightening rerun improved kill rate again; the remaining residue is now mostly concentrated in credential loading, folder resolution, and metadata/download helpers rather than broad reach gaps. |
| `repository` | `b1f1d35bee28` | 568 | 74 | 66 | 0 | The concentrated read-model laws fixed the helper blind spot and cut survivor density sharply; the main remaining issue is timeout-heavy hydration/archive-stat coverage, not reach. |
| `site-builder` | `2bdb267e93b7` | 245 | 228 | 1 | 0 | Streaming/site generation laws materially improved this area and it is now mainly a regression guard. |
| `source-detection` | `38d65c004a2a` | 816 | 330 | 5 | 0 | Clean post-helper-contract rerun sharply improved kill rate while keeping reach complete; survivor mass is now much narrower and still concentrated in ZIP filtering, emit paths, and provider sniffing. |
| `providers-semantics` | `c0596770631e` | 805 | 504 | 2 | 0 | First focused concentration rerun after source semantic test collapse; reach is still clean and survivor mass remains concentrated in `extract_content_blocks`, `to_meta`, and fallback Claude Code harmonization. |
| `sources-parse` | `c0596770631e` | 3455 | 2482 | 10 | 0 | First focused concentration rerun after source semantic test collapse; the broad source surface remains fully reachable, with remaining work concentrated in drive-client seams and provider semantic survivor mass. |

### Readiness Call

- `004` is complete, and the immediate follow-up reruns are complete.
- The first post-`005` focused concentration rerun wave is also complete on clean SHA `c0596770631e`.
- We are ready for the next targeted law/property wave.
- The current highest-yield next fronts are:
  1. `sources-parse`
  2. `providers-semantics`
  3. `cli-query`
  4. `source-detection`
  5. `drive-client`
  6. `repository`
- We are not ready to claim source/provider/harmonization semantics are exhaustively specified.
  The reruns removed reach failures, but they did not saturate the semantic space.
- The dominant structural issues are now:
  - survivor concentration in `polylogue.schemas.unified` and provider viewport shaping,
  - high survivor density in source parsing/detection and query orchestration,
  - source detection still carrying a narrower but still meaningful survivor cluster around ZIP filtering, emit paths, and provider sniffing,
  - drive-client transport/auth behavior still carrying a narrower but meaningful survivor cluster in auth/load/download helpers.
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
>>>>>>> theirs

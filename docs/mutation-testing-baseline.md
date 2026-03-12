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

Recorded on `2026-03-13`.

### Repo-Wide Lint

- Command: `nix develop -c ruff check .`
- Result: `All checks passed!`

### Full Test Suite

- Command: `nix develop -c pytest -q -n 0`
- Result: `2798 passed, 1 warning in 198.33s`
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
- the clean second concentration-program reruns on commit `e07c4baebfe6`

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

### Second Concentration Program Clean Baselines

These are the latest clean reruns after the bulk source-edit concentration pass
on commit `e07c4baebfe6`. They supersede the older rows above for the touched
campaigns.

| Campaign | Commit | Killed | Survived | Timeout | Not checked | Interpretation |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `models` | `e07c4baebfe6` | 138 | 22 | 3 | 3 | Still a healthy semantic surface, but the concentrated owner file exposed a small remaining blind spot in `mainline_messages` / `project` alongside the old `_is_chatgpt_thinking` / `extract_thinking` survivor cluster. |
| `ui-core` | `e07c4baebfe6` | 11 | 15 | 0 | 0 | Reach is complete and runtime is cheap, but the remaining surface is underspecified around `PlainConsole.print`; this is now a small focused projection-contract problem rather than an ownership problem. |
| `cli-run` | `e07c4baebfe6` | 186 | 89 | 0 | 8 | The concentration pass removed the old timeout wall entirely, but `not_checked` persists in rich progress observers and survivor density is still concentrated in plain progress formatting/observer behavior. |
| `cli-query` | `e07c4baebfe6` | 992 | 964 | 6 | 0 | Slightly better than the earlier concentrated baseline, but still a high-survivor orchestration surface; remaining debt is mostly in `_async_execute_query`, modifier/delete action flows, and result-shaping routes. |
| `filters` | `e07c4baebfe6` | 453 | 55 | 89 | 0 | Still strong overall and fully reachable, but the new concentrated owner exposes more honest survivor mass in `pick`, `has_branches`, `_needs_content_loading`, and summary-sort helpers. Timeout mass remains the main remaining cost center. |
| `providers-semantics` | `e07c4baebfe6` | 785 | 489 | 2 | 0 | Reach stays complete, but this remains materially underspecified around shared content-block/meta extraction in `polylogue.schemas.unified` and provider viewport shaping. |
| `sources-parse` | `e07c4baebfe6` | 3538 | 2365 | 9 | 0 | Reach stays complete and timeout mass dropped further, but the survivor frontier is still large; the dominant clusters remain shared semantic extraction, Drive auth/filter helpers, JSON/file iteration, and source parsing orchestration. |

### Bulk Suite Densification Pre-Commit Baselines

These are the latest post-edit reruns from the bulk whole-suite densification
pass. They were recorded before committing the test rewrite batch, so the
campaign artifacts correctly report `Dirty = yes` and commit `eb43cfd48e98`
even though the measured tree is exactly the one that passed
`pytest -q -n 0` above.

Use these as the current empirical state of the touched fronts, but do not
confuse them with clean post-commit baselines.

| Campaign | Commit | Killed | Survived | Timeout | Not checked | Interpretation |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `filters` | `eb43cfd48e98` | 451 | 59 | 87 | 0 | Slight regression from the previous clean concentration rerun. The suite is smaller and still fully reaches the surface, but the remaining weak cluster is sharper and more honest around `pick`, `_needs_content_loading`, `has_branches`, and summary-sort/describe behavior. |
| `models` | `eb43cfd48e98` | 120 | 41 | 3 | 2 | The owner file is much smaller, but survivor density worsened and a small `mainline_messages` blind spot remains. The remaining debt is concentrated in `_is_chatgpt_thinking`, `extract_thinking`, and message-pair traversal. |
| `hybrid` | `eb43cfd48e98` | 113 | 20 | 0 | 3 | Slight improvement over the older clean baseline. The remaining gaps are narrow and still centered on provider construction and ranked-conversation resolution. |
| `repository` | `eb43cfd48e98` | 534 | 134 | 40 | 1 | Mixed result: timeout mass dropped substantially, but survivor density rose and one `conversation_exists_by_hash` mutant slipped to `not_checked`. The concentrated owner is smaller, but archive stats, save/search similarity, hydration, and record conversion still need stronger laws. |
| `cli-query` | `eb43cfd48e98` | 1022 | 932 | 8 | 0 | Best result of this pass. Kills improved materially and survivors dropped versus the prior clean rerun, while timeout mass stayed low. The remaining frontier is still mostly `_async_execute_query` and modifier/delete action routing. |
| `source-detection` | `eb43cfd48e98` | 819 | 331 | 1 | 0 | Slight regression in kill/survivor balance, but timeout noise dropped and reach stayed complete. The remaining concentrated weak spots are `filter_entries`, emit paths, and grouped/individual save-bundle behavior. |
| `providers-semantics` | `eb43cfd48e98` | 778 | 496 | 2 | 0 | Slight regression with the same overall story: reach is complete, but `polylogue.schemas.unified` still carries too much semantic authority. Shared block/meta/reasoning extraction remains under-specified. |
| `sources-parse` | `eb43cfd48e98` | 3531 | 2366 | 15 | 0 | Broad source reach remains complete, but this pass did not improve the broad source frontier overall. The main survivor mass is still shared semantic extraction plus Drive auth/filter/file iteration helpers, with modest timeout growth. |

### Maximal Remaining-Owner Compaction Pre-Commit Baselines

These are the latest post-edit reruns from the maximal remaining-owner
compaction pass. They were recorded before committing the test rewrite batch, so
the campaign artifacts correctly report `Dirty = yes` and commit
`c1fd5ce60e82` even though the measured tree is exactly the one that passed the
repo-wide verification baseline above.

Use these as the current empirical state of the touched fronts.

| Campaign | Commit | Killed | Survived | Timeout | Not checked | Interpretation |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `ui-core` | `c1fd5ce60e82` | 11 | 15 | 0 | 0 | Flat overall. The suite owner is smaller, but `PlainConsole.print` still dominates the remaining survivor mass. This is now a tiny focused projection-contract problem. |
| `cli-query` | `c1fd5ce60e82` | 1013 | 935 | 14 | 0 | Material improvement over the earlier concentrated baseline. The owner rewrite reduced survivors while keeping full reach; the remaining frontier is still `_async_execute_query` and modifier/delete action flow. |
| `drive-client` | `c1fd5ce60e82` | 584 | 296 | 3 | 0 | Slight improvement again. The remaining weak cluster is still credential loading, folder resolution, metadata/download helpers, and retry/auth transport behavior. |
| `source-detection` | `c1fd5ce60e82` | 934 | 217 | 0 | 0 | Strong improvement. The compacted source-owner contracts substantially improved kill density and removed timeout noise; the remaining cluster is concentrated in `filter_entries`, `_emit_individual`, and Drive payload sniffing. |
| `providers-semantics` | `c1fd5ce60e82` | 784 | 489 | 3 | 0 | Essentially flat to slightly worse than the previous pre-commit state. The suite is still fully reaching the surface, but `extract_content_blocks`, `to_meta`, and fallback Claude Code harmonization remain the dominant semantic weak spots. |
| `sources-parse` | `c1fd5ce60e82` | 3886 | 2019 | 7 | 0 | Strong improvement on the broadest source front. Kills increased substantially, survivors dropped materially, and timeout mass stayed low. The remaining frontier is now sharply concentrated in shared semantic extraction, Drive credential/transport helpers, file iteration, and content-block/reasoning helpers. |

### Refactor-First Follow-Up Pre-Commit Baselines

These are the latest focused reruns after frontloading production refactors in
`ConversationFilter` and `DriveClient`, then realigning the owner tests around
those seams. The artifacts were recorded before committing the batch, so they
correctly report `Dirty = yes` on commit `033d5d6f3130`.

| Campaign | Commit | Killed | Survived | Timeout | Not checked | Interpretation |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `filters` | `033d5d6f3130` | 457 | 45 | 97 | 0 | Modest but real improvement over the previous clean follow-up (`457/50/106/0`). The new execution-plan seam did not change reach, but it sharpened the remaining frontier into `pick`, `has_branches`, `count`, summary sorting, and description rendering rather than diffuse planner blindness. Timeout mass is still the main remaining cost center. |
| `drive-client` | `033d5d6f3130` | 564 | 286 | 0 | 0 | Strong improvement over the prior follow-up (`560/340/0/0`). Splitting cached-token loading, refresh transitions, folder-resolution helpers, and `DriveFile` construction reduced survivor density without introducing new runtime cost. The remaining cluster is narrower: `_load_credentials`, `resolve_folder_id`, `iter_json_files`, auth/manual flow, and service retry wiring. |

### Targeted Tightening Pre-Commit Baselines

These are the latest focused reruns after tightening the remaining
`ConversationFilter` picker/count/summary contracts and Drive auth/service
contracts on top of the refactor-first seams. The artifacts were recorded
before committing the batch, so they correctly report `Dirty = yes` on commit
`122d78613e3f`.

| Campaign | Commit | Killed | Survived | Timeout | Not checked | Interpretation |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `filters` | `122d78613e3f` | 470 | 38 | 91 | 0 | Best focused result so far on this front. Reach stays complete, survivors dropped again, and timeout mass shrank modestly. The remaining weak cluster is now sharply centered on `pick`, `_describe_active_filters`, `count`, `list_summaries`, and a smaller summary-sort tail. |
| `drive-client` | `122d78613e3f` | 571 | 279 | 0 | 0 | Another real improvement. Reach is complete, kills increased, and survivors fell again without any timeout cost. The remaining residue is mostly `_load_credentials`, `resolve_folder_id`, `iter_json_files`, `_load_cached_credentials`, `_run_manual_auth_flow`, `_refresh_credentials_if_needed`, and `_service_handle`. |

### Readiness Call

- `004` is complete, and the immediate follow-up reruns are complete.
- The first post-`005` focused concentration rerun wave is complete on clean SHAs `c0596770631e`, `47a9b1cff33f`, `a3440a0f1a4b`, and `027519a11118`.
- The bulk second concentration-program reruns are complete on clean SHA `e07c4baebfe6`.
- The whole-suite densification pass is also measured as a dirty pre-commit
  rerun set on `eb43cfd48e98`.
- The maximal remaining-owner compaction pass is now measured as a newer dirty
  pre-commit rerun set on `c1fd5ce60e82`.
- The refactor-first follow-up pass is now superseded by a newer targeted
  tightening rerun set on `122d78613e3f`.
- We are ready for the next targeted law/property wave.
- The current highest-yield next fronts are:
  1. `providers-semantics`
  2. `repository`
  3. `filters`
  4. `models`
  5. `cli-run`
  6. `drive-client`
  7. `cli-query`
  8. `sources-parse`
  9. `source-detection`
  10. `ui-core`
- We are not ready to claim source/provider/harmonization semantics are exhaustively specified.
  The reruns removed reach failures, but they did not saturate the semantic space.
- The dominant structural issues are now:
  - survivor concentration in `polylogue.schemas.unified` and provider viewport shaping,
  - `providers-semantics` concentration reduced duplication but also revealed lost mutation signal around adapter content-block/meta extraction,
  - query orchestration still carrying a large but now sharper survivor frontier,
  - source parsing no longer looks like a reach problem; it is a concentrated semantic-utility problem,
  - the whole-suite densification pass improved suite size and runtime more than mutation density in source/semantic fronts, so the next pass should bias toward stronger semantic oracles rather than further deletions alone,
  - rich/plain CLI progress observer behavior is now isolated enough to target directly,
  - `filters` concentration removed ownership noise but left a smaller, sharper `pick`/sort/loading survivor cluster,
  - source parsing now carrying a clearer residual cluster in shared extraction helpers plus a small timeout cluster around ChatGPT pair iteration / Drive transport,
  - source detection now carries a much narrower residual cluster around ZIP filtering, emit paths, and provider sniffing,
  - drive-client transport/auth behavior still carrying a narrower but meaningful survivor cluster in auth/load/download helpers.
  - the latest tightening pass substantially improved `filters` and
    `drive-client`, but both still have sharply concentrated semantic survivor
    tails rather than broad reach gaps.
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

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

## Focused Mutation Wave: `polylogue.lib.filters`

- Recorded on `2026-03-11`
- Targeted verification before the mutation run:
  - `nix develop -c ruff check tests/unit/core/test_filters.py tests/unit/core/test_filters_props.py tests/unit/core/test_filters_adv.py`
  - `nix develop -c pytest -q tests/unit/core/test_filters.py tests/unit/core/test_filters_props.py tests/unit/core/test_filters_adv.py`
  - Result: `252 passed in 23.38s`
- Mutation command: `nix develop -c mutmut run "polylogue.lib.filters*"`
- Result: `597` filter mutants checked at `4.27 mutations/second`

| Status | Count |
| --- | ---: |
| Killed | 457 |
| Survived | 35 |
| Timeout | 105 |
| No tests | 0 |

### Key Improvement

- Relative to the earlier narrow historical baseline for `polylogue.lib.filters` (`0 killed / 0 survived / 597 no tests / 0 timeout`), this wave eliminated the `no tests` blind spot entirely and converted most of the module into real signal.

### Dominant Survivor Clusters

| Function cluster | Surviving mutants |
| --- | ---: |
| `ConversationFilter.pick` | 10 |
| `ConversationFilter._describe_active_filters` | 6 |
| `ConversationFilter.list_summaries` | 6 |
| `ConversationFilter._apply_summary_filters` | 3 |
| `ConversationFilter.is_continuation` | 3 |
| `ConversationFilter.is_sidechain` | 3 |
| `ConversationFilter._has_post_filters` | 1 |
| `ConversationFilter.delete` | 1 |
| `ConversationFilter.has_branches` | 1 |
| `ConversationFilter.is_root` | 1 |

### Dominant Timeout Clusters

| Function cluster | Timeout mutants |
| --- | ---: |
| `ConversationFilter._apply_filters` | 28 |
| `ConversationFilter._apply_sort` | 20 |
| `ConversationFilter.__init__` | 15 |
| `ConversationFilter._fetch_generic` | 10 |
| `ConversationFilter._apply_common_filters` | 8 |
| `ConversationFilter._execute_pipeline` | 6 |
| `ConversationFilter._apply_sort_generic` | 5 |
| `ConversationFilter._effective_fetch_limit` | 5 |
| `ConversationFilter._needs_content_loading` | 4 |
| `ConversationFilter.list` | 4 |

### Interpretation

- The high-value structural wins from this wave are already realized:
  - `count()` is fully killed across its fast/summary/full paths.
  - `_fetch_summary_candidates()` is fully killed.
  - `first()`, `parent()`, and `count()` are fully killed.
  - `pick()` is materially improved but still the largest real survivor pocket.
- The remaining timeout mass is concentrated in larger pipeline-style helpers. Further progress there likely requires cheaper fixtures or smaller helper-level contracts, not just more end-to-end examples.
- The remaining survivors are no longer a broad blindness problem. They are a bounded residue in formatting/summary/branch-control behavior, which is the right point to pause and reassess instead of blindly accreting tests.

### Follow-up Hardening Pass

- After this baseline was recorded, a second narrow hardening pass added exact contracts for:
  - picker numbering, truncation, unknown-date rendering, and selection bounds
  - exact `list_summaries()` guidance text and summary-filter post-processing
  - negative branch predicate semantics
  - multi-delete counting
  - multi-value `describe()` joins and similarity truncation
- Those follow-up tests are verified by targeted pytest runs, but the exact post-pass focused mutation count is not recorded here yet because current broad mutmut configuration re-enters full-tree stats/clean-test phases after the test inventory changes, even for explicit survivor-key reruns.

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

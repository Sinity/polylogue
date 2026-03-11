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
- Configured mutation targets:
  - `polylogue/lib/models.py`
  - `polylogue/lib/filters.py`
  - `polylogue/lib/roles.py`
  - `polylogue/lib/timestamps.py`
  - `polylogue/lib/hashing.py`
  - `polylogue/lib/json.py`
  - `polylogue/storage/search_providers/fts5.py`
  - `polylogue/storage/search_providers/hybrid.py`
- Configured test selection:
  - `tests/unit/core/test_models.py`
  - `tests/unit/core/test_message_laws.py`
  - `tests/unit/core/test_properties.py`
  - `tests/unit/core/test_json.py`
  - `tests/unit/core/test_filters_props.py`
  - `tests/unit/core/test_json_laws.py`
  - `tests/unit/core/test_timestamp_guards.py`
  - `tests/unit/core/test_hashing.py`
  - `tests/unit/storage/test_fts5.py`
  - `tests/unit/storage/test_fts5_laws.py`
  - `tests/unit/storage/test_hybrid.py`
  - `tests/unit/storage/test_hybrid_laws.py`
- Harness note: mutmut now forces `pytest_add_cli_args = ["-n", "0", "-p", "no:randomly", "-p", "no:random-order", "--benchmark-disable"]` so it does not inherit repo-wide `pytest -n auto` and break inside xdist workers.

## Latest Mutation Results

- Command: `nix develop -c mutmut run`
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

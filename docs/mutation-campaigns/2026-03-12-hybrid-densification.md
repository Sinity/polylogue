# Mutmut Campaign: `hybrid`

- Recorded on `2026-03-12T09:57:11.806235+00:00`
- Commit: `eb43cfd48e989a58780604e9cbf9d3ba93700ff8`
- Worktree dirty: `yes`
- Description: Hybrid search fusion and ranked-conversation resolution
- Workspace: `/tmp/nix-shell.smrgrF/mutmut-hybrid-3pivorxf/repo`
- Command: `mutmut run`

## Scope

- Mutated paths: `polylogue/storage/search_providers/hybrid.py`
- Selected tests: `tests/unit/storage/test_hybrid.py`, `tests/unit/storage/test_hybrid_laws.py`

## Counts

| Status | Count |
| --- | ---: |
| Killed | 113 |
| Survived | 20 |
| Timeout | 0 |
| Not checked | 3 |
| Suspicious | 0 |
| Skipped | 0 |

- Runtime: `7.52s`
- Exit code: `0`

## Dominant Survivors

| Function | Count |
| --- | ---: |
| `search_conversations` | 6 |
| `search_scored` | 4 |
| `__init__` | 1 |
| `polylogue.storage.search_providers.hybrid.x__resolve_ranked_conversation_ids__mutmut_4` | 1 |
| `polylogue.storage.search_providers.hybrid.x__resolve_ranked_conversation_ids__mutmut_13` | 1 |
| `polylogue.storage.search_providers.hybrid.x__resolve_ranked_conversation_ids__mutmut_14` | 1 |
| `polylogue.storage.search_providers.hybrid.x_create_hybrid_provider__mutmut_1` | 1 |
| `polylogue.storage.search_providers.hybrid.x_create_hybrid_provider__mutmut_5` | 1 |
| `polylogue.storage.search_providers.hybrid.x_create_hybrid_provider__mutmut_6` | 1 |
| `polylogue.storage.search_providers.hybrid.x_create_hybrid_provider__mutmut_7` | 1 |

## Dominant Timeouts

| Function | Count |
| --- | ---: |
| _none_ | 0 |

## Dominant Not-Checked Clusters

| Function | Count |
| --- | ---: |
| `index` | 2 |
| `search` | 1 |

## Survivor Keys

- `polylogue.storage.search_providers.hybrid.xǁHybridSearchProviderǁ__init____mutmut_1`
- `polylogue.storage.search_providers.hybrid.xǁHybridSearchProviderǁsearch_scored__mutmut_11`
- `polylogue.storage.search_providers.hybrid.xǁHybridSearchProviderǁsearch_scored__mutmut_12`
- `polylogue.storage.search_providers.hybrid.xǁHybridSearchProviderǁsearch_scored__mutmut_14`
- `polylogue.storage.search_providers.hybrid.xǁHybridSearchProviderǁsearch_scored__mutmut_27`
- `polylogue.storage.search_providers.hybrid.xǁHybridSearchProviderǁsearch_conversations__mutmut_1`
- `polylogue.storage.search_providers.hybrid.xǁHybridSearchProviderǁsearch_conversations__mutmut_2`
- `polylogue.storage.search_providers.hybrid.xǁHybridSearchProviderǁsearch_conversations__mutmut_3`
- `polylogue.storage.search_providers.hybrid.xǁHybridSearchProviderǁsearch_conversations__mutmut_5`
- `polylogue.storage.search_providers.hybrid.xǁHybridSearchProviderǁsearch_conversations__mutmut_8`
- `polylogue.storage.search_providers.hybrid.xǁHybridSearchProviderǁsearch_conversations__mutmut_10`
- `polylogue.storage.search_providers.hybrid.x__resolve_ranked_conversation_ids__mutmut_4`
- `polylogue.storage.search_providers.hybrid.x__resolve_ranked_conversation_ids__mutmut_13`
- `polylogue.storage.search_providers.hybrid.x__resolve_ranked_conversation_ids__mutmut_14`
- `polylogue.storage.search_providers.hybrid.x_create_hybrid_provider__mutmut_1`
- `polylogue.storage.search_providers.hybrid.x_create_hybrid_provider__mutmut_5`
- `polylogue.storage.search_providers.hybrid.x_create_hybrid_provider__mutmut_6`
- `polylogue.storage.search_providers.hybrid.x_create_hybrid_provider__mutmut_7`
- `polylogue.storage.search_providers.hybrid.x_create_hybrid_provider__mutmut_9`
- `polylogue.storage.search_providers.hybrid.x_create_hybrid_provider__mutmut_12`

## Not-Checked Keys

- `polylogue.storage.search_providers.hybrid.xǁHybridSearchProviderǁindex__mutmut_1`
- `polylogue.storage.search_providers.hybrid.xǁHybridSearchProviderǁindex__mutmut_2`
- `polylogue.storage.search_providers.hybrid.xǁHybridSearchProviderǁsearch__mutmut_1`

## Source Worktree Status

- ` M tests/unit/cli/test_helpers.py`
- ` M tests/unit/cli/test_query_exec.py`
- ` M tests/unit/core/test_conversation_semantics.py`
- ` M tests/unit/core/test_dates.py`
- ` M tests/unit/core/test_filters_adv.py`
- ` M tests/unit/core/test_filters_props.py`
- ` M tests/unit/core/test_hashing.py`
- ` M tests/unit/core/test_message_laws.py`
- ` M tests/unit/core/test_models.py`
- ` D tests/unit/core/test_properties.py`
- ` M tests/unit/sources/test_claude.py`
- ` M tests/unit/sources/test_content_extraction.py`
- ` M tests/unit/sources/test_extraction.py`
- ` M tests/unit/sources/test_parsers.py`
- ` M tests/unit/sources/test_seeded_parser_contracts.py`
- ` M tests/unit/storage/test_backend.py`
- ` M tests/unit/storage/test_hybrid_laws.py`
- ` M tests/unit/storage/test_repository_laws.py`
- ` M tests/unit/storage/test_store_ops.py`
- ` M tests/unit/storage/test_vec.py`

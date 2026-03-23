# Mutmut Campaign: `filters`

- Recorded on `2026-03-12T09:53:08.674594+00:00`
- Commit: `eb43cfd48e989a58780604e9cbf9d3ba93700ff8`
- Worktree dirty: `yes`
- Description: ConversationFilter semantics and summary/picker contracts
- Workspace: `/tmp/nix-shell.7CFxpV/mutmut-filters-jxbih_ar/repo`
- Command: `mutmut run`

## Scope

- Mutated paths: `polylogue/lib/filters.py`
- Selected tests: `tests/unit/core/test_filters.py`, `tests/unit/core/test_filters_adv.py`, `tests/unit/core/test_filters_props.py`

## Counts

| Status | Count |
| --- | ---: |
| Killed | 451 |
| Survived | 59 |
| Timeout | 87 |
| Not checked | 0 |
| Suspicious | 0 |
| Skipped | 0 |

- Runtime: `133.97s`
- Exit code: `0`

## Dominant Survivors

| Function | Count |
| --- | ---: |
| `pick` | 21 |
| `_needs_content_loading` | 10 |
| `has_branches` | 8 |
| `_describe_active_filters` | 4 |
| `count` | 4 |
| `_apply_summary_sort` | 3 |
| `until` | 2 |
| `first` | 2 |
| `_apply_summary_filters` | 2 |
| `list_summaries` | 2 |

## Dominant Timeouts

| Function | Count |
| --- | ---: |
| `__init__` | 15 |
| `_apply_sort` | 14 |
| `list` | 12 |
| `_fetch_generic` | 9 |
| `_apply_filters` | 8 |
| `_fetch_candidates` | 7 |
| `_apply_common_filters` | 6 |
| `_apply_sort_generic` | 5 |
| `_effective_fetch_limit` | 5 |
| `_execute_pipeline` | 5 |

## Dominant Not-Checked Clusters

| Function | Count |
| --- | ---: |
| _none_ | 0 |

## Survivor Keys

- `polylogue.lib.filters.xǁConversationFilterǁsince__mutmut_6`
- `polylogue.lib.filters.xǁConversationFilterǁuntil__mutmut_4`
- `polylogue.lib.filters.xǁConversationFilterǁuntil__mutmut_5`
- `polylogue.lib.filters.xǁConversationFilterǁhas_branches__mutmut_3`
- `polylogue.lib.filters.xǁConversationFilterǁhas_branches__mutmut_6`
- `polylogue.lib.filters.xǁConversationFilterǁhas_branches__mutmut_7`
- `polylogue.lib.filters.xǁConversationFilterǁhas_branches__mutmut_8`
- `polylogue.lib.filters.xǁConversationFilterǁhas_branches__mutmut_9`
- `polylogue.lib.filters.xǁConversationFilterǁhas_branches__mutmut_10`
- `polylogue.lib.filters.xǁConversationFilterǁhas_branches__mutmut_11`
- `polylogue.lib.filters.xǁConversationFilterǁhas_branches__mutmut_12`
- `polylogue.lib.filters.xǁConversationFilterǁ_describe_active_filters__mutmut_13`
- `polylogue.lib.filters.xǁConversationFilterǁ_describe_active_filters__mutmut_16`
- `polylogue.lib.filters.xǁConversationFilterǁ_describe_active_filters__mutmut_19`
- `polylogue.lib.filters.xǁConversationFilterǁ_describe_active_filters__mutmut_47`
- `polylogue.lib.filters.xǁConversationFilterǁfirst__mutmut_2`
- `polylogue.lib.filters.xǁConversationFilterǁfirst__mutmut_3`
- `polylogue.lib.filters.xǁConversationFilterǁcount__mutmut_20`
- `polylogue.lib.filters.xǁConversationFilterǁcount__mutmut_21`
- `polylogue.lib.filters.xǁConversationFilterǁcount__mutmut_22`
- `polylogue.lib.filters.xǁConversationFilterǁcount__mutmut_23`
- `polylogue.lib.filters.xǁConversationFilterǁpick__mutmut_5`
- `polylogue.lib.filters.xǁConversationFilterǁpick__mutmut_9`
- `polylogue.lib.filters.xǁConversationFilterǁpick__mutmut_10`
- `polylogue.lib.filters.xǁConversationFilterǁpick__mutmut_11`
- ... 34 more

## Timeout Keys

- `polylogue.lib.filters.xǁConversationFilterǁ__init____mutmut_2`
- `polylogue.lib.filters.xǁConversationFilterǁ__init____mutmut_11`
- `polylogue.lib.filters.xǁConversationFilterǁ__init____mutmut_12`
- `polylogue.lib.filters.xǁConversationFilterǁ__init____mutmut_13`
- `polylogue.lib.filters.xǁConversationFilterǁ__init____mutmut_14`
- `polylogue.lib.filters.xǁConversationFilterǁ__init____mutmut_15`
- `polylogue.lib.filters.xǁConversationFilterǁ__init____mutmut_16`
- `polylogue.lib.filters.xǁConversationFilterǁ__init____mutmut_17`
- `polylogue.lib.filters.xǁConversationFilterǁ__init____mutmut_18`
- `polylogue.lib.filters.xǁConversationFilterǁ__init____mutmut_22`
- `polylogue.lib.filters.xǁConversationFilterǁ__init____mutmut_23`
- `polylogue.lib.filters.xǁConversationFilterǁ__init____mutmut_25`
- `polylogue.lib.filters.xǁConversationFilterǁ__init____mutmut_30`
- `polylogue.lib.filters.xǁConversationFilterǁ__init____mutmut_32`
- `polylogue.lib.filters.xǁConversationFilterǁ__init____mutmut_34`
- `polylogue.lib.filters.xǁConversationFilterǁ_apply_common_filters__mutmut_1`
- `polylogue.lib.filters.xǁConversationFilterǁ_apply_common_filters__mutmut_8`
- `polylogue.lib.filters.xǁConversationFilterǁ_apply_common_filters__mutmut_9`
- `polylogue.lib.filters.xǁConversationFilterǁ_apply_common_filters__mutmut_11`
- `polylogue.lib.filters.xǁConversationFilterǁ_apply_common_filters__mutmut_12`
- `polylogue.lib.filters.xǁConversationFilterǁ_apply_common_filters__mutmut_16`
- `polylogue.lib.filters.xǁConversationFilterǁ_apply_filters__mutmut_1`
- `polylogue.lib.filters.xǁConversationFilterǁ_apply_filters__mutmut_4`
- `polylogue.lib.filters.xǁConversationFilterǁ_apply_filters__mutmut_6`
- `polylogue.lib.filters.xǁConversationFilterǁ_apply_filters__mutmut_18`
- ... 62 more

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

## Notes

- Targets the historical largest no-test blind spot.
- Timeout tail is expected in filter pipeline helpers.

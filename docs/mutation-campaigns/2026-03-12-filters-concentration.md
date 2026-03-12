# Mutmut Campaign: `filters`

- Recorded on `2026-03-12T07:03:20.122895+00:00`
- Commit: `e07c4baebfe68af194fd423cd8fa0ecab515ca01`
- Worktree dirty: `no`
- Description: ConversationFilter semantics and summary/picker contracts
- Workspace: `/tmp/nix-shell.7VouyF/mutmut-filters-vcqxcpzy/repo`
- Command: `mutmut run`

## Scope

- Mutated paths: `polylogue/lib/filters.py`
- Selected tests: `tests/unit/core/test_filters.py`, `tests/unit/core/test_filters_adv.py`, `tests/unit/core/test_filters_props.py`

## Counts

| Status | Count |
| --- | ---: |
| Killed | 453 |
| Survived | 55 |
| Timeout | 89 |
| Not checked | 0 |
| Suspicious | 0 |
| Skipped | 0 |

- Runtime: `160.77s`
- Exit code: `0`

## Dominant Survivors

| Function | Count |
| --- | ---: |
| `pick` | 21 |
| `has_branches` | 9 |
| `count` | 4 |
| `_needs_content_loading` | 4 |
| `_apply_summary_filters` | 3 |
| `_apply_summary_sort` | 3 |
| `until` | 2 |
| `first` | 2 |
| `list_summaries` | 2 |
| `since` | 1 |

## Dominant Timeouts

| Function | Count |
| --- | ---: |
| `_apply_sort` | 20 |
| `__init__` | 15 |
| `list` | 12 |
| `_apply_filters` | 10 |
| `_fetch_generic` | 9 |
| `_apply_common_filters` | 6 |
| `_apply_sort_generic` | 5 |
| `_effective_fetch_limit` | 5 |
| `_execute_pipeline` | 4 |
| `_fetch_candidates` | 3 |

## Dominant Not-Checked Clusters

| Function | Count |
| --- | ---: |
| _none_ | 0 |

## Survivor Keys

- `polylogue.lib.filters.xǁConversationFilterǁsince__mutmut_6`
- `polylogue.lib.filters.xǁConversationFilterǁuntil__mutmut_4`
- `polylogue.lib.filters.xǁConversationFilterǁuntil__mutmut_5`
- `polylogue.lib.filters.xǁConversationFilterǁis_continuation__mutmut_1`
- `polylogue.lib.filters.xǁConversationFilterǁis_sidechain__mutmut_1`
- `polylogue.lib.filters.xǁConversationFilterǁis_root__mutmut_1`
- `polylogue.lib.filters.xǁConversationFilterǁparent__mutmut_2`
- `polylogue.lib.filters.xǁConversationFilterǁhas_branches__mutmut_1`
- `polylogue.lib.filters.xǁConversationFilterǁhas_branches__mutmut_3`
- `polylogue.lib.filters.xǁConversationFilterǁhas_branches__mutmut_6`
- `polylogue.lib.filters.xǁConversationFilterǁhas_branches__mutmut_7`
- `polylogue.lib.filters.xǁConversationFilterǁhas_branches__mutmut_8`
- `polylogue.lib.filters.xǁConversationFilterǁhas_branches__mutmut_9`
- `polylogue.lib.filters.xǁConversationFilterǁhas_branches__mutmut_10`
- `polylogue.lib.filters.xǁConversationFilterǁhas_branches__mutmut_11`
- `polylogue.lib.filters.xǁConversationFilterǁhas_branches__mutmut_12`
- `polylogue.lib.filters.xǁConversationFilterǁfirst__mutmut_2`
- `polylogue.lib.filters.xǁConversationFilterǁfirst__mutmut_3`
- `polylogue.lib.filters.xǁConversationFilterǁcount__mutmut_20`
- `polylogue.lib.filters.xǁConversationFilterǁcount__mutmut_21`
- `polylogue.lib.filters.xǁConversationFilterǁcount__mutmut_22`
- `polylogue.lib.filters.xǁConversationFilterǁcount__mutmut_23`
- `polylogue.lib.filters.xǁConversationFilterǁpick__mutmut_5`
- `polylogue.lib.filters.xǁConversationFilterǁpick__mutmut_9`
- `polylogue.lib.filters.xǁConversationFilterǁpick__mutmut_10`
- ... 30 more

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
- `polylogue.lib.filters.xǁConversationFilterǁ_apply_filters__mutmut_8`
- ... 64 more

## Notes

- Targets the historical largest no-test blind spot.
- Timeout tail is expected in filter pipeline helpers.

# Mutmut Campaign: `filters`

- Recorded on `2026-03-13T01:17:53.474410+00:00`
- Commit: `122d78613e3fd400352d0faa98e47e8d6bda12f2`
- Worktree dirty: `yes`
- Description: ConversationFilter semantics and summary/picker contracts
- Workspace: `/tmp/nix-shell.aU8gdR/mutmut-filters-na6okb2p/repo`
- Command: `mutmut run`

## Scope

- Mutated paths: `polylogue/lib/filters.py`
- Selected tests: `tests/unit/core/test_filters.py`, `tests/unit/core/test_filters_adv.py`, `tests/unit/core/test_filters_props.py`

## Counts

| Status | Count |
| --- | ---: |
| Killed | 470 |
| Survived | 38 |
| Timeout | 91 |
| Not checked | 0 |
| Suspicious | 0 |
| Skipped | 0 |

- Runtime: `142.62s`
- Exit code: `0`

## Dominant Survivors

| Function | Count |
| --- | ---: |
| `pick` | 18 |
| `_describe_active_filters` | 4 |
| `until` | 2 |
| `_can_count_in_sql` | 2 |
| `first` | 2 |
| `count` | 2 |
| `_apply_summary_filters` | 2 |
| `_apply_summary_sort` | 2 |
| `list_summaries` | 2 |
| `since` | 1 |

## Dominant Timeouts

| Function | Count |
| --- | ---: |
| `__init__` | 15 |
| `_apply_sort` | 14 |
| `list` | 12 |
| `_fetch_generic` | 10 |
| `_needs_content_loading` | 10 |
| `_apply_common_filters` | 6 |
| `_apply_sort_generic` | 5 |
| `_effective_fetch_limit` | 5 |
| `_apply_filters` | 4 |
| `_fetch_candidates` | 4 |

## Dominant Not-Checked Clusters

| Function | Count |
| --- | ---: |
| _none_ | 0 |

## Survivor Keys

- `polylogue.lib.filters.xǁConversationFilterǁsince__mutmut_6`
- `polylogue.lib.filters.xǁConversationFilterǁuntil__mutmut_4`
- `polylogue.lib.filters.xǁConversationFilterǁuntil__mutmut_5`
- `polylogue.lib.filters.xǁConversationFilterǁ_describe_active_filters__mutmut_13`
- `polylogue.lib.filters.xǁConversationFilterǁ_describe_active_filters__mutmut_16`
- `polylogue.lib.filters.xǁConversationFilterǁ_describe_active_filters__mutmut_19`
- `polylogue.lib.filters.xǁConversationFilterǁ_describe_active_filters__mutmut_47`
- `polylogue.lib.filters.xǁConversationFilterǁ_can_count_in_sql__mutmut_6`
- `polylogue.lib.filters.xǁConversationFilterǁ_can_count_in_sql__mutmut_7`
- `polylogue.lib.filters.xǁConversationFilterǁfirst__mutmut_2`
- `polylogue.lib.filters.xǁConversationFilterǁfirst__mutmut_3`
- `polylogue.lib.filters.xǁConversationFilterǁcount__mutmut_4`
- `polylogue.lib.filters.xǁConversationFilterǁcount__mutmut_7`
- `polylogue.lib.filters.xǁConversationFilterǁdelete__mutmut_7`
- `polylogue.lib.filters.xǁConversationFilterǁpick__mutmut_5`
- `polylogue.lib.filters.xǁConversationFilterǁpick__mutmut_9`
- `polylogue.lib.filters.xǁConversationFilterǁpick__mutmut_10`
- `polylogue.lib.filters.xǁConversationFilterǁpick__mutmut_11`
- `polylogue.lib.filters.xǁConversationFilterǁpick__mutmut_12`
- `polylogue.lib.filters.xǁConversationFilterǁpick__mutmut_13`
- `polylogue.lib.filters.xǁConversationFilterǁpick__mutmut_14`
- `polylogue.lib.filters.xǁConversationFilterǁpick__mutmut_16`
- `polylogue.lib.filters.xǁConversationFilterǁpick__mutmut_17`
- `polylogue.lib.filters.xǁConversationFilterǁpick__mutmut_18`
- `polylogue.lib.filters.xǁConversationFilterǁpick__mutmut_19`
- ... 13 more

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
- `polylogue.lib.filters.xǁConversationFilterǁ_apply_filters__mutmut_25`
- ... 66 more

## Source Worktree Status

- ` M polylogue/cli/click_app.py`
- ` M tests/unit/core/test_filters.py`
- ` M tests/unit/core/test_filters_adv.py`
- ` M tests/unit/sources/test_drive_client_laws.py`

## Notes

- Targets the historical largest no-test blind spot.
- Timeout tail is expected in filter pipeline helpers.

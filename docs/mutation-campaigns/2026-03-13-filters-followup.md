# Mutmut Campaign: `filters`

- Recorded on `2026-03-13T01:02:52.561695+00:00`
- Commit: `033d5d6f3130a0a178be6c93b2a154f23d7c3ac0`
- Worktree dirty: `yes`
- Description: ConversationFilter semantics and summary/picker contracts
- Workspace: `/tmp/nix-shell.O6rYx4/mutmut-filters-txdiy5k4/repo`
- Command: `mutmut run`

## Scope

- Mutated paths: `polylogue/lib/filters.py`
- Selected tests: `tests/unit/core/test_filters.py`, `tests/unit/core/test_filters_adv.py`, `tests/unit/core/test_filters_props.py`

## Counts

| Status | Count |
| --- | ---: |
| Killed | 457 |
| Survived | 45 |
| Timeout | 97 |
| Not checked | 0 |
| Suspicious | 0 |
| Skipped | 0 |

- Runtime: `140.28s`
- Exit code: `0`

## Dominant Survivors

| Function | Count |
| --- | ---: |
| `pick` | 18 |
| `has_branches` | 4 |
| `_describe_active_filters` | 4 |
| `count` | 4 |
| `_apply_summary_sort` | 3 |
| `until` | 2 |
| `_can_count_in_sql` | 2 |
| `first` | 2 |
| `_apply_summary_filters` | 2 |
| `list_summaries` | 2 |

## Dominant Timeouts

| Function | Count |
| --- | ---: |
| `_apply_sort` | 20 |
| `__init__` | 15 |
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
- `polylogue.lib.filters.xǁConversationFilterǁhas_branches__mutmut_3`
- `polylogue.lib.filters.xǁConversationFilterǁhas_branches__mutmut_4`
- `polylogue.lib.filters.xǁConversationFilterǁhas_branches__mutmut_5`
- `polylogue.lib.filters.xǁConversationFilterǁhas_branches__mutmut_6`
- `polylogue.lib.filters.xǁConversationFilterǁ_describe_active_filters__mutmut_13`
- `polylogue.lib.filters.xǁConversationFilterǁ_describe_active_filters__mutmut_16`
- `polylogue.lib.filters.xǁConversationFilterǁ_describe_active_filters__mutmut_19`
- `polylogue.lib.filters.xǁConversationFilterǁ_describe_active_filters__mutmut_47`
- `polylogue.lib.filters.xǁConversationFilterǁ_can_count_in_sql__mutmut_6`
- `polylogue.lib.filters.xǁConversationFilterǁ_can_count_in_sql__mutmut_7`
- `polylogue.lib.filters.xǁConversationFilterǁfirst__mutmut_2`
- `polylogue.lib.filters.xǁConversationFilterǁfirst__mutmut_3`
- `polylogue.lib.filters.xǁConversationFilterǁcount__mutmut_4`
- `polylogue.lib.filters.xǁConversationFilterǁcount__mutmut_5`
- `polylogue.lib.filters.xǁConversationFilterǁcount__mutmut_6`
- `polylogue.lib.filters.xǁConversationFilterǁcount__mutmut_7`
- `polylogue.lib.filters.xǁConversationFilterǁdelete__mutmut_7`
- `polylogue.lib.filters.xǁConversationFilterǁpick__mutmut_5`
- `polylogue.lib.filters.xǁConversationFilterǁpick__mutmut_9`
- `polylogue.lib.filters.xǁConversationFilterǁpick__mutmut_10`
- `polylogue.lib.filters.xǁConversationFilterǁpick__mutmut_11`
- `polylogue.lib.filters.xǁConversationFilterǁpick__mutmut_12`
- ... 20 more

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
- ... 72 more

## Source Worktree Status

- ` M polylogue/lib/filters.py`
- ` M polylogue/sources/drive_client.py`
- ` M tests/unit/core/test_filters.py`
- ` M tests/unit/core/test_filters_adv.py`
- ` M tests/unit/core/test_filters_props.py`
- ` M tests/unit/sources/test_drive_client_laws.py`
- ` M tests/unit/sources/test_drive_utils.py`

## Notes

- Targets the historical largest no-test blind spot.
- Timeout tail is expected in filter pipeline helpers.

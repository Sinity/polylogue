# Mutmut Campaign: `models`

- Recorded on `2026-03-12T09:56:43.482654+00:00`
- Commit: `eb43cfd48e989a58780604e9cbf9d3ba93700ff8`
- Worktree dirty: `yes`
- Description: Message/Conversation semantic helpers and pairing logic
- Workspace: `/tmp/nix-shell.hWWMSo/mutmut-models-_p8lwfeb/repo`
- Command: `mutmut run`

## Scope

- Mutated paths: `polylogue/lib/models.py`
- Selected tests: `tests/unit/core/test_models.py`, `tests/unit/core/test_message_laws.py`, `tests/unit/core/test_conversation_semantics.py`

## Counts

| Status | Count |
| --- | ---: |
| Killed | 120 |
| Survived | 41 |
| Timeout | 3 |
| Not checked | 2 |
| Suspicious | 0 |
| Skipped | 0 |

- Runtime: `27.95s`
- Exit code: `0`

## Dominant Survivors

| Function | Count |
| --- | ---: |
| `_is_chatgpt_thinking` | 22 |
| `extract_thinking` | 15 |
| `iter_pairs` | 2 |
| `to_text` | 2 |

## Dominant Timeouts

| Function | Count |
| --- | ---: |
| `iter_pairs` | 3 |

## Dominant Not-Checked Clusters

| Function | Count |
| --- | ---: |
| `mainline_messages` | 2 |

## Survivor Keys

- `polylogue.lib.models.xǁMessageǁ_is_chatgpt_thinking__mutmut_5`
- `polylogue.lib.models.xǁMessageǁ_is_chatgpt_thinking__mutmut_7`
- `polylogue.lib.models.xǁMessageǁ_is_chatgpt_thinking__mutmut_11`
- `polylogue.lib.models.xǁMessageǁ_is_chatgpt_thinking__mutmut_14`
- `polylogue.lib.models.xǁMessageǁ_is_chatgpt_thinking__mutmut_16`
- `polylogue.lib.models.xǁMessageǁ_is_chatgpt_thinking__mutmut_21`
- `polylogue.lib.models.xǁMessageǁ_is_chatgpt_thinking__mutmut_23`
- `polylogue.lib.models.xǁMessageǁ_is_chatgpt_thinking__mutmut_26`
- `polylogue.lib.models.xǁMessageǁ_is_chatgpt_thinking__mutmut_30`
- `polylogue.lib.models.xǁMessageǁ_is_chatgpt_thinking__mutmut_31`
- `polylogue.lib.models.xǁMessageǁ_is_chatgpt_thinking__mutmut_33`
- `polylogue.lib.models.xǁMessageǁ_is_chatgpt_thinking__mutmut_34`
- `polylogue.lib.models.xǁMessageǁ_is_chatgpt_thinking__mutmut_35`
- `polylogue.lib.models.xǁMessageǁ_is_chatgpt_thinking__mutmut_36`
- `polylogue.lib.models.xǁMessageǁ_is_chatgpt_thinking__mutmut_38`
- `polylogue.lib.models.xǁMessageǁ_is_chatgpt_thinking__mutmut_39`
- `polylogue.lib.models.xǁMessageǁ_is_chatgpt_thinking__mutmut_40`
- `polylogue.lib.models.xǁMessageǁ_is_chatgpt_thinking__mutmut_41`
- `polylogue.lib.models.xǁMessageǁ_is_chatgpt_thinking__mutmut_42`
- `polylogue.lib.models.xǁMessageǁ_is_chatgpt_thinking__mutmut_43`
- `polylogue.lib.models.xǁMessageǁ_is_chatgpt_thinking__mutmut_44`
- `polylogue.lib.models.xǁMessageǁ_is_chatgpt_thinking__mutmut_45`
- `polylogue.lib.models.xǁMessageǁextract_thinking__mutmut_14`
- `polylogue.lib.models.xǁMessageǁextract_thinking__mutmut_16`
- `polylogue.lib.models.xǁMessageǁextract_thinking__mutmut_17`
- ... 16 more

## Timeout Keys

- `polylogue.lib.models.xǁConversationǁiter_pairs__mutmut_16`
- `polylogue.lib.models.xǁConversationǁiter_pairs__mutmut_17`
- `polylogue.lib.models.xǁConversationǁiter_pairs__mutmut_20`

## Not-Checked Keys

- `polylogue.lib.models.xǁConversationǁmainline_messages__mutmut_1`
- `polylogue.lib.models.xǁConversationǁmainline_messages__mutmut_2`

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

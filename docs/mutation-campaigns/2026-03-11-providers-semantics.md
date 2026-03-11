# Mutmut Campaign: `providers-semantics`

- Recorded on `2026-03-11T10:01:27.813065+00:00`
- Commit: `2bdb267e93b79f1f0dc863f86b5ed859e4e0dbdd`
- Worktree dirty: `no`
- Description: Provider semantic extraction, harmonization, and viewport contracts
- Workspace: `/tmp/nix-shell.rzHew1/mutmut-providers-semantics-1bj5aku1/repo`
- Command: `mutmut run`

## Scope

- Mutated paths: `polylogue/sources/providers`, `polylogue/schemas/unified.py`
- Selected tests: `tests/unit/sources/test_parse_laws.py`, `tests/unit/sources/test_harmonization_contracts.py`, `tests/unit/sources/test_provider_viewport_laws.py`, `tests/unit/sources/test_providers.py`, `tests/unit/sources/test_null_guard_properties.py`, `tests/unit/sources/test_models.py`

## Counts

| Status | Count |
| --- | ---: |
| Killed | 415 |
| Survived | 652 |
| Timeout | 3 |
| Not checked | 112 |
| Suspicious | 0 |
| Skipped | 0 |

- Runtime: `50.22s`
- Exit code: `0`

## Dominant Survivors

| Function | Count |
| --- | ---: |
| `extract_content_blocks` | 139 |
| `to_meta` | 70 |
| `to_content_blocks` | 34 |
| `extract_reasoning_traces` | 20 |
| `extract_tool_calls` | 4 |
| `polylogue.schemas.unified.x_extract_reasoning_traces__mutmut_9` | 1 |
| `polylogue.schemas.unified.x_extract_reasoning_traces__mutmut_18` | 1 |
| `polylogue.schemas.unified.x_extract_reasoning_traces__mutmut_19` | 1 |
| `polylogue.schemas.unified.x_extract_reasoning_traces__mutmut_20` | 1 |
| `polylogue.schemas.unified.x_extract_reasoning_traces__mutmut_35` | 1 |

## Dominant Timeouts

| Function | Count |
| --- | ---: |
| `iter_user_assistant_pairs` | 3 |

## Dominant Not-Checked Clusters

| Function | Count |
| --- | ---: |
| `polylogue.schemas.unified.x__missing_role__mutmut_1` | 1 |
| `polylogue.schemas.unified.x__missing_role__mutmut_2` | 1 |
| `polylogue.schemas.unified.x__missing_role__mutmut_3` | 1 |
| `polylogue.schemas.unified.x__missing_role__mutmut_4` | 1 |
| `polylogue.schemas.unified.x_extract_from_provider_meta__mutmut_1` | 1 |
| `polylogue.schemas.unified.x_extract_from_provider_meta__mutmut_2` | 1 |
| `polylogue.schemas.unified.x_extract_from_provider_meta__mutmut_3` | 1 |
| `polylogue.schemas.unified.x_extract_from_provider_meta__mutmut_4` | 1 |
| `polylogue.schemas.unified.x_extract_from_provider_meta__mutmut_5` | 1 |
| `polylogue.schemas.unified.x_extract_from_provider_meta__mutmut_6` | 1 |

## Survivor Keys

- `polylogue.schemas.unified.x_extract_reasoning_traces__mutmut_9`
- `polylogue.schemas.unified.x_extract_reasoning_traces__mutmut_18`
- `polylogue.schemas.unified.x_extract_reasoning_traces__mutmut_19`
- `polylogue.schemas.unified.x_extract_reasoning_traces__mutmut_20`
- `polylogue.schemas.unified.x_extract_reasoning_traces__mutmut_35`
- `polylogue.schemas.unified.x_extract_tool_calls__mutmut_14`
- `polylogue.schemas.unified.x_extract_tool_calls__mutmut_16`
- `polylogue.schemas.unified.x_extract_tool_calls__mutmut_19`
- `polylogue.schemas.unified.x_extract_tool_calls__mutmut_22`
- `polylogue.schemas.unified.x_extract_tool_calls__mutmut_24`
- `polylogue.schemas.unified.x_extract_tool_calls__mutmut_37`
- `polylogue.schemas.unified.x_extract_tool_calls__mutmut_39`
- `polylogue.schemas.unified.x_extract_content_blocks__mutmut_7`
- `polylogue.schemas.unified.x_extract_content_blocks__mutmut_9`
- `polylogue.schemas.unified.x_extract_content_blocks__mutmut_12`
- `polylogue.schemas.unified.x_extract_content_blocks__mutmut_13`
- `polylogue.schemas.unified.x_extract_content_blocks__mutmut_19`
- `polylogue.schemas.unified.x_extract_content_blocks__mutmut_22`
- `polylogue.schemas.unified.x_extract_content_blocks__mutmut_23`
- `polylogue.schemas.unified.x_extract_content_blocks__mutmut_24`
- `polylogue.schemas.unified.x_extract_content_blocks__mutmut_25`
- `polylogue.schemas.unified.x_extract_content_blocks__mutmut_26`
- `polylogue.schemas.unified.x_extract_content_blocks__mutmut_32`
- `polylogue.schemas.unified.x_extract_content_blocks__mutmut_35`
- `polylogue.schemas.unified.x_extract_content_blocks__mutmut_36`
- ... 627 more

## Timeout Keys

- `polylogue.sources.providers.chatgpt.xǁChatGPTConversationǁiter_user_assistant_pairs__mutmut_18`
- `polylogue.sources.providers.chatgpt.xǁChatGPTConversationǁiter_user_assistant_pairs__mutmut_19`
- `polylogue.sources.providers.chatgpt.xǁChatGPTConversationǁiter_user_assistant_pairs__mutmut_21`

## Not-Checked Keys

- `polylogue.schemas.unified.x__missing_role__mutmut_1`
- `polylogue.schemas.unified.x__missing_role__mutmut_2`
- `polylogue.schemas.unified.x__missing_role__mutmut_3`
- `polylogue.schemas.unified.x__missing_role__mutmut_4`
- `polylogue.schemas.unified.x_extract_from_provider_meta__mutmut_1`
- `polylogue.schemas.unified.x_extract_from_provider_meta__mutmut_2`
- `polylogue.schemas.unified.x_extract_from_provider_meta__mutmut_3`
- `polylogue.schemas.unified.x_extract_from_provider_meta__mutmut_4`
- `polylogue.schemas.unified.x_extract_from_provider_meta__mutmut_5`
- `polylogue.schemas.unified.x_extract_from_provider_meta__mutmut_6`
- `polylogue.schemas.unified.x_extract_from_provider_meta__mutmut_7`
- `polylogue.schemas.unified.x_extract_from_provider_meta__mutmut_8`
- `polylogue.schemas.unified.x_extract_from_provider_meta__mutmut_9`
- `polylogue.schemas.unified.x_extract_from_provider_meta__mutmut_10`
- `polylogue.schemas.unified.x_extract_from_provider_meta__mutmut_11`
- `polylogue.schemas.unified.x_extract_from_provider_meta__mutmut_12`
- `polylogue.schemas.unified.x_extract_from_provider_meta__mutmut_13`
- `polylogue.schemas.unified.x_is_message_record__mutmut_1`
- `polylogue.schemas.unified.x_is_message_record__mutmut_2`
- `polylogue.schemas.unified.x_is_message_record__mutmut_3`
- `polylogue.schemas.unified.x_is_message_record__mutmut_4`
- `polylogue.schemas.unified.x_is_message_record__mutmut_5`
- `polylogue.schemas.unified.x_is_message_record__mutmut_6`
- `polylogue.schemas.unified.x_is_message_record__mutmut_7`
- `polylogue.schemas.unified.x_is_message_record__mutmut_8`
- ... 87 more

## Notes

- Directly relevant to the next provider-law wave.

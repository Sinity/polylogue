# Mutmut Campaign: `providers-semantics`

- Recorded on `2026-03-11T18:11:49.054310+00:00`
- Commit: `a27de694650ddb2c16aa40338cd2bf5bb0ab9719`
- Worktree dirty: `no`
- Description: Provider semantic extraction, harmonization, and viewport contracts
- Workspace: `/tmp/nix-shell.VVjV3k/mutmut-providers-semantics-aiyt_hu2/repo`
- Command: `mutmut run`

## Scope

- Mutated paths: `polylogue/sources/providers`, `polylogue/schemas/unified.py`
- Selected tests: `tests/unit/sources/test_parse_laws.py`, `tests/unit/sources/test_harmonization_contracts.py`, `tests/unit/sources/test_provider_viewport_laws.py`, `tests/unit/sources/test_providers.py`, `tests/unit/sources/test_viewport_protocol.py`, `tests/unit/sources/test_extraction.py`, `tests/unit/sources/test_unified_semantic_laws.py`, `tests/unit/sources/test_null_guard_properties.py`, `tests/unit/sources/test_models.py`

## Counts

| Status | Count |
| --- | ---: |
| Killed | 1009 |
| Survived | 620 |
| Timeout | 2 |
| Not checked | 0 |
| Suspicious | 0 |
| Skipped | 0 |

- Runtime: `62.23s`
- Exit code: `0`

## Dominant Survivors

| Function | Count |
| --- | ---: |
| `extract_content_blocks` | 124 |
| `to_meta` | 45 |
| `to_content_blocks` | 19 |
| `extract_reasoning_traces` | 18 |
| `extract_tool_calls` | 3 |
| `polylogue.schemas.unified.x__missing_role__mutmut_2` | 1 |
| `polylogue.schemas.unified.x_extract_reasoning_traces__mutmut_9` | 1 |
| `polylogue.schemas.unified.x_extract_reasoning_traces__mutmut_18` | 1 |
| `polylogue.schemas.unified.x_extract_reasoning_traces__mutmut_19` | 1 |
| `polylogue.schemas.unified.x_extract_reasoning_traces__mutmut_20` | 1 |

## Dominant Timeouts

| Function | Count |
| --- | ---: |
| `iter_user_assistant_pairs` | 2 |

## Dominant Not-Checked Clusters

| Function | Count |
| --- | ---: |
| _none_ | 0 |

## Survivor Keys

- `polylogue.schemas.unified.x__missing_role__mutmut_2`
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
- ... 595 more

## Timeout Keys

- `polylogue.sources.providers.chatgpt.xǁChatGPTConversationǁiter_user_assistant_pairs__mutmut_18`
- `polylogue.sources.providers.chatgpt.xǁChatGPTConversationǁiter_user_assistant_pairs__mutmut_21`

## Notes

- Directly relevant to the next provider-law wave.

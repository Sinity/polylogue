# Mutmut Campaign: `providers-semantics`

- Recorded on `2026-03-11T20:12:38.428888+00:00`
- Commit: `c0596770631e91333026afbe4414d4249a934e72`
- Worktree dirty: `no`
- Description: Provider semantic extraction, harmonization, and viewport contracts
- Workspace: `/tmp/nix-shell.honHyV/mutmut-providers-semantics-zwhxu__s/repo`
- Command: `mutmut run`

## Scope

- Mutated paths: `polylogue/sources/providers`, `polylogue/schemas/unified.py`
- Selected tests: `tests/unit/sources/test_parse_laws.py`, `tests/unit/sources/test_harmonization_contracts.py`, `tests/unit/sources/test_provider_viewport_laws.py`, `tests/unit/sources/test_providers.py`, `tests/unit/sources/test_viewport_protocol.py`, `tests/unit/sources/test_extraction.py`, `tests/unit/sources/test_unified_semantic_laws.py`, `tests/unit/sources/test_null_guard_properties.py`, `tests/unit/sources/test_models.py`

## Counts

| Status | Count |
| --- | ---: |
| Killed | 805 |
| Survived | 504 |
| Timeout | 2 |
| Not checked | 0 |
| Suspicious | 0 |
| Skipped | 0 |

- Runtime: `48.27s`
- Exit code: `0`

## Dominant Survivors

| Function | Count |
| --- | ---: |
| `extract_content_blocks` | 99 |
| `to_meta` | 39 |
| `extract_reasoning_traces` | 23 |
| `to_content_blocks` | 7 |
| `extract_tool_calls` | 3 |
| `polylogue.schemas.unified.x__missing_role__mutmut_2` | 1 |
| `polylogue.schemas.unified.x__harmonize_viewport_message__mutmut_27` | 1 |
| `polylogue.schemas.unified.x__extract_with_adapter__mutmut_6` | 1 |
| `polylogue.schemas.unified.x__extract_with_adapter__mutmut_7` | 1 |
| `polylogue.schemas.unified.x__extract_with_adapter__mutmut_8` | 1 |

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
- `polylogue.schemas.unified.x__harmonize_viewport_message__mutmut_27`
- `polylogue.schemas.unified.x__extract_with_adapter__mutmut_6`
- `polylogue.schemas.unified.x__extract_with_adapter__mutmut_7`
- `polylogue.schemas.unified.x__extract_with_adapter__mutmut_8`
- `polylogue.schemas.unified.x__extract_with_adapter__mutmut_9`
- `polylogue.schemas.unified.x__extract_with_adapter__mutmut_49`
- `polylogue.schemas.unified.x__fallback_extract_claude_code__mutmut_3`
- `polylogue.schemas.unified.x__fallback_extract_claude_code__mutmut_5`
- `polylogue.schemas.unified.x__fallback_extract_claude_code__mutmut_8`
- `polylogue.schemas.unified.x__fallback_extract_claude_code__mutmut_9`
- `polylogue.schemas.unified.x__fallback_extract_claude_code__mutmut_10`
- `polylogue.schemas.unified.x__fallback_extract_claude_code__mutmut_12`
- `polylogue.schemas.unified.x__fallback_extract_claude_code__mutmut_13`
- `polylogue.schemas.unified.x__fallback_extract_claude_code__mutmut_14`
- `polylogue.schemas.unified.x__fallback_extract_claude_code__mutmut_18`
- `polylogue.schemas.unified.x__fallback_extract_claude_code__mutmut_22`
- `polylogue.schemas.unified.x__fallback_extract_claude_code__mutmut_23`
- `polylogue.schemas.unified.x__fallback_extract_claude_code__mutmut_24`
- `polylogue.schemas.unified.x__fallback_extract_claude_code__mutmut_25`
- `polylogue.schemas.unified.x__fallback_extract_claude_code__mutmut_26`
- `polylogue.schemas.unified.x__fallback_extract_claude_code__mutmut_31`
- `polylogue.schemas.unified.x__fallback_extract_claude_code__mutmut_32`
- `polylogue.schemas.unified.x__fallback_extract_claude_code__mutmut_33`
- `polylogue.schemas.unified.x__fallback_extract_claude_code__mutmut_34`
- ... 479 more

## Timeout Keys

- `polylogue.sources.providers.chatgpt.xǁChatGPTConversationǁiter_user_assistant_pairs__mutmut_18`
- `polylogue.sources.providers.chatgpt.xǁChatGPTConversationǁiter_user_assistant_pairs__mutmut_21`

## Notes

- Directly relevant to the next provider-law wave.

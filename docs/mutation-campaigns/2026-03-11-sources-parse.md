# Mutmut Campaign: `sources-parse`

- Recorded on `2026-03-11T10:02:18.125197+00:00`
- Commit: `2bdb267e93b79f1f0dc863f86b5ed859e4e0dbdd`
- Worktree dirty: `no`
- Description: Provider detection, parsing, harmonization, and parser laws
- Workspace: `/tmp/nix-shell.rzHew1/mutmut-sources-parse-hq_n8ojl/repo`
- Command: `mutmut run`

## Scope

- Mutated paths: `polylogue/sources`, `polylogue/schemas/unified.py`
- Selected tests: `tests/unit/sources/test_parse_laws.py`, `tests/unit/sources/test_parsers_props.py`, `tests/unit/sources/test_harmonization_contracts.py`, `tests/unit/sources/test_provider_viewport_laws.py`, `tests/unit/sources/test_source_laws.py`, `tests/unit/sources/test_providers.py`, `tests/unit/sources/test_parsers.py`, `tests/unit/sources/test_parsers_base.py`, `tests/unit/sources/test_parser_misc.py`, `tests/unit/sources/test_parser_edge.py`, `tests/unit/sources/test_parsers_chatgpt.py`, `tests/unit/sources/test_null_guard_properties.py`, `tests/unit/sources/test_models.py`

## Counts

| Status | Count |
| --- | ---: |
| Killed | 1651 |
| Survived | 2390 |
| Timeout | 7 |
| Not checked | 1706 |
| Suspicious | 0 |
| Skipped | 0 |

- Runtime: `229.47s`
- Exit code: `0`

## Dominant Survivors

| Function | Count |
| --- | ---: |
| `extract_content_blocks` | 139 |
| `to_meta` | 71 |
| `filter_entries` | 35 |
| `to_content_blocks` | 34 |
| `_emit_individual` | 25 |
| `extract_reasoning_traces` | 24 |
| `emit` | 8 |
| `_maybe_enrich` | 7 |
| `_emit_grouped` | 6 |
| `extract_tool_calls` | 4 |

## Dominant Timeouts

| Function | Count |
| --- | ---: |
| `iter_user_assistant_pairs` | 3 |
| `to_content_blocks` | 1 |
| `polylogue.sources.source.x_parse_payload__mutmut_42` | 1 |
| `polylogue.sources.source.x_parse_payload__mutmut_44` | 1 |
| `polylogue.sources.source.x_parse_payload__mutmut_46` | 1 |

## Dominant Not-Checked Clusters

| Function | Count |
| --- | ---: |
| `_load_credentials` | 250 |
| `iter_json_files` | 95 |
| `resolve_folder_id` | 76 |
| `download_to_path` | 74 |
| `_service_handle` | 53 |
| `get_metadata` | 48 |
| `download_json_payload` | 47 |
| `save` | 36 |
| `_call_with_retry` | 32 |
| `download_bytes` | 27 |

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
- `polylogue.schemas.unified.x_extract_tool_calls__mutmut_44`
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
- ... 2365 more

## Timeout Keys

- `polylogue.sources.providers.chatgpt.xǁChatGPTMessageǁto_content_blocks__mutmut_1`
- `polylogue.sources.providers.chatgpt.xǁChatGPTConversationǁiter_user_assistant_pairs__mutmut_18`
- `polylogue.sources.providers.chatgpt.xǁChatGPTConversationǁiter_user_assistant_pairs__mutmut_19`
- `polylogue.sources.providers.chatgpt.xǁChatGPTConversationǁiter_user_assistant_pairs__mutmut_21`
- `polylogue.sources.source.x_parse_payload__mutmut_42`
- `polylogue.sources.source.x_parse_payload__mutmut_44`
- `polylogue.sources.source.x_parse_payload__mutmut_46`

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
- ... 1681 more

## Notes

- Broadest campaign here; best run after law-wave work lands.

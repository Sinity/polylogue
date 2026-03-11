# Mutmut Campaign: `sources-parse`

- Recorded on `2026-03-11T18:11:49.054341+00:00`
- Commit: `a27de694650ddb2c16aa40338cd2bf5bb0ab9719`
- Worktree dirty: `no`
- Description: Provider detection, parsing, harmonization, and parser laws
- Workspace: `/tmp/nix-shell.ZRAoE6/mutmut-sources-parse-fhdm7nyq/repo`
- Command: `mutmut run`

## Scope

- Mutated paths: `polylogue/sources`, `polylogue/schemas/unified.py`
- Selected tests: `tests/unit/sources/test_parse_laws.py`, `tests/unit/sources/test_parsers_props.py`, `tests/unit/sources/test_harmonization_contracts.py`, `tests/unit/sources/test_provider_viewport_laws.py`, `tests/unit/sources/test_source_laws.py`, `tests/unit/sources/test_providers.py`, `tests/unit/sources/test_viewport_protocol.py`, `tests/unit/sources/test_extraction.py`, `tests/unit/sources/test_unified_semantic_laws.py`, `tests/unit/sources/test_parsers.py`, `tests/unit/sources/test_parsers_base.py`, `tests/unit/sources/test_parser_misc.py`, `tests/unit/sources/test_parser_edge.py`, `tests/unit/sources/test_parsers_chatgpt.py`, `tests/unit/sources/test_parsers_drive.py`, `tests/unit/sources/test_drive_client_laws.py`, `tests/unit/sources/test_drive_ops.py`, `tests/unit/sources/test_drive_resilience.py`, `tests/unit/sources/test_drive_utils.py`, `tests/unit/sources/test_source_ops.py`, `tests/unit/sources/test_raw_capture.py`, `tests/unit/sources/test_null_guard_properties.py`, `tests/unit/sources/test_models.py`, `tests/unit/sources/test_token_store.py`

## Counts

| Status | Count |
| --- | ---: |
| Killed | 3644 |
| Survived | 2608 |
| Timeout | 12 |
| Not checked | 0 |
| Suspicious | 0 |
| Skipped | 0 |

- Runtime: `280.24s`
- Exit code: `0`

## Dominant Survivors

| Function | Count |
| --- | ---: |
| `extract_content_blocks` | 124 |
| `_load_credentials` | 101 |
| `iter_json_files` | 43 |
| `to_meta` | 41 |
| `filter_entries` | 35 |
| `resolve_folder_id` | 30 |
| `to_content_blocks` | 19 |
| `save` | 19 |
| `load` | 18 |
| `extract_reasoning_traces` | 17 |

## Dominant Timeouts

| Function | Count |
| --- | ---: |
| `iter_user_assistant_pairs` | 3 |
| `_download_request` | 2 |
| `polylogue.sources.drive.x_iter_drive_conversations__mutmut_4` | 1 |
| `polylogue.sources.drive.x_iter_drive_raw_data__mutmut_3` | 1 |
| `_call_with_retry` | 1 |
| `iter_json_files` | 1 |
| `polylogue.sources.source.x_parse_payload__mutmut_41` | 1 |
| `polylogue.sources.source.x_parse_payload__mutmut_44` | 1 |
| `polylogue.sources.source.x_parse_payload__mutmut_46` | 1 |

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
- ... 2583 more

## Timeout Keys

- `polylogue.sources.drive.x_iter_drive_conversations__mutmut_4`
- `polylogue.sources.drive.x_iter_drive_raw_data__mutmut_3`
- `polylogue.sources.drive_client.xǁDriveClientǁ_call_with_retry__mutmut_6`
- `polylogue.sources.drive_client.xǁDriveClientǁiter_json_files__mutmut_84`
- `polylogue.sources.drive_client.xǁDriveClientǁ_download_request__mutmut_14`
- `polylogue.sources.drive_client.xǁDriveClientǁ_download_request__mutmut_15`
- `polylogue.sources.providers.chatgpt.xǁChatGPTConversationǁiter_user_assistant_pairs__mutmut_18`
- `polylogue.sources.providers.chatgpt.xǁChatGPTConversationǁiter_user_assistant_pairs__mutmut_19`
- `polylogue.sources.providers.chatgpt.xǁChatGPTConversationǁiter_user_assistant_pairs__mutmut_21`
- `polylogue.sources.source.x_parse_payload__mutmut_41`
- `polylogue.sources.source.x_parse_payload__mutmut_44`
- `polylogue.sources.source.x_parse_payload__mutmut_46`

## Notes

- Broadest campaign here; best run after law-wave work lands.

# Mutmut Campaign: `sources-parse`

- Recorded on `2026-03-11T19:22:10.711397+00:00`
- Commit: `e759af23458dfdc67e1a820513f09f3828460458`
- Worktree dirty: `no`
- Description: Provider detection, parsing, harmonization, and parser laws
- Workspace: `/tmp/nix-shell.cEiqVv/mutmut-sources-parse-_9gaegv4/repo`
- Command: `mutmut run`

## Scope

- Mutated paths: `polylogue/sources`, `polylogue/schemas/unified.py`
- Selected tests: `tests/unit/sources/test_parse_laws.py`, `tests/unit/sources/test_parsers_props.py`, `tests/unit/sources/test_harmonization_contracts.py`, `tests/unit/sources/test_provider_viewport_laws.py`, `tests/unit/sources/test_source_laws.py`, `tests/unit/sources/test_providers.py`, `tests/unit/sources/test_viewport_protocol.py`, `tests/unit/sources/test_extraction.py`, `tests/unit/sources/test_unified_semantic_laws.py`, `tests/unit/sources/test_parsers.py`, `tests/unit/sources/test_parsers_base.py`, `tests/unit/sources/test_parser_misc.py`, `tests/unit/sources/test_parser_edge.py`, `tests/unit/sources/test_parsers_chatgpt.py`, `tests/unit/sources/test_parsers_drive.py`, `tests/unit/sources/test_drive_client_laws.py`, `tests/unit/sources/test_drive_ops.py`, `tests/unit/sources/test_drive_resilience.py`, `tests/unit/sources/test_drive_utils.py`, `tests/unit/sources/test_source_ops.py`, `tests/unit/sources/test_raw_capture.py`, `tests/unit/sources/test_null_guard_properties.py`, `tests/unit/sources/test_models.py`, `tests/unit/sources/test_token_store.py`

## Counts

| Status | Count |
| --- | ---: |
| Killed | 3494 |
| Survived | 2467 |
| Timeout | 11 |
| Not checked | 0 |
| Suspicious | 0 |
| Skipped | 0 |

- Runtime: `241.31s`
- Exit code: `0`

## Dominant Survivors

| Function | Count |
| --- | ---: |
| `_load_credentials` | 101 |
| `extract_content_blocks` | 99 |
| `iter_json_files` | 43 |
| `to_meta` | 39 |
| `filter_entries` | 35 |
| `resolve_folder_id` | 30 |
| `save` | 19 |
| `load` | 18 |
| `get_metadata` | 16 |
| `_emit_individual` | 16 |

## Dominant Timeouts

| Function | Count |
| --- | ---: |
| `_download_request` | 2 |
| `iter_user_assistant_pairs` | 2 |
| `polylogue.sources.drive.x_iter_drive_conversations__mutmut_4` | 1 |
| `polylogue.sources.drive.x_iter_drive_raw_data__mutmut_3` | 1 |
| `_call_with_retry` | 1 |
| `iter_json_files` | 1 |
| `polylogue.sources.source.x_parse_payload__mutmut_42` | 1 |
| `polylogue.sources.source.x_parse_payload__mutmut_43` | 1 |
| `polylogue.sources.source.x_parse_payload__mutmut_45` | 1 |

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
- `polylogue.schemas.unified.x__fallback_extract_claude_code__mutmut_15`
- `polylogue.schemas.unified.x__fallback_extract_claude_code__mutmut_18`
- `polylogue.schemas.unified.x__fallback_extract_claude_code__mutmut_22`
- `polylogue.schemas.unified.x__fallback_extract_claude_code__mutmut_23`
- `polylogue.schemas.unified.x__fallback_extract_claude_code__mutmut_24`
- `polylogue.schemas.unified.x__fallback_extract_claude_code__mutmut_25`
- `polylogue.schemas.unified.x__fallback_extract_claude_code__mutmut_26`
- `polylogue.schemas.unified.x__fallback_extract_claude_code__mutmut_28`
- `polylogue.schemas.unified.x__fallback_extract_claude_code__mutmut_31`
- `polylogue.schemas.unified.x__fallback_extract_claude_code__mutmut_32`
- ... 2442 more

## Timeout Keys

- `polylogue.sources.drive.x_iter_drive_conversations__mutmut_4`
- `polylogue.sources.drive.x_iter_drive_raw_data__mutmut_3`
- `polylogue.sources.drive_client.xǁDriveClientǁ_call_with_retry__mutmut_6`
- `polylogue.sources.drive_client.xǁDriveClientǁiter_json_files__mutmut_84`
- `polylogue.sources.drive_client.xǁDriveClientǁ_download_request__mutmut_14`
- `polylogue.sources.drive_client.xǁDriveClientǁ_download_request__mutmut_15`
- `polylogue.sources.providers.chatgpt.xǁChatGPTConversationǁiter_user_assistant_pairs__mutmut_18`
- `polylogue.sources.providers.chatgpt.xǁChatGPTConversationǁiter_user_assistant_pairs__mutmut_21`
- `polylogue.sources.source.x_parse_payload__mutmut_42`
- `polylogue.sources.source.x_parse_payload__mutmut_43`
- `polylogue.sources.source.x_parse_payload__mutmut_45`

## Notes

- Broadest campaign here; best run after law-wave work lands.

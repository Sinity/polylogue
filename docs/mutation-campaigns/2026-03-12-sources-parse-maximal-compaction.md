# Mutmut Campaign: `sources-parse`

- Recorded on `2026-03-12T14:09:59.893525+00:00`
- Commit: `c1fd5ce60e8216a714acfc59072597ee40955a66`
- Worktree dirty: `yes`
- Description: Provider detection, parsing, harmonization, and parser laws
- Workspace: `/tmp/nix-shell.fuxVaQ/mutmut-sources-parse-_7dqwdfl/repo`
- Command: `mutmut run`

## Scope

- Mutated paths: `polylogue/sources`, `polylogue/schemas/unified.py`
- Selected tests: `tests/unit/sources/test_parse_laws.py`, `tests/unit/sources/test_parsers_props.py`, `tests/unit/sources/test_content_extraction.py`, `tests/unit/sources/test_source_laws.py`, `tests/unit/sources/test_extraction.py`, `tests/unit/sources/test_unified_semantic_laws.py`, `tests/unit/sources/test_parsers.py`, `tests/unit/sources/test_parsers_base.py`, `tests/unit/sources/test_edge_cases.py`, `tests/unit/sources/test_parser_edge.py`, `tests/unit/sources/test_parsers_chatgpt.py`, `tests/unit/sources/test_parsers_drive.py`, `tests/unit/sources/test_drive_client_laws.py`, `tests/unit/sources/test_drive_ops.py`, `tests/unit/sources/test_drive_utils.py`, `tests/unit/sources/test_null_guard_properties.py`, `tests/unit/sources/test_models.py`, `tests/unit/sources/test_token_store.py`

## Counts

| Status | Count |
| --- | ---: |
| Killed | 3886 |
| Survived | 2019 |
| Timeout | 7 |
| Not checked | 0 |
| Suspicious | 0 |
| Skipped | 0 |

- Runtime: `180.61s`
- Exit code: `0`

## Dominant Survivors

| Function | Count |
| --- | ---: |
| `extract_content_blocks` | 104 |
| `_load_credentials` | 75 |
| `to_meta` | 41 |
| `iter_json_files` | 37 |
| `resolve_folder_id` | 33 |
| `filter_entries` | 32 |
| `get_metadata` | 20 |
| `save` | 19 |
| `load` | 18 |
| `extract_reasoning_traces` | 16 |

## Dominant Timeouts

| Function | Count |
| --- | ---: |
| `iter_user_assistant_pairs` | 3 |
| `_download_request` | 2 |
| `polylogue.sources.drive.x_iter_drive_raw_data__mutmut_3` | 1 |
| `_call_with_retry` | 1 |

## Dominant Not-Checked Clusters

| Function | Count |
| --- | ---: |
| _none_ | 0 |

## Survivor Keys

- `polylogue.schemas.unified.x__missing_role__mutmut_2`
- `polylogue.schemas.unified.x__harmonize_viewport_message__mutmut_13`
- `polylogue.schemas.unified.x__harmonize_viewport_message__mutmut_27`
- `polylogue.schemas.unified.x__validate_claude_code_record__mutmut_5`
- `polylogue.schemas.unified.x__validate_claude_code_record__mutmut_6`
- `polylogue.schemas.unified.x__validate_claude_code_record__mutmut_7`
- `polylogue.schemas.unified.x__validate_claude_code_record__mutmut_8`
- `polylogue.schemas.unified.x__validate_claude_code_record__mutmut_9`
- `polylogue.schemas.unified.x__extract_with_adapter__mutmut_2`
- `polylogue.schemas.unified.x__extract_with_adapter__mutmut_3`
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
- ... 1994 more

## Timeout Keys

- `polylogue.sources.drive.x_iter_drive_raw_data__mutmut_3`
- `polylogue.sources.drive_client.xǁDriveClientǁ_call_with_retry__mutmut_6`
- `polylogue.sources.drive_client.xǁDriveClientǁ_download_request__mutmut_14`
- `polylogue.sources.drive_client.xǁDriveClientǁ_download_request__mutmut_15`
- `polylogue.sources.providers.chatgpt.xǁChatGPTConversationǁiter_user_assistant_pairs__mutmut_18`
- `polylogue.sources.providers.chatgpt.xǁChatGPTConversationǁiter_user_assistant_pairs__mutmut_19`
- `polylogue.sources.providers.chatgpt.xǁChatGPTConversationǁiter_user_assistant_pairs__mutmut_21`

## Source Worktree Status

- ` M tests/unit/cli/test_query_exec_laws.py`
- ` M tests/unit/core/test_health_core.py`
- ` M tests/unit/sources/test_drive_utils.py`
- ` M tests/unit/sources/test_source_laws.py`
- ` M tests/unit/sources/test_unified_semantic_laws.py`
- ` M tests/unit/ui/test_rendering.py`

## Notes

- Broadest campaign here; best run after law-wave work lands.

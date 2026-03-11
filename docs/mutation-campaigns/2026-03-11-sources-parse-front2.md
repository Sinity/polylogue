# Mutmut Campaign: `sources-parse`

- Recorded on `2026-03-11T22:24:46.072807+00:00`
- Commit: `47a9b1cff33f61d745bae5cea90e20e1f75749d9`
- Worktree dirty: `no`
- Description: Provider detection, parsing, harmonization, and parser laws
- Workspace: `/tmp/nix-shell.ANfJ44/mutmut-sources-parse-5b5lholf/repo`
- Command: `mutmut run`

## Scope

- Mutated paths: `polylogue/sources`, `polylogue/schemas/unified.py`
- Selected tests: `tests/unit/sources/test_parse_laws.py`, `tests/unit/sources/test_parsers_props.py`, `tests/unit/sources/test_harmonization_contracts.py`, `tests/unit/sources/test_source_laws.py`, `tests/unit/sources/test_providers.py`, `tests/unit/sources/test_viewport_protocol.py`, `tests/unit/sources/test_extraction.py`, `tests/unit/sources/test_unified_semantic_laws.py`, `tests/unit/sources/test_parsers.py`, `tests/unit/sources/test_parsers_base.py`, `tests/unit/sources/test_parser_misc.py`, `tests/unit/sources/test_parser_edge.py`, `tests/unit/sources/test_parsers_chatgpt.py`, `tests/unit/sources/test_parsers_drive.py`, `tests/unit/sources/test_drive_client_laws.py`, `tests/unit/sources/test_drive_ops.py`, `tests/unit/sources/test_drive_utils.py`, `tests/unit/sources/test_source_ops.py`, `tests/unit/sources/test_raw_capture.py`, `tests/unit/sources/test_null_guard_properties.py`, `tests/unit/sources/test_models.py`, `tests/unit/sources/test_token_store.py`

## Counts

| Status | Count |
| --- | ---: |
| Killed | 3597 |
| Survived | 2319 |
| Timeout | 31 |
| Not checked | 0 |
| Suspicious | 0 |
| Skipped | 0 |

- Runtime: `265.78s`
- Exit code: `0`

## Dominant Survivors

| Function | Count |
| --- | ---: |
| `extract_content_blocks` | 100 |
| `_load_credentials` | 75 |
| `to_meta` | 39 |
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
| `_download_request` | 2 |
| `iter_user_assistant_pairs` | 2 |
| `polylogue.sources.drive.x_iter_drive_raw_data__mutmut_3` | 1 |
| `_call_with_retry` | 1 |
| `polylogue.sources.parsers.base.x_content_blocks_from_segments__mutmut_2` | 1 |
| `polylogue.sources.parsers.base.x_content_blocks_from_segments__mutmut_5` | 1 |
| `polylogue.sources.parsers.base.x_content_blocks_from_segments__mutmut_6` | 1 |
| `polylogue.sources.parsers.base.x_content_blocks_from_segments__mutmut_9` | 1 |
| `polylogue.sources.parsers.base.x_content_blocks_from_segments__mutmut_10` | 1 |
| `polylogue.sources.parsers.base.x_content_blocks_from_segments__mutmut_11` | 1 |

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
- `polylogue.schemas.unified.x__extract_with_adapter__mutmut_11`
- `polylogue.schemas.unified.x__extract_with_adapter__mutmut_16`
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
- ... 2294 more

## Timeout Keys

- `polylogue.sources.drive.x_iter_drive_raw_data__mutmut_3`
- `polylogue.sources.drive_client.xǁDriveClientǁ_call_with_retry__mutmut_6`
- `polylogue.sources.drive_client.xǁDriveClientǁ_download_request__mutmut_14`
- `polylogue.sources.drive_client.xǁDriveClientǁ_download_request__mutmut_15`
- `polylogue.sources.parsers.base.x_content_blocks_from_segments__mutmut_2`
- `polylogue.sources.parsers.base.x_content_blocks_from_segments__mutmut_5`
- `polylogue.sources.parsers.base.x_content_blocks_from_segments__mutmut_6`
- `polylogue.sources.parsers.base.x_content_blocks_from_segments__mutmut_9`
- `polylogue.sources.parsers.base.x_content_blocks_from_segments__mutmut_10`
- `polylogue.sources.parsers.base.x_content_blocks_from_segments__mutmut_11`
- `polylogue.sources.parsers.base.x_content_blocks_from_segments__mutmut_12`
- `polylogue.sources.parsers.base.x_content_blocks_from_segments__mutmut_13`
- `polylogue.sources.parsers.base.x_content_blocks_from_segments__mutmut_14`
- `polylogue.sources.parsers.base.x_content_blocks_from_segments__mutmut_15`
- `polylogue.sources.parsers.base.x_content_blocks_from_segments__mutmut_16`
- `polylogue.sources.parsers.base.x_content_blocks_from_segments__mutmut_18`
- `polylogue.sources.parsers.base.x_content_blocks_from_segments__mutmut_21`
- `polylogue.sources.parsers.base.x_content_blocks_from_segments__mutmut_23`
- `polylogue.sources.parsers.base.x_content_blocks_from_segments__mutmut_26`
- `polylogue.sources.parsers.base.x_content_blocks_from_segments__mutmut_27`
- `polylogue.sources.parsers.base.x_content_blocks_from_segments__mutmut_32`
- `polylogue.sources.parsers.base.x_content_blocks_from_segments__mutmut_37`
- `polylogue.sources.parsers.base.x_content_blocks_from_segments__mutmut_38`
- `polylogue.sources.parsers.base.x_content_blocks_from_segments__mutmut_39`
- `polylogue.sources.parsers.base.x_content_blocks_from_segments__mutmut_40`
- ... 6 more

## Notes

- Broadest campaign here; best run after law-wave work lands.

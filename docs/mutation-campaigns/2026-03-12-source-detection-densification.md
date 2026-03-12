# Mutmut Campaign: `source-detection`

- Recorded on `2026-03-12T10:00:29.229712+00:00`
- Commit: `eb43cfd48e989a58780604e9cbf9d3ba93700ff8`
- Worktree dirty: `yes`
- Description: Source detection, sniffing, and parser dispatch
- Workspace: `/tmp/nix-shell.ccW0oW/mutmut-source-detection-i0xvik39/repo`
- Command: `mutmut run`

## Scope

- Mutated paths: `polylogue/sources/source.py`
- Selected tests: `tests/unit/sources/test_source_laws.py`, `tests/unit/sources/test_parsers.py`, `tests/unit/sources/test_parsers_base.py`, `tests/unit/sources/test_edge_cases.py`, `tests/unit/sources/test_parser_edge.py`, `tests/unit/sources/test_parsers_chatgpt.py`, `tests/unit/sources/test_parsers_props.py`, `tests/unit/sources/test_parsers_drive.py`

## Counts

| Status | Count |
| --- | ---: |
| Killed | 819 |
| Survived | 331 |
| Timeout | 1 |
| Not checked | 0 |
| Suspicious | 0 |
| Skipped | 0 |

- Runtime: `57.49s`
- Exit code: `0`

## Dominant Survivors

| Function | Count |
| --- | ---: |
| `filter_entries` | 32 |
| `_emit_individual` | 13 |
| `emit` | 6 |
| `_emit_grouped` | 4 |
| `polylogue.sources.source.x_save_bundle__mutmut_5` | 1 |
| `polylogue.sources.source.x_save_bundle__mutmut_9` | 1 |
| `polylogue.sources.source.x__decode_json_bytes__mutmut_3` | 1 |
| `polylogue.sources.source.x__decode_json_bytes__mutmut_13` | 1 |
| `polylogue.sources.source.x__decode_json_bytes__mutmut_14` | 1 |
| `polylogue.sources.source.x__decode_json_bytes__mutmut_22` | 1 |

## Dominant Timeouts

| Function | Count |
| --- | ---: |
| `polylogue.sources.source.x_parse_payload__mutmut_43` | 1 |

## Dominant Not-Checked Clusters

| Function | Count |
| --- | ---: |
| _none_ | 0 |

## Survivor Keys

- `polylogue.sources.source.x_save_bundle__mutmut_5`
- `polylogue.sources.source.x_save_bundle__mutmut_9`
- `polylogue.sources.source.x__decode_json_bytes__mutmut_3`
- `polylogue.sources.source.x__decode_json_bytes__mutmut_13`
- `polylogue.sources.source.x__decode_json_bytes__mutmut_14`
- `polylogue.sources.source.x__decode_json_bytes__mutmut_22`
- `polylogue.sources.source.x__decode_json_bytes__mutmut_25`
- `polylogue.sources.source.x__decode_json_bytes__mutmut_28`
- `polylogue.sources.source.x__decode_json_bytes__mutmut_29`
- `polylogue.sources.source.x__decode_json_bytes__mutmut_30`
- `polylogue.sources.source.x__decode_json_bytes__mutmut_31`
- `polylogue.sources.source.x__decode_json_bytes__mutmut_32`
- `polylogue.sources.source.x__decode_json_bytes__mutmut_33`
- `polylogue.sources.source.x_detect_provider__mutmut_10`
- `polylogue.sources.source.x_detect_provider__mutmut_11`
- `polylogue.sources.source.x_detect_provider__mutmut_12`
- `polylogue.sources.source.x_detect_provider__mutmut_13`
- `polylogue.sources.source.x_detect_provider__mutmut_14`
- `polylogue.sources.source.x_detect_provider__mutmut_18`
- `polylogue.sources.source.x_detect_provider__mutmut_19`
- `polylogue.sources.source.x_detect_provider__mutmut_20`
- `polylogue.sources.source.x_detect_provider__mutmut_22`
- `polylogue.sources.source.x_detect_provider__mutmut_34`
- `polylogue.sources.source.x_detect_provider__mutmut_35`
- `polylogue.sources.source.x_detect_provider__mutmut_42`
- ... 306 more

## Timeout Keys

- `polylogue.sources.source.x_parse_payload__mutmut_43`

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

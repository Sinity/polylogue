# Mutmut Campaign: `source-detection`

- Recorded on `2026-03-12T14:09:59.882621+00:00`
- Commit: `c1fd5ce60e8216a714acfc59072597ee40955a66`
- Worktree dirty: `yes`
- Description: Source detection, sniffing, and parser dispatch
- Workspace: `/tmp/nix-shell.4yITYv/mutmut-source-detection-s2loyf8s/repo`
- Command: `mutmut run`

## Scope

- Mutated paths: `polylogue/sources/source.py`
- Selected tests: `tests/unit/sources/test_source_laws.py`, `tests/unit/sources/test_parsers.py`, `tests/unit/sources/test_parsers_base.py`, `tests/unit/sources/test_edge_cases.py`, `tests/unit/sources/test_parser_edge.py`, `tests/unit/sources/test_parsers_chatgpt.py`, `tests/unit/sources/test_parsers_props.py`, `tests/unit/sources/test_parsers_drive.py`

## Counts

| Status | Count |
| --- | ---: |
| Killed | 934 |
| Survived | 217 |
| Timeout | 0 |
| Not checked | 0 |
| Suspicious | 0 |
| Skipped | 0 |

- Runtime: `53.96s`
- Exit code: `0`

## Dominant Survivors

| Function | Count |
| --- | ---: |
| `filter_entries` | 32 |
| `_emit_individual` | 13 |
| `polylogue.sources.source.x_save_bundle__mutmut_5` | 1 |
| `polylogue.sources.source.x_save_bundle__mutmut_9` | 1 |
| `polylogue.sources.source.x__looks_like_chunked_conversation__mutmut_1` | 1 |
| `polylogue.sources.source.x__looks_like_chunked_conversation_list__mutmut_1` | 1 |
| `polylogue.sources.source.x_parse_drive_payload__mutmut_1` | 1 |
| `polylogue.sources.source.x_parse_drive_payload__mutmut_2` | 1 |
| `polylogue.sources.source.x_parse_drive_payload__mutmut_3` | 1 |
| `polylogue.sources.source.x_parse_drive_payload__mutmut_4` | 1 |

## Dominant Timeouts

| Function | Count |
| --- | ---: |
| _none_ | 0 |

## Dominant Not-Checked Clusters

| Function | Count |
| --- | ---: |
| _none_ | 0 |

## Survivor Keys

- `polylogue.sources.source.x_save_bundle__mutmut_5`
- `polylogue.sources.source.x_save_bundle__mutmut_9`
- `polylogue.sources.source.x__looks_like_chunked_conversation__mutmut_1`
- `polylogue.sources.source.x__looks_like_chunked_conversation_list__mutmut_1`
- `polylogue.sources.source.x_parse_drive_payload__mutmut_1`
- `polylogue.sources.source.x_parse_drive_payload__mutmut_2`
- `polylogue.sources.source.x_parse_drive_payload__mutmut_3`
- `polylogue.sources.source.x_parse_drive_payload__mutmut_4`
- `polylogue.sources.source.x_parse_drive_payload__mutmut_5`
- `polylogue.sources.source.x_parse_drive_payload__mutmut_6`
- `polylogue.sources.source.x_parse_drive_payload__mutmut_7`
- `polylogue.sources.source.x_parse_drive_payload__mutmut_8`
- `polylogue.sources.source.x_parse_drive_payload__mutmut_9`
- `polylogue.sources.source.x_parse_drive_payload__mutmut_12`
- `polylogue.sources.source.x_parse_drive_payload__mutmut_13`
- `polylogue.sources.source.x_parse_drive_payload__mutmut_14`
- `polylogue.sources.source.x_parse_drive_payload__mutmut_17`
- `polylogue.sources.source.x_parse_drive_payload__mutmut_18`
- `polylogue.sources.source.x_parse_drive_payload__mutmut_32`
- `polylogue.sources.source.x_parse_drive_payload__mutmut_39`
- `polylogue.sources.source.x_parse_drive_payload__mutmut_40`
- `polylogue.sources.source.x_parse_drive_payload__mutmut_41`
- `polylogue.sources.source.x_parse_drive_payload__mutmut_42`
- `polylogue.sources.source.x_parse_drive_payload__mutmut_43`
- `polylogue.sources.source.x_parse_drive_payload__mutmut_44`
- ... 192 more

## Source Worktree Status

- ` M tests/unit/cli/test_query_exec_laws.py`
- ` M tests/unit/core/test_health_core.py`
- ` M tests/unit/sources/test_drive_utils.py`
- ` M tests/unit/sources/test_source_laws.py`
- ` M tests/unit/sources/test_unified_semantic_laws.py`
- ` M tests/unit/ui/test_rendering.py`

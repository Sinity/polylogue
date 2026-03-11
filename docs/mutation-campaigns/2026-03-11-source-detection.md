# Mutmut Campaign: `source-detection`

- Recorded on `2026-03-11T10:00:02.085699+00:00`
- Commit: `2bdb267e93b79f1f0dc863f86b5ed859e4e0dbdd`
- Worktree dirty: `no`
- Description: Source detection, sniffing, and parser dispatch
- Workspace: `/tmp/nix-shell.rzHew1/mutmut-source-detection-c2ywoac5/repo`
- Command: `mutmut run`

## Scope

- Mutated paths: `polylogue/sources/source.py`
- Selected tests: `tests/unit/sources/test_source_laws.py`, `tests/unit/sources/test_providers.py`, `tests/unit/sources/test_parsers.py`, `tests/unit/sources/test_parsers_base.py`, `tests/unit/sources/test_parser_misc.py`, `tests/unit/sources/test_parser_edge.py`, `tests/unit/sources/test_parsers_chatgpt.py`, `tests/unit/sources/test_parsers_props.py`

## Counts

| Status | Count |
| --- | ---: |
| Killed | 563 |
| Survived | 455 |
| Timeout | 3 |
| Not checked | 127 |
| Suspicious | 0 |
| Skipped | 0 |

- Runtime: `85.65s`
- Exit code: `0`

## Dominant Survivors

| Function | Count |
| --- | ---: |
| `filter_entries` | 35 |
| `_emit_individual` | 24 |
| `emit` | 8 |
| `_maybe_enrich` | 7 |
| `_emit_grouped` | 4 |
| `_make_raw` | 2 |
| `polylogue.sources.source.x_save_bundle__mutmut_5` | 1 |
| `polylogue.sources.source.x_save_bundle__mutmut_9` | 1 |
| `polylogue.sources.source.x__decode_json_bytes__mutmut_3` | 1 |
| `polylogue.sources.source.x__decode_json_bytes__mutmut_13` | 1 |

## Dominant Timeouts

| Function | Count |
| --- | ---: |
| `polylogue.sources.source.x_parse_payload__mutmut_41` | 1 |
| `polylogue.sources.source.x_parse_payload__mutmut_42` | 1 |
| `polylogue.sources.source.x_iter_source_conversations_with_raw__mutmut_6` | 1 |

## Dominant Not-Checked Clusters

| Function | Count |
| --- | ---: |
| `polylogue.sources.source.x_iter_source_raw_data__mutmut_1` | 1 |
| `polylogue.sources.source.x_iter_source_raw_data__mutmut_2` | 1 |
| `polylogue.sources.source.x_iter_source_raw_data__mutmut_3` | 1 |
| `polylogue.sources.source.x_iter_source_raw_data__mutmut_4` | 1 |
| `polylogue.sources.source.x_iter_source_raw_data__mutmut_5` | 1 |
| `polylogue.sources.source.x_iter_source_raw_data__mutmut_6` | 1 |
| `polylogue.sources.source.x_iter_source_raw_data__mutmut_7` | 1 |
| `polylogue.sources.source.x_iter_source_raw_data__mutmut_8` | 1 |
| `polylogue.sources.source.x_iter_source_raw_data__mutmut_9` | 1 |
| `polylogue.sources.source.x_iter_source_raw_data__mutmut_10` | 1 |

## Survivor Keys

- `polylogue.sources.source.x_save_bundle__mutmut_5`
- `polylogue.sources.source.x_save_bundle__mutmut_9`
- `polylogue.sources.source.x__decode_json_bytes__mutmut_3`
- `polylogue.sources.source.x__decode_json_bytes__mutmut_13`
- `polylogue.sources.source.x__decode_json_bytes__mutmut_14`
- `polylogue.sources.source.x__decode_json_bytes__mutmut_15`
- `polylogue.sources.source.x__decode_json_bytes__mutmut_22`
- `polylogue.sources.source.x__decode_json_bytes__mutmut_23`
- `polylogue.sources.source.x__decode_json_bytes__mutmut_24`
- `polylogue.sources.source.x__decode_json_bytes__mutmut_25`
- `polylogue.sources.source.x__decode_json_bytes__mutmut_26`
- `polylogue.sources.source.x__decode_json_bytes__mutmut_27`
- `polylogue.sources.source.x__decode_json_bytes__mutmut_28`
- `polylogue.sources.source.x__decode_json_bytes__mutmut_29`
- `polylogue.sources.source.x__decode_json_bytes__mutmut_30`
- `polylogue.sources.source.x__decode_json_bytes__mutmut_31`
- `polylogue.sources.source.x__decode_json_bytes__mutmut_32`
- `polylogue.sources.source.x__decode_json_bytes__mutmut_33`
- `polylogue.sources.source.x_detect_provider__mutmut_3`
- `polylogue.sources.source.x_detect_provider__mutmut_4`
- `polylogue.sources.source.x_detect_provider__mutmut_10`
- `polylogue.sources.source.x_detect_provider__mutmut_11`
- `polylogue.sources.source.x_detect_provider__mutmut_12`
- `polylogue.sources.source.x_detect_provider__mutmut_13`
- `polylogue.sources.source.x_detect_provider__mutmut_14`
- ... 430 more

## Timeout Keys

- `polylogue.sources.source.x_parse_payload__mutmut_41`
- `polylogue.sources.source.x_parse_payload__mutmut_42`
- `polylogue.sources.source.x_iter_source_conversations_with_raw__mutmut_6`

## Not-Checked Keys

- `polylogue.sources.source.x_iter_source_raw_data__mutmut_1`
- `polylogue.sources.source.x_iter_source_raw_data__mutmut_2`
- `polylogue.sources.source.x_iter_source_raw_data__mutmut_3`
- `polylogue.sources.source.x_iter_source_raw_data__mutmut_4`
- `polylogue.sources.source.x_iter_source_raw_data__mutmut_5`
- `polylogue.sources.source.x_iter_source_raw_data__mutmut_6`
- `polylogue.sources.source.x_iter_source_raw_data__mutmut_7`
- `polylogue.sources.source.x_iter_source_raw_data__mutmut_8`
- `polylogue.sources.source.x_iter_source_raw_data__mutmut_9`
- `polylogue.sources.source.x_iter_source_raw_data__mutmut_10`
- `polylogue.sources.source.x_iter_source_raw_data__mutmut_11`
- `polylogue.sources.source.x_iter_source_raw_data__mutmut_12`
- `polylogue.sources.source.x_iter_source_raw_data__mutmut_13`
- `polylogue.sources.source.x_iter_source_raw_data__mutmut_14`
- `polylogue.sources.source.x_iter_source_raw_data__mutmut_15`
- `polylogue.sources.source.x_iter_source_raw_data__mutmut_16`
- `polylogue.sources.source.x_iter_source_raw_data__mutmut_17`
- `polylogue.sources.source.x_iter_source_raw_data__mutmut_18`
- `polylogue.sources.source.x_iter_source_raw_data__mutmut_19`
- `polylogue.sources.source.x_iter_source_raw_data__mutmut_20`
- `polylogue.sources.source.x_iter_source_raw_data__mutmut_21`
- `polylogue.sources.source.x_iter_source_raw_data__mutmut_22`
- `polylogue.sources.source.x_iter_source_raw_data__mutmut_23`
- `polylogue.sources.source.x_iter_source_raw_data__mutmut_24`
- `polylogue.sources.source.x_iter_source_raw_data__mutmut_25`
- ... 102 more

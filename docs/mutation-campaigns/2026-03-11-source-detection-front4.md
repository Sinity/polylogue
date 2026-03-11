# Mutmut Campaign: `source-detection`

- Recorded on `2026-03-11T22:03:06.297653+00:00`
- Commit: `38d65c004a2a7cb37d9eca7a098786d01d790bb2`
- Worktree dirty: `no`
- Description: Source detection, sniffing, and parser dispatch
- Workspace: `/tmp/nix-shell.j5I5bL/mutmut-source-detection-6nh5qxq5/repo`
- Command: `mutmut run`

## Scope

- Mutated paths: `polylogue/sources/source.py`
- Selected tests: `tests/unit/sources/test_source_laws.py`, `tests/unit/sources/test_providers.py`, `tests/unit/sources/test_parsers.py`, `tests/unit/sources/test_parsers_base.py`, `tests/unit/sources/test_parser_misc.py`, `tests/unit/sources/test_parser_edge.py`, `tests/unit/sources/test_parsers_chatgpt.py`, `tests/unit/sources/test_parsers_props.py`, `tests/unit/sources/test_source_ops.py`, `tests/unit/sources/test_raw_capture.py`

## Counts

| Status | Count |
| --- | ---: |
| Killed | 816 |
| Survived | 330 |
| Timeout | 5 |
| Not checked | 0 |
| Suspicious | 0 |
| Skipped | 0 |

- Runtime: `56.15s`
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
| `polylogue.sources.source.x_parse_payload__mutmut_40` | 1 |
| `polylogue.sources.source.x_parse_payload__mutmut_43` | 1 |
| `polylogue.sources.source.x_parse_payload__mutmut_45` | 1 |
| `_make_raw` | 1 |
| `polylogue.sources.source.x_iter_source_conversations_with_raw__mutmut_6` | 1 |

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
- ... 305 more

## Timeout Keys

- `polylogue.sources.source.x_parse_payload__mutmut_40`
- `polylogue.sources.source.x_parse_payload__mutmut_43`
- `polylogue.sources.source.x_parse_payload__mutmut_45`
- `polylogue.sources.source.xǁ_ConversationEmitterǁ_make_raw__mutmut_7`
- `polylogue.sources.source.x_iter_source_conversations_with_raw__mutmut_6`

# Mutmut Campaign: `source-detection`

- Recorded on `2026-03-11T20:28:01.139536+00:00`
- Commit: `e058c8240959fc530a1d97cc2d47f15840189cf1`
- Worktree dirty: `no`
- Description: Source detection, sniffing, and parser dispatch
- Workspace: `/tmp/nix-shell.tdwI27/mutmut-source-detection-eup4wkek/repo`
- Command: `mutmut run`

## Scope

- Mutated paths: `polylogue/sources/source.py`
- Selected tests: `tests/unit/sources/test_source_laws.py`, `tests/unit/sources/test_providers.py`, `tests/unit/sources/test_parsers.py`, `tests/unit/sources/test_parsers_base.py`, `tests/unit/sources/test_parser_misc.py`, `tests/unit/sources/test_parser_edge.py`, `tests/unit/sources/test_parsers_chatgpt.py`, `tests/unit/sources/test_parsers_props.py`, `tests/unit/sources/test_source_ops.py`, `tests/unit/sources/test_raw_capture.py`

## Counts

| Status | Count |
| --- | ---: |
| Killed | 702 |
| Survived | 445 |
| Timeout | 4 |
| Not checked | 0 |
| Suspicious | 0 |
| Skipped | 0 |

- Runtime: `121.54s`
- Exit code: `0`

## Dominant Survivors

| Function | Count |
| --- | ---: |
| `filter_entries` | 36 |
| `_emit_individual` | 16 |
| `emit` | 7 |
| `_maybe_enrich` | 7 |
| `_emit_grouped` | 4 |
| `polylogue.sources.source.x_save_bundle__mutmut_5` | 1 |
| `polylogue.sources.source.x_save_bundle__mutmut_9` | 1 |
| `polylogue.sources.source.x__decode_json_bytes__mutmut_3` | 1 |
| `polylogue.sources.source.x__decode_json_bytes__mutmut_13` | 1 |
| `polylogue.sources.source.x__decode_json_bytes__mutmut_14` | 1 |

## Dominant Timeouts

| Function | Count |
| --- | ---: |
| `polylogue.sources.source.x_parse_payload__mutmut_40` | 1 |
| `polylogue.sources.source.x_parse_payload__mutmut_43` | 1 |
| `polylogue.sources.source.x_parse_payload__mutmut_45` | 1 |
| `_emit_grouped` | 1 |

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
- `polylogue.sources.source.x__decode_json_bytes__mutmut_15`
- `polylogue.sources.source.x__decode_json_bytes__mutmut_22`
- `polylogue.sources.source.x__decode_json_bytes__mutmut_23`
- `polylogue.sources.source.x__decode_json_bytes__mutmut_25`
- `polylogue.sources.source.x__decode_json_bytes__mutmut_26`
- `polylogue.sources.source.x__decode_json_bytes__mutmut_27`
- `polylogue.sources.source.x__decode_json_bytes__mutmut_30`
- `polylogue.sources.source.x__decode_json_bytes__mutmut_31`
- `polylogue.sources.source.x__decode_json_bytes__mutmut_32`
- `polylogue.sources.source.x__decode_json_bytes__mutmut_33`
- `polylogue.sources.source.x_detect_provider__mutmut_4`
- `polylogue.sources.source.x_detect_provider__mutmut_10`
- `polylogue.sources.source.x_detect_provider__mutmut_11`
- `polylogue.sources.source.x_detect_provider__mutmut_12`
- `polylogue.sources.source.x_detect_provider__mutmut_13`
- `polylogue.sources.source.x_detect_provider__mutmut_14`
- `polylogue.sources.source.x_detect_provider__mutmut_17`
- `polylogue.sources.source.x_detect_provider__mutmut_18`
- `polylogue.sources.source.x_detect_provider__mutmut_19`
- `polylogue.sources.source.x_detect_provider__mutmut_20`
- ... 420 more

## Timeout Keys

- `polylogue.sources.source.x_parse_payload__mutmut_40`
- `polylogue.sources.source.x_parse_payload__mutmut_43`
- `polylogue.sources.source.x_parse_payload__mutmut_45`
- `polylogue.sources.source.xǁ_ConversationEmitterǁ_emit_grouped__mutmut_7`

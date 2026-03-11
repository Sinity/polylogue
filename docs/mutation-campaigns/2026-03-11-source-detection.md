# Mutmut Campaign: `source-detection`

- Recorded on `2026-03-11T06:35:18.888629+00:00`
- Commit: `147e689d15caf23fc4036c3af6211af4f71bbaad`
- Worktree dirty: `no`
- Description: Source detection, sniffing, and parser dispatch
- Workspace: `/tmp/nix-shell.VSHgLm/mutmut-source-detection-0g1yt3av/repo`
- Command: `mutmut run`

## Scope

- Mutated paths: `polylogue/sources/source.py`
- Selected tests: `tests/unit/sources/test_parsers.py`, `tests/unit/sources/test_parsers_base.py`, `tests/unit/sources/test_parser_misc.py`, `tests/unit/sources/test_parser_edge.py`, `tests/unit/sources/test_parsers_chatgpt.py`, `tests/unit/sources/test_parsers_props.py`

## Counts

| Status | Count |
| --- | ---: |
| Killed | 41 |
| Survived | 197 |
| Timeout | 0 |
| Not checked | 910 |
| Suspicious | 0 |
| Skipped | 0 |

- Runtime: `19.66s`
- Exit code: `0`

## Dominant Survivors

| Function | Count |
| --- | ---: |
| `polylogue.sources.source.x_save_bundle__mutmut_5` | 1 |
| `polylogue.sources.source.x_save_bundle__mutmut_9` | 1 |
| `polylogue.sources.source.x_detect_provider__mutmut_1` | 1 |
| `polylogue.sources.source.x_detect_provider__mutmut_2` | 1 |
| `polylogue.sources.source.x_detect_provider__mutmut_3` | 1 |
| `polylogue.sources.source.x_detect_provider__mutmut_4` | 1 |
| `polylogue.sources.source.x_detect_provider__mutmut_5` | 1 |
| `polylogue.sources.source.x_detect_provider__mutmut_6` | 1 |
| `polylogue.sources.source.x_detect_provider__mutmut_7` | 1 |
| `polylogue.sources.source.x_detect_provider__mutmut_8` | 1 |

## Dominant Timeouts

| Function | Count |
| --- | ---: |
| _none_ | 0 |

## Dominant Not-Checked Clusters

| Function | Count |
| --- | ---: |
| `filter_entries` | 69 |
| `_emit_individual` | 59 |
| `_emit_grouped` | 32 |
| `emit` | 23 |
| `_make_raw` | 14 |
| `_maybe_enrich` | 10 |
| `__init__` | 4 |
| `polylogue.sources.source.x__decode_json_bytes__mutmut_1` | 1 |
| `polylogue.sources.source.x__decode_json_bytes__mutmut_2` | 1 |
| `polylogue.sources.source.x__decode_json_bytes__mutmut_3` | 1 |

## Survivor Keys

- `polylogue.sources.source.x_save_bundle__mutmut_5`
- `polylogue.sources.source.x_save_bundle__mutmut_9`
- `polylogue.sources.source.x_detect_provider__mutmut_1`
- `polylogue.sources.source.x_detect_provider__mutmut_2`
- `polylogue.sources.source.x_detect_provider__mutmut_3`
- `polylogue.sources.source.x_detect_provider__mutmut_4`
- `polylogue.sources.source.x_detect_provider__mutmut_5`
- `polylogue.sources.source.x_detect_provider__mutmut_6`
- `polylogue.sources.source.x_detect_provider__mutmut_7`
- `polylogue.sources.source.x_detect_provider__mutmut_8`
- `polylogue.sources.source.x_detect_provider__mutmut_9`
- `polylogue.sources.source.x_detect_provider__mutmut_10`
- `polylogue.sources.source.x_detect_provider__mutmut_11`
- `polylogue.sources.source.x_detect_provider__mutmut_12`
- `polylogue.sources.source.x_detect_provider__mutmut_13`
- `polylogue.sources.source.x_detect_provider__mutmut_14`
- `polylogue.sources.source.x_detect_provider__mutmut_17`
- `polylogue.sources.source.x_detect_provider__mutmut_18`
- `polylogue.sources.source.x_detect_provider__mutmut_19`
- `polylogue.sources.source.x_detect_provider__mutmut_20`
- `polylogue.sources.source.x_detect_provider__mutmut_22`
- `polylogue.sources.source.x_detect_provider__mutmut_23`
- `polylogue.sources.source.x_detect_provider__mutmut_24`
- `polylogue.sources.source.x_detect_provider__mutmut_25`
- `polylogue.sources.source.x_detect_provider__mutmut_26`
- ... 172 more

## Not-Checked Keys

- `polylogue.sources.source.x__decode_json_bytes__mutmut_1`
- `polylogue.sources.source.x__decode_json_bytes__mutmut_2`
- `polylogue.sources.source.x__decode_json_bytes__mutmut_3`
- `polylogue.sources.source.x__decode_json_bytes__mutmut_4`
- `polylogue.sources.source.x__decode_json_bytes__mutmut_5`
- `polylogue.sources.source.x__decode_json_bytes__mutmut_6`
- `polylogue.sources.source.x__decode_json_bytes__mutmut_7`
- `polylogue.sources.source.x__decode_json_bytes__mutmut_8`
- `polylogue.sources.source.x__decode_json_bytes__mutmut_9`
- `polylogue.sources.source.x__decode_json_bytes__mutmut_10`
- `polylogue.sources.source.x__decode_json_bytes__mutmut_11`
- `polylogue.sources.source.x__decode_json_bytes__mutmut_12`
- `polylogue.sources.source.x__decode_json_bytes__mutmut_13`
- `polylogue.sources.source.x__decode_json_bytes__mutmut_14`
- `polylogue.sources.source.x__decode_json_bytes__mutmut_15`
- `polylogue.sources.source.x__decode_json_bytes__mutmut_16`
- `polylogue.sources.source.x__decode_json_bytes__mutmut_17`
- `polylogue.sources.source.x__decode_json_bytes__mutmut_18`
- `polylogue.sources.source.x__decode_json_bytes__mutmut_19`
- `polylogue.sources.source.x__decode_json_bytes__mutmut_20`
- `polylogue.sources.source.x__decode_json_bytes__mutmut_21`
- `polylogue.sources.source.x__decode_json_bytes__mutmut_22`
- `polylogue.sources.source.x__decode_json_bytes__mutmut_23`
- `polylogue.sources.source.x__decode_json_bytes__mutmut_24`
- `polylogue.sources.source.x__decode_json_bytes__mutmut_25`
- ... 885 more

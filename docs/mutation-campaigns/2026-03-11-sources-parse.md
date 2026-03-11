# Mutmut Campaign: `sources-parse`

- Recorded on `2026-03-11T06:36:14.451521+00:00`
- Commit: `147e689d15caf23fc4036c3af6211af4f71bbaad`
- Worktree dirty: `no`
- Description: Provider detection, parsing, harmonization, and parser laws
- Workspace: `/tmp/nix-shell.IEIqa8/mutmut-sources-parse-94_p8ptc/repo`
- Command: `mutmut run`

## Scope

- Mutated paths: `polylogue/sources`, `polylogue/schemas/unified.py`
- Selected tests: `tests/unit/sources/test_parse_laws.py`, `tests/unit/sources/test_parsers_props.py`, `tests/unit/sources/test_harmonization_contracts.py`, `tests/unit/sources/test_providers.py`, `tests/unit/sources/test_parsers.py`, `tests/unit/sources/test_parsers_base.py`, `tests/unit/sources/test_parser_misc.py`, `tests/unit/sources/test_parser_edge.py`, `tests/unit/sources/test_parsers_chatgpt.py`, `tests/unit/sources/test_null_guard_properties.py`, `tests/unit/sources/test_models.py`

## Counts

| Status | Count |
| --- | ---: |
| Killed | 1353 |
| Survived | 2307 |
| Timeout | 0 |
| Not checked | 2094 |
| Suspicious | 0 |
| Skipped | 0 |

- Runtime: `133.77s`
- Exit code: `0`

## Dominant Survivors

| Function | Count |
| --- | ---: |
| `extract_content_blocks` | 104 |
| `filter_entries` | 57 |
| `_emit_individual` | 35 |
| `extract_reasoning_traces` | 27 |
| `emit` | 12 |
| `_emit_grouped` | 9 |
| `_maybe_enrich` | 7 |
| `to_content_blocks` | 5 |
| `_make_raw` | 4 |
| `__init__` | 3 |

## Dominant Timeouts

| Function | Count |
| --- | ---: |
| _none_ | 0 |

## Dominant Not-Checked Clusters

| Function | Count |
| --- | ---: |
| `_load_credentials` | 250 |
| `to_meta` | 102 |
| `iter_json_files` | 95 |
| `resolve_folder_id` | 76 |
| `download_to_path` | 74 |
| `extract_content_blocks` | 69 |
| `_service_handle` | 53 |
| `get_metadata` | 48 |
| `download_json_payload` | 47 |
| `to_content_blocks` | 46 |

## Survivor Keys

- `polylogue.schemas.unified.x_extract_reasoning_traces__mutmut_4`
- `polylogue.schemas.unified.x_extract_reasoning_traces__mutmut_9`
- `polylogue.schemas.unified.x_extract_reasoning_traces__mutmut_18`
- `polylogue.schemas.unified.x_extract_reasoning_traces__mutmut_19`
- `polylogue.schemas.unified.x_extract_reasoning_traces__mutmut_20`
- `polylogue.schemas.unified.x_extract_reasoning_traces__mutmut_21`
- `polylogue.schemas.unified.x_extract_reasoning_traces__mutmut_22`
- `polylogue.schemas.unified.x_extract_reasoning_traces__mutmut_23`
- `polylogue.schemas.unified.x_extract_reasoning_traces__mutmut_24`
- `polylogue.schemas.unified.x_extract_reasoning_traces__mutmut_25`
- `polylogue.schemas.unified.x_extract_reasoning_traces__mutmut_26`
- `polylogue.schemas.unified.x_extract_reasoning_traces__mutmut_27`
- `polylogue.schemas.unified.x_extract_reasoning_traces__mutmut_28`
- `polylogue.schemas.unified.x_extract_reasoning_traces__mutmut_31`
- `polylogue.schemas.unified.x_extract_reasoning_traces__mutmut_34`
- `polylogue.schemas.unified.x_extract_reasoning_traces__mutmut_35`
- `polylogue.schemas.unified.x_extract_tool_calls__mutmut_4`
- `polylogue.schemas.unified.x_extract_tool_calls__mutmut_14`
- `polylogue.schemas.unified.x_extract_tool_calls__mutmut_16`
- `polylogue.schemas.unified.x_extract_tool_calls__mutmut_19`
- `polylogue.schemas.unified.x_extract_tool_calls__mutmut_20`
- `polylogue.schemas.unified.x_extract_tool_calls__mutmut_21`
- `polylogue.schemas.unified.x_extract_tool_calls__mutmut_22`
- `polylogue.schemas.unified.x_extract_tool_calls__mutmut_24`
- `polylogue.schemas.unified.x_extract_tool_calls__mutmut_25`
- ... 2282 more

## Not-Checked Keys

- `polylogue.schemas.unified.x__missing_role__mutmut_1`
- `polylogue.schemas.unified.x__missing_role__mutmut_2`
- `polylogue.schemas.unified.x__missing_role__mutmut_3`
- `polylogue.schemas.unified.x__missing_role__mutmut_4`
- `polylogue.schemas.unified.x_extract_codex_text__mutmut_1`
- `polylogue.schemas.unified.x_extract_codex_text__mutmut_2`
- `polylogue.schemas.unified.x_extract_codex_text__mutmut_3`
- `polylogue.schemas.unified.x_extract_codex_text__mutmut_4`
- `polylogue.schemas.unified.x_extract_codex_text__mutmut_5`
- `polylogue.schemas.unified.x_extract_codex_text__mutmut_6`
- `polylogue.schemas.unified.x_extract_codex_text__mutmut_7`
- `polylogue.schemas.unified.x_extract_codex_text__mutmut_8`
- `polylogue.schemas.unified.x_extract_codex_text__mutmut_9`
- `polylogue.schemas.unified.x_extract_codex_text__mutmut_10`
- `polylogue.schemas.unified.x_extract_codex_text__mutmut_11`
- `polylogue.schemas.unified.x_extract_codex_text__mutmut_12`
- `polylogue.schemas.unified.x_extract_codex_text__mutmut_13`
- `polylogue.schemas.unified.x_extract_codex_text__mutmut_14`
- `polylogue.schemas.unified.x_extract_codex_text__mutmut_15`
- `polylogue.schemas.unified.x_extract_codex_text__mutmut_16`
- `polylogue.schemas.unified.x_extract_codex_text__mutmut_17`
- `polylogue.schemas.unified.x_extract_codex_text__mutmut_18`
- `polylogue.schemas.unified.x_extract_codex_text__mutmut_19`
- `polylogue.schemas.unified.x_extract_codex_text__mutmut_20`
- `polylogue.schemas.unified.x_extract_codex_text__mutmut_21`
- ... 2069 more

## Notes

- Broadest campaign here; best run after law-wave work lands.

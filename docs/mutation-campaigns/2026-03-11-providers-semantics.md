# Mutmut Campaign: `providers-semantics`

- Recorded on `2026-03-11T06:35:38.887231+00:00`
- Commit: `147e689d15caf23fc4036c3af6211af4f71bbaad`
- Worktree dirty: `no`
- Description: Provider semantic extraction, harmonization, and viewport contracts
- Workspace: `/tmp/nix-shell.fzgbVy/mutmut-providers-semantics-yuhxe36u/repo`
- Command: `mutmut run`

## Scope

- Mutated paths: `polylogue/sources/providers`, `polylogue/schemas/unified.py`
- Selected tests: `tests/unit/sources/test_parse_laws.py`, `tests/unit/sources/test_harmonization_contracts.py`, `tests/unit/sources/test_providers.py`, `tests/unit/sources/test_null_guard_properties.py`, `tests/unit/sources/test_models.py`

## Counts

| Status | Count |
| --- | ---: |
| Killed | 162 |
| Survived | 588 |
| Timeout | 0 |
| Not checked | 432 |
| Suspicious | 0 |
| Skipped | 0 |

- Runtime: `35.24s`
- Exit code: `0`

## Dominant Survivors

| Function | Count |
| --- | ---: |
| `extract_content_blocks` | 104 |
| `extract_reasoning_traces` | 27 |
| `to_content_blocks` | 5 |
| `polylogue.schemas.unified.x_extract_reasoning_traces__mutmut_4` | 1 |
| `polylogue.schemas.unified.x_extract_reasoning_traces__mutmut_9` | 1 |
| `polylogue.schemas.unified.x_extract_reasoning_traces__mutmut_18` | 1 |
| `polylogue.schemas.unified.x_extract_reasoning_traces__mutmut_19` | 1 |
| `polylogue.schemas.unified.x_extract_reasoning_traces__mutmut_20` | 1 |
| `polylogue.schemas.unified.x_extract_reasoning_traces__mutmut_21` | 1 |
| `polylogue.schemas.unified.x_extract_reasoning_traces__mutmut_22` | 1 |

## Dominant Timeouts

| Function | Count |
| --- | ---: |
| _none_ | 0 |

## Dominant Not-Checked Clusters

| Function | Count |
| --- | ---: |
| `to_meta` | 102 |
| `extract_content_blocks` | 69 |
| `to_content_blocks` | 46 |
| `iter_user_assistant_pairs` | 23 |
| `to_tool_call` | 18 |
| `to_reasoning_trace` | 8 |
| `to_token_usage` | 8 |
| `extract_reasoning_traces` | 6 |
| `extract_tool_calls` | 6 |
| `polylogue.schemas.unified.x__missing_role__mutmut_1` | 1 |

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
- ... 563 more

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
- ... 407 more

## Notes

- Directly relevant to the next provider-law wave.

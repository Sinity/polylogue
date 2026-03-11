# Mutmut Campaign: `schema-inference`

- Recorded on `2026-03-11T06:28:48.293670+00:00`
- Commit: `147e689d15caf23fc4036c3af6211af4f71bbaad`
- Worktree dirty: `no`
- Description: Schema inference and privacy heuristics
- Workspace: `/tmp/nix-shell.PyeMc7/mutmut-schema-inference-06k8y_li/repo`
- Command: `mutmut run`

## Scope

- Mutated paths: `polylogue/schemas/schema_inference.py`
- Selected tests: `tests/unit/core/test_schema.py`, `tests/unit/core/test_schema_privacy.py`

## Counts

| Status | Count |
| --- | ---: |
| Killed | 536 |
| Survived | 759 |
| Timeout | 3 |
| Not checked | 0 |
| Suspicious | 0 |
| Skipped | 0 |

- Runtime: `130.77s`
- Exit code: `0`

## Dominant Survivors

| Function | Count |
| --- | ---: |
| `polylogue.schemas.schema_inference.x_is_dynamic_key__mutmut_8` | 1 |
| `polylogue.schemas.schema_inference.x_is_dynamic_key__mutmut_10` | 1 |
| `polylogue.schemas.schema_inference.x_is_dynamic_key__mutmut_18` | 1 |
| `polylogue.schemas.schema_inference.x_is_dynamic_key__mutmut_20` | 1 |
| `polylogue.schemas.schema_inference.x__looks_pathlike_key__mutmut_1` | 1 |
| `polylogue.schemas.schema_inference.x__looks_pathlike_key__mutmut_2` | 1 |
| `polylogue.schemas.schema_inference.x__looks_pathlike_key__mutmut_3` | 1 |
| `polylogue.schemas.schema_inference.x__looks_pathlike_key__mutmut_4` | 1 |
| `polylogue.schemas.schema_inference.x__looks_pathlike_key__mutmut_5` | 1 |
| `polylogue.schemas.schema_inference.x__looks_pathlike_key__mutmut_6` | 1 |

## Dominant Timeouts

| Function | Count |
| --- | ---: |
| `polylogue.schemas.schema_inference.x_load_samples_from_sessions__mutmut_17` | 1 |
| `polylogue.schemas.schema_inference.x_generate_provider_schema__mutmut_36` | 1 |
| `polylogue.schemas.schema_inference.x_generate_provider_schema__mutmut_40` | 1 |

## Dominant Not-Checked Clusters

| Function | Count |
| --- | ---: |
| _none_ | 0 |

## Survivor Keys

- `polylogue.schemas.schema_inference.x_is_dynamic_key__mutmut_8`
- `polylogue.schemas.schema_inference.x_is_dynamic_key__mutmut_10`
- `polylogue.schemas.schema_inference.x_is_dynamic_key__mutmut_18`
- `polylogue.schemas.schema_inference.x_is_dynamic_key__mutmut_20`
- `polylogue.schemas.schema_inference.x__looks_pathlike_key__mutmut_1`
- `polylogue.schemas.schema_inference.x__looks_pathlike_key__mutmut_2`
- `polylogue.schemas.schema_inference.x__looks_pathlike_key__mutmut_3`
- `polylogue.schemas.schema_inference.x__looks_pathlike_key__mutmut_4`
- `polylogue.schemas.schema_inference.x__looks_pathlike_key__mutmut_5`
- `polylogue.schemas.schema_inference.x__looks_pathlike_key__mutmut_6`
- `polylogue.schemas.schema_inference.x__looks_pathlike_key__mutmut_7`
- `polylogue.schemas.schema_inference.x__looks_pathlike_key__mutmut_8`
- `polylogue.schemas.schema_inference.x__looks_pathlike_key__mutmut_9`
- `polylogue.schemas.schema_inference.x__looks_pathlike_key__mutmut_10`
- `polylogue.schemas.schema_inference.x__looks_pathlike_key__mutmut_11`
- `polylogue.schemas.schema_inference.x__looks_pathlike_key__mutmut_12`
- `polylogue.schemas.schema_inference.x__looks_pathlike_key__mutmut_13`
- `polylogue.schemas.schema_inference.x__looks_pathlike_key__mutmut_14`
- `polylogue.schemas.schema_inference.x__looks_pathlike_key__mutmut_15`
- `polylogue.schemas.schema_inference.x__looks_pathlike_key__mutmut_16`
- `polylogue.schemas.schema_inference.x__should_collapse_high_cardinality_keys__mutmut_1`
- `polylogue.schemas.schema_inference.x__should_collapse_high_cardinality_keys__mutmut_2`
- `polylogue.schemas.schema_inference.x__should_collapse_high_cardinality_keys__mutmut_3`
- `polylogue.schemas.schema_inference.x__should_collapse_high_cardinality_keys__mutmut_4`
- `polylogue.schemas.schema_inference.x__should_collapse_high_cardinality_keys__mutmut_6`
- ... 734 more

## Timeout Keys

- `polylogue.schemas.schema_inference.x_load_samples_from_sessions__mutmut_17`
- `polylogue.schemas.schema_inference.x_generate_provider_schema__mutmut_36`
- `polylogue.schemas.schema_inference.x_generate_provider_schema__mutmut_40`

# Mutmut Campaign: `schema-inference`

- Recorded on `2026-03-12T03:35:21.908118+00:00`
- Commit: `856caf495bab96724189df08115a5410192f2877`
- Worktree dirty: `no`
- Description: Schema inference and privacy heuristics
- Workspace: `/tmp/nix-shell.cKVsFm/mutmut-schema-inference-640s2lad/repo`
- Command: `mutmut run`

## Scope

- Mutated paths: `polylogue/schemas/schema_inference.py`
- Selected tests: `tests/unit/core/test_schema_generation.py`, `tests/unit/core/test_schema_laws.py`, `tests/unit/core/test_schema_privacy.py`

## Counts

| Status | Count |
| --- | ---: |
| Killed | 561 |
| Survived | 707 |
| Timeout | 30 |
| Not checked | 0 |
| Suspicious | 0 |
| Skipped | 0 |

- Runtime: `309.28s`
- Exit code: `0`

## Dominant Survivors

| Function | Count |
| --- | ---: |
| `polylogue.schemas.schema_inference.x_is_dynamic_key__mutmut_8` | 1 |
| `polylogue.schemas.schema_inference.x_is_dynamic_key__mutmut_10` | 1 |
| `polylogue.schemas.schema_inference.x_is_dynamic_key__mutmut_18` | 1 |
| `polylogue.schemas.schema_inference.x_is_dynamic_key__mutmut_20` | 1 |
| `polylogue.schemas.schema_inference.x__looks_pathlike_key__mutmut_5` | 1 |
| `polylogue.schemas.schema_inference.x__looks_pathlike_key__mutmut_7` | 1 |
| `polylogue.schemas.schema_inference.x__looks_pathlike_key__mutmut_8` | 1 |
| `polylogue.schemas.schema_inference.x__looks_pathlike_key__mutmut_9` | 1 |
| `polylogue.schemas.schema_inference.x__looks_pathlike_key__mutmut_10` | 1 |
| `polylogue.schemas.schema_inference.x__looks_pathlike_key__mutmut_11` | 1 |

## Dominant Timeouts

| Function | Count |
| --- | ---: |
| `polylogue.schemas.schema_inference.x__looks_pathlike_key__mutmut_1` | 1 |
| `polylogue.schemas.schema_inference.x__looks_pathlike_key__mutmut_2` | 1 |
| `polylogue.schemas.schema_inference.x__looks_pathlike_key__mutmut_3` | 1 |
| `polylogue.schemas.schema_inference.x__looks_pathlike_key__mutmut_4` | 1 |
| `polylogue.schemas.schema_inference.x__looks_pathlike_key__mutmut_6` | 1 |
| `polylogue.schemas.schema_inference.x__looks_pathlike_key__mutmut_12` | 1 |
| `polylogue.schemas.schema_inference.x__looks_pathlike_key__mutmut_13` | 1 |
| `polylogue.schemas.schema_inference.x__looks_pathlike_key__mutmut_14` | 1 |
| `polylogue.schemas.schema_inference.x__detect_numeric_format__mutmut_3` | 1 |
| `polylogue.schemas.schema_inference.x__iter_samples_from_sessions__mutmut_16` | 1 |

## Dominant Not-Checked Clusters

| Function | Count |
| --- | ---: |
| _none_ | 0 |

## Survivor Keys

- `polylogue.schemas.schema_inference.x_is_dynamic_key__mutmut_8`
- `polylogue.schemas.schema_inference.x_is_dynamic_key__mutmut_10`
- `polylogue.schemas.schema_inference.x_is_dynamic_key__mutmut_18`
- `polylogue.schemas.schema_inference.x_is_dynamic_key__mutmut_20`
- `polylogue.schemas.schema_inference.x__looks_pathlike_key__mutmut_5`
- `polylogue.schemas.schema_inference.x__looks_pathlike_key__mutmut_7`
- `polylogue.schemas.schema_inference.x__looks_pathlike_key__mutmut_8`
- `polylogue.schemas.schema_inference.x__looks_pathlike_key__mutmut_9`
- `polylogue.schemas.schema_inference.x__looks_pathlike_key__mutmut_10`
- `polylogue.schemas.schema_inference.x__looks_pathlike_key__mutmut_11`
- `polylogue.schemas.schema_inference.x__looks_pathlike_key__mutmut_15`
- `polylogue.schemas.schema_inference.x__looks_pathlike_key__mutmut_16`
- `polylogue.schemas.schema_inference.x__should_collapse_high_cardinality_keys__mutmut_1`
- `polylogue.schemas.schema_inference.x__should_collapse_high_cardinality_keys__mutmut_2`
- `polylogue.schemas.schema_inference.x__should_collapse_high_cardinality_keys__mutmut_3`
- `polylogue.schemas.schema_inference.x__should_collapse_high_cardinality_keys__mutmut_4`
- `polylogue.schemas.schema_inference.x__should_collapse_high_cardinality_keys__mutmut_6`
- `polylogue.schemas.schema_inference.x__should_collapse_high_cardinality_keys__mutmut_7`
- `polylogue.schemas.schema_inference.x__should_collapse_high_cardinality_keys__mutmut_8`
- `polylogue.schemas.schema_inference.x__should_collapse_high_cardinality_keys__mutmut_9`
- `polylogue.schemas.schema_inference.x__should_collapse_high_cardinality_keys__mutmut_10`
- `polylogue.schemas.schema_inference.x__should_collapse_high_cardinality_keys__mutmut_11`
- `polylogue.schemas.schema_inference.x_collapse_dynamic_keys__mutmut_18`
- `polylogue.schemas.schema_inference.x_collapse_dynamic_keys__mutmut_19`
- `polylogue.schemas.schema_inference.x_collapse_dynamic_keys__mutmut_20`
- ... 682 more

## Timeout Keys

- `polylogue.schemas.schema_inference.x__looks_pathlike_key__mutmut_1`
- `polylogue.schemas.schema_inference.x__looks_pathlike_key__mutmut_2`
- `polylogue.schemas.schema_inference.x__looks_pathlike_key__mutmut_3`
- `polylogue.schemas.schema_inference.x__looks_pathlike_key__mutmut_4`
- `polylogue.schemas.schema_inference.x__looks_pathlike_key__mutmut_6`
- `polylogue.schemas.schema_inference.x__looks_pathlike_key__mutmut_12`
- `polylogue.schemas.schema_inference.x__looks_pathlike_key__mutmut_13`
- `polylogue.schemas.schema_inference.x__looks_pathlike_key__mutmut_14`
- `polylogue.schemas.schema_inference.x__detect_numeric_format__mutmut_3`
- `polylogue.schemas.schema_inference.x__iter_samples_from_sessions__mutmut_16`
- `polylogue.schemas.schema_inference.x_load_samples_from_sessions__mutmut_14`
- `polylogue.schemas.schema_inference.x_load_samples_from_sessions__mutmut_17`
- `polylogue.schemas.schema_inference.x_generate_provider_schema__mutmut_12`
- `polylogue.schemas.schema_inference.x_generate_provider_schema__mutmut_26`
- `polylogue.schemas.schema_inference.x_generate_provider_schema__mutmut_36`
- `polylogue.schemas.schema_inference.x_generate_provider_schema__mutmut_40`
- `polylogue.schemas.schema_inference.x_generate_provider_schema__mutmut_174`
- `polylogue.schemas.schema_inference.x_generate_provider_schema__mutmut_175`
- `polylogue.schemas.schema_inference.x_generate_provider_schema__mutmut_176`
- `polylogue.schemas.schema_inference.x_generate_provider_schema__mutmut_177`
- `polylogue.schemas.schema_inference.x_generate_provider_schema__mutmut_178`
- `polylogue.schemas.schema_inference.x_generate_provider_schema__mutmut_179`
- `polylogue.schemas.schema_inference.x_generate_provider_schema__mutmut_180`
- `polylogue.schemas.schema_inference.x_generate_provider_schema__mutmut_183`
- `polylogue.schemas.schema_inference.x_generate_provider_schema__mutmut_184`
- ... 5 more

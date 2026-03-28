# Mutmut Campaign: `schema-core`

- Recorded on `2026-03-12T03:44:19.397336+00:00`
- Commit: `856caf495bab96724189df08115a5410192f2877`
- Worktree dirty: `no`
- Description: Schema generation, privacy, verification, and safety contracts
- Workspace: `/tmp/nix-shell.ksczLl/mutmut-schema-core-_67lj43n/repo`
- Command: `mutmut run`

## Scope

- Mutated paths: `polylogue/schemas/schema_inference.py`, `polylogue/schemas/validator.py`, `polylogue/schemas/verification.py`
- Selected tests: `tests/unit/core/test_schema_validation.py`, `tests/unit/core/test_schema_generation.py`, `tests/unit/core/test_schema_annotations.py`, `tests/unit/core/test_schema_laws.py`, `tests/unit/core/test_schema_privacy.py`, `tests/unit/core/test_schema_verification.py`, `tests/unit/storage/test_schema_safety.py`

## Counts

| Status | Count |
| --- | ---: |
| Killed | 795 |
| Survived | 883 |
| Timeout | 16 |
| Not checked | 0 |
| Suspicious | 0 |
| Skipped | 0 |

- Runtime: `326.14s`
- Exit code: `0`

## Dominant Survivors

| Function | Count |
| --- | ---: |
| `_detect_drift` | 29 |
| `validation_samples` | 6 |
| `__init__` | 5 |
| `_looks_dynamic_key` | 4 |
| `validate` | 2 |
| `_format_error` | 2 |
| `polylogue.schemas.schema_inference.x_is_dynamic_key__mutmut_8` | 1 |
| `polylogue.schemas.schema_inference.x_is_dynamic_key__mutmut_10` | 1 |
| `polylogue.schemas.schema_inference.x_is_dynamic_key__mutmut_18` | 1 |
| `polylogue.schemas.schema_inference.x_is_dynamic_key__mutmut_20` | 1 |

## Dominant Timeouts

| Function | Count |
| --- | ---: |
| `polylogue.schemas.schema_inference.x__annotate_schema__mutmut_15` | 1 |
| `polylogue.schemas.schema_inference.x__annotate_schema__mutmut_17` | 1 |
| `polylogue.schemas.schema_inference.x__annotate_schema__mutmut_18` | 1 |
| `polylogue.schemas.schema_inference.x__annotate_schema__mutmut_19` | 1 |
| `polylogue.schemas.schema_inference.x__annotate_schema__mutmut_20` | 1 |
| `polylogue.schemas.schema_inference.x__annotate_schema__mutmut_21` | 1 |
| `polylogue.schemas.schema_inference.x__annotate_schema__mutmut_22` | 1 |
| `polylogue.schemas.schema_inference.x__annotate_schema__mutmut_70` | 1 |
| `polylogue.schemas.schema_inference.x__iter_samples_from_sessions__mutmut_16` | 1 |
| `polylogue.schemas.schema_inference.x_load_samples_from_sessions__mutmut_17` | 1 |

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
- ... 858 more

## Timeout Keys

- `polylogue.schemas.schema_inference.x__annotate_schema__mutmut_15`
- `polylogue.schemas.schema_inference.x__annotate_schema__mutmut_17`
- `polylogue.schemas.schema_inference.x__annotate_schema__mutmut_18`
- `polylogue.schemas.schema_inference.x__annotate_schema__mutmut_19`
- `polylogue.schemas.schema_inference.x__annotate_schema__mutmut_20`
- `polylogue.schemas.schema_inference.x__annotate_schema__mutmut_21`
- `polylogue.schemas.schema_inference.x__annotate_schema__mutmut_22`
- `polylogue.schemas.schema_inference.x__annotate_schema__mutmut_70`
- `polylogue.schemas.schema_inference.x__iter_samples_from_sessions__mutmut_16`
- `polylogue.schemas.schema_inference.x_load_samples_from_sessions__mutmut_17`
- `polylogue.schemas.schema_inference.x_generate_provider_schema__mutmut_26`
- `polylogue.schemas.schema_inference.x_generate_provider_schema__mutmut_36`
- `polylogue.schemas.schema_inference.x_generate_provider_schema__mutmut_40`
- `polylogue.schemas.schema_inference.x_generate_provider_schema__mutmut_50`
- `polylogue.schemas.schema_inference.x_generate_provider_schema__mutmut_52`
- `polylogue.schemas.schema_inference.x_generate_provider_schema__mutmut_53`

## Notes

- Larger campaign; use when law and privacy work are stable.

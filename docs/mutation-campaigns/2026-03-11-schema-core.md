# Mutmut Campaign: `schema-core`

- Recorded on `2026-03-11T06:25:22.335205+00:00`
- Commit: `147e689d15caf23fc4036c3af6211af4f71bbaad`
- Worktree dirty: `no`
- Description: Schema generation, privacy, verification, and safety contracts
- Workspace: `/tmp/nix-shell.cjPCW2/mutmut-schema-core-e_yexbrl/repo`
- Command: `mutmut run`

## Scope

- Mutated paths: `polylogue/schemas/schema_inference.py`, `polylogue/schemas/validator.py`, `polylogue/schemas/verification.py`
- Selected tests: `tests/unit/core/test_schema.py`, `tests/unit/core/test_schema_privacy.py`, `tests/unit/core/test_schema_verification.py`, `tests/unit/storage/test_schema_safety.py`

## Counts

| Status | Count |
| --- | ---: |
| Killed | 765 |
| Survived | 900 |
| Timeout | 29 |
| Not checked | 0 |
| Suspicious | 0 |
| Skipped | 0 |

- Runtime: `205.63s`
- Exit code: `0`

## Dominant Survivors

| Function | Count |
| --- | ---: |
| `_detect_drift` | 29 |
| `validation_samples` | 12 |
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
| `polylogue.schemas.schema_inference.x_load_samples_from_sessions__mutmut_17` | 1 |
| `polylogue.schemas.schema_inference.x_generate_provider_schema__mutmut_36` | 1 |
| `polylogue.schemas.schema_inference.x_generate_provider_schema__mutmut_40` | 1 |
| `polylogue.schemas.schema_inference.x_generate_provider_schema__mutmut_71` | 1 |
| `polylogue.schemas.schema_inference.x_generate_provider_schema__mutmut_72` | 1 |
| `polylogue.schemas.schema_inference.x_generate_provider_schema__mutmut_73` | 1 |
| `polylogue.schemas.schema_inference.x_generate_provider_schema__mutmut_74` | 1 |
| `polylogue.schemas.schema_inference.x_generate_provider_schema__mutmut_75` | 1 |
| `polylogue.schemas.schema_inference.x_generate_provider_schema__mutmut_76` | 1 |
| `polylogue.schemas.schema_inference.x_generate_provider_schema__mutmut_77` | 1 |

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
- ... 875 more

## Timeout Keys

- `polylogue.schemas.schema_inference.x_load_samples_from_sessions__mutmut_17`
- `polylogue.schemas.schema_inference.x_generate_provider_schema__mutmut_36`
- `polylogue.schemas.schema_inference.x_generate_provider_schema__mutmut_40`
- `polylogue.schemas.schema_inference.x_generate_provider_schema__mutmut_71`
- `polylogue.schemas.schema_inference.x_generate_provider_schema__mutmut_72`
- `polylogue.schemas.schema_inference.x_generate_provider_schema__mutmut_73`
- `polylogue.schemas.schema_inference.x_generate_provider_schema__mutmut_74`
- `polylogue.schemas.schema_inference.x_generate_provider_schema__mutmut_75`
- `polylogue.schemas.schema_inference.x_generate_provider_schema__mutmut_76`
- `polylogue.schemas.schema_inference.x_generate_provider_schema__mutmut_77`
- `polylogue.schemas.schema_inference.x_generate_provider_schema__mutmut_78`
- `polylogue.schemas.schema_inference.x_generate_provider_schema__mutmut_79`
- `polylogue.schemas.schema_inference.x_generate_provider_schema__mutmut_80`
- `polylogue.schemas.schema_inference.x_generate_provider_schema__mutmut_81`
- `polylogue.schemas.schema_inference.x_generate_provider_schema__mutmut_82`
- `polylogue.schemas.schema_inference.x_generate_provider_schema__mutmut_83`
- `polylogue.schemas.schema_inference.x_generate_provider_schema__mutmut_84`
- `polylogue.schemas.schema_inference.x_generate_provider_schema__mutmut_85`
- `polylogue.schemas.schema_inference.x_generate_provider_schema__mutmut_86`
- `polylogue.schemas.schema_inference.x_generate_provider_schema__mutmut_87`
- `polylogue.schemas.schema_inference.x_generate_provider_schema__mutmut_88`
- `polylogue.schemas.schema_inference.x_generate_provider_schema__mutmut_89`
- `polylogue.schemas.schema_inference.x_generate_provider_schema__mutmut_90`
- `polylogue.schemas.schema_inference.x_generate_provider_schema__mutmut_91`
- `polylogue.schemas.schema_inference.x_generate_provider_schema__mutmut_92`
- ... 4 more

## Notes

- Larger campaign; use when law and privacy work are stable.

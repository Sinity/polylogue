# Mutmut Campaign: `schema-validation`

- Recorded on `2026-03-11T07:32:42.298458+00:00`
- Commit: `d1e704d7a2ba9f3ab5bd2357487c6d5c967eddb5`
- Worktree dirty: `no`
- Description: Schema validator and verification contracts
- Workspace: `/tmp/nix-shell.QLNA20/mutmut-schema-validation-_qcbr7ln/repo`
- Command: `mutmut run`

## Scope

- Mutated paths: `polylogue/schemas/validator.py`, `polylogue/schemas/verification.py`
- Selected tests: `tests/unit/core/test_schema.py`, `tests/unit/core/test_schema_laws.py`, `tests/unit/core/test_schema_verification.py`, `tests/unit/storage/test_schema_safety.py`

## Counts

| Status | Count |
| --- | ---: |
| Killed | 235 |
| Survived | 161 |
| Timeout | 0 |
| Not checked | 0 |
| Suspicious | 0 |
| Skipped | 0 |

- Runtime: `43.36s`
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
| `polylogue.schemas.validator.x_validate_provider_export__mutmut_1` | 1 |
| `polylogue.schemas.validator.x_validate_provider_export__mutmut_4` | 1 |
| `polylogue.schemas.validator.x_validate_provider_export__mutmut_6` | 1 |
| `polylogue.schemas.verification.x__verification_provider_clause__mutmut_3` | 1 |

## Dominant Timeouts

| Function | Count |
| --- | ---: |
| _none_ | 0 |

## Dominant Not-Checked Clusters

| Function | Count |
| --- | ---: |
| _none_ | 0 |

## Survivor Keys

- `polylogue.schemas.validator.xǁSchemaValidatorǁ__init____mutmut_1`
- `polylogue.schemas.validator.xǁSchemaValidatorǁ__init____mutmut_3`
- `polylogue.schemas.validator.xǁSchemaValidatorǁ__init____mutmut_4`
- `polylogue.schemas.validator.xǁSchemaValidatorǁ__init____mutmut_5`
- `polylogue.schemas.validator.xǁSchemaValidatorǁ__init____mutmut_6`
- `polylogue.schemas.validator.xǁSchemaValidatorǁvalidate__mutmut_10`
- `polylogue.schemas.validator.xǁSchemaValidatorǁvalidate__mutmut_14`
- `polylogue.schemas.validator.xǁSchemaValidatorǁvalidation_samples__mutmut_1`
- `polylogue.schemas.validator.xǁSchemaValidatorǁvalidation_samples__mutmut_2`
- `polylogue.schemas.validator.xǁSchemaValidatorǁvalidation_samples__mutmut_3`
- `polylogue.schemas.validator.xǁSchemaValidatorǁvalidation_samples__mutmut_4`
- `polylogue.schemas.validator.xǁSchemaValidatorǁvalidation_samples__mutmut_7`
- `polylogue.schemas.validator.xǁSchemaValidatorǁvalidation_samples__mutmut_8`
- `polylogue.schemas.validator.xǁSchemaValidatorǁ_format_error__mutmut_6`
- `polylogue.schemas.validator.xǁSchemaValidatorǁ_format_error__mutmut_7`
- `polylogue.schemas.validator.xǁSchemaValidatorǁ_detect_drift__mutmut_12`
- `polylogue.schemas.validator.xǁSchemaValidatorǁ_detect_drift__mutmut_14`
- `polylogue.schemas.validator.xǁSchemaValidatorǁ_detect_drift__mutmut_23`
- `polylogue.schemas.validator.xǁSchemaValidatorǁ_detect_drift__mutmut_25`
- `polylogue.schemas.validator.xǁSchemaValidatorǁ_detect_drift__mutmut_26`
- `polylogue.schemas.validator.xǁSchemaValidatorǁ_detect_drift__mutmut_27`
- `polylogue.schemas.validator.xǁSchemaValidatorǁ_detect_drift__mutmut_28`
- `polylogue.schemas.validator.xǁSchemaValidatorǁ_detect_drift__mutmut_29`
- `polylogue.schemas.validator.xǁSchemaValidatorǁ_detect_drift__mutmut_30`
- `polylogue.schemas.validator.xǁSchemaValidatorǁ_detect_drift__mutmut_46`
- ... 136 more

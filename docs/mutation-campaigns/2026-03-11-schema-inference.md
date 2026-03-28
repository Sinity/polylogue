# Mutmut Campaign: `schema-inference`

- Recorded on `2026-03-11T07:33:56.368190+00:00`
- Commit: `d1e704d7a2ba9f3ab5bd2357487c6d5c967eddb5`
- Worktree dirty: `no`
- Description: Schema inference and privacy heuristics
- Workspace: `/tmp/nix-shell.dYTHuX/mutmut-schema-inference-pdbrpok9/repo`
- Command: `mutmut run`

## Scope

- Mutated paths: `polylogue/schemas/schema_inference.py`
- Selected tests: `tests/unit/core/test_schema.py`, `tests/unit/core/test_schema_laws.py`, `tests/unit/core/test_schema_privacy.py`

## Counts

| Status | Count |
| --- | ---: |
| Killed | 534 |
| Survived | 317 |
| Timeout | 447 |
| Not checked | 0 |
| Suspicious | 0 |
| Skipped | 0 |

- Runtime: `497.54s`
- Exit code: `0`

## Dominant Survivors

| Function | Count |
| --- | ---: |
| `polylogue.schemas.schema_inference.x_is_dynamic_key__mutmut_8` | 1 |
| `polylogue.schemas.schema_inference.x_is_dynamic_key__mutmut_10` | 1 |
| `polylogue.schemas.schema_inference.x_is_dynamic_key__mutmut_18` | 1 |
| `polylogue.schemas.schema_inference.x_is_dynamic_key__mutmut_20` | 1 |
| `polylogue.schemas.schema_inference.x__should_collapse_high_cardinality_keys__mutmut_1` | 1 |
| `polylogue.schemas.schema_inference.x__should_collapse_high_cardinality_keys__mutmut_2` | 1 |
| `polylogue.schemas.schema_inference.x__should_collapse_high_cardinality_keys__mutmut_3` | 1 |
| `polylogue.schemas.schema_inference.x__should_collapse_high_cardinality_keys__mutmut_4` | 1 |
| `polylogue.schemas.schema_inference.x__should_collapse_high_cardinality_keys__mutmut_6` | 1 |
| `polylogue.schemas.schema_inference.x__should_collapse_high_cardinality_keys__mutmut_7` | 1 |

## Dominant Timeouts

| Function | Count |
| --- | ---: |
| `polylogue.schemas.schema_inference.x__looks_pathlike_key__mutmut_1` | 1 |
| `polylogue.schemas.schema_inference.x__looks_pathlike_key__mutmut_2` | 1 |
| `polylogue.schemas.schema_inference.x__looks_pathlike_key__mutmut_3` | 1 |
| `polylogue.schemas.schema_inference.x__looks_pathlike_key__mutmut_4` | 1 |
| `polylogue.schemas.schema_inference.x__looks_pathlike_key__mutmut_5` | 1 |
| `polylogue.schemas.schema_inference.x__looks_pathlike_key__mutmut_6` | 1 |
| `polylogue.schemas.schema_inference.x__looks_pathlike_key__mutmut_7` | 1 |
| `polylogue.schemas.schema_inference.x__looks_pathlike_key__mutmut_8` | 1 |
| `polylogue.schemas.schema_inference.x__looks_pathlike_key__mutmut_9` | 1 |
| `polylogue.schemas.schema_inference.x__looks_pathlike_key__mutmut_10` | 1 |

## Dominant Not-Checked Clusters

| Function | Count |
| --- | ---: |
| _none_ | 0 |

## Survivor Keys

- `polylogue.schemas.schema_inference.x_is_dynamic_key__mutmut_8`
- `polylogue.schemas.schema_inference.x_is_dynamic_key__mutmut_10`
- `polylogue.schemas.schema_inference.x_is_dynamic_key__mutmut_18`
- `polylogue.schemas.schema_inference.x_is_dynamic_key__mutmut_20`
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
- `polylogue.schemas.schema_inference.x_collapse_dynamic_keys__mutmut_21`
- `polylogue.schemas.schema_inference.x_collapse_dynamic_keys__mutmut_22`
- `polylogue.schemas.schema_inference.x_collapse_dynamic_keys__mutmut_23`
- `polylogue.schemas.schema_inference.x_collapse_dynamic_keys__mutmut_27`
- `polylogue.schemas.schema_inference.x_collapse_dynamic_keys__mutmut_29`
- `polylogue.schemas.schema_inference.x_collapse_dynamic_keys__mutmut_47`
- `polylogue.schemas.schema_inference.x_collapse_dynamic_keys__mutmut_48`
- `polylogue.schemas.schema_inference.x_collapse_dynamic_keys__mutmut_49`
- ... 292 more

## Timeout Keys

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
- `polylogue.schemas.schema_inference.x__detect_string_format__mutmut_1`
- `polylogue.schemas.schema_inference.x__detect_string_format__mutmut_2`
- `polylogue.schemas.schema_inference.x__detect_string_format__mutmut_3`
- `polylogue.schemas.schema_inference.x__detect_string_format__mutmut_4`
- `polylogue.schemas.schema_inference.x__detect_numeric_format__mutmut_1`
- `polylogue.schemas.schema_inference.x__detect_numeric_format__mutmut_2`
- `polylogue.schemas.schema_inference.x__detect_numeric_format__mutmut_3`
- `polylogue.schemas.schema_inference.x__detect_numeric_format__mutmut_4`
- `polylogue.schemas.schema_inference.x__detect_numeric_format__mutmut_5`
- ... 422 more

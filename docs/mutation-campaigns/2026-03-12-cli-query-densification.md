# Mutmut Campaign: `cli-query`

- Recorded on `2026-03-12T09:59:22.612021+00:00`
- Commit: `eb43cfd48e989a58780604e9cbf9d3ba93700ff8`
- Worktree dirty: `yes`
- Description: Query command planning, action routing, and summary output contracts
- Workspace: `/tmp/nix-shell.E1DGhF/mutmut-cli-query-zuls1a49/repo`
- Command: `mutmut run`

## Scope

- Mutated paths: `polylogue/cli/query.py`, `polylogue/cli/query_plan.py`, `polylogue/cli/query_actions.py`, `polylogue/cli/query_output.py`
- Selected tests: `tests/unit/cli/test_query.py`, `tests/unit/cli/test_query_exec.py`, `tests/unit/cli/test_query_exec_laws.py`, `tests/unit/cli/test_query_fmt.py`, `tests/unit/cli/test_query_plan.py`

## Counts

| Status | Count |
| --- | ---: |
| Killed | 1022 |
| Survived | 932 |
| Timeout | 8 |
| Not checked | 0 |
| Suspicious | 0 |
| Skipped | 0 |

- Runtime: `66.24s`
- Exit code: `0`

## Dominant Survivors

| Function | Count |
| --- | ---: |
| `polylogue.cli.query.x__create_query_vector_provider__mutmut_1` | 1 |
| `polylogue.cli.query.x__create_query_vector_provider__mutmut_3` | 1 |
| `polylogue.cli.query.x__create_query_vector_provider__mutmut_5` | 1 |
| `polylogue.cli.query.x__async_execute_query__mutmut_2` | 1 |
| `polylogue.cli.query.x__async_execute_query__mutmut_12` | 1 |
| `polylogue.cli.query.x__async_execute_query__mutmut_56` | 1 |
| `polylogue.cli.query.x__async_execute_query__mutmut_57` | 1 |
| `polylogue.cli.query.x__async_execute_query__mutmut_58` | 1 |
| `polylogue.cli.query.x__async_execute_query__mutmut_59` | 1 |
| `polylogue.cli.query.x__async_execute_query__mutmut_60` | 1 |

## Dominant Timeouts

| Function | Count |
| --- | ---: |
| `polylogue.cli.query_actions.x__apply_modifiers__mutmut_16` | 1 |
| `polylogue.cli.query_actions.x__apply_modifiers__mutmut_17` | 1 |
| `polylogue.cli.query_actions.x__apply_modifiers__mutmut_19` | 1 |
| `polylogue.cli.query_actions.x__apply_modifiers__mutmut_21` | 1 |
| `polylogue.cli.query_actions.x__delete_conversations__mutmut_3` | 1 |
| `polylogue.cli.query_actions.x__delete_conversations__mutmut_69` | 1 |
| `polylogue.cli.query_actions.x__delete_conversations__mutmut_72` | 1 |
| `polylogue.cli.query_actions.x__delete_conversations__mutmut_73` | 1 |

## Dominant Not-Checked Clusters

| Function | Count |
| --- | ---: |
| _none_ | 0 |

## Survivor Keys

- `polylogue.cli.query.x__create_query_vector_provider__mutmut_1`
- `polylogue.cli.query.x__create_query_vector_provider__mutmut_3`
- `polylogue.cli.query.x__create_query_vector_provider__mutmut_5`
- `polylogue.cli.query.x__async_execute_query__mutmut_2`
- `polylogue.cli.query.x__async_execute_query__mutmut_12`
- `polylogue.cli.query.x__async_execute_query__mutmut_56`
- `polylogue.cli.query.x__async_execute_query__mutmut_57`
- `polylogue.cli.query.x__async_execute_query__mutmut_58`
- `polylogue.cli.query.x__async_execute_query__mutmut_59`
- `polylogue.cli.query.x__async_execute_query__mutmut_60`
- `polylogue.cli.query.x__async_execute_query__mutmut_61`
- `polylogue.cli.query.x__async_execute_query__mutmut_62`
- `polylogue.cli.query.x__async_execute_query__mutmut_63`
- `polylogue.cli.query.x__async_execute_query__mutmut_74`
- `polylogue.cli.query.x__async_execute_query__mutmut_76`
- `polylogue.cli.query.x__async_execute_query__mutmut_77`
- `polylogue.cli.query.x__async_execute_query__mutmut_80`
- `polylogue.cli.query.x__async_execute_query__mutmut_86`
- `polylogue.cli.query.x__async_execute_query__mutmut_88`
- `polylogue.cli.query.x__async_execute_query__mutmut_91`
- `polylogue.cli.query.x__async_execute_query__mutmut_108`
- `polylogue.cli.query.x__async_execute_query__mutmut_109`
- `polylogue.cli.query.x__async_execute_query__mutmut_110`
- `polylogue.cli.query.x__async_execute_query__mutmut_111`
- `polylogue.cli.query.x__async_execute_query__mutmut_112`
- ... 907 more

## Timeout Keys

- `polylogue.cli.query_actions.x__apply_modifiers__mutmut_16`
- `polylogue.cli.query_actions.x__apply_modifiers__mutmut_17`
- `polylogue.cli.query_actions.x__apply_modifiers__mutmut_19`
- `polylogue.cli.query_actions.x__apply_modifiers__mutmut_21`
- `polylogue.cli.query_actions.x__delete_conversations__mutmut_3`
- `polylogue.cli.query_actions.x__delete_conversations__mutmut_69`
- `polylogue.cli.query_actions.x__delete_conversations__mutmut_72`
- `polylogue.cli.query_actions.x__delete_conversations__mutmut_73`

## Source Worktree Status

- ` M tests/unit/cli/test_helpers.py`
- ` M tests/unit/cli/test_query_exec.py`
- ` M tests/unit/core/test_conversation_semantics.py`
- ` M tests/unit/core/test_dates.py`
- ` M tests/unit/core/test_filters_adv.py`
- ` M tests/unit/core/test_filters_props.py`
- ` M tests/unit/core/test_hashing.py`
- ` M tests/unit/core/test_message_laws.py`
- ` M tests/unit/core/test_models.py`
- ` D tests/unit/core/test_properties.py`
- ` M tests/unit/sources/test_claude.py`
- ` M tests/unit/sources/test_content_extraction.py`
- ` M tests/unit/sources/test_extraction.py`
- ` M tests/unit/sources/test_parsers.py`
- ` M tests/unit/sources/test_seeded_parser_contracts.py`
- ` M tests/unit/storage/test_backend.py`
- ` M tests/unit/storage/test_hybrid_laws.py`
- ` M tests/unit/storage/test_repository_laws.py`
- ` M tests/unit/storage/test_store_ops.py`
- ` M tests/unit/storage/test_vec.py`

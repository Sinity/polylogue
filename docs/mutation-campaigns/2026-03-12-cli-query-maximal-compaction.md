# Mutmut Campaign: `cli-query`

- Recorded on `2026-03-12T14:08:09.840738+00:00`
- Commit: `c1fd5ce60e8216a714acfc59072597ee40955a66`
- Worktree dirty: `yes`
- Description: Query command planning, action routing, and summary output contracts
- Workspace: `/tmp/nix-shell.JJuv2z/mutmut-cli-query-10xd3he4/repo`
- Command: `mutmut run`

## Scope

- Mutated paths: `polylogue/cli/query.py`, `polylogue/cli/query_plan.py`, `polylogue/cli/query_actions.py`, `polylogue/cli/query_output.py`
- Selected tests: `tests/unit/cli/test_query.py`, `tests/unit/cli/test_query_exec.py`, `tests/unit/cli/test_query_exec_laws.py`, `tests/unit/cli/test_query_fmt.py`, `tests/unit/cli/test_query_plan.py`

## Counts

| Status | Count |
| --- | ---: |
| Killed | 1013 |
| Survived | 935 |
| Timeout | 14 |
| Not checked | 0 |
| Suspicious | 0 |
| Skipped | 0 |

- Runtime: `71.48s`
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
| `polylogue.cli.query_actions.x__apply_modifiers__mutmut_3` | 1 |
| `polylogue.cli.query_actions.x__apply_modifiers__mutmut_16` | 1 |
| `polylogue.cli.query_actions.x__apply_modifiers__mutmut_17` | 1 |
| `polylogue.cli.query_actions.x__apply_modifiers__mutmut_22` | 1 |
| `polylogue.cli.query_actions.x__apply_modifiers__mutmut_63` | 1 |
| `polylogue.cli.query_actions.x__apply_modifiers__mutmut_67` | 1 |
| `polylogue.cli.query_actions.x__apply_modifiers__mutmut_68` | 1 |
| `polylogue.cli.query_actions.x__apply_modifiers__mutmut_69` | 1 |
| `polylogue.cli.query_actions.x__apply_modifiers__mutmut_70` | 1 |
| `polylogue.cli.query_actions.x__apply_modifiers__mutmut_71` | 1 |

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
- ... 910 more

## Timeout Keys

- `polylogue.cli.query_actions.x__apply_modifiers__mutmut_3`
- `polylogue.cli.query_actions.x__apply_modifiers__mutmut_16`
- `polylogue.cli.query_actions.x__apply_modifiers__mutmut_17`
- `polylogue.cli.query_actions.x__apply_modifiers__mutmut_22`
- `polylogue.cli.query_actions.x__apply_modifiers__mutmut_63`
- `polylogue.cli.query_actions.x__apply_modifiers__mutmut_67`
- `polylogue.cli.query_actions.x__apply_modifiers__mutmut_68`
- `polylogue.cli.query_actions.x__apply_modifiers__mutmut_69`
- `polylogue.cli.query_actions.x__apply_modifiers__mutmut_70`
- `polylogue.cli.query_actions.x__apply_modifiers__mutmut_71`
- `polylogue.cli.query_actions.x__apply_modifiers__mutmut_72`
- `polylogue.cli.query_actions.x__apply_modifiers__mutmut_73`
- `polylogue.cli.query_actions.x__delete_conversations__mutmut_3`
- `polylogue.cli.query_output.x__output_stats_by_summaries__mutmut_18`

## Source Worktree Status

- ` M tests/unit/cli/test_query_exec_laws.py`
- ` M tests/unit/core/test_health_core.py`
- ` M tests/unit/sources/test_drive_utils.py`
- ` M tests/unit/sources/test_source_laws.py`
- ` M tests/unit/sources/test_unified_semantic_laws.py`
- ` M tests/unit/ui/test_rendering.py`

# Mutmut Campaign: `cli-query`

- Recorded on `2026-03-12T04:14:00.022579+00:00`
- Commit: `58264c2c47beaaa5522139c54700c20910833267`
- Worktree dirty: `no`
- Description: Query command planning, action routing, and summary output contracts
- Workspace: `/tmp/nix-shell.HnMjMk/mutmut-cli-query-iehom3tw/repo`
- Command: `mutmut run`

## Scope

- Mutated paths: `polylogue/cli/query.py`, `polylogue/cli/query_plan.py`, `polylogue/cli/query_actions.py`, `polylogue/cli/query_output.py`
- Selected tests: `tests/unit/cli/test_query.py`, `tests/unit/cli/test_query_exec.py`, `tests/unit/cli/test_query_exec_laws.py`, `tests/unit/cli/test_query_fmt.py`, `tests/unit/cli/test_query_plan.py`

## Counts

| Status | Count |
| --- | ---: |
| Killed | 954 |
| Survived | 985 |
| Timeout | 23 |
| Not checked | 0 |
| Suspicious | 0 |
| Skipped | 0 |

- Runtime: `107.32s`
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
| `polylogue.cli.query_actions.x__apply_modifiers__mutmut_8` | 1 |
| `polylogue.cli.query_actions.x__apply_modifiers__mutmut_9` | 1 |
| `polylogue.cli.query_actions.x__apply_modifiers__mutmut_17` | 1 |
| `polylogue.cli.query_actions.x__apply_modifiers__mutmut_19` | 1 |
| `polylogue.cli.query_actions.x__apply_modifiers__mutmut_21` | 1 |
| `polylogue.cli.query_actions.x__apply_modifiers__mutmut_22` | 1 |
| `polylogue.cli.query_actions.x__apply_modifiers__mutmut_62` | 1 |
| `polylogue.cli.query_actions.x__apply_modifiers__mutmut_67` | 1 |
| `polylogue.cli.query_actions.x__apply_modifiers__mutmut_68` | 1 |

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
- `polylogue.cli.query.x__async_execute_query__mutmut_83`
- `polylogue.cli.query.x__async_execute_query__mutmut_84`
- `polylogue.cli.query.x__async_execute_query__mutmut_86`
- `polylogue.cli.query.x__async_execute_query__mutmut_88`
- `polylogue.cli.query.x__async_execute_query__mutmut_91`
- `polylogue.cli.query.x__async_execute_query__mutmut_108`
- `polylogue.cli.query.x__async_execute_query__mutmut_109`
- `polylogue.cli.query.x__async_execute_query__mutmut_110`
- ... 960 more

## Timeout Keys

- `polylogue.cli.query_actions.x__apply_modifiers__mutmut_3`
- `polylogue.cli.query_actions.x__apply_modifiers__mutmut_8`
- `polylogue.cli.query_actions.x__apply_modifiers__mutmut_9`
- `polylogue.cli.query_actions.x__apply_modifiers__mutmut_17`
- `polylogue.cli.query_actions.x__apply_modifiers__mutmut_19`
- `polylogue.cli.query_actions.x__apply_modifiers__mutmut_21`
- `polylogue.cli.query_actions.x__apply_modifiers__mutmut_22`
- `polylogue.cli.query_actions.x__apply_modifiers__mutmut_62`
- `polylogue.cli.query_actions.x__apply_modifiers__mutmut_67`
- `polylogue.cli.query_actions.x__apply_modifiers__mutmut_68`
- `polylogue.cli.query_actions.x__apply_modifiers__mutmut_69`
- `polylogue.cli.query_actions.x__apply_modifiers__mutmut_70`
- `polylogue.cli.query_actions.x__apply_modifiers__mutmut_71`
- `polylogue.cli.query_actions.x__apply_modifiers__mutmut_72`
- `polylogue.cli.query_actions.x__apply_modifiers__mutmut_73`
- `polylogue.cli.query_actions.x__apply_modifiers__mutmut_74`
- `polylogue.cli.query_actions.x__delete_conversations__mutmut_3`
- `polylogue.cli.query_actions.x__delete_conversations__mutmut_69`
- `polylogue.cli.query_actions.x__delete_conversations__mutmut_81`
- `polylogue.cli.query_actions.x__delete_conversations__mutmut_83`
- `polylogue.cli.query_actions.x__delete_conversations__mutmut_84`
- `polylogue.cli.query_actions.x__delete_conversations__mutmut_85`
- `polylogue.cli.query_output.x__output_stats_by_summaries__mutmut_15`

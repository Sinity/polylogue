# Mutmut Campaign: `cli-query`

- Recorded on `2026-03-11T20:28:01.134608+00:00`
- Commit: `e058c8240959fc530a1d97cc2d47f15840189cf1`
- Worktree dirty: `no`
- Description: Query command planning, action routing, and summary output contracts
- Workspace: `/tmp/nix-shell.KNu3Le/mutmut-cli-query-u7dwl8v7/repo`
- Command: `mutmut run`

## Scope

- Mutated paths: `polylogue/cli/query.py`, `polylogue/cli/query_plan.py`, `polylogue/cli/query_actions.py`, `polylogue/cli/query_output.py`
- Selected tests: `tests/unit/cli/test_query.py`, `tests/unit/cli/test_query_exec.py`, `tests/unit/cli/test_query_exec_laws.py`, `tests/unit/cli/test_query_fmt.py`, `tests/unit/cli/test_query_plan.py`

## Counts

| Status | Count |
| --- | ---: |
| Killed | 935 |
| Survived | 1007 |
| Timeout | 20 |
| Not checked | 0 |
| Suspicious | 0 |
| Skipped | 0 |

- Runtime: `90.02s`
- Exit code: `0`

## Dominant Survivors

| Function | Count |
| --- | ---: |
| `polylogue.cli.query.x__async_execute_query__mutmut_1` | 1 |
| `polylogue.cli.query.x__async_execute_query__mutmut_2` | 1 |
| `polylogue.cli.query.x__async_execute_query__mutmut_3` | 1 |
| `polylogue.cli.query.x__async_execute_query__mutmut_4` | 1 |
| `polylogue.cli.query.x__async_execute_query__mutmut_5` | 1 |
| `polylogue.cli.query.x__async_execute_query__mutmut_6` | 1 |
| `polylogue.cli.query.x__async_execute_query__mutmut_7` | 1 |
| `polylogue.cli.query.x__async_execute_query__mutmut_8` | 1 |
| `polylogue.cli.query.x__async_execute_query__mutmut_9` | 1 |
| `polylogue.cli.query.x__async_execute_query__mutmut_11` | 1 |

## Dominant Timeouts

| Function | Count |
| --- | ---: |
| `polylogue.cli.query_actions.x__apply_modifiers__mutmut_8` | 1 |
| `polylogue.cli.query_actions.x__apply_modifiers__mutmut_16` | 1 |
| `polylogue.cli.query_actions.x__apply_modifiers__mutmut_17` | 1 |
| `polylogue.cli.query_actions.x__apply_modifiers__mutmut_19` | 1 |
| `polylogue.cli.query_actions.x__apply_modifiers__mutmut_21` | 1 |
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

- `polylogue.cli.query.x__async_execute_query__mutmut_1`
- `polylogue.cli.query.x__async_execute_query__mutmut_2`
- `polylogue.cli.query.x__async_execute_query__mutmut_3`
- `polylogue.cli.query.x__async_execute_query__mutmut_4`
- `polylogue.cli.query.x__async_execute_query__mutmut_5`
- `polylogue.cli.query.x__async_execute_query__mutmut_6`
- `polylogue.cli.query.x__async_execute_query__mutmut_7`
- `polylogue.cli.query.x__async_execute_query__mutmut_8`
- `polylogue.cli.query.x__async_execute_query__mutmut_9`
- `polylogue.cli.query.x__async_execute_query__mutmut_11`
- `polylogue.cli.query.x__async_execute_query__mutmut_12`
- `polylogue.cli.query.x__async_execute_query__mutmut_13`
- `polylogue.cli.query.x__async_execute_query__mutmut_14`
- `polylogue.cli.query.x__async_execute_query__mutmut_15`
- `polylogue.cli.query.x__async_execute_query__mutmut_16`
- `polylogue.cli.query.x__async_execute_query__mutmut_17`
- `polylogue.cli.query.x__async_execute_query__mutmut_18`
- `polylogue.cli.query.x__async_execute_query__mutmut_19`
- `polylogue.cli.query.x__async_execute_query__mutmut_20`
- `polylogue.cli.query.x__async_execute_query__mutmut_23`
- `polylogue.cli.query.x__async_execute_query__mutmut_24`
- `polylogue.cli.query.x__async_execute_query__mutmut_25`
- `polylogue.cli.query.x__async_execute_query__mutmut_26`
- `polylogue.cli.query.x__async_execute_query__mutmut_27`
- `polylogue.cli.query.x__async_execute_query__mutmut_28`
- ... 982 more

## Timeout Keys

- `polylogue.cli.query_actions.x__apply_modifiers__mutmut_8`
- `polylogue.cli.query_actions.x__apply_modifiers__mutmut_16`
- `polylogue.cli.query_actions.x__apply_modifiers__mutmut_17`
- `polylogue.cli.query_actions.x__apply_modifiers__mutmut_19`
- `polylogue.cli.query_actions.x__apply_modifiers__mutmut_21`
- `polylogue.cli.query_actions.x__apply_modifiers__mutmut_67`
- `polylogue.cli.query_actions.x__apply_modifiers__mutmut_68`
- `polylogue.cli.query_actions.x__apply_modifiers__mutmut_69`
- `polylogue.cli.query_actions.x__apply_modifiers__mutmut_70`
- `polylogue.cli.query_actions.x__apply_modifiers__mutmut_71`
- `polylogue.cli.query_actions.x__apply_modifiers__mutmut_72`
- `polylogue.cli.query_actions.x__apply_modifiers__mutmut_73`
- `polylogue.cli.query_actions.x__delete_conversations__mutmut_69`
- `polylogue.cli.query_actions.x__delete_conversations__mutmut_72`
- `polylogue.cli.query_actions.x__delete_conversations__mutmut_73`
- `polylogue.cli.query_actions.x__delete_conversations__mutmut_81`
- `polylogue.cli.query_actions.x__delete_conversations__mutmut_82`
- `polylogue.cli.query_actions.x__delete_conversations__mutmut_83`
- `polylogue.cli.query_actions.x__delete_conversations__mutmut_84`
- `polylogue.cli.query_actions.x__delete_conversations__mutmut_110`

# Mutmut Campaign: `cli-run`

- Recorded on `2026-03-12T04:14:00.022530+00:00`
- Commit: `58264c2c47beaaa5522139c54700c20910833267`
- Worktree dirty: `no`
- Description: Run command execution, display, and watch contracts
- Workspace: `/tmp/nix-shell.x0yF2w/mutmut-cli-run-cxrkn0qh/repo`
- Command: `mutmut run`

## Scope

- Mutated paths: `polylogue/cli/commands/run.py`
- Selected tests: `tests/unit/cli/test_run.py`, `tests/unit/cli/test_run_int.py`, `tests/unit/cli/test_run_laws.py`

## Counts

| Status | Count |
| --- | ---: |
| Killed | 167 |
| Survived | 21 |
| Timeout | 87 |
| Not checked | 8 |
| Suspicious | 0 |
| Skipped | 0 |

- Runtime: `106.61s`
- Exit code: `0`

## Dominant Survivors

| Function | Count |
| --- | ---: |
| `on_progress` | 18 |
| `__init__` | 2 |
| `polylogue.cli.commands.run.x__display_result__mutmut_33` | 1 |

## Dominant Timeouts

| Function | Count |
| --- | ---: |
| `__init__` | 12 |
| `on_completed` | 6 |
| `polylogue.cli.commands.run.x__format_elapsed__mutmut_5` | 1 |
| `polylogue.cli.commands.run.x__execute_sync_once__mutmut_2` | 1 |
| `polylogue.cli.commands.run.x__execute_sync_once__mutmut_5` | 1 |
| `polylogue.cli.commands.run.x__execute_sync_once__mutmut_6` | 1 |
| `polylogue.cli.commands.run.x__execute_sync_once__mutmut_7` | 1 |
| `polylogue.cli.commands.run.x__execute_sync_once__mutmut_8` | 1 |
| `polylogue.cli.commands.run.x__execute_sync_once__mutmut_9` | 1 |
| `polylogue.cli.commands.run.x__execute_sync_once__mutmut_10` | 1 |

## Dominant Not-Checked Clusters

| Function | Count |
| --- | ---: |
| `on_progress` | 8 |

## Survivor Keys

- `polylogue.cli.commands.run.xǁ_PlainProgressObserverǁon_progress__mutmut_7`
- `polylogue.cli.commands.run.xǁ_PlainProgressObserverǁon_progress__mutmut_12`
- `polylogue.cli.commands.run.xǁ_PlainProgressObserverǁon_progress__mutmut_14`
- `polylogue.cli.commands.run.xǁ_PlainProgressObserverǁon_progress__mutmut_17`
- `polylogue.cli.commands.run.xǁ_PlainProgressObserverǁon_progress__mutmut_23`
- `polylogue.cli.commands.run.xǁ_PlainProgressObserverǁon_progress__mutmut_24`
- `polylogue.cli.commands.run.xǁ_PlainProgressObserverǁon_progress__mutmut_25`
- `polylogue.cli.commands.run.xǁ_PlainProgressObserverǁon_progress__mutmut_31`
- `polylogue.cli.commands.run.xǁ_PlainProgressObserverǁon_progress__mutmut_32`
- `polylogue.cli.commands.run.xǁ_PlainProgressObserverǁon_progress__mutmut_33`
- `polylogue.cli.commands.run.xǁ_PlainProgressObserverǁon_progress__mutmut_37`
- `polylogue.cli.commands.run.xǁ_PlainProgressObserverǁon_progress__mutmut_40`
- `polylogue.cli.commands.run.xǁ_PlainProgressObserverǁon_progress__mutmut_42`
- `polylogue.cli.commands.run.xǁ_PlainProgressObserverǁon_progress__mutmut_44`
- `polylogue.cli.commands.run.xǁ_PlainProgressObserverǁon_progress__mutmut_45`
- `polylogue.cli.commands.run.xǁ_PlainProgressObserverǁon_progress__mutmut_46`
- `polylogue.cli.commands.run.xǁ_PlainProgressObserverǁon_progress__mutmut_48`
- `polylogue.cli.commands.run.xǁ_PlainProgressObserverǁon_progress__mutmut_49`
- `polylogue.cli.commands.run.xǁ_RichProgressObserverǁ__init____mutmut_1`
- `polylogue.cli.commands.run.xǁ_RichProgressObserverǁ__init____mutmut_2`
- `polylogue.cli.commands.run.x__display_result__mutmut_33`

## Timeout Keys

- `polylogue.cli.commands.run.x__format_elapsed__mutmut_5`
- `polylogue.cli.commands.run.xǁ_PlainProgressObserverǁ__init____mutmut_1`
- `polylogue.cli.commands.run.xǁ_PlainProgressObserverǁ__init____mutmut_2`
- `polylogue.cli.commands.run.xǁ_PlainProgressObserverǁ__init____mutmut_3`
- `polylogue.cli.commands.run.xǁ_PlainProgressObserverǁ__init____mutmut_5`
- `polylogue.cli.commands.run.xǁ_PlainProgressObserverǁ__init____mutmut_6`
- `polylogue.cli.commands.run.xǁ_PlainProgressObserverǁ__init____mutmut_8`
- `polylogue.cli.commands.run.xǁ_PlainProgressObserverǁ__init____mutmut_9`
- `polylogue.cli.commands.run.xǁ_PlainProgressObserverǁ__init____mutmut_10`
- `polylogue.cli.commands.run.xǁ_PlainProgressObserverǁ__init____mutmut_11`
- `polylogue.cli.commands.run.xǁ_PlainProgressObserverǁ__init____mutmut_14`
- `polylogue.cli.commands.run.xǁ_PlainProgressObserverǁ__init____mutmut_16`
- `polylogue.cli.commands.run.xǁ_PlainProgressObserverǁ__init____mutmut_17`
- `polylogue.cli.commands.run.xǁ_PlainProgressObserverǁon_completed__mutmut_6`
- `polylogue.cli.commands.run.xǁ_PlainProgressObserverǁon_completed__mutmut_8`
- `polylogue.cli.commands.run.xǁ_PlainProgressObserverǁon_completed__mutmut_9`
- `polylogue.cli.commands.run.xǁ_PlainProgressObserverǁon_completed__mutmut_11`
- `polylogue.cli.commands.run.xǁ_PlainProgressObserverǁon_completed__mutmut_13`
- `polylogue.cli.commands.run.xǁ_PlainProgressObserverǁon_completed__mutmut_15`
- `polylogue.cli.commands.run.x__execute_sync_once__mutmut_2`
- `polylogue.cli.commands.run.x__execute_sync_once__mutmut_5`
- `polylogue.cli.commands.run.x__execute_sync_once__mutmut_6`
- `polylogue.cli.commands.run.x__execute_sync_once__mutmut_7`
- `polylogue.cli.commands.run.x__execute_sync_once__mutmut_8`
- `polylogue.cli.commands.run.x__execute_sync_once__mutmut_9`
- ... 62 more

## Not-Checked Keys

- `polylogue.cli.commands.run.xǁ_RichProgressObserverǁon_progress__mutmut_1`
- `polylogue.cli.commands.run.xǁ_RichProgressObserverǁon_progress__mutmut_2`
- `polylogue.cli.commands.run.xǁ_RichProgressObserverǁon_progress__mutmut_3`
- `polylogue.cli.commands.run.xǁ_RichProgressObserverǁon_progress__mutmut_4`
- `polylogue.cli.commands.run.xǁ_RichProgressObserverǁon_progress__mutmut_5`
- `polylogue.cli.commands.run.xǁ_RichProgressObserverǁon_progress__mutmut_6`
- `polylogue.cli.commands.run.xǁ_RichProgressObserverǁon_progress__mutmut_7`
- `polylogue.cli.commands.run.xǁ_RichProgressObserverǁon_progress__mutmut_8`

# Mutmut Campaign: `cli-run`

- Recorded on `2026-03-12T07:01:48.281647+00:00`
- Commit: `e07c4baebfe68af194fd423cd8fa0ecab515ca01`
- Worktree dirty: `no`
- Description: Run command execution, display, and watch contracts
- Workspace: `/tmp/nix-shell.7NA3G4/mutmut-cli-run-q38_uox4/repo`
- Command: `mutmut run`

## Scope

- Mutated paths: `polylogue/cli/commands/run.py`
- Selected tests: `tests/unit/cli/test_run.py`, `tests/unit/cli/test_run_int.py`, `tests/unit/cli/test_run_laws.py`

## Counts

| Status | Count |
| --- | ---: |
| Killed | 186 |
| Survived | 89 |
| Timeout | 0 |
| Not checked | 8 |
| Suspicious | 0 |
| Skipped | 0 |

- Runtime: `37.44s`
- Exit code: `0`

## Dominant Survivors

| Function | Count |
| --- | ---: |
| `on_progress` | 18 |
| `__init__` | 11 |
| `on_completed` | 6 |
| `polylogue.cli.commands.run.x__format_elapsed__mutmut_5` | 1 |
| `polylogue.cli.commands.run.x__execute_sync_once__mutmut_5` | 1 |
| `polylogue.cli.commands.run.x__execute_sync_once__mutmut_7` | 1 |
| `polylogue.cli.commands.run.x__execute_sync_once__mutmut_9` | 1 |
| `polylogue.cli.commands.run.x__execute_sync_once__mutmut_10` | 1 |
| `polylogue.cli.commands.run.x__execute_sync_once__mutmut_14` | 1 |
| `polylogue.cli.commands.run.x__execute_sync_once__mutmut_16` | 1 |

## Dominant Timeouts

| Function | Count |
| --- | ---: |
| _none_ | 0 |

## Dominant Not-Checked Clusters

| Function | Count |
| --- | ---: |
| `on_progress` | 8 |

## Survivor Keys

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
- ... 64 more

## Not-Checked Keys

- `polylogue.cli.commands.run.xǁ_RichProgressObserverǁon_progress__mutmut_1`
- `polylogue.cli.commands.run.xǁ_RichProgressObserverǁon_progress__mutmut_2`
- `polylogue.cli.commands.run.xǁ_RichProgressObserverǁon_progress__mutmut_3`
- `polylogue.cli.commands.run.xǁ_RichProgressObserverǁon_progress__mutmut_4`
- `polylogue.cli.commands.run.xǁ_RichProgressObserverǁon_progress__mutmut_5`
- `polylogue.cli.commands.run.xǁ_RichProgressObserverǁon_progress__mutmut_6`
- `polylogue.cli.commands.run.xǁ_RichProgressObserverǁon_progress__mutmut_7`
- `polylogue.cli.commands.run.xǁ_RichProgressObserverǁon_progress__mutmut_8`

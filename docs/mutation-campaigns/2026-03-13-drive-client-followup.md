# Mutmut Campaign: `drive-client`

- Recorded on `2026-03-13T01:02:52.544157+00:00`
- Commit: `033d5d6f3130a0a178be6c93b2a154f23d7c3ac0`
- Worktree dirty: `yes`
- Description: Drive auth, transport, JSON payload parsing, and ingest attachment contracts
- Workspace: `/tmp/nix-shell.ILedBE/mutmut-drive-client-mct0na3b/repo`
- Command: `mutmut run`

## Scope

- Mutated paths: `polylogue/sources/drive_client.py`
- Selected tests: `tests/unit/sources/test_drive_client_laws.py`, `tests/unit/sources/test_drive_utils.py`, `tests/unit/sources/test_drive_ops.py`

## Counts

| Status | Count |
| --- | ---: |
| Killed | 564 |
| Survived | 286 |
| Timeout | 0 |
| Not checked | 0 |
| Suspicious | 0 |
| Skipped | 0 |

- Runtime: `21.75s`
- Exit code: `0`

## Dominant Survivors

| Function | Count |
| --- | ---: |
| `_load_credentials` | 58 |
| `resolve_folder_id` | 25 |
| `iter_json_files` | 18 |
| `_load_cached_credentials` | 13 |
| `_run_manual_auth_flow` | 13 |
| `_refresh_credentials_if_needed` | 10 |
| `__init__` | 9 |
| `_call_with_retry` | 9 |
| `download_to_path` | 9 |
| `_service_handle` | 8 |

## Dominant Timeouts

| Function | Count |
| --- | ---: |
| _none_ | 0 |

## Dominant Not-Checked Clusters

| Function | Count |
| --- | ---: |
| _none_ | 0 |

## Survivor Keys

- `polylogue.sources.drive_client.x_default_credentials_path__mutmut_1`
- `polylogue.sources.drive_client.x_default_credentials_path__mutmut_7`
- `polylogue.sources.drive_client.x_default_credentials_path__mutmut_8`
- `polylogue.sources.drive_client.x_default_credentials_path__mutmut_14`
- `polylogue.sources.drive_client.x_default_token_path__mutmut_1`
- `polylogue.sources.drive_client.x_default_token_path__mutmut_7`
- `polylogue.sources.drive_client.x_default_token_path__mutmut_8`
- `polylogue.sources.drive_client.x_default_token_path__mutmut_14`
- `polylogue.sources.drive_client.x__import_module__mutmut_1`
- `polylogue.sources.drive_client.x__import_module__mutmut_3`
- `polylogue.sources.drive_client.x__import_module__mutmut_6`
- `polylogue.sources.drive_client.x__import_module__mutmut_7`
- `polylogue.sources.drive_client.x__import_module__mutmut_8`
- `polylogue.sources.drive_client.x__import_module__mutmut_9`
- `polylogue.sources.drive_client.x__import_module__mutmut_10`
- `polylogue.sources.drive_client.x__import_module__mutmut_11`
- `polylogue.sources.drive_client.x__parse_modified_time__mutmut_3`
- `polylogue.sources.drive_client.x__parse_modified_time__mutmut_4`
- `polylogue.sources.drive_client.x__parse_modified_time__mutmut_10`
- `polylogue.sources.drive_client.x__parse_modified_time__mutmut_11`
- `polylogue.sources.drive_client.x__looks_like_id__mutmut_3`
- `polylogue.sources.drive_client.x__looks_like_id__mutmut_9`
- `polylogue.sources.drive_client.x__resolve_credentials_path__mutmut_1`
- `polylogue.sources.drive_client.x__resolve_credentials_path__mutmut_7`
- `polylogue.sources.drive_client.x__resolve_credentials_path__mutmut_8`
- ... 261 more

## Source Worktree Status

- ` M polylogue/lib/filters.py`
- ` M polylogue/sources/drive_client.py`
- ` M tests/unit/core/test_filters.py`
- ` M tests/unit/core/test_filters_adv.py`
- ` M tests/unit/core/test_filters_props.py`
- ` M tests/unit/sources/test_drive_client_laws.py`
- ` M tests/unit/sources/test_drive_utils.py`

## Notes

- Targets the historical Drive not_checked cluster with focused tests.

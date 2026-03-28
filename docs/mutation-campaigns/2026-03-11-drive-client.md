# Mutmut Campaign: `drive-client`

- Recorded on `2026-03-11T17:59:08.012030+00:00`
- Commit: `7e7c310037f9f8cf89ba9c016d8eb4713d1b2f3d`
- Worktree dirty: `no`
- Description: Drive auth, transport, JSON payload parsing, and ingest attachment contracts
- Workspace: `/tmp/nix-shell.wZpdv1/mutmut-drive-client-izbqzmtq/repo`
- Command: `mutmut run`

## Scope

- Mutated paths: `polylogue/sources/drive_client.py`
- Selected tests: `tests/unit/sources/test_drive_client_laws.py`, `tests/unit/sources/test_drive_utils.py`, `tests/unit/sources/test_drive_resilience.py`, `tests/unit/sources/test_drive_ops.py`

## Counts

| Status | Count |
| --- | ---: |
| Killed | 553 |
| Survived | 327 |
| Timeout | 4 |
| Not checked | 0 |
| Suspicious | 0 |
| Skipped | 0 |

- Runtime: `31.81s`
- Exit code: `0`

## Dominant Survivors

| Function | Count |
| --- | ---: |
| `_load_credentials` | 101 |
| `iter_json_files` | 43 |
| `resolve_folder_id` | 30 |
| `get_metadata` | 16 |
| `_call_with_retry` | 13 |
| `download_to_path` | 10 |
| `_run_manual_auth_flow` | 9 |
| `_service_handle` | 8 |
| `_persist_token` | 7 |
| `__init__` | 6 |

## Dominant Timeouts

| Function | Count |
| --- | ---: |
| `_download_request` | 2 |
| `_call_with_retry` | 1 |
| `iter_json_files` | 1 |

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
- `polylogue.sources.drive_client.x__resolve_credentials_path__mutmut_2`
- `polylogue.sources.drive_client.x__resolve_credentials_path__mutmut_3`
- `polylogue.sources.drive_client.x__resolve_credentials_path__mutmut_7`
- ... 302 more

## Timeout Keys

- `polylogue.sources.drive_client.xǁDriveClientǁ_call_with_retry__mutmut_6`
- `polylogue.sources.drive_client.xǁDriveClientǁiter_json_files__mutmut_84`
- `polylogue.sources.drive_client.xǁDriveClientǁ_download_request__mutmut_14`
- `polylogue.sources.drive_client.xǁDriveClientǁ_download_request__mutmut_15`

## Notes

- Targets the historical Drive not_checked cluster with focused tests.

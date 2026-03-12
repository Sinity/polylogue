# Mutmut Campaign: `drive-client`

- Recorded on `2026-03-12T14:08:09.842595+00:00`
- Commit: `c1fd5ce60e8216a714acfc59072597ee40955a66`
- Worktree dirty: `yes`
- Description: Drive auth, transport, JSON payload parsing, and ingest attachment contracts
- Workspace: `/tmp/nix-shell.QSL7U8/mutmut-drive-client-p05js1in/repo`
- Command: `mutmut run`

## Scope

- Mutated paths: `polylogue/sources/drive_client.py`
- Selected tests: `tests/unit/sources/test_drive_client_laws.py`, `tests/unit/sources/test_drive_utils.py`, `tests/unit/sources/test_drive_ops.py`

## Counts

| Status | Count |
| --- | ---: |
| Killed | 584 |
| Survived | 296 |
| Timeout | 3 |
| Not checked | 0 |
| Suspicious | 0 |
| Skipped | 0 |

- Runtime: `31.54s`
- Exit code: `0`

## Dominant Survivors

| Function | Count |
| --- | ---: |
| `_load_credentials` | 75 |
| `iter_json_files` | 37 |
| `resolve_folder_id` | 33 |
| `get_metadata` | 20 |
| `__init__` | 9 |
| `_call_with_retry` | 9 |
| `_run_manual_auth_flow` | 9 |
| `download_to_path` | 9 |
| `_service_handle` | 8 |
| `_persist_token` | 6 |

## Dominant Timeouts

| Function | Count |
| --- | ---: |
| `_download_request` | 2 |
| `_call_with_retry` | 1 |

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
- `polylogue.sources.drive_client.x__resolve_credentials_path__mutmut_7`
- `polylogue.sources.drive_client.x__resolve_credentials_path__mutmut_8`
- `polylogue.sources.drive_client.x__resolve_credentials_path__mutmut_14`
- ... 271 more

## Timeout Keys

- `polylogue.sources.drive_client.xǁDriveClientǁ_call_with_retry__mutmut_6`
- `polylogue.sources.drive_client.xǁDriveClientǁ_download_request__mutmut_14`
- `polylogue.sources.drive_client.xǁDriveClientǁ_download_request__mutmut_15`

## Source Worktree Status

- ` M tests/unit/cli/test_query_exec_laws.py`
- ` M tests/unit/core/test_health_core.py`
- ` M tests/unit/sources/test_drive_utils.py`
- ` M tests/unit/sources/test_source_laws.py`
- ` M tests/unit/sources/test_unified_semantic_laws.py`
- ` M tests/unit/ui/test_rendering.py`

## Notes

- Targets the historical Drive not_checked cluster with focused tests.

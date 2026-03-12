# Mutmut Campaign: `drive-client`

- Recorded on `2026-03-13T00:45:26.367307+00:00`
- Commit: `e94d1f7270c8d43f07f2a08b03adcdfbfeec769a`
- Worktree dirty: `yes`
- Description: Drive auth, transport, JSON payload parsing, and ingest attachment contracts
- Workspace: `/tmp/nix-shell.ZLOrb8/mutmut-drive-client-z6f9bf9o/repo`
- Command: `mutmut run`

## Scope

- Mutated paths: `polylogue/sources/drive_client.py`
- Selected tests: `tests/unit/sources/test_drive_client_laws.py`, `tests/unit/sources/test_drive_utils.py`, `tests/unit/sources/test_drive_ops.py`

## Counts

| Status | Count |
| --- | ---: |
| Killed | 560 |
| Survived | 340 |
| Timeout | 0 |
| Not checked | 0 |
| Suspicious | 0 |
| Skipped | 0 |

- Runtime: `20.79s`
- Exit code: `0`

## Dominant Survivors

| Function | Count |
| --- | ---: |
| `_load_credentials` | 58 |
| `iter_json_files` | 39 |
| `resolve_folder_id` | 38 |
| `get_metadata` | 32 |
| `_load_cached_credentials` | 17 |
| `download_to_path` | 14 |
| `_run_manual_auth_flow` | 13 |
| `_refresh_credentials_if_needed` | 10 |
| `__init__` | 9 |
| `_call_with_retry` | 9 |

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
- ... 315 more

## Source Worktree Status

- ` M polylogue/lib/filters.py`
- ` M polylogue/sources/drive_client.py`
- ` M tests/unit/core/test_filters.py`
- ` M tests/unit/core/test_filters_adv.py`
- ` M tests/unit/core/test_filters_props.py`
- ` M tests/unit/sources/test_drive_client_laws.py`
- ` M tests/unit/ui/test_golden.py`

## Notes

- Targets the historical Drive not_checked cluster with focused tests.

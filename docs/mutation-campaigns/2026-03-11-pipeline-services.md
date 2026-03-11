# Mutmut Campaign: `pipeline-services`

- Recorded on `2026-03-11T07:46:50.470356+00:00`
- Commit: `d1e704d7a2ba9f3ab5bd2357487c6d5c967eddb5`
- Worktree dirty: `no`
- Description: Acquire/validate/parse planning and stage contracts
- Workspace: `/tmp/nix-shell.FgyBpx/mutmut-pipeline-services-3wi4xniq/repo`
- Command: `mutmut run`

## Scope

- Mutated paths: `polylogue/pipeline/services`
- Selected tests: `tests/unit/pipeline/test_services.py`, `tests/unit/pipeline/test_service_laws.py`

## Counts

| Status | Count |
| --- | ---: |
| Killed | 736 |
| Survived | 595 |
| Timeout | 84 |
| Not checked | 246 |
| Suspicious | 0 |
| Skipped | 0 |

- Runtime: `365.67s`
- Exit code: `0`

## Dominant Survivors

| Function | Count |
| --- | ---: |
| `evaluate_raw_records` | 164 |
| `build_plan` | 112 |
| `_process_raw_batch` | 81 |
| `validate_raw_ids` | 42 |
| `parse_from_raw` | 39 |
| `get_index_status` | 24 |
| `_persist_record` | 22 |
| `update_index` | 17 |
| `ingest_sources` | 17 |
| `_make_raw_record` | 14 |

## Dominant Timeouts

| Function | Count |
| --- | ---: |
| `visit_sources` | 47 |
| `_iter_raw_record_stream` | 23 |
| `_iter_source_raw_stream` | 5 |
| `acquire_sources` | 4 |
| `_process_raw_batch` | 3 |
| `__init__` | 2 |

## Dominant Not-Checked Clusters

| Function | Count |
| --- | ---: |
| `render_conversations` | 110 |
| `_iter_drive_raw_stream` | 35 |
| `store_records` | 32 |
| `scan_sources` | 20 |
| `rebuild_index` | 20 |
| `ensure_index_exists` | 15 |
| `__init__` | 6 |
| `record_failure` | 5 |
| `record_success` | 3 |

## Survivor Keys

- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁ_persist_record__mutmut_9`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁ_persist_record__mutmut_10`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁ_persist_record__mutmut_11`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁ_persist_record__mutmut_12`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁ_persist_record__mutmut_13`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁ_persist_record__mutmut_14`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁ_persist_record__mutmut_15`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁ_persist_record__mutmut_16`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁ_persist_record__mutmut_17`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁ_persist_record__mutmut_18`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁ_persist_record__mutmut_19`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁ_persist_record__mutmut_20`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁ_persist_record__mutmut_21`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁ_persist_record__mutmut_22`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁ_persist_record__mutmut_23`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁ_persist_record__mutmut_24`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁ_persist_record__mutmut_25`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁ_persist_record__mutmut_26`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁ_persist_record__mutmut_27`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁ_persist_record__mutmut_28`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁ_persist_record__mutmut_29`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁ_persist_record__mutmut_30`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁacquire_sources__mutmut_3`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁacquire_sources__mutmut_5`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁacquire_sources__mutmut_13`
- ... 570 more

## Timeout Keys

- `polylogue.pipeline.services.acquisition.xǁScanResultǁ__init____mutmut_1`
- `polylogue.pipeline.services.acquisition.xǁScanResultǁ__init____mutmut_5`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁvisit_sources__mutmut_1`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁvisit_sources__mutmut_2`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁvisit_sources__mutmut_3`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁvisit_sources__mutmut_8`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁvisit_sources__mutmut_9`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁvisit_sources__mutmut_12`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁvisit_sources__mutmut_13`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁvisit_sources__mutmut_14`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁvisit_sources__mutmut_16`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁvisit_sources__mutmut_17`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁvisit_sources__mutmut_18`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁvisit_sources__mutmut_19`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁvisit_sources__mutmut_22`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁvisit_sources__mutmut_23`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁvisit_sources__mutmut_24`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁvisit_sources__mutmut_25`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁvisit_sources__mutmut_27`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁvisit_sources__mutmut_28`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁvisit_sources__mutmut_29`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁvisit_sources__mutmut_30`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁvisit_sources__mutmut_34`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁvisit_sources__mutmut_35`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁvisit_sources__mutmut_36`
- ... 59 more

## Not-Checked Keys

- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁscan_sources__mutmut_1`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁscan_sources__mutmut_2`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁscan_sources__mutmut_3`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁscan_sources__mutmut_4`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁscan_sources__mutmut_5`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁscan_sources__mutmut_6`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁscan_sources__mutmut_7`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁscan_sources__mutmut_8`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁscan_sources__mutmut_9`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁscan_sources__mutmut_10`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁscan_sources__mutmut_11`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁscan_sources__mutmut_12`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁscan_sources__mutmut_13`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁscan_sources__mutmut_14`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁscan_sources__mutmut_15`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁscan_sources__mutmut_16`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁscan_sources__mutmut_17`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁscan_sources__mutmut_18`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁscan_sources__mutmut_19`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁscan_sources__mutmut_20`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁstore_records__mutmut_1`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁstore_records__mutmut_2`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁstore_records__mutmut_3`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁstore_records__mutmut_4`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁstore_records__mutmut_5`
- ... 221 more

## Notes

- Likely to need more helper-level laws to reduce timeout noise.

# Mutmut Campaign: `pipeline-services`

- Recorded on `2026-03-12T03:40:45.985215+00:00`
- Commit: `856caf495bab96724189df08115a5410192f2877`
- Worktree dirty: `no`
- Description: Acquire/validate/parse planning and stage contracts
- Workspace: `/tmp/nix-shell.kKBw8M/mutmut-pipeline-services-86nqrv_y/repo`
- Command: `mutmut run`

## Scope

- Mutated paths: `polylogue/pipeline/services`
- Selected tests: `tests/unit/pipeline/test_acquisition_service.py`, `tests/unit/pipeline/test_validation_service.py`, `tests/unit/pipeline/test_planning_service.py`, `tests/unit/pipeline/test_parsing_service.py`, `tests/unit/pipeline/test_render_service.py`, `tests/unit/pipeline/test_indexing.py`, `tests/unit/pipeline/test_ingest_state.py`, `tests/unit/pipeline/test_service_laws.py`

## Counts

| Status | Count |
| --- | ---: |
| Killed | 841 |
| Survived | 648 |
| Timeout | 84 |
| Not checked | 35 |
| Suspicious | 0 |
| Skipped | 0 |

- Runtime: `207.35s`
- Exit code: `0`

## Dominant Survivors

| Function | Count |
| --- | ---: |
| `evaluate_raw_records` | 164 |
| `build_plan` | 112 |
| `_process_raw_batch` | 81 |
| `render_conversations` | 45 |
| `validate_raw_ids` | 45 |
| `parse_from_raw` | 39 |
| `_persist_record` | 22 |
| `ingest_sources` | 17 |
| `_make_raw_record` | 14 |
| `__init__` | 14 |

## Dominant Timeouts

| Function | Count |
| --- | ---: |
| `visit_sources` | 42 |
| `_iter_raw_record_stream` | 23 |
| `acquire_sources` | 8 |
| `_iter_source_raw_stream` | 5 |
| `_process_raw_batch` | 3 |
| `render_conversations` | 2 |
| `__init__` | 1 |

## Dominant Not-Checked Clusters

| Function | Count |
| --- | ---: |
| `_iter_drive_raw_stream` | 35 |

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
- ... 623 more

## Timeout Keys

- `polylogue.pipeline.services.acquisition.xǁScanResultǁ__init____mutmut_4`
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
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁvisit_sources__mutmut_23`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁvisit_sources__mutmut_24`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁvisit_sources__mutmut_25`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁvisit_sources__mutmut_28`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁvisit_sources__mutmut_29`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁvisit_sources__mutmut_30`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁvisit_sources__mutmut_37`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁvisit_sources__mutmut_38`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁvisit_sources__mutmut_39`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁvisit_sources__mutmut_41`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁvisit_sources__mutmut_42`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁvisit_sources__mutmut_43`
- ... 59 more

## Not-Checked Keys

- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁ_iter_drive_raw_stream__mutmut_1`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁ_iter_drive_raw_stream__mutmut_2`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁ_iter_drive_raw_stream__mutmut_3`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁ_iter_drive_raw_stream__mutmut_4`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁ_iter_drive_raw_stream__mutmut_5`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁ_iter_drive_raw_stream__mutmut_6`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁ_iter_drive_raw_stream__mutmut_7`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁ_iter_drive_raw_stream__mutmut_8`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁ_iter_drive_raw_stream__mutmut_9`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁ_iter_drive_raw_stream__mutmut_10`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁ_iter_drive_raw_stream__mutmut_11`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁ_iter_drive_raw_stream__mutmut_12`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁ_iter_drive_raw_stream__mutmut_13`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁ_iter_drive_raw_stream__mutmut_14`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁ_iter_drive_raw_stream__mutmut_15`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁ_iter_drive_raw_stream__mutmut_16`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁ_iter_drive_raw_stream__mutmut_17`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁ_iter_drive_raw_stream__mutmut_18`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁ_iter_drive_raw_stream__mutmut_19`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁ_iter_drive_raw_stream__mutmut_20`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁ_iter_drive_raw_stream__mutmut_21`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁ_iter_drive_raw_stream__mutmut_22`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁ_iter_drive_raw_stream__mutmut_23`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁ_iter_drive_raw_stream__mutmut_24`
- `polylogue.pipeline.services.acquisition.xǁAcquisitionServiceǁ_iter_drive_raw_stream__mutmut_25`
- ... 10 more

## Notes

- Likely to need more helper-level laws to reduce timeout noise.

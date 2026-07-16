---
cluster: incremental-rebuild-equivalence
readiness: prepared-not-execution-grade
git_head: 21f78b4db2ba62ff44b5f16dfab96067bc249b4c
generated_by: census/dossier.py
---

# Incremental versus rebuild storage equivalence

> Evidence packet, not a deletion verdict or coverage gate.

## Responsibility

Rebuilding derived tiers from durable inputs yields the same public/logical archive facts and declared identities as incremental ingestion, without importing user overlay state into content identity.

## Readiness

`prepared-not-execution-grade`

- cluster prerequisite requires coordinator receipt: seeded-artifact-integrity
- upstream Bead contract requires merged-source coordinator receipt: polylogue-1xc.14.1
- no realized sensitivity artifact found for the cluster

## Baseline dependencies

- `seeded-artifact-integrity`: Requires a stable realized workload artifact and independent planted facts.
- `polylogue-1xc.14.1`: Replay one identical workload identity through incremental and rebuild routes; do not define another corpus.

## Authoritative routes

| Path | Exists | Resolved symbols | Missing symbols |
| --- | --- | --- | --- |
| `polylogue/storage/sqlite/archive_tiers/write.py` | yes | — | — |
| `polylogue/cli/commands/maintenance/_rebuild_index.py` | yes | `_rebuild_index_selection_plan`, `rebuild_index_command` | — |
| `devtools/index_fast_forward.py` | yes | `plan_index`, `validate_clone`, `fast_forward_clone` | — |
| `polylogue/storage/insights/session/rebuild.py` | yes | `rebuild_session_insights_sync`, `rebuild_session_insights_async` | — |

## Independent obligations

- incremental and source replay expose equal public/logical facts
- FTS and materialized insights are neither stale nor missing
- reordered independent artifacts converge
- user tags/corrections/annotations do not change ingest content identity
- durable migration, blob, lineage, and recovery obligations remain in their focused suites

## Proposed survivor tests

- tests/unit/storage/test_incremental_rebuild_equivalence.py::test_incremental_and_rebuild_public_facts_match
- tests/unit/storage/test_incremental_rebuild_equivalence.py::test_overlay_state_does_not_change_rebuild_identity

## Sensitivity witnesses

- temporary-production-mutation: Skip one materializer, leave FTS stale, reorder replay incorrectly, or include user overlay state in content identity; equivalence must fail.

Realized artifacts: `0`.

## Candidate scope

Tests: `tests/unit/storage/test_index_fast_forward_lifecycle.py`, `tests/unit/storage/test_session_insight_rebuild_progress.py`, `tests/unit/storage/test_convergence_stale_to_healthy.py`

Helpers: `tests/infra/semantic_facts.py`, `tests/property/test_write_path_state_machine.py`

Planned: `tests/unit/storage/test_incremental_rebuild_equivalence.py`

Avoid: `polylogue/storage/sqlite/migrations`, `tests/unit/storage/test_migrations.py`, `tests/unit/storage/test_blob_gc.py`, `tests/unit/storage/test_lineage_normalization.py`

## Deletion candidates requiring dominance proof

- narrow rebuild/refresh parity tests with no unique crash, migration, security, or diagnostic branch
- private row-order comparisons dominated by public fact equality
- duplicated sync/async examples after both routes enter one law

## Evidence inventory

- pytest receipts: `1`
- testmon available/matching tests: `True` / `1870`
- coverage contexts available: `False`
- coupling findings: `0`
- fixture inventory rows: `1`
- mutation artifacts: `0`

## Recent path history

- `f0c1b489b84cd04aac840315e7e55fa23eb97e39	2026-07-16	fix: restore archive contract verification (#2932) (#2932)`
- `d2573d438f9b041ad832a2a0a801c7ce7abda445	2026-07-16	fix(capture): recover replaced browser snapshots (#2930)`
- `36001d023b2cfe793cb19fdd7c42a87597356f48	2026-07-16	feat(sinex): wire durable publication convergence (#2925)`
- `b55f3fd9697083d44466613091604a21c7324ae6	2026-07-16	fix(lineage): compose sibling variants canonically (#2922)`
- `d6501ac4615efa30cb0e2413c97614a4bf44b253	2026-07-16	fix(storage): make raw replay batches component-aware (#2915)`
- `41cb11f8739afd303b77eafacdc92d3e88183469	2026-07-15	refactor(storage): consolidate _table_exists, fix drift found while verifying (#2912)`
- `5d99611f4aaca2eabbc8621a173692140ed165d3	2026-07-15	refactor(polylogue-dab): stop materializing run-projection cache rows (#2898)`
- `9461741faf14381e61c16c1f58b26fd7b6c13d50	2026-07-15	fix(storage): skip backup manifest requirement for additive-only migrations (#2905)`
- `866dab24d5e38bc111ac188629243ad530707551	2026-07-14	feat(insights): comparative judgment core for rxdo.9.11-.9.15 (#2889)`
- `c5b36d3b65fa1102adaf1ea8ed86d2440c75cf7c	2026-07-14	feat(insights): rxdo.9 measurement substrate primitives (#2888)`
- `d068d64821c6dc440013a28134e2f245fe7074b2	2026-07-14	refactor(architecture): sqlite leak sweep, staleness unify, control-center decomposition (#2900)`
- `89166362b9aee8c304b27a69f68ec1b74606f634	2026-07-14	feat(query-dsl,daemon,api): real production query evaluator + finding provenance (rxdo cluster) (#2899)`

## Permitted worker checks

```bash
devtools test tests/unit/storage/test_incremental_rebuild_equivalence.py
```

The coordinator must refresh this dossier after the upstream merge, after collecting
per-test coverage contexts, and after sensitivity execution.

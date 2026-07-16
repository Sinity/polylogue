---
cluster: convergence-restart-retry
readiness: prepared-not-execution-grade
git_head: 21f78b4db2ba62ff44b5f16dfab96067bc249b4c
generated_by: census/dossier.py
---

# Convergence debt across restart, retry, and quiescence

> Evidence packet, not a deletion verdict or coverage gate.

## Responsibility

Bounded convergence work that remains pending is durably recorded, survives restart, retries without duplication, and reaches the same final facts as uninterrupted execution.

## Readiness

`prepared-not-execution-grade`

- cluster prerequisite requires coordinator receipt: seeded-artifact-integrity
- upstream Bead contract requires merged-source coordinator receipt: polylogue-1xc.14.1
- no realized sensitivity artifact found for the cluster

## Baseline dependencies

- `seeded-artifact-integrity`: Use independent facts attached to the realized named workload artifact for final comparison.
- `polylogue-1xc.14.1`: Reuse realized active-growing and partial-convergence variants and the shared workload receipt.

## Authoritative routes

| Path | Exists | Resolved symbols | Missing symbols |
| --- | --- | --- | --- |
| `polylogue/daemon/convergence.py` | yes | `ConvergenceStage`, `DaemonConverger` | — |
| `polylogue/daemon/convergence_stages.py` | yes | — | — |
| `polylogue/sources/live/convergence_debt.py` | yes | `convergence_debt_from_states`, `convergence_debt_from_state` | — |
| `polylogue/sources/live/convergence_debt_retry.py` | yes | `convergence_debt_retry_at`, `same_pending_convergence_debt` | — |
| `polylogue/daemon/convergence_debt_status.py` | yes | — | — |

## Independent obligations

- false_means_pending records remaining work without reporting failure
- debt survives process restart
- session-scoped retry does not widen to irrelevant backlog
- retries are idempotent and do not duplicate material
- final durable/public facts equal uninterrupted execution
- status and alert projections agree with debt and quiescence

## Proposed survivor tests

- tests/unit/daemon/test_convergence_restart_law.py::test_pending_debt_survives_restart_and_reaches_quiescence
- tests/unit/daemon/test_convergence_restart_law.py::test_retry_is_scoped_idempotent_and_matches_uninterrupted

## Sensitivity witnesses

- temporary-production-mutation: Remove debt persistence, false_means_pending handling, session scoping, or retry execution; durable/public final-state law must fail.

Realized artifacts: `0`.

## Candidate scope

Tests: `tests/unit/daemon/test_daemon_convergence.py`, `tests/unit/daemon/test_convergence_stages.py`, `tests/unit/daemon/test_convergence_final_state.py`, `tests/unit/daemon/test_convergence_debt_alert.py`

Helpers: `tests/infra/frozen_clock.py`

Planned: `tests/infra/convergence_harness.py`, `tests/unit/daemon/test_convergence_restart_law.py`

Avoid: `tests/unit/daemon/test_http_server.py`, `tests/unit/daemon/test_maintenance_endpoints.py`, `polylogue/daemon/http.py`

## Deletion candidates requiring dominance proof

- internal repository/converger call-order tests whose only obligation is covered by the restart law
- sleep/polling tests replaced by frozen barriers
- duplicate status projections after durable/public agreement is proved

## Evidence inventory

- pytest receipts: `1`
- testmon available/matching tests: `True` / `238`
- coverage contexts available: `False`
- coupling findings: `6`
- fixture inventory rows: `1`
- mutation artifacts: `0`

## Recent path history

- `f0c1b489b84cd04aac840315e7e55fa23eb97e39	2026-07-16	fix: restore archive contract verification (#2932) (#2932)`
- `36001d023b2cfe793cb19fdd7c42a87597356f48	2026-07-16	feat(sinex): wire durable publication convergence (#2925)`
- `b6c78adfcd666358307daf64ac97e8d695a8b854	2026-07-16	feat(archive): expose exact-source freshness (#2924)`
- `d6501ac4615efa30cb0e2413c97614a4bf44b253	2026-07-16	fix(storage): make raw replay batches component-aware (#2915)`
- `41cb11f8739afd303b77eafacdc92d3e88183469	2026-07-15	refactor(storage): consolidate _table_exists, fix drift found while verifying (#2912)`
- `5d99611f4aaca2eabbc8621a173692140ed165d3	2026-07-15	refactor(polylogue-dab): stop materializing run-projection cache rows (#2898)`
- `d068d64821c6dc440013a28134e2f245fe7074b2	2026-07-14	refactor(architecture): sqlite leak sweep, staleness unify, control-center decomposition (#2900)`
- `89166362b9aee8c304b27a69f68ec1b74606f634	2026-07-14	feat(query-dsl,daemon,api): real production query evaluator + finding provenance (rxdo cluster) (#2899)`
- `beff1130b31f40eaa1c9e3581325326a5af9c6c5	2026-07-14	refactor(paths): extract sibling_index_db helper, sweep 17 sites (#2894)`
- `a952221cdcc4813ffcc4c9c18c4fd8981d5bbb2a	2026-07-13	feat(query): materialize watched query relations (#2826)`
- `3082c72f0c046d184e6c5088d31cf87faac548e6	2026-07-13	feat(cli): add hot daemon read routing (#2827)`
- `5ff5b0e0f83d67c900d161444455e1baa3bef492	2026-07-13	fix(daemon): record lifecycle heartbeats and termination forensics (#2802)`

## Permitted worker checks

```bash
devtools test tests/unit/daemon/test_convergence_restart_law.py
```

The coordinator must refresh this dossier after the upstream merge, after collecting
per-test coverage contexts, and after sensitivity execution.

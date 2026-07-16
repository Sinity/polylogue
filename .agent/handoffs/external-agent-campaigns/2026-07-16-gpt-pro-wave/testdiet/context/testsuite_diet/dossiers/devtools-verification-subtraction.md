---
cluster: devtools-verification-subtraction
readiness: prepared-not-execution-grade
git_head: 21f78b4db2ba62ff44b5f16dfab96067bc249b4c
generated_by: census/dossier.py
---

# Devtools verification subtraction without proof loops

> Evidence packet, not a deletion verdict or coverage gate.

## Responsibility

Retired conductor and self-referential verification machinery are removed while executable lanes, live inventory/docs checks, mutation/benchmark receipts, and branch-local dev-loop behavior remain.

## Readiness

`prepared-not-execution-grade`

- upstream Bead contract requires merged-source coordinator receipt: polylogue-b054.1.1.3
- upstream Bead contract requires merged-source coordinator receipt: polylogue-b054.1.1.4
- upstream Bead contract requires merged-source coordinator receipt: polylogue-b054.1.1.5
- no realized sensitivity artifact found for the cluster

## Baseline dependencies

- `polylogue-b054.1.1.3`: Preserve the complete seed/resource receipt and avoid its implementation files.
- `polylogue-b054.1.1.4`: Preserve the real production-mutation/testmon proof and avoid its implementation files.
- `polylogue-b054.1.1.5`: Preserve repeated isolated/xdist hang-witness evidence and avoid its implementation files.

## Authoritative routes

| Path | Exists | Resolved symbols | Missing symbols |
| --- | --- | --- | --- |
| `devtools/devloop_temporal.py` | yes | `operating_log_events`, `structured_devloop_events`, `build_report`, `main` | — |
| `devtools/command_catalog.py` | yes | `CommandSpec` | — |
| `devtools/verify_closure_matrix.py` | yes | `_validate`, `main` | — |
| `devtools/scenario_coverage.py` | yes | `build_runtime_scenario_coverage` | — |
| `devtools/verify_manifests.py` | yes | `check_coverage_gaps`, `check_campaign_coverage_catalog`, `check_test_coverage_domains`, `main` | — |

## Independent obligations

- retired temporal conductor has no live consumer
- branch-local dev-loop operations remain
- executable validation lanes propagate failures
- mutation and benchmark receipt freshness checks real artifacts
- live docs inventory detects an undocumented public surface
- authored mirrors cannot claim behavioral completeness

## Proposed survivor tests

- focused command-catalog dispatch tests
- live docs inventory mutation test
- executable validation-lane failure propagation test
- mutation/benchmark real-receipt freshness tests
- branch-local dev-loop focused tests

## Sensitivity witnesses

- consumer-and-behavior-differential: Deleting devloop_temporal leaves live routes green; deleting dev_loop or bypassing an executable lane/docs inventory must fail focused survivors.

Realized artifacts: `0`.

## Candidate scope

Tests: `tests/unit/devtools/test_devloop_temporal.py`, `tests/unit/devtools/test_verify_closure_matrix.py`, `tests/unit/devtools/test_scenario_coverage.py`, `tests/unit/devtools/test_verify_manifests.py`, `tests/unit/devtools/test_render_quality_reference.py`

Helpers: —

Planned: —

Avoid: `devtools/dev_loop.py`, `tests/unit/devtools/test_dev_loop.py`, `polylogue/mcp`, `polylogue/web`

## Deletion candidates requiring dominance proof

- devtools/devloop_temporal.py
- tests/unit/devtools/test_devloop_temporal.py
- workspace temporal-devloop CommandSpec
- verify_closure_matrix path/prose mirror
- scenario coverage completeness claims
- verify_manifests authored coverage-gap/domain/CI-substring checks

## Evidence inventory

- pytest receipts: `1`
- testmon available/matching tests: `True` / `155`
- coverage contexts available: `False`
- coupling findings: `1`
- fixture inventory rows: `0`
- mutation artifacts: `0`

## Recent path history

- `f0c1b489b84cd04aac840315e7e55fa23eb97e39	2026-07-16	fix: restore archive contract verification (#2932) (#2932)`
- `10a1212f4b17b9359494949dccb032456f26c4ef	2026-07-09	feat(devtools): accept bead: owners in coverage-manifest gap records (#2611)`
- `3b90e9e33eb1e66f50e019a822a18622bc4f57ca	2026-07-05	fix(daemon): align browser capture spool watching`
- `349898106783020294e52e30d3a2296021094940	2026-07-04	feat: integrate archive convergence and capture platform (#2534)`
- `c37cb0dc390c539e0a121993d1e970d7977c9587	2026-07-03	fix(devtools): keep dev-loop daemon source-only`
- `d40d00a312e5530a76e3e1ba29f3ddc37ce89d82	2026-07-02	feat: converge dogfood branch state (#2504)`
- `ce2f62ac65e6e7c29558a74f1dd10ba78d0999e2	2026-07-02	feat(read): add projection render specs (#2503)`
- `7120bde619a1ffaaef6a82afcba13de904adf80f	2026-07-02	fix(archive): report evidence and convergence honestly (#2502)`
- `c6c1aa659633d22600b5f206f3de4a3c21d589b1	2026-06-24	fix(devtools): align browser plan with Chromium smokes (#2374)`
- `373f9e3c34fc747ee83604f3e75bf6e43e983a1f	2026-06-24	fix(ops): restore runtime materialization trust (#2351)`
- `8ae4eaf8660a1f26fa39f8b3b94ae45bf269708c	2026-06-24	feat: integrate query-action archive diagnostics (#2350)`
- `6049c778615a12825eb6fd878652533521a9ef52	2026-06-23	fix: stabilize live Polylogue dogfooding loop (#2319)`

## Permitted worker checks

```bash
devtools test tests/unit/devtools/test_devloop_temporal.py
devtools test -k 'command_catalog or docs_coverage or validation_lane or mutation_catalog or benchmark_catalog'
```

The coordinator must refresh this dossier after the upstream merge, after collecting
per-test coverage contexts, and after sensitivity execution.

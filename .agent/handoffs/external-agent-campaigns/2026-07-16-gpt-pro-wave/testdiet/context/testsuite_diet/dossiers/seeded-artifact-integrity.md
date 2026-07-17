---
cluster: seeded-artifact-integrity
readiness: prepared-not-execution-grade
git_head: 21f78b4db2ba62ff44b5f16dfab96067bc249b4c
generated_by: census/dossier.py
---

# Seeded artifact integrity over realized workload canaries

> Evidence packet, not a deletion verdict or coverage gate.

## Responsibility

A realized named workload canary is built through provider generation and real ingest, cached under shared workload/profile/build/archive identity, published atomically only after archive and independently planted-fact validation, and cloned without mutating its immutable base.

## Readiness

`prepared-not-execution-grade`

- upstream Bead contract requires merged-source coordinator receipt: polylogue-1xc.14
- upstream Bead contract requires merged-source coordinator receipt: polylogue-b054.1.1.3
- upstream Bead contract requires merged-source coordinator receipt: polylogue-b054.1.1.4
- no realized sensitivity artifact found for the cluster

## Baseline dependencies

- `polylogue-1xc.14.1`: Treat bounded package/archive profiles, correlated variants, named tiers, C-03, privacy review, and workload/profile/build/archive identity as realized authority.
- `polylogue-1xc.14`: Reuse its workload phase/resource/cancellation/cleanup receipt contract; do not mint a Diet receipt vocabulary.
- `polylogue-b054.1.1.3`: Reuse complete seed identity, node outcomes, and physical resource accounting.
- `polylogue-b054.1.1.4`: Reuse the real affected-selection production-mutation mechanism for sensitivity.

## Authoritative routes

| Path | Exists | Resolved symbols | Missing symbols |
| --- | --- | --- | --- |
| `tests/conftest.py` | yes | `_managed_pytest_temp_root`, `_clone_archive_template`, `seeded_db` | — |
| `tests/infra/corpus_fixtures.py` | yes | `_seed_db`, `corpus_seeded_db` | — |
| `polylogue/scenarios/corpus.py` | yes | `CorpusSpec`, `build_default_corpus_specs` | — |
| `polylogue/schemas/synthetic/runtime.py` | yes | `_generate_from_schema` | — |
| `tests/infra/semantic_facts.py` | yes | `SessionFacts`, `ArchiveFacts` | — |

## Independent obligations

- cache identity extends the shared workload/profile/build/archive identity with schema, DDL, parser, and materializer inputs
- any ingest failure prevents publication
- independent canary facts and archive quick-check/counts/hashes agree
- publish is atomic across interruption and concurrent builders
- writable clones cannot mutate the immutable base
- resource and cleanup evidence uses the shared 1xc.14/b054 receipts

## Proposed survivor tests

- tests/unit/core/test_corpus_artifacts.py::test_artifact_identity_and_atomic_publication
- tests/unit/core/test_corpus_artifacts.py::test_artifact_rejects_ingest_failure_and_corruption
- tests/unit/core/test_corpus_artifacts.py::test_writable_clone_preserves_base

## Sensitivity witnesses

- real-testmon-production-mutation: Use the b054.1.1.4 reversible production-mutation mechanism to convert one provider ingest exception back into a warning; the ordinary affected gate must select and fail the publication test.
- fault-injection: Interrupt before directory rename and corrupt a completed DB/manifest; reuse must reject both.

Realized artifacts: `0`.

## Candidate scope

Tests: `tests/conftest.py`, `tests/infra/corpus_fixtures.py`, `tests/infra/semantic_facts.py`

Helpers: `tests/infra/state_machines.py`, `tests/infra/growth_budgets.py`, `tests/infra/scale_fixtures.py`, `tests/benchmarks/conftest.py`

Planned: `tests/infra/corpus_artifacts.py`, `tests/unit/core/test_corpus_artifacts.py`

Avoid: `polylogue/schemas/generation/workload_profiles.py`, `polylogue/schemas/generation/archive_workload_profile.py`, `polylogue/schemas/field_stats/distributions.py`, `polylogue/schemas/synthetic/runtime.py`, `tests/unit/core/test_schema_workload_profiles.py`

## Deletion candidates requiring dominance proof

- tests/conftest.py:seeded_repository
- tests/conftest.py:seeded_db_writable
- tests/infra/state_machines.py
- tests/infra/growth_budgets.py
- tests/infra/test_growth_budgets.py
- benchmark-only realistic seeder after all consumers migrate

## Evidence inventory

- pytest receipts: `0`
- testmon available/matching tests: `True` / `15913`
- coverage contexts available: `False`
- coupling findings: `0`
- fixture inventory rows: `3`
- mutation artifacts: `0`

## Recent path history

- `f0c1b489b84cd04aac840315e7e55fa23eb97e39	2026-07-16	fix: restore archive contract verification (#2932) (#2932)`
- `25b9ccdbfd7967b68378d06273d8e9c32f4f91c4	2026-06-19	feat(providers): report importer package completeness (#2180) (#2189)`
- `9d8caef4bdfd062292f6af22088ff3e42d958db6	2026-06-19	refactor(devtools): group benchmark commands (#2126) (#2138)`
- `4b41f1306185619fa2ccb876ee0eac3bd04a559d	2026-06-18	fix(devtools): make verify runs observable and bounded (#2110) (#2111)`
- `7135ed34fca4676e5aa4ce57b7e0cdf7e1c391a8	2026-06-18	feat(assertions): export user-tier assertions (#1883) (#2098)`
- `8647cebb9a4386333877be004f0724b55aaa4fcb	2026-06-16	fix(tests): stop default pytest basetemps from using shm (#1995)`
- `e1fc62dae0e11b60cd009f555944d6fea23bcb44	2026-06-13	refactor: rename content_blocks → blocks in runtime models and tests (#1791) (#1857)`
- `2c47c1c17c090c8c2cf5933274a415f08cce73fc	2026-06-09	test: stabilize devtools verify --all under xdist load (#1803)`
- `ac9cfeb0b2614140c940ad44b637c0689d939202	2026-06-09	refactor: current split-file archive as the sole storage architecture (#1787)`
- `e789ca5e24e5b6538911b090e5b19bc3068bdd34	2026-06-05	fix(tests): isolate concurrent pytest runs with per-run tmpfs basetemp (#1785)`
- `7f5dacd2bf1a140499fabda11549e1327ad32c24	2026-05-31	fix(storage): close leaked sqlite connections, unsuppress ResourceWarning (#1772)`
- `78d993873a1e97b5ef15ce0d621bd8bd1cb66579	2026-05-29	refactor(verifiability): delete ceremonial verification infrastructure (#1737)`

## Permitted worker checks

```bash
devtools test tests/unit/core/test_corpus_artifacts.py
devtools test -k seeded_db
```

The coordinator must refresh this dossier after the upstream merge, after collecting
per-test coverage contexts, and after sensitivity execution.

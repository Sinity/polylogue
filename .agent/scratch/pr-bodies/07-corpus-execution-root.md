## Summary

PR 7/9. Stacked on `feature/refactor/stack-06-scenario-substrate`.

Second half of the scenario-substrate unification wave. Routes synthetic harnesses through `CorpusSpec`, collapses execution layers into one shared root, and models runtime paths declaratively.

**Size note:** 57 commits, the largest PR in the series. Each commit is small and mechanically coherent. Natural split point: `refactor: move scenario execution into shared substrate` (~30 commits before, ~27 after).

## Problem

Even after PR 6 unified metadata and catalog authorship, several semantic roots remained split:

- Synthetic harness seeding was too low-dimensional — `count`, `messages_per_conversation`, `seed`, `style` only. Provider mix, topology, pathology, scale knobs were all separate ad hoc arguments. Parser schema tests, helper tests, large-archive specs, and test fixtures each authored their own synthetic input shape.
- Inferred schema corpora existed but were not projected into first-class scenarios, not persisted in the schema list, and not executed through the same path as authored corpora.
- Showcase execution, validation lane execution, benchmark campaign execution, and pytest execution each carried their own subprocess/cli/execution vocabulary. Shared execution specs didn't exist.
- Showcase exercises carried an `ExerciseScenario` intermediate layer introduced in PR 6, which was now an unnecessary wrapper.
- Runtime paths for archive ingest, publication, embeddings, source acquisition were not declared as runtime artifact paths — the coverage map couldn't reference them.
- Memory-budget lanes were hand-written command blocks rather than compiled from shared execution specs.
- Lane entries for validation lanes, synthetic benchmark lanes, and scale lanes had diverged into multiple shapes.
- Durable benchmark scenarios and mutation campaigns each had layered authored roots with no reason for the extra indirection.
- Message FTS runtime wasn't modeled as an explicit operation/loop, even though it participated in the artifact graph.

## Solution

- **`CorpusSpec` as single input**: route synthetic workflows, test fixtures, parser schema tests, helper tests, large-archive specs through one authoring surface. Project inferred schema corpora into `CorpusSpec`, persist them in the schema list, execute them directly through `cda52da4 refactor: execute inferred corpus specs directly`.
- **`ExecutionSpec`** unifies subprocess / pytest / CLI execution under one substrate. Canonicalize pytest execution specs (`f0ee361a`), centralize Polylogue CLI execution semantics (`96c0e8bb`), unify showcase execution roots (`5ea11068`), serialize showcase execution payloads (`b9293a55`), share execution specs across validation and benchmark scenarios (`0636f794`).
- **Showcase exercise model collapse** — `Exercise` now extends `ExecutableScenario` directly, dropping the intermediate `ExerciseScenario` layer (`13994f6b`).
- **Runtime path modeling** — archive ingest (`47517b0f`), publication (`035c98e9`), embeddings (`eb86d835`), source acquisition (`8b9be3e5`), message FTS runtime loop (`59481e80`) all declared as runtime artifact paths. Runtime registries route through the authored scenario catalog (`fe3fe138`).
- **Scenario sources self-project** into runtime targets (`00d24556`); authored executable scenarios and inferred corpus scenarios compile through the same catalog (`283b4bdf`, `4d06ff7e`, `6c776f6a`).
- **Memory budget lanes** now compile from shared executions (`4914bec3`).
- **Lane entry unification** — `ValidationLaneEntry` and synthetic-benchmark lane entry shapes collapsed (`9fb41402`); scale lanes folded into the shared validation catalog (`0e55704a`); shared lane config reused across scale lanes (`56ffeebc`).
- **Catalog collapse** — durable benchmark scenario layers collapsed (`f333289d`); mutation campaign layers collapsed (`e6daef87`); benchmark entries authored directly (`9e41e16f`). Multi-spec synthetic fixture writing shared (`a1de4a2c`).
- **Corpus request routing** — showcase seeding modeled with corpus requests (`9644ed84`); synthetic corpus selection routed through requests (`713b946c`); corpus sources threaded through synthetic harnesses (`22319b6d`).
- **Pipeline probe execution** typed explicitly (`8a7ef648`).

## Verification

- `pytest -q --ignore=tests/integration`
- `pytest -q tests/unit/scenarios tests/unit/showcase tests/unit/devtools/test_validation_lanes.py tests/unit/devtools/test_benchmark_campaign.py tests/unit/devtools/test_benchmark_campaigns.py tests/unit/devtools/test_campaign_report.py tests/unit/core/test_operator_inference.py tests/unit/core/test_operator_models.py tests/integration/test_schema_operator_workflow.py tests/unit/devtools/test_pipeline_probe.py tests/unit/devtools/test_quality_registry.py tests/unit/devtools/test_render_quality_reference.py tests/unit/devtools/test_scenario_projections.py`
- `devtools render-all --check`
- `ruff check polylogue tests devtools`

Commits on this branch: 57 (delta against `feature/refactor/stack-06-scenario-substrate`).

## Stack

Base: `feature/refactor/stack-06-scenario-substrate`. Next: `feature/refactor/stack-08-final-unification`.

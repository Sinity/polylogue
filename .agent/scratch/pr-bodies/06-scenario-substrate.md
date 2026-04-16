## Summary

PR 6/9. Stacked on `feature/refactor/stack-05-artifact-graph`.

First half of the scenario-substrate unification wave. Introduces shared metadata roots, authored operation specs, a typed QA request model, and scenario-first catalog extractions.

**Size note:** 54 commits. Each commit moves authorship up one layer mechanically. If this is too large to review as one unit, a natural split point is at `refactor: project benchmarks from catalog entries` (~35 commits before, ~19 after).

## Problem

Before this wave, verification, showcase, benchmark, and QA surfaces each maintained parallel authoring vocabularies:

- `Exercise` was a separate root model in `polylogue/showcase/exercise_models.py` — CLI-invocation-shaped and not derivable from anything higher.
- Benchmark campaigns had their own registry shape in `devtools/benchmark_campaign.py` (durable) and `devtools/benchmark_campaigns.py` (synthetic), each with a hand-curated `CAMPAIGN_REGISTRY` dict.
- Validation lanes had yet another registry shape in `devtools/validation_lane_base.py`.
- `QualityRegistry` aggregated these separate registries instead of describing one deeper model.
- Showcase exercise metadata, benchmark campaign metadata, and mutation campaign metadata each had their own `origin` / `artifact_targets` / `operation_targets` / `tags` serialization, so they drifted.
- The artifact graph introduced in PR 5 modeled layers and repair targets but didn't yet model operations as first-class metadata.
- QA invocations (`polylogue audit`) were driven by loose CLI argument bags, not typed plans.
- The scenario-projection inventory was not exposed as a devtools surface, so there was no way to ask "which scenarios cover this runtime target?"
- Runtime scenario coverage was not tracked; uncovered runtime paths were not reported.
- Benchmark scenarios couldn't carry artifact/operation targets, so the coverage map couldn't include them.

## Solution

- **`ScenarioMetadata`** in `polylogue/scenarios/metadata.py` becomes the one vocabulary for `origin`, `artifact_targets`, `operation_targets`, `tags` across showcase exercises, benchmark campaigns, mutation campaigns, validation lanes, and the quality registry. All coercion/serialization logic flows through `from_payload()`, `from_object()`, `to_payload()`.
- **`OperationSpec` + `RUNTIME_OPERATION_SPECS`** in `polylogue/operations/specs.py` — authored runtime operations (raw validation backlog planning, parse backlog planning, action-event materialization, action-event indexing, action-event health projection). The artifact graph now consumes this catalog instead of owning operation lists itself.
- **`ExerciseScenario`** authored root — `polylogue/showcase/scenario_models.py` becomes the new semantic source for generated CLI-backed showcase items; `Exercise` becomes a compiled execution artifact.
- **Benchmark compilation** — `BENCHMARK_SCENARIOS` + `compile_benchmark_scenarios(...)` replace the hand-built durable campaign registry. `SyntheticBenchmarkScenario` + `SYNTHETIC_BENCHMARK_REGISTRY` + `run_synthetic_benchmark_campaign(...)` replace the free-form synthetic `CAMPAIGN_REGISTRY` dispatch. Durable and synthetic campaign result artifacts now preserve full metadata.
- **QA request model** — `polylogue/showcase/qa_request.py` and adjacent files type QA invocations end-to-end: stage selection, capture/snapshot intent, finalization flow, invocation plans, CLI option derivation.
- **Catalog extractions** — separate benchmark, mutation, validation, and scenario-projection catalogs decoupled from `QualityRegistry`. Scenario-projection inventory exposed as a devtools surface with filter support. Quality registry now carries preserved synthetic benchmark scenario metadata as a distinct category.
- **Runtime coverage map** — runtime paths declared as artifact path targets; scenario coverage resolved through the artifact graph. Uncovered runtime coverage is now reported explicitly.
- **Session product repair semantics** — mapped into the artifact graph so scenario-bearing projections can reference them.

## Verification

- `pytest -q --ignore=tests/integration`
- `pytest -q tests/unit/scenarios/test_metadata.py tests/unit/showcase/test_scenario_models.py tests/unit/showcase/test_exercise_catalog.py tests/unit/showcase/test_report.py tests/unit/cli/test_qa.py tests/unit/devtools/test_benchmark_campaign.py tests/unit/devtools/test_benchmark_campaigns.py tests/unit/devtools/test_campaign_report.py tests/unit/devtools/test_quality_registry.py tests/unit/devtools/test_render_quality_reference.py tests/unit/devtools/test_scenario_projections.py tests/unit/operations/test_specs.py tests/unit/core/test_artifact_graph.py tests/unit/devtools/test_artifact_graph.py`
- `devtools render-quality-reference --check`
- `devtools render-all --check`
- `ruff check polylogue tests devtools`

Commits on this branch: 54 (delta against `feature/refactor/stack-05-artifact-graph`).

## Stack

Base: `feature/refactor/stack-05-artifact-graph`. Next: `feature/refactor/stack-07-corpus-execution-root`.

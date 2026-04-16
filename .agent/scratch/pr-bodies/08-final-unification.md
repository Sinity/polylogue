## Summary

PR 8/9. Stacked on `feature/refactor/stack-07-corpus-execution-root`.

Final unification layer and sqlite runtime collapse. Each refactor is a clean cap on top of the earlier substrate waves.

## Problem

After PRs 5–7 landed, a few semantic roots were still split across parallel locations:

- Three validation/assertion types in `polylogue/showcase/exercise_models.py`, `devtools/validation_lane_base.py`, and `devtools/benchmark_models.py` — each describing "what must be true after a workload" in its own shape.
- Validation lane families (contracts/live/hardening/etc.) were hand-authored as parallel lists, even though they shared stage shape and wiring.
- `CorpusSpec` was half generation-request and half observed-profile blob — inference, payloads, grouping, rendering, and scenario compilation all consumed both kinds of fields from the same flat object.
- Maintenance repair targets were defined three times: in storage repair constants, artifact node metadata, and execution/runtime mappings. Target name strings were hand-maintained in parallel.
- Product query surfaces and live/memory-budget CLI surfaces were hand-authored as separate command lists, even after PR 7 introduced shared execution specs.
- Sync and async SQLite paths carried their own schema ensure logic, DB-path resolution, connection caching, and connection profile semantics. `polylogue/paths.py` was a broad facade; `polylogue/cli/run_execution_workflow.py` existed as dead compatibility.

## Solution

- **`AssertionSpec`** (`polylogue/scenarios/assertions.py`) — one outcome vocabulary across showcase, validation lanes, and benchmarks. Replaces the three parallel types. All consumers (`exercise_models.py`, `catalog_loader.py`, `generators.py`, `showcase_runner_support.py`, `validation_lane_base.py`, `validation_lane_runtime.py`, `benchmark_models.py`) import the same root.
- **`ValidationLaneFamily.from_stages(...)`** (`devtools/validation_family_models.py` + `devtools/validation_catalog.py`) — composite lanes declare stages via the family compiler instead of open-coding each lane. (Contract and live catalogs still use hand-authored dictionaries with shared `pytest_lane()` / `devtools_lane()` helpers — they are consistent with the new substrate and can migrate to the family compiler as follow-up work; not runtime drift.)
- **`CorpusProfile`** extracted from `CorpusSpec` (`polylogue/scenarios/corpus.py`). Profile tokens, anchor kind, observed artifact counts, scope counts, observation windows, and inferred package/cluster profiles now live on `CorpusProfile`. `CorpusSpec` carries a single `profile: CorpusProfile` field. `polylogue/schemas/operator_inference.py` routes inferred profile metadata through this model.
- **`MaintenanceTargetSpec`** catalog (`polylogue/maintenance_targets.py`) — owns repair target identity, maintenance category, destructive flag, preview semantics, and doctor-operation mapping for every consumer. `storage/repair.py` and `artifact_graph.py` consume the catalog; no hardcoded target name literals remain in maintenance code.
- **`CliSurfaceFamily`** (`polylogue/scenarios/cli_surfaces.py`) compiles product query surfaces (`polylogue/scenarios/product_surfaces.py`) and operational/memory-budget surfaces (`polylogue/scenarios/operational_surfaces.py`) from shared builders. Showcase JSON-contract generation (`polylogue/showcase/generators.py`) and live/memory-budget lane compilation (`devtools/validation_lane_catalog_live.py`) consume the same substrate.
- **SQLite runtime collapse** — `polylogue/paths.py` becomes a single lazy facade exporting path accessors from `paths_roots.py`. `polylogue/storage/backends/connection.py` imports from `polylogue.paths` rather than re-deriving. Sync and async paths share schema runtime (`8c39814a refactor: unify sqlite schema and db path runtime`) and connection profile semantics (`1677f8ee refactor: share sqlite connection profiles`). `polylogue/cli/run_execution_workflow.py` is removed as dead.
- **Scenario catalog cached** for faster projection (`e6f0d046`); showcase generators authored directly from exercises (`ad064f8d`); synthetic benchmark runners derived from campaigns (`c4548b21`); scenario execution runtime centralized (`f11eaec9`); execution metadata moved into execution specs (`538c7384`); synthetic runner execution runtime unified (`22385d1d`).

## Verification

- `pytest -q --ignore=tests/integration`
- `pytest -q tests/unit/scenarios/test_assertions.py tests/unit/scenarios/test_corpus.py tests/unit/scenarios/test_operational_surfaces.py tests/unit/scenarios/test_product_surfaces.py tests/unit/showcase/test_exercise_catalog.py tests/unit/devtools/test_validation_lanes.py tests/unit/storage/test_backend.py tests/unit/storage/test_parse_tracking.py tests/unit/core/test_maintenance_targets.py tests/unit/showcase/test_generators.py tests/unit/devtools/test_artifact_graph.py tests/unit/core/test_operator_inference.py`
- `devtools render-all --check`
- `ruff check polylogue tests devtools`

Commits on this branch: 14 (delta against `feature/refactor/stack-07-corpus-execution-root`).

## Stack

Base: `feature/refactor/stack-07-corpus-execution-root`. Next: `feature/fix/stack-09-runtime-fixes`.

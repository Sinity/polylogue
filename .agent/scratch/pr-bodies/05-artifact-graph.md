## Summary

PR 5/9. Stacked on `feature/perf/stack-04-parse-hardening`.

The first shared semantic substrate: introduces an explicit runtime artifact graph and unifies storage/schema bootstrap across sync and async paths. Small, focused PR (4 commits) that lays the foundation for PRs 6–8.

## Problem

Before this substrate existed, runtime correctness was scattered across parallel helper modules that could drift silently:

- Action-event row state, FTS state, and repair accounting were defined three times — in `action_event_status.py`, `derived_status_products.py`, and `repair.py` — with subtly different rules. Extra stale `action_events_fts` rows could make readiness false while remaining undercounted or invisible in debt and retrieval-evidence projections.
- Raw ingest backlog semantics were split between SQL backlog predicates (`planning_backlog.py`) and scan-time in-memory raw-state decisions (`planning_runtime.py`), including force-reparse behavior. The two paths could diverge quietly.
- Runtime operations were not modeled as first-class metadata. Health, repair, and verification logic had to rediscover dependencies indirectly, which is why drift showed up in places like action-event read-model vs FTS health/debt accounting.
- Sync (`schema_upgrade.py`) and async (`async_sqlite_schema.py`) schema bootstrap carried weaker-on-async current-version extension coverage, different session-product backfill sequences, and different mismatch messaging. Correctness changes landed on one path and lagged on the other.

## Solution

- **`polylogue/storage/action_event_artifacts.py`** — single semantic model for action-event rows, FTS state, and repair accounting. The path projections (`action_event_status.py`, `derived_status_products.py`, `derived_status.py`, `embedding_stats_support.py`, `repair.py`) now compose against this model. Explicit concepts: missing conversations, stale rows, pending FTS rows, stale extra FTS rows, canonical repair count/detail, canonical row/FTS readiness.
- **`polylogue/storage/raw_ingest_artifacts.py`** — single semantic model for validation backlog eligibility, parse backlog eligibility, force-reparse eligibility, and quarantine classification, with shared SQL query specs for backlog selection. Both `planning_backlog.py` and `planning_runtime.py` now consume this model.
- **`polylogue/artifact_graph.py`** — first explicit runtime artifact/dependency graph. Current coverage: `raw_validation_state → validation_backlog → parse_backlog → parse_quarantine` and `tool_use_source_blocks → action_event_rows → action_event_fts → action_event_health`. Names artifact layers, dependency edges, repair targets, and health surfaces. `devtools/artifact_graph.py` provides the control-plane projection.
- **`polylogue/storage/backends/schema_bootstrap.py`** — shared bootstrap planning for sync and async paths: shared snapshot capture, current-version extension plan, schema mismatch message, and sync/async plan application helpers. Backend-specific SQL execution differences are preserved; drift between the two paths is not.

## Verification

- `pytest -q tests/unit/storage/test_action_event_artifacts.py tests/unit/storage/test_raw_ingest_artifacts.py tests/unit/core/test_artifact_graph.py tests/unit/devtools/test_artifact_graph.py tests/unit/storage/test_backend.py tests/unit/storage/test_parse_tracking.py tests/unit/storage/test_derived_status.py tests/unit/storage/test_repair.py tests/unit/storage/test_fts5.py tests/unit/cli/test_run.py tests/unit/pipeline/test_parsing_service.py tests/unit/pipeline/test_run_sources.py tests/integration/test_health.py`
- `ruff check polylogue/storage/action_event_artifacts.py polylogue/storage/raw_ingest_artifacts.py polylogue/artifact_graph.py polylogue/storage/backends/schema_bootstrap.py devtools/artifact_graph.py`
- `python -m devtools.artifact_graph --json`

Commits on this branch: 4 (delta against `feature/perf/stack-04-parse-hardening`).

## Stack

Base: `feature/perf/stack-04-parse-hardening`. Next: `feature/refactor/stack-06-scenario-substrate`.

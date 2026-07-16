# 15. polylogue-f2qv.5 — Version-gate provider-usage projection so stale rollups self-heal

Priority: **P2**  
Lane: **usage-cost-correctness**  
Readiness: **ready-now / convergence-path**

Depends on packet(s): polylogue-f2qv.1, polylogue-f2qv.2

## Why this is urgent / critical-path

Usage/cost fixes do not help existing archives if stale `session_model_usage` rows never rederive. Manual full index rebuilds are the wrong maintenance model for derived read-model staleness.

## Static diagnosis / likely mechanism

Root cause from bead and source anchors: provider usage rows are written during ingest (`polylogue/storage/sqlite/archive_tiers/write.py:618`) but are not in the session insight rebuild/convergence path. The status gate covers profiles/logical/work/threads, not provider usage. Archive debt can only recommend manual rebuild (`polylogue/operations/archive_debt.py:854+`, zero-token rows `:899+`).

## Implementation plan

Implementation shape:
1. Add a materializer version for provider-usage/session-model-usage projection.
2. Store that version in an existing insight materialization table or a new lightweight table keyed by session id.
3. Extend session-insight status/staleness detection to include provider usage.
4. Extend the periodic convergence loop to refresh stale provider usage rows for sessions whose source blocks carry usage.
5. Archive-debt should report stale provider usage as drainable by convergence, not only full rebuild.
6. Add a version bump fixture: seed old-version rows, run convergence/drain, assert new rows derived.

## Test plan

Tests:
- old-version `session_model_usage` row becomes stale.
- daemon/convergence drain refreshes it without `maintenance rebuild-index`.
- zero-token debt over sessions with source usage drains to zero after convergence.
- archive_debt message changes from manual-only to convergence-drainable.

## Verification command / proof

`devtools test tests/unit/storage/insights tests/unit/operations/test_archive_debt.py -k 'provider_usage or materializer_version or stale'`

## Pitfalls

Do not put this in a one-off repair command only. The acceptance condition is automagic convergence on daemon run.

## Files/functions to inspect or touch

- `polylogue/storage/sqlite/archive_tiers/write.py:618`
- `polylogue/storage/insights/session/status.py`
- `polylogue/storage/insights/session/rebuild.py`
- `polylogue/insights/registry.py`
- `polylogue/operations/archive_debt.py:854+`

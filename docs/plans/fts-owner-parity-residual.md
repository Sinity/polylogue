# FTS owner parity residual

## Scope

This change makes freshness evidence honest. It does not replace the existing
startup, identity-drift, orphan-audit, or debt-retry owners. Keeping those
owners avoids a second scheduler while preserving the behavior that already
repairs and observes the archive.

## Required proof before consolidating owners

Any replacement owner must demonstrate all of the following against the real
archive routes before an existing loop is removed:

1. It publishes an exact, generation-bound snapshot after an archive-wide
   verification and marks bounded work stale.
2. It detects trigger and count drift, including equal-count identity
   substitution, and schedules the appropriate repair.
3. It finds and repairs orphaned FTS rows without overlooking derived
   surfaces.
4. It records retryable debt, drains it after a transient failure, and keeps
   the debt observable until completion.
5. It preserves the current telemetry and reports catch-up progress.
6. It schedules stale work at the current cadence, observes repair budgets,
   and does not turn a bounded pass into a global scan.
7. It resumes safely after restart, including previously recorded debt and a
   stale freshness ledger.

## Delivery order

First add the replacement as a shadow observer with parity tests for the seven
requirements above. Then route one responsibility at a time while the old
owner remains available as a fallback. Remove an old loop only after the
replacement has passed its matching restart and debt tests and the telemetry
shows no unowned work.

The current schema/freshness work supplies the evidence contract that this
future consolidation must use; it deliberately does not claim scheduler
parity.

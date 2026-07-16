# 17. polylogue-1xc.12 — Add FTS drift gauges and metamorphic trigger-coherence tests with rowid-reuse protection

Priority: **P2**  
Lane: **search-integrity**  
Readiness: **spec-first then code**

## Why this is urgent / critical-path

FTS readiness as a boolean hides whether drift is 1 row or catastrophic. Count agreement is not enough when rowids can be reused.

## Static diagnosis / likely mechanism

Mechanism from bead: `messages_fts.rowid == blocks.rowid == docsize.id` is the keystone identity, but SQLite rowid reuse can make a ghost FTS row bind to a different block. Existing readiness checks are count/boolean-oriented; exact reconciliation must compare rowid plus block identity, and a ledger can itself drift.

## Implementation plan

Implementation shape:
1. Inspect `polylogue/storage/sqlite/archive_tiers/index.py` FTS trigger DDL and the current readiness checks in `polylogue/storage/archive_readiness.py` and `polylogue/daemon/fts_startup.py`.
2. Define drift classes: missing FTS row, ghost/excess row, mismatched identity, empty-text transition drift, ledger-vs-exact disagreement.
3. Add O(1) gauges from an `fts_freshness_state`/similar ledger where available; do not count full FTS tables on metrics scrape.
4. Add periodic exact reconciliation that samples or runs bounded full checks outside hot scrape paths.
5. Add `ops.db` drift sample history with retention.
6. Add Hypothesis/stateful tests that apply arbitrary insert/update/delete sequences through real triggers and assert exact convergence.
7. Add a rowid-reuse regression that fails unless identity is checked beyond count equality.

## Test plan

Tests:
- rowid reuse scenario creates equal counts but mismatched identity and the check catches it.
- Hypothesis op sequences over blocks/search text converge to zero drift.
- metrics endpoint emits gauges without table scans.
- exact reconciliation can repair or at least classify ledger drift.

## Verification command / proof

`devtools test tests/unit/storage/test_fts*.py tests/unit/daemon/test_*metrics*.py -k 'fts or rowid or drift or metamorphic'`

## Pitfalls

Contentless FTS cannot be checked by selecting text back. Use docsize/rowid plus block identity or a ledger. Keep heavy reconciliation out of the scrape path.

## Files/functions to inspect or touch

- `polylogue/storage/sqlite/archive_tiers/index.py`
- `polylogue/daemon/fts_startup.py`
- `polylogue/storage/archive_readiness.py`
- `polylogue/storage/sqlite/archive_tiers/archive.py:8112+`
- `ops.db metrics/telemetry modules`

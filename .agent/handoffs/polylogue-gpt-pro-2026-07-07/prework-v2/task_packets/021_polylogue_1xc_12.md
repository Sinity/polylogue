# 021. polylogue-1xc.12 — FTS drift gauges + metamorphic coherence tests; rowid-reuse requires block_id check

Priority/type/status: **P2 / bug / open**. Lane: **08-scale-performance-live**. Release: **B/G-storage-live-performance**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

FTS readiness is too boolean: operators need drift MAGNITUDE and tests need to prove trigger coherence under arbitrary block mutation. Keystone identity: messages_fts.rowid == blocks.rowid == docsize.id — and SQLite ROWID REUSE means a ghost FTS row can bind to a DIFFERENT block after delete+insert, so count agreement is insufficient: exact checks must join on rowid AND confirm block_id. Add: Prometheus gauges from the fts_freshness_state ledger (O(1), no COUNT on scrape), ops.db fts_drift_samples history with retention, metamorphic property tests (arbitrary insert/update/delete sequences through the REAL triggers => 0 missing / 0 excess, incl. empty-text transitions and repair convergence), and periodic exact reconciliation because the ledger itself can be the thing that drifted.

## Existing design note

Anchor files: polylogue/storage/sqlite/archive_tiers/index.py ~L307-318 (messages_fts_ai/ad/au triggers; threads_fts ~L449+), polylogue/daemon/fts_startup.py (startup repair), polylogue/storage/archive_readiness.py (current boolean readiness). Keystone identity: messages_fts.rowid == blocks.rowid == docsize.id, and SQLite ROWID REUSE means a ghost FTS row can silently attach to an unrelated reinserted block — drift checks must therefore compare block_id, not rowid existence. Deliverables: (1) drift GAUGES (counts of missing/ghost/mismatched rows) surfaced through readiness instead of a boolean; (2) metamorphic trigger-coherence tests: apply arbitrary insert/delete/update block mutations, assert FTS row set converges to exactly the search_text-bearing blocks (Hypothesis stateful fits; tests/infra strategies exist). Pitfall: FTS is contentless (content='') — you cannot SELECT text back; compare via docsize/rowid+blocks join.

## Acceptance criteria

Rowid-reuse test fails unless block_id equality checked; hypothesis op-sequence tests green; gauges scrape without table scans; ledger-vs-exact agreement periodically verified. Verify: property tests + metrics endpoint test.

## Static mechanism / likely defect

FTS coherence currently has multiple DDL/repair/readiness paths and risks rowid reuse drift unless checks prove rowid maps to the expected block_id/content.

## Source anchors to inspect first

- `polylogue/storage/sqlite/archive_tiers/index.py:296` — messages_fts table DDL lives in archive_tiers/index.py.
- `polylogue/storage/fts/sql.py:11` — messages_fts DDL copy also exists in storage/fts/sql.py.
- `polylogue/storage/fts/fts_lifecycle.py:292` — ensure_fts_triggers_sync owns runtime repair/recreation.
- `polylogue/storage/fts/fts_lifecycle.py:512` — Thread FTS rebuild logic is duplicated in lifecycle repair path.
- `polylogue/daemon/convergence_stages.py:988` — daemon convergence has another repair/readiness path for archive messages_fts.
- `polylogue/archive/query/spec.py:440` — SessionQuerySpec is the core read selection object.
- `polylogue/archive/query/spec.py:498` — from_params centralizes parameter lowering.
- `polylogue/cli/query_contracts.py:70` — QueryOutputSpec is current output/render contract seam.
- `polylogue/cli/archive_query.py:164` — CLI archive query execution remains a dense orchestration point.
- `polylogue/daemon/http.py:1734` — Daemon split-archive route already has from_params path for some cases.
- `polylogue/daemon/http.py:3296` — Another daemon path builds SessionQuerySpec from raw query params.
- `polylogue/mcp/query_contracts.py:54` — MCP query normalization has a distinct lowering path.

## Implementation plan

1. Add drift gauges that compare blocks↔messages_fts_docsize plus a block_id/content-hash sentinel, not only row counts.
2. Create metamorphic tests: insert/update/delete/reuse rowid patterns, then ensure FTS and structural reads converge.
3. Unify trigger DDL declarations or generate repair DDL from one source.
4. Expose drift counts in daemon status with repair command hints.

## Tests to add

- Rowid reuse fixture detects stale FTS row with wrong block_id.
- Trigger deletion/recreation is idempotent.
- Global drift repair updates only needed windows or states why full repair is required.

## Verification commands

- ``devtools test tests/unit/storage/test_fts*.py tests/unit/daemon/test_*metrics*.py -k 'fts or rowid or drift or metamorphic'``

## Pitfalls

- Do not optimize by hiding degraded/partial state; status must get more truthful, not prettier.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.

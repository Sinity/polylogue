# 081. polylogue-4ts.4 — Wrap lineage composition reads in a single read transaction

Priority/type/status: **P2 / bug / open**. Lane: **03-lineage-compaction-truth**. Release: **F-lineage-compaction**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

Composition uses multiple autocommit SELECTs; a concurrent parent re-ingest between reads yields a torn transcript. Hold one deferred read transaction across the recursion (pattern: fts_invariant_snapshot_sync). GH issue thread (body + comments) is input, not authority; this bead's scope statement wins where they conflict.

## Existing design note

Code-confirmed (gh#2476): get_messages / read_archive_session_envelope / _composed_db_signatures compose via multiple autocommit SELECTs (edge read -> recursive parent read -> own read); a parent re-ingest between reads yields a torn transcript. Fix: hold one deferred read transaction across the whole composition recursion — pattern to copy: fts_invariant_snapshot_sync. Apply to BOTH sync and async paths (twin-path trap, see bd memories). Test: interleave a parent full-replace between edge-read and parent-read via a hook/monkeypatch; assert composed transcript is either old-consistent or new-consistent, never mixed.

## Acceptance criteria

1. Both the sync path (read_archive_session_envelope, _composed_db_signatures) and the async path (get_messages, plus batch/paginated composition) hold ONE deferred read transaction across the full inheritance recursion (edge read -> recursive parent read -> own read), following the fts_invariant_snapshot_sync pattern. 2. A regression test interleaves a parent full-replace (DELETE + re-INSERT) between the edge-read and the parent-read via hook/monkeypatch and asserts the composed transcript is wholly old-consistent or wholly new-consistent, never torn — asserted on BOTH the sync and async paths (the twin-path trap is an explicit checkable item, not incidental). Verify: `devtools test tests/unit/storage/` selection covering the composition paths passes; the interleaving test fails on current main if the snapshot is missing and passes after the fix.

## Static mechanism / likely defect

Likely mechanism from bead title: lineage composition performs multiple separate reads/connection steps, so concurrent ingest/refresh can produce an impossible graph. This is especially dangerous for branch/shared-prefix accounting and compaction context.

## Source anchors to inspect first

- `polylogue/archive/session/threads.py` — Session/thread lineage read and composition model.
- `polylogue/insights/topology.py` — Topology/lineage derived insight code.
- `polylogue/daemon/lineage_startup.py` — Daemon lineage startup/convergence path.
- `polylogue/archive/coverage.py` — Completeness/truncation cues live here.
- `polylogue/insights/postmortem.py` — Compaction/continuation postmortem evidence is mined here.
- `polylogue/core/refs.py` — Existing ref model should be extended, not bypassed.
- `polylogue/storage/sqlite/archive_tiers/user_write.py:901` — Findings should reuse assertion/candidate lifecycle.
- `polylogue/surfaces/payloads.py:747` — Surface payloads currently expose branch/message refs; expand consistently.
- `polylogue/mcp/payloads.py:377` — MCP message payloads carry variant/branch metadata today.

## Implementation plan

1. Implementation shape:
2. 1. Locate lineage composition entrypoint and list every DB read it performs.
3. 2. Ensure the public composition call opens one read connection and begins a read transaction/snapshot before the first query.
4. 3. Pass that connection through lower helpers instead of helpers reopening connections.
5. 4. If both source and index tiers are needed, document consistency model; prefer a single tier/projection for composition or record cross-tier snapshot caveat.
6. 5. Add optional `composition_read_id` or `snapshot_started_at_ms` in debug output for diagnostics.

## Tests to add

- two-connection fixture: begin composition read, mutate lineage/messages on another connection, finish composition; result reflects one consistent before/after state, not mixed.
- helper tests assert no lower function opens a new connection when a connection is supplied.
- existing lineage composition tests still pass.

## Verification commands

- ``devtools test tests/unit/lineage tests/unit/storage -k 'composition or transaction or snapshot or lineage'``

## Pitfalls

- Do not broaden scope beyond the bead acceptance criteria; make a failing test first, then patch the smallest shared seam.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.

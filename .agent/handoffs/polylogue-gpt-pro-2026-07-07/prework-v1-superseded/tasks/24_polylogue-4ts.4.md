# 24. polylogue-4ts.4 — Read lineage composition from one transaction/snapshot

Priority: **P2**  
Lane: **lineage-truth**  
Readiness: **needs-source-confirmation then storage patch**

## Why this is urgent / critical-path

Composed session reads that stitch sessions, edges, messages, compactions, and shared prefixes must not mix rows from different write moments.

## Static diagnosis / likely mechanism

Likely mechanism from bead title: lineage composition performs multiple separate reads/connection steps, so concurrent ingest/refresh can produce an impossible graph. This is especially dangerous for branch/shared-prefix accounting and compaction context.

## Implementation plan

Implementation shape:
1. Locate lineage composition entrypoint and list every DB read it performs.
2. Ensure the public composition call opens one read connection and begins a read transaction/snapshot before the first query.
3. Pass that connection through lower helpers instead of helpers reopening connections.
4. If both source and index tiers are needed, document consistency model; prefer a single tier/projection for composition or record cross-tier snapshot caveat.
5. Add optional `composition_read_id` or `snapshot_started_at_ms` in debug output for diagnostics.

## Test plan

Tests:
- two-connection fixture: begin composition read, mutate lineage/messages on another connection, finish composition; result reflects one consistent before/after state, not mixed.
- helper tests assert no lower function opens a new connection when a connection is supplied.
- existing lineage composition tests still pass.

## Verification command / proof

`devtools test tests/unit/lineage tests/unit/storage -k 'composition or transaction or snapshot or lineage'`

## Pitfalls

Do not serialize writers globally to hide this. The read path needs a coherent snapshot; normal ingest concurrency should continue.

## Files/functions to inspect or touch

- `polylogue/lineage*`
- `polylogue/storage/sqlite/archive_tiers/* lineage/read helpers`
- `composition renderers`

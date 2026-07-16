# 28. polylogue-9e5.4 — Static get-modify-put race-window audit of shared SQLite writer paths

Priority: **P2**  
Lane: **storage-correctness**  
Readiness: **ready-now / audit-artifact**

## Why this is urgent / critical-path

Some correctness failures require two actors. Before filing random race bugs, classify which read-modify-write paths are actually split across transactions and which are safe.

## Static diagnosis / likely mechanism

Bead notes give the method: static sweep for SELECT-then-UPDATE/INSERT, manual upsert emulation, status transitions, shared writer APIs, and connection boundaries. Named candidates include write effects, blob GC, daemon cursors, embeddings, FTS readiness, MCP mutation handlers, and CLI ops writers.

## Implementation plan

Implementation shape:
1. `rg` for read-then-write patterns and connection open boundaries.
2. For each candidate, record file:function, invariant, connection/transaction boundary, interleaving, expected consequence, and verdict: safe-by-single-transaction, safe-by-unique/upsert, needs-harness, or confirmed-bug.
3. Only create implementation bug beads for confirmed windows with concrete two-connection repro sketches.
4. Document refuted windows as safe with reason so agents do not re-triage them.

## Test plan

No broad product tests unless a confirmed race is found. For top 1-2 confirmed windows, add a minimal two-connection repro test in the follow-up bug, not this audit packet.

## Verification command / proof

Review committed `race-window-table.md/json`. Optional: `devtools test -k <new_race_test>` only for confirmed follow-up.

## Pitfalls

Do not file “maybe race” beads. A race bug needs a concrete interleaving and observable lost/stale effect.

## Files/functions to inspect or touch

- `polylogue/archive/write_effects.py`
- `polylogue/storage/blob_gc.py`
- `polylogue/daemon/cursor*`
- `polylogue/storage/embeddings/*`
- `polylogue/storage/archive_readiness.py`
- `polylogue/mcp/server_mutation_tools.py`
- `CLI ops writers`

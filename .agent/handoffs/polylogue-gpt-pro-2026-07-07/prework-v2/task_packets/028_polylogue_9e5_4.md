# 028. polylogue-9e5.4 — Get->modify->put race audit across daemon/CLI/MCP writers

Priority/type/status: **P2 / task / open**. Lane: **06-agent-context-coordination**. Release: **D-agent-context-coordination**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

Sweep multi-step read-then-write sequences on separate connections: blob leases, ingest_cursor updates, embedding_status transitions, fts_freshness_state. Three writer surfaces (daemon, CLI, MCP mutation role) share the same SQLite files. This technique found 16 bugs in the sibling project. Output: confirmed race windows as bug beads with interleaving repro sketches.

## Existing design note

Static get->modify->put race audit across the shared-SQLite writers (promoted from the 2026-07-04 notes sidecar; static sweep first, not tests). Trace the named read-then-write sequences — blob leases (polylogue/archive/write_effects.py, storage/blob_gc.py), ingest_cursor updates (daemon/cursor* stores), embedding_status transitions (storage/embeddings/*), and fts_freshness_state/readiness helpers — plus MCP-mutation-role and CLI ops writers that share the same DB files. For each candidate record: file:function, connection boundary, transaction boundary, the invariant, the possible two-actor interleaving, the expected lost/stale effect, and a classification (safe-by-single-transaction, safe-by-unique/upsert, needs-harness, or bug). Pitfall: some sequences are already single-transaction (commit_archive_write_effects) — do not file those as bugs; note them as safe-with-reason so future agents don't re-triage.

## Acceptance criteria

1. A committed race-window table (sequence, writers, one-txn vs split, invariant, verdict) covers the four named sequences plus the shared writer surfaces; refuted windows are documented safe with the reason. 2. Each CONFIRMED window (concrete two-actor interleaving + reproducible consequence) is filed as a separate bug bead with a minimal two-connection repro sketch naming the exact table rows — implementation left to the follow-up bead. 3. No product-code mutation in this bead; an optional focused proof harness runs only if a real bug bead is created. Verify: artifact review; where a bug bead is created, `devtools test -k <race_test>` exercises the top 1-2 suspected windows (no broad tests for the audit itself).

## Static mechanism / likely defect

Bead notes give the method: static sweep for SELECT-then-UPDATE/INSERT, manual upsert emulation, status transitions, shared writer APIs, and connection boundaries. Named candidates include write effects, blob GC, daemon cursors, embeddings, FTS readiness, MCP mutation handlers, and CLI ops writers.

## Source anchors to inspect first

- `polylogue/insights/audit.py:173` — build_insight_rigor_audit_report is the audit entry point.
- `polylogue/insights/audit.py:194` — Current code iterates list_rigor_contracts, not the product registry.
- `polylogue/insights/audit.py:216` — Registry lookup is secondary and skipped for products without contracts.
- `polylogue/insights/rigor.py:85` — _RIGOR_MATRIX declares only a subset of registered products.
- `polylogue/insights/registry.py:294` — INSIGHT_REGISTRY is the universe the audit should iterate.
- `polylogue/archive/actions/followup.py` — Action/followup classification is a real structural analytics input.
- `polylogue/archive/actions/fields.py` — Action fields determine what can be measured without prose heuristics.
- `polylogue/insights/registry.py:294` — Insight registry should become measure/product registry input.
- `scripts/agent_forensics.py` — Existing forensics script is a proof artifact and candidate product surface.

## Implementation plan

1. Implementation shape:
2. 1. `rg` for read-then-write patterns and connection open boundaries.
3. 2. For each candidate, record file:function, invariant, connection/transaction boundary, interleaving, expected consequence, and verdict: safe-by-single-transaction, safe-by-unique/upsert, needs-harness, or confirmed-bug.
4. 3. Only create implementation bug beads for confirmed windows with concrete two-connection repro sketches.
5. 4. Document refuted windows as safe with reason so agents do not re-triage them.

## Tests to add

- No broad product tests unless a confirmed race is found. For top 1-2 confirmed windows, add a minimal two-connection repro test in the follow-up bug, not this audit packet.

## Verification commands

- `Review committed `race-window-table.md/json`. Optional: `devtools test -k <new_race_test>` only for confirmed follow-up.`

## Pitfalls

- Do not add direct context injection or trusted memory writes outside the scheduler/candidate policy.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.

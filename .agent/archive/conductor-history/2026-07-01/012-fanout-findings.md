---
created: "2026-06-28"
purpose: "Synthesis of 20-agent research/audit fan-out over polylogue (post lineage/cost/attachment work)"
status: active
project: polylogue
---

# 20-agent fan-out findings — prioritized

## DEPLOY-BLOCKERS (fix before re-ingest / live daemon)

### H1 — parent re-ingest silently breaks every child's composition  ★ top
`session_links.branch_point_message_id REFERENCES messages(message_id) ON DELETE SET NULL`
(index.py:324). Full-replace re-ingest of a parent does `DELETE FROM messages WHERE
session_id=?` (write.py:1532) → the cascade NULLs `branch_point_message_id` on every
child edge. Composition filters `branch_point_message_id IS NOT NULL`, so the child
reverts to **tail-only (inherited prefix vanishes)** — permanently, because deferred
re-extraction is gated on `inheritance IS NULL` and the edge is already
`prefix-sharing`. Bites continuously in production (Codex/Claude parents grow → re-ingest).
**Fix:** message_id is deterministic across re-ingest, so drop the FK action — make
`branch_point_message_id TEXT` (plain, no FK). It stays valid when the parent is
re-created with the same ids. (Schema bump v13→v14.)

### Lineage-2 — web constructs not sliced to the tail → FK crash
`_write_web_constructs(conn, session, …)` iterates `session.messages` (full list) while
messages/blocks were sliced to the tail (write.py:1551). A web-construct on an inherited
prefix message → INSERT against a non-existent message_id/block_id → FK violation aborts
that session's ingest. Low trigger (Codex/Claude forks rarely have web constructs) but a
hard crash. **Fix:** pass the sliced `messages` to `_write_web_constructs`.

### Lineage-3 — acompact subagent auto-compactions get the WRONG parent (~39/187)
`agent-acompact-*` also fires when a **Task subagent** compacts its own context. ~39 of
187 have <90% overlap with the main session; 9 have 0% overlap (e.g. `1796b263` subagent
"implementing a 12-commit testing overhaul"). The parser assigns `parent = main SID`
anyway → composition prepends the main session's transcript onto a subagent that never
saw it (wrong content) and dedup saves nothing. 148/187 (genuine main-session compactions)
are correct. **Fix:** distinguish main-session compaction from subagent compaction (test
prefix content/UUID membership against the main session, or detect a fresh task-prompt
head) before assigning the main session as parent.

### Lineage-4 — transcript reads that bypass composition show truncated forks
`get_messages_paginated` (CLI `messages.py:60`), `get_messages_batch`, and `iter_messages`
(streaming/export/repository) query `messages` directly → return **tail-only** for a
prefix-sharing child. CLI pagination + exports of a fork show a truncated transcript.
(`get_messages` and `read_archive_session_envelope` compose correctly.) **Fix:** route
these through composition or add the prefix-sharing edge handling.

### Lineage-5 — child cost rollup still counts the inherited prefix
`_write_session_events` gets the full `session.session_events` (write.py:445), so the
child's `session_model_usage` rollup includes inherited-prefix tokens; storage is
deduped but cost is not. `_reextract_prefix_tail_db` refreshes counts but not usage
rollups. Undercuts the cost-correctness goal. **Fix:** slice usage events to the tail /
recompute rollups after extraction.

## HIGH (correctness, not strictly deploy-gating)

- **Concurrency-3:** lineage composition is multiple non-atomic autocommit SELECTs; a
  concurrent parent re-ingest between the edge read and the parent read yields a **torn
  transcript** (over-long/truncated). Read-only, self-healing. Fix: wrap composition in
  one deferred read txn (pattern exists: `fts_invariant_snapshot_sync`). Don't run
  queries during the re-ingest.
- **MCP-1/2:** `facets`, `aggregate_sessions`, `correlate_sessions` silently cap scoped
  aggregates at the page limit (10 / 1000) — wrong totals. Pre-existing bug class. Fix:
  `replace(spec, limit=None)`.
- **MCP-5:** `get_session_tree` per-node `message_count` double-counts the inherited
  prefix (summary-only, no wire blowup). Add a logical/dedup mode.
- **Signature M5/M6:** parent(DB) vs child(parsed) signature mismatch on variant_index>0
  and scalar tool_input canonicalization → under-dedup (safe direction, no loss) but
  reduces effectiveness. Align variant filtering + `_canonical_json` of scalars.
- **Parser-1:** ChatGPT image/asset-only nodes silently dropped (`if not text: continue`
  before blocks built) — routine loss for image users. Move skip after block build.
- **Parser-2:** Antigravity brain-artifact non-UTF-8 body → UnicodeDecodeError (not OSError)
  → session dropped. Broaden except / errors="replace".

## MEDIUM (perf — old pre-dedup re-ingest hit these 16K×; current baseline ~2.4K)

- **Perf-1:** `_composed_db_signatures` recomputes parent composition (full SHA over all
  block text, recursive) per fork-child, unmemoized → super-linear on wide/deep lineages,
  inside the ingest txn. Memoize per ingest, or persist per-message signatures.
- **Perf-2:** `idx_messages_session_sortkey` referenced in docstring but **does not exist**
  → every keyset/paginated message read does a temp B-tree sort (claimed-linear is O(M²)).
  Add the index.
- **Perf-3:** `delete_session_rows_sql` scans the whole `messages_fts_docsize` per session
  write → per-session full-docsize scans during re-ingest. Use correlated EXISTS.
- **Perf-4:** no global `sessions(sort_key_ms DESC)` index → bare `list_sessions` full-scans
  + temp sort.

## LOW / cleanup

- Dead code: `persist_session_commits` + `session_commit_edge_to_row` (no-op stubs),
  `ArchiveWebConstructRow` (never built). `tests/fuzz/README.md` stale `polylogue.lib.timestamps`.
- Security (all LOW, none exploitable today): `_VALID_HEX` accepts trailing `\n` + unbounded
  length (use `re.fullmatch(r"[0-9a-f]{64}")`); `sanitize_path` symlink check is theater
  (CWD-relative, path never opened); zip per-entry 10 GiB cap, no aggregate cap.
- Codebase is otherwise notably clean (no TODO debt, SQL parameterized, secrets gated).

## Re-ingest reality (corrected by the impact agent)
`ops reset --database` deletes **source.db + index.db + embeddings.db + ops.db** and rebuilds
from the **on-disk corpus** (~41 GB: ~/.claude 16G + ~/.codex 21G + chatlog 3.8G), not from
source.db. `user.db` + blob store preserved. **embeddings.db is deleted and NOT regenerated**
(embedding disabled) → semantic search goes dark until `ops embed backfill`. Runtime ~2-6 h,
I/O then CPU/write bound. Snapshot first; run under `sinnix-scope background`; don't query
mid-run (WAL). Capture `ops diagnostics workload --json` before/after.

## Cost residual (the ~10%), quantified
- Codex live rollups still double-count cache reads (naive 295B vs 139.3B = 2.12×); corrected
  input+output = 150.5B = 1.08×, **per-thread median 1.0024**. Closes on re-ingest.
- Claude +9.6%: lineage/resume duplication +6.1pp (closes via logical-session basis),
  comparison-window staleness +3.5pp (stats-cache through 06-21), server_tool_use ~0
  (all-zero corpus), cache-read pricing policy is the cost (not token) axis.

## Build-ready designs delivered (ready to implement)
- WS2 forensics fold: ~70% already materialized; only gaps = reasoning-token lane +
  temporal usage-timeline insight + optional markdown render. Drop the script's inferior
  hand-rolled credit constants.
- `devtools lab probe cost-reconciliation`: full design + external-store field maps.
- `polylogue topic-pack`: staged multi-channel retrieval spec; most primitives exist.
- Compaction boundary-range (#7) + non-inline attachment acquisition (#8): full designs.
- Issue triage: 25 open; #2467/#2468/#2316/#2456 narrow-not-close after merge; quick wins
  #2435 Grok, #2421 blob-ref audit; #2465 scale-hardening is highest operational risk.

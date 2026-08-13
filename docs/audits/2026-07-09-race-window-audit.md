# Get-\>modify-\>put race audit across daemon/CLI/MCP writers

**Date**: 2026-07-09
**Bead**: polylogue-9e5.4
**Method**: static read of connection/transaction boundaries around every
candidate read-then-write sequence named in the bead, plus other shared-writer
surfaces discovered while tracing them. No product code was changed to
produce this audit. Two race windows were substantiated with a minimal
two-connection/two-step proof test (evidence only, not a fix) — see
"Proof harness" below.

## Method

For each sequence: read the exact function(s), classify the connection
boundary (one connection/one `with conn:` block spanning read+write, vs.
separate `open_connection`/`sqlite3.connect` calls), classify the transaction
boundary, state the invariant, construct a concrete two-actor interleaving,
and give a verdict:

- **safe-by-single-transaction** — read and write share one open transaction;
  no other connection can observe/mutate the intermediate state.
- **safe-by-unique/upsert** — the write is a full-row `INSERT ... ON CONFLICT
  DO UPDATE` keyed by a real uniqueness constraint, and every writer computes
  the SAME deterministic absolute value (not a delta based on a prior read),
  so interleaving produces "last write wins" with no corrupted intermediate
  state.
- **needs-harness** — a plausible unlocked read-then-write gap exists but
  static reading alone can't establish whether two real actors ever touch the
  same key concurrently in production.
- **bug** — concrete two-actor interleaving with a reproducible bad outcome
  (lost update, stale read that gets persisted, or a safety mechanism that
  never actually engages).

## Race-window table

| # | Sequence | File:function | Connection boundary | Txn boundary | Invariant | Verdict |
|---|----------|---------------|----------------------|---------------|-----------|---------|
| 1a | Blob-lease acquire/release inside a write | `polylogue/archive/write_effects.py:commit_archive_write_effects` | Deliberately split: `acquire_blob_leases` on a **fresh immediate-commit connection** (line 76), the main data commit on the caller's `conn`, `release_operation_leases` back on `conn` after commit (or a fresh connection on the failure path, `_release_leases_on_failure`) | Each piece is its own single-statement commit; the split is intentional (comment lines 68-71) so the lease is visible to a concurrent GC connection before the data txn commits | Lease must be visible to GC before the referencing row commits, and released only after that row is durable | **safe-by-single-transaction reasoning holds for the acquire/release pair itself** — `INSERT OR IGNORE`/`DELETE` are each atomic single statements; the split-connection design is correct **if invoked**. See 1b for why it currently is not. |
| 1b | Blob-lease **reachability** from real ingest | `polylogue/pipeline/services/ingest_batch/_core.py:_commit_sync_ingest_side_effects` (only production caller of `commit_archive_write_effects`) | N/A — the payload it builds never includes `_blob_hashes`/`_operation_id` | N/A | GC safety invariant #2 ("never delete a blob with an active lease", `blob_gc.py:11`) requires a lease to exist while a blob is acquired-but-not-yet-referenced | **BUG** — filed as polylogue-v7e0. `has_lease = bool(blob_hashes and operation_id)` (`write_effects.py:72`) is always `False` in production: a repo-wide grep confirms `_blob_hashes`/`_operation_id` are never set outside `write_effects.py`'s own `payload.get(...)` defaults and the unit tests that call `commit_archive_write_effects` directly. `acquire_blob_leases`/`release_operation_leases` are otherwise only referenced from `blob_gc.py` itself and tests. `WriteOperation.BLOB_STORE` is declared and never constructed anywhere. |
| 2 | Blob GC check-then-unlink pass | `polylogue/storage/blob_gc.py:run_blob_gc_report` | One connection/one pass for the whole loop (`conn = open_connection(db_path)` at top, closed in `finally`); file `unlink()` itself is a separate, non-transactional OS call | `_reference_surfaces` (SELECT) and `_has_active_lease` (SELECT) run inside the same open connection as the eventual `INSERT INTO gc_generations` commit, but the file delete happens *between* those reads and that commit, with no lock preventing another writer from referencing the blob in between | Never delete a blob that a concurrent ingest is about to reference | **needs-harness given 1b.** With leases dead, the only defense against deleting a blob whose DB reference hasn't committed yet is the `MIN_AGE_S=60` + previous-generation-timestamp age gate (`run_blob_gc_report:394-405`). This is a real gap only when a single ingest's acquire→commit span exceeds ~60s (plausible for the documented multi-GiB streaming Claude Code path) **and** an operator or scheduled job runs `polylogue maintenance blob-gc --yes` (CLI, `cli/commands/maintenance.py:1790`) during that window — i.e. exactly the CLI-vs-daemon shared-writer scenario the bead is about. Not filed as a separate bug; it is a direct consequence of 1b and will close together with it. |
| 3 | Ingest cursor failure bookkeeping | `polylogue/sources/live/cursor.py:CursorStore.mark_failed` / `.mark_excluded` / `.reset_failures` | `get_record(path)` opens+closes one `_connect_ops()` connection; the subsequent `self.set(...)` opens+closes a **second, independent** `_connect_ops()` connection. No lock spans the pair (`best_effort_cursor_write` is a lock-**retry** wrapper, not a cross-call lock) | Two separate single-statement transactions | `ingest_cursor.failure_count` accumulates real parse failures 1:1 so `_MAX_CURSOR_FAILURES_BEFORE_EXCLUDE=5` fires after 5 true failures, and exponential backoff (`delay_s = 60*2**(failures-1)`) is computed from the true count | **BUG** — filed as polylogue-qug2. Two actors calling `mark_failed(path)`/`reset_failures(path)`/`mark_excluded(path)` for the **same** `source_path` near-simultaneously (e.g. the live daemon watcher tailing a file while an operator's `polylogue import`/reprocess CLI batch-parses the same directory — both construct a `CursorStore` over the same `ops.db`) both read the same stale `failure_count`, both compute `+1` independently, and the second `set()`'s full-row upsert (`upsert_ingest_cursor`, `ops_write.py:135` `ON CONFLICT DO UPDATE SET failure_count = excluded.failure_count`) overwrites the first — one real failure is never counted. Consequence: delayed poison-pill exclusion / under-lengthened backoff, not data loss. Confirmed with a proof test (see below). |
| 3b | Convergence-debt attempt counting | `polylogue/sources/live/cursor.py:CursorStore._sync_convergence_debt_to_ops` | Same shape as #3: a `SELECT attempts, next_retry_at, last_error` on one `_connect_ops()` connection, Python computes `attempts_delta`/`retry_at`, then a **second** `_connect_ops()` connection commits `add_archive_convergence_debt` | Two separate transactions | `convergence_debt.attempts` should count real failed convergence attempts for a `(stage, target_type, target_id)` key | Same root cause as #3 (not filed as a separate bead — same fix will address both call sites in `cursor.py`). |
| 4 | Embedding `needs_reindex` transition on success | `polylogue/storage/embeddings/materialization.py:_record_archive_embedding_success` (embeds session, then blind `needs_reindex = 0`) vs. `polylogue/daemon/convergence_stages.py:_reconcile_embedding_config_change` (bulk `UPDATE embedding_status SET needs_reindex = 1` on model/dimension change, line 676) | Each write is its own single-statement upsert/UPDATE on its own connection — individually atomic | Individually safe-by-upsert (keyed on `session_id` PK), but the **pair** is not: neither write is conditioned on the other's generation/version | A session marked `needs_reindex=1` because the configured embedding model changed must stay `needs_reindex=1` until it is actually re-embedded **under the new model** | **BUG** — filed as polylogue-y337. `_reconcile_embedding_config_change` runs on every `_archive_embed_check*` probe call (`convergence_stages.py:1233,1268,1306`), not just at daemon startup. If it detects a model/dimension change and bulk-marks all rows `needs_reindex=1` *while* an in-flight `_embed_archive_sessions_sync`/`embed_archive_session_sync` pass for some session is mid-flight (already past its read of messages, still computing embeddings under the **old** model/provider), that pass's terminal `_record_archive_embedding_success` unconditionally sets `needs_reindex = 0` (`materialization.py:1044-1049`), silently clobbering the just-set reindex requirement. The session is left marked "fresh" while holding embeddings from the superseded model/dimension. Confirmed with a proof test (see below). |
| 5 | FTS freshness snapshot writes | `polylogue/storage/fts/freshness.py:record_fts_surface_state_sync` / `mark_all_fts_stale_sync` | `INSERT ... ON CONFLICT(surface) DO UPDATE` — single statement, keyed on `surface` PK | Whatever transaction the caller is already in | `fts_freshness_state` reflects "state as of last probe" | **safe-by-unique/upsert + self-healing.** Every writer stamps a freshly-recomputed absolute snapshot (state + counts as of that read), never a delta, so a losing writer just leaves a one-cycle-stale snapshot that the next probe corrects — matching the documented `false_means_pending`/convergence-retry model (`daemon/convergence.py`). The one place this snapshot participates in a hard atomicity requirement — `suspend_fts_triggers_sync` calling `mark_all_fts_stale_sync` right before dropping FTS triggers for a bulk write — runs on the **same connection, same transaction** as the surrounding `commit_archive_write_effects`/ingest-batch `BEGIN IMMEDIATE` (`pipeline/services/ingest_batch/_core.py:975-979`, `storage/fts/fts_lifecycle.py:256-263`). This is exactly the "already single-transaction, do not misclassify" pattern the bead calls out. |
| 6 | `commit_archive_write_effects` overall | `polylogue/archive/write_effects.py:commit_archive_write_effects` | One caller-owned `conn`; FTS trigger repair + `conn.commit()` all inside the function; blob-lease acquire/release are the *only* pieces deliberately outside that transaction (see #1a) | `ensure_fts_triggers_sync` → `repair_message_fts_index_sync` → `conn.commit()` is one unbroken sequence on one connection before the function returns | Row materialization, FTS repair, and commit land atomically together (#1242) | **safe-by-single-transaction**, confirmed by reading lines 84-116 directly — this is the sequence the bead explicitly warns not to misfile as a bug. |

## Bug beads filed

All three carry `discovered-from:polylogue-9e5.4`. No fixes were implemented;
each bead's repro is the two-connection/two-step sketch from the table above
plus (for 4.2 and 4.3) a runnable proof test.

- **polylogue-v7e0** — Blob-lease safety mechanism (`pending_blob_refs`,
  `acquire_blob_leases`/`release_operation_leases`) is dead code: no real
  ingest caller populates `_blob_hashes`/`_operation_id`, so GC's "never
  delete a leased blob" invariant never actually engages; the sole real
  defense is the `MIN_AGE_S` timing heuristic. Includes the derived GC
  check-then-unlink exposure (table row #2) as the same root cause.
- **polylogue-qug2** — `CursorStore.mark_failed`/`mark_excluded`/
  `reset_failures` (and `_sync_convergence_debt_to_ops`) do an unlocked
  get-on-one-connection, set-on-another read-modify-write; two concurrent
  callers touching the same `source_path`/subject lose an increment.
- **polylogue-y337** — `_record_archive_embedding_success`'s unconditional
  `needs_reindex = 0` can silently clobber a concurrent
  `_reconcile_embedding_config_change`'s `needs_reindex = 1` bulk marker,
  leaving stale-model embeddings marked fresh.

## Proof harness

Two minimal, deterministic (no real threads — the interleaving is driven
explicitly by call order, which is the strongest and least flaky way to
demonstrate a two-actor race) tests were added as evidence, not a fix:

- `tests/unit/sources/test_cursor_failure_count_race_evidence.py::test_mark_failed_lost_update_when_two_actors_read_before_either_writes`
  — seeds `failure_count=2`, has two "actors" call the real
  `CursorStore.get_record`/`.set` in the exact interleaved order the table
  describes, and asserts the final `failure_count` is `3`, not `4` — the
  literal lost update. **Result: passes, i.e. reproduces the bug.**
- `tests/unit/storage/test_embedding_needs_reindex_race_evidence.py::test_embedding_success_write_clobbers_concurrent_reindex_request`
  — builds a bare `embedding_status` table (the real DDL fragment), seeds a
  `needs_reindex=0` row, runs the real config-change bulk-mark SQL, then the
  real `_record_archive_embedding_success`, and asserts the row ends up
  `needs_reindex=0` despite the intervening mark. **Result: passes, i.e.
  reproduces the bug.**

Verification: `devtools test -k test_mark_failed_lost_update_when_two_actors_read_before_either_writes` and `devtools test -k test_embedding_success_write_clobbers_concurrent_reindex_request` both pass locally (only these two new tests; no broad run for the audit itself).

## Sequences classified safe (no bead filed)

- `commit_archive_write_effects` (#6) — safe-by-single-transaction, matches
  the bead's own flagged pitfall exactly.
- Blob-lease acquire/release mechanics in isolation (#1a) — safe-by-design
  if invoked; the bug is that it is never invoked (#1b).
- `fts_freshness_state` writes (#5) — safe-by-upsert + self-healing probe
  design; the one atomicity-sensitive use is already single-transaction.
- `embedding_status` per-message/per-session upserts in the non-racing case
  — safe-by-upsert (keyed on `session_id`/`message_id` PKs, deterministic
  absolute writes).

## Other shared-writer surfaces surveyed, no further findings

`session_profiles` upserts and the `gc_generations` insert (`blob_gc.py:497`)
are both single-statement, single-transaction writes with no preceding
cross-connection read of the same row; not included as separate table rows
above because they do not fit the get-modify-put shape at all (they are pure
inserts/upserts of freshly-computed values, the same reasoning as row #5).

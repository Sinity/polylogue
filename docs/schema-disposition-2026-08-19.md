# Schema disposition audit — full column inventory (gvzkr)

Date: 2026-08-19. Committed artifact for `polylogue-gvzkr`'s column-level schema-disposition
acceptance criteria. Extends the 2026-08-04 table-level seed
(`.agent/scratch/polylogue-gvzkr-schema-disposition-2026-08-04.md`, not committed — a working
scratch file) to full column granularity across all five SQLite tiers, then normalizes and
recomputes it against live DDL for this commit.

**Every number in the summary table below is recomputed directly from the row table further
down in this file** — never hand-entered — specifically to avoid the header/body disagreement
that the source working notes had. That disagreement turned out to be pervasive, not isolated
to one tier: see "Correction versus the working notes" below.

## Coverage summary (machine-derivable — recomputed from the row table, not hand-maintained)

| tier | tables | columns | KEEP | PURGE | UNCLEAR |
|---|---:|---:|---:|---:|---:|
| source.db | 37 | 357 | 273 | 76 | 8 |
| index.db | 61 | 650 | 639 | 11 | 0 |
| embeddings.db | 6 | 50 | 48 | 2 | 0 |
| ops.db | 16 | 158 | 138 | 20 | 0 |
| user.db | 16 | 121 | 120 | 0 | 1 |
| **total** | **136** | **1336** | **1218** | **109** | **9** |

Column counts are enumerated from live `PRAGMA table_info` output against each tier's canonical
`*_DDL` constant in `polylogue/storage/sqlite/archive_tiers/{source,index,embeddings,ops,user}.py`
(embeddings.db's `message_embeddings` vec0 virtual table was enumerated from its `CREATE VIRTUAL
TABLE ... USING vec0(...)` DDL text directly, since the `vec0` loadable extension is not present
in this environment to execute it). index.db's 33 FTS5 shadow-table columns (`messages_fts_{config,data,docsize,idx}`, `blocks_command_trigram_{config,data,docsize,idx}`, `session_work_events_fts_{config,content,data,docsize,idx}`) are included as individual rows with an inherited KEEP verdict, per the audit's own rule that they track their parent FTS table's disposition rather than being independently classified.

## Correction versus the working notes

The working scratch files this artifact normalizes had a real, pervasive defect: **every tier's
own header summary line undercounted its own body table**, not just ops.db's (which is the one
case the dispatching instruction already knew about). Recomputing from live DDL via
`PRAGMA table_info` and reconciling against each per-tier working file's row content found:

- **source.db**: working notes claimed 33 tables / 208 columns. Live DDL has **37 tables / 357
  columns** — the working notes' own body text already named all 37 tables (2 fully detailed
  column-by-column, 28 more given table-level KEEP/PURGE/UNCLEAR verdicts, 7 delegated as
  PURGE-pending-execution), it just never summed them. This file expands every table-level
  verdict to one row per real DDL column so the AC's "every column exactly once" holds; no new
  columns were audited beyond what the working notes already covered per table.
- **index.db**: working notes' own body (617 individually classified + 33 inherited FTS-shadow)
  already summed correctly to 650, matching live DDL exactly. No correction needed here beyond
  transcription.
- **embeddings.db**: working notes claimed 49 columns; the body table itself has 50 rows,
  matching live DDL (the header simply mis-added KEEP 47 + PURGE 2).
- **ops.db**: working notes claimed 130 columns (aggregate) / 129 (per-tier header), both wrong;
  the per-tier body has 158 rows, matching live DDL exactly — this is the same defect the
  dispatching instruction flagged, confirmed and fixed the same way (recompute from the body).
- **user.db**: working notes claimed 92 columns; the body table has 121 rows, matching live DDL
  exactly.

Net effect: the true total is **1,336 columns across 136 tables/views**, not the ~1,124 the
aggregate working file estimated. The disposition *content* (which columns are KEEP/PURGE/
UNCLEAR and why) is unchanged from the working notes — this was a counting/summation defect in
the header lines, not a re-audit of any column's evidence.

## UNCLEAR — adjudicate at PR review, not guessed here

9 individual column rows are UNCLEAR, covering 2 distinct findings (both carried from the
underlying audit — this pass did not resolve either):

1. **`source.excised_content`** (8 of 8 columns: `removed_hash`, `hash_kind`, `reason`, `actor`,
   `prior_revision`, `span_start`, `span_end`, `excised_at_ms`) — the table's own DDL comment
   (`polylogue/storage/sqlite/archive_tiers/source.py:837-840`) states it is "never queried for
   its own sake by a reader — it exists purely as a write-time gate plus forensic trail," which
   argues against PURGE (it is deliberately write-mostly by design, not accidentally dead), but
   no column-level production SELECT was found either, which argues against a confident KEEP.
   **What would settle it:** confirm with whoever owns the excision/GC subsystem whether the
   forensic-trail read path is genuinely unimplemented-but-wanted, or was deliberately never
   built because the write-time gate is sufficient on its own.
2. **`user.query_names.supersedes_query_hash`** (1 column) — written on every INSERT/UPSERT
   (`polylogue/storage/sqlite/query_objects.py:128-136`) but never appears in any production
   SELECT list; the column name and a DDL comment suggest it was meant to support
   renamed/superseded-query-name tracking, a real feature shape, not obviously dead weight.
   **What would settle it:** confirm whether renamed-query tracking (the `polylogue-4p1`
   query-object cluster) ever shipped a reader for this column, or was left half-built.

Both are graded UNCLEAR rather than PURGE deliberately, per durable/write-mostly-by-design
conservatism — a wrong PURGE on either would need an explicit-consent-gated destructive
migration to undo. Do not guess a verdict for either at PR review without the owner input named
above.

## Full column table

One row per column, every column exactly once. `tier.table.column` | verdict | evidence
citation | unlocks / owning reference.

### source.db

| tier.table.column | verdict | evidence | unlocks / owning reference |
|---|---|---|---|
| `source.audit_continuity_control.singleton` | KEEP | polylogue/storage/sqlite/audit_continuity.py (every column selected across lines 282-672), polylogue/operations/durable_change_train.py:1002-1560, polylogue/maintenance/raw_authority_recovery.py (generic-row dump excludes this table by name as "volatile"). singleton/prepared_at_ms confirmed via CHECK constraint + INSERT, not an explicit SELECT. | n/a |
| `source.audit_continuity_control.committed_generation` | KEEP | polylogue/storage/sqlite/audit_continuity.py (every column selected across lines 282-672), polylogue/operations/durable_change_train.py:1002-1560, polylogue/maintenance/raw_authority_recovery.py (generic-row dump excludes this table by name as "volatile"). singleton/prepared_at_ms confirmed via CHECK constraint + INSERT, not an explicit SELECT. | n/a |
| `source.audit_continuity_control.committed_head_sha256` | KEEP | polylogue/storage/sqlite/audit_continuity.py (every column selected across lines 282-672), polylogue/operations/durable_change_train.py:1002-1560, polylogue/maintenance/raw_authority_recovery.py (generic-row dump excludes this table by name as "volatile"). singleton/prepared_at_ms confirmed via CHECK constraint + INSERT, not an explicit SELECT. | n/a |
| `source.audit_continuity_control.pending_mutation_id` | KEEP | polylogue/storage/sqlite/audit_continuity.py (every column selected across lines 282-672), polylogue/operations/durable_change_train.py:1002-1560, polylogue/maintenance/raw_authority_recovery.py (generic-row dump excludes this table by name as "volatile"). singleton/prepared_at_ms confirmed via CHECK constraint + INSERT, not an explicit SELECT. | n/a |
| `source.audit_continuity_control.pending_payload_json` | KEEP | polylogue/storage/sqlite/audit_continuity.py (every column selected across lines 282-672), polylogue/operations/durable_change_train.py:1002-1560, polylogue/maintenance/raw_authority_recovery.py (generic-row dump excludes this table by name as "volatile"). singleton/prepared_at_ms confirmed via CHECK constraint + INSERT, not an explicit SELECT. | n/a |
| `source.audit_continuity_control.pending_payload_sha256` | KEEP | polylogue/storage/sqlite/audit_continuity.py (every column selected across lines 282-672), polylogue/operations/durable_change_train.py:1002-1560, polylogue/maintenance/raw_authority_recovery.py (generic-row dump excludes this table by name as "volatile"). singleton/prepared_at_ms confirmed via CHECK constraint + INSERT, not an explicit SELECT. | n/a |
| `source.audit_continuity_control.prepared_at_ms` | KEEP | polylogue/storage/sqlite/audit_continuity.py (every column selected across lines 282-672), polylogue/operations/durable_change_train.py:1002-1560, polylogue/maintenance/raw_authority_recovery.py (generic-row dump excludes this table by name as "volatile"). singleton/prepared_at_ms confirmed via CHECK constraint + INSERT, not an explicit SELECT. | n/a |
| `source.blob_publication_reservations.publication_id` | KEEP | blob_store.py, blob_gc.py (seed-carried) | n/a |
| `source.blob_publication_reservations.blob_hash` | KEEP | blob_store.py, blob_gc.py (seed-carried) | n/a |
| `source.blob_publication_reservations.size_bytes` | KEEP | blob_store.py, blob_gc.py (seed-carried) | n/a |
| `source.blob_publication_reservations.publisher_id` | KEEP | blob_store.py, blob_gc.py (seed-carried) | n/a |
| `source.blob_publication_reservations.reserved_at_ms` | KEEP | blob_store.py, blob_gc.py (seed-carried) | n/a |
| `source.blob_refs.blob_hash` | KEEP | blob_gc.py:265,316, GC candidate scan | n/a |
| `source.blob_refs.ref_id` | KEEP | blob_gc.py referent-table resolution (ref_type/ref_id join), excision.py:322-551 | n/a |
| `source.blob_refs.ref_type` | KEEP | blob_gc.py:171-297, archive_verification.py:1107-1143, blob_ref_liveness.py:217 | n/a |
| `source.blob_refs.source_path` | KEEP | repair.py:568,857 SELECT blob_hash, source_path, size_bytes FROM blob_refs | n/a |
| `source.blob_refs.size_bytes` | KEEP | same repair.py:568,857 select | n/a |
| `source.blob_refs.acquired_at_ms` | KEEP | source_write.py:1374,1382 SELECT acquired_at_ms, rowid FROM blob_refs; durable_change_train.py:2158 copy-forward read | n/a |
| `source.excised_content.removed_hash` | UNCLEAR | DDL comment at source.py:837-840: "never queried for its own sake by a reader - exists purely as a write-time gate plus forensic trail." Deliberately write-mostly by design (write-time re-acquisition gate IS its function), not accidentally write-only - so PURGE would be wrong, but no column-level production SELECT found either (seed-carried UNCLEAR). | operator call - deliberately write-mostly by design, not a confident PURGE |
| `source.excised_content.hash_kind` | UNCLEAR | DDL comment at source.py:837-840: "never queried for its own sake by a reader - exists purely as a write-time gate plus forensic trail." Deliberately write-mostly by design (write-time re-acquisition gate IS its function), not accidentally write-only - so PURGE would be wrong, but no column-level production SELECT found either (seed-carried UNCLEAR). | operator call - deliberately write-mostly by design, not a confident PURGE |
| `source.excised_content.reason` | UNCLEAR | DDL comment at source.py:837-840: "never queried for its own sake by a reader - exists purely as a write-time gate plus forensic trail." Deliberately write-mostly by design (write-time re-acquisition gate IS its function), not accidentally write-only - so PURGE would be wrong, but no column-level production SELECT found either (seed-carried UNCLEAR). | operator call - deliberately write-mostly by design, not a confident PURGE |
| `source.excised_content.actor` | UNCLEAR | DDL comment at source.py:837-840: "never queried for its own sake by a reader - exists purely as a write-time gate plus forensic trail." Deliberately write-mostly by design (write-time re-acquisition gate IS its function), not accidentally write-only - so PURGE would be wrong, but no column-level production SELECT found either (seed-carried UNCLEAR). | operator call - deliberately write-mostly by design, not a confident PURGE |
| `source.excised_content.prior_revision` | UNCLEAR | DDL comment at source.py:837-840: "never queried for its own sake by a reader - exists purely as a write-time gate plus forensic trail." Deliberately write-mostly by design (write-time re-acquisition gate IS its function), not accidentally write-only - so PURGE would be wrong, but no column-level production SELECT found either (seed-carried UNCLEAR). | operator call - deliberately write-mostly by design, not a confident PURGE |
| `source.excised_content.span_start` | UNCLEAR | DDL comment at source.py:837-840: "never queried for its own sake by a reader - exists purely as a write-time gate plus forensic trail." Deliberately write-mostly by design (write-time re-acquisition gate IS its function), not accidentally write-only - so PURGE would be wrong, but no column-level production SELECT found either (seed-carried UNCLEAR). | operator call - deliberately write-mostly by design, not a confident PURGE |
| `source.excised_content.span_end` | UNCLEAR | DDL comment at source.py:837-840: "never queried for its own sake by a reader - exists purely as a write-time gate plus forensic trail." Deliberately write-mostly by design (write-time re-acquisition gate IS its function), not accidentally write-only - so PURGE would be wrong, but no column-level production SELECT found either (seed-carried UNCLEAR). | operator call - deliberately write-mostly by design, not a confident PURGE |
| `source.excised_content.excised_at_ms` | UNCLEAR | DDL comment at source.py:837-840: "never queried for its own sake by a reader - exists purely as a write-time gate plus forensic trail." Deliberately write-mostly by design (write-time re-acquisition gate IS its function), not accidentally write-only - so PURGE would be wrong, but no column-level production SELECT found either (seed-carried UNCLEAR). | operator call - deliberately write-mostly by design, not a confident PURGE |
| `source.gc_generations.generation_id` | KEEP | blob_store.py, blob_gc.py (seed-carried) | n/a |
| `source.gc_generations.started_at_ms` | KEEP | blob_store.py, blob_gc.py (seed-carried) | n/a |
| `source.gc_generations.completed_at_ms` | KEEP | blob_store.py, blob_gc.py (seed-carried) | n/a |
| `source.gc_generations.reclaimed_count` | KEEP | blob_store.py, blob_gc.py (seed-carried) | n/a |
| `source.gc_generations.reclaimed_bytes` | KEEP | blob_store.py, blob_gc.py (seed-carried) | n/a |
| `source.history_sidecars.sidecar_id` | KEEP | source_write.py, schema_inference_gate.py, archive_verification.py, blob_gc.py (seed-carried) | n/a |
| `source.history_sidecars.origin` | KEEP | source_write.py, schema_inference_gate.py, archive_verification.py, blob_gc.py (seed-carried) | n/a |
| `source.history_sidecars.source_path` | KEEP | source_write.py, schema_inference_gate.py, archive_verification.py, blob_gc.py (seed-carried) | n/a |
| `source.history_sidecars.payload_json` | KEEP | source_write.py, schema_inference_gate.py, archive_verification.py, blob_gc.py (seed-carried) | n/a |
| `source.history_sidecars.observed_at_ms` | KEEP | source_write.py, schema_inference_gate.py, archive_verification.py, blob_gc.py (seed-carried) | n/a |
| `source.history_sidecars.content_hash` | KEEP | source_write.py, schema_inference_gate.py, archive_verification.py, blob_gc.py (seed-carried) | n/a |
| `source.otlp_spans.span_id` | PURGE | rg -n 'otlp_spans' polylogue devtools tests finds only DDL/migration/migration-test references, no production SELECT. Durable-tier drop requires polylogue-60i5 admission (seed-carried PURGE candidate, re-confirmed). | requires polylogue-60i5 durable-migration admission before any drop |
| `source.otlp_spans.trace_id` | PURGE | rg -n 'otlp_spans' polylogue devtools tests finds only DDL/migration/migration-test references, no production SELECT. Durable-tier drop requires polylogue-60i5 admission (seed-carried PURGE candidate, re-confirmed). | requires polylogue-60i5 durable-migration admission before any drop |
| `source.otlp_spans.parent_span_id` | PURGE | rg -n 'otlp_spans' polylogue devtools tests finds only DDL/migration/migration-test references, no production SELECT. Durable-tier drop requires polylogue-60i5 admission (seed-carried PURGE candidate, re-confirmed). | requires polylogue-60i5 durable-migration admission before any drop |
| `source.otlp_spans.origin` | PURGE | rg -n 'otlp_spans' polylogue devtools tests finds only DDL/migration/migration-test references, no production SELECT. Durable-tier drop requires polylogue-60i5 admission (seed-carried PURGE candidate, re-confirmed). | requires polylogue-60i5 durable-migration admission before any drop |
| `source.otlp_spans.session_native_id` | PURGE | rg -n 'otlp_spans' polylogue devtools tests finds only DDL/migration/migration-test references, no production SELECT. Durable-tier drop requires polylogue-60i5 admission (seed-carried PURGE candidate, re-confirmed). | requires polylogue-60i5 durable-migration admission before any drop |
| `source.otlp_spans.name` | PURGE | rg -n 'otlp_spans' polylogue devtools tests finds only DDL/migration/migration-test references, no production SELECT. Durable-tier drop requires polylogue-60i5 admission (seed-carried PURGE candidate, re-confirmed). | requires polylogue-60i5 durable-migration admission before any drop |
| `source.otlp_spans.kind` | PURGE | rg -n 'otlp_spans' polylogue devtools tests finds only DDL/migration/migration-test references, no production SELECT. Durable-tier drop requires polylogue-60i5 admission (seed-carried PURGE candidate, re-confirmed). | requires polylogue-60i5 durable-migration admission before any drop |
| `source.otlp_spans.attributes_json` | PURGE | rg -n 'otlp_spans' polylogue devtools tests finds only DDL/migration/migration-test references, no production SELECT. Durable-tier drop requires polylogue-60i5 admission (seed-carried PURGE candidate, re-confirmed). | requires polylogue-60i5 durable-migration admission before any drop |
| `source.otlp_spans.events_json` | PURGE | rg -n 'otlp_spans' polylogue devtools tests finds only DDL/migration/migration-test references, no production SELECT. Durable-tier drop requires polylogue-60i5 admission (seed-carried PURGE candidate, re-confirmed). | requires polylogue-60i5 durable-migration admission before any drop |
| `source.otlp_spans.started_at_ms` | PURGE | rg -n 'otlp_spans' polylogue devtools tests finds only DDL/migration/migration-test references, no production SELECT. Durable-tier drop requires polylogue-60i5 admission (seed-carried PURGE candidate, re-confirmed). | requires polylogue-60i5 durable-migration admission before any drop |
| `source.otlp_spans.ended_at_ms` | PURGE | rg -n 'otlp_spans' polylogue devtools tests finds only DDL/migration/migration-test references, no production SELECT. Durable-tier drop requires polylogue-60i5 admission (seed-carried PURGE candidate, re-confirmed). | requires polylogue-60i5 durable-migration admission before any drop |
| `source.otlp_spans.received_at_ms` | PURGE | rg -n 'otlp_spans' polylogue devtools tests finds only DDL/migration/migration-test references, no production SELECT. Durable-tier drop requires polylogue-60i5 admission (seed-carried PURGE candidate, re-confirmed). | requires polylogue-60i5 durable-migration admission before any drop |
| `source.raw_append_chain_backfill_receipts.raw_id` | KEEP | seed-carried, re-confirmed | n/a |
| `source.raw_append_chain_backfill_receipts.logical_source_key` | KEEP | seed-carried, re-confirmed | n/a |
| `source.raw_append_chain_backfill_receipts.source_path` | KEEP | seed-carried, re-confirmed | n/a |
| `source.raw_append_chain_backfill_receipts.blob_hash` | KEEP | seed-carried, re-confirmed | n/a |
| `source.raw_append_chain_backfill_receipts.blob_size` | KEEP | seed-carried, re-confirmed | n/a |
| `source.raw_append_chain_backfill_receipts.append_start_offset` | KEEP | seed-carried, re-confirmed | n/a |
| `source.raw_append_chain_backfill_receipts.append_end_offset` | KEEP | seed-carried, re-confirmed | n/a |
| `source.raw_append_chain_backfill_receipts.matched_after_codex_header_strip` | KEEP | seed-carried, re-confirmed | n/a |
| `source.raw_append_chain_backfill_receipts.previous_revision_authority` | KEEP | seed-carried, re-confirmed | n/a |
| `source.raw_append_chain_backfill_receipts.compared_at_ms` | KEEP | seed-carried, re-confirmed | n/a |
| `source.raw_append_chain_backfill_receipts.tool_version` | KEEP | seed-carried, re-confirmed | n/a |
| `source.raw_append_chain_backfill_receipts.backup_manifest_path` | KEEP | seed-carried, re-confirmed | n/a |
| `source.raw_append_chain_backfill_receipts.detail` | KEEP | seed-carried, re-confirmed | n/a |
| `source.raw_artifacts.artifact_id` | KEEP | schema_inference_gate.py, blob_gc.py:86 explicit table list, maintenance/archive_verification.py (seed-carried) | n/a |
| `source.raw_artifacts.raw_id` | KEEP | schema_inference_gate.py, blob_gc.py:86 explicit table list, maintenance/archive_verification.py (seed-carried) | n/a |
| `source.raw_artifacts.origin` | KEEP | schema_inference_gate.py, blob_gc.py:86 explicit table list, maintenance/archive_verification.py (seed-carried) | n/a |
| `source.raw_artifacts.source_path` | KEEP | schema_inference_gate.py, blob_gc.py:86 explicit table list, maintenance/archive_verification.py (seed-carried) | n/a |
| `source.raw_artifacts.source_index` | KEEP | schema_inference_gate.py, blob_gc.py:86 explicit table list, maintenance/archive_verification.py (seed-carried) | n/a |
| `source.raw_artifacts.artifact_kind` | KEEP | schema_inference_gate.py, blob_gc.py:86 explicit table list, maintenance/archive_verification.py (seed-carried) | n/a |
| `source.raw_artifacts.support_status` | KEEP | schema_inference_gate.py, blob_gc.py:86 explicit table list, maintenance/archive_verification.py (seed-carried) | n/a |
| `source.raw_artifacts.classification_reason` | KEEP | schema_inference_gate.py, blob_gc.py:86 explicit table list, maintenance/archive_verification.py (seed-carried) | n/a |
| `source.raw_artifacts.parse_as_session` | KEEP | schema_inference_gate.py, blob_gc.py:86 explicit table list, maintenance/archive_verification.py (seed-carried) | n/a |
| `source.raw_artifacts.schema_eligible` | KEEP | schema_inference_gate.py, blob_gc.py:86 explicit table list, maintenance/archive_verification.py (seed-carried) | n/a |
| `source.raw_artifacts.malformed_jsonl_lines` | KEEP | schema_inference_gate.py, blob_gc.py:86 explicit table list, maintenance/archive_verification.py (seed-carried) | n/a |
| `source.raw_artifacts.decode_error` | KEEP | schema_inference_gate.py, blob_gc.py:86 explicit table list, maintenance/archive_verification.py (seed-carried) | n/a |
| `source.raw_artifacts.cohort_id` | KEEP | schema_inference_gate.py, blob_gc.py:86 explicit table list, maintenance/archive_verification.py (seed-carried) | n/a |
| `source.raw_artifacts.link_group_key` | KEEP | schema_inference_gate.py, blob_gc.py:86 explicit table list, maintenance/archive_verification.py (seed-carried) | n/a |
| `source.raw_artifacts.sidecar_agent_type` | KEEP | schema_inference_gate.py, blob_gc.py:86 explicit table list, maintenance/archive_verification.py (seed-carried) | n/a |
| `source.raw_artifacts.first_observed_at_ms` | KEEP | schema_inference_gate.py, blob_gc.py:86 explicit table list, maintenance/archive_verification.py (seed-carried) | n/a |
| `source.raw_artifacts.last_observed_at_ms` | KEEP | schema_inference_gate.py, blob_gc.py:86 explicit table list, maintenance/archive_verification.py (seed-carried) | n/a |
| `source.raw_authority_artifact_census_checkpoint_members.census_id` | KEEP | raw_authority_artifact_census.py:216,282,290 (SELECT raw_id ... WHERE census_id = ? ORDER BY ordinal) | n/a |
| `source.raw_authority_artifact_census_checkpoint_members.ordinal` | KEEP | raw_authority_artifact_census.py:216,282,290 (SELECT raw_id ... WHERE census_id = ? ORDER BY ordinal) | n/a |
| `source.raw_authority_artifact_census_checkpoint_members.raw_id` | KEEP | raw_authority_artifact_census.py:216,282,290 (SELECT raw_id ... WHERE census_id = ? ORDER BY ordinal) | n/a |
| `source.raw_authority_artifact_census_checkpoints.census_id` | KEEP | raw_authority_artifact_census.py:239 SELECT candidate_count, universe_sha256, universe_complete, snapshot_max_raw_rowid, ... | n/a |
| `source.raw_authority_artifact_census_checkpoints.universe_sha256` | KEEP | raw_authority_artifact_census.py:239 SELECT candidate_count, universe_sha256, universe_complete, snapshot_max_raw_rowid, ... | n/a |
| `source.raw_authority_artifact_census_checkpoints.candidate_count` | KEEP | raw_authority_artifact_census.py:239 SELECT candidate_count, universe_sha256, universe_complete, snapshot_max_raw_rowid, ... | n/a |
| `source.raw_authority_artifact_census_checkpoints.universe_complete` | KEEP | raw_authority_artifact_census.py:239 SELECT candidate_count, universe_sha256, universe_complete, snapshot_max_raw_rowid, ... | n/a |
| `source.raw_authority_artifact_census_checkpoints.snapshot_max_raw_rowid` | KEEP | raw_authority_artifact_census.py:239 SELECT candidate_count, universe_sha256, universe_complete, snapshot_max_raw_rowid, ... | n/a |
| `source.raw_authority_artifact_census_checkpoints.materialized_after_rowid` | KEEP | raw_authority_artifact_census.py:239 SELECT candidate_count, universe_sha256, universe_complete, snapshot_max_raw_rowid, ... | n/a |
| `source.raw_authority_artifact_census_checkpoints.index_generation` | KEEP | raw_authority_artifact_census.py:239 SELECT candidate_count, universe_sha256, universe_complete, snapshot_max_raw_rowid, ... | n/a |
| `source.raw_authority_artifact_census_checkpoints.index_identity_sha256` | KEEP | raw_authority_artifact_census.py:239 SELECT candidate_count, universe_sha256, universe_complete, snapshot_max_raw_rowid, ... | n/a |
| `source.raw_authority_artifact_census_checkpoints.next_after_raw_id` | KEEP | raw_authority_artifact_census.py:239 SELECT candidate_count, universe_sha256, universe_complete, snapshot_max_raw_rowid, ... | n/a |
| `source.raw_authority_artifact_census_checkpoints.last_receipt_id` | KEEP | raw_authority_artifact_census.py:239 SELECT candidate_count, universe_sha256, universe_complete, snapshot_max_raw_rowid, ... | n/a |
| `source.raw_authority_artifact_census_checkpoints.completed_at_ms` | KEEP | raw_authority_artifact_census.py:239 SELECT candidate_count, universe_sha256, universe_complete, snapshot_max_raw_rowid, ... | n/a |
| `source.raw_authority_artifact_census_checkpoints.created_at_ms` | KEEP | raw_authority_artifact_census.py:239 SELECT candidate_count, universe_sha256, universe_complete, snapshot_max_raw_rowid, ... | n/a |
| `source.raw_authority_artifact_census_receipts.receipt_id` | KEEP | polylogue/maintenance/raw_authority_artifact_census.py (receipt applied-at/tool-version bookkeeping for bounded census pages) | n/a |
| `source.raw_authority_artifact_census_receipts.receipt_sha256` | KEEP | polylogue/maintenance/raw_authority_artifact_census.py (receipt applied-at/tool-version bookkeeping for bounded census pages) | n/a |
| `source.raw_authority_artifact_census_receipts.receipt_json` | KEEP | polylogue/maintenance/raw_authority_artifact_census.py (receipt applied-at/tool-version bookkeeping for bounded census pages) | n/a |
| `source.raw_authority_artifact_census_receipts.backup_manifest_path` | KEEP | polylogue/maintenance/raw_authority_artifact_census.py (receipt applied-at/tool-version bookkeeping for bounded census pages) | n/a |
| `source.raw_authority_artifact_census_receipts.applied_at_ms` | KEEP | polylogue/maintenance/raw_authority_artifact_census.py (receipt applied-at/tool-version bookkeeping for bounded census pages) | n/a |
| `source.raw_authority_artifact_census_receipts.tool_version` | KEEP | polylogue/maintenance/raw_authority_artifact_census.py (receipt applied-at/tool-version bookkeeping for bounded census pages) | n/a |
| `source.raw_authority_blockers.blocker_id` | PURGE | Table-level PURGE-pending-execution inherited from the 2026-08-04 seed; not re-derived at column granularity this pass per the team-lead's cite-don't-re-derive instruction for delegated tables. Current writers still feed open P0 reconciliation work. | blocked on polylogue-w6hql/polylogue-lr6dx clearing before this is an executable drop |
| `source.raw_authority_blockers.plan_id` | PURGE | Table-level PURGE-pending-execution inherited from the 2026-08-04 seed; not re-derived at column granularity this pass per the team-lead's cite-don't-re-derive instruction for delegated tables. Current writers still feed open P0 reconciliation work. | blocked on polylogue-w6hql/polylogue-lr6dx clearing before this is an executable drop |
| `source.raw_authority_blockers.census_id` | PURGE | Table-level PURGE-pending-execution inherited from the 2026-08-04 seed; not re-derived at column granularity this pass per the team-lead's cite-don't-re-derive instruction for delegated tables. Current writers still feed open P0 reconciliation work. | blocked on polylogue-w6hql/polylogue-lr6dx clearing before this is an executable drop |
| `source.raw_authority_blockers.reason` | PURGE | Table-level PURGE-pending-execution inherited from the 2026-08-04 seed; not re-derived at column granularity this pass per the team-lead's cite-don't-re-derive instruction for delegated tables. Current writers still feed open P0 reconciliation work. | blocked on polylogue-w6hql/polylogue-lr6dx clearing before this is an executable drop |
| `source.raw_authority_blockers.expected_json` | PURGE | Table-level PURGE-pending-execution inherited from the 2026-08-04 seed; not re-derived at column granularity this pass per the team-lead's cite-don't-re-derive instruction for delegated tables. Current writers still feed open P0 reconciliation work. | blocked on polylogue-w6hql/polylogue-lr6dx clearing before this is an executable drop |
| `source.raw_authority_blockers.observed_json` | PURGE | Table-level PURGE-pending-execution inherited from the 2026-08-04 seed; not re-derived at column granularity this pass per the team-lead's cite-don't-re-derive instruction for delegated tables. Current writers still feed open P0 reconciliation work. | blocked on polylogue-w6hql/polylogue-lr6dx clearing before this is an executable drop |
| `source.raw_authority_blockers.created_at_ms` | PURGE | Table-level PURGE-pending-execution inherited from the 2026-08-04 seed; not re-derived at column granularity this pass per the team-lead's cite-don't-re-derive instruction for delegated tables. Current writers still feed open P0 reconciliation work. | blocked on polylogue-w6hql/polylogue-lr6dx clearing before this is an executable drop |
| `source.raw_authority_blockers.resolved_at_ms` | PURGE | Table-level PURGE-pending-execution inherited from the 2026-08-04 seed; not re-derived at column granularity this pass per the team-lead's cite-don't-re-derive instruction for delegated tables. Current writers still feed open P0 reconciliation work. | blocked on polylogue-w6hql/polylogue-lr6dx clearing before this is an executable drop |
| `source.raw_authority_blockers.resolution` | PURGE | Table-level PURGE-pending-execution inherited from the 2026-08-04 seed; not re-derived at column granularity this pass per the team-lead's cite-don't-re-derive instruction for delegated tables. Current writers still feed open P0 reconciliation work. | blocked on polylogue-w6hql/polylogue-lr6dx clearing before this is an executable drop |
| `source.raw_authority_census_plans.census_id` | PURGE | Table-level PURGE-pending-execution inherited from the 2026-08-04 seed; not re-derived at column granularity this pass per the team-lead's cite-don't-re-derive instruction for delegated tables. Current writers still feed open P0 reconciliation work. | blocked on polylogue-w6hql/polylogue-lr6dx clearing before this is an executable drop |
| `source.raw_authority_census_plans.plan_id` | PURGE | Table-level PURGE-pending-execution inherited from the 2026-08-04 seed; not re-derived at column granularity this pass per the team-lead's cite-don't-re-derive instruction for delegated tables. Current writers still feed open P0 reconciliation work. | blocked on polylogue-w6hql/polylogue-lr6dx clearing before this is an executable drop |
| `source.raw_authority_census_plans.ordinal` | PURGE | Table-level PURGE-pending-execution inherited from the 2026-08-04 seed; not re-derived at column granularity this pass per the team-lead's cite-don't-re-derive instruction for delegated tables. Current writers still feed open P0 reconciliation work. | blocked on polylogue-w6hql/polylogue-lr6dx clearing before this is an executable drop |
| `source.raw_authority_census_plans.selected` | PURGE | Table-level PURGE-pending-execution inherited from the 2026-08-04 seed; not re-derived at column granularity this pass per the team-lead's cite-don't-re-derive instruction for delegated tables. Current writers still feed open P0 reconciliation work. | blocked on polylogue-w6hql/polylogue-lr6dx clearing before this is an executable drop |
| `source.raw_authority_census_plans.outcome_status` | PURGE | Table-level PURGE-pending-execution inherited from the 2026-08-04 seed; not re-derived at column granularity this pass per the team-lead's cite-don't-re-derive instruction for delegated tables. Current writers still feed open P0 reconciliation work. | blocked on polylogue-w6hql/polylogue-lr6dx clearing before this is an executable drop |
| `source.raw_authority_census_plans.reason` | PURGE | Table-level PURGE-pending-execution inherited from the 2026-08-04 seed; not re-derived at column granularity this pass per the team-lead's cite-don't-re-derive instruction for delegated tables. Current writers still feed open P0 reconciliation work. | blocked on polylogue-w6hql/polylogue-lr6dx clearing before this is an executable drop |
| `source.raw_authority_census_plans.next_action` | PURGE | Table-level PURGE-pending-execution inherited from the 2026-08-04 seed; not re-derived at column granularity this pass per the team-lead's cite-don't-re-derive instruction for delegated tables. Current writers still feed open P0 reconciliation work. | blocked on polylogue-w6hql/polylogue-lr6dx clearing before this is an executable drop |
| `source.raw_authority_census_plans.application_receipt_json` | PURGE | Table-level PURGE-pending-execution inherited from the 2026-08-04 seed; not re-derived at column granularity this pass per the team-lead's cite-don't-re-derive instruction for delegated tables. Current writers still feed open P0 reconciliation work. | blocked on polylogue-w6hql/polylogue-lr6dx clearing before this is an executable drop |
| `source.raw_authority_census_plans.outcome_recorded` | PURGE | Table-level PURGE-pending-execution inherited from the 2026-08-04 seed; not re-derived at column granularity this pass per the team-lead's cite-don't-re-derive instruction for delegated tables. Current writers still feed open P0 reconciliation work. | blocked on polylogue-w6hql/polylogue-lr6dx clearing before this is an executable drop |
| `source.raw_authority_census_plans.recorded_at_ms` | PURGE | Table-level PURGE-pending-execution inherited from the 2026-08-04 seed; not re-derived at column granularity this pass per the team-lead's cite-don't-re-derive instruction for delegated tables. Current writers still feed open P0 reconciliation work. | blocked on polylogue-w6hql/polylogue-lr6dx clearing before this is an executable drop |
| `source.raw_authority_census_post_plans.census_id` | PURGE | Table-level PURGE-pending-execution inherited from the 2026-08-04 seed; not re-derived at column granularity this pass per the team-lead's cite-don't-re-derive instruction for delegated tables. Current writers still feed open P0 reconciliation work. | blocked on polylogue-w6hql/polylogue-lr6dx clearing before this is an executable drop |
| `source.raw_authority_census_post_plans.plan_id` | PURGE | Table-level PURGE-pending-execution inherited from the 2026-08-04 seed; not re-derived at column granularity this pass per the team-lead's cite-don't-re-derive instruction for delegated tables. Current writers still feed open P0 reconciliation work. | blocked on polylogue-w6hql/polylogue-lr6dx clearing before this is an executable drop |
| `source.raw_authority_census_post_plans.ordinal` | PURGE | Table-level PURGE-pending-execution inherited from the 2026-08-04 seed; not re-derived at column granularity this pass per the team-lead's cite-don't-re-derive instruction for delegated tables. Current writers still feed open P0 reconciliation work. | blocked on polylogue-w6hql/polylogue-lr6dx clearing before this is an executable drop |
| `source.raw_authority_censuses.census_id` | PURGE | Table-level PURGE-pending-execution inherited from the 2026-08-04 seed; not re-derived at column granularity this pass per the team-lead's cite-don't-re-derive instruction for delegated tables. Current writers still feed open P0 reconciliation work. | blocked on polylogue-w6hql/polylogue-lr6dx clearing before this is an executable drop |
| `source.raw_authority_censuses.sequence_no` | PURGE | Table-level PURGE-pending-execution inherited from the 2026-08-04 seed; not re-derived at column granularity this pass per the team-lead's cite-don't-re-derive instruction for delegated tables. Current writers still feed open P0 reconciliation work. | blocked on polylogue-w6hql/polylogue-lr6dx clearing before this is an executable drop |
| `source.raw_authority_censuses.scope_json` | PURGE | Table-level PURGE-pending-execution inherited from the 2026-08-04 seed; not re-derived at column granularity this pass per the team-lead's cite-don't-re-derive instruction for delegated tables. Current writers still feed open P0 reconciliation work. | blocked on polylogue-w6hql/polylogue-lr6dx clearing before this is an executable drop |
| `source.raw_authority_censuses.residual_json` | PURGE | Table-level PURGE-pending-execution inherited from the 2026-08-04 seed; not re-derived at column granularity this pass per the team-lead's cite-don't-re-derive instruction for delegated tables. Current writers still feed open P0 reconciliation work. | blocked on polylogue-w6hql/polylogue-lr6dx clearing before this is an executable drop |
| `source.raw_authority_censuses.parser_fingerprint` | PURGE | Table-level PURGE-pending-execution inherited from the 2026-08-04 seed; not re-derived at column granularity this pass per the team-lead's cite-don't-re-derive instruction for delegated tables. Current writers still feed open P0 reconciliation work. | blocked on polylogue-w6hql/polylogue-lr6dx clearing before this is an executable drop |
| `source.raw_authority_censuses.mode` | PURGE | Table-level PURGE-pending-execution inherited from the 2026-08-04 seed; not re-derived at column granularity this pass per the team-lead's cite-don't-re-derive instruction for delegated tables. Current writers still feed open P0 reconciliation work. | blocked on polylogue-w6hql/polylogue-lr6dx clearing before this is an executable drop |
| `source.raw_authority_censuses.lifecycle_status` | PURGE | Table-level PURGE-pending-execution inherited from the 2026-08-04 seed; not re-derived at column granularity this pass per the team-lead's cite-don't-re-derive instruction for delegated tables. Current writers still feed open P0 reconciliation work. | blocked on polylogue-w6hql/polylogue-lr6dx clearing before this is an executable drop |
| `source.raw_authority_censuses.quiescent` | PURGE | Table-level PURGE-pending-execution inherited from the 2026-08-04 seed; not re-derived at column granularity this pass per the team-lead's cite-don't-re-derive instruction for delegated tables. Current writers still feed open P0 reconciliation work. | blocked on polylogue-w6hql/polylogue-lr6dx clearing before this is an executable drop |
| `source.raw_authority_censuses.inventory_digest` | PURGE | Table-level PURGE-pending-execution inherited from the 2026-08-04 seed; not re-derived at column granularity this pass per the team-lead's cite-don't-re-derive instruction for delegated tables. Current writers still feed open P0 reconciliation work. | blocked on polylogue-w6hql/polylogue-lr6dx clearing before this is an executable drop |
| `source.raw_authority_censuses.residual_digest` | PURGE | Table-level PURGE-pending-execution inherited from the 2026-08-04 seed; not re-derived at column granularity this pass per the team-lead's cite-don't-re-derive instruction for delegated tables. Current writers still feed open P0 reconciliation work. | blocked on polylogue-w6hql/polylogue-lr6dx clearing before this is an executable drop |
| `source.raw_authority_censuses.plan_count` | PURGE | Table-level PURGE-pending-execution inherited from the 2026-08-04 seed; not re-derived at column granularity this pass per the team-lead's cite-don't-re-derive instruction for delegated tables. Current writers still feed open P0 reconciliation work. | blocked on polylogue-w6hql/polylogue-lr6dx clearing before this is an executable drop |
| `source.raw_authority_censuses.post_inventory_digest` | PURGE | Table-level PURGE-pending-execution inherited from the 2026-08-04 seed; not re-derived at column granularity this pass per the team-lead's cite-don't-re-derive instruction for delegated tables. Current writers still feed open P0 reconciliation work. | blocked on polylogue-w6hql/polylogue-lr6dx clearing before this is an executable drop |
| `source.raw_authority_censuses.post_residual_json` | PURGE | Table-level PURGE-pending-execution inherited from the 2026-08-04 seed; not re-derived at column granularity this pass per the team-lead's cite-don't-re-derive instruction for delegated tables. Current writers still feed open P0 reconciliation work. | blocked on polylogue-w6hql/polylogue-lr6dx clearing before this is an executable drop |
| `source.raw_authority_censuses.post_residual_digest` | PURGE | Table-level PURGE-pending-execution inherited from the 2026-08-04 seed; not re-derived at column granularity this pass per the team-lead's cite-don't-re-derive instruction for delegated tables. Current writers still feed open P0 reconciliation work. | blocked on polylogue-w6hql/polylogue-lr6dx clearing before this is an executable drop |
| `source.raw_authority_censuses.post_plan_count` | PURGE | Table-level PURGE-pending-execution inherited from the 2026-08-04 seed; not re-derived at column granularity this pass per the team-lead's cite-don't-re-derive instruction for delegated tables. Current writers still feed open P0 reconciliation work. | blocked on polylogue-w6hql/polylogue-lr6dx clearing before this is an executable drop |
| `source.raw_authority_censuses.postflight_at_ms` | PURGE | Table-level PURGE-pending-execution inherited from the 2026-08-04 seed; not re-derived at column granularity this pass per the team-lead's cite-don't-re-derive instruction for delegated tables. Current writers still feed open P0 reconciliation work. | blocked on polylogue-w6hql/polylogue-lr6dx clearing before this is an executable drop |
| `source.raw_authority_censuses.executable_plan_count` | PURGE | Table-level PURGE-pending-execution inherited from the 2026-08-04 seed; not re-derived at column granularity this pass per the team-lead's cite-don't-re-derive instruction for delegated tables. Current writers still feed open P0 reconciliation work. | blocked on polylogue-w6hql/polylogue-lr6dx clearing before this is an executable drop |
| `source.raw_authority_censuses.residual_plan_count` | PURGE | Table-level PURGE-pending-execution inherited from the 2026-08-04 seed; not re-derived at column granularity this pass per the team-lead's cite-don't-re-derive instruction for delegated tables. Current writers still feed open P0 reconciliation work. | blocked on polylogue-w6hql/polylogue-lr6dx clearing before this is an executable drop |
| `source.raw_authority_censuses.predecessor_census_id` | PURGE | Table-level PURGE-pending-execution inherited from the 2026-08-04 seed; not re-derived at column granularity this pass per the team-lead's cite-don't-re-derive instruction for delegated tables. Current writers still feed open P0 reconciliation work. | blocked on polylogue-w6hql/polylogue-lr6dx clearing before this is an executable drop |
| `source.raw_authority_censuses.fixed_point` | PURGE | Table-level PURGE-pending-execution inherited from the 2026-08-04 seed; not re-derived at column granularity this pass per the team-lead's cite-don't-re-derive instruction for delegated tables. Current writers still feed open P0 reconciliation work. | blocked on polylogue-w6hql/polylogue-lr6dx clearing before this is an executable drop |
| `source.raw_authority_censuses.created_at_ms` | PURGE | Table-level PURGE-pending-execution inherited from the 2026-08-04 seed; not re-derived at column granularity this pass per the team-lead's cite-don't-re-derive instruction for delegated tables. Current writers still feed open P0 reconciliation work. | blocked on polylogue-w6hql/polylogue-lr6dx clearing before this is an executable drop |
| `source.raw_authority_censuses.completed_at_ms` | PURGE | Table-level PURGE-pending-execution inherited from the 2026-08-04 seed; not re-derived at column granularity this pass per the team-lead's cite-don't-re-derive instruction for delegated tables. Current writers still feed open P0 reconciliation work. | blocked on polylogue-w6hql/polylogue-lr6dx clearing before this is an executable drop |
| `source.raw_authority_parser_census.raw_id` | PURGE | Table-level PURGE-pending-execution inherited from the 2026-08-04 seed; not re-derived at column granularity this pass per the team-lead's cite-don't-re-derive instruction for delegated tables. Current writers still feed open P0 reconciliation work. | blocked on polylogue-w6hql/polylogue-lr6dx clearing before this is an executable drop |
| `source.raw_authority_parser_census.parser_fingerprint` | PURGE | Table-level PURGE-pending-execution inherited from the 2026-08-04 seed; not re-derived at column granularity this pass per the team-lead's cite-don't-re-derive instruction for delegated tables. Current writers still feed open P0 reconciliation work. | blocked on polylogue-w6hql/polylogue-lr6dx clearing before this is an executable drop |
| `source.raw_authority_parser_census.status` | PURGE | Table-level PURGE-pending-execution inherited from the 2026-08-04 seed; not re-derived at column granularity this pass per the team-lead's cite-don't-re-derive instruction for delegated tables. Current writers still feed open P0 reconciliation work. | blocked on polylogue-w6hql/polylogue-lr6dx clearing before this is an executable drop |
| `source.raw_authority_parser_census.logical_keys_json` | PURGE | Table-level PURGE-pending-execution inherited from the 2026-08-04 seed; not re-derived at column granularity this pass per the team-lead's cite-don't-re-derive instruction for delegated tables. Current writers still feed open P0 reconciliation work. | blocked on polylogue-w6hql/polylogue-lr6dx clearing before this is an executable drop |
| `source.raw_authority_parser_census.detail` | PURGE | Table-level PURGE-pending-execution inherited from the 2026-08-04 seed; not re-derived at column granularity this pass per the team-lead's cite-don't-re-derive instruction for delegated tables. Current writers still feed open P0 reconciliation work. | blocked on polylogue-w6hql/polylogue-lr6dx clearing before this is an executable drop |
| `source.raw_authority_parser_census.censused_at_ms` | PURGE | Table-level PURGE-pending-execution inherited from the 2026-08-04 seed; not re-derived at column granularity this pass per the team-lead's cite-don't-re-derive instruction for delegated tables. Current writers still feed open P0 reconciliation work. | blocked on polylogue-w6hql/polylogue-lr6dx clearing before this is an executable drop |
| `source.raw_authority_plans.plan_id` | PURGE | Table-level PURGE-pending-execution inherited from the 2026-08-04 seed; not re-derived at column granularity this pass per the team-lead's cite-don't-re-derive instruction for delegated tables. Current writers still feed open P0 reconciliation work. | blocked on polylogue-w6hql/polylogue-lr6dx clearing before this is an executable drop |
| `source.raw_authority_plans.input_digest` | PURGE | Table-level PURGE-pending-execution inherited from the 2026-08-04 seed; not re-derived at column granularity this pass per the team-lead's cite-don't-re-derive instruction for delegated tables. Current writers still feed open P0 reconciliation work. | blocked on polylogue-w6hql/polylogue-lr6dx clearing before this is an executable drop |
| `source.raw_authority_plans.input_raw_ids_json` | PURGE | Table-level PURGE-pending-execution inherited from the 2026-08-04 seed; not re-derived at column granularity this pass per the team-lead's cite-don't-re-derive instruction for delegated tables. Current writers still feed open P0 reconciliation work. | blocked on polylogue-w6hql/polylogue-lr6dx clearing before this is an executable drop |
| `source.raw_authority_plans.logical_keys_json` | PURGE | Table-level PURGE-pending-execution inherited from the 2026-08-04 seed; not re-derived at column granularity this pass per the team-lead's cite-don't-re-derive instruction for delegated tables. Current writers still feed open P0 reconciliation work. | blocked on polylogue-w6hql/polylogue-lr6dx clearing before this is an executable drop |
| `source.raw_authority_plans.authority_witness_json` | PURGE | Table-level PURGE-pending-execution inherited from the 2026-08-04 seed; not re-derived at column granularity this pass per the team-lead's cite-don't-re-derive instruction for delegated tables. Current writers still feed open P0 reconciliation work. | blocked on polylogue-w6hql/polylogue-lr6dx clearing before this is an executable drop |
| `source.raw_authority_plans.source_preconditions_json` | PURGE | Table-level PURGE-pending-execution inherited from the 2026-08-04 seed; not re-derived at column granularity this pass per the team-lead's cite-don't-re-derive instruction for delegated tables. Current writers still feed open P0 reconciliation work. | blocked on polylogue-w6hql/polylogue-lr6dx clearing before this is an executable drop |
| `source.raw_authority_plans.index_preconditions_json` | PURGE | Table-level PURGE-pending-execution inherited from the 2026-08-04 seed; not re-derived at column granularity this pass per the team-lead's cite-don't-re-derive instruction for delegated tables. Current writers still feed open P0 reconciliation work. | blocked on polylogue-w6hql/polylogue-lr6dx clearing before this is an executable drop |
| `source.raw_authority_plans.created_at_ms` | PURGE | Table-level PURGE-pending-execution inherited from the 2026-08-04 seed; not re-derived at column granularity this pass per the team-lead's cite-don't-re-derive instruction for delegated tables. Current writers still feed open P0 reconciliation work. | blocked on polylogue-w6hql/polylogue-lr6dx clearing before this is an executable drop |
| `source.raw_authority_verdicts.raw_id` | KEEP | raw_authority_verdict_cache.py, blob_gc.py (seed-carried) | n/a |
| `source.raw_authority_verdicts.logical_source_key` | KEEP | raw_authority_verdict_cache.py, blob_gc.py (seed-carried) | n/a |
| `source.raw_authority_verdicts.verdict` | KEEP | raw_authority_verdict_cache.py, blob_gc.py (seed-carried) | n/a |
| `source.raw_authority_verdicts.cohort_member_count` | KEEP | raw_authority_verdict_cache.py, blob_gc.py (seed-carried) | n/a |
| `source.raw_authority_verdicts.cohort_fingerprint` | KEEP | raw_authority_verdict_cache.py, blob_gc.py (seed-carried) | n/a |
| `source.raw_authority_verdicts.computed_at_ms` | KEEP | raw_authority_verdict_cache.py, blob_gc.py (seed-carried) | n/a |
| `source.raw_byte_duplicate_supersession_receipts.raw_id` | KEEP | seed-carried, re-confirmed | n/a |
| `source.raw_byte_duplicate_supersession_receipts.blob_hash` | KEEP | seed-carried, re-confirmed | n/a |
| `source.raw_byte_duplicate_supersession_receipts.blob_size` | KEEP | seed-carried, re-confirmed | n/a |
| `source.raw_byte_duplicate_supersession_receipts.duplicate_of_raw_id` | KEEP | seed-carried, re-confirmed | n/a |
| `source.raw_byte_duplicate_supersession_receipts.duplicate_of_session_id` | KEEP | seed-carried, re-confirmed | n/a |
| `source.raw_byte_duplicate_supersession_receipts.previous_revision_authority` | KEEP | seed-carried, re-confirmed | n/a |
| `source.raw_byte_duplicate_supersession_receipts.promoted_at_ms` | KEEP | seed-carried, re-confirmed | n/a |
| `source.raw_byte_duplicate_supersession_receipts.tool_version` | KEEP | seed-carried, re-confirmed | n/a |
| `source.raw_byte_duplicate_supersession_receipts.backup_manifest_path` | KEEP | seed-carried, re-confirmed | n/a |
| `source.raw_byte_duplicate_supersession_receipts.detail` | KEEP | seed-carried, re-confirmed | n/a |
| `source.raw_capture_observations.raw_id` | KEEP | polylogue/storage/source_write.py (seed-carried, re-confirmed) | n/a |
| `source.raw_capture_observations.capture_mode` | KEEP | polylogue/storage/source_write.py (seed-carried, re-confirmed) | n/a |
| `source.raw_capture_observations.first_observed_at_ms` | KEEP | polylogue/storage/source_write.py (seed-carried, re-confirmed) | n/a |
| `source.raw_container_coordinates.raw_id` | KEEP | polylogue/storage/blob_integrity.py:1058-1064,1709-1712 (zip-container coordinate join and record path) | n/a |
| `source.raw_container_coordinates.coordinate_format` | KEEP | polylogue/storage/blob_integrity.py:1058-1064,1709-1712 (zip-container coordinate join and record path) | n/a |
| `source.raw_container_coordinates.entry_ordinal` | KEEP | polylogue/storage/blob_integrity.py:1058-1064,1709-1712 (zip-container coordinate join and record path) | n/a |
| `source.raw_container_coordinates.split_index` | KEEP | polylogue/storage/blob_integrity.py:1058-1064,1709-1712 (zip-container coordinate join and record path) | n/a |
| `source.raw_failure_disposition_receipts.raw_id` | KEEP | polylogue/maintenance/raw_failure_disposition_apply.py:116-191 (SELECT r.raw_id, r.origin, r.source_path, ... EXISTS (SELECT 1 FROM raw_failure_disposition_receipts ...)) | n/a |
| `source.raw_failure_disposition_receipts.artifact_id` | KEEP | polylogue/maintenance/raw_failure_disposition_apply.py:116-191 (SELECT r.raw_id, r.origin, r.source_path, ... EXISTS (SELECT 1 FROM raw_failure_disposition_receipts ...)) | n/a |
| `source.raw_failure_disposition_receipts.origin` | KEEP | polylogue/maintenance/raw_failure_disposition_apply.py:116-191 (SELECT r.raw_id, r.origin, r.source_path, ... EXISTS (SELECT 1 FROM raw_failure_disposition_receipts ...)) | n/a |
| `source.raw_failure_disposition_receipts.source_path` | KEEP | polylogue/maintenance/raw_failure_disposition_apply.py:116-191 (SELECT r.raw_id, r.origin, r.source_path, ... EXISTS (SELECT 1 FROM raw_failure_disposition_receipts ...)) | n/a |
| `source.raw_failure_disposition_receipts.source_index` | KEEP | polylogue/maintenance/raw_failure_disposition_apply.py:116-191 (SELECT r.raw_id, r.origin, r.source_path, ... EXISTS (SELECT 1 FROM raw_failure_disposition_receipts ...)) | n/a |
| `source.raw_failure_disposition_receipts.blob_hash` | KEEP | polylogue/maintenance/raw_failure_disposition_apply.py:116-191 (SELECT r.raw_id, r.origin, r.source_path, ... EXISTS (SELECT 1 FROM raw_failure_disposition_receipts ...)) | n/a |
| `source.raw_failure_disposition_receipts.blob_size` | KEEP | polylogue/maintenance/raw_failure_disposition_apply.py:116-191 (SELECT r.raw_id, r.origin, r.source_path, ... EXISTS (SELECT 1 FROM raw_failure_disposition_receipts ...)) | n/a |
| `source.raw_failure_disposition_receipts.previous_parse_error` | KEEP | polylogue/maintenance/raw_failure_disposition_apply.py:116-191 (SELECT r.raw_id, r.origin, r.source_path, ... EXISTS (SELECT 1 FROM raw_failure_disposition_receipts ...)) | n/a |
| `source.raw_failure_disposition_receipts.previous_validation_status` | KEEP | polylogue/maintenance/raw_failure_disposition_apply.py:116-191 (SELECT r.raw_id, r.origin, r.source_path, ... EXISTS (SELECT 1 FROM raw_failure_disposition_receipts ...)) | n/a |
| `source.raw_failure_disposition_receipts.previous_artifact_kind` | KEEP | polylogue/maintenance/raw_failure_disposition_apply.py:116-191 (SELECT r.raw_id, r.origin, r.source_path, ... EXISTS (SELECT 1 FROM raw_failure_disposition_receipts ...)) | n/a |
| `source.raw_failure_disposition_receipts.previous_support_status` | KEEP | polylogue/maintenance/raw_failure_disposition_apply.py:116-191 (SELECT r.raw_id, r.origin, r.source_path, ... EXISTS (SELECT 1 FROM raw_failure_disposition_receipts ...)) | n/a |
| `source.raw_failure_disposition_receipts.previous_classification_reason` | KEEP | polylogue/maintenance/raw_failure_disposition_apply.py:116-191 (SELECT r.raw_id, r.origin, r.source_path, ... EXISTS (SELECT 1 FROM raw_failure_disposition_receipts ...)) | n/a |
| `source.raw_failure_disposition_receipts.disposition_kind` | KEEP | polylogue/maintenance/raw_failure_disposition_apply.py:116-191 (SELECT r.raw_id, r.origin, r.source_path, ... EXISTS (SELECT 1 FROM raw_failure_disposition_receipts ...)) | n/a |
| `source.raw_failure_disposition_receipts.manifest_sha256` | KEEP | polylogue/maintenance/raw_failure_disposition_apply.py:116-191 (SELECT r.raw_id, r.origin, r.source_path, ... EXISTS (SELECT 1 FROM raw_failure_disposition_receipts ...)) | n/a |
| `source.raw_failure_disposition_receipts.disposed_at_ms` | KEEP | polylogue/maintenance/raw_failure_disposition_apply.py:116-191 (SELECT r.raw_id, r.origin, r.source_path, ... EXISTS (SELECT 1 FROM raw_failure_disposition_receipts ...)) | n/a |
| `source.raw_failure_disposition_receipts.tool_version` | KEEP | polylogue/maintenance/raw_failure_disposition_apply.py:116-191 (SELECT r.raw_id, r.origin, r.source_path, ... EXISTS (SELECT 1 FROM raw_failure_disposition_receipts ...)) | n/a |
| `source.raw_failure_disposition_receipts.backup_manifest_path` | KEEP | polylogue/maintenance/raw_failure_disposition_apply.py:116-191 (SELECT r.raw_id, r.origin, r.source_path, ... EXISTS (SELECT 1 FROM raw_failure_disposition_receipts ...)) | n/a |
| `source.raw_failure_disposition_receipts.detail` | KEEP | polylogue/maintenance/raw_failure_disposition_apply.py:116-191 (SELECT r.raw_id, r.origin, r.source_path, ... EXISTS (SELECT 1 FROM raw_failure_disposition_receipts ...)) | n/a |
| `source.raw_hook_events.hook_event_id` | KEEP | polylogue/maintenance/hook_payload_ref_reconciliation_receipt.py:195, hook_payload_ref_reconciliation_apply.py:220, polylogue/storage/hook_payload_ref_reconciliation.py:277,287 all SELECT/UPDATE it directly, incl. blob_hash | n/a |
| `source.raw_hook_events.origin` | KEEP | polylogue/maintenance/hook_payload_ref_reconciliation_receipt.py:195, hook_payload_ref_reconciliation_apply.py:220, polylogue/storage/hook_payload_ref_reconciliation.py:277,287 all SELECT/UPDATE it directly, incl. blob_hash | n/a |
| `source.raw_hook_events.native_id` | KEEP | polylogue/maintenance/hook_payload_ref_reconciliation_receipt.py:195, hook_payload_ref_reconciliation_apply.py:220, polylogue/storage/hook_payload_ref_reconciliation.py:277,287 all SELECT/UPDATE it directly, incl. blob_hash | n/a |
| `source.raw_hook_events.session_native_id` | KEEP | polylogue/maintenance/hook_payload_ref_reconciliation_receipt.py:195, hook_payload_ref_reconciliation_apply.py:220, polylogue/storage/hook_payload_ref_reconciliation.py:277,287 all SELECT/UPDATE it directly, incl. blob_hash | n/a |
| `source.raw_hook_events.source_path` | KEEP | polylogue/maintenance/hook_payload_ref_reconciliation_receipt.py:195, hook_payload_ref_reconciliation_apply.py:220, polylogue/storage/hook_payload_ref_reconciliation.py:277,287 all SELECT/UPDATE it directly, incl. blob_hash | n/a |
| `source.raw_hook_events.event_type` | KEEP | polylogue/maintenance/hook_payload_ref_reconciliation_receipt.py:195, hook_payload_ref_reconciliation_apply.py:220, polylogue/storage/hook_payload_ref_reconciliation.py:277,287 all SELECT/UPDATE it directly, incl. blob_hash | n/a |
| `source.raw_hook_events.payload_json` | KEEP | polylogue/maintenance/hook_payload_ref_reconciliation_receipt.py:195, hook_payload_ref_reconciliation_apply.py:220, polylogue/storage/hook_payload_ref_reconciliation.py:277,287 all SELECT/UPDATE it directly, incl. blob_hash | n/a |
| `source.raw_hook_events.observed_at_ms` | KEEP | polylogue/maintenance/hook_payload_ref_reconciliation_receipt.py:195, hook_payload_ref_reconciliation_apply.py:220, polylogue/storage/hook_payload_ref_reconciliation.py:277,287 all SELECT/UPDATE it directly, incl. blob_hash | n/a |
| `source.raw_hook_events.blob_hash` | KEEP | polylogue/maintenance/hook_payload_ref_reconciliation_receipt.py:195, hook_payload_ref_reconciliation_apply.py:220, polylogue/storage/hook_payload_ref_reconciliation.py:277,287 all SELECT/UPDATE it directly, incl. blob_hash | n/a |
| `source.raw_live_source_reconciliation_receipts.raw_id` | KEEP | seed-carried, re-confirmed | n/a |
| `source.raw_live_source_reconciliation_receipts.verdict` | KEEP | seed-carried, re-confirmed | n/a |
| `source.raw_live_source_reconciliation_receipts.previous_revision_authority` | KEEP | seed-carried, re-confirmed | n/a |
| `source.raw_live_source_reconciliation_receipts.source_path` | KEEP | seed-carried, re-confirmed | n/a |
| `source.raw_live_source_reconciliation_receipts.blob_hash` | KEEP | seed-carried, re-confirmed | n/a |
| `source.raw_live_source_reconciliation_receipts.blob_size` | KEEP | seed-carried, re-confirmed | n/a |
| `source.raw_live_source_reconciliation_receipts.compared_at_ms` | KEEP | seed-carried, re-confirmed | n/a |
| `source.raw_live_source_reconciliation_receipts.tool_version` | KEEP | seed-carried, re-confirmed | n/a |
| `source.raw_live_source_reconciliation_receipts.backup_manifest_path` | KEEP | seed-carried, re-confirmed | n/a |
| `source.raw_live_source_reconciliation_receipts.detail` | KEEP | seed-carried, re-confirmed | n/a |
| `source.raw_membership_census.raw_id` | PURGE | Table-level PURGE-pending-execution inherited from the 2026-08-04 seed; not re-derived at column granularity this pass per the team-lead's cite-don't-re-derive instruction for delegated tables. Current writers still feed open P0 reconciliation work. | blocked on polylogue-w6hql/polylogue-lr6dx clearing before this is an executable drop |
| `source.raw_membership_census.parser_fingerprint` | PURGE | Table-level PURGE-pending-execution inherited from the 2026-08-04 seed; not re-derived at column granularity this pass per the team-lead's cite-don't-re-derive instruction for delegated tables. Current writers still feed open P0 reconciliation work. | blocked on polylogue-w6hql/polylogue-lr6dx clearing before this is an executable drop |
| `source.raw_membership_census.status` | PURGE | Table-level PURGE-pending-execution inherited from the 2026-08-04 seed; not re-derived at column granularity this pass per the team-lead's cite-don't-re-derive instruction for delegated tables. Current writers still feed open P0 reconciliation work. | blocked on polylogue-w6hql/polylogue-lr6dx clearing before this is an executable drop |
| `source.raw_membership_census.member_count` | PURGE | Table-level PURGE-pending-execution inherited from the 2026-08-04 seed; not re-derived at column granularity this pass per the team-lead's cite-don't-re-derive instruction for delegated tables. Current writers still feed open P0 reconciliation work. | blocked on polylogue-w6hql/polylogue-lr6dx clearing before this is an executable drop |
| `source.raw_membership_census.censused_at_ms` | PURGE | Table-level PURGE-pending-execution inherited from the 2026-08-04 seed; not re-derived at column granularity this pass per the team-lead's cite-don't-re-derive instruction for delegated tables. Current writers still feed open P0 reconciliation work. | blocked on polylogue-w6hql/polylogue-lr6dx clearing before this is an executable drop |
| `source.raw_membership_census.detail` | PURGE | Table-level PURGE-pending-execution inherited from the 2026-08-04 seed; not re-derived at column granularity this pass per the team-lead's cite-don't-re-derive instruction for delegated tables. Current writers still feed open P0 reconciliation work. | blocked on polylogue-w6hql/polylogue-lr6dx clearing before this is an executable drop |
| `source.raw_membership_writeback_receipts.raw_id` | KEEP | seed-carried, re-confirmed | n/a |
| `source.raw_membership_writeback_receipts.logical_source_key` | KEEP | seed-carried, re-confirmed | n/a |
| `source.raw_membership_writeback_receipts.provider_session_id` | KEEP | seed-carried, re-confirmed | n/a |
| `source.raw_membership_writeback_receipts.membership_decision` | KEEP | seed-carried, re-confirmed | n/a |
| `source.raw_membership_writeback_receipts.previous_revision_authority` | KEEP | seed-carried, re-confirmed | n/a |
| `source.raw_membership_writeback_receipts.promoted_at_ms` | KEEP | seed-carried, re-confirmed | n/a |
| `source.raw_membership_writeback_receipts.tool_version` | KEEP | seed-carried, re-confirmed | n/a |
| `source.raw_membership_writeback_receipts.backup_manifest_path` | KEEP | seed-carried, re-confirmed | n/a |
| `source.raw_membership_writeback_receipts.detail` | KEEP | seed-carried, re-confirmed | n/a |
| `source.raw_non_session_duplicate_exclusion_receipts.raw_id` | KEEP | polylogue/maintenance/schema_inference_gate.py:288,327-338 (SELECT blob_hash, blob_size, indexed_twin_raw_id, ... FROM raw_non_session_duplicate_exclusion_receipts) | n/a |
| `source.raw_non_session_duplicate_exclusion_receipts.blob_hash` | KEEP | polylogue/maintenance/schema_inference_gate.py:288,327-338 (SELECT blob_hash, blob_size, indexed_twin_raw_id, ... FROM raw_non_session_duplicate_exclusion_receipts) | n/a |
| `source.raw_non_session_duplicate_exclusion_receipts.blob_size` | KEEP | polylogue/maintenance/schema_inference_gate.py:288,327-338 (SELECT blob_hash, blob_size, indexed_twin_raw_id, ... FROM raw_non_session_duplicate_exclusion_receipts) | n/a |
| `source.raw_non_session_duplicate_exclusion_receipts.indexed_twin_raw_id` | KEEP | polylogue/maintenance/schema_inference_gate.py:288,327-338 (SELECT blob_hash, blob_size, indexed_twin_raw_id, ... FROM raw_non_session_duplicate_exclusion_receipts) | n/a |
| `source.raw_non_session_duplicate_exclusion_receipts.indexed_twin_session_id` | KEEP | polylogue/maintenance/schema_inference_gate.py:288,327-338 (SELECT blob_hash, blob_size, indexed_twin_raw_id, ... FROM raw_non_session_duplicate_exclusion_receipts) | n/a |
| `source.raw_non_session_duplicate_exclusion_receipts.parser_fingerprint` | KEEP | polylogue/maintenance/schema_inference_gate.py:288,327-338 (SELECT blob_hash, blob_size, indexed_twin_raw_id, ... FROM raw_non_session_duplicate_exclusion_receipts) | n/a |
| `source.raw_non_session_duplicate_exclusion_receipts.excluded_at_ms` | KEEP | polylogue/maintenance/schema_inference_gate.py:288,327-338 (SELECT blob_hash, blob_size, indexed_twin_raw_id, ... FROM raw_non_session_duplicate_exclusion_receipts) | n/a |
| `source.raw_non_session_duplicate_exclusion_receipts.tool_version` | KEEP | polylogue/maintenance/schema_inference_gate.py:288,327-338 (SELECT blob_hash, blob_size, indexed_twin_raw_id, ... FROM raw_non_session_duplicate_exclusion_receipts) | n/a |
| `source.raw_non_session_duplicate_exclusion_receipts.detail` | KEEP | polylogue/maintenance/schema_inference_gate.py:288,327-338 (SELECT blob_hash, blob_size, indexed_twin_raw_id, ... FROM raw_non_session_duplicate_exclusion_receipts) | n/a |
| `source.raw_quarantine_group_dedup_receipts.raw_id` | KEEP | seed-carried, re-confirmed | n/a |
| `source.raw_quarantine_group_dedup_receipts.source_path` | KEEP | seed-carried, re-confirmed | n/a |
| `source.raw_quarantine_group_dedup_receipts.blob_hash` | KEEP | seed-carried, re-confirmed | n/a |
| `source.raw_quarantine_group_dedup_receipts.blob_size` | KEEP | seed-carried, re-confirmed | n/a |
| `source.raw_quarantine_group_dedup_receipts.representative_raw_id` | KEEP | seed-carried, re-confirmed | n/a |
| `source.raw_quarantine_group_dedup_receipts.representative_session_id` | KEEP | seed-carried, re-confirmed | n/a |
| `source.raw_quarantine_group_dedup_receipts.promoted_at_ms` | KEEP | seed-carried, re-confirmed | n/a |
| `source.raw_quarantine_group_dedup_receipts.tool_version` | KEEP | seed-carried, re-confirmed | n/a |
| `source.raw_quarantine_group_dedup_receipts.backup_manifest_path` | KEEP | seed-carried, re-confirmed | n/a |
| `source.raw_quarantine_group_dedup_receipts.detail` | KEEP | seed-carried, re-confirmed | n/a |
| `source.raw_session_memberships.raw_id` | KEEP | active reconciler consumer (seed-carried) | n/a |
| `source.raw_session_memberships.logical_source_key` | KEEP | active reconciler consumer (seed-carried) | n/a |
| `source.raw_session_memberships.provider_session_id` | KEEP | active reconciler consumer (seed-carried) | n/a |
| `source.raw_session_memberships.source_revision` | KEEP | active reconciler consumer (seed-carried) | n/a |
| `source.raw_session_memberships.normalized_content_hash` | KEEP | active reconciler consumer (seed-carried) | n/a |
| `source.raw_session_memberships.message_count` | KEEP | active reconciler consumer (seed-carried) | n/a |
| `source.raw_session_memberships.predecessor_raw_id` | KEEP | active reconciler consumer (seed-carried) | n/a |
| `source.raw_session_memberships.acquisition_generation` | KEEP | active reconciler consumer (seed-carried) | n/a |
| `source.raw_session_memberships.revision_authority` | KEEP | active reconciler consumer (seed-carried) | n/a |
| `source.raw_session_memberships.decision` | KEEP | active reconciler consumer (seed-carried) | n/a |
| `source.raw_session_memberships.decided_at_ms` | KEEP | active reconciler consumer (seed-carried) | n/a |
| `source.raw_sessions.raw_id` | KEEP | PK; used everywhere (raw_reads.py, raw_state.py, blob_gc.py) | n/a |
| `source.raw_sessions.origin` | KEEP | idx_raw_sessions_origin; filtered constantly in queries/raw_reads.py, daemon/status.py | n/a |
| `source.raw_sessions.capture_mode` | KEEP | raw_capture_observations dedup logic, source_write.py | n/a |
| `source.raw_sessions.native_id` | KEEP | idx_raw_sessions_origin_native; dedup/dispatch lookups | n/a |
| `source.raw_sessions.source_path` | KEEP | idx_raw_sessions_source_path; import_explain.py, repair.py | n/a |
| `source.raw_sessions.source_index` | KEEP | same index; raw_writes.py | n/a |
| `source.raw_sessions.blob_hash` | KEEP | idx_raw_sessions_blob_hash; blob_gc.py reference surface check | n/a |
| `source.raw_sessions.blob_size` | KEEP | raw_state.py, mappers_archive.py | n/a |
| `source.raw_sessions.acquired_at_ms` | KEEP | raw_reads.py, daemon/provenance.py | n/a |
| `source.raw_sessions.file_mtime_ms` | KEEP | raw_reads.py:192-198, daemon/provenance.py:132, blob_integrity.py:1058-1078, repair.py:503,747,1041 | n/a |
| `source.raw_sessions.parsed_at_ms` | KEEP | idx_raw_sessions_parse_ready; raw_state.py | n/a |
| `source.raw_sessions.parse_error` | KEEP | daemon/status.py:982, raw_state.py | n/a |
| `source.raw_sessions.validated_at_ms` | KEEP | idx_raw_sessions_parse_ready; raw_state.py:224 | n/a |
| `source.raw_sessions.validation_status` | KEEP | daemon/status.py:982, import_explain.py:281 | n/a |
| `source.raw_sessions.validation_error` | KEEP | daemon/provenance.py:137-266, daemon/status.py:982, import_explain.py:188-204, operations/archive_debt.py:246, mappers_archive.py:224 | n/a |
| `source.raw_sessions.validation_drift_count` | KEEP | raw_state.py:171, mappers_archive.py:225 | n/a |
| `source.raw_sessions.validation_mode` | KEEP | raw_writes.py, source_write.py:1009 | n/a |
| `source.raw_sessions.detection_warnings_json` | KEEP | import_explain.py:189-281, daemon/status.py:956-957 | n/a |
| `source.raw_sessions.logical_source_key` | KEEP | idx_raw_sessions_logical_revision; pervasive in raw_reconciler/revision_authority | n/a |
| `source.raw_sessions.revision_kind` | KEEP | CHECK-gated, raw_state.py, revision_governance.py | n/a |
| `source.raw_sessions.source_revision` | KEEP | raw_session_memberships joins, repair.py | n/a |
| `source.raw_sessions.predecessor_source_revision` | KEEP | revision_governance.py append-chain resolution | n/a |
| `source.raw_sessions.predecessor_raw_id` | KEEP | raw_append_chain_backfill_receipts linkage, revision_governance.py | n/a |
| `source.raw_sessions.baseline_raw_id` | KEEP | revision_governance.py append-chain baseline resolution | n/a |
| `source.raw_sessions.append_start_offset` | KEEP | revision_governance.py append-chain validation | n/a |
| `source.raw_sessions.append_end_offset` | KEEP | revision_governance.py append-chain validation, CHECK(>start) | n/a |
| `source.raw_sessions.acquisition_generation` | KEEP | idx_raw_sessions_logical_revision; raw_reconciler generation tracking | n/a |
| `source.raw_sessions.revision_authority` | KEEP | idx_raw_sessions_raw_authority_census_candidates; pervasive gate (61 hits) - CHECK'd, filtered, read by raw_authority_verdict_projection | n/a |
| `source.raw_sessions.revision_authority_evidence` | KEEP | raw_live_source_reconciliation_receipts/raw_append_chain_backfill_receipts provenance linkage | n/a |
| `source.raw_sessions.detected_provider` | KEEP | sampling_db.py:316-332, migration 033 rationale (detected-vs-declared provider drift) | n/a |
| `source.raw_unknown_export_reclassification_receipts.raw_id` | KEEP | polylogue/maintenance/unknown_export_reclassification_apply.py:116-220 (existence + reclassification-required gate) | n/a |
| `source.raw_unknown_export_reclassification_receipts.previous_origin` | KEEP | polylogue/maintenance/unknown_export_reclassification_apply.py:116-220 (existence + reclassification-required gate) | n/a |
| `source.raw_unknown_export_reclassification_receipts.new_origin` | KEEP | polylogue/maintenance/unknown_export_reclassification_apply.py:116-220 (existence + reclassification-required gate) | n/a |
| `source.raw_unknown_export_reclassification_receipts.previous_capture_mode` | KEEP | polylogue/maintenance/unknown_export_reclassification_apply.py:116-220 (existence + reclassification-required gate) | n/a |
| `source.raw_unknown_export_reclassification_receipts.new_capture_mode` | KEEP | polylogue/maintenance/unknown_export_reclassification_apply.py:116-220 (existence + reclassification-required gate) | n/a |
| `source.raw_unknown_export_reclassification_receipts.embedded_provider` | KEEP | polylogue/maintenance/unknown_export_reclassification_apply.py:116-220 (existence + reclassification-required gate) | n/a |
| `source.raw_unknown_export_reclassification_receipts.source_path` | KEEP | polylogue/maintenance/unknown_export_reclassification_apply.py:116-220 (existence + reclassification-required gate) | n/a |
| `source.raw_unknown_export_reclassification_receipts.blob_hash` | KEEP | polylogue/maintenance/unknown_export_reclassification_apply.py:116-220 (existence + reclassification-required gate) | n/a |
| `source.raw_unknown_export_reclassification_receipts.blob_size` | KEEP | polylogue/maintenance/unknown_export_reclassification_apply.py:116-220 (existence + reclassification-required gate) | n/a |
| `source.raw_unknown_export_reclassification_receipts.reclassified_at_ms` | KEEP | polylogue/maintenance/unknown_export_reclassification_apply.py:116-220 (existence + reclassification-required gate) | n/a |
| `source.raw_unknown_export_reclassification_receipts.tool_version` | KEEP | polylogue/maintenance/unknown_export_reclassification_apply.py:116-220 (existence + reclassification-required gate) | n/a |
| `source.raw_unknown_export_reclassification_receipts.backup_manifest_path` | KEEP | polylogue/maintenance/unknown_export_reclassification_apply.py:116-220 (existence + reclassification-required gate) | n/a |
| `source.raw_unknown_export_reclassification_receipts.index_reparse_required` | KEEP | polylogue/maintenance/unknown_export_reclassification_apply.py:116-220 (existence + reclassification-required gate) | n/a |
| `source.raw_unknown_export_reclassification_receipts.detail` | KEEP | polylogue/maintenance/unknown_export_reclassification_apply.py:116-220 (existence + reclassification-required gate) | n/a |
| `source.sinex_publication_obligations.object_id` | KEEP | polylogue/sinex/service.py, polylogue/sinex/obligations.py (seed-carried) | n/a |
| `source.sinex_publication_obligations.protocol_version` | KEEP | polylogue/sinex/service.py, polylogue/sinex/obligations.py (seed-carried) | n/a |
| `source.sinex_publication_obligations.revision_id` | KEEP | polylogue/sinex/service.py, polylogue/sinex/obligations.py (seed-carried) | n/a |
| `source.sinex_publication_obligations.manifest_digest` | KEEP | polylogue/sinex/service.py, polylogue/sinex/obligations.py (seed-carried) | n/a |
| `source.sinex_publication_obligations.mode` | KEEP | polylogue/sinex/service.py, polylogue/sinex/obligations.py (seed-carried) | n/a |
| `source.sinex_publication_obligations.status` | KEEP | polylogue/sinex/service.py, polylogue/sinex/obligations.py (seed-carried) | n/a |
| `source.sinex_publication_obligations.attempt_count` | KEEP | polylogue/sinex/service.py, polylogue/sinex/obligations.py (seed-carried) | n/a |
| `source.sinex_publication_obligations.last_attempt_at_ms` | KEEP | polylogue/sinex/service.py, polylogue/sinex/obligations.py (seed-carried) | n/a |
| `source.sinex_publication_obligations.last_receipt_state` | KEEP | polylogue/sinex/service.py, polylogue/sinex/obligations.py (seed-carried) | n/a |
| `source.sinex_publication_obligations.last_error` | KEEP | polylogue/sinex/service.py, polylogue/sinex/obligations.py (seed-carried) | n/a |
| `source.sinex_publication_obligations.created_at_ms` | KEEP | polylogue/sinex/service.py, polylogue/sinex/obligations.py (seed-carried) | n/a |
| `source.sinex_publication_obligations.updated_at_ms` | KEEP | polylogue/sinex/service.py, polylogue/sinex/obligations.py (seed-carried) | n/a |
| `source.sinex_publication_obligations.retired_at_ms` | KEEP | polylogue/sinex/service.py, polylogue/sinex/obligations.py (seed-carried) | n/a |
| `source.sinex_publication_obligations.next_attempt_at_ms` | KEEP | polylogue/sinex/service.py, polylogue/sinex/obligations.py (seed-carried) | n/a |
| `source.sinex_publication_payloads.object_id` | KEEP | polylogue/sinex/service.py, polylogue/sinex/obligations.py (seed-carried) | n/a |
| `source.sinex_publication_payloads.protocol_version` | KEEP | polylogue/sinex/service.py, polylogue/sinex/obligations.py (seed-carried) | n/a |
| `source.sinex_publication_payloads.revision_id` | KEEP | polylogue/sinex/service.py, polylogue/sinex/obligations.py (seed-carried) | n/a |
| `source.sinex_publication_payloads.manifest_digest` | KEEP | polylogue/sinex/service.py, polylogue/sinex/obligations.py (seed-carried) | n/a |
| `source.sinex_publication_payloads.manifest_bytes` | KEEP | polylogue/sinex/service.py, polylogue/sinex/obligations.py (seed-carried) | n/a |
| `source.sinex_publication_payloads.manifest_sha256` | KEEP | polylogue/sinex/service.py, polylogue/sinex/obligations.py (seed-carried) | n/a |
| `source.sinex_publication_payloads.manifest_size_bytes` | KEEP | polylogue/sinex/service.py, polylogue/sinex/obligations.py (seed-carried) | n/a |
| `source.sinex_publication_payloads.segment_count` | KEEP | polylogue/sinex/service.py, polylogue/sinex/obligations.py (seed-carried) | n/a |
| `source.sinex_publication_payloads.total_size_bytes` | KEEP | polylogue/sinex/service.py, polylogue/sinex/obligations.py (seed-carried) | n/a |
| `source.sinex_publication_payloads.staged_at_ms` | KEEP | polylogue/sinex/service.py, polylogue/sinex/obligations.py (seed-carried) | n/a |
| `source.sinex_publication_receipts.object_id` | KEEP | polylogue/sinex/service.py, polylogue/sinex/obligations.py (seed-carried) | n/a |
| `source.sinex_publication_receipts.protocol_version` | KEEP | polylogue/sinex/service.py, polylogue/sinex/obligations.py (seed-carried) | n/a |
| `source.sinex_publication_receipts.revision_id` | KEEP | polylogue/sinex/service.py, polylogue/sinex/obligations.py (seed-carried) | n/a |
| `source.sinex_publication_receipts.manifest_digest` | KEEP | polylogue/sinex/service.py, polylogue/sinex/obligations.py (seed-carried) | n/a |
| `source.sinex_publication_receipts.attempt_number` | KEEP | polylogue/sinex/service.py, polylogue/sinex/obligations.py (seed-carried) | n/a |
| `source.sinex_publication_receipts.request_id` | KEEP | polylogue/sinex/service.py, polylogue/sinex/obligations.py (seed-carried) | n/a |
| `source.sinex_publication_receipts.receipt_state` | KEEP | polylogue/sinex/service.py, polylogue/sinex/obligations.py (seed-carried) | n/a |
| `source.sinex_publication_receipts.receipt_detail` | KEEP | polylogue/sinex/service.py, polylogue/sinex/obligations.py (seed-carried) | n/a |
| `source.sinex_publication_receipts.error_code` | KEEP | polylogue/sinex/service.py, polylogue/sinex/obligations.py (seed-carried) | n/a |
| `source.sinex_publication_receipts.received_at_ms` | KEEP | polylogue/sinex/service.py, polylogue/sinex/obligations.py (seed-carried) | n/a |
| `source.sinex_publication_segments.object_id` | KEEP | polylogue/sinex/service.py, polylogue/sinex/obligations.py (seed-carried) | n/a |
| `source.sinex_publication_segments.protocol_version` | KEEP | polylogue/sinex/service.py, polylogue/sinex/obligations.py (seed-carried) | n/a |
| `source.sinex_publication_segments.revision_id` | KEEP | polylogue/sinex/service.py, polylogue/sinex/obligations.py (seed-carried) | n/a |
| `source.sinex_publication_segments.manifest_digest` | KEEP | polylogue/sinex/service.py, polylogue/sinex/obligations.py (seed-carried) | n/a |
| `source.sinex_publication_segments.position` | KEEP | polylogue/sinex/service.py, polylogue/sinex/obligations.py (seed-carried) | n/a |
| `source.sinex_publication_segments.segment_name` | KEEP | polylogue/sinex/service.py, polylogue/sinex/obligations.py (seed-carried) | n/a |
| `source.sinex_publication_segments.segment_bytes` | KEEP | polylogue/sinex/service.py, polylogue/sinex/obligations.py (seed-carried) | n/a |
| `source.sinex_publication_segments.segment_sha256` | KEEP | polylogue/sinex/service.py, polylogue/sinex/obligations.py (seed-carried) | n/a |
| `source.sinex_publication_segments.size_bytes` | KEEP | polylogue/sinex/service.py, polylogue/sinex/obligations.py (seed-carried) | n/a |
| `source.verified_blob_receipts.blob_hash` | KEEP | blob_verification.py (seed-carried) | n/a |
| `source.verified_blob_receipts.st_dev` | KEEP | blob_verification.py (seed-carried) | n/a |
| `source.verified_blob_receipts.st_ino` | KEEP | blob_verification.py (seed-carried) | n/a |
| `source.verified_blob_receipts.st_size` | KEEP | blob_verification.py (seed-carried) | n/a |
| `source.verified_blob_receipts.st_mtime_ns` | KEEP | blob_verification.py (seed-carried) | n/a |
| `source.verified_blob_receipts.st_ctime_ns` | KEEP | blob_verification.py (seed-carried) | n/a |
| `source.verified_blob_receipts.verified_at_ms` | KEEP | blob_verification.py (seed-carried) | n/a |

### index.db

| tier.table.column | verdict | evidence | unlocks / owning reference |
|---|---|---|---|
| `index.action_pairs.tool_use_block_id` | KEEP | polylogue/product/continuity_scenarios.py (+13 more outside-tier files) | n/a |
| `index.action_pairs.session_id` | KEEP | devtools/resume_ranking_eval.py (+398 more outside-tier files) | n/a |
| `index.action_pairs.message_id` | KEEP | devtools/lineage_validation.py (+139 more outside-tier files) | n/a |
| `index.action_pairs.tool_id` | KEEP | devtools/scale_regression_probe.py (+55 more outside-tier files) | n/a |
| `index.action_pairs.use_rank` | KEEP | polylogue/storage/sqlite/action_relation.py (+1 more outside-tier files) | n/a |
| `index.action_pairs.tool_name` | KEEP | devtools/affordance_usage.py (+78 more outside-tier files) | n/a |
| `index.action_pairs.semantic_type` | KEEP | polylogue/mcp/archive_support.py (+33 more outside-tier files) | n/a |
| `index.action_pairs.tool_command` | KEEP | devtools/claim_vs_evidence.py (+11 more outside-tier files) | n/a |
| `index.action_pairs.tool_path` | KEEP | devtools/affordance_usage.py (+14 more outside-tier files) | n/a |
| `index.action_pairs.tool_result_block_id` | KEEP | devtools/claim_vs_evidence_evidence.py (+8 more outside-tier files) | n/a |
| `index.action_pairs.is_error` | KEEP | polylogue/sinex/material_adapter.py (+52 more outside-tier files) | n/a |
| `index.action_pairs.exit_code` | KEEP | devtools/query_memory_budget.py (+73 more outside-tier files) | n/a |
| `index.actions.session_id` | KEEP | devtools/resume_ranking_eval.py (+398 more outside-tier files) | n/a |
| `index.actions.message_id` | KEEP | devtools/lineage_validation.py (+139 more outside-tier files) | n/a |
| `index.actions.tool_use_block_id` | KEEP | polylogue/product/continuity_scenarios.py (+13 more outside-tier files) | n/a |
| `index.actions.tool_name` | KEEP | polylogue/sinex/material_adapter.py (+78 more outside-tier files) | n/a |
| `index.actions.semantic_type` | KEEP | polylogue/sinex/material_adapter.py (+33 more outside-tier files) | n/a |
| `index.actions.tool_command` | KEEP | devtools/claim_vs_evidence.py (+11 more outside-tier files) | n/a |
| `index.actions.tool_path` | KEEP | devtools/affordance_usage.py (+14 more outside-tier files) | n/a |
| `index.actions.tool_input` | KEEP | polylogue/security/secret_scan.py (+43 more outside-tier files) | n/a |
| `index.actions.output_text` | KEEP | polylogue/product/continuity_scenarios.py (+22 more outside-tier files) | n/a |
| `index.actions.is_error` | KEEP | polylogue/sinex/material_adapter.py (+52 more outside-tier files) | n/a |
| `index.actions.exit_code` | KEEP | devtools/query_memory_budget.py (+73 more outside-tier files) | n/a |
| `index.actions.tool_result_block_id` | KEEP | devtools/claim_vs_evidence_evidence.py (+8 more outside-tier files) | n/a |
| `index.actions.result_state` | KEEP | polylogue/archive/actions/parsing.py (+6 more outside-tier files) | n/a |
| `index.agent_meta_sidecar_purge_receipts.session_id` | PURGE | write-only, see polylogue/maintenance/agent_meta_sidecar_purge_apply.py:127 (sole writer). Only other reference is polylogue/maintenance/reindex_canary.py:1890's TABLE-NAME-STRING existence check, not a column reader. | unlocks dropping the whole table + idx_agent_meta_sidecar_purge_receipts_purged_at + the one-time actuator devtools/agent_meta_sidecar_purge_apply.py + polylogue/maintenance/agent_meta_sidecar_purge_apply.py once its one-time evidence obligation is discharged (seed's highest-leverage index-tier candidate, polylogue-ioz7 v57) |
| `index.agent_meta_sidecar_purge_receipts.origin` | PURGE | write-only, see polylogue/maintenance/agent_meta_sidecar_purge_apply.py:127 (sole writer). Only other reference is polylogue/maintenance/reindex_canary.py:1890's TABLE-NAME-STRING existence check, not a column reader. | unlocks dropping the whole table + idx_agent_meta_sidecar_purge_receipts_purged_at + the one-time actuator devtools/agent_meta_sidecar_purge_apply.py + polylogue/maintenance/agent_meta_sidecar_purge_apply.py once its one-time evidence obligation is discharged (seed's highest-leverage index-tier candidate, polylogue-ioz7 v57) |
| `index.agent_meta_sidecar_purge_receipts.native_id` | PURGE | write-only, see polylogue/maintenance/agent_meta_sidecar_purge_apply.py:127 (sole writer). Only other reference is polylogue/maintenance/reindex_canary.py:1890's TABLE-NAME-STRING existence check, not a column reader. | unlocks dropping the whole table + idx_agent_meta_sidecar_purge_receipts_purged_at + the one-time actuator devtools/agent_meta_sidecar_purge_apply.py + polylogue/maintenance/agent_meta_sidecar_purge_apply.py once its one-time evidence obligation is discharged (seed's highest-leverage index-tier candidate, polylogue-ioz7 v57) |
| `index.agent_meta_sidecar_purge_receipts.raw_id` | PURGE | write-only, see polylogue/maintenance/agent_meta_sidecar_purge_apply.py:127 (sole writer). Only other reference is polylogue/maintenance/reindex_canary.py:1890's TABLE-NAME-STRING existence check, not a column reader. | unlocks dropping the whole table + idx_agent_meta_sidecar_purge_receipts_purged_at + the one-time actuator devtools/agent_meta_sidecar_purge_apply.py + polylogue/maintenance/agent_meta_sidecar_purge_apply.py once its one-time evidence obligation is discharged (seed's highest-leverage index-tier candidate, polylogue-ioz7 v57) |
| `index.agent_meta_sidecar_purge_receipts.source_path` | PURGE | write-only, see polylogue/maintenance/agent_meta_sidecar_purge_apply.py:127 (sole writer). Only other reference is polylogue/maintenance/reindex_canary.py:1890's TABLE-NAME-STRING existence check, not a column reader. | unlocks dropping the whole table + idx_agent_meta_sidecar_purge_receipts_purged_at + the one-time actuator devtools/agent_meta_sidecar_purge_apply.py + polylogue/maintenance/agent_meta_sidecar_purge_apply.py once its one-time evidence obligation is discharged (seed's highest-leverage index-tier candidate, polylogue-ioz7 v57) |
| `index.agent_meta_sidecar_purge_receipts.purged_at_ms` | PURGE | write-only, see polylogue/maintenance/agent_meta_sidecar_purge_apply.py:127 (sole writer). Only other reference is polylogue/maintenance/reindex_canary.py:1890's TABLE-NAME-STRING existence check, not a column reader. | unlocks dropping the whole table + idx_agent_meta_sidecar_purge_receipts_purged_at + the one-time actuator devtools/agent_meta_sidecar_purge_apply.py + polylogue/maintenance/agent_meta_sidecar_purge_apply.py once its one-time evidence obligation is discharged (seed's highest-leverage index-tier candidate, polylogue-ioz7 v57) |
| `index.agent_meta_sidecar_purge_receipts.tool_version` | PURGE | write-only, see polylogue/maintenance/agent_meta_sidecar_purge_apply.py:127 (sole writer). Only other reference is polylogue/maintenance/reindex_canary.py:1890's TABLE-NAME-STRING existence check, not a column reader. | unlocks dropping the whole table + idx_agent_meta_sidecar_purge_receipts_purged_at + the one-time actuator devtools/agent_meta_sidecar_purge_apply.py + polylogue/maintenance/agent_meta_sidecar_purge_apply.py once its one-time evidence obligation is discharged (seed's highest-leverage index-tier candidate, polylogue-ioz7 v57) |
| `index.agent_meta_sidecar_purge_receipts.backup_manifest_path` | PURGE | write-only, see polylogue/maintenance/agent_meta_sidecar_purge_apply.py:127 (sole writer). Only other reference is polylogue/maintenance/reindex_canary.py:1890's TABLE-NAME-STRING existence check, not a column reader. | unlocks dropping the whole table + idx_agent_meta_sidecar_purge_receipts_purged_at + the one-time actuator devtools/agent_meta_sidecar_purge_apply.py + polylogue/maintenance/agent_meta_sidecar_purge_apply.py once its one-time evidence obligation is discharged (seed's highest-leverage index-tier candidate, polylogue-ioz7 v57) |
| `index.agent_meta_sidecar_purge_receipts.detail` | PURGE | write-only, see polylogue/maintenance/agent_meta_sidecar_purge_apply.py:127 (sole writer). Only other reference is polylogue/maintenance/reindex_canary.py:1890's TABLE-NAME-STRING existence check, not a column reader. | unlocks dropping the whole table + idx_agent_meta_sidecar_purge_receipts_purged_at + the one-time actuator devtools/agent_meta_sidecar_purge_apply.py + polylogue/maintenance/agent_meta_sidecar_purge_apply.py once its one-time evidence obligation is discharged (seed's highest-leverage index-tier candidate, polylogue-ioz7 v57) |
| `index.attachment_native_ids.ref_id` | KEEP | polylogue/security/excision.py (+29 more outside-tier files) | n/a |
| `index.attachment_native_ids.id_kind` | KEEP | polylogue/storage/attachment_relink.py (+3 more outside-tier files) | n/a |
| `index.attachment_native_ids.native_id` | KEEP | devtools/storage_correctness_scenario.py (+97 more outside-tier files) | n/a |
| `index.attachment_refs.attachment_id` | KEEP | devtools/attachment_reacquisition_report.py (+41 more outside-tier files) | n/a |
| `index.attachment_refs.session_id` | KEEP | devtools/resume_ranking_eval.py (+398 more outside-tier files) | n/a |
| `index.attachment_refs.message_id` | KEEP | devtools/lineage_validation.py (+139 more outside-tier files) | n/a |
| `index.attachment_refs.position` | KEEP | devtools/verify_runs.py (+122 more outside-tier files) | n/a |
| `index.attachment_refs.upload_origin` | KEEP | polylogue/sinex/material_adapter.py (+15 more outside-tier files) | n/a |
| `index.attachment_refs.source_url` | KEEP | devtools/dev_loop.py (+25 more outside-tier files) | n/a |
| `index.attachment_refs.caption` | KEEP | polylogue/product/continuity_scenarios.py (+17 more outside-tier files) | n/a |
| `index.attachments.attachment_id` | KEEP | devtools/attachment_reacquisition_report.py (+41 more outside-tier files) | n/a |
| `index.attachments.display_name` | KEEP | devtools/affordance_usage.py (+31 more outside-tier files) | n/a |
| `index.attachments.media_type` | KEEP | polylogue/sinex/material_adapter.py (+23 more outside-tier files) | n/a |
| `index.attachments.byte_count` | KEEP | devtools/raw_byte_duplicate_supersession_apply.py (+24 more outside-tier files) | n/a |
| `index.attachments.blob_hash` | KEEP | devtools/storage_correctness_scenario.py (+106 more outside-tier files) | n/a |
| `index.attachments.acquisition_status` | KEEP | devtools/attachment_reacquisition_report.py (+15 more outside-tier files) | n/a |
| `index.attachments.ref_count` | KEEP | polylogue/storage/session_replacement.py (+5 more outside-tier files) | n/a |
| `index.blocks.message_id` | KEEP | devtools/claim_vs_evidence.py (+139 more outside-tier files) | n/a |
| `index.blocks.session_id` | KEEP | devtools/resume_ranking_eval.py (+398 more outside-tier files) | n/a |
| `index.blocks.position` | KEEP | devtools/storage_correctness_scenario.py (+122 more outside-tier files) | n/a |
| `index.blocks.block_type` | KEEP | devtools/rebuild_safety_scenario.py (+56 more outside-tier files) | n/a |
| `index.blocks.text` | KEEP | devtools/__init__.py (+393 more outside-tier files) | n/a |
| `index.blocks.tool_name` | KEEP | devtools/claim_vs_evidence.py (+78 more outside-tier files) | n/a |
| `index.blocks.tool_id` | KEEP | devtools/claim_vs_evidence_evidence.py (+55 more outside-tier files) | n/a |
| `index.blocks.tool_input` | KEEP | polylogue/security/secret_scan.py (+43 more outside-tier files) | n/a |
| `index.blocks.semantic_type` | KEEP | devtools/affordance_usage.py (+33 more outside-tier files) | n/a |
| `index.blocks.media_type` | KEEP | polylogue/sinex/material_adapter.py (+23 more outside-tier files) | n/a |
| `index.blocks.language` | KEEP | polylogue/config.py (+65 more outside-tier files) | n/a |
| `index.blocks.tool_result_is_error` | KEEP | devtools/claim_vs_evidence.py (+26 more outside-tier files) | n/a |
| `index.blocks.tool_result_exit_code` | KEEP | devtools/claim_vs_evidence.py (+24 more outside-tier files) | n/a |
| `index.blocks.tool_result_outcome_unknown_reason` | KEEP | polylogue/storage/hydrators.py (+5 more outside-tier files) | n/a |
| `index.blocks.signature` | KEEP | polylogue/agent_integration/spec.py (+53 more outside-tier files) | n/a |
| `index.blocks.content_hash` | KEEP | devtools/storage_correctness_scenario.py (+44 more outside-tier files) | n/a |
| `index.blocks_command_trigram.tool_detail_text` | KEEP | devtools/affordance_usage.py (+4 more outside-tier files) | n/a |
| `index.delegation_facts.delegation_id` | KEEP | polylogue/storage/sqlite/delegation_facts.py | n/a |
| `index.delegation_facts.parent_session_id` | KEEP | devtools/resume_ranking_eval.py (+48 more outside-tier files) | n/a |
| `index.delegation_facts.child_session_id` | KEEP | polylogue/demo/seed.py (+16 more outside-tier files) | n/a |
| `index.delegation_facts.mapping_state` | KEEP | polylogue/annotations/join.py (+11 more outside-tier files) | n/a |
| `index.delegation_facts.link_confidence` | KEEP | polylogue/storage/sqlite/delegation_facts.py (+2 more outside-tier files) | n/a |
| `index.delegation_facts.link_method` | KEEP | polylogue/storage/sqlite/delegation_facts.py (+3 more outside-tier files) | n/a |
| `index.delegation_facts.inheritance` | KEEP | devtools/storage_correctness_scenario.py (+20 more outside-tier files) | n/a |
| `index.delegation_facts.branch_point_message_id` | KEEP | devtools/storage_correctness_scenario.py (+9 more outside-tier files) | n/a |
| `index.delegation_facts.instruction_message_id` | KEEP | polylogue/insights/delegation_work_evidence.py (+3 more outside-tier files) | n/a |
| `index.delegation_facts.instruction_tool_use_block_id` | KEEP | polylogue/api/archive.py (+7 more outside-tier files) | n/a |
| `index.delegation_facts.instruction_payload` | KEEP | polylogue/insights/fable_packet.py (+2 more outside-tier files) | n/a |
| `index.delegation_facts.dispatch_turn_model` | KEEP | polylogue/annotations/join.py (+5 more outside-tier files) | n/a |
| `index.delegation_facts.requested_model` | KEEP | polylogue/surfaces/payloads.py (+4 more outside-tier files) | n/a |
| `index.delegation_facts.artifact_block_id` | KEEP | polylogue/surfaces/payloads.py (+2 more outside-tier files) | n/a |
| `index.delegation_facts.artifact_text` | KEEP | polylogue/surfaces/payloads.py (+2 more outside-tier files) | n/a |
| `index.delegation_facts.result_is_error` | KEEP | polylogue/insights/delegation_work_evidence.py (+3 more outside-tier files) | n/a |
| `index.delegation_facts.result_exit_code` | KEEP | polylogue/storage/sqlite/delegation_facts.py (+2 more outside-tier files) | n/a |
| `index.delegation_facts.result_status` | KEEP | polylogue/cli/archive_query.py (+8 more outside-tier files) | n/a |
| `index.delegation_facts.parent_origin` | KEEP | devtools/lineage_validation.py (+5 more outside-tier files) | n/a |
| `index.delegation_facts.parent_session_dominant_model` | KEEP | polylogue/surfaces/payloads.py (+2 more outside-tier files) | n/a |
| `index.delegation_facts.parent_session_dominant_model_family` | KEEP | polylogue/storage/sqlite/delegation_facts.py (+1 more outside-tier files) | n/a |
| `index.delegation_facts.parent_terminal_state` | KEEP | polylogue/surfaces/payloads.py (+2 more outside-tier files) | n/a |
| `index.delegation_facts.child_session_dominant_model` | KEEP | polylogue/surfaces/payloads.py (+4 more outside-tier files) | n/a |
| `index.delegation_facts.child_session_dominant_model_family` | KEEP | polylogue/storage/sqlite/delegation_facts.py (+1 more outside-tier files) | n/a |
| `index.delegation_facts.child_cost_usd` | KEEP | polylogue/storage/sqlite/delegation_facts.py (+1 more outside-tier files) | n/a |
| `index.delegation_facts.child_cost_is_estimated` | KEEP | polylogue/storage/sqlite/delegation_facts.py (+1 more outside-tier files) | n/a |
| `index.delegation_facts.child_tokens` | KEEP | polylogue/surfaces/payloads.py (+1 more outside-tier files) | n/a |
| `index.delegation_facts.child_wall_ms` | KEEP | polylogue/storage/sqlite/delegation_facts.py (+1 more outside-tier files) | n/a |
| `index.delegation_facts.child_terminal_state` | KEEP | polylogue/surfaces/payloads.py (+1 more outside-tier files) | n/a |
| `index.delegation_facts_source.parent_session_id` | KEEP | devtools/resume_ranking_eval.py (+48 more outside-tier files) | n/a |
| `index.delegation_facts_source.child_session_id` | KEEP | polylogue/coordination/envelope.py (+16 more outside-tier files) | n/a |
| `index.delegation_facts_source.mapping_state` | KEEP | polylogue/annotations/join.py (+11 more outside-tier files) | n/a |
| `index.delegation_facts_source.link_confidence` | KEEP | polylogue/storage/sqlite/delegation_facts.py (+2 more outside-tier files) | n/a |
| `index.delegation_facts_source.link_method` | KEEP | polylogue/storage/sqlite/delegation_facts.py (+3 more outside-tier files) | n/a |
| `index.delegation_facts_source.inheritance` | KEEP | devtools/storage_correctness_scenario.py (+20 more outside-tier files) | n/a |
| `index.delegation_facts_source.branch_point_message_id` | KEEP | devtools/storage_correctness_scenario.py (+9 more outside-tier files) | n/a |
| `index.delegation_facts_source.instruction_message_id` | KEEP | polylogue/surfaces/payloads.py (+3 more outside-tier files) | n/a |
| `index.delegation_facts_source.instruction_tool_use_block_id` | KEEP | polylogue/storage/sqlite/delegation_facts.py (+7 more outside-tier files) | n/a |
| `index.delegation_facts_source.instruction_payload` | KEEP | polylogue/insights/fable_packet.py (+2 more outside-tier files) | n/a |
| `index.delegation_facts_source.dispatch_turn_model` | KEEP | polylogue/annotations/join.py (+5 more outside-tier files) | n/a |
| `index.delegation_facts_source.requested_model` | KEEP | polylogue/storage/sqlite/delegation_facts.py (+4 more outside-tier files) | n/a |
| `index.delegation_facts_source.artifact_block_id` | KEEP | polylogue/storage/sqlite/delegation_facts.py (+2 more outside-tier files) | n/a |
| `index.delegation_facts_source.artifact_text` | KEEP | polylogue/insights/delegation_work_evidence.py (+2 more outside-tier files) | n/a |
| `index.delegation_facts_source.result_is_error` | KEEP | polylogue/surfaces/payloads.py (+3 more outside-tier files) | n/a |
| `index.delegation_facts_source.result_exit_code` | KEEP | polylogue/storage/sqlite/delegation_facts.py (+2 more outside-tier files) | n/a |
| `index.delegation_facts_source.result_status` | KEEP | polylogue/cli/archive_query.py (+8 more outside-tier files) | n/a |
| `index.delegation_facts_source.parent_origin` | KEEP | devtools/lineage_validation.py (+5 more outside-tier files) | n/a |
| `index.delegation_facts_source.parent_session_dominant_model` | KEEP | polylogue/surfaces/payloads.py (+2 more outside-tier files) | n/a |
| `index.delegation_facts_source.parent_session_dominant_model_family` | KEEP | polylogue/storage/sqlite/delegation_facts.py (+1 more outside-tier files) | n/a |
| `index.delegation_facts_source.parent_terminal_state` | KEEP | polylogue/core/enums.py (+2 more outside-tier files) | n/a |
| `index.delegation_facts_source.child_session_dominant_model` | KEEP | polylogue/archive/session/runtime.py (+4 more outside-tier files) | n/a |
| `index.delegation_facts_source.child_session_dominant_model_family` | KEEP | polylogue/surfaces/payloads.py (+1 more outside-tier files) | n/a |
| `index.delegation_facts_source.child_cost_usd` | KEEP | polylogue/surfaces/payloads.py (+1 more outside-tier files) | n/a |
| `index.delegation_facts_source.child_cost_is_estimated` | KEEP | polylogue/storage/sqlite/delegation_facts.py (+1 more outside-tier files) | n/a |
| `index.delegation_facts_source.child_tokens` | KEEP | polylogue/storage/sqlite/delegation_facts.py (+1 more outside-tier files) | n/a |
| `index.delegation_facts_source.child_wall_ms` | KEEP | polylogue/storage/sqlite/delegation_facts.py (+1 more outside-tier files) | n/a |
| `index.delegation_facts_source.child_terminal_state` | KEEP | polylogue/surfaces/payloads.py (+1 more outside-tier files) | n/a |
| `index.delegation_refresh_scope.parent_session_id` | KEEP | devtools/resume_ranking_eval.py (+48 more outside-tier files) | n/a |
| `index.delegations.parent_session_id` | KEEP | devtools/resume_ranking_eval.py (+48 more outside-tier files) | n/a |
| `index.delegations.child_session_id` | KEEP | polylogue/surfaces/payloads.py (+16 more outside-tier files) | n/a |
| `index.delegations.mapping_state` | KEEP | polylogue/annotations/join.py (+11 more outside-tier files) | n/a |
| `index.delegations.link_confidence` | KEEP | polylogue/surfaces/payloads.py (+2 more outside-tier files) | n/a |
| `index.delegations.link_method` | KEEP | polylogue/storage/sqlite/delegation_facts.py (+3 more outside-tier files) | n/a |
| `index.delegations.inheritance` | KEEP | devtools/storage_correctness_scenario.py (+20 more outside-tier files) | n/a |
| `index.delegations.branch_point_message_id` | KEEP | devtools/storage_correctness_scenario.py (+9 more outside-tier files) | n/a |
| `index.delegations.instruction_message_id` | KEEP | polylogue/insights/delegation_work_evidence.py (+3 more outside-tier files) | n/a |
| `index.delegations.instruction_tool_use_block_id` | KEEP | polylogue/surfaces/payloads.py (+7 more outside-tier files) | n/a |
| `index.delegations.instruction_payload` | KEEP | polylogue/surfaces/payloads.py (+2 more outside-tier files) | n/a |
| `index.delegations.dispatch_turn_model` | KEEP | polylogue/annotations/join.py (+5 more outside-tier files) | n/a |
| `index.delegations.requested_model` | KEEP | polylogue/storage/sqlite/delegation_facts.py (+4 more outside-tier files) | n/a |
| `index.delegations.artifact_block_id` | KEEP | polylogue/surfaces/payloads.py (+2 more outside-tier files) | n/a |
| `index.delegations.artifact_text` | KEEP | polylogue/insights/delegation_work_evidence.py (+2 more outside-tier files) | n/a |
| `index.delegations.result_is_error` | KEEP | polylogue/insights/delegation_work_evidence.py (+3 more outside-tier files) | n/a |
| `index.delegations.result_exit_code` | KEEP | polylogue/archive/query/discovery.py (+2 more outside-tier files) | n/a |
| `index.delegations.result_status` | KEEP | polylogue/core/enums.py (+8 more outside-tier files) | n/a |
| `index.delegations.parent_origin` | KEEP | devtools/lineage_validation.py (+5 more outside-tier files) | n/a |
| `index.delegations.parent_session_dominant_model` | KEEP | polylogue/storage/sqlite/delegation_facts.py (+2 more outside-tier files) | n/a |
| `index.delegations.parent_session_dominant_model_family` | KEEP | polylogue/surfaces/payloads.py (+1 more outside-tier files) | n/a |
| `index.delegations.parent_terminal_state` | KEEP | polylogue/core/enums.py (+2 more outside-tier files) | n/a |
| `index.delegations.child_session_dominant_model` | KEEP | polylogue/archive/session/runtime.py (+4 more outside-tier files) | n/a |
| `index.delegations.child_session_dominant_model_family` | KEEP | polylogue/surfaces/payloads.py (+1 more outside-tier files) | n/a |
| `index.delegations.child_cost_usd` | KEEP | polylogue/surfaces/payloads.py (+1 more outside-tier files) | n/a |
| `index.delegations.child_cost_is_estimated` | KEEP | polylogue/storage/sqlite/delegation_facts.py (+1 more outside-tier files) | n/a |
| `index.delegations.child_tokens` | KEEP | polylogue/surfaces/payloads.py (+1 more outside-tier files) | n/a |
| `index.delegations.child_wall_ms` | KEEP | polylogue/surfaces/payloads.py (+1 more outside-tier files) | n/a |
| `index.delegations.child_terminal_state` | KEEP | polylogue/surfaces/payloads.py (+1 more outside-tier files) | n/a |
| `index.derived_refresh_guard.guard_name` | KEEP | polylogue/maintenance/sharded_rebuild.py (+1 more outside-tier files) | n/a |
| `index.file_edits.tool_use_block_id` | KEEP | polylogue/product/continuity_scenarios.py (+13 more outside-tier files) | n/a |
| `index.file_edits.session_id` | KEEP | devtools/resume_ranking_eval.py (+398 more outside-tier files) | n/a |
| `index.file_edits.message_id` | KEEP | devtools/lineage_validation.py (+139 more outside-tier files) | n/a |
| `index.file_edits.file_path` | KEEP | devtools/verify.py (+26 more outside-tier files) | n/a |
| `index.file_edits.structured_patch_json` | KEEP | polylogue/storage/sqlite/queries/file_edits.py (+1 more outside-tier files) | n/a |
| `index.file_edits.original_file` | KEEP | polylogue/mcp/server_cutover.py (+8 more outside-tier files) | n/a |
| `index.file_edits.old_string` | KEEP | polylogue/pipeline/semantic_metadata.py (+9 more outside-tier files) | n/a |
| `index.file_edits.new_string` | KEEP | polylogue/demo/seed.py (+9 more outside-tier files) | n/a |
| `index.file_edits.replace_all` | KEEP | polylogue/sources/parsers/base_models.py (+5 more outside-tier files) | n/a |
| `index.file_edits.user_modified` | KEEP | polylogue/storage/runtime/archive/records.py (+5 more outside-tier files) | n/a |
| `index.file_edits.observed_at_ms` | KEEP | devtools/daemon_workload_probe.py (+57 more outside-tier files) | n/a |
| `index.fts_freshness_state.surface` | KEEP | polylogue/config.py (+302 more outside-tier files) | n/a |
| `index.fts_freshness_state.state` | KEEP | polylogue/config.py (+374 more outside-tier files) | n/a |
| `index.fts_freshness_state.checked_at` | KEEP | polylogue/daemon/notification_backends/journald.py (+16 more outside-tier files) | n/a |
| `index.fts_freshness_state.source_rows` | KEEP | devtools/verify_agent_integration.py (+25 more outside-tier files) | n/a |
| `index.fts_freshness_state.indexed_rows` | KEEP | devtools/storage_correctness_scenario.py (+18 more outside-tier files) | n/a |
| `index.fts_freshness_state.missing_rows` | KEEP | polylogue/storage/archive_readiness.py (+14 more outside-tier files) | n/a |
| `index.fts_freshness_state.excess_rows` | KEEP | polylogue/daemon/convergence_stages.py (+13 more outside-tier files) | n/a |
| `index.fts_freshness_state.duplicate_rows` | KEEP | polylogue/maintenance/archive_verification.py (+14 more outside-tier files) | n/a |
| `index.fts_freshness_state.detail` | KEEP | polylogue/daemon_client.py (+218 more outside-tier files) | n/a |
| `index.insight_materialization.insight_type` | KEEP | devtools/daemon_workload_probe.py (+11 more outside-tier files) | n/a |
| `index.insight_materialization.session_id` | KEEP | devtools/resume_ranking_eval.py (+398 more outside-tier files) | n/a |
| `index.insight_materialization.materializer_version` | KEEP | devtools/resume_ranking_eval.py (+44 more outside-tier files) | n/a |
| `index.insight_materialization.materialized_at_ms` | KEEP | devtools/rebuild_safety_scenario.py (+3 more outside-tier files) | n/a |
| `index.insight_materialization.source_updated_at_ms` | KEEP | polylogue/storage/insights/session/rebuild.py (+2 more outside-tier files) | n/a |
| `index.insight_materialization.source_sort_key_ms` | KEEP | polylogue/storage/repair.py (+3 more outside-tier files) | n/a |
| `index.insight_materialization.input_high_water_mark_ms` | KEEP | polylogue/storage/insights/session/rebuild.py | n/a |
| `index.insight_materialization.input_high_water_mark_source` | KEEP | polylogue/daemon/http.py (+22 more outside-tier files) | n/a |
| `index.insight_materialization.input_row_count` | KEEP | polylogue/daemon/http.py (+19 more outside-tier files) | n/a |
| `index.messages.session_id` | KEEP | devtools/resume_ranking_eval.py (+398 more outside-tier files) | n/a |
| `index.messages.native_id` | KEEP | devtools/storage_correctness_scenario.py (+97 more outside-tier files) | n/a |
| `index.messages.parent_message_id` | KEEP | polylogue/pipeline/services/ingest_batch/_core.py (+21 more outside-tier files) | n/a |
| `index.messages.position` | KEEP | devtools/storage_correctness_scenario.py (+122 more outside-tier files) | n/a |
| `index.messages.role` | KEEP | polylogue/config.py (+209 more outside-tier files) | n/a |
| `index.messages.message_type` | KEEP | devtools/render_openapi.py (+70 more outside-tier files) | n/a |
| `index.messages.material_origin` | KEEP | devtools/claim_vs_evidence.py (+59 more outside-tier files) | n/a |
| `index.messages.model_name` | KEEP | polylogue/sinex/material_adapter.py (+40 more outside-tier files) | n/a |
| `index.messages.model_effort` | KEEP | polylogue/sinex/material_adapter.py (+6 more outside-tier files) | n/a |
| `index.messages.sender_name` | KEEP | polylogue/sinex/material_adapter.py (+5 more outside-tier files) | n/a |
| `index.messages.recipient` | KEEP | polylogue/config.py (+12 more outside-tier files) | n/a |
| `index.messages.delivery_status` | KEEP | polylogue/sinex/material_adapter.py (+4 more outside-tier files) | n/a |
| `index.messages.end_turn` | KEEP | polylogue/sinex/material_adapter.py (+11 more outside-tier files) | n/a |
| `index.messages.user_context_text` | KEEP | polylogue/sinex/material_adapter.py (+4 more outside-tier files) | n/a |
| `index.messages.has_tool_use` | KEEP | devtools/synthetic_benchmark_runtime.py (+31 more outside-tier files) | n/a |
| `index.messages.has_thinking` | KEEP | devtools/render_openapi.py (+29 more outside-tier files) | n/a |
| `index.messages.has_paste` | KEEP | polylogue/mcp/archive_support.py (+23 more outside-tier files) | n/a |
| `index.messages.paste_boundary` | KEEP | polylogue/archive/message/paste_detection.py (+1 more outside-tier files) | n/a |
| `index.messages.variant_index` | KEEP | devtools/storage_correctness_scenario.py (+55 more outside-tier files) | n/a |
| `index.messages.is_active_path` | KEEP | devtools/storage_correctness_scenario.py (+36 more outside-tier files) | n/a |
| `index.messages.is_active_leaf` | KEEP | polylogue/sinex/material_adapter.py (+26 more outside-tier files) | n/a |
| `index.messages.word_count` | KEEP | devtools/proof_world_real_slice.py (+60 more outside-tier files) | n/a |
| `index.messages.input_tokens` | KEEP | devtools/claim_vs_evidence.py (+42 more outside-tier files) | n/a |
| `index.messages.output_tokens` | KEEP | devtools/claim_vs_evidence.py (+42 more outside-tier files) | n/a |
| `index.messages.cache_read_tokens` | KEEP | devtools/claim_vs_evidence.py (+38 more outside-tier files) | n/a |
| `index.messages.cache_write_tokens` | KEEP | polylogue/sinex/material_adapter.py (+38 more outside-tier files) | n/a |
| `index.messages.duration_ms` | KEEP | devtools/query_memory_budget.py (+58 more outside-tier files) | n/a |
| `index.messages.content_hash` | KEEP | devtools/storage_correctness_scenario.py (+44 more outside-tier files) | n/a |
| `index.messages.occurred_at_ms` | KEEP | polylogue/sinex/material_adapter.py (+30 more outside-tier files) | n/a |
| `index.messages.stop_reason` | KEEP | polylogue/core/enums.py (+17 more outside-tier files) | n/a |
| `index.messages_fts.block_id` | KEEP | polylogue/security/secret_scan.py (+40 more outside-tier files) | n/a |
| `index.messages_fts.message_id` | KEEP | devtools/lineage_validation.py (+139 more outside-tier files) | n/a |
| `index.messages_fts.session_id` | KEEP | devtools/resume_ranking_eval.py (+398 more outside-tier files) | n/a |
| `index.messages_fts.block_type` | KEEP | polylogue/sinex/material_adapter.py (+56 more outside-tier files) | n/a |
| `index.messages_fts.text` | KEEP | polylogue/agent_integration/spec.py (+393 more outside-tier files) | n/a |
| `index.messages_fts_identity.rowid` | KEEP | devtools/daemon_workload_probe.py (+43 more outside-tier files) | n/a |
| `index.messages_fts_identity.block_id` | KEEP | polylogue/security/secret_scan.py (+40 more outside-tier files) | n/a |
| `index.messages_fts_identity.source_hash` | KEEP | devtools/rebuild_safety_scenario.py (+6 more outside-tier files) | n/a |
| `index.messages_fts_identity.recipe_id` | KEEP | devtools/rebuild_safety_scenario.py (+2 more outside-tier files) | n/a |
| `index.paste_spans.message_id` | KEEP | devtools/lineage_validation.py (+139 more outside-tier files) | n/a |
| `index.paste_spans.session_id` | KEEP | devtools/resume_ranking_eval.py (+398 more outside-tier files) | n/a |
| `index.paste_spans.position` | KEEP | devtools/storage_correctness_scenario.py (+122 more outside-tier files) | n/a |
| `index.paste_spans.start_offset` | KEEP | polylogue/daemon/web_shell_paste.py (+8 more outside-tier files) | n/a |
| `index.paste_spans.end_offset` | KEEP | polylogue/sources/live/batch.py (+6 more outside-tier files) | n/a |
| `index.paste_spans.boundary_state` | KEEP | polylogue/sources/assembly_claude_code.py (+3 more outside-tier files) | n/a |
| `index.paste_spans.source_event_id` | KEEP | polylogue/sources/parsers/base_models.py | n/a |
| `index.paste_spans.source_marker` | KEEP | polylogue/sources/assembly_claude_code.py (+2 more outside-tier files) | n/a |
| `index.paste_spans.content_hash` | KEEP | devtools/storage_correctness_scenario.py (+44 more outside-tier files) | n/a |
| `index.paste_spans.observed_at_ms` | KEEP | devtools/daemon_workload_probe.py (+57 more outside-tier files) | n/a |
| `index.query_unit_frame_state.singleton` | KEEP | devtools/bead_cluster.py (+28 more outside-tier files) | n/a |
| `index.query_unit_frame_state.epoch` | KEEP | devtools/temporal_archive_aggregates.py (+31 more outside-tier files) | n/a |
| `index.raw_revision_applications.decision_id` | KEEP | devtools/raw_authority_scale_proof.py (+6 more outside-tier files) | n/a |
| `index.raw_revision_applications.raw_id` | KEEP | devtools/storage_correctness_scenario.py (+181 more outside-tier files) | n/a |
| `index.raw_revision_applications.session_id` | KEEP | devtools/resume_ranking_eval.py (+398 more outside-tier files) | n/a |
| `index.raw_revision_applications.logical_source_key` | KEEP | devtools/raw_authority_restart_proof.py (+51 more outside-tier files) | n/a |
| `index.raw_revision_applications.source_revision` | KEEP | devtools/raw_authority_scale_proof.py (+22 more outside-tier files) | n/a |
| `index.raw_revision_applications.acquisition_generation` | KEEP | devtools/raw_authority_scale_proof.py (+28 more outside-tier files) | n/a |
| `index.raw_revision_applications.decision` | KEEP | devtools/why.py (+114 more outside-tier files) | n/a |
| `index.raw_revision_applications.accepted_raw_id` | KEEP | polylogue/archive/revision_replay.py (+7 more outside-tier files) | n/a |
| `index.raw_revision_applications.accepted_source_revision` | KEEP | polylogue/archive/revision_replay.py (+6 more outside-tier files) | n/a |
| `index.raw_revision_applications.accepted_content_hash` | KEEP | polylogue/maintenance/cursor_authority_reconcile.py (+4 more outside-tier files) | n/a |
| `index.raw_revision_applications.baseline_raw_id` | KEEP | polylogue/archive/revision_authority.py (+23 more outside-tier files) | n/a |
| `index.raw_revision_applications.predecessor_raw_id` | KEEP | polylogue/sources/revision_backfill.py (+25 more outside-tier files) | n/a |
| `index.raw_revision_applications.append_end_offset` | KEEP | polylogue/archive/revision_authority.py (+20 more outside-tier files) | n/a |
| `index.raw_revision_applications.detail` | KEEP | polylogue/daemon_client.py (+218 more outside-tier files) | n/a |
| `index.raw_revision_applications.decided_at_ms` | KEEP | devtools/raw_authority_scale_proof.py (+14 more outside-tier files) | n/a |
| `index.raw_revision_heads.logical_source_key` | KEEP | devtools/raw_byte_duplicate_supersession_apply.py (+51 more outside-tier files) | n/a |
| `index.raw_revision_heads.session_id` | KEEP | devtools/command_catalog.py (+398 more outside-tier files) | n/a |
| `index.raw_revision_heads.accepted_raw_id` | KEEP | polylogue/storage/repair.py (+7 more outside-tier files) | n/a |
| `index.raw_revision_heads.accepted_source_revision` | KEEP | polylogue/maintenance/reindex_canary.py (+6 more outside-tier files) | n/a |
| `index.raw_revision_heads.accepted_content_hash` | KEEP | polylogue/storage/repair.py (+4 more outside-tier files) | n/a |
| `index.raw_revision_heads.accepted_frontier_kind` | KEEP | polylogue/archive/session_revision_membership.py (+5 more outside-tier files) | n/a |
| `index.raw_revision_heads.accepted_frontier` | KEEP | polylogue/storage/repair.py (+5 more outside-tier files) | n/a |
| `index.raw_revision_heads.acquisition_generation` | KEEP | devtools/raw_authority_scale_proof.py (+28 more outside-tier files) | n/a |
| `index.raw_revision_heads.append_end_offset` | KEEP | polylogue/maintenance/rebuild_index.py (+20 more outside-tier files) | n/a |
| `index.raw_revision_heads.decided_at_ms` | KEEP | devtools/raw_authority_scale_proof.py (+14 more outside-tier files) | n/a |
| `index.repo_checkouts.repo_id` | KEEP | polylogue/insights/correlation_view.py (+7 more outside-tier files) | n/a |
| `index.repo_checkouts.root_path` | KEEP | devtools/verify_layering.py (+6 more outside-tier files) | n/a |
| `index.repo_checkouts.first_seen_at_ms` | KEEP | polylogue/storage/insights/session/repo_observations.py | n/a |
| `index.repo_checkouts.last_seen_at_ms` | KEEP | polylogue/storage/insights/session/repo_observations.py | n/a |
| `index.repos.repo_id` | KEEP | polylogue/insights/session_label.py (+7 more outside-tier files) | n/a |
| `index.repos.origin_url` | KEEP | polylogue/insights/session_label.py (+2 more outside-tier files) | n/a |
| `index.repos.root_path` | KEEP | devtools/verify_layering.py (+6 more outside-tier files) | n/a |
| `index.repos.repo_name` | KEEP | polylogue/storage/insights/session/repo_observations.py (+7 more outside-tier files) | n/a |
| `index.repos.first_seen_at_ms` | KEEP | polylogue/storage/insights/session/repo_observations.py | n/a |
| `index.repos.last_seen_at_ms` | KEEP | polylogue/storage/insights/session/repo_observations.py | n/a |
| `index.session_agent_policies.session_id` | KEEP | devtools/resume_ranking_eval.py (+398 more outside-tier files) | n/a |
| `index.session_agent_policies.source_message_id` | KEEP | polylogue/material_protocol/v1/records.py (+13 more outside-tier files) | n/a |
| `index.session_agent_policies.position` | KEEP | devtools/storage_correctness_scenario.py (+122 more outside-tier files) | n/a |
| `index.session_agent_policies.approval_policy` | KEEP | polylogue/api/archive.py (+2 more outside-tier files) | n/a |
| `index.session_agent_policies.sandbox_policy` | KEEP | polylogue/storage/sqlite/queries/session_agent_policies.py (+2 more outside-tier files) | n/a |
| `index.session_agent_policies.network_policy` | KEEP | polylogue/api/archive.py (+2 more outside-tier files) | n/a |
| `index.session_agent_policies.observed_at_ms` | KEEP | devtools/daemon_workload_probe.py (+57 more outside-tier files) | n/a |
| `index.session_commits.session_id` | KEEP | polylogue/material_protocol/v1/records.py (+398 more outside-tier files) | n/a |
| `index.session_commits.commit_sha` | KEEP | polylogue/insights/correlation_view.py (+6 more outside-tier files) | n/a |
| `index.session_commits.repo_id` | KEEP | polylogue/storage/insights/session/repo_observations.py (+7 more outside-tier files) | n/a |
| `index.session_commits.detection_type` | KEEP | polylogue/api/archive.py (+4 more outside-tier files) | n/a |
| `index.session_commits.method` | KEEP | polylogue/daemon_client.py (+75 more outside-tier files) | n/a |
| `index.session_commits.confidence` | KEEP | polylogue/config.py (+116 more outside-tier files) | n/a |
| `index.session_commits.evidence_json` | KEEP | devtools/lineage_validation.py (+6 more outside-tier files) | n/a |
| `index.session_commits.created_at_ms` | KEEP | devtools/claim_vs_evidence_evidence.py (+68 more outside-tier files) | n/a |
| `index.session_events.session_id` | KEEP | devtools/resume_ranking_eval.py (+398 more outside-tier files) | n/a |
| `index.session_events.source_message_id` | KEEP | polylogue/api/archive.py (+13 more outside-tier files) | n/a |
| `index.session_events.source_message_provider_id` | KEEP | polylogue/sinex/material_adapter.py (+24 more outside-tier files) | n/a |
| `index.session_events.position` | KEEP | devtools/storage_correctness_scenario.py (+122 more outside-tier files) | n/a |
| `index.session_events.event_type` | KEEP | devtools/turso_probe.py (+63 more outside-tier files) | n/a |
| `index.session_events.summary` | KEEP | devtools/verify.py (+280 more outside-tier files) | n/a |
| `index.session_events.payload_json` | KEEP | devtools/daemon_workload_probe.py (+36 more outside-tier files) | n/a |
| `index.session_events.occurred_at_ms` | KEEP | devtools/affordance_usage.py (+30 more outside-tier files) | n/a |
| `index.session_latency_profiles.session_id` | KEEP | devtools/resume_ranking_eval.py (+398 more outside-tier files) | n/a |
| `index.session_latency_profiles.materializer_version` | KEEP | devtools/resume_ranking_eval.py (+44 more outside-tier files) | n/a |
| `index.session_latency_profiles.materialized_at` | KEEP | devtools/resume_ranking_eval.py (+34 more outside-tier files) | n/a |
| `index.session_latency_profiles.source_updated_at` | KEEP | devtools/resume_ranking_eval.py (+36 more outside-tier files) | n/a |
| `index.session_latency_profiles.source_sort_key` | KEEP | polylogue/insights/archive_models.py (+24 more outside-tier files) | n/a |
| `index.session_latency_profiles.input_high_water_mark` | KEEP | polylogue/daemon/http.py (+22 more outside-tier files) | n/a |
| `index.session_latency_profiles.input_high_water_mark_source` | KEEP | polylogue/daemon/http.py (+22 more outside-tier files) | n/a |
| `index.session_latency_profiles.input_row_count` | KEEP | polylogue/daemon/http.py (+19 more outside-tier files) | n/a |
| `index.session_latency_profiles.source_name` | KEEP | polylogue/agent_integration/spec.py (+145 more outside-tier files) | n/a |
| `index.session_latency_profiles.title` | KEEP | devtools/resume_ranking_eval.py (+227 more outside-tier files) | n/a |
| `index.session_latency_profiles.first_message_at` | KEEP | polylogue/insights/portfolio.py (+23 more outside-tier files) | n/a |
| `index.session_latency_profiles.last_message_at` | KEEP | devtools/resume_ranking_eval.py (+23 more outside-tier files) | n/a |
| `index.session_latency_profiles.canonical_session_date` | KEEP | polylogue/insights/archive_models.py (+26 more outside-tier files) | n/a |
| `index.session_latency_profiles.median_tool_call_ms` | KEEP | polylogue/archive/semantic/timing.py (+7 more outside-tier files) | n/a |
| `index.session_latency_profiles.p90_tool_call_ms` | KEEP | polylogue/insights/archive_models.py (+7 more outside-tier files) | n/a |
| `index.session_latency_profiles.max_tool_call_ms` | KEEP | polylogue/archive/semantic/timing.py (+7 more outside-tier files) | n/a |
| `index.session_latency_profiles.stuck_tool_count` | KEEP | polylogue/storage/insights/session/latency_profiles.py (+7 more outside-tier files) | n/a |
| `index.session_latency_profiles.median_agent_response_ms` | KEEP | polylogue/archive/semantic/timing.py (+6 more outside-tier files) | n/a |
| `index.session_latency_profiles.median_user_response_ms` | KEEP | polylogue/insights/archive_models.py (+6 more outside-tier files) | n/a |
| `index.session_latency_profiles.tool_call_count_by_category_json` | KEEP | polylogue/insights/archive.py (+4 more outside-tier files) | n/a |
| `index.session_latency_profiles.evidence_payload_json` | KEEP | polylogue/storage/sqlite/queries/session_insight_timeline_reads.py (+9 more outside-tier files) | n/a |
| `index.session_latency_profiles.search_text` | KEEP | devtools/daemon_workload_probe.py (+43 more outside-tier files) | n/a |
| `index.session_links.src_session_id` | KEEP | devtools/storage_correctness_scenario.py (+18 more outside-tier files) | n/a |
| `index.session_links.dst_origin` | KEEP | devtools/lineage_validation.py (+12 more outside-tier files) | n/a |
| `index.session_links.dst_native_id` | KEEP | devtools/lineage_validation.py (+15 more outside-tier files) | n/a |
| `index.session_links.link_type` | KEEP | devtools/lineage_validation.py (+19 more outside-tier files) | n/a |
| `index.session_links.resolved_dst_session_id` | KEEP | devtools/storage_correctness_scenario.py (+21 more outside-tier files) | n/a |
| `index.session_links.branch_point_message_id` | KEEP | devtools/storage_correctness_scenario.py (+9 more outside-tier files) | n/a |
| `index.session_links.inheritance` | KEEP | devtools/storage_correctness_scenario.py (+20 more outside-tier files) | n/a |
| `index.session_links.status` | KEEP | polylogue/daemon_client.py (+400 more outside-tier files) | n/a |
| `index.session_links.parent_tool_use_block_id` | KEEP | polylogue/archive/topology/edge.py (+4 more outside-tier files) | n/a |
| `index.session_links.method` | KEEP | polylogue/daemon_client.py (+75 more outside-tier files) | n/a |
| `index.session_links.confidence` | KEEP | polylogue/config.py (+116 more outside-tier files) | n/a |
| `index.session_links.evidence_json` | KEEP | devtools/lineage_validation.py (+6 more outside-tier files) | n/a |
| `index.session_links.observed_at_ms` | KEEP | devtools/daemon_workload_probe.py (+57 more outside-tier files) | n/a |
| `index.session_links.resolved_at_ms` | KEEP | devtools/daemon_workload_probe.py (+8 more outside-tier files) | n/a |
| `index.session_model_usage.session_id` | KEEP | devtools/resume_ranking_eval.py (+398 more outside-tier files) | n/a |
| `index.session_model_usage.model_name` | KEEP | devtools/claim_vs_evidence.py (+40 more outside-tier files) | n/a |
| `index.session_model_usage.input_tokens` | KEEP | devtools/claim_vs_evidence.py (+42 more outside-tier files) | n/a |
| `index.session_model_usage.output_tokens` | KEEP | devtools/claim_vs_evidence.py (+42 more outside-tier files) | n/a |
| `index.session_model_usage.cache_read_tokens` | KEEP | devtools/claim_vs_evidence.py (+38 more outside-tier files) | n/a |
| `index.session_model_usage.cache_write_tokens` | KEEP | devtools/claim_vs_evidence.py (+38 more outside-tier files) | n/a |
| `index.session_model_usage.message_count` | KEEP | devtools/agent_meta_sidecar_sweep_report.py (+127 more outside-tier files) | n/a |
| `index.session_model_usage.cost_usd` | KEEP | devtools/daemon_workload_probe.py (+19 more outside-tier files) | n/a |
| `index.session_model_usage.cost_credits` | KEEP | devtools/daemon_workload_probe.py (+8 more outside-tier files) | n/a |
| `index.session_model_usage.cost_provenance` | KEEP | devtools/claim_vs_evidence.py (+19 more outside-tier files) | n/a |
| `index.session_phases.session_id` | KEEP | devtools/command_catalog.py (+398 more outside-tier files) | n/a |
| `index.session_phases.position` | KEEP | devtools/storage_correctness_scenario.py (+122 more outside-tier files) | n/a |
| `index.session_phases.start_index` | KEEP | polylogue/archive/session/extraction.py (+21 more outside-tier files) | n/a |
| `index.session_phases.end_index` | KEEP | polylogue/archive/session/extraction.py (+20 more outside-tier files) | n/a |
| `index.session_phases.started_at_ms` | KEEP | devtools/storage_correctness_scenario.py (+29 more outside-tier files) | n/a |
| `index.session_phases.ended_at_ms` | KEEP | polylogue/pipeline/ids.py (+5 more outside-tier files) | n/a |
| `index.session_phases.duration_ms` | KEEP | devtools/query_memory_budget.py (+58 more outside-tier files) | n/a |
| `index.session_phases.tool_counts_json` | KEEP | polylogue/storage/insights/session/storage.py (+3 more outside-tier files) | n/a |
| `index.session_phases.word_count` | KEEP | devtools/scale_regression_probe.py (+60 more outside-tier files) | n/a |
| `index.session_phases.input_high_water_mark` | KEEP | polylogue/daemon/http.py (+22 more outside-tier files) | n/a |
| `index.session_phases.input_high_water_mark_source` | KEEP | polylogue/insights/temporal_source.py (+22 more outside-tier files) | n/a |
| `index.session_phases.evidence_json` | KEEP | devtools/lineage_validation.py (+6 more outside-tier files) | n/a |
| `index.session_phases.inference_json` | KEEP | polylogue/storage/insights/session/storage.py (+2 more outside-tier files) | n/a |
| `index.session_phases.search_text` | KEEP | devtools/daemon_workload_probe.py (+43 more outside-tier files) | n/a |
| `index.session_profiles.session_id` | KEEP | devtools/resume_ranking_eval.py (+398 more outside-tier files) | n/a |
| `index.session_profiles.logical_session_id` | KEEP | devtools/resume_ranking_eval.py (+18 more outside-tier files) | n/a |
| `index.session_profiles.materializer_version` | KEEP | devtools/resume_ranking_eval.py (+44 more outside-tier files) | n/a |
| `index.session_profiles.materialized_at` | KEEP | devtools/resume_ranking_eval.py (+34 more outside-tier files) | n/a |
| `index.session_profiles.source_updated_at` | KEEP | devtools/resume_ranking_eval.py (+36 more outside-tier files) | n/a |
| `index.session_profiles.source_sort_key` | KEEP | polylogue/insights/archive_summaries.py (+24 more outside-tier files) | n/a |
| `index.session_profiles.input_high_water_mark` | KEEP | polylogue/daemon/http.py (+22 more outside-tier files) | n/a |
| `index.session_profiles.input_high_water_mark_source` | KEEP | polylogue/insights/archive_models.py (+22 more outside-tier files) | n/a |
| `index.session_profiles.input_row_count` | KEEP | polylogue/insights/archive_summaries.py (+19 more outside-tier files) | n/a |
| `index.session_profiles.source_name` | KEEP | polylogue/agent_integration/spec.py (+145 more outside-tier files) | n/a |
| `index.session_profiles.title` | KEEP | devtools/resume_ranking_eval.py (+227 more outside-tier files) | n/a |
| `index.session_profiles.first_message_at` | KEEP | polylogue/insights/registry.py (+23 more outside-tier files) | n/a |
| `index.session_profiles.last_message_at` | KEEP | devtools/resume_ranking_eval.py (+23 more outside-tier files) | n/a |
| `index.session_profiles.canonical_session_date` | KEEP | polylogue/storage/insights/timeline/records.py (+26 more outside-tier files) | n/a |
| `index.session_profiles.repo_paths_json` | KEEP | polylogue/storage/insights/session/storage.py (+2 more outside-tier files) | n/a |
| `index.session_profiles.repo_names_json` | KEEP | polylogue/demo/seed.py (+4 more outside-tier files) | n/a |
| `index.session_profiles.tags_json` | KEEP | polylogue/storage/insights/session/status.py (+3 more outside-tier files) | n/a |
| `index.session_profiles.auto_tags_json` | KEEP | polylogue/storage/insights/session/status.py (+3 more outside-tier files) | n/a |
| `index.session_profiles.message_count` | KEEP | devtools/agent_meta_sidecar_sweep_report.py (+127 more outside-tier files) | n/a |
| `index.session_profiles.substantive_count` | KEEP | polylogue/insights/session_analytics.py (+12 more outside-tier files) | n/a |
| `index.session_profiles.attachment_count` | KEEP | polylogue/pipeline/services/ingest_worker.py (+15 more outside-tier files) | n/a |
| `index.session_profiles.work_event_count` | KEEP | devtools/daemon_workload_probe.py (+11 more outside-tier files) | n/a |
| `index.session_profiles.phase_count` | KEEP | devtools/temporal_read_profile.py (+16 more outside-tier files) | n/a |
| `index.session_profiles.word_count` | KEEP | devtools/proof_world_real_slice.py (+60 more outside-tier files) | n/a |
| `index.session_profiles.tool_use_count` | KEEP | devtools/scale_regression_probe.py (+24 more outside-tier files) | n/a |
| `index.session_profiles.thinking_count` | KEEP | devtools/scale_regression_probe.py (+23 more outside-tier files) | n/a |
| `index.session_profiles.total_cost_usd` | KEEP | polylogue/demo/seed.py (+26 more outside-tier files) | n/a |
| `index.session_profiles.total_duration_ms` | KEEP | devtools/pipeline_probe/result.py (+20 more outside-tier files) | n/a |
| `index.session_profiles.engaged_duration_ms` | KEEP | polylogue/archive/session/models.py (+10 more outside-tier files) | n/a |
| `index.session_profiles.tool_active_duration_ms` | KEEP | polylogue/archive/session/models.py (+10 more outside-tier files) | n/a |
| `index.session_profiles.wall_duration_ms` | KEEP | polylogue/archive/session/models.py (+24 more outside-tier files) | n/a |
| `index.session_profiles.workflow_shape` | KEEP | devtools/resume_ranking_eval.py (+21 more outside-tier files) | n/a |
| `index.session_profiles.workflow_shape_method` | PURGE | Zero production writer AND zero production reader: only appears in the DDL (index.py:1632) and 2 test files that SELECT the full row (test_archive_tiers_write.py:788, test_archive_tiers_archive.py:1772) without asserting a real value. Sibling columns workflow_shape/workflow_shape_confidence/workflow_shape_features_json are heavily read+written (archive/session/runtime.py:_workflow_shape, storage/insights/session/{rebuild,storage,profiles}.py, insights/{archive,resume,archive_rollups}.py, storage/sqlite/archive_tiers/archive.py, storage/sqlite/queries/mappers_insight_profiles.py) — workflow_shape_method alone was never wired to a producer. | n/a (no owning cleanup bead found; new finding — file one alongside the 818fy rebuild DDL window) |
| `index.session_profiles.workflow_shape_confidence` | KEEP | polylogue/archive/session/models.py (+10 more outside-tier files) | n/a |
| `index.session_profiles.workflow_shape_features_json` | KEEP | polylogue/storage/sqlite/queries/mappers_insight_profiles.py (+4 more outside-tier files) | n/a |
| `index.session_profiles.terminal_state` | KEEP | devtools/resume_ranking_eval.py (+32 more outside-tier files) | n/a |
| `index.session_profiles.terminal_state_method` | KEEP | polylogue/insights/archive_models.py (+9 more outside-tier files) | n/a |
| `index.session_profiles.terminal_state_confidence` | KEEP | polylogue/archive/session/models.py (+12 more outside-tier files) | n/a |
| `index.session_profiles.terminal_state_evidence_json` | KEEP | polylogue/storage/insights/session/rebuild.py (+4 more outside-tier files) | n/a |
| `index.session_profiles.cost_is_estimated` | KEEP | polylogue/storage/insights/session/rebuild.py (+12 more outside-tier files) | n/a |
| `index.session_profiles.thinking_duration_ms` | KEEP | polylogue/archive/session/models.py (+10 more outside-tier files) | n/a |
| `index.session_profiles.output_duration_ms` | KEEP | polylogue/archive/session/models.py (+9 more outside-tier files) | n/a |
| `index.session_profiles.tool_duration_ms` | KEEP | polylogue/insights/archive_models.py (+10 more outside-tier files) | n/a |
| `index.session_profiles.latency_percentiles_ms_json` | KEEP | polylogue/storage/insights/session/storage.py (+3 more outside-tier files) | n/a |
| `index.session_profiles.tool_calls_per_minute` | KEEP | polylogue/insights/rigor.py (+10 more outside-tier files) | n/a |
| `index.session_profiles.timing_provenance` | KEEP | polylogue/insights/timeline_renderer.py (+15 more outside-tier files) | n/a |
| `index.session_profiles.total_input_tokens` | KEEP | polylogue/insights/archive_models.py (+14 more outside-tier files) | n/a |
| `index.session_profiles.total_output_tokens` | KEEP | polylogue/archive/session/models.py (+14 more outside-tier files) | n/a |
| `index.session_profiles.total_cache_read_tokens` | KEEP | polylogue/insights/portfolio.py (+14 more outside-tier files) | n/a |
| `index.session_profiles.total_cache_write_tokens` | KEEP | polylogue/archive/semantic/cost_compute.py (+14 more outside-tier files) | n/a |
| `index.session_profiles.total_credit_cost` | KEEP | polylogue/insights/archive_models.py (+11 more outside-tier files) | n/a |
| `index.session_profiles.cost_provenance` | KEEP | devtools/claim_vs_evidence.py (+19 more outside-tier files) | n/a |
| `index.session_profiles.per_model_cost_json` | KEEP | polylogue/storage/insights/session/rebuild.py (+7 more outside-tier files) | n/a |
| `index.session_profiles.evidence_payload_json` | KEEP | polylogue/insights/archive.py (+9 more outside-tier files) | n/a |
| `index.session_profiles.inference_payload_json` | KEEP | devtools/scale_regression_probe.py (+4 more outside-tier files) | n/a |
| `index.session_profiles.enrichment_payload_json` | KEEP | polylogue/storage/insights/session/storage.py (+1 more outside-tier files) | n/a |
| `index.session_profiles.evidence_search_text` | KEEP | polylogue/storage/sqlite/queries/session_insight_profile_reads.py (+5 more outside-tier files) | n/a |
| `index.session_profiles.inference_search_text` | KEEP | polylogue/storage/insights/session/rebuild.py (+5 more outside-tier files) | n/a |
| `index.session_profiles.enrichment_search_text` | KEEP | polylogue/storage/insights/session/storage.py (+5 more outside-tier files) | n/a |
| `index.session_profiles.enrichment_version` | KEEP | polylogue/insights/archive_models.py (+6 more outside-tier files) | n/a |
| `index.session_profiles.enrichment_family` | KEEP | polylogue/insights/archive_models.py (+5 more outside-tier files) | n/a |
| `index.session_profiles.inference_version` | KEEP | polylogue/storage/insights/timeline/records.py (+12 more outside-tier files) | n/a |
| `index.session_profiles.inference_family` | KEEP | polylogue/insights/archive_models.py (+9 more outside-tier files) | n/a |
| `index.session_profiles.search_text` | KEEP | devtools/daemon_workload_probe.py (+43 more outside-tier files) | n/a |
| `index.session_profiles.duration_ms` | KEEP | devtools/query_memory_budget.py (+58 more outside-tier files) | n/a |
| `index.session_profiles.cost_credits` | KEEP | devtools/daemon_workload_probe.py (+8 more outside-tier files) | n/a |
| `index.session_profiles.cost_usd` | KEEP | devtools/daemon_workload_probe.py (+19 more outside-tier files) | n/a |
| `index.session_profiles.priced_with` | KEEP | polylogue/archive/session/models.py (+6 more outside-tier files) | n/a |
| `index.session_profiles.priced_at_ms` | KEEP | devtools/rebuild_safety_scenario.py (+4 more outside-tier files) | n/a |
| `index.session_profiles.primary_model_name` | KEEP | polylogue/archive/session/models.py (+7 more outside-tier files) | n/a |
| `index.session_profiles.primary_model_family` | KEEP | polylogue/storage/insights/session/rebuild.py (+7 more outside-tier files) | n/a |
| `index.session_provider_usage_events.session_id` | KEEP | devtools/resume_ranking_eval.py (+398 more outside-tier files) | n/a |
| `index.session_provider_usage_events.source_message_id` | KEEP | polylogue/material_protocol/v1/records.py (+13 more outside-tier files) | n/a |
| `index.session_provider_usage_events.position` | KEEP | devtools/storage_correctness_scenario.py (+122 more outside-tier files) | n/a |
| `index.session_provider_usage_events.provider_event_type` | KEEP | devtools/claim_vs_evidence.py (+1 more outside-tier files) | n/a |
| `index.session_provider_usage_events.model_name` | KEEP | devtools/claim_vs_evidence.py (+40 more outside-tier files) | n/a |
| `index.session_provider_usage_events.last_input_tokens` | KEEP | polylogue/storage/usage.py | n/a |
| `index.session_provider_usage_events.last_output_tokens` | KEEP | polylogue/storage/usage.py | n/a |
| `index.session_provider_usage_events.last_cached_input_tokens` | KEEP | polylogue/storage/usage.py | n/a |
| `index.session_provider_usage_events.last_cache_write_tokens` | KEEP | polylogue/storage/usage.py | n/a |
| `index.session_provider_usage_events.last_reasoning_output_tokens` | KEEP | devtools/claim_vs_evidence.py (+1 more outside-tier files) | n/a |
| `index.session_provider_usage_events.last_total_tokens` | KEEP | polylogue/storage/usage.py | n/a |
| `index.session_provider_usage_events.total_input_tokens` | KEEP | polylogue/insights/archive_models.py (+14 more outside-tier files) | n/a |
| `index.session_provider_usage_events.total_output_tokens` | KEEP | polylogue/storage/usage.py (+14 more outside-tier files) | n/a |
| `index.session_provider_usage_events.total_cached_input_tokens` | KEEP | polylogue/storage/usage.py | n/a |
| `index.session_provider_usage_events.total_cache_write_tokens` | KEEP | polylogue/archive/session/models.py (+14 more outside-tier files) | n/a |
| `index.session_provider_usage_events.total_reasoning_output_tokens` | KEEP | polylogue/storage/usage.py | n/a |
| `index.session_provider_usage_events.total_tokens` | KEEP | devtools/cost_reconciliation_probe.py (+16 more outside-tier files) | n/a |
| `index.session_provider_usage_events.occurred_at_ms` | KEEP | polylogue/sinex/material_adapter.py (+30 more outside-tier files) | n/a |
| `index.session_refs.session_id` | KEEP | devtools/resume_ranking_eval.py (+398 more outside-tier files) | n/a |
| `index.session_refs.position` | KEEP | devtools/storage_correctness_scenario.py (+122 more outside-tier files) | n/a |
| `index.session_refs.kind` | KEEP | polylogue/config.py (+334 more outside-tier files) | n/a |
| `index.session_refs.repo` | KEEP | devtools/resume_ranking_eval.py (+123 more outside-tier files) | n/a |
| `index.session_refs.ref_number` | KEEP | polylogue/storage/sqlite/queries/mappers_archive.py (+1 more outside-tier files) | n/a |
| `index.session_refs.url` | KEEP | polylogue/config.py (+62 more outside-tier files) | n/a |
| `index.session_refs.observed_at_ms` | KEEP | devtools/daemon_workload_probe.py (+57 more outside-tier files) | n/a |
| `index.session_repos.session_id` | KEEP | devtools/resume_ranking_eval.py (+398 more outside-tier files) | n/a |
| `index.session_repos.repo_id` | KEEP | polylogue/storage/runtime/archive/records.py (+7 more outside-tier files) | n/a |
| `index.session_repos.root_path` | KEEP | devtools/verify_layering.py (+6 more outside-tier files) | n/a |
| `index.session_repos.branch_name` | KEEP | polylogue/storage/insights/session/repo_observations.py (+1 more outside-tier files) | n/a |
| `index.session_repos.observed_at_ms` | KEEP | devtools/daemon_workload_probe.py (+57 more outside-tier files) | n/a |
| `index.session_tag_rollups.tag` | KEEP | polylogue/agent_integration/spec.py (+92 more outside-tier files) | n/a |
| `index.session_tag_rollups.bucket_day` | KEEP | polylogue/storage/insights/session/refresh.py (+8 more outside-tier files) | n/a |
| `index.session_tag_rollups.source_name` | KEEP | polylogue/agent_integration/spec.py (+145 more outside-tier files) | n/a |
| `index.session_tag_rollups.materializer_version` | KEEP | devtools/resume_ranking_eval.py (+44 more outside-tier files) | n/a |
| `index.session_tag_rollups.materialized_at` | KEEP | devtools/resume_ranking_eval.py (+34 more outside-tier files) | n/a |
| `index.session_tag_rollups.source_updated_at` | KEEP | devtools/resume_ranking_eval.py (+36 more outside-tier files) | n/a |
| `index.session_tag_rollups.source_sort_key` | KEEP | polylogue/daemon/http.py (+24 more outside-tier files) | n/a |
| `index.session_tag_rollups.input_high_water_mark` | KEEP | polylogue/storage/insights/timeline/records.py (+22 more outside-tier files) | n/a |
| `index.session_tag_rollups.input_high_water_mark_source` | KEEP | polylogue/daemon/http.py (+22 more outside-tier files) | n/a |
| `index.session_tag_rollups.input_row_count` | KEEP | polylogue/daemon/http.py (+19 more outside-tier files) | n/a |
| `index.session_tag_rollups.session_count` | KEEP | devtools/daemon_workload_probe.py (+84 more outside-tier files) | n/a |
| `index.session_tag_rollups.logical_session_count` | KEEP | devtools/cost_reconciliation_probe.py (+11 more outside-tier files) | n/a |
| `index.session_tag_rollups.logical_session_ids_json` | KEEP | polylogue/storage/insights/session/storage.py (+2 more outside-tier files) | n/a |
| `index.session_tag_rollups.explicit_count` | KEEP | polylogue/storage/insights/aggregate/records.py (+8 more outside-tier files) | n/a |
| `index.session_tag_rollups.auto_count` | KEEP | polylogue/insights/archive.py (+8 more outside-tier files) | n/a |
| `index.session_tag_rollups.repo_breakdown_json` | KEEP | polylogue/storage/insights/session/storage.py (+2 more outside-tier files) | n/a |
| `index.session_tag_rollups.search_text` | KEEP | devtools/daemon_workload_probe.py (+43 more outside-tier files) | n/a |
| `index.session_tags.session_id` | KEEP | devtools/resume_ranking_eval.py (+398 more outside-tier files) | n/a |
| `index.session_tags.tag` | KEEP | devtools/antigravity_phantom_sweep_report.py (+92 more outside-tier files) | n/a |
| `index.session_tags.tag_source` | KEEP | polylogue/sources/parsers/base_models.py | n/a |
| `index.session_tags.method` | KEEP | polylogue/daemon_client.py (+75 more outside-tier files) | n/a |
| `index.session_tags.confidence` | KEEP | polylogue/config.py (+116 more outside-tier files) | n/a |
| `index.session_tags.evidence_json` | KEEP | devtools/lineage_validation.py (+6 more outside-tier files) | n/a |
| `index.session_work_events.session_id` | KEEP | devtools/resume_ranking_eval.py (+398 more outside-tier files) | n/a |
| `index.session_work_events.position` | KEEP | devtools/storage_correctness_scenario.py (+122 more outside-tier files) | n/a |
| `index.session_work_events.work_event_type` | KEEP | polylogue/insights/rigor.py (+5 more outside-tier files) | n/a |
| `index.session_work_events.summary` | KEEP | devtools/verify.py (+280 more outside-tier files) | n/a |
| `index.session_work_events.confidence` | KEEP | polylogue/config.py (+116 more outside-tier files) | n/a |
| `index.session_work_events.start_index` | KEEP | polylogue/api/archive.py (+21 more outside-tier files) | n/a |
| `index.session_work_events.end_index` | KEEP | polylogue/archive/session/extraction.py (+20 more outside-tier files) | n/a |
| `index.session_work_events.started_at_ms` | KEEP | devtools/storage_correctness_scenario.py (+29 more outside-tier files) | n/a |
| `index.session_work_events.ended_at_ms` | KEEP | polylogue/pipeline/ids.py (+5 more outside-tier files) | n/a |
| `index.session_work_events.duration_ms` | KEEP | devtools/query_memory_budget.py (+58 more outside-tier files) | n/a |
| `index.session_work_events.file_paths_json` | KEEP | polylogue/storage/insights/session/storage.py (+3 more outside-tier files) | n/a |
| `index.session_work_events.tools_used_json` | KEEP | polylogue/storage/insights/session/storage.py (+3 more outside-tier files) | n/a |
| `index.session_work_events.input_high_water_mark` | KEEP | polylogue/storage/insights/timeline/records.py (+22 more outside-tier files) | n/a |
| `index.session_work_events.input_high_water_mark_source` | KEEP | polylogue/insights/provenance.py (+22 more outside-tier files) | n/a |
| `index.session_work_events.evidence_json` | KEEP | devtools/lineage_validation.py (+6 more outside-tier files) | n/a |
| `index.session_work_events.inference_json` | KEEP | polylogue/storage/insights/session/storage.py (+2 more outside-tier files) | n/a |
| `index.session_work_events.search_text` | KEEP | devtools/daemon_workload_probe.py (+43 more outside-tier files) | n/a |
| `index.session_work_events_fts.event_id` | KEEP | devtools/turso_probe.py (+39 more outside-tier files) | n/a |
| `index.session_work_events_fts.session_id` | KEEP | devtools/resume_ranking_eval.py (+398 more outside-tier files) | n/a |
| `index.session_work_events_fts.work_event_type` | KEEP | polylogue/insights/rigor.py (+5 more outside-tier files) | n/a |
| `index.session_work_events_fts.text` | KEEP | polylogue/agent_integration/spec.py (+393 more outside-tier files) | n/a |
| `index.session_working_dirs.session_id` | KEEP | devtools/resume_ranking_eval.py (+398 more outside-tier files) | n/a |
| `index.session_working_dirs.path` | KEEP | polylogue/daemon_client.py (+617 more outside-tier files) | n/a |
| `index.session_working_dirs.position` | KEEP | devtools/storage_correctness_scenario.py (+122 more outside-tier files) | n/a |
| `index.sessions.native_id` | KEEP | devtools/storage_correctness_scenario.py (+97 more outside-tier files) | n/a |
| `index.sessions.origin` | KEEP | polylogue/config.py (+396 more outside-tier files) | n/a |
| `index.sessions.parent_session_id` | KEEP | devtools/resume_ranking_eval.py (+48 more outside-tier files) | n/a |
| `index.sessions.root_session_id` | KEEP | devtools/temporal_archive_aggregates.py (+11 more outside-tier files) | n/a |
| `index.sessions.raw_id` | KEEP | devtools/storage_correctness_scenario.py (+181 more outside-tier files) | n/a |
| `index.sessions.parser_fingerprint` | KEEP | devtools/daemon_live_benchmark.py (+21 more outside-tier files) | n/a |
| `index.sessions.lowering_fingerprint` | KEEP | devtools/index_fast_forward.py (+6 more outside-tier files) | n/a |
| `index.sessions.branch_type` | KEEP | devtools/storage_correctness_scenario.py (+42 more outside-tier files) | n/a |
| `index.sessions.active_leaf_message_id` | KEEP | polylogue/mcp/payloads.py (+5 more outside-tier files) | n/a |
| `index.sessions.title` | KEEP | devtools/resume_ranking_eval.py (+227 more outside-tier files) | n/a |
| `index.sessions.session_kind` | KEEP | devtools/read_package.py (+18 more outside-tier files) | n/a |
| `index.sessions.title_source` | KEEP | polylogue/sinex/material_adapter.py (+24 more outside-tier files) | n/a |
| `index.sessions.title_ref` | KEEP | polylogue/api/archive.py (+12 more outside-tier files) | n/a |
| `index.sessions.title_confidence` | KEEP | polylogue/sources/dispatch.py (+11 more outside-tier files) | n/a |
| `index.sessions.display_name` | KEEP | devtools/affordance_usage.py (+31 more outside-tier files) | n/a |
| `index.sessions.run_settings_json` | KEEP | polylogue/sources/origin_specs.py (+4 more outside-tier files) | n/a |
| `index.sessions.pending_drafts_json` | KEEP | polylogue/sources/origin_specs.py (+7 more outside-tier files) | n/a |
| `index.sessions.git_branch` | KEEP | polylogue/coordination/envelope.py (+38 more outside-tier files) | n/a |
| `index.sessions.git_repository_url` | KEEP | polylogue/context/preamble.py (+26 more outside-tier files) | n/a |
| `index.sessions.provider_project_ref` | KEEP | polylogue/sinex/material_adapter.py (+17 more outside-tier files) | n/a |
| `index.sessions.commit_hash` | KEEP | polylogue/sources/providers/codex.py (+4 more outside-tier files) | n/a |
| `index.sessions.instructions_text` | KEEP | polylogue/sinex/material_adapter.py (+7 more outside-tier files) | n/a |
| `index.sessions.reported_duration_ms` | KEEP | polylogue/sinex/material_adapter.py (+10 more outside-tier files) | n/a |
| `index.sessions.reported_cost_usd` | KEEP | polylogue/sinex/material_adapter.py (+12 more outside-tier files) | n/a |
| `index.sessions.message_count` | KEEP | devtools/proof_world_real_slice.py (+127 more outside-tier files) | n/a |
| `index.sessions.word_count` | KEEP | devtools/proof_world_real_slice.py (+60 more outside-tier files) | n/a |
| `index.sessions.tool_use_count` | KEEP | devtools/scale_regression_probe.py (+24 more outside-tier files) | n/a |
| `index.sessions.thinking_count` | KEEP | devtools/scale_regression_probe.py (+23 more outside-tier files) | n/a |
| `index.sessions.paste_count` | KEEP | polylogue/sources/assembly_claude_code.py (+8 more outside-tier files) | n/a |
| `index.sessions.user_message_count` | KEEP | polylogue/archive/query/metadata.py (+8 more outside-tier files) | n/a |
| `index.sessions.authored_user_message_count` | KEEP | polylogue/hooks/__init__.py (+9 more outside-tier files) | n/a |
| `index.sessions.assistant_message_count` | KEEP | polylogue/insights/rigor.py (+8 more outside-tier files) | n/a |
| `index.sessions.system_message_count` | KEEP | polylogue/storage/sqlite/async_sqlite.py (+4 more outside-tier files) | n/a |
| `index.sessions.tool_message_count` | KEEP | polylogue/storage/insights/session/rebuild.py (+5 more outside-tier files) | n/a |
| `index.sessions.user_word_count` | KEEP | polylogue/storage/insights/session/rebuild.py (+4 more outside-tier files) | n/a |
| `index.sessions.authored_user_word_count` | KEEP | polylogue/schemas/generation/archive_workload_profile.py (+3 more outside-tier files) | n/a |
| `index.sessions.assistant_word_count` | KEEP | polylogue/archive/query/metadata.py (+4 more outside-tier files) | n/a |
| `index.sessions.content_hash` | KEEP | devtools/storage_correctness_scenario.py (+44 more outside-tier files) | n/a |
| `index.sessions.created_at_ms` | KEEP | polylogue/security/secret_scan.py (+68 more outside-tier files) | n/a |
| `index.sessions.updated_at_ms` | KEEP | devtools/turso_probe.py (+61 more outside-tier files) | n/a |
| `index.thread_sessions.thread_id` | KEEP | devtools/cost_reconciliation_probe.py (+35 more outside-tier files) | n/a |
| `index.thread_sessions.session_id` | KEEP | devtools/resume_ranking_eval.py (+398 more outside-tier files) | n/a |
| `index.thread_sessions.position` | KEEP | devtools/storage_correctness_scenario.py (+122 more outside-tier files) | n/a |
| `index.threads.thread_id` | KEEP | devtools/merge_gate.py (+35 more outside-tier files) | n/a |
| `index.threads.dominant_repo_id` | PURGE | Zero production writer AND zero production reader: only appears in DDL (index.py:1148) and 1 test harness broad SELECT (tests/infra/convergence_harness.py:847). The live column is the plain-TEXT sibling `dominant_repo` (heavily used: archive/session/threads.py, daemon/http.py:1074, daemon/web_shell.py:2585, insights/{archive,resume,registry,rigor}.py, storage/insights/session/{threads,storage}.py, storage/sqlite/queries/mappers_insight_timelines.py). `dominant_repo_id` (FK to repos.repo_id) was declared for a repo-identity join that was never implemented at the write side. | n/a (no owning cleanup bead found; new finding — file one alongside the 818fy rebuild DDL window) |
| `index.threads.materializer_version` | KEEP | devtools/resume_ranking_eval.py (+44 more outside-tier files) | n/a |
| `index.threads.materialized_at` | KEEP | devtools/resume_ranking_eval.py (+34 more outside-tier files) | n/a |
| `index.threads.source_updated_at` | KEEP | devtools/resume_ranking_eval.py (+36 more outside-tier files) | n/a |
| `index.threads.input_high_water_mark` | KEEP | polylogue/insights/provenance.py (+22 more outside-tier files) | n/a |
| `index.threads.input_high_water_mark_source` | KEEP | polylogue/storage/sqlite/queries/session_insight_timeline_reads.py (+22 more outside-tier files) | n/a |
| `index.threads.input_row_count` | KEEP | polylogue/daemon/http.py (+19 more outside-tier files) | n/a |
| `index.threads.start_time` | KEEP | polylogue/core/metrics.py (+20 more outside-tier files) | n/a |
| `index.threads.end_time` | KEEP | polylogue/core/metrics.py (+21 more outside-tier files) | n/a |
| `index.threads.dominant_repo` | KEEP | polylogue/cli/shared/resume_rendering.py (+14 more outside-tier files) | n/a |
| `index.threads.session_ids_json` | KEEP | polylogue/api/archive.py (+6 more outside-tier files) | n/a |
| `index.threads.session_count` | KEEP | devtools/daemon_workload_probe.py (+84 more outside-tier files) | n/a |
| `index.threads.depth` | KEEP | devtools/verify_doc_commands.py (+59 more outside-tier files) | n/a |
| `index.threads.branch_count` | KEEP | polylogue/archive/session/documents.py (+10 more outside-tier files) | n/a |
| `index.threads.total_messages` | KEEP | polylogue/archive/coverage.py (+36 more outside-tier files) | n/a |
| `index.threads.total_cost_usd` | KEEP | polylogue/demo/seed.py (+26 more outside-tier files) | n/a |
| `index.threads.wall_duration_ms` | KEEP | polylogue/archive/session/models.py (+24 more outside-tier files) | n/a |
| `index.threads.work_event_breakdown_json` | KEEP | polylogue/storage/insights/session/storage.py (+3 more outside-tier files) | n/a |
| `index.threads.payload_json` | KEEP | devtools/daemon_workload_probe.py (+36 more outside-tier files) | n/a |
| `index.threads.search_text` | KEEP | devtools/daemon_workload_probe.py (+43 more outside-tier files) | n/a |
| `index.threads.created_at_ms` | KEEP | polylogue/security/secret_scan.py (+68 more outside-tier files) | n/a |
| `index.web_content_constructs.session_id` | KEEP | devtools/resume_ranking_eval.py (+398 more outside-tier files) | n/a |
| `index.web_content_constructs.message_id` | KEEP | devtools/lineage_validation.py (+139 more outside-tier files) | n/a |
| `index.web_content_constructs.block_id` | KEEP | polylogue/security/secret_scan.py (+40 more outside-tier files) | n/a |
| `index.web_content_constructs.position` | KEEP | devtools/storage_correctness_scenario.py (+122 more outside-tier files) | n/a |
| `index.web_content_constructs.provider` | KEEP | polylogue/config.py (+391 more outside-tier files) | n/a |
| `index.web_content_constructs.construct_type` | KEEP | polylogue/demo/constructs.py (+11 more outside-tier files) | n/a |
| `index.web_content_constructs.provider_key` | KEEP | devtools/dev_loop.py (+9 more outside-tier files) | n/a |
| `index.web_content_constructs.title` | KEEP | polylogue/agent_integration/spec.py (+227 more outside-tier files) | n/a |
| `index.web_content_constructs.url` | KEEP | polylogue/config.py (+62 more outside-tier files) | n/a |
| `index.web_content_constructs.text` | KEEP | polylogue/agent_integration/spec.py (+393 more outside-tier files) | n/a |
| `index.web_content_constructs.source_id` | KEEP | devtools/pr_scope.py (+9 more outside-tier files) | n/a |
| `index.web_content_constructs.group_id` | KEEP | polylogue/sources/dispatch.py (+6 more outside-tier files) | n/a |
| `index.web_content_constructs.group_title` | KEEP | polylogue/sources/parsers/base_support.py (+6 more outside-tier files) | n/a |
| `index.web_content_constructs.query` | KEEP | polylogue/daemon_client.py (+380 more outside-tier files) | n/a |
| `index.web_content_constructs.asset_pointer` | KEEP | polylogue/storage/runtime/archive/records.py (+8 more outside-tier files) | n/a |
| `index.web_content_constructs.mime_type` | KEEP | devtools/attachment_reacquisition_report.py (+40 more outside-tier files) | n/a |
| `index.web_content_constructs.status` | KEEP | polylogue/daemon_client.py (+400 more outside-tier files) | n/a |
| `index.web_content_constructs.task_id` | KEEP | polylogue/insights/run_projection.py (+15 more outside-tier files) | n/a |
| `index.web_content_constructs.task_type` | KEEP | polylogue/storage/runtime/archive/records.py (+5 more outside-tier files) | n/a |
| `index.web_content_constructs.rank` | KEEP | devtools/resume_ranking_eval.py (+54 more outside-tier files) | n/a |
| `index.web_content_constructs.start_index` | KEEP | polylogue/insights/archive_models.py (+21 more outside-tier files) | n/a |
| `index.web_content_constructs.end_index` | KEEP | polylogue/storage/insights/timeline/records.py (+20 more outside-tier files) | n/a |
| `index.work_evidence_edges.graph_id` | KEEP | polylogue/storage/query_models.py (+12 more outside-tier files) | n/a |
| `index.work_evidence_edges.edge_ref` | KEEP | polylogue/storage/sqlite/queries/work_evidence.py (+5 more outside-tier files) | n/a |
| `index.work_evidence_edges.edge_kind` | KEEP | polylogue/insights/claude_workflow_materializer.py (+6 more outside-tier files) | n/a |
| `index.work_evidence_edges.source_ref` | KEEP | polylogue/telemetry/otel_projection.py (+16 more outside-tier files) | n/a |
| `index.work_evidence_edges.target_ref` | KEEP | polylogue/agent_integration/spec.py (+63 more outside-tier files) | n/a |
| `index.work_evidence_edges.evidence_refs_json` | KEEP | polylogue/coordination/envelope.py (+6 more outside-tier files) | n/a |
| `index.work_evidence_edges.corpus_snapshot_ref` | KEEP | polylogue/insights/claude_workflow_materializer.py (+7 more outside-tier files) | n/a |
| `index.work_evidence_edges.authority` | KEEP | polylogue/config.py (+197 more outside-tier files) | n/a |
| `index.work_evidence_edges.confidence` | KEEP | polylogue/config.py (+116 more outside-tier files) | n/a |
| `index.work_evidence_edges.occurred_at_ms` | KEEP | polylogue/sinex/material_adapter.py (+30 more outside-tier files) | n/a |
| `index.work_evidence_edges.association_state` | KEEP | polylogue/insights/claude_workflow_materializer.py (+6 more outside-tier files) | n/a |
| `index.work_evidence_graphs.graph_id` | KEEP | polylogue/insights/claude_workflow_materializer.py (+12 more outside-tier files) | n/a |
| `index.work_evidence_graphs.corpus_snapshot_ref` | KEEP | polylogue/operations/incident_evidence_materialization.py (+7 more outside-tier files) | n/a |
| `index.work_evidence_nodes.graph_id` | KEEP | polylogue/operations/incident_evidence_materialization.py (+12 more outside-tier files) | n/a |
| `index.work_evidence_nodes.node_ref` | KEEP | polylogue/insights/claude_workflow_materializer.py (+1 more outside-tier files) | n/a |
| `index.work_evidence_nodes.node_kind` | KEEP | polylogue/insights/claude_workflow_materializer.py (+1 more outside-tier files) | n/a |
| `index.work_evidence_nodes.label` | KEEP | devtools/resume_ranking_eval.py (+195 more outside-tier files) | n/a |
| `index.work_evidence_nodes.evidence_refs_json` | KEEP | polylogue/insights/claude_workflow_materializer.py (+6 more outside-tier files) | n/a |
| `index.work_evidence_nodes.corpus_snapshot_ref` | KEEP | polylogue/insights/claude_workflow_materializer.py (+7 more outside-tier files) | n/a |
| `index.work_evidence_nodes.authority` | KEEP | polylogue/config.py (+197 more outside-tier files) | n/a |
| `index.work_evidence_nodes.confidence` | KEEP | polylogue/config.py (+116 more outside-tier files) | n/a |
| `index.work_evidence_nodes.occurred_at_ms` | KEEP | polylogue/sinex/material_adapter.py (+30 more outside-tier files) | n/a |
| `index.work_evidence_nodes.actor_ref` | KEEP | polylogue/annotations/importer.py (+25 more outside-tier files) | n/a |
| `index.work_evidence_nodes.execution_context_id` | KEEP | polylogue/insights/claude_workflow_materializer.py (+8 more outside-tier files) | n/a |
| `index.work_evidence_nodes.execution_context_known_json` | KEEP | polylogue/insights/claude_workflow_materializer.py (+1 more outside-tier files) | n/a |
| `index.work_evidence_nodes.execution_context_unknown_json` | KEEP | polylogue/insights/claude_workflow_materializer.py (+1 more outside-tier files) | n/a |
| `index.work_evidence_nodes.execution_context_addressed` | KEEP | polylogue/insights/claude_workflow_materializer.py (+1 more outside-tier files) | n/a |
| `index.work_evidence_nodes.association_state` | KEEP | polylogue/insights/claude_workflow_materializer.py (+6 more outside-tier files) | n/a |
| `index.work_evidence_nodes.claim_text` | KEEP | devtools/continuity_evidence.py (+8 more outside-tier files) | n/a |
| `index.messages_fts_config.k` | KEEP | SQLite FTS5 virtual-table implementation storage; inherits KEEP from parent `messages_fts` (confirmed KEEP above). Not independently classified per the seed's rule (the correct removal trigger is dropping the parent FTS object, not a standalone column PURGE on shadow tables — see the `threads_fts` precedent in f2bad6e62). | n/a |
| `index.messages_fts_config.v` | KEEP | SQLite FTS5 virtual-table implementation storage; inherits KEEP from parent `messages_fts` (confirmed KEEP above). Not independently classified per the seed's rule (the correct removal trigger is dropping the parent FTS object, not a standalone column PURGE on shadow tables — see the `threads_fts` precedent in f2bad6e62). | n/a |
| `index.messages_fts_data.id` | KEEP | SQLite FTS5 virtual-table implementation storage; inherits KEEP from parent `messages_fts` (confirmed KEEP above). Not independently classified per the seed's rule (the correct removal trigger is dropping the parent FTS object, not a standalone column PURGE on shadow tables — see the `threads_fts` precedent in f2bad6e62). | n/a |
| `index.messages_fts_data.block` | KEEP | SQLite FTS5 virtual-table implementation storage; inherits KEEP from parent `messages_fts` (confirmed KEEP above). Not independently classified per the seed's rule (the correct removal trigger is dropping the parent FTS object, not a standalone column PURGE on shadow tables — see the `threads_fts` precedent in f2bad6e62). | n/a |
| `index.messages_fts_docsize.id` | KEEP | SQLite FTS5 virtual-table implementation storage; inherits KEEP from parent `messages_fts` (confirmed KEEP above). Not independently classified per the seed's rule (the correct removal trigger is dropping the parent FTS object, not a standalone column PURGE on shadow tables — see the `threads_fts` precedent in f2bad6e62). | n/a |
| `index.messages_fts_docsize.sz` | KEEP | SQLite FTS5 virtual-table implementation storage; inherits KEEP from parent `messages_fts` (confirmed KEEP above). Not independently classified per the seed's rule (the correct removal trigger is dropping the parent FTS object, not a standalone column PURGE on shadow tables — see the `threads_fts` precedent in f2bad6e62). | n/a |
| `index.messages_fts_docsize.origin` | KEEP | SQLite FTS5 virtual-table implementation storage; inherits KEEP from parent `messages_fts` (confirmed KEEP above). Not independently classified per the seed's rule (the correct removal trigger is dropping the parent FTS object, not a standalone column PURGE on shadow tables — see the `threads_fts` precedent in f2bad6e62). | n/a |
| `index.messages_fts_idx.segid` | KEEP | SQLite FTS5 virtual-table implementation storage; inherits KEEP from parent `messages_fts` (confirmed KEEP above). Not independently classified per the seed's rule (the correct removal trigger is dropping the parent FTS object, not a standalone column PURGE on shadow tables — see the `threads_fts` precedent in f2bad6e62). | n/a |
| `index.messages_fts_idx.term` | KEEP | SQLite FTS5 virtual-table implementation storage; inherits KEEP from parent `messages_fts` (confirmed KEEP above). Not independently classified per the seed's rule (the correct removal trigger is dropping the parent FTS object, not a standalone column PURGE on shadow tables — see the `threads_fts` precedent in f2bad6e62). | n/a |
| `index.messages_fts_idx.pgno` | KEEP | SQLite FTS5 virtual-table implementation storage; inherits KEEP from parent `messages_fts` (confirmed KEEP above). Not independently classified per the seed's rule (the correct removal trigger is dropping the parent FTS object, not a standalone column PURGE on shadow tables — see the `threads_fts` precedent in f2bad6e62). | n/a |
| `index.blocks_command_trigram_config.k` | KEEP | SQLite FTS5 virtual-table implementation storage; inherits KEEP from parent `blocks_command_trigram` (confirmed KEEP above). Not independently classified per the seed's rule (the correct removal trigger is dropping the parent FTS object, not a standalone column PURGE on shadow tables — see the `threads_fts` precedent in f2bad6e62). | n/a |
| `index.blocks_command_trigram_config.v` | KEEP | SQLite FTS5 virtual-table implementation storage; inherits KEEP from parent `blocks_command_trigram` (confirmed KEEP above). Not independently classified per the seed's rule (the correct removal trigger is dropping the parent FTS object, not a standalone column PURGE on shadow tables — see the `threads_fts` precedent in f2bad6e62). | n/a |
| `index.blocks_command_trigram_data.id` | KEEP | SQLite FTS5 virtual-table implementation storage; inherits KEEP from parent `blocks_command_trigram` (confirmed KEEP above). Not independently classified per the seed's rule (the correct removal trigger is dropping the parent FTS object, not a standalone column PURGE on shadow tables — see the `threads_fts` precedent in f2bad6e62). | n/a |
| `index.blocks_command_trigram_data.block` | KEEP | SQLite FTS5 virtual-table implementation storage; inherits KEEP from parent `blocks_command_trigram` (confirmed KEEP above). Not independently classified per the seed's rule (the correct removal trigger is dropping the parent FTS object, not a standalone column PURGE on shadow tables — see the `threads_fts` precedent in f2bad6e62). | n/a |
| `index.blocks_command_trigram_docsize.id` | KEEP | SQLite FTS5 virtual-table implementation storage; inherits KEEP from parent `blocks_command_trigram` (confirmed KEEP above). Not independently classified per the seed's rule (the correct removal trigger is dropping the parent FTS object, not a standalone column PURGE on shadow tables — see the `threads_fts` precedent in f2bad6e62). | n/a |
| `index.blocks_command_trigram_docsize.sz` | KEEP | SQLite FTS5 virtual-table implementation storage; inherits KEEP from parent `blocks_command_trigram` (confirmed KEEP above). Not independently classified per the seed's rule (the correct removal trigger is dropping the parent FTS object, not a standalone column PURGE on shadow tables — see the `threads_fts` precedent in f2bad6e62). | n/a |
| `index.blocks_command_trigram_idx.segid` | KEEP | SQLite FTS5 virtual-table implementation storage; inherits KEEP from parent `blocks_command_trigram` (confirmed KEEP above). Not independently classified per the seed's rule (the correct removal trigger is dropping the parent FTS object, not a standalone column PURGE on shadow tables — see the `threads_fts` precedent in f2bad6e62). | n/a |
| `index.blocks_command_trigram_idx.term` | KEEP | SQLite FTS5 virtual-table implementation storage; inherits KEEP from parent `blocks_command_trigram` (confirmed KEEP above). Not independently classified per the seed's rule (the correct removal trigger is dropping the parent FTS object, not a standalone column PURGE on shadow tables — see the `threads_fts` precedent in f2bad6e62). | n/a |
| `index.blocks_command_trigram_idx.pgno` | KEEP | SQLite FTS5 virtual-table implementation storage; inherits KEEP from parent `blocks_command_trigram` (confirmed KEEP above). Not independently classified per the seed's rule (the correct removal trigger is dropping the parent FTS object, not a standalone column PURGE on shadow tables — see the `threads_fts` precedent in f2bad6e62). | n/a |
| `index.session_work_events_fts_config.k` | KEEP | SQLite FTS5 virtual-table implementation storage; inherits KEEP from parent `session_work_events_fts` (confirmed KEEP above). Not independently classified per the seed's rule (the correct removal trigger is dropping the parent FTS object, not a standalone column PURGE on shadow tables — see the `threads_fts` precedent in f2bad6e62). | n/a |
| `index.session_work_events_fts_config.v` | KEEP | SQLite FTS5 virtual-table implementation storage; inherits KEEP from parent `session_work_events_fts` (confirmed KEEP above). Not independently classified per the seed's rule (the correct removal trigger is dropping the parent FTS object, not a standalone column PURGE on shadow tables — see the `threads_fts` precedent in f2bad6e62). | n/a |
| `index.session_work_events_fts_content.id` | KEEP | SQLite FTS5 virtual-table implementation storage; inherits KEEP from parent `session_work_events_fts` (confirmed KEEP above). Not independently classified per the seed's rule (the correct removal trigger is dropping the parent FTS object, not a standalone column PURGE on shadow tables — see the `threads_fts` precedent in f2bad6e62). | n/a |
| `index.session_work_events_fts_content.c0` | KEEP | SQLite FTS5 virtual-table implementation storage; inherits KEEP from parent `session_work_events_fts` (confirmed KEEP above). Not independently classified per the seed's rule (the correct removal trigger is dropping the parent FTS object, not a standalone column PURGE on shadow tables — see the `threads_fts` precedent in f2bad6e62). | n/a |
| `index.session_work_events_fts_content.c1` | KEEP | SQLite FTS5 virtual-table implementation storage; inherits KEEP from parent `session_work_events_fts` (confirmed KEEP above). Not independently classified per the seed's rule (the correct removal trigger is dropping the parent FTS object, not a standalone column PURGE on shadow tables — see the `threads_fts` precedent in f2bad6e62). | n/a |
| `index.session_work_events_fts_content.c2` | KEEP | SQLite FTS5 virtual-table implementation storage; inherits KEEP from parent `session_work_events_fts` (confirmed KEEP above). Not independently classified per the seed's rule (the correct removal trigger is dropping the parent FTS object, not a standalone column PURGE on shadow tables — see the `threads_fts` precedent in f2bad6e62). | n/a |
| `index.session_work_events_fts_content.c3` | KEEP | SQLite FTS5 virtual-table implementation storage; inherits KEEP from parent `session_work_events_fts` (confirmed KEEP above). Not independently classified per the seed's rule (the correct removal trigger is dropping the parent FTS object, not a standalone column PURGE on shadow tables — see the `threads_fts` precedent in f2bad6e62). | n/a |
| `index.session_work_events_fts_data.id` | KEEP | SQLite FTS5 virtual-table implementation storage; inherits KEEP from parent `session_work_events_fts` (confirmed KEEP above). Not independently classified per the seed's rule (the correct removal trigger is dropping the parent FTS object, not a standalone column PURGE on shadow tables — see the `threads_fts` precedent in f2bad6e62). | n/a |
| `index.session_work_events_fts_data.block` | KEEP | SQLite FTS5 virtual-table implementation storage; inherits KEEP from parent `session_work_events_fts` (confirmed KEEP above). Not independently classified per the seed's rule (the correct removal trigger is dropping the parent FTS object, not a standalone column PURGE on shadow tables — see the `threads_fts` precedent in f2bad6e62). | n/a |
| `index.session_work_events_fts_docsize.id` | KEEP | SQLite FTS5 virtual-table implementation storage; inherits KEEP from parent `session_work_events_fts` (confirmed KEEP above). Not independently classified per the seed's rule (the correct removal trigger is dropping the parent FTS object, not a standalone column PURGE on shadow tables — see the `threads_fts` precedent in f2bad6e62). | n/a |
| `index.session_work_events_fts_docsize.sz` | KEEP | SQLite FTS5 virtual-table implementation storage; inherits KEEP from parent `session_work_events_fts` (confirmed KEEP above). Not independently classified per the seed's rule (the correct removal trigger is dropping the parent FTS object, not a standalone column PURGE on shadow tables — see the `threads_fts` precedent in f2bad6e62). | n/a |
| `index.session_work_events_fts_idx.segid` | KEEP | SQLite FTS5 virtual-table implementation storage; inherits KEEP from parent `session_work_events_fts` (confirmed KEEP above). Not independently classified per the seed's rule (the correct removal trigger is dropping the parent FTS object, not a standalone column PURGE on shadow tables — see the `threads_fts` precedent in f2bad6e62). | n/a |
| `index.session_work_events_fts_idx.term` | KEEP | SQLite FTS5 virtual-table implementation storage; inherits KEEP from parent `session_work_events_fts` (confirmed KEEP above). Not independently classified per the seed's rule (the correct removal trigger is dropping the parent FTS object, not a standalone column PURGE on shadow tables — see the `threads_fts` precedent in f2bad6e62). | n/a |
| `index.session_work_events_fts_idx.pgno` | KEEP | SQLite FTS5 virtual-table implementation storage; inherits KEEP from parent `session_work_events_fts` (confirmed KEEP above). Not independently classified per the seed's rule (the correct removal trigger is dropping the parent FTS object, not a standalone column PURGE on shadow tables — see the `threads_fts` precedent in f2bad6e62). | n/a |

### embeddings.db

| tier.table.column | verdict | evidence | unlocks / owning reference |
|---|---|---|---|
| `embeddings.message_embeddings.embedding_input_hash (vec0 PK)` | KEEP | `polylogue/storage/search_providers/sqlite_vec_queries.py` joins on it for vector search | n/a |
| `embeddings.message_embeddings.embedding (vec0 float[1024])` | KEEP | `sqlite_vec_queries.py` — the vector itself, semantic search core | n/a |
| `embeddings.message_embeddings.model (vec0 aux)` | KEEP | `sqlite_vec_queries.py` filters/reports by model | n/a |
| `embeddings.message_embeddings_meta.embedding_input_hash` | KEEP | `embedding_write.py:read_embedding_meta` JOIN key; `sql.py:MISSING_META_MESSAGES_SQL` | n/a |
| `embeddings.message_embeddings_meta.model` | KEEP | `status_payload.py` MODEL_COUNTS query (`storage/embeddings/sql.py:65-70`), `embedding_write.py:read_embedding_meta` row["model"] | n/a |
| `embeddings.message_embeddings_meta.dimension` | KEEP | `sql.py:DIMENSION_COUNTS_SQL`, `embedding_write.py:read_embedding_meta` row["dimension"] | n/a |
| `embeddings.message_embeddings_meta.embedded_at_ms` | KEEP | `sql.py:EMBEDDED_AT_BOUNDS_SQL` (MIN/MAX), `status_payload.py:916` bounds query, `embedding_write.py:read_embedding_meta` | n/a |
| `embeddings.message_embeddings_meta.recipe_hash` | KEEP | `daemon/convergence_stages.py:1238-1247` WHERE recipe_hash != ? drives reindex trigger (`meta_recipe_changed`) | n/a |
| `embeddings.message_embeddings_meta.output_contract_hash` | KEEP | write path only writes it; **no direct SELECT of this column from `message_embeddings_meta` found** — `convergence_stages.py`'s output_contract_hash comparisons all read from `embedding_derivation_state`, not this table. Distinct from the KEEP derivation_state column of the same name. | UNCLEAR-adjacent; flagged, not flipped (low confidence single miss, keep conservative) |
| `embeddings.message_embedding_refs.message_id` | KEEP | PK; `embedding_write.py:read_embedding_meta` JOIN, `sqlite_vec_queries.py` JOINs, `session_replacement.py` DELETE...WHERE session_id (session-scoped, message_id is join target) | n/a |
| `embeddings.message_embedding_refs.session_id` | KEEP | `reconcile.py:320` `SELECT COUNT(*) ... WHERE session_id = ?`, `session_replacement.py` DELETE WHERE session_id, idx_message_embedding_refs_session | n/a |
| `embeddings.message_embedding_refs.origin` | KEEP | `sqlite_vec_queries.py:258` "applied post-join against `message_embedding_refs.origin`" — origin filter in semantic search | n/a |
| `embeddings.message_embedding_refs.embedding_input_hash` | KEEP | `embedding_write.py:read_embedding_meta` JOIN key to meta table; core content-address | n/a |
| `embeddings.message_embedding_refs.embedded_at_ms` | KEEP | `reconcile.py:431-513` `SELECT r.embedded_at_ms ...` staleness/reconciliation window checks | n/a |
| `embeddings.embedding_status.session_id` | KEEP | PK; joined everywhere (`status_payload.py`, `reconcile.py`, `metrics.py`) | n/a |
| `embeddings.embedding_status.origin` | KEEP | `embedding_write.py:904-923 read_embedding_status` SELECTs and returns `origin` on `ArchiveEmbeddingStatus`, consumed by `mark_session_embedding_error` callers | n/a |
| `embeddings.embedding_status.message_count_embedded` | KEEP | `status_payload.py:792-799` `SELECT COALESCE(SUM(e.message_count_embedded), 0)` | n/a |
| `embeddings.embedding_status.last_embedded_at_ms` | KEEP | `reconcile.py:464-542` `SELECT session_id, last_embedded_at_ms` staleness window; `embedding_write.py:read_embedding_status` | n/a |
| `embeddings.embedding_status.needs_reindex` | KEEP | `sql.py:EMBEDDED_SESSIONS_SQL/PENDING_SESSIONS_SQL`, `status_payload.py` (`COALESCE(e.needs_reindex,0)=0`), `daemon/metrics.py:722-723` | n/a |
| `embeddings.embedding_status.error_message` | KEEP | `sql.py:EMBEDDING_FAILURE_COUNT_SQL`, `status_payload.py` blocked/failure detection, `daemon/metrics.py:727` | n/a |
| `embeddings.embedding_derivation_state.session_id` | KEEP | PK; WHERE clauses throughout `embedding_write.py` (lines 434-441, 568-577), `convergence_stages.py` RETURNING session_id | n/a |
| `embeddings.embedding_derivation_state.origin` | KEEP | `embedding_write.py:493` attempt tuple includes origin read from the attempt object built off this row (via `ArchiveEmbeddingAttempt.origin=attempt.origin` chain) — same attempt-projection pattern as embedding_status.origin | n/a |
| `embeddings.embedding_derivation_state.generation` | KEEP | WHERE predicate gating idempotent apply (`embedding_write.py:434-441, 568-577`), `convergence_stages.py:1271` `SET generation = generation + 1` reindex bump | n/a |
| `embeddings.embedding_derivation_state.derivation_key` | KEEP | WHERE predicate (`embedding_write.py:434-441, 568-577`), computed/compared via `polylogue_embedding_derivation_key(...)` in `convergence_stages.py:1269-1271` | n/a |
| `embeddings.embedding_derivation_state.source_hash` | KEEP | Compared against freshly computed digest (`embedding_write.py:426-428`), used in derivation_key recompute (`convergence_stages.py`) | n/a |
| `embeddings.embedding_derivation_state.recipe_hash` | KEEP | `convergence_stages.py:1282` `WHERE recipe_hash != ? OR output_contract_hash != ?` drives the reindex-on-recipe-change trigger | n/a |
| `embeddings.embedding_derivation_state.output_contract_hash` | KEEP | Same WHERE clause as recipe_hash above (`convergence_stages.py:1282`) | n/a |
| `embeddings.embedding_derivation_state.attempt_state` | KEEP | WHERE predicate (`attempt_state = 'pending'`) gating every apply path (`embedding_write.py:434-441, 568-577`) | n/a |
| `embeddings.embedding_derivation_state.message_count` | PURGE | **Write-only.** Set at INSERT (`embedding_write.py:126`), reset to 0 on reindex (`embedding_write.py:136,211`, `convergence_stages.py:1280`), updated on success (`embedding_write.py:471,477`) — no SELECT/WHERE/ORDER BY of this column anywhere in `polylogue/` outside those writes. `embedding_status.message_count_embedded` is the actually-read analog. | Flip from seed's blanket KEEP; small, low-risk column drop riding the 818fy derived-tier rebuild window |
| `embeddings.embedding_derivation_state.updated_at_ms` | PURGE | **Write-only.** Set on every INSERT/UPDATE (`embedding_write.py:126,137,471,477`; `convergence_stages.py:1280`) — no SELECT of it from `embedding_derivation_state` anywhere (the `ORDER BY updated_at_ms DESC` at `embedding_write.py:794` belongs to the unrelated `embedding_failures` table). | Flip from seed's blanket KEEP; same rebuild window as message_count |
| `embeddings.embedding_failures.failure_id` | KEEP | PK; `embed.py` CLI resolve-failure command, `list_active_embedding_failures`/`read_embedding_failure` SELECT+return it | n/a |
| `embeddings.embedding_failures.session_id` | KEEP | SELECTed and returned throughout (`embedding_write.py:783-820`), used in resolve/supersede WHERE clauses | n/a |
| `embeddings.embedding_failures.origin` | KEEP | SELECTed in `list_active_embedding_failures`/`read_embedding_failure`, surfaced in `status_payload.py:_active_failure_details` | n/a |
| `embeddings.embedding_failures.message_refs_json` | KEEP | SELECTed and JSON-decoded in `status_payload.py:290` and `embedding_write.py:_failure_from_row` | n/a |
| `embeddings.embedding_failures.provider` | KEEP | SELECTed/returned (`status_payload.py:299`, `embedding_write.py` failure rows) | n/a |
| `embeddings.embedding_failures.model` | KEEP | SELECTed/returned (`status_payload.py:300`) | n/a |
| `embeddings.embedding_failures.error_class` | KEEP | SELECTed/returned (`status_payload.py:301`) | n/a |
| `embeddings.embedding_failures.error_message` | KEEP | SELECTed/returned (`status_payload.py:302`) | n/a |
| `embeddings.embedding_failures.retryable` | KEEP | SELECTed/returned (`status_payload.py:303`), gates CLI resolution options | n/a |
| `embeddings.embedding_failures.lifecycle_state` | KEEP | WHERE filter everywhere (`lifecycle_state IN ('retryable','terminal')`), SELECTed/returned | n/a |
| `embeddings.embedding_failures.created_at_ms` | KEEP | SELECTed/returned (`status_payload.py:305`) | n/a |
| `embeddings.embedding_failures.updated_at_ms` | KEEP | ORDER BY + SELECTed/returned (`embedding_write.py:794`, `status_payload.py:306`) | n/a |
| `embeddings.embedding_failures.resolved_at_ms` | KEEP | SELECTed/returned (`embedding_write.py:869`) | n/a |
| `embeddings.embedding_failures.resolution_action` | KEEP | SELECTed/returned (`status_payload.py:307`, `cli/commands/embed.py:348`) | n/a |
| `embeddings.embedding_failures.resolution_note` | KEEP | SELECTed/returned, surfaced via CLI (`embedding_write.py:791,808`, `cli/commands/embed.py:348`) | n/a |
| `embeddings.embedding_failures.superseded_by` | KEEP | SELECTed/returned, surfaced via CLI (`embedding_write.py:791,808`, `cli/commands/embed.py:349`) | n/a |
| `embeddings.embedding_failures.generation` | KEEP | SELECTed via `_embedding_failure_identity_select` (`embedding_write.py:832-843`), returned on `ArchiveEmbeddingFailure` | n/a |
| `embeddings.embedding_failures.derivation_key` | KEEP | Same identity-select path (`embedding_write.py:844`) | n/a |
| `embeddings.embedding_failures.source_hash` | KEEP | Same identity-select path (`embedding_write.py:844`) | n/a |
| `embeddings.embedding_failures.recipe_hash` | KEEP | Same identity-select path (`embedding_write.py:844`) | n/a |

### ops.db

| tier.table.column | verdict | evidence | unlocks / owning reference |
|---|---|---|---|
| `ops.ingest_cursor.source_path` | KEEP | PK; selected everywhere, e.g. cursor.py:1075 | n/a |
| `ops.ingest_cursor.origin` | KEEP | cursor.py:1075 SELECT list | n/a |
| `ops.ingest_cursor.stat_size` | KEEP | cursor.py:1075; archive_verification.py:2309 | n/a |
| `ops.ingest_cursor.byte_offset` | KEEP | cursor.py:1075; raw_retention.py:1850 (frontier authority) | n/a |
| `ops.ingest_cursor.last_complete_newline` | KEEP | cursor.py:1075 | n/a |
| `ops.ingest_cursor.record_count` | KEEP | cursor.py:1075 | n/a |
| `ops.ingest_cursor.last_record_ts_ms` | KEEP | cursor.py:1075 | n/a |
| `ops.ingest_cursor.parser_fingerprint` | KEEP | cursor.py:1075 | n/a |
| `ops.ingest_cursor.content_fingerprint` | KEEP | cursor.py:1075; used in list_retry_records WHERE | n/a |
| `ops.ingest_cursor.tail_hash` | KEEP | cursor.py:1075 | n/a |
| `ops.ingest_cursor.st_dev` | KEEP | cursor.py:1075 | n/a |
| `ops.ingest_cursor.st_ino` | KEEP | cursor.py:1075 | n/a |
| `ops.ingest_cursor.mtime_ns` | KEEP | cursor.py:1075 | n/a |
| `ops.ingest_cursor.failure_count` | KEEP | cursor.py:1075; status.py:1273; cli/commands/status.py:653 | n/a |
| `ops.ingest_cursor.next_retry_at` | KEEP | cursor.py:1075; archive_verification.py:2013 | n/a |
| `ops.ingest_cursor.excluded` | KEEP | cursor.py:1075,1389; status.py:1279; archive_verification.py:2000 | n/a |
| `ops.ingest_cursor.deferred_end_offset` | KEEP | cursor.py:1075; raw_retention.py:1850 | n/a |
| `ops.ingest_cursor.updated_at_ms` | KEEP | cursor.py:1075; archive_verification.py:2001,2309 | n/a |
| `ops.ingest_attempts.attempt_id` | KEEP | PK; WHERE key in cursor_authority_reconcile.py:714, metrics.py:666 | n/a |
| `ops.ingest_attempts.source_path` | KEEP | cursor.py:352; source_freshness.py:1030 | n/a |
| `ops.ingest_attempts.origin` | KEEP | daemon status/metrics payload builders (broad) | n/a |
| `ops.ingest_attempts.status` | KEEP | owned by polylogue-oj4oo (status vocabulary unification), cite not re-derive; real reader e.g. daemon/metrics.py:299,1663; status.py | n/a |
| `ops.ingest_attempts.phase` | KEEP | daemon/health.py:1122 `SELECT phase, error_message`; source_freshness.py:1030 | n/a |
| `ops.ingest_attempts.storage_route` | KEEP | daemon/metrics.py:639 `GROUP BY storage_route`; cli/commands/status.py | n/a |
| `ops.ingest_attempts.started_at_ms` | KEEP | status.py:1515; metrics.py:358 | n/a |
| `ops.ingest_attempts.heartbeat_at_ms` | KEEP | status.py:1516 | n/a |
| `ops.ingest_attempts.finished_at_ms` | KEEP | status.py:1517; metrics.py:358,662 | n/a |
| `ops.ingest_attempts.parsed_raw_count` | KEEP | status.py:1518; metrics.py:662 | n/a |
| `ops.ingest_attempts.materialized_count` | KEEP | status.py:1519; metrics.py:662 | n/a |
| `ops.ingest_attempts.error_message` | KEEP | status.py:1520; health.py:1122 | n/a |
| `ops.ingest_attempts.source_paths_json` | KEEP | cursor.py:352 `SELECT source_path, source_paths_json WHERE status='running'` | n/a |
| `ops.ingest_attempts.outcome_code` | KEEP | cursor_authority_reconcile.py:710 `outcome_code, retryable, diagnostic, remediation` (polylogue-cnu3 typed disposition) | n/a |
| `ops.ingest_attempts.retryable` | KEEP | cursor_authority_reconcile.py:710 | n/a |
| `ops.ingest_attempts.evidence_ref` | PURGE | write-only: only appears in ops_write.py's INSERT/upsert (`evidence_ref = disposition.evidence_ref`); zero SELECT of this column anywhere in polylogue/devtools/tests (checked cursor_authority_reconcile.py, status.py, health.py, metrics.py, source_freshness.py, cli/commands/status.py — none select it) | new gvzkr finding; candidate for 6kur repair-surface cull once ingest_attempts is touched |
| `ops.ingest_attempts.diagnostic` | KEEP | cursor_authority_reconcile.py:710 | n/a |
| `ops.ingest_attempts.remediation` | KEEP | cursor_authority_reconcile.py:710 | n/a |
| `ops.embedding_catchup_runs.run_id` | KEEP | `list_embedding_catchup_runs`/`read_embedding_catchup_run` (ops_write.py:1230-1264), called by daemon/metrics.py:868 and storage/embeddings/status_payload.py:1104 | n/a |
| `ops.embedding_catchup_runs.started_at_ms` | KEEP | same | n/a |
| `ops.embedding_catchup_runs.finished_at_ms` | KEEP | same | n/a |
| `ops.embedding_catchup_runs.status` | KEEP | same; owned by oj4oo vocabulary, cite not re-derive | n/a |
| `ops.embedding_catchup_runs.origin` | KEEP | same | n/a |
| `ops.embedding_catchup_runs.scanned_sessions` | KEEP | same | n/a |
| `ops.embedding_catchup_runs.embedded_sessions` | KEEP | same | n/a |
| `ops.embedding_catchup_runs.skipped_sessions` | KEEP | same | n/a |
| `ops.embedding_catchup_runs.error_count` | KEEP | same | n/a |
| `ops.embedding_catchup_runs.embedded_messages` | KEEP | same | n/a |
| `ops.embedding_catchup_runs.estimated_cost_usd` | KEEP | same | n/a |
| `ops.embedding_catchup_runs.error_message` | KEEP | same | n/a |
| `ops.convergence_debt.debt_id` | KEEP | PK; ORDER BY debt_id DESC (convergence_debt_status.py:106) | n/a |
| `ops.convergence_debt.stage` | KEEP | `_REQUIRED_CONVERGENCE_DEBT_COLUMNS` (convergence_debt_status.py:18-30) full projection; metrics.py:405 GROUP BY | n/a |
| `ops.convergence_debt.target_type` | KEEP | same; source_freshness.py:1685 WHERE | n/a |
| `ops.convergence_debt.target_id` | KEEP | same | n/a |
| `ops.convergence_debt.status` | KEEP | same; metrics.py:405 | n/a |
| `ops.convergence_debt.priority` | KEEP | ORDER BY priority DESC (convergence_debt_status.py:106); ops_write.py:1688 | n/a |
| `ops.convergence_debt.attempts` | KEEP | required-columns set; cursor.py read-modify-write | n/a |
| `ops.convergence_debt.last_error` | KEEP | required-columns set; cursor.py:476 read | n/a |
| `ops.convergence_debt.next_retry_at` | KEEP | required-columns set; cursor.py:476 | n/a |
| `ops.convergence_debt.materializer_version` | KEEP | required-columns set (convergence_debt_status.py:29); daemon/cli.py:1818 etc. surfaces `debt.materializer_version` | n/a |
| `ops.convergence_debt.created_at_ms` | KEEP | required-columns set (convergence_debt_status.py:30) | n/a |
| `ops.convergence_debt.updated_at_ms` | KEEP | required-columns set; ORDER BY updated_at_ms DESC | n/a |
| `ops.cursor_lag_samples.sample_id` | PURGE | only in test-only `read_cursor_lag_sample`/`list_cursor_lag_samples`; no production SELECT projects it | narrows the accessor functions' claim to a real reader before any 6kur cull touches them |
| `ops.cursor_lag_samples.family` | KEEP | `cursor_lag_baseline.py:298` WHERE family = ? | n/a |
| `ops.cursor_lag_samples.source_path` | PURGE | test-only reader only (same as sample_id) | same |
| `ops.cursor_lag_samples.lag_ms` | KEEP | `cursor_lag_baseline.py:296` SELECT lag_ms | n/a |
| `ops.cursor_lag_samples.stuck_file_count` | PURGE | test-only reader only | same |
| `ops.cursor_lag_samples.p50_lag_ms` | PURGE | test-only reader only (note: this is a *stored, precomputed* percentile, distinct from `daemon/cursor_lag_baseline.py`'s own `rolling_median_lag_s` which is computed fresh from `lag_ms` samples, not read from this column) | same |
| `ops.cursor_lag_samples.p95_lag_ms` | PURGE | test-only reader only, same caveat as p50_lag_ms | same |
| `ops.cursor_lag_samples.severity` | PURGE | test-only reader only | same |
| `ops.cursor_lag_samples.sampled_at_ms` | KEEP | `cursor_lag_baseline.py:298` WHERE sampled_at_ms >= ?; also DELETE retention predicate | n/a |
| `ops.slo_samples.sample_id` | KEEP | `list_slo_samples` (operations/slo.py:120-124) full projection, called by `daemon/slo.py:425` | n/a |
| `ops.slo_samples.label` | KEEP | same; also WHERE label = ? | n/a |
| `ops.slo_samples.scope` | KEEP | same; also WHERE scope = ? | n/a |
| `ops.slo_samples.value` | KEEP | same | n/a |
| `ops.slo_samples.observed_at_ms` | KEEP | same; WHERE observed_at_ms >= ? | n/a |
| `ops.slo_samples.window_start_ms` | KEEP | same | n/a |
| `ops.slo_samples.window_end_ms` | KEEP | same | n/a |
| `ops.slo_samples.metadata_json` | KEEP | same | n/a |
| `ops.daemon_stage_events.event_id` | KEEP | ORDER BY event_id DESC (metrics.py:545); PK lookups in ops_write.py:1127 | n/a |
| `ops.daemon_stage_events.attempt_id` | KEEP | catchup_status.py:206; status.py:1621 (payload lookup keyed by attempt_id) | n/a |
| `ops.daemon_stage_events.stage` | KEEP | catchup_status.py:206; archive_readiness.py:165 WHERE stage = ? | n/a |
| `ops.daemon_stage_events.status` | KEEP | catchup_status.py:206; archive_readiness.py:165 | n/a |
| `ops.daemon_stage_events.observed_at_ms` | KEEP | catchup_status.py:206; archive_readiness.py:165; metrics.py:544,659 ORDER BY | n/a |
| `ops.daemon_stage_events.payload_json` | KEEP | catchup_status.py:206; metrics.py:544,659 (storage_route sniffed from payload); status.py:1621 | n/a |
| `ops.daemon_events.id` | KEEP | `daemon/events.py:199,210,247,252,317,361,368,392` — every SELECT list | n/a |
| `ops.daemon_events.ts_ms` | KEEP | same | n/a |
| `ops.daemon_events.kind` | KEEP | same; also GROUP BY kind (events.py:588) | n/a |
| `ops.daemon_events.operation_id` | KEEP | same | n/a |
| `ops.daemon_events.payload_json` | KEEP | same | n/a |
| `ops.judgment_scheduler_receipts.operation_id` | KEEP | `_read_latest_judgment_scheduler_receipt` (ops_write.py:224-249) full projection, called via `operations/judgment_scheduler.py:read_latest_judgment_scheduler_receipt`, consumed by `api/archive.py:1743` and `daemon/judgment_automation.py:379`, surfaced through `cli/commands/judge.py:250-277` | n/a |
| `ops.judgment_scheduler_receipts.observed_at_ms` | KEEP | same; also ORDER BY | n/a |
| `ops.judgment_scheduler_receipts.status` | KEEP | same; cli/commands/judge.py:252 | n/a |
| `ops.judgment_scheduler_receipts.reason` | KEEP | same; judge.py:254 | n/a |
| `ops.judgment_scheduler_receipts.retryable` | KEEP | same; judge.py:253 | n/a |
| `ops.judgment_scheduler_receipts.retry_route` | KEEP | same; judge.py:254 | n/a |
| `ops.judgment_scheduler_receipts.batch_limit` | KEEP | same; judge.py:255 | n/a |
| `ops.judgment_scheduler_receipts.considered` | KEEP | same; judge.py:258 | n/a |
| `ops.judgment_scheduler_receipts.accepted` | KEEP | same; judge.py:259 | n/a |
| `ops.judgment_scheduler_receipts.rejected` | KEEP | same; judge.py:260 | n/a |
| `ops.judgment_scheduler_receipts.escalated` | KEEP | same; judge.py:261 | n/a |
| `ops.judgment_scheduler_receipts.idempotent` | KEEP | same; judge.py:262 | n/a |
| `ops.judgment_scheduler_receipts.failed` | KEEP | same; judge.py:263 | n/a |
| `ops.judgment_scheduler_receipts.receipt_persistence_degraded` | KEEP | same; judge.py:273-276 | n/a |
| `ops.judgment_scheduler_receipts.receipt_persistence_recovered` | KEEP | same; judge.py:277 | n/a |
| `ops.daemon_lifecycle.run_id` | KEEP | `latest_daemon_lifecycle` (ops_write.py:1098-1112) full projection, called by `daemon/lifecycle.py:206` | n/a |
| `ops.daemon_lifecycle.started_at_ms` | KEEP | same; also ORDER BY; api/archive.py:1728 separate query | n/a |
| `ops.daemon_lifecycle.stopped_at_ms` | KEEP | same; api/archive.py:1727 | n/a |
| `ops.daemon_lifecycle.last_heartbeat_at_ms` | KEEP | same; api/archive.py:1727 | n/a |
| `ops.daemon_lifecycle.signal` | KEEP | same | n/a |
| `ops.daemon_lifecycle.exit_kind` | KEEP | same | n/a |
| `ops.daemon_lifecycle.details_json` | KEEP | same | n/a |
| `ops.secret_scan_status.session_id` | KEEP | `secret_scan.py:444,498` LEFT JOIN key + `st.session_id IS NULL` check | n/a |
| `ops.secret_scan_status.scanner_version` | KEEP | `secret_scan.py:446,500` `st.scanner_version < ?` (drives the whole "pending" concept) | n/a |
| `ops.secret_scan_status.scanned_at_ms` | PURGE | write-only: only in `_record_secret_scan_status` INSERT/UPSERT (secret_scan.py:522-531); never selected | corrects the seed's table-level verdict; a source.db/index.db-independent, purely ops.db-scoped drop |
| `ops.secret_scan_status.blocks_scanned` | PURGE | write-only, same evidence | same |
| `ops.secret_scan_status.candidates_found` | PURGE | write-only, same evidence | same |
| `ops.mcp_call_log.call_id` | KEEP | ops_write.py:1482,1508; diagnostics.py:938 | n/a |
| `ops.mcp_call_log.tool_name` | KEEP | same; also WHERE tool_name = ? filter | n/a |
| `ops.mcp_call_log.session_id` | KEEP | same; also EXISTS-subquery join key against mcp_call_session_refs | n/a |
| `ops.mcp_call_log.started_at_ms` | KEEP | same; ORDER BY | n/a |
| `ops.mcp_call_log.finished_at_ms` | KEEP | same | n/a |
| `ops.mcp_call_log.duration_ms` | KEEP | same | n/a |
| `ops.mcp_call_log.success` | KEEP | same | n/a |
| `ops.mcp_call_log.error_detail` | KEEP | same | n/a |
| `ops.mcp_call_session_refs.call_id` | KEEP | `list_mcp_calls` EXISTS-subquery join key (ops_write.py:1489-1492), reached from diagnostics.py | n/a |
| `ops.mcp_call_session_refs.session_id` | KEEP | same subquery predicate | n/a |
| `ops.mcp_call_session_refs.relation` | KEEP | only read inside `record_mcp_call`'s own idempotency/conflict check (ops_write.py:1439, `existing_refs != desired_refs` guard) — enforces a real write invariant (rejects a call_id replayed with different session-ref roles) but is never externally surfaced: `list_mcp_calls`/`ArchiveMcpCallLogEntry` do not project `relation` at all (weak) | flag for operator: candidate to actually surface 'primary' vs 'member' in the diagnostics payload, or accept it as write-path-only and downgrade to PURGE later — recommend UNCLEAR-leaning-KEEP, not a guess either way |
| `ops.route_observations.observation_id` | KEEP | ops_write.py:1620 SELECT list; PK dedup in prune | n/a |
| `ops.route_observations.trace_id` | KEEP | same | n/a |
| `ops.route_observations.surface` | KEEP | same; WHERE surface = ? filter | n/a |
| `ops.route_observations.route` | KEEP | same; WHERE route = ? filter | n/a |
| `ops.route_observations.verb` | KEEP | same | n/a |
| `ops.route_observations.daemon_path` | KEEP | same | n/a |
| `ops.route_observations.phase` | KEEP | same | n/a |
| `ops.route_observations.started_at_ms` | KEEP | same; ORDER BY; WHERE since_ms filter | n/a |
| `ops.route_observations.duration_ms` | KEEP | same | n/a |
| `ops.route_observations.status` | KEEP | same | n/a |
| `ops.route_observations.git_head` | KEEP | same | n/a |
| `ops.route_observations.archive_epoch` | KEEP | same | n/a |
| `ops.route_observations.attributes_json` | KEEP | same | n/a |
| `ops.route_observations.sampled` | KEEP | same | n/a |
| `ops.fts_drift_samples.sample_id` | PURGE | test-only reader (`list_fts_drift_samples`); zero production SELECT | new gvzkr finding — candidate to drop the table + its one writer (`drift_sampling.py:125`) together, or wire a diagnostics-surface reader if the telemetry is wanted |
| `ops.fts_drift_samples.surface` | PURGE | same | same |
| `ops.fts_drift_samples.state` | PURGE | same | same |
| `ops.fts_drift_samples.source_rows` | PURGE | same | same |
| `ops.fts_drift_samples.indexed_rows` | PURGE | same | same |
| `ops.fts_drift_samples.missing_rows` | PURGE | same | same |
| `ops.fts_drift_samples.excess_rows` | PURGE | same | same |
| `ops.fts_drift_samples.duplicate_rows` | PURGE | same | same |
| `ops.fts_drift_samples.identity_mismatch_rows` | PURGE | same | same |
| `ops.fts_drift_samples.sampled_at_ms` | PURGE | same | same |
| `ops.schema_drift_samples.sample_id` | KEEP | `summarize_schema_drift_since` (ops_write.py:557-600) DISTINCT/classification query, called from `polylogue/insights/schema_drift.py:54` (production insight, not test-only) | n/a |
| `ops.schema_drift_samples.origin` | KEEP | same; WHERE origin = ? | n/a |
| `ops.schema_drift_samples.element_kind` | KEEP | `_schema_drift_sample_from_row`/`list_schema_drift_samples` full projection — TEST-ONLY caller, but `element_kind` is not read by `summarize_schema_drift_since` itself; keeping KEEP because the DDL comments (polylogue-da1/polylogue-u6tl) document this as a load-bearing drift-sentinel dimension and `list_schema_drift_samples` is one query away from production wiring, not dead code — flagging as weak-KEEP rather than guessing PURGE | operator call: wire `element_kind` into the schema_drift insight surface, or accept UNCLEAR |
| `ops.schema_drift_samples.classification` | KEEP | `summarize_schema_drift_since` line 590 `SELECT classification, native_id_example`; drives risky/benign split | n/a |
| `ops.schema_drift_samples.unseen_key_signature` | KEEP | same caveat as element_kind — part of `list_schema_drift_samples`' full projection (test-only caller) but not touched by `summarize_schema_drift_since`; DDL comment names it as part of the dedup key `(origin, element_kind, unseen_key_signature)` used at write time, not confirmed read in production (weak) | operator call, same as element_kind |
| `ops.schema_drift_samples.native_id_example` | KEEP | `summarize_schema_drift_since` line 590; feeds `risky_examples`/`example_native_ids` | n/a |
| `ops.schema_drift_samples.raw_id` | KEEP | same caveat as element_kind/unseen_key_signature (weak) | operator call, same as element_kind |
| `ops.schema_drift_samples.observed_at_ms` | KEEP | `summarize_schema_drift_since` WHERE observed_at_ms >= ?; ops_write.py:582 DISTINCT origin query | n/a |

### user.db

| tier.table.column | verdict | evidence | unlocks / owning reference |
|---|---|---|---|
| `user.query_unit_frame_state.singleton` | KEEP | `archive/query/transaction.py:100,103` `SELECT epoch FROM query_unit_frame_state WHERE singleton = 1` | n/a |
| `user.query_unit_frame_state.epoch` | KEEP | same citation — the query-cache-invalidation generation counter | n/a |
| `user.assertions.assertion_id` | KEEP | PK; `_ASSERTION_COLUMNS` (user_write.py:2360-2363), used in every assertion reader | n/a |
| `user.assertions.scope_ref` | KEEP | `_ASSERTION_COLUMNS`, `idx_assertions_scope_kind_status` | n/a |
| `user.assertions.target_ref` | KEEP | `_ASSERTION_COLUMNS`, `idx_assertions_target_kind*` (2 indexes) | n/a |
| `user.assertions.key` | KEEP | `_ASSERTION_COLUMNS` | n/a |
| `user.assertions.kind` | KEEP | `_ASSERTION_COLUMNS`, closed-vocabulary AssertionKind filter throughout | n/a |
| `user.assertions.value_json` | KEEP | `_ASSERTION_COLUMNS` | n/a |
| `user.assertions.body_text` | KEEP | `_ASSERTION_COLUMNS` | n/a |
| `user.assertions.author_ref` | KEEP | `_ASSERTION_COLUMNS` | n/a |
| `user.assertions.author_kind` | KEEP | `_ASSERTION_COLUMNS` | n/a |
| `user.assertions.evidence_refs_json` | KEEP | `_ASSERTION_COLUMNS` | n/a |
| `user.assertions.status` | KEEP | `_ASSERTION_COLUMNS`, `idx_assertions_kind_status_updated` | n/a |
| `user.assertions.visibility` | KEEP | `_ASSERTION_COLUMNS`, `idx_assertions_target_kind_status_visibility` | n/a |
| `user.assertions.confidence` | KEEP | `_ASSERTION_COLUMNS` | n/a |
| `user.assertions.staleness_json` | KEEP | `_ASSERTION_COLUMNS` | n/a |
| `user.assertions.context_policy_json` | KEEP | `_ASSERTION_COLUMNS` — gates MCP/API context injection | n/a |
| `user.assertions.supersedes_json` | KEEP | `_ASSERTION_COLUMNS` | n/a |
| `user.assertions.created_at_ms` | KEEP | `_ASSERTION_COLUMNS`, ordering everywhere | n/a |
| `user.assertions.updated_at_ms` | KEEP | `_ASSERTION_COLUMNS`, `idx_assertions_kind_status_updated`, epoch-bump triggers | n/a |
| `user.queries.query_hash` | KEEP | PK; `query_objects.py:198-236` `get_query`/`list_watched_queries` | n/a |
| `user.queries.canonical_plan_json` | KEEP | `query_objects.py:201,222` SELECT list, decoded as executable plan | n/a |
| `user.queries.grain` | KEEP | `query_objects.py:201,222` SELECT list | n/a |
| `user.queries.lane` | KEEP | `query_objects.py:201,222` SELECT list | n/a |
| `user.queries.rank_policy` | KEEP | `query_objects.py:201,222` SELECT list | n/a |
| `user.queries.created_at_ms` | KEEP | ordering context in query_objects.py evaluation-receipt/result-set flows | n/a |
| `user.queries.definition_protocol_version` | KEEP | `query_objects.py:201,222` SELECT list | n/a |
| `user.query_names.name` | KEEP | PK; `query_objects.py:226-236` `list_watched_queries` JOIN | n/a |
| `user.query_names.query_hash` | KEEP | `query_objects.py:128,226-236` FK join target | n/a |
| `user.query_names.supersedes_query_hash` | UNCLEAR | Write-only — see finding above. Not flagged PURGE per durable-tier conservatism. | n/a |
| `user.query_names.updated_at_ms` | KEEP | `idx_query_names_query_hash`/`idx_query_names_watch` ordering, `query_objects.py:128` upsert-then-read cycle | n/a |
| `user.query_names.watch` | KEEP | `query_objects.py:226-236` `WHERE n.watch = 1`, `idx_query_names_watch` | n/a |
| `user.result_sets.result_set_id` | KEEP | PK; `query_objects.py:248-300` `get_result_set`/`get_latest_result_set`/`get_watched_query_baseline` | n/a |
| `user.result_sets.query_hash` | KEEP | `query_objects.py:248-300` SELECT list + FK filters | n/a |
| `user.result_sets.grain` | KEEP | `query_objects.py:248-300` SELECT list | n/a |
| `user.result_sets.corpus_epoch` | KEEP | `query_objects.py:248-300` SELECT list | n/a |
| `user.result_sets.member_count` | KEEP | `query_objects.py:248-300` SELECT list | n/a |
| `user.result_sets.membership_merkle_root` | KEEP | `query_objects.py:248-300` SELECT list — integrity proof | n/a |
| `user.result_sets.ordered_rank_hash` | KEEP | `query_objects.py:248-300` SELECT list — integrity proof | n/a |
| `user.result_sets.exactness` | KEEP | `query_objects.py:248-300` SELECT list | n/a |
| `user.result_sets.persistence_class` | KEEP | `query_objects.py:248-300` SELECT list, `get_latest_result_set` filter | n/a |
| `user.result_sets.created_at_ms` | KEEP | `idx_result_sets_query_epoch` ordering, `get_latest_result_set` ORDER BY | n/a |
| `user.result_set_members.result_set_id` | KEEP | `query_objects.py:259` `WHERE result_set_id = ?` | n/a |
| `user.result_set_members.rank` | KEEP | `query_objects.py:259` `ORDER BY rank` | n/a |
| `user.result_set_members.member_ref` | KEEP | `query_objects.py:259` `SELECT member_ref` | n/a |
| `user.query_edges.src_query_hash` | KEEP | `query_objects.py:429-432` recursive reachability CTE | n/a |
| `user.query_edges.dst_query_hash` | KEEP | `query_objects.py:429-432` recursive reachability CTE | n/a |
| `user.query_edges.edge_kind` | KEEP | `query_objects.py:429` `WHERE ... edge_kind = ?` | n/a |
| `user.query_edges.created_at_ms` | KEEP | `idx_query_edges_dst_kind` ordering | n/a |
| `user.retained_query_runs.run_id` | KEEP | PK; `query_objects.py:339,358` full-row read | n/a |
| `user.retained_query_runs.query_hash` | KEEP | `query_objects.py:339,358` SELECT list, trigger-enforced consistency | n/a |
| `user.retained_query_runs.result_set_id` | KEEP | `query_objects.py:339,358` SELECT list, trigger-enforced consistency | n/a |
| `user.retained_query_runs.retained_at_ms` | KEEP | `query_objects.py:339` SELECT list | n/a |
| `user.query_evaluation_receipts.receipt_id` | KEEP | PK; `query_objects.py:390-394` full-row read | n/a |
| `user.query_evaluation_receipts.query_hash` | KEEP | `query_objects.py:390-394` SELECT list, `idx_query_evaluation_receipts_query_time` | n/a |
| `user.query_evaluation_receipts.result_set_id` | KEEP | `query_objects.py:390-394` SELECT list, trigger-enforced consistency | n/a |
| `user.query_evaluation_receipts.source_generation` | KEEP | `query_objects.py:390-394` SELECT list | n/a |
| `user.query_evaluation_receipts.user_generation` | KEEP | `query_objects.py:390-394` SELECT list | n/a |
| `user.query_evaluation_receipts.index_generation` | KEEP | `query_objects.py:390-394` SELECT list | n/a |
| `user.query_evaluation_receipts.runtime_build_ref` | KEEP | `query_objects.py:390-394` SELECT list | n/a |
| `user.query_evaluation_receipts.model_refs_json` | KEEP | `query_objects.py:390-394` SELECT list | n/a |
| `user.query_evaluation_receipts.resolved_bounds_json` | KEEP | `query_objects.py:390-394` SELECT list | n/a |
| `user.query_evaluation_receipts.degradation_json` | KEEP | `query_objects.py:390-394` SELECT list | n/a |
| `user.query_evaluation_receipts.created_at_ms` | KEEP | `query_objects.py:390-394` SELECT list, `idx_query_evaluation_receipts_query_time` | n/a |
| `user.watched_query_baselines.query_hash` | KEEP | PK; `query_objects.py:289-300` JOIN target | n/a |
| `user.watched_query_baselines.result_set_id` | KEEP | `query_objects.py:289-300` JOIN target | n/a |
| `user.watched_query_baselines.updated_at_ms` | KEEP | referenced in watch-refresh comparison logic (put_watched_query_baseline read-before-write) | n/a |
| `user.result_set_holdout_policies.result_set_id` | KEEP | PK; `holdout_cohorts.py:124-131` `get_holdout_policy` | n/a |
| `user.result_set_holdout_policies.frame` | KEEP | `holdout_cohorts.py:126-131` SELECT list | n/a |
| `user.result_set_holdout_policies.selection_definition_json` | KEEP | `holdout_cohorts.py:126-131` SELECT list | n/a |
| `user.result_set_holdout_policies.intended_confirmation_use` | KEEP | `holdout_cohorts.py:126-131` SELECT list | n/a |
| `user.result_set_holdout_policies.authority` | KEEP | `holdout_cohorts.py:126-131` SELECT list | n/a |
| `user.result_set_holdout_policies.created_epoch` | KEEP | `holdout_cohorts.py:126-131` SELECT list | n/a |
| `user.result_set_holdout_policies.created_at_ms` | KEEP | table-level provenance timestamp, read in policy listing paths (same module) | n/a |
| `user.holdout_access_receipts.receipt_id` | KEEP | PK; `holdout_cohorts.py:199-217` `list_holdout_access_receipts` | n/a |
| `user.holdout_access_receipts.result_set_id` | KEEP | `holdout_cohorts.py:199-223` SELECT + filter, `idx_holdout_access_receipts_result_set` | n/a |
| `user.holdout_access_receipts.accessor_ref` | KEEP | `holdout_cohorts.py:199-217` SELECT list | n/a |
| `user.holdout_access_receipts.declared_confirmation` | KEEP | `holdout_cohorts.py:199-217` SELECT list | n/a |
| `user.holdout_access_receipts.contamination` | KEEP | `holdout_cohorts.py:199-226` SELECT list, `has_holdout_contamination` permanent-taint check | n/a |
| `user.holdout_access_receipts.reason` | KEEP | `holdout_cohorts.py:199-217` SELECT list | n/a |
| `user.holdout_access_receipts.accessed_at_ms` | KEEP | `holdout_cohorts.py:199-217` SELECT list + ORDER BY | n/a |
| `user.annotation_schemas.schema_id` | KEEP | PK; `user_annotations.py:69-98` full-row read | n/a |
| `user.annotation_schemas.schema_version` | KEEP | `user_annotations.py:69-98` SELECT list + ORDER BY | n/a |
| `user.annotation_schemas.definition_json` | KEEP | `user_annotations.py:69-98` SELECT list — the schema itself | n/a |
| `user.annotation_schemas.definition_sha256` | KEEP | `user_annotations.py:69-98` SELECT list — integrity fingerprint | n/a |
| `user.annotation_schemas.registered_at_ms` | KEEP | `user_annotations.py:69-98` SELECT list | n/a |
| `user.annotation_batches.batch_id` | KEEP | PK; `user_annotations.py:230-241` full-row read | n/a |
| `user.annotation_batches.schema_id` | KEEP | `user_annotations.py:230-241` SELECT list, FK | n/a |
| `user.annotation_batches.schema_version` | KEEP | `user_annotations.py:230-241` SELECT list, FK | n/a |
| `user.annotation_batches.target_ref` | KEEP | `user_annotations.py:230-241` SELECT list, `idx_annotation_batches_schema_target_time` | n/a |
| `user.annotation_batches.source_result_ref` | KEEP | `user_annotations.py:230-241` SELECT list, `idx_annotation_batches_source_result_time` | n/a |
| `user.annotation_batches.actor_ref` | KEEP | `user_annotations.py:230-241` SELECT list | n/a |
| `user.annotation_batches.model_ref` | KEEP | `user_annotations.py:230-241` SELECT list | n/a |
| `user.annotation_batches.prompt_ref` | KEEP | `user_annotations.py:230-241` SELECT list | n/a |
| `user.annotation_batches.total_count` | KEEP | `user_annotations.py:230-241` SELECT list | n/a |
| `user.annotation_batches.valid_count` | KEEP | `user_annotations.py:230-241` SELECT list | n/a |
| `user.annotation_batches.invalid_count` | KEEP | `user_annotations.py:230-241` SELECT list | n/a |
| `user.annotation_batches.abstained_count` | KEEP | `user_annotations.py:230-241` SELECT list | n/a |
| `user.annotation_batches.assertion_refs_json` | KEEP | `user_annotations.py:230-241` SELECT list — links to assertion rows | n/a |
| `user.annotation_batches.validation_failures_json` | KEEP | `user_annotations.py:230-241` SELECT list | n/a |
| `user.annotation_batches.metadata_json` | KEEP | `user_annotations.py:230-241` SELECT list | n/a |
| `user.annotation_batches.created_at_ms` | KEEP | `user_annotations.py:230-241` SELECT list, `idx_annotation_batches_*` ordering | n/a |
| `user.user_settings.setting_key` | KEEP | PK; `user_settings_write.py:108,132` full-row read | n/a |
| `user.user_settings.value_json` | KEEP | `user_settings_write.py:108,132` SELECT list | n/a |
| `user.user_settings.updated_at_ms` | KEEP | `user_settings_write.py:108,132` SELECT list | n/a |
| `user.user_settings.author_ref` | KEEP | `user_settings_write.py:108,132` SELECT list | n/a |
| `user.context_deliveries.snapshot_ref` | KEEP | PK; `context_delivery_write.py:173,214` full-row read | n/a |
| `user.context_deliveries.recipient_ref` | KEEP | `idx_context_deliveries_recipient_time`, `context_delivery_write.py:173` | n/a |
| `user.context_deliveries.run_ref` | KEEP | `idx_context_deliveries_run_time`, `context_delivery_write.py:173` | n/a |
| `user.context_deliveries.boundary` | KEEP | `context_delivery_write.py:173` full-row read | n/a |
| `user.context_deliveries.inheritance_mode` | KEEP | `context_delivery_write.py:173` full-row read | n/a |
| `user.context_deliveries.context_image_json` | KEEP | `context_delivery_write.py:173` full-row read — the delivered payload | n/a |
| `user.context_deliveries.context_image_sha256` | KEEP | `context_delivery_write.py:173` full-row read — integrity fingerprint | n/a |
| `user.context_deliveries.segment_refs_json` | KEEP | `context_delivery_write.py:173` full-row read | n/a |
| `user.context_deliveries.evidence_refs_json` | KEEP | `context_delivery_write.py:173` full-row read | n/a |
| `user.context_deliveries.assertion_refs_json` | KEEP | `context_delivery_write.py:173` full-row read | n/a |
| `user.context_deliveries.omissions_json` | KEEP | `context_delivery_write.py:173` full-row read | n/a |
| `user.context_deliveries.caveats_json` | KEEP | `context_delivery_write.py:173` full-row read | n/a |
| `user.context_deliveries.metadata_json` | KEEP | `context_delivery_write.py:173` full-row read | n/a |
| `user.context_deliveries.delivered_by_ref` | KEEP | `context_delivery_write.py:173` full-row read | n/a |
| `user.context_deliveries.delivered_at_ms` | KEEP | `idx_context_deliveries_*` ordering, `context_delivery_write.py:214` ORDER BY | n/a |

## Delegated / not re-derived this pass

Per the team-lead's explicit instruction, the following were cited from prior work rather than
re-audited column-by-column in this normalization pass:

- `source.raw_membership_census`, `raw_authority_parser_census`, `raw_authority_censuses`,
  `raw_authority_plans`, `raw_authority_census_plans`, `raw_authority_census_post_plans`,
  `raw_authority_blockers` (64 columns total): PURGE-pending-execution per `polylogue-w6hql`/
  `polylogue-lr6dx` — current writers still feed open P0 reconciliation work; not an executable
  drop yet.
- `ops.ingest_attempts.status`/`.outcome_code`, `ops.embedding_catchup_runs.status`/
  `.embedded_sessions`/`.skipped_sessions`/`.error_count` (run-status vocabulary columns):
  ownership cited to `polylogue-oj4oo`, not re-derived — their KEEP verdict here reflects real
  readers today, independent of `oj4oo`'s pending unification work.
- `source.blob_refs`, `raw_artifacts`, `history_sidecars`, `verified_blob_receipts`: checked
  against `polylogue-r9xsj`'s required acceptance surface before considering PURGE — all
  independently confirmed KEEP by reader evidence anyway, so r9xsj's blocking status was never
  invoked as a fallback for these.

## Ten most consequential PURGE findings (ranked by what they unlock)

Carried from the underlying audit's ranking (see individual rows above for full evidence):

1. `index.agent_meta_sidecar_purge_receipts.*` (9 columns) — sole writer is the one-time
   `agent_meta_sidecar_purge_apply.py` actuator; unlocks dropping the table, its index, and the
   actuator once the purge's evidence obligation is discharged.
2. `ops.fts_drift_samples.*` (10 columns) — full write→retention→read cycle exists, but its sole
   reader (`list_fts_drift_samples`) has zero production or test callers.
3. `ops.secret_scan_status.{scanned_at_ms,blocks_scanned,candidates_found}` (3 of 5 columns) —
   the other 2 columns are genuinely read via a cross-database JOIN a same-DB grep missed; a
   real partial-table PURGE, exactly the shape column granularity exists to catch.
4. `ops.cursor_lag_samples.*` minus identity/join columns (6 of 9) — the daemon computes its own
   rolling median fresh rather than reading the stored precomputed percentiles.
5. `user.query_names.{supersedes_query_hash,updated_at_ms}` / `user.watched_query_baselines.
   updated_at_ms` — flips the seed's blanket "no durable user-tier PURGE" call for the
   `polylogue-4p1` query-object cluster (note: `supersedes_query_hash` is graded UNCLEAR above,
   not PURGE, per this file's conservative re-grading — the other two remain PURGE).
6. `embeddings.embedding_derivation_state.{message_count,updated_at_ms}` — flips the seed's
   blanket KEEP; `embedding_status.message_count_embedded` is the actually-read analog.
7. `index.session_profiles.workflow_shape_method` — zero production writer AND reader; its three
   sibling `workflow_shape_*` columns are heavily used.
8. `index.threads.dominant_repo_id` — an FK-typed join column never wired at the write side; the
   plain-TEXT sibling `dominant_repo` is the live column.
9. `source.otlp_spans.*` (12 columns, seed-carried) — only DDL/migration/migration-test
   references; requires `polylogue-60i5` durable-migration admission before any drop.
10. `ops.ingest_attempts.evidence_ref` — write-only; its siblings are all read by
    `cursor_authority_reconcile.py:710`, but this one column never is.

## Verification performed for this normalization pass

- Every tier's table/column inventory was enumerated from live DDL: `SOURCE_DDL`, `INDEX_DDL`,
  `OPS_DDL`, `USER_DDL` executed against an in-memory SQLite connection and read back via
  `PRAGMA table_info`; `EMBEDDINGS_DDL` parsed as text (its `vec0` virtual table requires a
  loadable extension not present in this environment).
- index.db, ops.db, user.db, embeddings.db: every column-level verdict, evidence citation, and
  unlocks note was carried verbatim from the underlying per-tier working files
  (`.agent/scratch/gvzkr-cols-{index,ops,user,embeddings}.md`, not committed), whose row counts
  matched live DDL exactly once independently recomputed here.
- source.db: the 2 tables given full column-by-column treatment in the underlying working file
  (`raw_sessions`, `blob_refs`) kept their per-column evidence verbatim; the other 35 tables'
  table-level verdicts (KEEP-all / PURGE-candidate / UNCLEAR / PURGE-pending-execution) were
  expanded to one row per live-DDL column, since the working file's per-column granularity
  stopped at the table level for those 35 tables. No column's disposition was re-derived from
  scratch — only expanded from an existing table-level verdict to match live DDL's actual
  column list for that table.
- Anti-vacuity check performed mechanically (not a new lint — a one-off script run during
  authoring, per the dispatching instruction): every row's `tier.table.column` key was asserted
  unique, and the row count for the tier was asserted equal to the DDL-derived column count
  before this file was written. No duplicate or missing column exists in the table above.
- No live-archive values, personal data, or absolute filesystem paths outside this repository
  appear in any row (checked by substring scan for `/realm/`, `/home/`, and the operator's
  username across every evidence/unlocks cell before commit).

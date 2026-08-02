-- polylogue-u19l: the "act" half of live-source-verification-based
-- quarantine reconciliation. polylogue.storage.live_source_reconciliation
-- classifies quarantined raw_sessions rows against their live source file's
-- current bytes but never mutates anything; a new, explicitly
-- operator-invoked actuator (devtools workspace
-- raw-live-source-reconciliation-apply) promotes only exact-match /
-- codex-header-strip-match rows to revision_authority='byte_proven'.
--
-- raw_sessions.revision_authority is a CHECK-constrained closed vocabulary
-- ('asserted' / 'byte_proven' / 'quarantined') -- widening it needs a table
-- rebuild, not an additive ALTER, so this migration deliberately does not
-- add a new revision_authority value. Instead it records the DIFFERENT
-- evidence mechanism in a new, nullable column: NULL means "whatever
-- produced this row's current revision_authority before this column
-- existed" (the pre-existing revision-graph reconciler, for every row
-- promoted before this migration); 'live_source_verification_v1' means "this
-- row's revision_authority was promoted because its archived bytes were
-- re-verified against the still-present live source file", not because the
-- revision-graph reconciler established its position in a revision chain.
ALTER TABLE raw_sessions ADD COLUMN revision_authority_evidence TEXT
    CHECK(revision_authority_evidence IS NULL OR revision_authority_evidence IN ('live_source_verification_v1'));

-- One immutable receipt per promoted row: what was compared, what matched,
-- when, by which tool version, and against which verified backup manifest.
-- This is the durable audit trail the promotion pass writes in the same
-- transaction as the revision_authority UPDATE -- never edited afterward.
CREATE TABLE IF NOT EXISTS raw_live_source_reconciliation_receipts (
    raw_id                      TEXT PRIMARY KEY REFERENCES raw_sessions(raw_id) ON DELETE CASCADE,
    verdict                     TEXT NOT NULL CHECK(verdict IN ('exact_match', 'codex_header_strip_match')),
    previous_revision_authority TEXT NOT NULL CHECK(previous_revision_authority IN ('asserted', 'byte_proven', 'quarantined')),
    source_path                 TEXT NOT NULL,
    blob_hash                   BLOB NOT NULL CHECK(length(blob_hash) = 32),
    blob_size                   INTEGER NOT NULL CHECK(blob_size >= 0),
    compared_at_ms              INTEGER NOT NULL CHECK(compared_at_ms >= 0),
    tool_version                TEXT NOT NULL,
    backup_manifest_path        TEXT NOT NULL,
    detail                      TEXT NOT NULL DEFAULT ''
) STRICT;

CREATE INDEX IF NOT EXISTS idx_raw_live_source_reconciliation_receipts_compared_at
ON raw_live_source_reconciliation_receipts(compared_at_ms);

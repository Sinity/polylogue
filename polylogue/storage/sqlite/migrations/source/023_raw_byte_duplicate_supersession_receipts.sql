-- polylogue-6753s: 4,305 raw_sessions rows (17.6 of 22.9 GiB, measured
-- 2026-08-03) are revision_authority='quarantined' with logical_source_key
-- IS NULL -- never touched by the reconciler, because every reconciliation
-- path keys off logical_source_key -- and are byte-identical
-- (same blob_hash) to some OTHER raw_sessions row that already has a
-- materialized session in index.db. These are re-acquisitions/re-syncs of
-- content the archive already indexed, not missing content. A new,
-- explicitly operator-invoked actuator (devtools workspace
-- raw-byte-duplicate-supersession-apply) promotes exactly these rows to
-- revision_authority='byte_proven'.
--
-- Deliberately does NOT widen revision_authority_evidence's closed CHECK
-- vocabulary (that requires a full raw_sessions table rebuild, migration
-- 021's own precedent for why that is expensive) and does NOT touch
-- predecessor_raw_id / baseline_raw_id / acquisition_generation
-- (revision-graph linkage owned by classify_raw_revision_cohort) or the
-- duplicate twin's own raw_sessions row at all. Mirrors
-- raw_membership_writeback_receipts (migration 019): a distinct,
-- independently-verifiable evidence mechanism gets its own dedicated
-- receipt table recording what was actually proven, rather than reusing
-- revision_authority_evidence for a mechanism it was never scoped to.
CREATE TABLE IF NOT EXISTS raw_byte_duplicate_supersession_receipts (
    raw_id                      TEXT PRIMARY KEY REFERENCES raw_sessions(raw_id) ON DELETE CASCADE,
    blob_hash                   BLOB NOT NULL CHECK(length(blob_hash) = 32),
    blob_size                   INTEGER NOT NULL CHECK(blob_size >= 0),
    duplicate_of_raw_id         TEXT NOT NULL,
    duplicate_of_session_id     TEXT NOT NULL,
    previous_revision_authority TEXT NOT NULL CHECK(previous_revision_authority IN ('asserted', 'byte_proven', 'quarantined')),
    promoted_at_ms              INTEGER NOT NULL CHECK(promoted_at_ms >= 0),
    tool_version                TEXT NOT NULL,
    backup_manifest_path        TEXT NOT NULL,
    detail                      TEXT NOT NULL DEFAULT ''
) STRICT;

CREATE INDEX IF NOT EXISTS idx_raw_byte_duplicate_supersession_receipts_promoted_at
ON raw_byte_duplicate_supersession_receipts(promoted_at_ms);

CREATE INDEX IF NOT EXISTS idx_raw_byte_duplicate_supersession_receipts_duplicate_of
ON raw_byte_duplicate_supersession_receipts(duplicate_of_raw_id);

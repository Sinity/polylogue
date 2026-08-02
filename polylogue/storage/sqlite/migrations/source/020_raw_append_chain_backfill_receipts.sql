-- polylogue-lb39z (Phase 1, item 3): 2,712 raw_sessions rows are
-- revision_kind='append', revision_authority='quarantined', and have no
-- raw_session_memberships row at all -- a genuine fixed point, because the
-- only mechanism that ever promotes an append raw
-- (_promote_contiguous_append_evidence) requires its byte-contiguous
-- predecessor to already be byte_proven. A new, explicitly operator-invoked
-- actuator (devtools workspace raw-append-chain-backfill-apply) proves each
-- such row's own claimed [append_start_offset:append_end_offset) byte range
-- directly against its live source file's current bytes -- a proof that does
-- not depend on any ancestor's authority -- and promotes exact matches to
-- revision_authority='byte_proven'.
--
-- Reuses the existing revision_authority_evidence='live_source_verification_v1'
-- value (migration 018): the proof mechanism (byte-window comparison against
-- the live source file) is identical to polylogue-u19l's actuator; only the
-- target population (membershipless append rows specifically) differs, and
-- that distinction is what this dedicated receipt table records.
CREATE TABLE IF NOT EXISTS raw_append_chain_backfill_receipts (
    raw_id                          TEXT PRIMARY KEY REFERENCES raw_sessions(raw_id) ON DELETE CASCADE,
    logical_source_key              TEXT,
    source_path                     TEXT NOT NULL,
    blob_hash                       BLOB NOT NULL CHECK(length(blob_hash) = 32),
    blob_size                       INTEGER NOT NULL CHECK(blob_size >= 0),
    append_start_offset             INTEGER NOT NULL CHECK(append_start_offset >= 0),
    append_end_offset               INTEGER NOT NULL CHECK(append_end_offset > append_start_offset),
    matched_after_codex_header_strip INTEGER NOT NULL CHECK(matched_after_codex_header_strip IN (0, 1)),
    previous_revision_authority     TEXT NOT NULL CHECK(previous_revision_authority IN ('asserted', 'byte_proven', 'quarantined')),
    compared_at_ms                  INTEGER NOT NULL CHECK(compared_at_ms >= 0),
    tool_version                    TEXT NOT NULL,
    backup_manifest_path            TEXT NOT NULL,
    detail                          TEXT NOT NULL DEFAULT ''
) STRICT;

CREATE INDEX IF NOT EXISTS idx_raw_append_chain_backfill_receipts_compared_at
ON raw_append_chain_backfill_receipts(compared_at_ms);

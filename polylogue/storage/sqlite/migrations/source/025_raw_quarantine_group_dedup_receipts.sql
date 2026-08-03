-- polylogue-zm4w8: 1,777 raw_sessions rows (22.2 GiB, measured 2026-08-03) are
-- pure redundant duplicates -- same source_path AND same blob_hash as another
-- raw_sessions row -- among the codex-session quarantine backlog. ZERO of
-- these duplicate blob_hash values have any non-quarantined ("indexed")
-- twin anywhere in raw_sessions, so raw_byte_duplicate_supersession_apply
-- (which only matches a quarantined raw against an ALREADY-INDEXED twin)
-- does not and cannot catch this class: every member of one of these groups
-- starts out quarantined, with nothing yet materialized for any of them.
--
-- A new, explicitly operator-invoked actuator (devtools workspace
-- raw-quarantine-group-dedup-apply) promotes exactly ONE representative raw
-- per (source_path, blob_hash) group through the real ingest/materialization
-- path (ParsingService.parse_from_raw -> write_parsed_session_to_archive ->
-- refresh_session_insights_bulk) so it becomes a genuine indexed session,
-- then marks the rest of the group's raws revision_authority='byte_proven'
-- (reusing raw_byte_duplicate_supersession's exact "quarantined, now proven
-- byte-identical to an indexed twin" precedent -- revision_authority is a
-- closed 3-value CHECK vocabulary with no 'superseded' member, and widening
-- it needs a full raw_sessions table rebuild, migration 021's own precedent
-- for why that is expensive). This receipt table is the durable, per-row
-- record of *which* representative raw and materialized session each
-- duplicate was superseded by -- never silently, exactly like
-- raw_byte_duplicate_supersession_receipts (migration 023) records its own
-- distinct evidence mechanism.
CREATE TABLE IF NOT EXISTS raw_quarantine_group_dedup_receipts (
    raw_id                     TEXT PRIMARY KEY REFERENCES raw_sessions(raw_id) ON DELETE CASCADE,
    source_path                TEXT NOT NULL,
    blob_hash                  BLOB NOT NULL CHECK(length(blob_hash) = 32),
    blob_size                  INTEGER NOT NULL CHECK(blob_size >= 0),
    representative_raw_id      TEXT NOT NULL,
    representative_session_id  TEXT NOT NULL,
    promoted_at_ms              INTEGER NOT NULL CHECK(promoted_at_ms >= 0),
    tool_version                TEXT NOT NULL,
    backup_manifest_path        TEXT NOT NULL,
    detail                      TEXT NOT NULL DEFAULT ''
) STRICT;

CREATE INDEX IF NOT EXISTS idx_raw_quarantine_group_dedup_receipts_promoted_at
ON raw_quarantine_group_dedup_receipts(promoted_at_ms);

CREATE INDEX IF NOT EXISTS idx_raw_quarantine_group_dedup_receipts_representative
ON raw_quarantine_group_dedup_receipts(representative_raw_id);

"""Source-tier raw-capture DDL fragment for archive.

source.db stores acquired bytes and source evidence. Parsed/read-model tables
live in index.db and are rebuilt from this tier.
"""

from __future__ import annotations

from polylogue.core.enums import ArtifactSupportStatus, Origin, Provider, ValidationMode, ValidationStatus
from polylogue.storage.sqlite.archive_tiers.common import check, nullable_check

SOURCE_SCHEMA_VERSION = 12

SOURCE_DDL = f"""
CREATE TABLE IF NOT EXISTS raw_sessions (
    raw_id                  TEXT PRIMARY KEY,
    origin                  TEXT NOT NULL CHECK ({check("origin", Origin)}),
    capture_mode            TEXT CHECK ({nullable_check("capture_mode", Provider)}),
    native_id               TEXT,
    source_path             TEXT NOT NULL,
    source_index            INTEGER NOT NULL DEFAULT 0,
    blob_hash               BLOB NOT NULL CHECK(length(blob_hash) = 32),
    blob_size               INTEGER NOT NULL CHECK(blob_size >= 0),
    acquired_at_ms          INTEGER NOT NULL,
    file_mtime_ms           INTEGER,
    parsed_at_ms            INTEGER,
    parse_error             TEXT,
    validated_at_ms         INTEGER,
    validation_status       TEXT CHECK ({nullable_check("validation_status", ValidationStatus)}),
    validation_error        TEXT,
    validation_drift_count  INTEGER NOT NULL DEFAULT 0 CHECK(validation_drift_count >= 0),
    validation_mode         TEXT CHECK ({nullable_check("validation_mode", ValidationMode)}),
    detection_warnings_json TEXT NOT NULL DEFAULT '[]'
    ,logical_source_key      TEXT
    ,revision_kind           TEXT NOT NULL DEFAULT 'unknown'
        CHECK(revision_kind IN ('full', 'append', 'unknown'))
    ,source_revision         TEXT
    ,predecessor_source_revision TEXT
    ,predecessor_raw_id      TEXT
    ,baseline_raw_id         TEXT
    ,append_start_offset     INTEGER CHECK(append_start_offset >= 0)
    ,append_end_offset       INTEGER CHECK(append_end_offset > append_start_offset)
    ,acquisition_generation  INTEGER CHECK(acquisition_generation >= 0)
    ,revision_authority      TEXT NOT NULL DEFAULT 'quarantined'
        CHECK(revision_authority IN ('asserted', 'byte_proven', 'quarantined'))
) STRICT;

CREATE INDEX IF NOT EXISTS idx_raw_sessions_origin
ON raw_sessions(origin);

CREATE INDEX IF NOT EXISTS idx_raw_sessions_origin_native
ON raw_sessions(origin, native_id)
WHERE native_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_raw_sessions_source_path
ON raw_sessions(source_path, source_index);

CREATE INDEX IF NOT EXISTS idx_raw_sessions_parse_ready
ON raw_sessions(raw_id)
WHERE parsed_at_ms IS NULL
  AND validated_at_ms IS NOT NULL
  AND (validation_status IS NULL OR validation_status != 'failed');

CREATE INDEX IF NOT EXISTS idx_raw_sessions_logical_revision
ON raw_sessions(logical_source_key, acquisition_generation, raw_id)
WHERE logical_source_key IS NOT NULL;

CREATE TABLE IF NOT EXISTS raw_session_memberships (
    raw_id                  TEXT NOT NULL REFERENCES raw_sessions(raw_id) ON DELETE CASCADE,
    logical_source_key      TEXT NOT NULL,
    provider_session_id     TEXT NOT NULL,
    source_revision         TEXT NOT NULL,
    normalized_content_hash BLOB NOT NULL CHECK(length(normalized_content_hash) = 32),
    message_count           INTEGER NOT NULL CHECK(message_count >= 0),
    predecessor_raw_id      TEXT,
    acquisition_generation  INTEGER NOT NULL DEFAULT 0 CHECK(acquisition_generation >= 0),
    revision_authority      TEXT NOT NULL DEFAULT 'quarantined'
        CHECK(revision_authority IN ('byte_proven', 'quarantined')),
    decision                TEXT CHECK(decision IN (
                                'applied', 'superseded_equivalent', 'superseded_prefix',
                                'ambiguous', 'deferred'
                            )),
    decided_at_ms           INTEGER CHECK(decided_at_ms >= 0),
    PRIMARY KEY(raw_id, logical_source_key),
    CHECK((decision IS NULL) = (decided_at_ms IS NULL))
) STRICT;

CREATE INDEX IF NOT EXISTS idx_raw_session_memberships_logical
ON raw_session_memberships(logical_source_key, acquisition_generation, raw_id);

CREATE INDEX IF NOT EXISTS idx_raw_session_memberships_pending
ON raw_session_memberships(raw_id)
WHERE decision IS NULL OR decision IN ('ambiguous', 'deferred');

CREATE TABLE IF NOT EXISTS raw_membership_census (
    raw_id             TEXT PRIMARY KEY REFERENCES raw_sessions(raw_id) ON DELETE CASCADE,
    parser_fingerprint TEXT NOT NULL,
    status             TEXT NOT NULL CHECK(status IN ('complete', 'failed', 'non_session')),
    member_count       INTEGER NOT NULL CHECK(member_count >= 0),
    censused_at_ms     INTEGER NOT NULL CHECK(censused_at_ms >= 0),
    detail             TEXT NOT NULL DEFAULT ''
) STRICT;

CREATE TABLE IF NOT EXISTS blob_refs (
    blob_hash       BLOB NOT NULL CHECK(length(blob_hash) = 32),
    ref_id          TEXT NOT NULL,
    ref_type        TEXT NOT NULL CHECK(ref_type IN ('raw_payload', 'attachment', 'sidecar')),
    source_path     TEXT,
    size_bytes      INTEGER NOT NULL CHECK(size_bytes >= 0),
    acquired_at_ms  INTEGER NOT NULL,
    PRIMARY KEY(blob_hash, ref_type, ref_id)
) STRICT;

CREATE INDEX IF NOT EXISTS idx_blob_refs_ref_id
ON blob_refs(ref_id);

CREATE TABLE IF NOT EXISTS blob_publication_reservations (
    publication_id   TEXT PRIMARY KEY,
    blob_hash        BLOB NOT NULL CHECK(length(blob_hash) = 32),
    size_bytes       INTEGER NOT NULL CHECK(size_bytes >= 0),
    publisher_id     TEXT NOT NULL,
    reserved_at_ms   INTEGER NOT NULL
) STRICT;

CREATE INDEX IF NOT EXISTS idx_blob_publication_reservations_hash
ON blob_publication_reservations(blob_hash);

CREATE TABLE IF NOT EXISTS gc_generations (
    generation_id    TEXT PRIMARY KEY,
    started_at_ms    INTEGER NOT NULL,
    completed_at_ms  INTEGER,
    reclaimed_count  INTEGER NOT NULL DEFAULT 0 CHECK(reclaimed_count >= 0),
    reclaimed_bytes  INTEGER NOT NULL DEFAULT 0 CHECK(reclaimed_bytes >= 0)
) STRICT;

CREATE TABLE IF NOT EXISTS raw_artifacts (
    artifact_id              TEXT PRIMARY KEY,
    raw_id                   TEXT NOT NULL REFERENCES raw_sessions(raw_id) ON DELETE CASCADE,
    origin                   TEXT NOT NULL CHECK ({check("origin", Origin)}),
    source_path              TEXT NOT NULL,
    source_index             INTEGER NOT NULL DEFAULT 0,
    artifact_kind            TEXT NOT NULL,
    support_status           TEXT NOT NULL CHECK ({check("support_status", ArtifactSupportStatus)}),
    classification_reason    TEXT NOT NULL,
    parse_as_session         INTEGER NOT NULL DEFAULT 0 CHECK(parse_as_session IN (0, 1)),
    schema_eligible          INTEGER NOT NULL DEFAULT 0 CHECK(schema_eligible IN (0, 1)),
    malformed_jsonl_lines    INTEGER NOT NULL DEFAULT 0 CHECK(malformed_jsonl_lines >= 0),
    decode_error             TEXT,
    cohort_id                TEXT,
    link_group_key           TEXT,
    sidecar_agent_type       TEXT,
    first_observed_at_ms     INTEGER NOT NULL,
    last_observed_at_ms      INTEGER NOT NULL
) STRICT;

CREATE UNIQUE INDEX IF NOT EXISTS idx_raw_artifacts_source_identity
ON raw_artifacts(origin, source_path, source_index);

CREATE INDEX IF NOT EXISTS idx_raw_artifacts_raw_id
ON raw_artifacts(raw_id);

CREATE TABLE IF NOT EXISTS raw_hook_events (
    hook_event_id   TEXT PRIMARY KEY,
    origin          TEXT NOT NULL CHECK ({check("origin", Origin)}),
    native_id       TEXT,
    session_native_id TEXT,
    source_path     TEXT NOT NULL,
    event_type      TEXT NOT NULL,
    payload_json    TEXT NOT NULL,
    observed_at_ms  INTEGER NOT NULL
) STRICT;

CREATE INDEX IF NOT EXISTS idx_raw_hook_events_session
ON raw_hook_events(origin, session_native_id, observed_at_ms);

CREATE TABLE IF NOT EXISTS otlp_spans (
    span_id           TEXT PRIMARY KEY,
    trace_id          TEXT NOT NULL,
    parent_span_id    TEXT,
    origin            TEXT CHECK ({nullable_check("origin", Origin)}),
    session_native_id TEXT,
    name              TEXT NOT NULL,
    kind              TEXT,
    attributes_json   TEXT NOT NULL DEFAULT '{{}}',
    events_json       TEXT NOT NULL DEFAULT '[]',
    started_at_ms     INTEGER,
    ended_at_ms       INTEGER,
    received_at_ms    INTEGER NOT NULL
) STRICT;

CREATE INDEX IF NOT EXISTS idx_otlp_spans_trace
ON otlp_spans(trace_id, started_at_ms DESC);

CREATE INDEX IF NOT EXISTS idx_otlp_spans_session
ON otlp_spans(origin, session_native_id, started_at_ms DESC)
WHERE session_native_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS history_sidecars (
    sidecar_id      TEXT PRIMARY KEY,
    origin          TEXT NOT NULL CHECK ({check("origin", Origin)}),
    source_path     TEXT NOT NULL,
    payload_json    TEXT NOT NULL,
    observed_at_ms  INTEGER NOT NULL,
    content_hash    BLOB NOT NULL CHECK(length(content_hash) = 32)
) STRICT;

CREATE UNIQUE INDEX IF NOT EXISTS idx_history_sidecars_path_hash
ON history_sidecars(origin, source_path, content_hash);

CREATE TABLE IF NOT EXISTS sinex_publication_obligations (
    object_id           TEXT NOT NULL,
    protocol_version     TEXT NOT NULL CHECK(protocol_version != ''),
    revision_id          TEXT NOT NULL,
    manifest_digest      TEXT NOT NULL,
    mode                 TEXT NOT NULL CHECK(mode IN ('mirror', 'primary')),
    status               TEXT NOT NULL DEFAULT 'pending'
                             CHECK(status IN ('pending', 'publishing', 'confirmed', 'durable_debt', 'rejected')),
    attempt_count        INTEGER NOT NULL DEFAULT 0 CHECK(attempt_count >= 0),
    last_attempt_at_ms   INTEGER,
    last_receipt_state   TEXT,
    last_error           TEXT,
    created_at_ms        INTEGER NOT NULL,
    updated_at_ms        INTEGER NOT NULL,
    retired_at_ms        INTEGER,
    next_attempt_at_ms   INTEGER,
    PRIMARY KEY(object_id, protocol_version, revision_id, manifest_digest)
) STRICT;

CREATE INDEX IF NOT EXISTS idx_sinex_publication_obligations_pending
ON sinex_publication_obligations(status, next_attempt_at_ms, created_at_ms)
WHERE status IN ('pending', 'publishing', 'durable_debt');

CREATE INDEX IF NOT EXISTS idx_sinex_publication_obligations_object
ON sinex_publication_obligations(object_id, created_at_ms DESC);

CREATE TABLE IF NOT EXISTS sinex_publication_payloads (
    object_id            TEXT NOT NULL,
    protocol_version     TEXT NOT NULL,
    revision_id          TEXT NOT NULL,
    manifest_digest      TEXT NOT NULL,
    manifest_bytes       BLOB NOT NULL,
    manifest_sha256      TEXT NOT NULL CHECK(length(manifest_sha256) = 64),
    manifest_size_bytes  INTEGER NOT NULL CHECK(manifest_size_bytes >= 0),
    segment_count        INTEGER NOT NULL CHECK(segment_count >= 0),
    total_size_bytes     INTEGER NOT NULL CHECK(total_size_bytes >= manifest_size_bytes),
    staged_at_ms         INTEGER NOT NULL CHECK(staged_at_ms >= 0),
    PRIMARY KEY(object_id, protocol_version, revision_id, manifest_digest),
    FOREIGN KEY(object_id, protocol_version, revision_id, manifest_digest)
        REFERENCES sinex_publication_obligations(
            object_id, protocol_version, revision_id, manifest_digest
        ) ON DELETE CASCADE
) STRICT;

CREATE TABLE IF NOT EXISTS sinex_publication_segments (
    object_id          TEXT NOT NULL,
    protocol_version   TEXT NOT NULL,
    revision_id        TEXT NOT NULL,
    manifest_digest    TEXT NOT NULL,
    position           INTEGER NOT NULL CHECK(position >= 0),
    segment_name       TEXT NOT NULL CHECK(segment_name != ''),
    segment_bytes      BLOB NOT NULL,
    segment_sha256     TEXT NOT NULL CHECK(length(segment_sha256) = 64),
    size_bytes         INTEGER NOT NULL CHECK(size_bytes >= 0),
    PRIMARY KEY(object_id, protocol_version, revision_id, manifest_digest, position),
    UNIQUE(object_id, protocol_version, revision_id, manifest_digest, segment_name),
    FOREIGN KEY(object_id, protocol_version, revision_id, manifest_digest)
        REFERENCES sinex_publication_payloads(
            object_id, protocol_version, revision_id, manifest_digest
        ) ON DELETE CASCADE
) STRICT;

CREATE TABLE IF NOT EXISTS sinex_publication_receipts (
    object_id          TEXT NOT NULL,
    protocol_version   TEXT NOT NULL,
    revision_id        TEXT NOT NULL,
    manifest_digest    TEXT NOT NULL,
    attempt_number     INTEGER NOT NULL CHECK(attempt_number > 0),
    request_id         TEXT NOT NULL,
    receipt_state      TEXT CHECK(receipt_state IS NULL OR receipt_state IN (
                            'raw_accepted', 'persisted_confirmed', 'durable_debt',
                            'spool_accepted_lossless', 'rejected'
                        )),
    receipt_detail     TEXT NOT NULL DEFAULT '',
    error_code         TEXT,
    received_at_ms     INTEGER NOT NULL CHECK(received_at_ms >= 0),
    PRIMARY KEY(object_id, protocol_version, revision_id, manifest_digest, attempt_number),
    FOREIGN KEY(object_id, protocol_version, revision_id, manifest_digest)
        REFERENCES sinex_publication_obligations(
            object_id, protocol_version, revision_id, manifest_digest
        ) ON DELETE CASCADE
) STRICT;

CREATE INDEX IF NOT EXISTS idx_sinex_publication_receipts_recent
ON sinex_publication_receipts(received_at_ms DESC);

-- Durable removed-content ledger (polylogue-27m). A row here is the
-- authoritative "this content is forgotten on purpose" marker for
-- standalone/off-mode excision: the acquire-time write chokepoint
-- (``write_source_raw_session``/``write_source_raw_session_blob_ref``)
-- refuses to re-store a raw payload whose ``blob_hash`` matches
-- ``removed_hash``, so an ordinary re-ingest of unmodified source files
-- cannot resurrect excised content even after an index.db rebuild.
-- ``span_start``/``span_end`` are populated only for a sub-payload
-- excision (e.g. a detected secret candidate span); both are NULL for a
-- whole-raw-session excision. This table is never queried for its own
-- sake by a reader -- it exists purely as a write-time gate plus forensic
-- trail, so no secret span coordinates ever carry the removed literal,
-- only byte offsets into the (now-deleted) payload.
CREATE TABLE IF NOT EXISTS excised_content (
    removed_hash    BLOB NOT NULL CHECK(length(removed_hash) = 32),
    hash_kind       TEXT NOT NULL DEFAULT 'blob_hash' CHECK(hash_kind IN ('blob_hash')),
    reason          TEXT NOT NULL,
    actor           TEXT NOT NULL,
    prior_revision  TEXT,
    span_start      INTEGER CHECK(span_start IS NULL OR span_start >= 0),
    span_end        INTEGER CHECK(span_end IS NULL OR span_end > span_start),
    excised_at_ms   INTEGER NOT NULL,
    PRIMARY KEY(removed_hash, hash_kind)
) STRICT;

"""

__all__ = ["SOURCE_DDL", "SOURCE_SCHEMA_VERSION"]

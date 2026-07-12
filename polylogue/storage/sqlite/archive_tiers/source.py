"""Source-tier raw-capture DDL fragment for archive.

source.db stores acquired bytes and source evidence. Parsed/read-model tables
live in index.db and are rebuilt from this tier.
"""

from __future__ import annotations

from polylogue.core.enums import ArtifactSupportStatus, Origin, Provider, ValidationMode, ValidationStatus
from polylogue.storage.sqlite.archive_tiers.common import check, nullable_check

SOURCE_SCHEMA_VERSION = 9

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

"""

__all__ = ["SOURCE_DDL", "SOURCE_SCHEMA_VERSION"]

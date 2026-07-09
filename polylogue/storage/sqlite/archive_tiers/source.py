"""Source-tier raw-capture DDL fragment for archive.

source.db stores acquired bytes and source evidence. Parsed/read-model tables
live in index.db and are rebuilt from this tier.
"""

from __future__ import annotations

from polylogue.core.enums import ArtifactSupportStatus, Origin, ValidationMode, ValidationStatus
from polylogue.storage.sqlite.archive_tiers.common import check, nullable_check

SOURCE_SCHEMA_VERSION = 3

SOURCE_DDL = f"""
CREATE TABLE IF NOT EXISTS raw_sessions (
    raw_id                  TEXT PRIMARY KEY,
    origin                  TEXT NOT NULL CHECK ({check("origin", Origin)}),
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

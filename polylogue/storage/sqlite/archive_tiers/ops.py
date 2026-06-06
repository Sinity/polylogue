"""Disposable ops-tier DDL for the archive."""

from __future__ import annotations

from polylogue.core.enums import Origin
from polylogue.storage.sqlite.archive_tiers.common import check, nullable_check

OPS_SCHEMA_VERSION = 1

OPS_DDL = f"""
CREATE TABLE IF NOT EXISTS ingest_cursor (
    source_path          TEXT PRIMARY KEY,
    origin               TEXT CHECK ({check("origin", Origin)} OR origin IS NULL),
    stat_size            INTEGER,
    byte_offset          INTEGER,
    last_complete_newline INTEGER,
    record_count         INTEGER NOT NULL DEFAULT 0 CHECK(record_count >= 0),
    last_record_ts_ms    INTEGER,
    parser_fingerprint   TEXT,
    content_fingerprint  TEXT,
    tail_hash            TEXT,
    st_dev               INTEGER,
    st_ino               INTEGER,
    mtime_ns             INTEGER,
    failure_count        INTEGER NOT NULL DEFAULT 0 CHECK(failure_count >= 0),
    next_retry_at        TEXT,
    excluded             INTEGER NOT NULL DEFAULT 0 CHECK(excluded IN (0, 1)),
    updated_at_ms        INTEGER NOT NULL
) STRICT;

CREATE INDEX IF NOT EXISTS idx_ingest_cursor_attention
ON ingest_cursor(failure_count, excluded, source_path);

CREATE TABLE IF NOT EXISTS ingest_attempts (
    attempt_id             TEXT PRIMARY KEY,
    source_path            TEXT,
    origin                 TEXT CHECK ({check("origin", Origin)} OR origin IS NULL),
    status                 TEXT NOT NULL CHECK(status IN ('running', 'completed', 'failed', 'interrupted')),
    phase                  TEXT,
    started_at_ms          INTEGER NOT NULL,
    heartbeat_at_ms        INTEGER,
    finished_at_ms         INTEGER,
    parsed_raw_count       INTEGER NOT NULL DEFAULT 0 CHECK(parsed_raw_count >= 0),
    materialized_count     INTEGER NOT NULL DEFAULT 0 CHECK(materialized_count >= 0),
    error_message          TEXT,
    source_paths_json      TEXT NOT NULL DEFAULT '[]'
) STRICT;

CREATE INDEX IF NOT EXISTS idx_ingest_attempts_status
ON ingest_attempts(status, heartbeat_at_ms);

CREATE TABLE IF NOT EXISTS convergence_debt (
    debt_id        TEXT PRIMARY KEY,
    stage          TEXT NOT NULL,
    target_type    TEXT NOT NULL,
    target_id      TEXT NOT NULL,
    status         TEXT NOT NULL DEFAULT 'failed' CHECK(status IN ('failed', 'deferred')),
    priority       INTEGER NOT NULL DEFAULT 0,
    attempts       INTEGER NOT NULL DEFAULT 0 CHECK(attempts >= 0),
    last_error     TEXT,
    next_retry_at  TEXT,
    materializer_version TEXT,
    created_at_ms  INTEGER NOT NULL,
    updated_at_ms  INTEGER NOT NULL,
    UNIQUE(stage, target_type, target_id)
) STRICT;

CREATE INDEX IF NOT EXISTS idx_convergence_debt_stage
ON convergence_debt(stage, priority DESC, updated_at_ms);

CREATE TABLE IF NOT EXISTS cursor_lag_samples (
    sample_id        TEXT PRIMARY KEY,
    family           TEXT NOT NULL,
    source_path      TEXT,
    lag_ms           INTEGER NOT NULL CHECK(lag_ms >= 0),
    stuck_file_count INTEGER NOT NULL DEFAULT 1 CHECK(stuck_file_count >= 0),
    p50_lag_ms       INTEGER NOT NULL DEFAULT 0 CHECK(p50_lag_ms >= 0),
    p95_lag_ms       INTEGER NOT NULL DEFAULT 0 CHECK(p95_lag_ms >= 0),
    severity         TEXT NOT NULL CHECK(severity IN ('info', 'warning', 'error', 'critical')),
    sampled_at_ms    INTEGER NOT NULL
) STRICT;

CREATE INDEX IF NOT EXISTS idx_cursor_lag_samples_family_time
ON cursor_lag_samples(family, sampled_at_ms DESC);

CREATE TABLE IF NOT EXISTS daemon_stage_events (
    event_id       TEXT PRIMARY KEY,
    attempt_id     TEXT,
    stage          TEXT NOT NULL,
    status         TEXT NOT NULL,
    observed_at_ms INTEGER NOT NULL,
    payload_json   TEXT NOT NULL DEFAULT '{{}}'
) STRICT;

CREATE TABLE IF NOT EXISTS daemon_events (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_ms          INTEGER NOT NULL,
    kind           TEXT NOT NULL,
    operation_id   TEXT,
    payload_json   TEXT NOT NULL DEFAULT '{{}}'
) STRICT;

CREATE INDEX IF NOT EXISTS idx_daemon_events_kind ON daemon_events(kind);
CREATE INDEX IF NOT EXISTS idx_daemon_events_ts ON daemon_events(ts_ms);

CREATE TABLE IF NOT EXISTS embedding_catchup_runs (
    run_id              TEXT PRIMARY KEY,
    started_at_ms       INTEGER NOT NULL,
    finished_at_ms      INTEGER,
    status              TEXT NOT NULL CHECK(status IN ('running', 'completed', 'failed', 'cancelled')),
    origin              TEXT CHECK ({nullable_check("origin", Origin)}),
    scanned_sessions    INTEGER NOT NULL DEFAULT 0 CHECK(scanned_sessions >= 0),
    embedded_messages   INTEGER NOT NULL DEFAULT 0 CHECK(embedded_messages >= 0),
    estimated_cost_usd  REAL,
    error_message       TEXT
) STRICT;

CREATE TABLE IF NOT EXISTS otlp_spans (
    trace_id        TEXT NOT NULL,
    span_id         TEXT NOT NULL,
    parent_span_id  TEXT,
    origin          TEXT CHECK ({nullable_check("origin", Origin)}),
    name            TEXT NOT NULL,
    kind            TEXT,
    started_at_ms   INTEGER,
    ended_at_ms     INTEGER,
    attributes_json TEXT NOT NULL DEFAULT '{{}}',
    events_json     TEXT NOT NULL DEFAULT '[]',
    PRIMARY KEY(trace_id, span_id)
) STRICT;

CREATE TABLE IF NOT EXISTS otlp_telemetry (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    received_at_ms   INTEGER NOT NULL,
    signal_type      TEXT NOT NULL,
    content_type     TEXT NOT NULL,
    payload          BLOB NOT NULL,
    resource_count   INTEGER NOT NULL DEFAULT 0,
    span_count       INTEGER,
    metric_count     INTEGER,
    log_record_count INTEGER
) STRICT;

CREATE INDEX IF NOT EXISTS idx_otlp_telemetry_signal_time
ON otlp_telemetry(signal_type, received_at_ms DESC);
"""

__all__ = ["OPS_DDL", "OPS_SCHEMA_VERSION"]

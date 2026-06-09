"""User-tier DDL for archive."""

from __future__ import annotations

USER_SCHEMA_VERSION = 1

USER_DDL = """
CREATE TABLE IF NOT EXISTS marks (
    mark_id         TEXT PRIMARY KEY,
    target_type     TEXT NOT NULL CHECK(target_type IN ('session', 'message', 'block', 'attachment', 'paste_span', 'work_event', 'phase', 'thread')),
    target_id       TEXT NOT NULL,
    mark_type       TEXT NOT NULL,
    label           TEXT,
    created_at_ms   INTEGER NOT NULL,
    updated_at_ms   INTEGER NOT NULL,
    metadata_json   TEXT NOT NULL DEFAULT '{}'
) STRICT;

CREATE INDEX IF NOT EXISTS idx_marks_target
ON marks(target_type, target_id);

CREATE TABLE IF NOT EXISTS annotations (
    annotation_id  TEXT PRIMARY KEY,
    target_type    TEXT NOT NULL CHECK(target_type IN ('session', 'message', 'block', 'attachment', 'paste_span', 'work_event', 'phase', 'thread')),
    target_id      TEXT NOT NULL,
    body           TEXT NOT NULL,
    created_at_ms  INTEGER NOT NULL,
    updated_at_ms  INTEGER NOT NULL
) STRICT;

CREATE INDEX IF NOT EXISTS idx_annotations_target
ON annotations(target_type, target_id);

CREATE TABLE IF NOT EXISTS corrections (
    correction_id    TEXT PRIMARY KEY,
    target_type      TEXT NOT NULL CHECK(target_type IN ('session', 'message', 'insight')),
    target_id        TEXT NOT NULL,
    correction_type  TEXT NOT NULL CHECK(correction_type IN ('tag_reject', 'tag_accept', 'summary_override')),
    payload_json     TEXT NOT NULL DEFAULT '{}',
    created_at_ms    INTEGER NOT NULL,
    updated_at_ms    INTEGER NOT NULL,
    UNIQUE(target_type, target_id, correction_type)
) STRICT;

CREATE TABLE IF NOT EXISTS suppressions (
    session_id       TEXT PRIMARY KEY,
    reason           TEXT,
    mode             TEXT NOT NULL DEFAULT 'hide' CHECK(mode IN ('hide', 'freeze')),
    created_at_ms    INTEGER NOT NULL,
    updated_at_ms    INTEGER NOT NULL
) STRICT;

CREATE TABLE IF NOT EXISTS session_tags (
    session_id     TEXT NOT NULL,
    tag            TEXT NOT NULL,
    tag_source     TEXT NOT NULL CHECK(tag_source = 'user'),
    method         TEXT,
    confidence     REAL CHECK(confidence IS NULL OR confidence BETWEEN 0 AND 1),
    evidence_json  TEXT,
    PRIMARY KEY(session_id, tag, tag_source)
) STRICT;

CREATE TABLE IF NOT EXISTS session_metadata (
    session_id      TEXT NOT NULL,
    key             TEXT NOT NULL,
    value_json      TEXT NOT NULL,
    created_at_ms   INTEGER NOT NULL,
    updated_at_ms   INTEGER NOT NULL,
    PRIMARY KEY(session_id, key)
) STRICT;

CREATE TABLE IF NOT EXISTS saved_views (
    view_id          TEXT PRIMARY KEY,
    name             TEXT NOT NULL UNIQUE,
    query_json       TEXT NOT NULL,
    created_at_ms    INTEGER NOT NULL,
    updated_at_ms    INTEGER NOT NULL
) STRICT;

CREATE TABLE IF NOT EXISTS recall_packs (
    recall_pack_id   TEXT PRIMARY KEY,
    name             TEXT NOT NULL,
    payload_json     TEXT NOT NULL,
    created_at_ms    INTEGER NOT NULL,
    updated_at_ms    INTEGER NOT NULL
) STRICT;

CREATE TABLE IF NOT EXISTS workspaces (
    workspace_id     TEXT PRIMARY KEY,
    name             TEXT NOT NULL UNIQUE,
    settings_json    TEXT NOT NULL DEFAULT '{}',
    created_at_ms    INTEGER NOT NULL,
    updated_at_ms    INTEGER NOT NULL
) STRICT;

CREATE TABLE IF NOT EXISTS blackboard_notes (
    note_id          TEXT PRIMARY KEY,
    target_type      TEXT CHECK(target_type IN ('session', 'message', 'block', 'attachment', 'paste_span', 'work_event', 'phase', 'thread') OR target_type IS NULL),
    target_id        TEXT,
    body             TEXT NOT NULL,
    created_at_ms    INTEGER NOT NULL,
    updated_at_ms    INTEGER NOT NULL
) STRICT;
"""

__all__ = ["USER_DDL", "USER_SCHEMA_VERSION"]

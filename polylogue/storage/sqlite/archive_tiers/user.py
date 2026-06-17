"""User-tier DDL for archive."""

from __future__ import annotations

USER_SCHEMA_VERSION = 2

USER_DDL = """
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

-- Unified evidence-linked user assertion (#1883). Marks, annotations,
-- corrections, suppressions, saved views, recall packs, workspaces, and
-- blackboard notes are represented here directly. ``kind`` carries the closed
-- v0 vocabulary defined by ``AssertionKind`` in user_write.py.
CREATE TABLE IF NOT EXISTS assertions (
    assertion_id        TEXT PRIMARY KEY,
    scope_ref           TEXT,
    target_ref          TEXT NOT NULL,
    key                 TEXT,
    kind                TEXT NOT NULL,
    value_json          TEXT,
    body_text           TEXT,
    author_ref          TEXT,
    author_kind         TEXT,
    evidence_refs_json  TEXT,
    status              TEXT,
    visibility          TEXT,
    confidence          REAL,
    staleness_json      TEXT,
    context_policy_json TEXT,
    supersedes_json     TEXT,
    created_at_ms       INTEGER NOT NULL,
    updated_at_ms       INTEGER NOT NULL
) STRICT;

CREATE INDEX IF NOT EXISTS idx_assertions_target_kind
ON assertions(target_ref, kind);
"""

__all__ = ["USER_DDL", "USER_SCHEMA_VERSION"]

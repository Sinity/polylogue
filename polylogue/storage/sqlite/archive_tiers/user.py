"""User-tier DDL for archive."""

from __future__ import annotations

USER_SCHEMA_VERSION = 3

USER_DDL = """
-- Unified evidence-linked user assertion. Marks, annotations,
-- corrections, suppressions, tags, metadata, saved views, recall packs,
-- workspaces, and blackboard notes are represented here directly. ``kind``
-- carries the vocabulary defined by ``AssertionKind`` in user_write.py.
CREATE TABLE IF NOT EXISTS assertions (
    assertion_id        TEXT PRIMARY KEY,
    scope_ref           TEXT,
    target_ref          TEXT NOT NULL,
    key                 TEXT,
    kind                TEXT NOT NULL,
    value_json          TEXT,
    body_text           TEXT,
    author_ref          TEXT DEFAULT 'user:local',
    author_kind         TEXT DEFAULT 'user',
    evidence_refs_json  TEXT DEFAULT '[]',
    status              TEXT DEFAULT 'active',
    visibility          TEXT DEFAULT 'private',
    confidence          REAL,
    staleness_json      TEXT,
    context_policy_json TEXT DEFAULT '{"inject":false}',
    supersedes_json     TEXT DEFAULT '[]',
    created_at_ms       INTEGER NOT NULL,
    updated_at_ms       INTEGER NOT NULL
) STRICT;

CREATE INDEX IF NOT EXISTS idx_assertions_target_kind
ON assertions(target_ref, kind);

CREATE INDEX IF NOT EXISTS idx_assertions_kind_status_updated
ON assertions(kind, status, updated_at_ms);

CREATE INDEX IF NOT EXISTS idx_assertions_target_kind_status_visibility
ON assertions(target_ref, kind, status, visibility);
"""

__all__ = ["USER_DDL", "USER_SCHEMA_VERSION"]

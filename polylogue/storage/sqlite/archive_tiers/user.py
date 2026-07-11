"""User-tier DDL for archive."""

from __future__ import annotations

USER_SCHEMA_VERSION = 5

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

-- Durable user/workspace settings. These are intentionally separate from
-- assertions: settings are state, not epistemic claims.
CREATE TABLE IF NOT EXISTS user_settings (
    setting_key    TEXT PRIMARY KEY,
    value_json     TEXT NOT NULL,
    updated_at_ms  INTEGER NOT NULL,
    author_ref     TEXT NOT NULL DEFAULT 'user:local'
) STRICT;

-- Immutable evidence of an exact compiled context image crossing a named
-- delivery boundary. Actor and recipient refs are provenance; authorization
-- remains the responsibility of the authenticated product surface.
CREATE TABLE IF NOT EXISTS context_deliveries (
    snapshot_ref           TEXT PRIMARY KEY,
    recipient_ref          TEXT NOT NULL,
    run_ref                TEXT,
    boundary               TEXT NOT NULL CHECK(length(trim(boundary)) > 0),
    inheritance_mode       TEXT NOT NULL DEFAULT 'explicit',
    context_image_json     TEXT NOT NULL CHECK(json_valid(context_image_json)),
    context_image_sha256   TEXT NOT NULL CHECK(length(context_image_sha256) = 64),
    segment_refs_json      TEXT NOT NULL DEFAULT '[]' CHECK(json_valid(segment_refs_json)),
    evidence_refs_json     TEXT NOT NULL DEFAULT '[]' CHECK(json_valid(evidence_refs_json)),
    assertion_refs_json    TEXT NOT NULL DEFAULT '[]' CHECK(json_valid(assertion_refs_json)),
    omissions_json         TEXT NOT NULL DEFAULT '[]' CHECK(json_valid(omissions_json)),
    caveats_json           TEXT NOT NULL DEFAULT '[]' CHECK(json_valid(caveats_json)),
    metadata_json          TEXT NOT NULL DEFAULT '{}' CHECK(json_valid(metadata_json)),
    delivered_by_ref       TEXT NOT NULL,
    delivered_at_ms        INTEGER NOT NULL CHECK(delivered_at_ms >= 0)
) STRICT;

CREATE INDEX IF NOT EXISTS idx_context_deliveries_recipient_time
ON context_deliveries(recipient_ref, delivered_at_ms DESC);

CREATE INDEX IF NOT EXISTS idx_context_deliveries_run_time
ON context_deliveries(run_ref, delivered_at_ms DESC);
"""

__all__ = ["USER_DDL", "USER_SCHEMA_VERSION"]

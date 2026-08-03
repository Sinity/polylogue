"""Durable authority-journal DDL for ``audit.db``.

The audit tier stores authority and lifecycle metadata only. Domain payloads
remain in their owning tiers and are linked by typed receipt references.
"""

from __future__ import annotations

AUDIT_SCHEMA_VERSION = 1

AUDIT_DDL = """
CREATE TABLE IF NOT EXISTS archive_authority (
    archive_instance_id TEXT PRIMARY KEY,
    created_at_ms       INTEGER NOT NULL CHECK(created_at_ms >= 0),
    authority_format    INTEGER NOT NULL CHECK(authority_format = 1)
) STRICT;

CREATE TABLE IF NOT EXISTS operation_previews (
    preview_id                 TEXT PRIMARY KEY,
    operation_name             TEXT NOT NULL,
    operation_version          INTEGER NOT NULL CHECK(operation_version >= 1),
    archive_instance_id        TEXT NOT NULL,
    archive_identity_digest    TEXT NOT NULL,
    plan_hash                  TEXT NOT NULL,
    parameter_digest           TEXT NOT NULL,
    target_digest              TEXT NOT NULL,
    target_count               INTEGER NOT NULL CHECK(target_count >= 0),
    destructive_class          TEXT NOT NULL CHECK(destructive_class IN (
        'additive', 'reversible', 'maintenance', 'reset', 'delete', 'excise'
    )),
    required_confirmation      TEXT NOT NULL CHECK(required_confirmation IN (
        'role_only', 'confirm_flag', 'bound_token'
    )),
    required_capability_count  INTEGER NOT NULL CHECK(required_capability_count >= 0),
    principal_actor_ref        TEXT NOT NULL,
    principal_surface          TEXT NOT NULL CHECK(principal_surface IN (
        'cli', 'api', 'mcp', 'daemon', 'maintenance', 'internal'
    )),
    role_label                 TEXT,
    state                      TEXT NOT NULL CHECK(state IN (
        'prepared', 'consumed', 'expired', 'stale', 'cancelled'
    )),
    created_at_ms              INTEGER NOT NULL CHECK(created_at_ms >= 0),
    expires_at_ms              INTEGER NOT NULL CHECK(expires_at_ms >= created_at_ms),
    plan_format                TEXT NOT NULL CHECK(plan_format = 'polylogue.mutation-plan/v1'),
    plan_json                  TEXT NOT NULL,
    UNIQUE(operation_name, operation_version, plan_hash, principal_actor_ref, created_at_ms)
) STRICT;

CREATE INDEX IF NOT EXISTS idx_operation_previews_plan_hash
ON operation_previews(plan_hash);
CREATE INDEX IF NOT EXISTS idx_operation_previews_state_expiry
ON operation_previews(state, expires_at_ms);

CREATE TABLE IF NOT EXISTS operation_preview_targets (
    preview_id          TEXT NOT NULL REFERENCES operation_previews(preview_id) ON DELETE CASCADE,
    ordinal             INTEGER NOT NULL CHECK(ordinal >= 0),
    target_kind         TEXT NOT NULL,
    target_ref          TEXT NOT NULL,
    identity_digest     TEXT NOT NULL,
    effect_identity     TEXT NOT NULL,
    durability          TEXT NOT NULL CHECK(durability IN ('durable', 'derived', 'disposable', 'external')),
    recovery_policy     TEXT NOT NULL CHECK(recovery_policy IN (
        'rebuild', 'restore_verified_backup', 'reauthenticate',
        'retry_convergent', 'reconcile_required', 'none'
    )),
    PRIMARY KEY(preview_id, ordinal)
) STRICT;
CREATE INDEX IF NOT EXISTS idx_operation_preview_targets_ref
ON operation_preview_targets(target_ref);

CREATE TABLE IF NOT EXISTS operation_preview_capabilities (
    preview_id  TEXT NOT NULL REFERENCES operation_previews(preview_id) ON DELETE CASCADE,
    capability  TEXT NOT NULL,
    PRIMARY KEY(preview_id, capability)
) STRICT;

CREATE TABLE IF NOT EXISTS operation_authorizations (
    authorization_id       TEXT PRIMARY KEY,
    preview_id             TEXT NOT NULL REFERENCES operation_previews(preview_id),
    actor_ref              TEXT NOT NULL,
    surface                TEXT NOT NULL CHECK(surface IN ('cli', 'api', 'mcp', 'daemon', 'maintenance', 'internal')),
    role_label             TEXT,
    confirmation_strength  TEXT NOT NULL CHECK(confirmation_strength IN ('role_only', 'confirm_flag', 'bound_token')),
    token_sha256           TEXT NOT NULL UNIQUE CHECK(length(token_sha256) = 64),
    state                  TEXT NOT NULL CHECK(state IN ('active', 'consumed', 'expired', 'revoked')),
    issued_at_ms           INTEGER NOT NULL CHECK(issued_at_ms >= 0),
    expires_at_ms          INTEGER NOT NULL CHECK(expires_at_ms >= issued_at_ms),
    consumed_at_ms         INTEGER
) STRICT;
CREATE INDEX IF NOT EXISTS idx_operation_authorizations_preview
ON operation_authorizations(preview_id, state);

CREATE TABLE IF NOT EXISTS operation_authorization_capabilities (
    authorization_id  TEXT NOT NULL REFERENCES operation_authorizations(authorization_id) ON DELETE CASCADE,
    capability        TEXT NOT NULL,
    PRIMARY KEY(authorization_id, capability)
) STRICT;

CREATE TABLE IF NOT EXISTS operation_runs (
    operation_id              TEXT PRIMARY KEY,
    preview_id                TEXT NOT NULL REFERENCES operation_previews(preview_id),
    initial_authorization_id  TEXT NOT NULL REFERENCES operation_authorizations(authorization_id),
    parent_operation_id       TEXT REFERENCES operation_runs(operation_id),
    operation_name            TEXT NOT NULL,
    operation_version         INTEGER NOT NULL CHECK(operation_version >= 1),
    archive_instance_id       TEXT NOT NULL,
    archive_identity_digest   TEXT NOT NULL,
    plan_hash                 TEXT NOT NULL,
    parameter_digest          TEXT NOT NULL,
    target_digest             TEXT NOT NULL,
    target_count              INTEGER NOT NULL CHECK(target_count >= 0),
    actor_ref                 TEXT NOT NULL,
    surface                   TEXT NOT NULL CHECK(surface IN ('cli', 'api', 'mcp', 'daemon', 'maintenance', 'internal')),
    role_label                TEXT,
    idempotency_key_hash      TEXT,
    status                    TEXT NOT NULL CHECK(status IN ('pending', 'running', 'completed', 'failed', 'interrupted')),
    terminal_reason           TEXT,
    affected_count            INTEGER NOT NULL DEFAULT 0 CHECK(affected_count >= 0),
    rejected_count            INTEGER NOT NULL DEFAULT 0 CHECK(rejected_count >= 0),
    failed_count              INTEGER NOT NULL DEFAULT 0 CHECK(failed_count >= 0),
    unknown_count             INTEGER NOT NULL DEFAULT 0 CHECK(unknown_count >= 0),
    effect_identity           TEXT,
    domain_receipt_kind       TEXT,
    domain_receipt_ref        TEXT,
    requested_at_ms           INTEGER NOT NULL CHECK(requested_at_ms >= 0),
    started_at_ms             INTEGER,
    updated_at_ms             INTEGER NOT NULL CHECK(updated_at_ms >= requested_at_ms),
    completed_at_ms           INTEGER,
    cancel_requested_at_ms    INTEGER,
    error_code                TEXT,
    error_class               TEXT,
    error_summary             TEXT,
    unknown_reason            TEXT
) STRICT;
CREATE UNIQUE INDEX IF NOT EXISTS idx_operation_runs_idempotency
ON operation_runs(archive_instance_id, operation_name, operation_version, idempotency_key_hash)
WHERE idempotency_key_hash IS NOT NULL;

CREATE TABLE IF NOT EXISTS operation_run_capabilities (
    operation_id  TEXT NOT NULL REFERENCES operation_runs(operation_id) ON DELETE CASCADE,
    capability    TEXT NOT NULL,
    PRIMARY KEY(operation_id, capability)
) STRICT;

CREATE TABLE IF NOT EXISTS operation_targets (
    operation_id                 TEXT NOT NULL REFERENCES operation_runs(operation_id) ON DELETE CASCADE,
    ordinal                      INTEGER NOT NULL CHECK(ordinal >= 0),
    target_kind                  TEXT NOT NULL,
    target_ref                   TEXT NOT NULL,
    identity_digest              TEXT NOT NULL,
    effect_identity              TEXT NOT NULL,
    state                        TEXT NOT NULL CHECK(state IN (
        'pending', 'running', 'applied', 'already_satisfied', 'rejected',
        'failed', 'unknown', 'acknowledged', 'cancelled'
    )),
    attempt_count                INTEGER NOT NULL DEFAULT 0 CHECK(attempt_count >= 0),
    current_attempt_id          TEXT,
    domain_receipt_kind         TEXT,
    domain_receipt_ref          TEXT,
    pre_archive_identity_digest TEXT,
    post_archive_identity_digest TEXT,
    started_at_ms               INTEGER,
    completed_at_ms             INTEGER,
    error_code                  TEXT,
    error_class                 TEXT,
    error_summary               TEXT,
    unknown_reason              TEXT,
    acknowledged_by             TEXT,
    acknowledged_at_ms          INTEGER,
    acknowledgement_reason     TEXT,
    PRIMARY KEY(operation_id, ordinal)
) STRICT;
CREATE INDEX IF NOT EXISTS idx_operation_targets_ref
ON operation_targets(target_ref);
CREATE INDEX IF NOT EXISTS idx_operation_targets_state
ON operation_targets(operation_id, state, ordinal);

CREATE TABLE IF NOT EXISTS operation_attempts (
    attempt_id                    TEXT PRIMARY KEY,
    operation_id                  TEXT NOT NULL REFERENCES operation_runs(operation_id) ON DELETE CASCADE,
    target_ordinal                INTEGER,
    authorization_id              TEXT NOT NULL REFERENCES operation_authorizations(authorization_id),
    worker_id                     TEXT,
    lease_expires_at_ms           INTEGER,
    prepared_precondition_digest  TEXT,
    state                         TEXT NOT NULL CHECK(state IN (
        'running', 'applied', 'failed', 'unknown', 'reconciled', 'cancelled'
    )),
    started_at_ms                 INTEGER NOT NULL CHECK(started_at_ms >= 0),
    finished_at_ms                INTEGER,
    error_code                    TEXT,
    error_class                   TEXT,
    error_summary                 TEXT,
    unknown_reason                TEXT
) STRICT;
CREATE UNIQUE INDEX IF NOT EXISTS idx_operation_attempts_running_target
ON operation_attempts(operation_id, target_ordinal)
WHERE state = 'running';

CREATE TABLE IF NOT EXISTS operation_events (
    operation_id   TEXT NOT NULL REFERENCES operation_runs(operation_id) ON DELETE CASCADE,
    sequence       INTEGER NOT NULL CHECK(sequence >= 1),
    target_ordinal INTEGER,
    attempt_id     TEXT,
    event_type     TEXT NOT NULL,
    from_state     TEXT,
    to_state       TEXT,
    actor_ref      TEXT,
    occurred_at_ms INTEGER NOT NULL CHECK(occurred_at_ms >= 0),
    detail_format  TEXT NOT NULL DEFAULT 'polylogue.audit-event/v1',
    detail_json    TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY(operation_id, sequence)
) STRICT;
CREATE INDEX IF NOT EXISTS idx_operation_events_type_time
ON operation_events(event_type, occurred_at_ms);
"""

__all__ = ["AUDIT_DDL", "AUDIT_SCHEMA_VERSION"]

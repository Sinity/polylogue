"""User-tier DDL for archive."""

from __future__ import annotations

USER_SCHEMA_VERSION = 8

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

CREATE INDEX IF NOT EXISTS idx_assertions_scope_kind_status
ON assertions(scope_ref, kind, status);

-- Immutable, content-addressed canonical query plans. Names, runs, and
-- result snapshots intentionally live in separate relations.
CREATE TABLE IF NOT EXISTS queries (
    query_hash          TEXT PRIMARY KEY NOT NULL CHECK(length(query_hash) = 64 AND query_hash NOT GLOB '*[^0-9a-f]*'),
    canonical_plan_json TEXT NOT NULL CHECK(json_valid(canonical_plan_json) AND json_type(canonical_plan_json) = 'object'),
    grain               TEXT NOT NULL CHECK(length(trim(grain)) > 0),
    lane                TEXT NOT NULL CHECK(length(trim(lane)) > 0),
    rank_policy         TEXT NOT NULL CHECK(length(trim(rank_policy)) > 0),
    definition_protocol_version TEXT NOT NULL CHECK(length(trim(definition_protocol_version)) > 0),
    created_at_ms       INTEGER NOT NULL CHECK(created_at_ms >= 0)
) STRICT;

CREATE TABLE IF NOT EXISTS query_names (
    name                    TEXT PRIMARY KEY NOT NULL CHECK(length(trim(name)) > 0),
    query_hash              TEXT NOT NULL REFERENCES queries(query_hash) ON UPDATE RESTRICT ON DELETE RESTRICT,
    supersedes_query_hash   TEXT REFERENCES queries(query_hash) ON UPDATE RESTRICT ON DELETE RESTRICT,
    watch                   INTEGER NOT NULL DEFAULT 0 CHECK(watch IN (0, 1)),
    updated_at_ms           INTEGER NOT NULL CHECK(updated_at_ms >= 0)
) STRICT;

CREATE INDEX IF NOT EXISTS idx_query_names_query_hash
ON query_names(query_hash, updated_at_ms DESC);

CREATE INDEX IF NOT EXISTS idx_query_names_watch
ON query_names(watch, updated_at_ms DESC, name);

CREATE TABLE IF NOT EXISTS result_sets (
    result_set_id          TEXT PRIMARY KEY NOT NULL CHECK(length(trim(result_set_id)) > 0),
    query_hash             TEXT NOT NULL REFERENCES queries(query_hash) ON UPDATE RESTRICT ON DELETE RESTRICT,
    grain                  TEXT NOT NULL CHECK(length(trim(grain)) > 0),
    corpus_epoch           TEXT NOT NULL CHECK(length(trim(corpus_epoch)) > 0),
    member_count           INTEGER NOT NULL CHECK(member_count >= 0),
    membership_merkle_root TEXT NOT NULL CHECK(length(membership_merkle_root) = 64 AND membership_merkle_root NOT GLOB '*[^0-9a-f]*'),
    ordered_rank_hash      TEXT NOT NULL CHECK(length(ordered_rank_hash) = 64 AND ordered_rank_hash NOT GLOB '*[^0-9a-f]*'),
    exactness              TEXT NOT NULL CHECK(exactness IN ('exact', 'capped', 'sampled', 'estimate')),
    persistence_class      TEXT NOT NULL CHECK(persistence_class IN ('routine', 'watch', 'pinned', 'finding', 'cohort')),
    created_at_ms          INTEGER NOT NULL CHECK(created_at_ms >= 0)
) STRICT;

CREATE INDEX IF NOT EXISTS idx_result_sets_query_epoch
ON result_sets(query_hash, corpus_epoch, created_at_ms DESC);

CREATE TABLE IF NOT EXISTS result_set_members (
    result_set_id TEXT NOT NULL REFERENCES result_sets(result_set_id) ON DELETE CASCADE,
    rank          INTEGER NOT NULL CHECK(rank >= 0),
    member_ref    TEXT NOT NULL CHECK(length(trim(member_ref)) > 0),
    PRIMARY KEY (result_set_id, rank),
    UNIQUE (result_set_id, member_ref)
) STRICT;

CREATE TABLE IF NOT EXISTS query_edges (
    src_query_hash TEXT NOT NULL REFERENCES queries(query_hash) ON UPDATE RESTRICT ON DELETE RESTRICT,
    dst_query_hash TEXT NOT NULL REFERENCES queries(query_hash) ON UPDATE RESTRICT ON DELETE RESTRICT,
    edge_kind      TEXT NOT NULL CHECK(edge_kind IN ('operand-of', 'refines', 'supersedes', 'derived-from', 'same-as')),
    created_at_ms  INTEGER NOT NULL CHECK(created_at_ms >= 0),
    PRIMARY KEY (src_query_hash, dst_query_hash, edge_kind)
) STRICT;

CREATE INDEX IF NOT EXISTS idx_query_edges_dst_kind
ON query_edges(dst_query_hash, edge_kind, created_at_ms DESC);

-- Durable retained run mappings are intentionally separate from disposable
-- ops telemetry: only explicitly retained relations may back `from query-run:`.
CREATE TABLE IF NOT EXISTS retained_query_runs (
    run_id          TEXT PRIMARY KEY NOT NULL CHECK(run_id GLOB 'qr_*' AND length(run_id) > 3),
    query_hash      TEXT NOT NULL REFERENCES queries(query_hash) ON UPDATE RESTRICT ON DELETE RESTRICT,
    result_set_id   TEXT NOT NULL REFERENCES result_sets(result_set_id) ON UPDATE RESTRICT ON DELETE RESTRICT,
    retained_at_ms  INTEGER NOT NULL CHECK(retained_at_ms >= 0)
) STRICT;

CREATE TABLE IF NOT EXISTS query_evaluation_receipts (
    receipt_id              TEXT PRIMARY KEY NOT NULL CHECK(length(trim(receipt_id)) > 0),
    query_hash              TEXT NOT NULL REFERENCES queries(query_hash) ON UPDATE RESTRICT ON DELETE RESTRICT,
    result_set_id           TEXT REFERENCES result_sets(result_set_id) ON UPDATE RESTRICT ON DELETE RESTRICT,
    source_generation       TEXT NOT NULL CHECK(length(trim(source_generation)) > 0),
    user_generation         TEXT NOT NULL CHECK(length(trim(user_generation)) > 0),
    index_generation        TEXT NOT NULL CHECK(length(trim(index_generation)) > 0),
    runtime_build_ref       TEXT NOT NULL CHECK(length(trim(runtime_build_ref)) > 0),
    model_refs_json         TEXT NOT NULL DEFAULT '[]' CHECK(json_valid(model_refs_json) AND json_type(model_refs_json) = 'array'),
    resolved_bounds_json    TEXT NOT NULL DEFAULT '{}' CHECK(json_valid(resolved_bounds_json) AND json_type(resolved_bounds_json) = 'object'),
    degradation_json        TEXT NOT NULL DEFAULT '{}' CHECK(json_valid(degradation_json) AND json_type(degradation_json) = 'object'),
    created_at_ms           INTEGER NOT NULL CHECK(created_at_ms >= 0)
) STRICT;

CREATE INDEX IF NOT EXISTS idx_query_evaluation_receipts_query_time
ON query_evaluation_receipts(query_hash, created_at_ms DESC);

-- Result manifests are immutable and membership-addressed, so a separate
-- pointer records the last watch evaluation even when membership recurs.
CREATE TABLE IF NOT EXISTS watched_query_baselines (
    query_hash      TEXT PRIMARY KEY NOT NULL REFERENCES queries(query_hash) ON UPDATE RESTRICT ON DELETE RESTRICT,
    result_set_id   TEXT NOT NULL REFERENCES result_sets(result_set_id) ON UPDATE RESTRICT ON DELETE RESTRICT,
    updated_at_ms   INTEGER NOT NULL CHECK(updated_at_ms >= 0)
) STRICT;

-- Immutable versioned annotation construct definitions. Definition JSON is
-- canonical and fingerprinted in Python; the row-level identity cannot be
-- reused for a different construct after registration.
CREATE TABLE IF NOT EXISTS annotation_schemas (
    schema_id          TEXT NOT NULL CHECK(length(trim(schema_id)) > 0),
    schema_version     INTEGER NOT NULL CHECK(schema_version >= 1),
    definition_json    TEXT NOT NULL CHECK(
        json_valid(definition_json)
        AND json_type(definition_json) = 'object'
        AND json_extract(definition_json, '$.schema_id') IS schema_id
        AND json_extract(definition_json, '$.version') IS schema_version
    ),
    definition_sha256  TEXT NOT NULL CHECK(
        length(definition_sha256) = 64
        AND definition_sha256 NOT GLOB '*[^0-9a-f]*'
    ),
    registered_at_ms   INTEGER NOT NULL CHECK(registered_at_ms >= 0),
    PRIMARY KEY (schema_id, schema_version)
) STRICT;

INSERT OR IGNORE INTO annotation_schemas (
    schema_id, schema_version, definition_json, definition_sha256, registered_at_ms
) VALUES (
    'delegation.discourse',
    1,
    '{"abstain_field":"abstain","description":"Evidence-backed discourse role and applicability for one delegation attempt.","evidence_policy":"required","fields":[{"description":"How the work order frames the requested action.","enum_values":["imperative","collaborative","goal_delegation","question","mixed","not_observed"],"maximum":null,"minimum":null,"name":"directive_mode","required":true,"value_type":"enum"},{"description":"How explicitly the work order constrains forbidden actions.","enum_values":["none","implicit","explicit","multiple"],"maximum":null,"minimum":null,"name":"prohibitions","required":true,"value_type":"enum"},{"description":"How much execution discretion the delegate receives.","enum_values":["low","bounded","high","unspecified"],"maximum":null,"minimum":null,"name":"autonomy","required":true,"value_type":"enum"},{"description":"How specifically the expected result shape is declared.","enum_values":["unspecified","informal","structured","machine_readable"],"maximum":null,"minimum":null,"name":"output_contract","required":true,"value_type":"enum"},{"description":"How the work order bounds the implementation surface.","enum_values":["open","bounded","owned_paths","owned_and_avoid_paths"],"maximum":null,"minimum":null,"name":"scope_control","required":true,"value_type":"enum"},{"description":"The strongest explicit verification obligation.","enum_values":["none","self_check","focused_tests","broad_gate"],"maximum":null,"minimum":null,"name":"verification_demand","required":true,"value_type":"enum"},{"description":"Whether checkpoints or escalation conditions are specified.","enum_values":["none","checkpoint","escalation","both"],"maximum":null,"minimum":null,"name":"checkpoint_escalation","required":true,"value_type":"enum"},{"description":"The interpersonal frame expressed by the work order.","enum_values":["directive","collaborative","advisory","evaluative","mixed"],"maximum":null,"minimum":null,"name":"relational_frame","required":true,"value_type":"enum"},{"description":"How much rationale for constraints and choices is exposed.","enum_values":["none","partial","explicit"],"maximum":null,"minimum":null,"name":"rationale_visibility","required":true,"value_type":"enum"},{"description":"Whether the discourse construct applies to this delegation.","enum_values":[],"maximum":null,"minimum":null,"name":"applicable","required":true,"value_type":"boolean"},{"description":"Label confidence on the closed interval from zero to one.","enum_values":[],"maximum":1.0,"minimum":0.0,"name":"confidence","required":true,"value_type":"number"},{"description":"True when available evidence is insufficient to label this delegation.","enum_values":[],"maximum":null,"minimum":null,"name":"abstain","required":false,"value_type":"boolean"},{"description":"Concise evidence-grounded rationale for the label or abstention.","enum_values":[],"maximum":null,"minimum":null,"name":"rationale","required":false,"value_type":"string"}],"format":"polylogue.annotation-schema/v1","schema_id":"delegation.discourse","status":"active","target_ref_kinds":["delegation"],"title":"Delegation discourse","version":1}',
    '7cb761fc365caaf40ca98a96c4d6d809284fa3aa651656f26e7b01e2476f06e9',
    0
);

-- Write-once annotation import provenance. Rows remain assertions and link
-- back through assertions.scope_ref = 'annotation-batch:<batch_id>'.
CREATE TABLE IF NOT EXISTS annotation_batches (
    batch_id                  TEXT PRIMARY KEY NOT NULL CHECK(length(trim(batch_id)) > 0),
    schema_id                 TEXT NOT NULL,
    schema_version            INTEGER NOT NULL CHECK(schema_version >= 1),
    target_ref                TEXT NOT NULL CHECK(length(trim(target_ref)) > 0),
    source_result_ref         TEXT NOT NULL CHECK(length(trim(source_result_ref)) > 0),
    actor_ref                 TEXT NOT NULL CHECK(length(trim(actor_ref)) > 0),
    model_ref                 TEXT NOT NULL CHECK(length(trim(model_ref)) > 0),
    prompt_ref                TEXT NOT NULL CHECK(length(trim(prompt_ref)) > 0),
    total_count               INTEGER NOT NULL CHECK(total_count >= 0),
    valid_count               INTEGER NOT NULL CHECK(valid_count >= 0),
    invalid_count             INTEGER NOT NULL CHECK(invalid_count >= 0),
    abstained_count           INTEGER NOT NULL CHECK(abstained_count >= 0 AND abstained_count <= valid_count),
    assertion_refs_json       TEXT NOT NULL DEFAULT '[]' CHECK(
        json_valid(assertion_refs_json)
        AND json_type(assertion_refs_json) = 'array'
        AND json_array_length(assertion_refs_json) = valid_count
    ),
    validation_failures_json  TEXT NOT NULL DEFAULT '[]' CHECK(
        json_valid(validation_failures_json)
        AND json_type(validation_failures_json) = 'array'
        AND json_array_length(validation_failures_json) = invalid_count
    ),
    metadata_json             TEXT NOT NULL DEFAULT '{}' CHECK(
        json_valid(metadata_json)
        AND json_type(metadata_json) = 'object'
    ),
    created_at_ms             INTEGER NOT NULL CHECK(created_at_ms >= 0),
    CHECK(valid_count + invalid_count = total_count),
    FOREIGN KEY (schema_id, schema_version)
        REFERENCES annotation_schemas(schema_id, schema_version)
        ON UPDATE RESTRICT ON DELETE RESTRICT
) STRICT;

CREATE INDEX IF NOT EXISTS idx_annotation_batches_schema_target_time
ON annotation_batches(schema_id, schema_version, target_ref, created_at_ms DESC, batch_id);

CREATE INDEX IF NOT EXISTS idx_annotation_batches_source_result_time
ON annotation_batches(source_result_ref, created_at_ms DESC, batch_id);


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

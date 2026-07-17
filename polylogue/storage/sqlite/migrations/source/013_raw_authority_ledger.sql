-- Durable, restart-safe conservation ledger for raw authority reconciliation.
CREATE TABLE raw_authority_parser_census (
    raw_id                  TEXT PRIMARY KEY REFERENCES raw_sessions(raw_id) ON DELETE CASCADE,
    parser_fingerprint      TEXT NOT NULL,
    status                  TEXT NOT NULL CHECK(status IN ('complete', 'failed')),
    logical_keys_json       TEXT NOT NULL CHECK(json_valid(logical_keys_json)),
    detail                  TEXT NOT NULL DEFAULT '',
    censused_at_ms          INTEGER NOT NULL CHECK(censused_at_ms >= 0)
) STRICT;

CREATE TABLE raw_authority_censuses (
    census_id               TEXT PRIMARY KEY,
    sequence_no             INTEGER NOT NULL UNIQUE CHECK(sequence_no > 0),
    scope_json              TEXT NOT NULL CHECK(json_valid(scope_json)),
    residual_json           TEXT NOT NULL CHECK(json_valid(residual_json)),
    parser_fingerprint      TEXT NOT NULL,
    mode                    TEXT NOT NULL CHECK(mode IN ('census', 'dry_run', 'apply')),
    lifecycle_status        TEXT NOT NULL CHECK(lifecycle_status IN ('planned', 'completed', 'interrupted')),
    quiescent               INTEGER NOT NULL CHECK(quiescent IN (0, 1)),
    inventory_digest        TEXT NOT NULL CHECK(length(inventory_digest) = 64),
    residual_digest         TEXT NOT NULL CHECK(length(residual_digest) = 64),
    plan_count              INTEGER NOT NULL CHECK(plan_count >= 0),
    post_inventory_digest   TEXT CHECK(post_inventory_digest IS NULL OR length(post_inventory_digest) = 64),
    post_residual_json      TEXT CHECK(post_residual_json IS NULL OR json_valid(post_residual_json)),
    post_residual_digest    TEXT CHECK(post_residual_digest IS NULL OR length(post_residual_digest) = 64),
    post_plan_count         INTEGER CHECK(post_plan_count IS NULL OR post_plan_count >= 0),
    postflight_at_ms        INTEGER CHECK(postflight_at_ms IS NULL OR postflight_at_ms >= created_at_ms),
    executable_plan_count   INTEGER NOT NULL CHECK(executable_plan_count >= 0),
    residual_plan_count     INTEGER NOT NULL CHECK(residual_plan_count >= 0),
    predecessor_census_id   TEXT REFERENCES raw_authority_censuses(census_id),
    fixed_point             INTEGER NOT NULL DEFAULT 0 CHECK(fixed_point IN (0, 1)),
    created_at_ms           INTEGER NOT NULL CHECK(created_at_ms >= 0),
    completed_at_ms         INTEGER CHECK(completed_at_ms IS NULL OR completed_at_ms >= created_at_ms),
    CHECK(plan_count >= executable_plan_count),
    CHECK(plan_count >= residual_plan_count),
    CHECK(plan_count = executable_plan_count + residual_plan_count),
    CHECK(
        (lifecycle_status = 'planned' AND completed_at_ms IS NULL)
        OR (lifecycle_status IN ('completed', 'interrupted') AND completed_at_ms IS NOT NULL)
    ),
    CHECK(
        (lifecycle_status = 'planned' AND post_inventory_digest IS NULL
            AND post_residual_json IS NULL AND post_residual_digest IS NULL
            AND post_plan_count IS NULL AND postflight_at_ms IS NULL)
        OR (lifecycle_status IN ('completed', 'interrupted')
            AND post_inventory_digest IS NOT NULL AND post_residual_json IS NOT NULL
            AND post_residual_digest IS NOT NULL AND post_plan_count IS NOT NULL
            AND postflight_at_ms IS NOT NULL)
    )
) STRICT;

CREATE TABLE raw_authority_plans (
    plan_id                  TEXT PRIMARY KEY,
    input_digest             TEXT NOT NULL CHECK(length(input_digest) = 64),
    input_raw_ids_json       TEXT NOT NULL CHECK(json_valid(input_raw_ids_json)),
    logical_keys_json        TEXT NOT NULL CHECK(json_valid(logical_keys_json)),
    authority_witness_json   TEXT NOT NULL CHECK(json_valid(authority_witness_json)),
    source_preconditions_json TEXT NOT NULL CHECK(json_valid(source_preconditions_json)),
    index_preconditions_json TEXT NOT NULL CHECK(json_valid(index_preconditions_json)),
    created_at_ms            INTEGER NOT NULL CHECK(created_at_ms >= 0)
) STRICT;

CREATE TABLE raw_authority_census_plans (
    census_id          TEXT NOT NULL REFERENCES raw_authority_censuses(census_id) ON DELETE CASCADE,
    plan_id            TEXT NOT NULL REFERENCES raw_authority_plans(plan_id),
    ordinal            INTEGER NOT NULL CHECK(ordinal >= 0),
    selected           INTEGER NOT NULL CHECK(selected IN (0, 1)),
    outcome_status     TEXT NOT NULL CHECK(outcome_status IN (
                           'executed', 'retryable', 'deferred', 'terminal',
                           'rejected_stale', 'carried_forward'
                       )),
    reason             TEXT NOT NULL,
    next_action        TEXT NOT NULL,
    application_receipt_json TEXT NOT NULL DEFAULT '{}' CHECK(json_valid(application_receipt_json)),
    outcome_recorded   INTEGER NOT NULL DEFAULT 0 CHECK(outcome_recorded IN (0, 1)),
    recorded_at_ms     INTEGER NOT NULL CHECK(recorded_at_ms >= 0),
    PRIMARY KEY(census_id, plan_id),
    UNIQUE(census_id, ordinal)
) STRICT;

CREATE INDEX idx_raw_authority_census_plans_status
ON raw_authority_census_plans(census_id, outcome_status, ordinal);

CREATE INDEX idx_raw_authority_census_plans_attempts
ON raw_authority_census_plans(plan_id, recorded_at_ms DESC)
WHERE selected = 1;

CREATE TABLE raw_authority_census_post_plans (
    census_id          TEXT NOT NULL REFERENCES raw_authority_censuses(census_id) ON DELETE CASCADE,
    plan_id            TEXT NOT NULL REFERENCES raw_authority_plans(plan_id),
    ordinal            INTEGER NOT NULL CHECK(ordinal >= 0),
    PRIMARY KEY(census_id, plan_id),
    UNIQUE(census_id, ordinal)
) STRICT;

CREATE TABLE raw_authority_blockers (
    blocker_id          TEXT PRIMARY KEY,
    plan_id             TEXT NOT NULL REFERENCES raw_authority_plans(plan_id),
    census_id           TEXT NOT NULL REFERENCES raw_authority_censuses(census_id),
    reason              TEXT NOT NULL,
    expected_json       TEXT NOT NULL CHECK(json_valid(expected_json)),
    observed_json       TEXT NOT NULL CHECK(json_valid(observed_json)),
    created_at_ms       INTEGER NOT NULL CHECK(created_at_ms >= 0),
    resolved_at_ms      INTEGER CHECK(resolved_at_ms IS NULL OR resolved_at_ms >= created_at_ms),
    resolution          TEXT,
    CHECK((resolved_at_ms IS NULL) = (resolution IS NULL))
) STRICT;

CREATE UNIQUE INDEX idx_raw_authority_blockers_open_plan
ON raw_authority_blockers(plan_id)
WHERE resolved_at_ms IS NULL;

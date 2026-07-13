-- Versioned planner definitions and durable receipts make a query execution
-- reproducible without treating canonical identity JSON as an executable plan.
ALTER TABLE queries ADD COLUMN definition_protocol_version TEXT NOT NULL
    DEFAULT 'polylogue.query-definition.v0'
    CHECK(length(trim(definition_protocol_version)) > 0);

ALTER TABLE query_names ADD COLUMN watch INTEGER NOT NULL DEFAULT 0
    CHECK(watch IN (0, 1));
CREATE INDEX IF NOT EXISTS idx_query_names_watch
ON query_names(watch, updated_at_ms DESC, name);

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

-- Holdout is an access POLICY on an existing result-set manifest, not a
-- second cohort/result relation type (rxdo.9.4). Additive: no existing
-- table or column is touched.
CREATE TABLE IF NOT EXISTS result_set_holdout_policies (
    result_set_id              TEXT PRIMARY KEY NOT NULL REFERENCES result_sets(result_set_id) ON UPDATE RESTRICT ON DELETE RESTRICT,
    frame                      TEXT NOT NULL CHECK(length(trim(frame)) > 0),
    selection_definition_json  TEXT NOT NULL CHECK(json_valid(selection_definition_json) AND json_type(selection_definition_json) = 'object'),
    intended_confirmation_use  TEXT NOT NULL CHECK(length(trim(intended_confirmation_use)) > 0),
    authority                  TEXT NOT NULL CHECK(length(trim(authority)) > 0),
    created_epoch               TEXT NOT NULL CHECK(length(trim(created_epoch)) > 0),
    created_at_ms               INTEGER NOT NULL CHECK(created_at_ms >= 0)
) STRICT;

CREATE TABLE IF NOT EXISTS holdout_access_receipts (
    receipt_id             TEXT PRIMARY KEY NOT NULL CHECK(length(trim(receipt_id)) > 0),
    result_set_id           TEXT NOT NULL REFERENCES result_set_holdout_policies(result_set_id) ON UPDATE RESTRICT ON DELETE RESTRICT,
    accessor_ref             TEXT NOT NULL CHECK(length(trim(accessor_ref)) > 0),
    declared_confirmation    INTEGER NOT NULL CHECK(declared_confirmation IN (0, 1)),
    contamination            INTEGER NOT NULL CHECK(contamination IN (0, 1)),
    reason                   TEXT,
    accessed_at_ms            INTEGER NOT NULL CHECK(accessed_at_ms >= 0)
) STRICT;
CREATE INDEX IF NOT EXISTS idx_holdout_access_receipts_result_set
ON holdout_access_receipts(result_set_id, accessed_at_ms DESC);

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

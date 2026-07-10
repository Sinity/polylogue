ALTER TABLE raw_sessions ADD COLUMN logical_source_key TEXT;
ALTER TABLE raw_sessions ADD COLUMN revision_kind TEXT NOT NULL DEFAULT 'unknown'
    CHECK(revision_kind IN ('full', 'append', 'unknown'));
ALTER TABLE raw_sessions ADD COLUMN source_revision TEXT;
ALTER TABLE raw_sessions ADD COLUMN predecessor_raw_id TEXT;
ALTER TABLE raw_sessions ADD COLUMN baseline_raw_id TEXT;
ALTER TABLE raw_sessions ADD COLUMN append_start_offset INTEGER CHECK(append_start_offset >= 0);
ALTER TABLE raw_sessions ADD COLUMN append_end_offset INTEGER CHECK(append_end_offset > append_start_offset);
ALTER TABLE raw_sessions ADD COLUMN acquisition_generation INTEGER CHECK(acquisition_generation >= 0);
ALTER TABLE raw_sessions ADD COLUMN revision_authority TEXT NOT NULL DEFAULT 'quarantined'
    CHECK(revision_authority IN ('asserted', 'byte_proven', 'quarantined'));

CREATE INDEX idx_raw_sessions_logical_revision
ON raw_sessions(logical_source_key, acquisition_generation, raw_id)
WHERE logical_source_key IS NOT NULL;

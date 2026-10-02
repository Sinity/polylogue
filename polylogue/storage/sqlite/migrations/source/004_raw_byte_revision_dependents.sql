-- migration-safety: additive-no-backup
CREATE INDEX idx_raw_sessions_predecessor_raw_id ON raw_sessions(predecessor_raw_id);
CREATE INDEX idx_raw_sessions_baseline_raw_id ON raw_sessions(baseline_raw_id);

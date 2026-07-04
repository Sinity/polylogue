DROP INDEX IF EXISTS idx_raw_sessions_origin_native;

CREATE INDEX IF NOT EXISTS idx_raw_sessions_origin_native
ON raw_sessions(origin, native_id)
WHERE native_id IS NOT NULL;

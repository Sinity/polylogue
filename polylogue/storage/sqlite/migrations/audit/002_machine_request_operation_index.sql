-- migration-safety: additive-with-backup
-- Keep operation-level machine-request lookups bounded by the durable
-- acceptance timeline.  The same index is present in fresh audit DDL.
CREATE INDEX IF NOT EXISTS idx_machine_requests_operation
ON machine_requests(operation_name, accepted_at_ms);

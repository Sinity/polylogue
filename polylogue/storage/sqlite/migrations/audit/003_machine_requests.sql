CREATE TABLE IF NOT EXISTS machine_requests (
    archive_identity TEXT NOT NULL,
    request_id TEXT NOT NULL,
    principal_ref TEXT NOT NULL,
    fingerprint TEXT NOT NULL CHECK(length(fingerprint) = 64),
    operation_name TEXT NOT NULL,
    artifact_kind TEXT NOT NULL,
    artifact_ref TEXT NOT NULL,
    accepted_at_ms INTEGER NOT NULL CHECK(accepted_at_ms >= 0),
    PRIMARY KEY(archive_identity, request_id)
) STRICT;
CREATE INDEX IF NOT EXISTS idx_machine_requests_artifact
ON machine_requests(artifact_kind, artifact_ref);

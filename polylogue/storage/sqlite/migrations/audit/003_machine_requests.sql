CREATE TABLE IF NOT EXISTS machine_requests (
    archive_identity TEXT NOT NULL,
    request_id TEXT NOT NULL,
    principal_ref TEXT NOT NULL,
    fingerprint TEXT NOT NULL CHECK(length(fingerprint) = 64),
    operation_name TEXT NOT NULL,
    artifact_kind TEXT NOT NULL,
    artifact_ref TEXT NOT NULL,
    accepted_at_ms INTEGER NOT NULL CHECK(accepted_at_ms >= 0),
    part_count INTEGER NOT NULL DEFAULT 1 CHECK(part_count >= 1),
    stop_reason TEXT,
    stopped_at_ms INTEGER,
    accepted_deadline_unix_ms INTEGER,
    PRIMARY KEY(archive_identity, request_id)
) STRICT;
CREATE INDEX IF NOT EXISTS idx_machine_requests_artifact
ON machine_requests(artifact_kind, artifact_ref);
CREATE TABLE IF NOT EXISTS machine_request_parts (
    archive_identity TEXT NOT NULL,
    request_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL CHECK(ordinal >= 0),
    artifact_ref TEXT NOT NULL,
    preview_ref TEXT NOT NULL REFERENCES operation_previews(preview_id),
    authorization_ref TEXT REFERENCES operation_authorizations(authorization_id),
    operation_id TEXT REFERENCES operation_runs(operation_id),
    PRIMARY KEY(archive_identity, request_id, ordinal),
    FOREIGN KEY(archive_identity, request_id) REFERENCES machine_requests(archive_identity, request_id)
) STRICT;
CREATE UNIQUE INDEX IF NOT EXISTS idx_machine_request_parts_authorization
ON machine_request_parts(authorization_ref) WHERE authorization_ref IS NOT NULL;

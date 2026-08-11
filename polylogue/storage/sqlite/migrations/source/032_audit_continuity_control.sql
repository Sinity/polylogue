CREATE TABLE audit_continuity_control (
    singleton            INTEGER PRIMARY KEY CHECK(singleton = 1),
    committed_generation INTEGER NOT NULL CHECK(committed_generation >= 0),
    committed_head_sha256 TEXT NOT NULL CHECK(length(committed_head_sha256) = 64),
    pending_mutation_id  TEXT UNIQUE,
    pending_payload_json TEXT,
    pending_payload_sha256 TEXT CHECK(pending_payload_sha256 IS NULL OR length(pending_payload_sha256) = 64),
    prepared_at_ms       INTEGER,
    CHECK(
        (pending_mutation_id IS NULL AND pending_payload_json IS NULL AND pending_payload_sha256 IS NULL AND prepared_at_ms IS NULL)
        OR
        (pending_mutation_id IS NOT NULL AND pending_payload_json IS NOT NULL AND pending_payload_sha256 IS NOT NULL AND prepared_at_ms IS NOT NULL AND prepared_at_ms >= 0)
    )
) STRICT;
INSERT INTO audit_continuity_control(
    singleton, committed_generation, committed_head_sha256,
    pending_mutation_id, pending_payload_json, pending_payload_sha256, prepared_at_ms
) VALUES (1, 0, '3230fdd585a4fd2d71b7d720bcfe5d697ff120fdb32aecde394e89d407c7198f', NULL, NULL, NULL, NULL);

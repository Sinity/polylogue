CREATE TABLE audit_continuity_head (
    singleton   INTEGER PRIMARY KEY CHECK(singleton = 1),
    generation  INTEGER NOT NULL CHECK(generation >= 0),
    head_sha256 TEXT NOT NULL CHECK(length(head_sha256) = 64),
    mutation_id TEXT,
    advanced_at_ms INTEGER NOT NULL CHECK(advanced_at_ms >= 0)
) STRICT;
INSERT INTO audit_continuity_head(
    singleton, generation, head_sha256, mutation_id, advanced_at_ms
) VALUES (1, 0, '3230fdd585a4fd2d71b7d720bcfe5d697ff120fdb32aecde394e89d407c7198f', NULL, 0);

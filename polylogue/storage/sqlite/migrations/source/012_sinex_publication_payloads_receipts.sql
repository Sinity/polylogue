-- migration-safety: additive-no-backup
-- Restart-safe exact Sinex publication payloads, receipts, and retry schedule
-- (polylogue-303r.2.1).
--
-- 011 is the durable excised-content migration present in the complete
-- repository but outside this job's selected source footprint.  This change
-- therefore advances source.db from 11 to 12; it deliberately does not edit
-- migration 010 and does not create an index.db migration chain.
ALTER TABLE sinex_publication_obligations
ADD COLUMN next_attempt_at_ms INTEGER;

UPDATE sinex_publication_obligations
SET next_attempt_at_ms = COALESCE(last_attempt_at_ms, created_at_ms)
WHERE status IN ('pending', 'publishing', 'durable_debt');

CREATE TABLE sinex_publication_payloads (
    object_id            TEXT NOT NULL,
    protocol_version     TEXT NOT NULL,
    revision_id          TEXT NOT NULL,
    manifest_digest      TEXT NOT NULL,
    manifest_bytes       BLOB NOT NULL,
    manifest_sha256      TEXT NOT NULL CHECK(length(manifest_sha256) = 64),
    manifest_size_bytes  INTEGER NOT NULL CHECK(manifest_size_bytes >= 0),
    segment_count        INTEGER NOT NULL CHECK(segment_count >= 0),
    total_size_bytes     INTEGER NOT NULL CHECK(total_size_bytes >= manifest_size_bytes),
    staged_at_ms         INTEGER NOT NULL CHECK(staged_at_ms >= 0),
    PRIMARY KEY(object_id, protocol_version, revision_id, manifest_digest),
    FOREIGN KEY(object_id, protocol_version, revision_id, manifest_digest)
        REFERENCES sinex_publication_obligations(
            object_id, protocol_version, revision_id, manifest_digest
        ) ON DELETE CASCADE
) STRICT;

CREATE TABLE sinex_publication_segments (
    object_id          TEXT NOT NULL,
    protocol_version   TEXT NOT NULL,
    revision_id        TEXT NOT NULL,
    manifest_digest    TEXT NOT NULL,
    position           INTEGER NOT NULL CHECK(position >= 0),
    segment_name       TEXT NOT NULL CHECK(segment_name != ''),
    segment_bytes      BLOB NOT NULL,
    segment_sha256     TEXT NOT NULL CHECK(length(segment_sha256) = 64),
    size_bytes         INTEGER NOT NULL CHECK(size_bytes >= 0),
    PRIMARY KEY(object_id, protocol_version, revision_id, manifest_digest, position),
    UNIQUE(object_id, protocol_version, revision_id, manifest_digest, segment_name),
    FOREIGN KEY(object_id, protocol_version, revision_id, manifest_digest)
        REFERENCES sinex_publication_payloads(
            object_id, protocol_version, revision_id, manifest_digest
        ) ON DELETE CASCADE
) STRICT;

CREATE TABLE sinex_publication_receipts (
    object_id          TEXT NOT NULL,
    protocol_version   TEXT NOT NULL,
    revision_id        TEXT NOT NULL,
    manifest_digest    TEXT NOT NULL,
    attempt_number     INTEGER NOT NULL CHECK(attempt_number > 0),
    request_id         TEXT NOT NULL,
    receipt_state      TEXT CHECK(receipt_state IS NULL OR receipt_state IN (
                            'raw_accepted', 'persisted_confirmed', 'durable_debt',
                            'spool_accepted_lossless', 'rejected'
                        )),
    receipt_detail     TEXT NOT NULL DEFAULT '',
    error_code         TEXT,
    received_at_ms     INTEGER NOT NULL CHECK(received_at_ms >= 0),
    PRIMARY KEY(object_id, protocol_version, revision_id, manifest_digest, attempt_number),
    FOREIGN KEY(object_id, protocol_version, revision_id, manifest_digest)
        REFERENCES sinex_publication_obligations(
            object_id, protocol_version, revision_id, manifest_digest
        ) ON DELETE CASCADE
) STRICT;

DROP INDEX idx_sinex_publication_obligations_pending;

CREATE INDEX idx_sinex_publication_obligations_pending
ON sinex_publication_obligations(status, next_attempt_at_ms, created_at_ms)
WHERE status IN ('pending', 'publishing', 'durable_debt');

CREATE INDEX idx_sinex_publication_receipts_recent
ON sinex_publication_receipts(received_at_ms DESC);

-- migration-safety: additive-no-backup
-- Durable Sinex-backed-mode publication obligation ledger (polylogue-303r.2).
--
-- One row per (object_id, protocol_version, revision_id, manifest_digest)
-- records that a normalized-session material revision must be published to
-- Sinex before the local index projection may treat that revision as the
-- backed-mode authoritative copy. The row is created in the SAME durable
-- source-tier transaction that records the acquired/normalized revision
-- (design: polylogue-303r.2). source.db is durable, so this ledger survives
-- process crashes and is never the disposable ops.db.
--
-- The primary key IS the idempotency key: retrying the same revision is a
-- no-op INSERT OR IGNORE at the call site, never a duplicate obligation.
CREATE TABLE sinex_publication_obligations (
    object_id           TEXT NOT NULL,
    protocol_version     TEXT NOT NULL CHECK(protocol_version != ''),
    revision_id          TEXT NOT NULL,
    manifest_digest      TEXT NOT NULL,
    mode                 TEXT NOT NULL CHECK(mode IN ('mirror', 'primary')),
    status               TEXT NOT NULL DEFAULT 'pending'
                             CHECK(status IN ('pending', 'publishing', 'confirmed', 'durable_debt', 'rejected')),
    attempt_count        INTEGER NOT NULL DEFAULT 0 CHECK(attempt_count >= 0),
    last_attempt_at_ms   INTEGER,
    last_receipt_state   TEXT,
    last_error           TEXT,
    created_at_ms        INTEGER NOT NULL,
    updated_at_ms        INTEGER NOT NULL,
    retired_at_ms        INTEGER,
    PRIMARY KEY(object_id, protocol_version, revision_id, manifest_digest)
) STRICT;

-- Draining pending/retryable obligations is the hot read: exclude terminal
-- states (confirmed, rejected) so a growing archive does not force the
-- drain loop to scan settled history.
CREATE INDEX idx_sinex_publication_obligations_pending
ON sinex_publication_obligations(status, created_at_ms)
WHERE status IN ('pending', 'publishing', 'durable_debt');

CREATE INDEX idx_sinex_publication_obligations_object
ON sinex_publication_obligations(object_id, created_at_ms DESC);

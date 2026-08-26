-- migration-safety: additive-backup-required
-- Hook events historically have one representative source_path. Preserve it
-- as first-observed compatibility evidence and add the recurring physical
-- carrier relation without rewriting existing rows.
CREATE TABLE hook_event_carriers (
    source_id       TEXT NOT NULL,
    relative_path   TEXT NOT NULL,
    hook_event_id   TEXT NOT NULL REFERENCES raw_hook_events(hook_event_id),
    blob_hash       BLOB NOT NULL CHECK(length(blob_hash) = 32),
    payload_digest  BLOB NOT NULL CHECK(length(payload_digest) = 32),
    carrier_role    TEXT NOT NULL CHECK(carrier_role IN ('primary-writable', 'legacy-read-only')),
    admitted_at_ms  INTEGER NOT NULL,
    PRIMARY KEY (source_id, relative_path)
) STRICT;

CREATE INDEX idx_hook_event_carriers_event
ON hook_event_carriers(hook_event_id, blob_hash);

-- Existing hook rows predate physical carrier identity. They remain valid
-- logical evidence; their first-observed representative is deliberately not
-- guessed into this table. New admission supplies complete carriers.

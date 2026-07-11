CREATE TABLE raw_session_memberships (
    raw_id                  TEXT NOT NULL REFERENCES raw_sessions(raw_id) ON DELETE CASCADE,
    logical_source_key      TEXT NOT NULL,
    provider_session_id     TEXT NOT NULL,
    source_revision         TEXT NOT NULL,
    normalized_content_hash BLOB NOT NULL CHECK(length(normalized_content_hash) = 32),
    message_count           INTEGER NOT NULL CHECK(message_count >= 0),
    predecessor_raw_id      TEXT,
    acquisition_generation  INTEGER NOT NULL DEFAULT 0 CHECK(acquisition_generation >= 0),
    revision_authority      TEXT NOT NULL DEFAULT 'quarantined'
        CHECK(revision_authority IN ('byte_proven', 'quarantined')),
    terminal_outcome        TEXT CHECK(terminal_outcome IN ('applied', 'superseded', 'ambiguous')),
    terminal_at_ms          INTEGER CHECK(terminal_at_ms >= 0),
    PRIMARY KEY(raw_id, logical_source_key),
    CHECK((terminal_outcome IS NULL) = (terminal_at_ms IS NULL))
) STRICT;

CREATE INDEX idx_raw_session_memberships_logical
ON raw_session_memberships(logical_source_key, acquisition_generation, raw_id);

CREATE INDEX idx_raw_session_memberships_pending
ON raw_session_memberships(raw_id)
WHERE terminal_outcome IS NULL;

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
    decision                TEXT CHECK(decision IN (
                                'applied', 'superseded_equivalent', 'superseded_prefix',
                                'ambiguous', 'deferred'
                            )),
    decided_at_ms           INTEGER CHECK(decided_at_ms >= 0),
    PRIMARY KEY(raw_id, logical_source_key),
    CHECK((decision IS NULL) = (decided_at_ms IS NULL))
) STRICT;

CREATE INDEX idx_raw_session_memberships_logical
ON raw_session_memberships(logical_source_key, acquisition_generation, raw_id);

CREATE INDEX idx_raw_session_memberships_pending
ON raw_session_memberships(raw_id)
WHERE decision IS NULL OR decision IN ('ambiguous', 'deferred');

CREATE TABLE raw_membership_census (
    raw_id             TEXT PRIMARY KEY REFERENCES raw_sessions(raw_id) ON DELETE CASCADE,
    parser_fingerprint TEXT NOT NULL,
    status             TEXT NOT NULL CHECK(status IN ('complete', 'failed', 'non_session')),
    member_count       INTEGER NOT NULL CHECK(member_count >= 0),
    censused_at_ms     INTEGER NOT NULL CHECK(censused_at_ms >= 0),
    detail             TEXT NOT NULL DEFAULT ''
) STRICT;

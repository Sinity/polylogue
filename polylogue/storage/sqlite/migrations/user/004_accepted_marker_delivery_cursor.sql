-- Add the source-stream position owned by the durable marker user sink.
--
-- Accepted marker inputs are immutable source history.  The user tier records
-- only the last source sequence that committed with canonical assertion
-- lowering, so a restart resumes from that durable boundary rather than
-- reconstructing candidates from the current index projection.
--
-- user.db is durable and cannot be rebuilt from source evidence.  Fresh DDL
-- alone would leave existing user tiers without this relation, so this
-- additive migration is required.  It neither rewrites assertions nor
-- changes the legacy per-session delivery rows.

CREATE TABLE IF NOT EXISTS accepted_marker_delivery_cursor (
    singleton         INTEGER PRIMARY KEY CHECK(singleton = 1),
    stream_id         TEXT NOT NULL CHECK(stream_id != ''),
    applied_sequence  INTEGER NOT NULL CHECK(applied_sequence >= 0),
    applied_at_ms     INTEGER NOT NULL CHECK(applied_at_ms >= 0)
) STRICT;

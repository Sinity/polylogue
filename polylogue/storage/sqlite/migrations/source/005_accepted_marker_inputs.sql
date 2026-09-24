-- migration-safety: additive
CREATE TABLE IF NOT EXISTS pending_accepted_marker_inputs (
    request_key TEXT PRIMARY KEY CHECK(length(request_key) = 64),
    raw_id TEXT NOT NULL CHECK(raw_id != ''),
    carrier_digest TEXT NOT NULL CHECK(length(carrier_digest) = 64),
    expected_incarnation_id TEXT NOT NULL CHECK(length(expected_incarnation_id) = 36),
    payload BLOB NOT NULL
) STRICT;
CREATE TABLE IF NOT EXISTS accepted_marker_stream (
    singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
    stream_id TEXT NOT NULL UNIQUE CHECK(stream_id != '')
) STRICT;
CREATE TABLE IF NOT EXISTS accepted_marker_inputs (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    identity TEXT NOT NULL UNIQUE CHECK(length(identity) = 64),
    raw_id TEXT NOT NULL CHECK(raw_id != ''),
    payload BLOB NOT NULL,
    index_incarnation_id TEXT CHECK(index_incarnation_id IS NULL OR length(index_incarnation_id) = 36),
    payload_sha256 TEXT NOT NULL CHECK(length(payload_sha256) = 64)
) STRICT;
CREATE TRIGGER IF NOT EXISTS accepted_marker_stream_no_update BEFORE UPDATE ON accepted_marker_stream
BEGIN SELECT RAISE(ABORT, 'accepted marker stream is immutable'); END;
CREATE TRIGGER IF NOT EXISTS accepted_marker_stream_no_delete BEFORE DELETE ON accepted_marker_stream
BEGIN SELECT RAISE(ABORT, 'accepted marker stream is immutable'); END;
CREATE TRIGGER IF NOT EXISTS accepted_marker_inputs_no_update BEFORE UPDATE ON accepted_marker_inputs
BEGIN SELECT RAISE(ABORT, 'accepted marker inputs are immutable'); END;
CREATE TRIGGER IF NOT EXISTS accepted_marker_inputs_no_delete BEFORE DELETE ON accepted_marker_inputs
BEGIN SELECT RAISE(ABORT, 'accepted marker inputs are immutable'); END;

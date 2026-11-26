-- Duplicate derivation also clears old marks after capture inputs disappear.
-- Unmarked sessions have no entries and need no message-table scan.
CREATE INDEX IF NOT EXISTS message_duplicate ON message(session_pk) WHERE duplicate_of IS NOT NULL;

-- A default page reads its unit's rows in the order of its first terms
-- (`time IS NULL, time, ...`), so these indexes carry the same
-- expressions: the page stops after its rows instead of sorting the table.
CREATE INDEX IF NOT EXISTS session_order ON session(last_activity_at IS NULL, last_activity_at DESC, id DESC);
CREATE INDEX IF NOT EXISTS message_order ON message(at IS NULL, at, session_order, id);
-- Explicit newest-first sorts break ties by ID in ascending order.
CREATE INDEX IF NOT EXISTS session_recent ON session(last_activity_at IS NULL, last_activity_at DESC, id);
CREATE INDEX IF NOT EXISTS message_recent ON message(at IS NULL, at DESC, id);
CREATE INDEX IF NOT EXISTS tool_call_order ON tool_call(at IS NULL, at, call_block_pk);
CREATE INDEX IF NOT EXISTS event_order ON event(at IS NULL, at, id);
CREATE INDEX IF NOT EXISTS message_at ON message(at, id);
CREATE INDEX IF NOT EXISTS message_session_at ON message(session_pk, at, id);
CREATE INDEX IF NOT EXISTS message_model_at ON message(model, at, id);
CREATE INDEX IF NOT EXISTS message_session_model_at ON message(session_pk, model, at, id);
-- The messages whose time is a period or of unknown precision, which are few
-- (a source that gives a date or a minute). A time condition reads the stored
-- time only of these and of their sessions; every other row is an instant,
-- decided by its millisecond column (query/fields.rs, MESSAGE_PERIODS).
CREATE INDEX IF NOT EXISTS message_period ON message(session_pk) WHERE at IS NOT NULL AND json_extract(at_time,'$[1].precision') IS NOT 'instant';
CREATE INDEX IF NOT EXISTS event_at ON event(at, id);
CREATE INDEX IF NOT EXISTS block_original_ordinal ON block(message_pk,coalesce(ordinal,cast(idx AS TEXT)));

-- Restored requests select all raw spellings of one NFC request text.
CREATE INDEX IF NOT EXISTS text_normalized ON text(normalized_hash,id);

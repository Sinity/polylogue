-- Foreign-key parent deletes need these child lookups during the first build too.
CREATE INDEX IF NOT EXISTS link_child  ON link(child_pk);
CREATE INDEX IF NOT EXISTS event_after   ON event(after_message_pk) WHERE after_message_pk IS NOT NULL;

-- Duplicate derivation also clears old marks after capture inputs disappear.
-- Ordinary unmarked sessions have no entries and need no message-table scan.
CREATE INDEX IF NOT EXISTS message_duplicate ON message(session_pk) WHERE duplicate_of IS NOT NULL;

-- A default page reads its unit's rows in the order of its first terms
-- (`time IS NULL, time, ...`), so these indexes carry those exact
-- expressions: the page stops after its rows instead of sorting the table.
CREATE INDEX IF NOT EXISTS session_order ON session(last_activity_at IS NULL, last_activity_at DESC, id DESC);
CREATE INDEX IF NOT EXISTS message_order ON message(at IS NULL, at, session_order, id);
-- Explicit newest-first sorts tie by semantic identity in ascending order.
CREATE INDEX IF NOT EXISTS session_recent ON session(last_activity_at IS NULL, last_activity_at DESC, id);
CREATE INDEX IF NOT EXISTS message_recent ON message(at IS NULL, at DESC, id);
CREATE INDEX IF NOT EXISTS tool_call_order ON tool_call(at IS NULL, at, call_block_pk);
CREATE INDEX IF NOT EXISTS event_order ON event(at IS NULL, at, id);
CREATE INDEX IF NOT EXISTS message_at ON message(at, id);
CREATE INDEX IF NOT EXISTS message_session_at ON message(session_pk, at, id);
CREATE INDEX IF NOT EXISTS message_model_at ON message(model, at, id);
CREATE INDEX IF NOT EXISTS message_session_model_at ON message(session_pk, model, at, id);
CREATE INDEX IF NOT EXISTS event_at ON event(at, id);
CREATE INDEX IF NOT EXISTS block_original_ordinal ON block(message_pk,coalesce(ordinal,cast(idx AS TEXT)));

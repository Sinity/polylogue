-- A default page reads its unit's rows in the order of its first terms
-- (`time IS NULL, time, ...`), so these indexes carry those exact
-- expressions: the page stops after its rows instead of sorting the table.
CREATE INDEX IF NOT EXISTS session_order ON session(last_activity_at IS NULL, last_activity_at DESC, id DESC);
CREATE INDEX IF NOT EXISTS message_order ON message(at IS NULL, at, session_order, id);
CREATE INDEX IF NOT EXISTS tool_call_order ON tool_call(at IS NULL, at, call_block_pk);
CREATE INDEX IF NOT EXISTS event_order ON event(at IS NULL, at, id);
CREATE INDEX IF NOT EXISTS message_at ON message(at, id);
CREATE INDEX IF NOT EXISTS message_session_at ON message(session_pk, at, id);
CREATE INDEX IF NOT EXISTS message_model_at ON message(model, at, id);
CREATE INDEX IF NOT EXISTS message_session_model_at ON message(session_pk, model, at, id);
CREATE INDEX IF NOT EXISTS event_at ON event(at, id);
CREATE INDEX IF NOT EXISTS block_original_ordinal ON block(message_pk,coalesce(ordinal,cast(idx AS TEXT)));

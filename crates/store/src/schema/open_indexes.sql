-- A default page reads its unit's rows in the order of its first terms
-- (`time IS NULL, time, ...`) as two parts, the dated rows and then the
-- undated ones (query/compile/statements.rs, page_of), so one index on
-- `(time, ...)` serves both parts in either direction and a time bound or
-- a continuation seeks in it.
CREATE INDEX IF NOT EXISTS session_activity ON session(last_activity_at, id);   -- default order, since:
CREATE INDEX IF NOT EXISTS message_time ON message(at, session_order, id);
CREATE INDEX IF NOT EXISTS event_time ON event(at, id);
CREATE INDEX IF NOT EXISTS document_time ON document(at, id);
CREATE INDEX IF NOT EXISTS tool_call_time ON tool_call(at, call_block_pk);
-- The longest stored period, read by one seek at the end of the index: a
-- lower time bound K reads the range at >= K - longest, which holds every
-- row whose period reaches K (query/fields.rs, TimeColumns).
CREATE INDEX IF NOT EXISTS message_period ON message(at_until - at) WHERE at_until IS NOT NULL;
CREATE INDEX IF NOT EXISTS event_period ON event(at_until - at) WHERE at_until IS NOT NULL;
-- The few messages a summary marks, off the active path or shown as
-- another's duplicate: a session with none is answered by one probe.
CREATE INDEX IF NOT EXISTS message_marked ON message(session_pk)
  WHERE on_active_path IS NOT NULL OR duplicate_of IS NOT NULL;
CREATE INDEX IF NOT EXISTS block_original_ordinal ON block(message_pk,coalesce(ordinal,cast(idx AS TEXT)));

-- Restored requests select all raw spellings of one NFC request text.
CREATE INDEX IF NOT EXISTS text_normalized ON text(normalized_hash,id);

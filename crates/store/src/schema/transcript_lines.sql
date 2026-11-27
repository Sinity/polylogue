-- Transcript lines (summary/lines.rs): per session, the ordered lines a
-- transcript can show, with token estimates, written by the session summary
-- from the session's assembled rows, off the writer. The line counters and
-- totals (`LineState`) are the `state` of the session's `transcript_lines` row
-- in `summary_state`.
CREATE TABLE transcript_line (
  session_pk        INTEGER NOT NULL REFERENCES session(pk),
  position               INTEGER NOT NULL,       -- place in the session's reading order, from 0
  source_key        BLOB,                   -- on the first line of a message version or an event: its identity
  message_pk        INTEGER,                -- on a message version's first line: the message
  event_pk          INTEGER,                -- on an event's line: the event
  time_bound        INTEGER,                -- on a message version's first line: the latest message time up to it,
                                            --   the session's first message aside; places events that name no message
  row_key           BLOB,                   -- on a unit's first line: its row's key in session order (summary/fold.rs,
                                            --   `Key`) when the line was written; a change at or before it reaches the unit
  body              BLOB NOT NULL,          -- the model IndexEntry, compact and without what the row,
                                            --   its unit's first row and call_id hold (items/entry_bytes.rs)
  class             INTEGER NOT NULL,       -- the model EntryClass: what decides whether a reading shows the line
  call_id           TEXT,                   -- on a call or result: its provider call ID
  counts            BLOB,                   -- on a unit's first line: per class, how many lines come before it,
                                            --   and the running sum of their full-form token estimates
  PRIMARY KEY (session_pk, position)
) STRICT, WITHOUT ROWID;
CREATE INDEX transcript_line_message ON transcript_line(message_pk) WHERE message_pk IS NOT NULL;
CREATE INDEX transcript_line_event ON transcript_line(event_pk) WHERE event_pk IS NOT NULL;
CREATE INDEX transcript_line_row ON transcript_line(session_pk, row_key) WHERE row_key IS NOT NULL;
-- A person's prompt (header kind 0, prompt author 2), a compaction (event kind 9) and a
-- microcompaction (event kind 10): the lines a reading finds by kind. The low six bits
-- of class are its kind and author.
CREATE INDEX transcript_line_prompt ON transcript_line(session_pk, position) WHERE class & 63 = 32;
CREATE INDEX transcript_line_compaction ON transcript_line(session_pk, position) WHERE class & 15 = 9;
CREATE INDEX transcript_line_microcompaction ON transcript_line(session_pk, position) WHERE class & 15 = 10;
-- The lines with a provider call ID, by session and call: a call and its
-- results found without reading the session's lines.
CREATE TABLE transcript_call (
  session_pk        INTEGER NOT NULL REFERENCES session(pk),
  call_id           TEXT NOT NULL,
  position               INTEGER NOT NULL,       -- the line (transcript_line.position)
  class             INTEGER NOT NULL,       -- the line's class
  PRIMARY KEY (session_pk, call_id, position)
) STRICT, WITHOUT ROWID;

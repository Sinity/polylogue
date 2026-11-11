-- The session item index: per physical session, the ordered layout items a
-- transcript can show, with token estimates, kept by the writer in the same
-- transaction as the rows they describe.
CREATE TABLE session_item (
  session_pk        INTEGER NOT NULL REFERENCES session(pk),
  seq               INTEGER NOT NULL,       -- place in the session's reading order, from 0
  unit              BLOB,                   -- on the first entry of a message version or an event: its identity
  message_pk        INTEGER,                -- on a message version's first entry: the message
  event_pk          INTEGER,                -- on an event's entry: the event
  time_bound        INTEGER,                -- on a message version's first entry: the latest message time up to it,
                                            --   the session's first message aside; places events that name no message
  entry             BLOB NOT NULL,          -- the model IndexEntry as JSON
  class             INTEGER NOT NULL,       -- the model EntryClass: what decides whether a reading shows the entry
  call_id           TEXT,                   -- on a call or result: its provider call ID
  counts            BLOB,                   -- on a unit's first entry: per class, how many entries come before it,
                                            --   and the running sum of their full-form token estimates
  PRIMARY KEY (session_pk, seq)
) STRICT, WITHOUT ROWID;
CREATE INDEX session_item_message ON session_item(message_pk) WHERE message_pk IS NOT NULL;
CREATE INDEX session_item_event ON session_item(event_pk) WHERE event_pk IS NOT NULL;
-- A person's prompt (header kind 0, prompt author 2), a compaction (event kind 9) and a
-- microcompaction (event kind 11): the entries a reading finds by kind. The low six bits
-- of class are its kind and author.
CREATE INDEX session_item_prompt ON session_item(session_pk, seq) WHERE class & 63 = 32;
CREATE INDEX session_item_compaction ON session_item(session_pk, seq) WHERE class & 15 = 9;
CREATE INDEX session_item_microcompaction ON session_item(session_pk, seq) WHERE class & 15 = 11;
CREATE INDEX session_item_call ON session_item(session_pk, call_id) WHERE call_id IS NOT NULL;
CREATE TABLE session_item_state (
  session_pk        INTEGER PRIMARY KEY REFERENCES session(pk),
  append_seq        INTEGER NOT NULL,       -- moves when entries are added at the end
  rewrite_version   INTEGER NOT NULL,       -- moves when anything already indexed changes; stales cursors
  links             BLOB NOT NULL,          -- identity of the links that compose this session as a child
  entries           INTEGER NOT NULL,       -- how many entries the session has
  counts            BLOB NOT NULL,          -- session_item.counts over every entry
  indexed_changed   INTEGER NOT NULL        -- session.changed when the entries were brought up to date
) STRICT;

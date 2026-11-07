-- The session item index: per physical session, the ordered layout items a
-- transcript can show, with token estimates, kept by the writer in the same
-- transaction as the rows they describe.
CREATE TABLE session_item (
  session_pk        INTEGER NOT NULL REFERENCES session(pk),
  seq               INTEGER NOT NULL,       -- place in the session's reading order, from 0
  unit              BLOB,                   -- on the first entry of a message version or an event: its identity
  entry             BLOB NOT NULL,          -- the model IndexEntry as JSON
  PRIMARY KEY (session_pk, seq)
) STRICT, WITHOUT ROWID;
CREATE TABLE session_item_state (
  session_pk        INTEGER PRIMARY KEY REFERENCES session(pk),
  append_seq        INTEGER NOT NULL,       -- moves when entries are added at the end
  rewrite_version   INTEGER NOT NULL,       -- moves when anything already indexed changes; stales cursors
  links             BLOB NOT NULL,          -- identity of the links that compose this session as a child
  indexed_changed   INTEGER NOT NULL        -- session.changed when the entries were brought up to date
) STRICT;

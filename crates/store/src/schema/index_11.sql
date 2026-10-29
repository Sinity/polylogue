-- Source signal storage, transformed by SQLite from the source4 declaration.
ALTER TABLE source_delivery ADD COLUMN signal BLOB;
CREATE TABLE "source_signal_event" (
  request TEXT NOT NULL,
  ordinal INTEGER NOT NULL CHECK(ordinal >= 0),
  delivery TEXT NOT NULL REFERENCES "source_delivery"(id),
  position INTEGER NOT NULL CHECK(position > 0),
  at INTEGER NOT NULL,
  body BLOB NOT NULL,
  commit_seq INTEGER NOT NULL,
  PRIMARY KEY (delivery, position),
  UNIQUE (request, ordinal)
) STRICT, WITHOUT ROWID;
CREATE TABLE "source_signal_state" (
  owner TEXT PRIMARY KEY,
  body BLOB NOT NULL,
  commit_seq INTEGER NOT NULL
) STRICT, WITHOUT ROWID;
CREATE INDEX source_signal_event_commit_seq ON "source_signal_event"(commit_seq);

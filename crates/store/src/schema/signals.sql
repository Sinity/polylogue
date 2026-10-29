-- Signal state commits with immutable deliveries and their returned receipts.
ALTER TABLE delivery ADD COLUMN signal BLOB;

CREATE TABLE signal_state (
  owner TEXT PRIMARY KEY,
  body BLOB NOT NULL,
  commit_seq INTEGER NOT NULL
) STRICT, WITHOUT ROWID;

CREATE TABLE signal_event (
  request TEXT NOT NULL,
  ordinal INTEGER NOT NULL CHECK(ordinal >= 0),
  delivery TEXT NOT NULL REFERENCES delivery(id),
  position INTEGER NOT NULL CHECK(position > 0),
  at INTEGER NOT NULL,
  body BLOB NOT NULL,
  commit_seq INTEGER NOT NULL,
  PRIMARY KEY (delivery, position),
  UNIQUE (request, ordinal)
) STRICT, WITHOUT ROWID;
CREATE INDEX signal_event_commit_seq ON signal_event(commit_seq);

CREATE TABLE assertion (
  id TEXT PRIMARY KEY,
  root TEXT NOT NULL,
  kind TEXT NOT NULL,
  key TEXT,
  value TEXT,
  author TEXT,
  via TEXT,
  machine TEXT,
  created_at TEXT,
  valid_until TEXT,
  imports TEXT NOT NULL DEFAULT '{"observed":[],"declared":[]}',
  historical BLOB,
  commit_seq INTEGER NOT NULL CHECK(commit_seq>=0),
  CHECK(historical IS NOT NULL OR author IS NOT NULL)
) STRICT;
CREATE TABLE assertion_link (
  assertion TEXT NOT NULL REFERENCES assertion(id),
  role TEXT NOT NULL,
  ref TEXT NOT NULL,
  seen_hash BLOB CHECK(seen_hash IS NULL OR length(seen_hash)=32),
  PRIMARY KEY(assertion,role,ref)
) STRICT, WITHOUT ROWID;
CREATE INDEX assertion_link_ref ON assertion_link(ref,role);
CREATE INDEX assertion_kind_key ON assertion(kind,key);
CREATE INDEX assertion_root ON assertion(root);
CREATE INDEX assertion_commit_seq ON assertion(commit_seq,id);

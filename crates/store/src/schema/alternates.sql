CREATE TABLE alternate (
  id TEXT PRIMARY KEY,
  root TEXT NOT NULL,
  base TEXT,
  source TEXT NOT NULL,
  source_hash BLOB NOT NULL CHECK(length(source_hash)=32),
  kind TEXT NOT NULL,
  from_lang TEXT,
  to_lang TEXT,
  author TEXT NOT NULL,
  via TEXT NOT NULL,
  method TEXT,
  machine TEXT NOT NULL,
  created_at TEXT NOT NULL,
  commit_seq INTEGER NOT NULL
) STRICT;
CREATE INDEX alternate_root ON alternate(root);
CREATE INDEX alternate_source ON alternate(source,kind,to_lang);
CREATE INDEX alternate_commit_seq ON alternate(commit_seq);
CREATE TABLE alternate_part (
  alternate TEXT NOT NULL REFERENCES alternate(id),
  position INTEGER NOT NULL CHECK(position>=0),
  source TEXT NOT NULL,
  source_hash BLOB NOT NULL CHECK(length(source_hash)=32),
  relation TEXT NOT NULL,
  body BLOB NOT NULL,
  PRIMARY KEY(alternate,position)
) STRICT, WITHOUT ROWID;

CREATE INDEX alternate_part_source ON alternate_part(alternate,source,position);

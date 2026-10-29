CREATE TABLE variant (
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
  created_at INTEGER NOT NULL,
  commit_seq INTEGER NOT NULL
) STRICT;
CREATE INDEX variant_root ON variant(root);
CREATE INDEX variant_source ON variant(source,kind,to_lang);
CREATE INDEX variant_commit_seq ON variant(commit_seq);
CREATE TABLE variant_part (
  variant TEXT NOT NULL REFERENCES variant(id),
  n INTEGER NOT NULL CHECK(n>=0),
  of TEXT NOT NULL,
  of_hash BLOB NOT NULL CHECK(length(of_hash)=32),
  relation TEXT NOT NULL,
  body BLOB NOT NULL,
  PRIMARY KEY(variant,n)
) STRICT, WITHOUT ROWID;

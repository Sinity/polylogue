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
  created_at TEXT NOT NULL,
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

CREATE INDEX variant_part_source ON variant_part(variant,of,n);

-- Observed ingress is computed locally; declared claims retain sender history.
CREATE TABLE variant_import (
  ref TEXT NOT NULL,
  variant TEXT NOT NULL REFERENCES variant(id),
  observed INTEGER NOT NULL CHECK(observed IN (0,1)),
  input TEXT NOT NULL,
  chain INTEGER NOT NULL,
  machine TEXT NOT NULL,
  actor TEXT NOT NULL,
  at TEXT NOT NULL,
  commit_seq INTEGER NOT NULL,
  PRIMARY KEY(ref,input,chain,machine,actor,observed,at)
) STRICT, WITHOUT ROWID;
CREATE UNIQUE INDEX variant_import_observed
  ON variant_import(ref,input,chain,machine,actor) WHERE observed=1;
CREATE INDEX variant_import_variant ON variant_import(variant);

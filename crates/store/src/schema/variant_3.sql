-- Legacy variants stored milliseconds; preserve those instants through the typed codec.
CREATE TEMP TABLE variant_part_upgrade AS SELECT * FROM @source.variant_part;
CREATE TABLE @source.variant_upgrade (
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
INSERT INTO @source.variant_upgrade
 SELECT id,root,base,source,source_hash,kind,from_lang,to_lang,author,via,method,machine,
        polylogue_legacy_variant_time(created_at),commit_seq FROM @source.variant;
DROP TABLE @source.variant_part;
DROP TABLE @source.variant;
ALTER TABLE @source.variant_upgrade RENAME TO @source.variant;
CREATE INDEX @source.variant_root ON @source.variant(root);
CREATE INDEX @source.variant_source ON @source.variant(source,kind,to_lang);
CREATE INDEX @source.variant_commit_seq ON @source.variant(commit_seq);
CREATE TABLE @source.variant_part (
  variant TEXT NOT NULL REFERENCES @source.variant(id),
  n INTEGER NOT NULL CHECK(n>=0),
  of TEXT NOT NULL,
  of_hash BLOB NOT NULL CHECK(length(of_hash)=32),
  relation TEXT NOT NULL,
  body BLOB NOT NULL,
  PRIMARY KEY(variant,n)
) STRICT, WITHOUT ROWID;

CREATE INDEX @source.variant_part_source ON @source.variant_part(variant,of,n);

-- Observed ingress is computed locally; declared claims retain sender history.
CREATE TABLE @source.variant_import (
  ref TEXT NOT NULL,
  variant TEXT NOT NULL REFERENCES @source.variant(id),
  observed INTEGER NOT NULL CHECK(observed IN (0,1)),
  input TEXT NOT NULL,
  chain INTEGER NOT NULL,
  machine TEXT NOT NULL,
  actor TEXT NOT NULL,
  at TEXT NOT NULL,
  commit_seq INTEGER NOT NULL,
  PRIMARY KEY(ref,input,chain,machine,actor,observed,at)
) STRICT, WITHOUT ROWID;
CREATE UNIQUE INDEX @source.variant_import_observed
  ON @source.variant_import(ref,input,chain,machine,actor) WHERE observed=1;
CREATE INDEX @source.variant_import_variant ON @source.variant_import(variant);

INSERT INTO @source.variant_part SELECT * FROM temp.variant_part_upgrade;
DROP TABLE temp.variant_part_upgrade;

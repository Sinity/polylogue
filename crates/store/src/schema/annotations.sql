CREATE TABLE annotation (
  id          TEXT PRIMARY KEY,   -- revision ID: the request ID, '<request>.<n>' for the n-th write of one request, or a derived markup ID
  root        TEXT NOT NULL,      -- the first revision's ID; equals id for a first revision
  kind        TEXT NOT NULL,
  key         TEXT,
  value       TEXT,               -- text, or canonical JSON checked against the kind's shape
  author      TEXT NOT NULL,      -- computed from the channel, never supplied by a caller
  via         TEXT NOT NULL,
  machine     TEXT NOT NULL,
  created_at  INTEGER,            -- Original authored Unix milliseconds; NULL when unknown
  valid_until INTEGER,            -- Unix milliseconds
  commit_seq  INTEGER NOT NULL CHECK(commit_seq>=0)
) STRICT;
CREATE INDEX annotation_kind_key ON annotation(kind,key);
CREATE INDEX annotation_root ON annotation(root);
CREATE INDEX annotation_commit_seq ON annotation(commit_seq,id);

-- Every reference an annotation makes, in its role: target, evidence, scope,
-- origin, revision, withdrawal and every relation are one fact.
CREATE TABLE annotation_link (
  annotation TEXT NOT NULL REFERENCES annotation(id),
  role      TEXT NOT NULL,
  ref       TEXT NOT NULL,        -- the stored reference form
  seen_hash BLOB CHECK(seen_hash IS NULL OR length(seen_hash)=32), -- BLAKE3 of the target text when written
  quote     TEXT CHECK(quote IS NULL OR json_valid(quote)),        -- span links: the exact text with up to 32 characters before and after
  PRIMARY KEY(annotation,role,ref)
) STRICT, WITHOUT ROWID;
CREATE INDEX annotation_link_ref ON annotation_link(ref,role,annotation);

-- Which revisions are current and withdrawn, recomputed in the writing
-- transaction for exactly the roots a write can affect. An index over the
-- `revises` and `withdraws` links, never a source of truth.
CREATE TABLE revision_state (
  id        TEXT PRIMARY KEY,
  root      TEXT NOT NULL,
  current   INTEGER NOT NULL CHECK(current IN (0,1)),
  withdrawn INTEGER NOT NULL CHECK(withdrawn IN (0,1))
) STRICT, WITHOUT ROWID;
CREATE INDEX revision_state_root ON revision_state(root,current,id);

-- The unique keys each revision's content holds, by the key's digest.
-- A write without bases finds the current holders of its keys here.
CREATE TABLE annotation_unique_key (
  key_hash  BLOB NOT NULL CHECK(length(key_hash)=32),
  id        TEXT NOT NULL REFERENCES annotation(id),
  PRIMARY KEY(key_hash,id)
) STRICT, WITHOUT ROWID;

-- A source can contain a fact already held through another physical ingress.
CREATE TABLE annotation_import (
  annotation TEXT NOT NULL REFERENCES annotation(id),
  input TEXT NOT NULL,
  chain INTEGER NOT NULL,
  machine TEXT NOT NULL,
  actor TEXT NOT NULL,
  at TEXT NOT NULL,
  commit_seq INTEGER NOT NULL,
  PRIMARY KEY(annotation,input,chain,machine,actor)
) STRICT, WITHOUT ROWID;

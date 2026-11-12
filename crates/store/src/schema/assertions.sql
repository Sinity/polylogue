CREATE TABLE assertion (
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
CREATE INDEX assertion_kind_key ON assertion(kind,key);
CREATE INDEX assertion_root ON assertion(root);
CREATE INDEX assertion_commit_seq ON assertion(commit_seq,id);

-- Every reference an assertion makes, in its role: target, evidence, scope,
-- origin, revision, withdrawal and every relation are one fact.
CREATE TABLE link (
  record    TEXT NOT NULL REFERENCES assertion(id),
  role      TEXT NOT NULL,
  ref       TEXT NOT NULL,        -- the stored reference form
  seen_hash BLOB CHECK(seen_hash IS NULL OR length(seen_hash)=32), -- BLAKE3 of the target text when written
  quote     TEXT CHECK(quote IS NULL OR json_valid(quote)),        -- span links: the exact text with up to 32 characters before and after
  PRIMARY KEY(record,role,ref)
) STRICT, WITHOUT ROWID;
CREATE INDEX link_ref ON link(ref,role,record);

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

-- The uniqueness slots each revision's content occupies, by the slot's digest.
-- A write without bases finds the current holders of its slots here.
CREATE TABLE assertion_slot (
  slot      BLOB NOT NULL CHECK(length(slot)=32),
  id        TEXT NOT NULL REFERENCES assertion(id),
  PRIMARY KEY(slot,id)
) STRICT, WITHOUT ROWID;

-- A source can contain a fact already held through another physical ingress.
CREATE TABLE assertion_import (
  assertion TEXT NOT NULL REFERENCES assertion(id),
  input TEXT NOT NULL,
  chain INTEGER NOT NULL,
  machine TEXT NOT NULL,
  actor TEXT NOT NULL,
  at TEXT NOT NULL,
  commit_seq INTEGER NOT NULL,
  PRIMARY KEY(assertion,input,chain,machine,actor)
) STRICT, WITHOUT ROWID;

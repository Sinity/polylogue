-- A reading and its selected rows share the same immutable typed payload.
CREATE TABLE block_value (
  pk INTEGER PRIMARY KEY,
  hash BLOB NOT NULL UNIQUE,
  body BLOB NOT NULL,
  data BLOB
) STRICT;
CREATE TABLE message_version_block (
  version_hash BLOB NOT NULL REFERENCES message_version(version_hash) ON DELETE CASCADE,
  idx INTEGER NOT NULL,
  value_pk INTEGER NOT NULL REFERENCES block_value(pk),
  PRIMARY KEY(version_hash,idx)
) STRICT, WITHOUT ROWID;
CREATE INDEX message_version_block_value ON message_version_block(value_pk);

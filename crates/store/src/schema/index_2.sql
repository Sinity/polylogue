CREATE TABLE message_projection (
  version_hash BLOB PRIMARY KEY REFERENCES message_version(version_hash) ON DELETE CASCADE,
  layout BLOB NOT NULL
) STRICT, WITHOUT ROWID;
ALTER TABLE block ADD COLUMN ordinal TEXT;

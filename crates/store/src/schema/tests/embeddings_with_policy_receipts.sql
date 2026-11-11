-- One vector per text, embedding key and chunk.
CREATE TABLE embedded (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,  -- explicit, so VACUUM never renumbers it; never reused,
                                     --   so the highest id is the embeddings part of a read's view
  text_hash   BLOB NOT NULL,         -- the text that was embedded
  model       TEXT NOT NULL,         -- the full embedding key: provider/model/dimensions/input type/normalization,
                                     --   e.g. 'voyage/voyage-4/1024/document/unit'
  chunk       INTEGER NOT NULL DEFAULT 0,  -- texts longer than the model's input are embedded in chunks
  dims        INTEGER NOT NULL,
  vector      BLOB NOT NULL,         -- dims little-endian f32 values
  embedded_at INTEGER NOT NULL,
  UNIQUE (text_hash, model, chunk)
) STRICT;

-- A text the provider permanently rejected (HTTP 400 and similar): recorded once
-- with the provider's error, listed in `status`, retried only when the
-- embedding key or the text changes.
CREATE TABLE embed_failure (
  text_hash   BLOB NOT NULL,
  model       TEXT NOT NULL,         -- the full embedding key, as in `embedded`
  error       TEXT NOT NULL,         -- the provider's error, as given
  failed_at   INTEGER NOT NULL,
  PRIMARY KEY (text_hash, model)
) STRICT, WITHOUT ROWID;

CREATE TABLE vector_policy_receipt (
 excision_id TEXT PRIMARY KEY,
 archive_id TEXT NOT NULL,
 policy_hash BLOB NOT NULL CHECK(length(policy_hash)=32)
) STRICT, WITHOUT ROWID;
CREATE TABLE vector_meta(key TEXT PRIMARY KEY,value TEXT NOT NULL) STRICT;
CREATE TRIGGER vector_receipt_insert AFTER INSERT ON vector_policy_receipt BEGIN
 DELETE FROM vector_meta WHERE key='policy_revision';
END;
CREATE TRIGGER vector_receipt_update AFTER UPDATE ON vector_policy_receipt BEGIN
 DELETE FROM vector_meta WHERE key='policy_revision';
END;
CREATE TRIGGER vector_receipt_delete AFTER DELETE ON vector_policy_receipt BEGIN
 DELETE FROM vector_meta WHERE key='policy_revision';
END;

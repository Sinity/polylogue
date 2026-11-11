-- The provider's vector for a `similar:` query text, kept so a text is paid
-- for once and ranks with model calls off. Not part of a model's revision:
-- the loaded document vectors do not change when one is stored.
CREATE TABLE query_vector (
  key         TEXT NOT NULL,         -- the full embedding key of the query, e.g.
                                     --   'voyage/voyage-4/1024/query/unit'
  text_hash   BLOB NOT NULL,         -- BLAKE3 of the query text's NFC bytes
  vector      BLOB NOT NULL,         -- little-endian f32 values
  embedded_at INTEGER NOT NULL,
  PRIMARY KEY (key, text_hash)
) STRICT, WITHOUT ROWID;

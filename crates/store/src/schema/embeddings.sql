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

-- Derived revision of each model, read in the same snapshot as its vectors.
CREATE TABLE embedding_revision (
  model TEXT PRIMARY KEY,
  revision INTEGER NOT NULL
) STRICT, WITHOUT ROWID;
INSERT INTO embedding_revision(model, revision)
SELECT model, 1 FROM embedded UNION SELECT model, 1 FROM embed_failure;
CREATE TRIGGER embedded_revision_insert AFTER INSERT ON embedded
BEGIN
  INSERT INTO embedding_revision(model, revision) VALUES (NEW.model, 1)
  ON CONFLICT(model) DO UPDATE SET revision = revision + 1;
END;
CREATE TRIGGER embedded_revision_update AFTER UPDATE ON embedded
WHEN OLD.text_hash <> NEW.text_hash OR OLD.model <> NEW.model OR OLD.chunk <> NEW.chunk OR OLD.dims <> NEW.dims OR OLD.vector <> NEW.vector OR OLD.embedded_at <> NEW.embedded_at
BEGIN
  INSERT INTO embedding_revision(model, revision) VALUES (NEW.model, 1)
  ON CONFLICT(model) DO UPDATE SET revision = revision + 1;
END;
CREATE TRIGGER embedded_revision_delete AFTER DELETE ON embedded
BEGIN
  INSERT INTO embedding_revision(model, revision) VALUES (OLD.model, 1)
  ON CONFLICT(model) DO UPDATE SET revision = revision + 1;
END;
CREATE TRIGGER embedded_revision_old_model AFTER UPDATE ON embedded
WHEN OLD.model <> NEW.model
BEGIN
  INSERT INTO embedding_revision(model, revision) VALUES (OLD.model, 1)
  ON CONFLICT(model) DO UPDATE SET revision = revision + 1;
END;
CREATE TRIGGER embed_failure_revision_insert AFTER INSERT ON embed_failure
BEGIN
  INSERT INTO embedding_revision(model, revision) VALUES (NEW.model, 1)
  ON CONFLICT(model) DO UPDATE SET revision = revision + 1;
END;
CREATE TRIGGER embed_failure_revision_update AFTER UPDATE ON embed_failure
WHEN OLD.text_hash <> NEW.text_hash OR OLD.model <> NEW.model OR OLD.error <> NEW.error OR OLD.failed_at <> NEW.failed_at
BEGIN
  INSERT INTO embedding_revision(model, revision) VALUES (NEW.model, 1)
  ON CONFLICT(model) DO UPDATE SET revision = revision + 1;
END;
CREATE TRIGGER embed_failure_revision_delete AFTER DELETE ON embed_failure
BEGIN
  INSERT INTO embedding_revision(model, revision) VALUES (OLD.model, 1)
  ON CONFLICT(model) DO UPDATE SET revision = revision + 1;
END;
CREATE TRIGGER embed_failure_revision_old_model AFTER UPDATE ON embed_failure
WHEN OLD.model <> NEW.model
BEGIN
  INSERT INTO embedding_revision(model, revision) VALUES (OLD.model, 1)
  ON CONFLICT(model) DO UPDATE SET revision = revision + 1;
END;

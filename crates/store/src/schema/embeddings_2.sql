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

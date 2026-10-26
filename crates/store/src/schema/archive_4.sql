-- How many inputs and revisions the archive holds and the bytes of its present
-- blobs, kept by the triggers below so that status does not scan the tables.
CREATE TABLE row_count (
  name          TEXT PRIMARY KEY,        -- input | revision | stored_bytes
  n             INTEGER NOT NULL
) STRICT, WITHOUT ROWID;
INSERT INTO row_count(name, n) VALUES
  ('input', (SELECT COUNT(*) FROM input)),
  ('revision', (SELECT COUNT(*) FROM revision)),
  ('stored_bytes', (SELECT COALESCE(SUM(size), 0) FROM blob WHERE state = 'present'));
CREATE TRIGGER input_added AFTER INSERT ON input BEGIN UPDATE row_count SET n = n + 1 WHERE name = 'input'; END;
CREATE TRIGGER input_removed AFTER DELETE ON input BEGIN UPDATE row_count SET n = n - 1 WHERE name = 'input'; END;
CREATE TRIGGER revision_added AFTER INSERT ON revision BEGIN UPDATE row_count SET n = n + 1 WHERE name = 'revision'; END;
CREATE TRIGGER revision_removed AFTER DELETE ON revision BEGIN UPDATE row_count SET n = n - 1 WHERE name = 'revision'; END;
CREATE TRIGGER blob_added AFTER INSERT ON blob WHEN new.state = 'present' BEGIN UPDATE row_count SET n = n + new.size WHERE name = 'stored_bytes'; END;
CREATE TRIGGER blob_changed AFTER UPDATE OF state, size ON blob BEGIN
  UPDATE row_count SET n = n + (CASE WHEN new.state = 'present' THEN new.size ELSE 0 END)
    - (CASE WHEN old.state = 'present' THEN old.size ELSE 0 END)
  WHERE name = 'stored_bytes';
END;
CREATE TRIGGER blob_removed AFTER DELETE ON blob WHEN old.state = 'present' BEGIN UPDATE row_count SET n = n - old.size WHERE name = 'stored_bytes'; END;

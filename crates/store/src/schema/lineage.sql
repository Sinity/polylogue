-- Structural products share the canonical relation projection with all readers.
CREATE TABLE session_lineage (
  session_pk INTEGER PRIMARY KEY REFERENCES session(pk) ON DELETE CASCADE,
  root TEXT,
  depth INTEGER,
  root_pending INTEGER NOT NULL CHECK(root_pending IN (0,1)),
  issue TEXT
) STRICT;
CREATE INDEX session_lineage_root ON session_lineage(root,session_pk);
CREATE TRIGGER link_insert_lineage AFTER INSERT ON link WHEN NEW.kind IN ('spawn','fork','continuation','resume') BEGIN
  INSERT INTO derive_due(session_pk,rows_changed,lineage)
  WITH RECURSIVE affected(id) AS (
    SELECT NEW.child_id
    UNION SELECT l.child_id FROM link l JOIN affected a ON l.parent_id=a.id
      WHERE l.kind IN ('spawn','fork','continuation','resume')
  ) SELECT s.pk,0,1 FROM affected a JOIN session s ON s.id=a.id WHERE true
  ON CONFLICT(session_pk) DO UPDATE SET lineage=1,marked=marked+1;
END;

CREATE TRIGGER link_delete_lineage AFTER DELETE ON link WHEN OLD.kind IN ('spawn','fork','continuation','resume') BEGIN
  INSERT INTO derive_due(session_pk,rows_changed,lineage)
  WITH RECURSIVE affected(id) AS (
    SELECT OLD.child_id
    UNION SELECT l.child_id FROM link l JOIN affected a ON l.parent_id=a.id
      WHERE l.kind IN ('spawn','fork','continuation','resume')
  ) SELECT s.pk,0,1 FROM affected a JOIN session s ON s.id=a.id WHERE true
  ON CONFLICT(session_pk) DO UPDATE SET lineage=1,marked=marked+1;
END;

CREATE TRIGGER link_update_lineage AFTER UPDATE OF parent_id,child_id,kind ON link
WHEN (NEW.parent_id IS NOT OLD.parent_id OR NEW.child_id IS NOT OLD.child_id OR NEW.kind IS NOT OLD.kind)
  AND (NEW.kind IN ('spawn','fork','continuation','resume') OR OLD.kind IN ('spawn','fork','continuation','resume')) BEGIN
  INSERT INTO derive_due(session_pk,rows_changed,lineage)
  WITH RECURSIVE affected(id) AS (
    SELECT NEW.child_id UNION SELECT OLD.child_id
    UNION SELECT l.child_id FROM link l JOIN affected a ON l.parent_id=a.id
      WHERE l.kind IN ('spawn','fork','continuation','resume')
  ) SELECT s.pk,0,1 FROM affected a JOIN session s ON s.id=a.id WHERE true
  ON CONFLICT(session_pk) DO UPDATE SET lineage=1,marked=marked+1;
END;

CREATE TRIGGER session_insert_lineage AFTER INSERT ON session  BEGIN
  INSERT INTO derive_due(session_pk,rows_changed,lineage)
  WITH RECURSIVE affected(id) AS (
    SELECT NEW.id
    UNION SELECT l.child_id FROM link l JOIN affected a ON l.parent_id=a.id
      WHERE l.kind IN ('spawn','fork','continuation','resume')
  ) SELECT s.pk,0,1 FROM affected a JOIN session s ON s.id=a.id WHERE true
  ON CONFLICT(session_pk) DO UPDATE SET lineage=1,marked=marked+1;
END;

CREATE TRIGGER session_delete_lineage AFTER DELETE ON session  BEGIN
  INSERT INTO derive_due(session_pk,rows_changed,lineage)
  WITH RECURSIVE affected(id) AS (
    SELECT OLD.id
    UNION SELECT l.child_id FROM link l JOIN affected a ON l.parent_id=a.id
      WHERE l.kind IN ('spawn','fork','continuation','resume')
  ) SELECT s.pk,0,1 FROM affected a JOIN session s ON s.id=a.id WHERE true
  ON CONFLICT(session_pk) DO UPDATE SET lineage=1,marked=marked+1;
END;

CREATE TRIGGER session_update_lineage AFTER UPDATE OF stub ON session WHEN NEW.stub IS NOT OLD.stub BEGIN
  INSERT INTO derive_due(session_pk,rows_changed,lineage)
  WITH RECURSIVE affected(id) AS (
    SELECT NEW.id
    UNION SELECT l.child_id FROM link l JOIN affected a ON l.parent_id=a.id
      WHERE l.kind IN ('spawn','fork','continuation','resume')
  ) SELECT s.pk,0,1 FROM affected a JOIN session s ON s.id=a.id WHERE true
  ON CONFLICT(session_pk) DO UPDATE SET lineage=1,marked=marked+1;
END;

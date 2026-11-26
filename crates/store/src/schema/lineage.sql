-- Structural products share the one lineage view with all readers.
CREATE TABLE session_lineage (
  session_pk INTEGER PRIMARY KEY REFERENCES session(pk) ON DELETE CASCADE,
  root TEXT,
  depth INTEGER,
  root_pending INTEGER NOT NULL CHECK(root_pending IN (0,1)),
  issue TEXT
) STRICT;
CREATE INDEX session_lineage_root ON session_lineage(root,session_pk);
-- The code that writes a link or a session marks the lineage it reaches due
-- (lineage::mark_due); no trigger marks a session.

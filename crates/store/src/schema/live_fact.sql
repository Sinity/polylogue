-- Live facts become the fold of each session's live evidence; the columns
-- of the latest hook event and the claim-command task go.
DROP INDEX IF EXISTS session_live;
ALTER TABLE session DROP COLUMN live_last_event;
ALTER TABLE session DROP COLUMN live_last_event_at;
ALTER TABLE session DROP COLUMN live_open_tool_pk;
ALTER TABLE session DROP COLUMN live_waiting_on;
ALTER TABLE session DROP COLUMN live_waiting_since;
ALTER TABLE session DROP COLUMN live_todo;
ALTER TABLE session DROP COLUMN live_task;
ALTER TABLE session DROP COLUMN live_task_evidence;
ALTER TABLE session ADD COLUMN live_facts TEXT;
ALTER TABLE session ADD COLUMN live_last_at INTEGER;
CREATE INDEX session_live ON session(live_last_at) WHERE live_last_at IS NOT NULL;
CREATE TABLE live_interval (
  session_pk INTEGER NOT NULL REFERENCES session(pk),
  aspect     TEXT NOT NULL,
  value      TEXT NOT NULL,
  start_at   INTEGER NOT NULL,
  end_at     INTEGER NOT NULL,
  closed_by  TEXT NOT NULL,
  PRIMARY KEY (session_pk, aspect, start_at)
) STRICT;
CREATE INDEX live_interval_end ON live_interval(end_at);
CREATE TABLE process (
  session_pk INTEGER NOT NULL REFERENCES session(pk),
  boot_id    TEXT NOT NULL,
  pid        INTEGER NOT NULL,
  start      INTEGER NOT NULL,
  parent_pid INTEGER,
  comm       TEXT,
  argv       TEXT,
  cwd        TEXT,
  started_at INTEGER,
  ended_at   INTEGER,
  role       TEXT NOT NULL,
  PRIMARY KEY (session_pk, boot_id, pid, start)
) STRICT;
-- Every session with live evidence is folded again by its next derivation.
UPDATE session SET live_reparse = 1 WHERE pk IN (
  SELECT session_pk FROM event WHERE session_pk IS NOT NULL
    AND (kind LIKE 'hook~_%' ESCAPE '~' OR kind LIKE 'process~_%' ESCAPE '~'
         OR kind LIKE 'registry~_%' ESCAPE '~' OR kind LIKE 'git~_%' ESCAPE '~'));
DELETE FROM derive_state WHERE name = 'session'
  AND session_pk IN (SELECT pk FROM session WHERE live_reparse = 1);

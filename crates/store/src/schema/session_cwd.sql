-- Every working directory a session recorded (its `cwd_json` members and its
-- last `cwd`), one row each, so `cwd:` finds sessions by a key range instead
-- of reading every session row. The triggers keep it equal to the session
-- row whichever statement writes the row.
CREATE TABLE session_cwd_member (
  cwd               TEXT NOT NULL,
  session_pk        INTEGER NOT NULL REFERENCES session(pk),
  PRIMARY KEY (cwd, session_pk)
) STRICT, WITHOUT ROWID;
CREATE INDEX session_cwd_member_session ON session_cwd_member(session_pk);
CREATE TRIGGER session_cwd_member_added AFTER INSERT ON session BEGIN
  INSERT OR IGNORE INTO session_cwd_member(cwd, session_pk)
    SELECT d.value, NEW.pk FROM json_each(json_insert(coalesce(NEW.cwd_json, '[]'), '$[#]', NEW.cwd)) d
     WHERE d.type = 'text';
END;
CREATE TRIGGER session_cwd_member_changed AFTER UPDATE OF cwd, cwd_json ON session BEGIN
  DELETE FROM session_cwd_member WHERE session_pk = OLD.pk;
  INSERT OR IGNORE INTO session_cwd_member(cwd, session_pk)
    SELECT d.value, NEW.pk FROM json_each(json_insert(coalesce(NEW.cwd_json, '[]'), '$[#]', NEW.cwd)) d
     WHERE d.type = 'text';
END;
CREATE TRIGGER session_cwd_member_removed AFTER DELETE ON session BEGIN
  DELETE FROM session_cwd_member WHERE session_pk = OLD.pk;
END;
INSERT OR IGNORE INTO session_cwd_member(cwd, session_pk)
  SELECT d.value, s.pk FROM session s, json_each(json_insert(coalesce(s.cwd_json, '[]'), '$[#]', s.cwd)) d
   WHERE d.type = 'text';

-- The change feed of live columns: sessions whose hook facts changed after a
-- write counter value (the daemon's scope index follows it).
CREATE INDEX session_live_changed ON session(live_changed) WHERE live_changed IS NOT NULL;

-- What a session-start brief reports about a session, kept by the derive step
-- (derivation `brief`), so a brief reads one row per listed session.
CREATE TABLE session_brief (
  session_pk          INTEGER PRIMARY KEY REFERENCES session(pk),
  last_answer_pk      INTEGER,              -- the last model message with a text block, not a duplicate
  files_changed       INTEGER NOT NULL,     -- distinct paths written or edited successfully
  recent_files_json   TEXT NOT NULL,        -- the four most recently changed of them, newest first
  commits_json        TEXT NOT NULL,        -- commit hashes its commands made, in hash order
  pull_requests_json  TEXT NOT NULL,        -- owner/name#number, in order
  unanswered_prompt   INTEGER NOT NULL,     -- its last message is a person's prompt, not a command
  unfinished_tool_call INTEGER NOT NULL     -- its newest tool call has no result
) STRICT;

-- Session summaries (summary.rs): the values of one session computed from
-- its assembled rows off the writer, for the sessions marked due.

-- The version each declared summary of a session was computed under; a
-- summary whose version moved is computed again for every session. `state`
-- is what a summary keeps beside its rows: the transcript lines' counters and
-- totals (summary/lines.rs, `LineState`), which cursors bind.
CREATE TABLE summary_state (
  name              TEXT NOT NULL,
  session_pk        INTEGER NOT NULL REFERENCES session(pk) ON DELETE CASCADE,
  version           TEXT NOT NULL,
  state             BLOB,
  PRIMARY KEY (name, session_pk)
) STRICT, WITHOUT ROWID;
CREATE INDEX summary_state_session ON summary_state(session_pk);

-- The folds' state after the row at `after`, in the session's row order
-- (summary/fold.rs, `Key`): kept at the end and at sparse earlier places,
-- so a run resumes from the nearest one before the earliest change. `state`
-- holds every fold's state and the fold versions it was made under.
CREATE TABLE summary_resume (
  session_pk        INTEGER NOT NULL REFERENCES session(pk) ON DELETE CASCADE,
  after             BLOB NOT NULL,
  rows              INTEGER NOT NULL,       -- how many rows the state stands after
  state             BLOB NOT NULL,
  PRIMARY KEY (session_pk, after)
) STRICT, WITHOUT ROWID;

-- Sessions whose summaries are behind, marked in the transaction that made
-- them so by the code that wrote the change: the summary worker finds its
-- work here, never by scanning sessions. `from_order` is the earliest row
-- key the changes reached: X'' from the start, X'FF' no row (a cause
-- outside the session's rows, or a version that moved). `repo_lookup` 1: a
-- repository lookup at or above its working directory changed; `lineage`
-- 1: its explicit ancestry changed. `marked` counts the markings: the
-- worker stores what it computed only while the count is the one it read.
CREATE TABLE summary_due (
  session_pk        INTEGER PRIMARY KEY REFERENCES session(pk) ON DELETE CASCADE,
  from_order        BLOB NOT NULL,
  repo_lookup       INTEGER NOT NULL DEFAULT 0,
  lineage           INTEGER NOT NULL DEFAULT 0,
  marked            INTEGER NOT NULL DEFAULT 0
) STRICT;

-- What each session contributes to completion: how many of its messages name
-- each model and how many of its calls use each tool, and its first typed
-- prompt (summary/values.rs). The completion catalog copies the rows changed
-- since it was published.
CREATE TABLE session_value (
  session_pk        INTEGER NOT NULL REFERENCES session(pk),
  field             TEXT NOT NULL,          -- model | tool
  value             TEXT NOT NULL,
  count                 INTEGER NOT NULL CHECK (count >= 0),  -- 0: the session no longer has the value; the
                                            --   row stays so the catalog sees the removal
  changed           INTEGER NOT NULL,       -- write counter of the transaction that last changed count
  PRIMARY KEY (session_pk, field, value)
) STRICT, WITHOUT ROWID;
CREATE INDEX session_value_changed ON session_value(changed);
CREATE TABLE session_first_prompt (
  session_pk        INTEGER PRIMARY KEY REFERENCES session(pk),
  message_pk        INTEGER,                -- the first typed prompt with text; NULL when there is none
  changed           INTEGER NOT NULL        -- write counter of the transaction that last changed it
) STRICT;
CREATE INDEX session_first_prompt_changed ON session_first_prompt(changed);

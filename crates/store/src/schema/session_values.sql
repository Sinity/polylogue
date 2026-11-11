-- What each session contributes to completion: how many of its messages name
-- each model and how many of its calls use each tool, and its first typed
-- prompt. The writer keeps them in the same transaction as the rows, from the
-- transaction's change summary, and the completion catalog copies the rows
-- changed since it was published.
CREATE TABLE session_value (
  session_pk        INTEGER NOT NULL REFERENCES session(pk),
  field             TEXT NOT NULL,          -- model | tool
  value             TEXT NOT NULL,
  n                 INTEGER NOT NULL CHECK (n >= 0),  -- 0: the session no longer has the value; the
                                            --   row stays so the catalog sees the removal
  changed           INTEGER NOT NULL,       -- write counter of the transaction that last changed n
  PRIMARY KEY (session_pk, field, value)
) STRICT, WITHOUT ROWID;
CREATE INDEX session_value_changed ON session_value(changed);
CREATE TABLE session_first_prompt (
  session_pk        INTEGER PRIMARY KEY REFERENCES session(pk),
  message_pk        INTEGER,                -- the first typed prompt with text; NULL when there is none
  changed           INTEGER NOT NULL        -- write counter of the transaction that last changed it
) STRICT;
CREATE INDEX session_first_prompt_changed ON session_first_prompt(changed);

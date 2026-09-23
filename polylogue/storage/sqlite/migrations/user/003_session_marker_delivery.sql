-- Add ``session_marker_delivery``: the sink-owned marker delivery position.
--
-- polylogue-bp12n.1 makes the applied position durable so a restart resumes
-- from a committed point instead of making profile validity depend on whether
-- the last lowering attempt happened to finish. The position commits in the
-- same transaction as the assertion rows it describes, which is why it lives
-- in user.db rather than a derived tier: a rebuildable tier could disagree
-- with the durable rows after a crash.
--
-- user.db is durable and irreplaceable -- it is never rebuilt from source
-- evidence -- so the fresh DDL alone would give the table to new archives and
-- silently leave every existing one without it. That is what this migration
-- exists to prevent.
--
-- This is purely additive: one new table and one new index, no existing table
-- touched, no rows read or rewritten, no constraint dropped. Row counts on
-- every pre-existing table are unchanged, so the train declares no row-change
-- allowance, and the statements are the same ones the fresh DDL emits so
-- migrated and freshly-created archives converge on identical schema.

CREATE TABLE IF NOT EXISTS session_marker_delivery (
    session_id       TEXT PRIMARY KEY,
    input_binding    TEXT NOT NULL,
    applied_at_ms    INTEGER NOT NULL CHECK(applied_at_ms >= 0)
) STRICT;

CREATE INDEX IF NOT EXISTS idx_session_marker_delivery_binding
ON session_marker_delivery(input_binding);

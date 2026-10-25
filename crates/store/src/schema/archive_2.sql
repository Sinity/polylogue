-- Backfill jobs: the operator asks the browser extension to capture a web
-- chat's older conversations. The job is stored once; what happened to it is
-- in backfill_job_event, and its state is the latest event (an expired claim
-- reads as waiting).
CREATE TABLE backfill_job (
  id                TEXT PRIMARY KEY,       -- UUIDv7
  provider          TEXT NOT NULL,          -- the web chat, by the extension's name for it (chatgpt, claude-ai, grok)
  since             INTEGER,                -- only conversations the provider changed at or after this time; NULL: all
  created_at        INTEGER NOT NULL,
  commit_seq        INTEGER NOT NULL
) STRICT;
CREATE INDEX backfill_job_commit_seq ON backfill_job(commit_seq);

-- What happened to a backfill job, in order: started | claimed | progress |
-- finished | failed | cancelled. Each claimed or progress event carries the
-- totals so far and where in the provider's conversation list to continue.
CREATE TABLE backfill_job_event (
  job               TEXT NOT NULL REFERENCES backfill_job(id),
  n                 INTEGER NOT NULL,       -- 1, 2, ...
  at                INTEGER NOT NULL,
  kind              TEXT NOT NULL,
  by                TEXT,                   -- the extension instance holding the claim, or the operator
  position          TEXT,                   -- the provider's own list position, as the extension reported it
  listed            INTEGER NOT NULL,       -- conversations the provider listed so far
  captured          INTEGER NOT NULL,       -- captured and stored
  unchanged         INTEGER NOT NULL,       -- already in the archive as the provider has them
  failed            INTEGER NOT NULL,       -- could not be captured
  detail            TEXT,
  commit_seq        INTEGER NOT NULL,
  PRIMARY KEY (job, n)
) STRICT, WITHOUT ROWID;
CREATE INDEX backfill_job_event_commit_seq ON backfill_job_event(commit_seq);

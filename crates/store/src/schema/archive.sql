-- The write counter (key 'write_counter'), incremented once per transaction.
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL) STRICT;

-- Machines by /etc/machine-id, with every host name seen with each.
-- `polylogue setting machine <old id>` makes a new machine continue an old
-- identity (moving an archive with the same provider paths); a changed machine
-- ID is otherwise a new machine.
CREATE TABLE machine (
  id            TEXT NOT NULL,           -- /etc/machine-id
  host          TEXT NOT NULL,           -- host name: a label for display, never a key
  first_seen    INTEGER NOT NULL,
  PRIMARY KEY (id, host)
) STRICT, WITHOUT ROWID;

-- The committed end of each append-only segment. Bytes after it are an
-- uncommitted tail, removed by the ordinary writer-open path.
CREATE TABLE segment (
  name          TEXT PRIMARY KEY,
  committed_end INTEGER NOT NULL CHECK(committed_end >= 0)
) STRICT, WITHOUT ROWID;

-- The complete shared extent descriptor is the hash index's byte authority.
CREATE TABLE blob (
  hash          BLOB PRIMARY KEY,
  size          INTEGER NOT NULL,
  stored_size   INTEGER NOT NULL,
  storage       BLOB NOT NULL
) STRICT;

-- One file as Polylogue tracks it: one location's history, or one imported file.
CREATE TABLE input (
  id            TEXT PRIMARY KEY,        -- hex BLAKE3 for an imported file that cannot change;
                                         -- UUIDv7 for a location that can grow or change
  format        TEXT NOT NULL            -- 'claude-code-jsonl', 'codex-rollout', 'codex-state-sqlite', 'chatgpt-export', ...
                                         -- (when it was first seen is a query over its observations, so an input
                                         --  named by content is the same row on every machine)
) STRICT;

-- Every distinct content of an input. Never deleted. The
-- observation that first saw a revision is a query (the earliest by seen_at, then
-- id), not a column, so the rows of an input named by content are identical on
-- every machine.
CREATE TABLE revision (
  input         TEXT NOT NULL REFERENCES input(id),
  number        INTEGER NOT NULL,        -- 1, 2, ... in order observed
  produced      TEXT NOT NULL,           -- first (a location's first revision) | imported | appended | rewritten
  chain         INTEGER NOT NULL,        -- the revision that started this revision chain (itself for first,
                                         --   imported and rewritten revisions)
  length        INTEGER NOT NULL,        -- of the revision's full bytes
  commit_seq    INTEGER NOT NULL,        -- the write counter at the insert
  PRIMARY KEY (input, number)
) STRICT, WITHOUT ROWID;
CREATE INDEX revision_commit_seq ON revision(commit_seq);

-- Where bytes come from: ranges of blobs placed at offsets in a revision chain
-- (appends share their chain's extents; a revision reads the extents covering
-- [0, length)). Every record has a byte range in its chain.
CREATE TABLE extent (
  input         TEXT NOT NULL,
  chain         INTEGER NOT NULL,        -- the revision number that started the chain
  offset        INTEGER NOT NULL,        -- position in the chain's bytes
  length        INTEGER NOT NULL,
  blob          BLOB NOT NULL REFERENCES blob(hash),
  blob_offset   INTEGER NOT NULL,
  commit_seq    INTEGER NOT NULL,
  PRIMARY KEY (input, chain, offset),
  FOREIGN KEY (input, chain) REFERENCES revision(input, number)
) STRICT, WITHOUT ROWID;
CREATE INDEX extent_blob ON extent(blob);

-- Each observation that found something (any class but unchanged). Insert-only.
-- The record of which input a path belongs to;
-- locations, first_seen, source:gone and message.first_seen are queries over it.
-- An input's latest observation is the last one inserted locally (the largest
-- rowid), not the largest seen_at: two changes found in one round share a
-- seen_at. Like commit_seq, that order is local and never compared
-- across archives. Its location is the path of its latest observation at a
-- location path: a filesystem path, 'drive:<file id>' (a Drive file) or
-- 'repo-lookup:<directory>' / 'agentctl-job:<job id>' (records the collector reads).
-- The current chain of an input
-- is the chain of the revision its location's latest observation produced or
-- confirmed, for every kind of location.
CREATE TABLE observation (
  id            TEXT PRIMARY KEY,        -- opaque observation identity
  input         TEXT NOT NULL REFERENCES input(id),
  machine       TEXT NOT NULL,           -- machine.id; paths from two machines never collide
  path          TEXT NOT NULL,           -- filesystem path, 'drive:<file id>', 'repo-lookup:<dir>', 'agentctl-job:<id>',
                                         --   or 'capture:<id>'
  class         TEXT NOT NULL,           -- first | appended | rewritten | deleted | back | imported
  revision      INTEGER,                 -- exact revision endpoint produced or confirmed
  chain         INTEGER,                 -- proven prefix only: revision NULL, length is observed bytes
  length        INTEGER,                 -- actual observed byte length
  device        INTEGER, inode INTEGER, size INTEGER,
  mtime_ns      INTEGER, ctime_ns INTEGER,
  details_json  TEXT,                    -- imported: the import member it was read as, and its format
  polylogue_version TEXT NOT NULL,
  seen_at       INTEGER NOT NULL,        -- Polylogue's clock
  commit_seq    INTEGER NOT NULL DEFAULT 0
) STRICT;
CREATE INDEX observation_input ON observation(input, seen_at);
CREATE INDEX observation_commit_seq ON observation(commit_seq);
CREATE INDEX observation_path ON observation(machine, path, seen_at);

-- Notes, tags, titles, saved queries, withdrawals and everything else people
-- and agents add.
-- One row per client write request: its ID and the hash of its content, so a resent
-- request (same ID, same content) writes nothing and a reused ID with other content is
-- refused as `conflict`. Rows created by the request take their IDs
-- from it. The request itself is kept: replayed in commit order the requests rebuild every row they wrote, which is how user
-- state moves to another archive (`export user-state`).
CREATE TABLE client_request (
  id            TEXT PRIMARY KEY,        -- UUIDv7 chosen by the client, once per user action
  content_hash  BLOB NOT NULL,           -- BLAKE3 of the request's sorted-key JSON, without the ID
  created_at    INTEGER NOT NULL,
  request       BLOB,                    -- the original request as JSON, zstd
  outcome       BLOB,                    -- the original typed outcome, zstd
  effect        BLOB,                    -- the immutable typed authored effect, zstd
  imports       TEXT NOT NULL DEFAULT '{"observed":[],"declared":[]}', -- import evidence and historical claims, JSON
  commit_seq    INTEGER NOT NULL DEFAULT 0  -- the write counter of the transaction that wrote it
) STRICT, WITHOUT ROWID;
CREATE INDEX client_request_commit_seq ON client_request(commit_seq,id);

CREATE TABLE note (
  id            TEXT PRIMARY KEY,        -- the writing request's ID (UUIDv7, chosen by the client), with '.<n>' for
                                         --   its n-th further row
  first_note    TEXT NOT NULL,           -- the first note of its revisions; = id for a first note
  kind          TEXT NOT NULL,           -- tag | note | title | decision | claim | lesson | blocker | question
                                         -- | finding | judgment | outcome | hidden | withdrawal | saved_query
                                         -- | forecast | goal | problem | correction
                                         -- | rubric | distillate | signal
                                         -- | comparison | page | slot | member | ping
  key           TEXT,                    -- key:value tags; a saved query's name
  value         TEXT,                    -- text or JSON; a saved query's text, plus `unit` and
                                         --   `denominator` when it defines a reported figure
  author        TEXT NOT NULL,           -- operator | agent session id | agent | script
  machine       TEXT NOT NULL,           -- where it was written
  created_at    INTEGER NOT NULL,        -- the writer's clock: display and the evidence cutoff only, never orders revisions
  valid_until   INTEGER,                 -- optional: shown as expired after it, left out of briefings
  commit_seq    INTEGER NOT NULL         -- the write counter of the transaction that wrote it
) STRICT;
CREATE INDEX note_commit_seq ON note(commit_seq);              -- the change feed
CREATE INDEX note_first_note ON note(first_note);
CREATE INDEX note_kind ON note(kind, key, value);        -- tag:, outcome:, hidden, query:<name>

-- What a note is about (one or more references).
CREATE TABLE note_target (
  note_id       TEXT NOT NULL REFERENCES note(id),
  ref           TEXT NOT NULL,           -- 'session:...', 'message:...', 'block:...', 'result:...', ...
  PRIMARY KEY (note_id, ref)
) STRICT;
CREATE INDEX note_target_ref ON note_target(ref);       -- notes about X; the hidden filter

-- Typed relations between notes and other things.
CREATE TABLE note_relation (
  note_id       TEXT NOT NULL REFERENCES note(id),
  kind          TEXT NOT NULL,           -- revises | withdraws | answers | challenges | derived_from
                                         -- | blocks | corrects | slot_of | member_of | on_page | subject (pages)
  other         TEXT NOT NULL,           -- a reference
  PRIMARY KEY (note_id, kind, other)
) STRICT;
CREATE INDEX note_relation_other ON note_relation(other, kind);  -- current revision; withdrawn?

-- A query result kept as it was (cited in a note, a briefing or a page).
CREATE TABLE kept_result (
  id            TEXT PRIMARY KEY,        -- UUIDv7; referenced as result:<id>
  name          TEXT,                    -- optional label
  query         TEXT NOT NULL,
  ran_at        INTEGER NOT NULL,
  read_at       TEXT NOT NULL,           -- what its read transaction read (the query's resolved time), as JSON
  rows_json     TEXT NOT NULL            -- row references (notes pinned to a revision), the displayed values,
                                         -- and the match rows of each match
) STRICT;

-- Settings (day.zone, subscription.anthropic, embeddings, ...).
-- The Sinex mode and address live in `$XDG_CONFIG_HOME/polylogue/sinex.toml`
-- instead, so a rebuild from Sinex can start from an empty archive. Insert-only:
-- a change is a new row, so the value in force at any time is kept. Each key has a fixed
-- scope in code: per machine (embeddings, signal.interrupts_per_hour)
-- or archive-wide under machine '*' (subscription.*, which belongs to a provider
-- account; day.logical_start and day.zone, one setting for the operator).
-- A machine reads its own rows for per-machine keys and '*' rows for the
-- rest; the current value is its row with the largest (set_at, id), the value
-- in force at time T the largest (set_at, id) with set_at <= T, and a key with
-- no row has the default.
CREATE TABLE setting (
  id            TEXT PRIMARY KEY,        -- UUIDv7 chosen by the writer
  machine       TEXT NOT NULL,           -- the machine it applies to (as note.machine), or '*' for an archive-wide key
  key           TEXT NOT NULL,
  value         TEXT NOT NULL,
  author        TEXT NOT NULL,           -- 'operator' or the agent session that set it
  set_at        INTEGER NOT NULL,        -- the writer's clock, shown by `polylogue setting`
  commit_seq    INTEGER NOT NULL         -- the write counter at the insert; never a key
) STRICT;
CREATE INDEX setting_current ON setting(machine, key, set_at, id);
CREATE INDEX setting_commit_seq ON setting(commit_seq);

-- Exactly what was put into a model's context, kept because it cannot be
-- recomputed. Prepared, delivered and used are
-- separate facts: a delivery row is what was prepared, with its exact bytes;
-- what happened to it is in delivery_event.
CREATE TABLE delivery (
  id                TEXT PRIMARY KEY,       -- UUIDv7 chosen when prepared; also in the injected reference line
  recipient_kind    TEXT NOT NULL,          -- session | operator
  recipient_id      TEXT,                   -- session id when recipient_kind = session (the session may arrive later)
  sender            TEXT NOT NULL,          -- hook | mcp | signal (model DeliverySender)
  event             TEXT NOT NULL,          -- session_start | resume | compaction | takeover | recall | signal | advice
  cwd               TEXT,                   -- where the recipient was working, when known
  at                INTEGER NOT NULL,       -- prepared at
  content_hash      BLOB NOT NULL,          -- BLAKE3 of the delivered text
  bytes             INTEGER NOT NULL,
  body              BLOB NOT NULL,          -- the delivered text, zstd
  refs_json         TEXT NOT NULL,          -- references included, in order
  left_out_json     TEXT NOT NULL DEFAULT '[]',  -- references and sections left out, each with its reason
  commit_seq        INTEGER NOT NULL
) STRICT;
CREATE INDEX delivery_recipient ON delivery(recipient_id, at);
CREATE INDEX delivery_sender ON delivery(sender, at);   -- signal notices; the evaluator's history
CREATE INDEX delivery_content ON delivery(content_hash);
CREATE INDEX delivery_commit_seq ON delivery(commit_seq);

-- What happened to a prepared delivery: delivered | failed | not_sent | used.
CREATE TABLE delivery_event (
  delivery          TEXT NOT NULL REFERENCES delivery(id),
  what              TEXT NOT NULL CHECK (what IN ('delivered','failed','not_sent','used')),
  at                INTEGER NOT NULL,
  detail            TEXT,                   -- failed/not_sent: the reason; used: the evidence
  request           TEXT NOT NULL,       -- original authored request identity
  commit_seq        INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (delivery, what, at)
) STRICT, WITHOUT ROWID;
CREATE INDEX delivery_event_commit_seq ON delivery_event(commit_seq);
CREATE UNIQUE INDEX delivery_event_request ON delivery_event(delivery,request);

-- Chat actions: turns the operator asks the browser extension to draft or
-- submit in a chat web page (`browser.actions`). The request is stored once;
-- what happened to it is in chat_action_event, and its state is the latest
-- event (an expired claim reads as queued).
CREATE TABLE chat_action (
  id                TEXT PRIMARY KEY,       -- UUIDv7
  request_key       TEXT NOT NULL UNIQUE,   -- the request's idempotency key: a resent request is the same action
  request_hash      BLOB NOT NULL,          -- BLAKE3 of the request's JSON; a reused key with other content is refused
  request_json      TEXT NOT NULL,          -- provider, operation, target, text, presentation and submit policy
  stage_only        INTEGER NOT NULL,       -- 1: fill the composer and stop; a submit is refused
  needs_approval    INTEGER NOT NULL,       -- 1: a submit into an existing conversation waits for the operator
  created_at        INTEGER NOT NULL,
  execution         TEXT NOT NULL DEFAULT 'live' CHECK(execution IN ('live','historical')),
  commit_seq        INTEGER NOT NULL
) STRICT;
CREATE INDEX chat_action_commit_seq ON chat_action(commit_seq);

-- Files the operator attached to a chat action, as blobs.
CREATE TABLE chat_action_attachment (
  action            TEXT NOT NULL REFERENCES chat_action(id),
  id                TEXT NOT NULL,          -- UUIDv7, for the extension to fetch it by
  blob              BLOB NOT NULL,
  name              TEXT NOT NULL,          -- a file name, never a path
  media_type        TEXT NOT NULL,
  size              INTEGER NOT NULL,
  PRIMARY KEY (action, id)
) STRICT, WITHOUT ROWID;
CREATE INDEX chat_action_attachment_blob ON chat_action_attachment(blob);

-- What happened to a chat action, in order: queued | awaiting_approval |
-- approved | declined | claimed | progress, or an outcome the extension
-- reported (drafted, submitted, outcome_unknown, rate_limited, failed, ...).
CREATE TABLE chat_action_event (
  action            TEXT NOT NULL REFERENCES chat_action(id),
  n                 INTEGER NOT NULL,       -- 1, 2, ...
  at                INTEGER NOT NULL,
  kind              TEXT NOT NULL,
  phase             TEXT NOT NULL,          -- the extension's step, as it reported it
  detail            TEXT,
  by                TEXT,                   -- the extension instance holding the claim, or the operator
  receipt_json      TEXT,                   -- the provider's record of the result, as the extension saw it
  request           TEXT NOT NULL,       -- original authored request identity
  commit_seq        INTEGER NOT NULL,
  PRIMARY KEY (action, n)
) STRICT, WITHOUT ROWID;
CREATE INDEX chat_action_event_commit_seq ON chat_action_event(commit_seq);
CREATE UNIQUE INDEX chat_action_event_request ON chat_action_event(action,request);

-- Exact member names and positions in retained container bytes.
CREATE TABLE observation_container (
    observation_id TEXT PRIMARY KEY REFERENCES observation(id),
    container_input TEXT NOT NULL REFERENCES input(id),
    member_index INTEGER NOT NULL CHECK(member_index>=0),
    kind TEXT NOT NULL,
    path_bytes BLOB NOT NULL
) STRICT;
CREATE INDEX observation_container_input ON observation_container(container_input);

-- Backfill jobs: the operator asks the browser extension to capture a web
-- chat's older conversations. The job is stored once; what happened to it is
-- in backfill_job_event, and its state is the latest event (an expired claim
-- reads as waiting).
CREATE TABLE backfill_job (
  id                TEXT PRIMARY KEY,       -- UUIDv7
  provider          TEXT NOT NULL,          -- the web chat, by the extension's name for it (chatgpt, claude-ai, grok)
  since             INTEGER,                -- only conversations the provider changed at or after this time; NULL: all
  created_at        INTEGER NOT NULL,
  execution         TEXT NOT NULL DEFAULT 'live' CHECK(execution IN ('live','historical')),
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
  request           TEXT NOT NULL,       -- original authored request identity
  commit_seq        INTEGER NOT NULL,
  PRIMARY KEY (job, n)
) STRICT, WITHOUT ROWID;
CREATE INDEX backfill_job_event_commit_seq ON backfill_job_event(commit_seq);
CREATE UNIQUE INDEX backfill_job_event_request ON backfill_job_event(job,request);

-- How many inputs and revisions the archive holds and the bytes of its present
-- blobs, kept by the triggers below so that status does not scan the tables.
CREATE TABLE row_count (
  name          TEXT PRIMARY KEY,        -- input | revision | stored_bytes
  n             INTEGER NOT NULL
) STRICT, WITHOUT ROWID;
INSERT INTO row_count(name, n) VALUES ('input', 0), ('revision', 0), ('stored_bytes', 0);
CREATE TRIGGER input_added AFTER INSERT ON input BEGIN UPDATE row_count SET n = n + 1 WHERE name = 'input'; END;
CREATE TRIGGER input_removed AFTER DELETE ON input BEGIN UPDATE row_count SET n = n - 1 WHERE name = 'input'; END;
CREATE TRIGGER revision_added AFTER INSERT ON revision BEGIN UPDATE row_count SET n = n + 1 WHERE name = 'revision'; END;
CREATE TRIGGER revision_removed AFTER DELETE ON revision BEGIN UPDATE row_count SET n = n - 1 WHERE name = 'revision'; END;
CREATE TRIGGER blob_added AFTER INSERT ON blob BEGIN UPDATE row_count SET n = n + new.size WHERE name = 'stored_bytes'; END;
CREATE TRIGGER blob_removed AFTER DELETE ON blob BEGIN UPDATE row_count SET n = n - old.size WHERE name = 'stored_bytes'; END;

-- The write counter (key 'write_counter'), incremented once per transaction,
-- and the archive's identity (key 'archive_id': 32 hex digits, random, written
-- once when the file is created), which every index of this archive records.
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL) STRICT;

-- Machines by /etc/machine-id, with every host name seen with each.
-- `polylogue setting machine <old id>` makes a new machine continue an old
-- identity (moving an archive with the same provider paths); a changed machine
-- ID is otherwise a new machine.
CREATE TABLE machine (
  id            TEXT NOT NULL,           -- /etc/machine-id
  host          TEXT NOT NULL,           -- host name: a label for display, never a key
  first_seen_at INTEGER NOT NULL,
  PRIMARY KEY (id, host)
) STRICT, WITHOUT ROWID;

-- The committed end of each append-only segment, recorded by the put that
-- appended up to it. Bytes after it are the tail of a put a crash stopped,
-- cut when the segment opens (bytes/segments.rs).
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
  id            TEXT PRIMARY KEY,        -- named as its origin declares (InputIdentity): a location is the first
                                         -- 32 hex digits of BLAKE3(machine, 0x00, real path bytes); a Codex rollout
                                         -- is 'codex:rollout:<session id>@<machine>'; an imported file is its full
                                         -- hex BLAKE3
  origin        TEXT NOT NULL            -- the origin's name: 'claude-code', 'codex', 'chatgpt', ...
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
  produced      TEXT NOT NULL CHECK (produced IN (@names(Produced))),
                                         -- first (a location's first revision) | imported | appended | rewritten
  chain         INTEGER NOT NULL,        -- the revision that started this revision chain (itself for first,
                                         --   imported and rewritten revisions)
  length        INTEGER NOT NULL,        -- of the revision's full bytes
  seen_at       INTEGER,                 -- Polylogue's clock when its bytes were read; NULL: unknown
  mtime_ns      INTEGER, ctime_ns INTEGER, -- the file's times at that read; NULL for bytes no file holds
  commit_seq    INTEGER NOT NULL,        -- the write counter at the insert
  PRIMARY KEY (input, number)
) STRICT, WITHOUT ROWID;
CREATE INDEX revision_commit_seq ON revision(commit_seq);

-- One row for each chain, written with its first revision: the earlier chain
-- of the same input whose bytes it begins like for longest (the newest of
-- equals; NULL when it agrees with none) and through which byte, as compared
-- when it was stored. Insert-only. How far any two chains of an input agree
-- is at least the smallest agreement on the path between them.
CREATE TABLE chain (
  input         TEXT NOT NULL,
  number        INTEGER NOT NULL,        -- the revision number that started it
  begins_like   INTEGER,
  agrees_to     INTEGER NOT NULL DEFAULT 0 CHECK (agrees_to >= 0),
  commit_seq    INTEGER NOT NULL,
  PRIMARY KEY (input, number),
  FOREIGN KEY (input, number) REFERENCES revision(input, number)
) STRICT, WITHOUT ROWID;

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

-- Each time an input was found at a location, as the location changed: its
-- first revision there, an import, a rewrite, an append found at another
-- location than the last, its return holding stored bytes, or its
-- disappearance. An append at the location the input was last observed at
-- writes no observation; its revision records it. Insert-only. An input's
-- latest observation is the one with the largest id (ids begin with the write
-- counter and an ordinal in fixed-width hex). Its location is the path bytes
-- of its latest observation: a filesystem path, 'drive:<file id>' (a Drive
-- file) or 'repo-lookup:<directory>' / 'agentctl-job:<job id>' (records the
-- collector reads). The current chain of an input is the chain of the revision
-- its location's latest observation produced or confirmed.
CREATE TABLE observation (
  id            TEXT PRIMARY KEY,        -- write counter, ordinal, UUID and input
  input         TEXT NOT NULL REFERENCES input(id),
  machine       TEXT NOT NULL,           -- machine.id; paths from two machines never collide
  class         TEXT NOT NULL CHECK (class IN (@names(ObservationClass))),
  revision      INTEGER,                 -- the revision it stored or found again
  chain         INTEGER,                 -- back holding a prefix: the chain, with no revision
  length        INTEGER,                 -- the length of that revision or prefix when observed
  device        INTEGER, inode INTEGER, size INTEGER,
  mtime_ns      INTEGER, ctime_ns INTEGER,
  details_json  TEXT,                    -- imported: the import member it was read as, and its format
  polylogue_version TEXT NOT NULL,
  seen_at       INTEGER NOT NULL,        -- Polylogue's clock
  commit_seq    INTEGER NOT NULL DEFAULT 0,
  path_bytes    BLOB NOT NULL            -- the location as read
) STRICT, WITHOUT ROWID;
CREATE INDEX observation_input ON observation(input, seen_at);
CREATE INDEX observation_commit_seq ON observation(commit_seq);
CREATE INDEX observation_path_bytes ON observation(machine,path_bytes,commit_seq);


-- A query result kept as it was (cited in a note, a briefing or a page).
CREATE TABLE saved_rows (
  id            TEXT PRIMARY KEY,        -- UUIDv7; referenced as result:<id>
  name          TEXT,                    -- optional label
  query         TEXT NOT NULL,
  ran_at        INTEGER NOT NULL,
  read_position TEXT NOT NULL,           -- what its read transaction read (the query's resolved time), as JSON
  rows_json     TEXT NOT NULL            -- row references (notes pinned to a revision), the displayed values,
                                         -- and the match rows of each match
) STRICT;

-- Exactly what was put into a model's context, kept because it cannot be
-- recomputed. Prepared, delivered and used are
-- separate facts: a delivery row is what was prepared, with its exact bytes;
-- what happened to it is in delivery_event.
CREATE TABLE delivery (
  id                TEXT PRIMARY KEY,       -- UUIDv7 chosen when prepared; also in the injected reference line
  recipient_kind    TEXT NOT NULL,          -- session | operator
  recipient_id      TEXT,                   -- session id when recipient_kind = session (the session may arrive later)
  sender            TEXT NOT NULL,          -- model DeliverySender
  event             TEXT NOT NULL,          -- model DeliveryEvent
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
  outcome           TEXT NOT NULL CHECK (outcome IN (@names(DeliveryOutcome))),
  at                INTEGER NOT NULL,
  detail            TEXT,                   -- failed/not_sent: the reason; used: the evidence
  request           TEXT NOT NULL,       -- original authored request identity
  commit_seq        INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (delivery, outcome, at)
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
  execution         TEXT NOT NULL DEFAULT 'live' CHECK(execution IN (@names(RunsHere))),
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
  position          INTEGER NOT NULL,       -- 1, 2, ...
  at                INTEGER NOT NULL,
  kind              TEXT NOT NULL,
  phase             TEXT NOT NULL,          -- the extension's step, as it reported it
  detail            TEXT,
  by                TEXT,                   -- the extension instance holding the claim, or the operator
  receipt_json      TEXT,                   -- the provider's record of the result, as the extension saw it
  request           TEXT NOT NULL,       -- original authored request identity
  commit_seq        INTEGER NOT NULL,
  PRIMARY KEY (action, position)
) STRICT, WITHOUT ROWID;
CREATE INDEX chat_action_event_commit_seq ON chat_action_event(commit_seq);
CREATE UNIQUE INDEX chat_action_event_request ON chat_action_event(action,request);

-- Exact member names and positions in retained container bytes.
CREATE TABLE observation_container (
    observation_id TEXT PRIMARY KEY REFERENCES observation(id),
    container_input TEXT NOT NULL REFERENCES input(id),
    member_position INTEGER NOT NULL CHECK(member_position>=0),
    kind TEXT NOT NULL,
    path_bytes BLOB NOT NULL
) STRICT;
CREATE INDEX observation_container_input ON observation_container(container_input);

-- Import jobs: the operator asks the browser extension to capture a web
-- chat's older conversations. The job is stored once; what happened to it is
-- in import_job_event, and its state is the latest event (an expired claim
-- reads as waiting).
CREATE TABLE import_job (
  id                TEXT PRIMARY KEY,       -- UUIDv7
  provider          TEXT NOT NULL,          -- the web chat, by the extension's name for it (chatgpt, claude-ai, grok)
  since             INTEGER,                -- only conversations the provider changed at or after this time; NULL: all
  created_at        INTEGER NOT NULL,
  execution         TEXT NOT NULL DEFAULT 'live' CHECK(execution IN (@names(RunsHere))),
  commit_seq        INTEGER NOT NULL
) STRICT;
CREATE INDEX import_job_commit_seq ON import_job(commit_seq);

-- What happened to an import job, in order: started | claimed | progress |
-- finished | failed | cancelled. Each claimed or progress event carries the
-- totals so far and where in the provider's conversation list to continue.
CREATE TABLE import_job_event (
  job               TEXT NOT NULL REFERENCES import_job(id),
  position          INTEGER NOT NULL,       -- 1, 2, ...
  at                INTEGER NOT NULL,
  kind              TEXT NOT NULL,
  by                TEXT,                   -- the extension instance holding the claim, or the operator
  provider_position TEXT,                   -- the provider's own list position, as the extension reported it
  listed            INTEGER NOT NULL,       -- conversations the provider listed so far
  captured          INTEGER NOT NULL,       -- captured and stored
  unchanged         INTEGER NOT NULL,       -- already in the archive as the provider has them
  failed            INTEGER NOT NULL,       -- could not be captured
  detail            TEXT,
  request           TEXT NOT NULL,       -- original authored request identity
  commit_seq        INTEGER NOT NULL,
  PRIMARY KEY (job, position)
) STRICT, WITHOUT ROWID;
CREATE INDEX import_job_event_commit_seq ON import_job_event(commit_seq);
CREATE UNIQUE INDEX import_job_event_request ON import_job_event(job,request);

-- How many inputs and revisions the archive holds and the bytes of its present
-- blobs, kept by the triggers below so that status does not scan the tables.
CREATE TABLE row_count (
  name          TEXT PRIMARY KEY,        -- input | revision | stored_bytes
  count         INTEGER NOT NULL
) STRICT, WITHOUT ROWID;
INSERT INTO row_count(name, count) VALUES ('input', 0), ('revision', 0), ('stored_bytes', 0);
CREATE TRIGGER input_added AFTER INSERT ON input BEGIN UPDATE row_count SET count = count + 1 WHERE name = 'input'; END;
CREATE TRIGGER input_removed AFTER DELETE ON input BEGIN UPDATE row_count SET count = count - 1 WHERE name = 'input'; END;
CREATE TRIGGER revision_added AFTER INSERT ON revision BEGIN UPDATE row_count SET count = count + 1 WHERE name = 'revision'; END;
CREATE TRIGGER revision_removed AFTER DELETE ON revision BEGIN UPDATE row_count SET count = count - 1 WHERE name = 'revision'; END;
CREATE TRIGGER blob_added AFTER INSERT ON blob BEGIN UPDATE row_count SET count = count + new.size WHERE name = 'stored_bytes'; END;
CREATE TRIGGER blob_removed AFTER DELETE ON blob BEGIN UPDATE row_count SET count = count - old.size WHERE name = 'stored_bytes'; END;

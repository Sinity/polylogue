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

-- Bytes by content. Small blobs inline, large ones as files under blobs/.
CREATE TABLE blob (
  hash          BLOB PRIMARY KEY,        -- BLAKE3 of the uncompressed bytes
  size          INTEGER NOT NULL,        -- uncompressed bytes
  stored_size   INTEGER NOT NULL,        -- compressed bytes
  place         TEXT NOT NULL,           -- inline | file
  encoding      TEXT NOT NULL,           -- plain (seekable zstd) | delta | page_patch
  base          BLOB REFERENCES blob(hash), -- delta: the plain blob it was made against (zstd patch-from, base as prefix)
  state         TEXT NOT NULL,           -- writing | present | deleting
  data          BLOB                     -- zstd bytes, when place = 'inline'
) STRICT;
CREATE INDEX blob_base ON blob(base) WHERE base IS NOT NULL;

-- One logical file: one location's history, or one imported file.
CREATE TABLE input (
  id            TEXT PRIMARY KEY,        -- hex BLAKE3 for an imported file that cannot change;
                                         -- UUIDv7 for a location that can grow or change
  format        TEXT NOT NULL            -- 'claude-code-jsonl', 'codex-rollout', 'codex-state-sqlite', 'chatgpt-export', ...
                                         -- (when it was first seen is a query over its observations, so an input
                                         --  named by content is the same row on every machine)
) STRICT;

-- Every distinct content of an input. Never deleted except by excision. The
-- observation that first saw a revision is a query (the earliest by seen_at, then
-- id), not a column, so the rows of an input named by content are identical on
-- every machine.
CREATE TABLE revision (
  input         TEXT NOT NULL REFERENCES input(id),
  number        INTEGER NOT NULL,        -- 1, 2, ... in order observed
  produced      TEXT NOT NULL,           -- first | imported | appended | head_rewritten | rewritten | snapshot;
                                         --   first: a location's first revision; head_rewritten starts a new chain
                                         --   whose extents are a new first line plus the previous chain's blob
                                         --   ranges after its first line (nothing is copied; storage/blobs.md)
  chain         INTEGER NOT NULL,        -- the revision that started this revision chain (itself for first,
                                         --   imported, head_rewritten, rewritten and snapshot revisions)
  base          INTEGER,                 -- snapshot patch: the plain snapshot revision it overlays
  subsumes      INTEGER,                 -- head_rewritten: the chain it subsumes (storage/blobs.md); a rebuild
                                         --   never reads a subsumed chain, whose records are all in this one
  length        INTEGER NOT NULL,        -- of the revision's full bytes
  blake3        BLOB,                    -- of the full bytes as observed, once complete; never changed,
                                         --   not even by excision (an excised revision is verified by its blobs)
  local_seq     INTEGER NOT NULL,        -- local order (Rules): the write counter at the insert; in `primary` mode
                                         --   the revision event carries it, and Sinex's replicated frontier is the
                                         --   largest W with every revision at or below W complete (joint #29); the event also carries
                                         --   prev_local_seq, the local_seq of the revision stored just before it in
                                         --   this archive's write order (0 for the first), so a gap left by a note or
                                         --   excision is told apart from a missing revision
  PRIMARY KEY (input, number)
) STRICT, WITHOUT ROWID;
CREATE INDEX revision_local ON revision(local_seq);

-- Where bytes come from: ranges of blobs placed at offsets in a revision chain
-- (appends share their chain's extents; a revision reads the extents covering
-- [0, length)) or in a snapshot patch revision (overlaid on its base). Every
-- record has a byte range in its chain. A head-rewritten chain's extents point
-- into the same blobs as the chain before it, shifted by the first line's change
-- in length, so the rewrite stores only the new first line and the appended bytes.
CREATE TABLE extent (
  input         TEXT NOT NULL,
  chain         INTEGER NOT NULL,        -- a revision number: the chain, or the patch revision
  offset        INTEGER NOT NULL,        -- position in the chain's bytes
  length        INTEGER NOT NULL,
  blob          BLOB NOT NULL REFERENCES blob(hash),
  blob_offset   INTEGER NOT NULL,
  PRIMARY KEY (input, chain, offset),
  FOREIGN KEY (input, chain) REFERENCES revision(input, number)
) STRICT, WITHOUT ROWID;
CREATE INDEX extent_blob ON extent(blob);

-- Each observation that found something (any class but unchanged). Insert-only.
-- The authority for which input a path belongs to (blobs.md, observing a file);
-- locations, first_seen, source:gone and message.first_seen are queries over it.
-- An input's latest observation is the last one inserted locally (the largest
-- rowid), not the largest seen_at: a rename and an append found in one round
-- share a seen_at. Like local_seq, that order is local and never compared
-- across archives. Its location is the path of its latest observation at a
-- location path: a filesystem path, 'drive:<file id>' (a Drive file) or
-- 'repo-lookup:<directory>' / 'agentctl-job:<job id>' (records the collector reads);
-- a 'zip:' or 'copy:' observation records where older bytes came from and is
-- never a location (blobs.md, order of chains). The current chain of an input
-- is the chain of the revision its location's latest observation produced or
-- confirmed, for every kind of location.
CREATE TABLE observation (
  id            TEXT PRIMARY KEY,        -- UUIDv7
  input         TEXT NOT NULL REFERENCES input(id),
  machine       TEXT NOT NULL,           -- machine.id; paths from two machines never collide
  path          TEXT NOT NULL,           -- filesystem path, 'drive:<file id>', 'repo-lookup:<dir>', 'agentctl-job:<id>',
                                         --   'zip:<input>/<member>', 'copy:<origin>', or 'capture:<id>'
  stands_for    TEXT,                    -- 'zip:' and 'copy:' observations: the location path these bytes are an
                                         --   older copy of, so the file found there later joins this input
  class         TEXT NOT NULL,           -- first | appended | head_rewritten | replaced | rewritten | renamed
                                         -- | deleted | back | imported | excised | download_attempt
  revision      INTEGER,                 -- the revision produced or confirmed (none for deleted and excised)
  excision      TEXT REFERENCES excision(id),  -- excised: the excision that changed this input's stored bytes
  spans_json    TEXT,                    -- excised: [[blob, offset, length], ...] the overwritten byte spans, by
                                         --   blob range, so every chain whose extents reuse that range sees them
  ids_json      TEXT,                    -- excised: content-derived IDs of the records overlapping a span,
                                         --   computed from the bytes before the overwrite ([[blob, offset, id], ...])
  text_hashes_json TEXT,                 -- excised: BLAKE3 of every text value that held an overwritten span,
                                         --   computed from the bytes before the overwrite
  length        INTEGER,                 -- that revision's length
  device        INTEGER, inode INTEGER, size INTEGER,
  mtime_ns      INTEGER, ctime_ns INTEGER, birth_ns INTEGER,
  mode          INTEGER, owner INTEGER, grp INTEGER,
  symlink_target TEXT,                   -- the path was a symlink: its target
  details_json  TEXT,                    -- download_attempt: URL, outcome, error and retry_at; no content
  provider_version TEXT,                 -- when the content states it
  polylogue_version TEXT NOT NULL,
  zone          TEXT,                    -- the collecting machine's IANA zone (model.md, Time: offsetless times)
  seen_at       INTEGER NOT NULL         -- Polylogue's clock
) STRICT;
CREATE INDEX observation_input ON observation(input, seen_at);
CREATE INDEX observation_path ON observation(machine, path, seen_at);
CREATE INDEX observation_stands_for ON observation(machine, stands_for) WHERE stands_for IS NOT NULL;

-- Notes, tags, titles, saved queries, withdrawals and everything else people
-- and agents add. Kinds and examples: user-data.md.
-- One row per client write request: its ID and the hash of its content, so a resent
-- request (same ID, same content) writes nothing and a reused ID with other content is
-- refused as `conflict` (surfaces/README.md). Rows created by the request take their IDs
-- from it.
CREATE TABLE client_request (
  id            TEXT PRIMARY KEY,        -- UUIDv7 chosen by the client, once per user action
  content_hash  BLOB NOT NULL,           -- BLAKE3 of the request's canonical JSON, without the ID
  created_at    INTEGER NOT NULL
) STRICT, WITHOUT ROWID;

CREATE TABLE note (
  id            TEXT PRIMARY KEY,        -- the writing request's ID (UUIDv7, chosen by the client), with '.<n>' for
                                         --   its n-th further row; for an inline marker, derived from the record's
                                         --   message ID, the marker line and its occurrence (user-data.md)
  root          TEXT NOT NULL,           -- first note of its revision family; = id for a first note
  kind          TEXT NOT NULL,           -- tag | note | title | decision | claim | lesson | blocker | question
                                         -- | finding | judgment | outcome | hidden | withdrawal | saved_query
                                         -- | forecast | goal | problem | correction
                                         -- | rubric | distillate | signal
                                         -- | comparison | page | slot | member | ping (user-data.md)
  key           TEXT,                    -- key:value tags; a saved query's name
  value         TEXT,                    -- text or JSON; a saved query's text, plus `unit` and
                                         --   `denominator` when it defines a reported figure (analysis.md §11)
  check_json    TEXT,                    -- claim only: its check, {"query", "expect": "rows"|"no_rows"|{"count": op, "n"}
                                         --   |{"aggregate": column, "op": op, "n"}}
                                         --   (user-data.md, checked claims); its state is computed at read, never stored
  author        TEXT NOT NULL,           -- operator | agent session id | agent | script (user-data.md)
  machine       TEXT NOT NULL,           -- where it was written
  created_at    INTEGER NOT NULL,        -- the writer's clock: display and the evidence cutoff only, never orders revisions
  valid_until   INTEGER,                 -- optional: shown as expired after it, left out of briefings (user-data.md)
  source        TEXT,                    -- for inline markers: the message it came from
  local_seq     INTEGER NOT NULL         -- local order (Rules); never copied in a union
) STRICT;
CREATE INDEX note_local ON note(local_seq);              -- the change feed
CREATE INDEX note_root ON note(root);
CREATE INDEX note_kind ON note(kind, key, value);        -- tag:, outcome:, hidden, query:<name>

-- What a note is about (one or more references).
CREATE TABLE note_target (
  note_id       TEXT NOT NULL REFERENCES note(id),
  ref           TEXT NOT NULL,           -- 'session:...', 'message:...', 'block:...', 'result:...', ...
  seen_hash     BLOB,                    -- BLAKE3 of the target's text when the note was written
  anchor_input  TEXT,                    -- optional exact anchor for a citation: the input, the revision chain,
  anchor_chain  INTEGER,                 --   the byte range of the cited record in that chain, and the parser
  anchor_start  INTEGER,                 --   version that read it (exact even if the index is rebuilt from
  anchor_end    INTEGER,                 --   another backend; a parser version is not a content version, so
  anchor_parser INTEGER,                 --   the chain is what pins the bytes). Bytes excised since resolve to
                                         --   "excised", never to the overwritten bytes (model.md, Identity)
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
  query_note    TEXT,                    -- the saved_query revision it ran, when it ran one
  ran_at        INTEGER NOT NULL,
  view          TEXT NOT NULL,           -- the view it read (storage/README.md: generation, index counter,
                                         --   archive counter, embeddings high-water id) and up_to_date, as JSON
  rows_json     TEXT NOT NULL            -- row references (notes pinned to a revision), the displayed values,
                                         -- and the match rows of each match (query.md, match rows)
) STRICT;

-- Content removed on purpose (how it works: below). The operator's standing
-- rule, not a work ledger: the loop removes whatever a row names, every round.
CREATE TABLE excision (
  id            TEXT PRIMARY KEY,        -- the request's ID; for the n-th further scope of one request,
                                         --   '<id>.<n>' (as note.id)
  scope         TEXT NOT NULL,           -- 'input:<id>', 'session:<id>', 'text:<blake3>', 'note:<id>'
  reason        TEXT,
  created_at    INTEGER NOT NULL,
  refused_json  TEXT NOT NULL DEFAULT '[]',  -- hashes and places only, never content: the (machine, path,
                                         --   device, inode) locations and the BLAKE3 of every revision and
                                         --   blob the excision removed; collect and store refuse them
  usage_json    TEXT NOT NULL DEFAULT '[]',  -- session scope: usage as numbers only, per (provider account,
                                         --   model, window): message count, tokens by kind (How excision works)
  message_ids_json TEXT NOT NULL DEFAULT '[]',  -- session scope: the excised session's message IDs (IDs only, never
                                         --   content), so copies of them elsewhere are counted nowhere, also after
                                         --   an index rebuild (How excision works)
  applied_upto  TEXT,                    -- text scope: local progress, never copied in a union: the last input
                                         --   ID scanned; 'end' once every input was. A union resets it to NULL.
  local_seq     INTEGER NOT NULL         -- local order (Rules); never copied in a union
) STRICT;

-- Variants: transformed copies of content (model.md, Variant). Content, not
-- claims, so not notes. A new version of a variant is a new row revising the old.
CREATE TABLE variant (
  id            TEXT PRIMARY KEY,        -- UUIDv7
  revises       TEXT REFERENCES variant(id),
  target        TEXT NOT NULL,           -- a reference: session, message, block, span (block:<id>#<start>-<end>) or note
  seen_hash     BLOB NOT NULL,           -- the target text it was made from
  kind          TEXT NOT NULL,           -- translation | transliteration | simplification | summary
  source_lang   TEXT,
  target_lang   TEXT,
  coverage      TEXT NOT NULL,           -- whole | partial
  body          TEXT NOT NULL,           -- the transformed text
  author        TEXT NOT NULL,           -- 'operator' or a session id (the model session that made it)
  machine       TEXT NOT NULL,
  created_at    INTEGER NOT NULL
) STRICT;
CREATE INDEX variant_target ON variant(target);

-- Which span of a variant comes from which span of its source.
CREATE TABLE variant_alignment (
  variant       TEXT NOT NULL REFERENCES variant(id),
  n             INTEGER NOT NULL,
  variant_start INTEGER NOT NULL,        -- Unicode code points in variant.body
  variant_end   INTEGER NOT NULL,
  source_ref    TEXT NOT NULL,           -- a span reference into the target
  relation      TEXT NOT NULL,           -- translates | transliterates | simplifies | summarizes | omits | expands | reorders
  exact         INTEGER NOT NULL,        -- 1 only for an exact copy, whose character map is then exact
  PRIMARY KEY (variant, n)
) STRICT, WITHOUT ROWID;

-- Settings (day.zone, subscription.anthropic, embeddings, ...).
-- The Sinex mode and address live in `$XDG_CONFIG_HOME/polylogue/sinex.toml`
-- instead, so a rebuild from Sinex can start from an empty archive. Insert-only:
-- a change is a new row, so the value in force at any time is kept (analysis.md,
-- billing mode: a plan applies from the time it was set). Each key has a fixed
-- scope in code: per machine (embeddings, signal.interrupts_per_hour)
-- or archive-wide under machine '*' (subscription.*, which belongs to a provider
-- account; day.logical_start and day.zone, one setting for the operator, joint #15).
-- A machine reads its own rows for per-machine keys and '*' rows for the rest; the current value is its row with the largest (set_at, id), the
-- value in force at time T the largest (set_at, id) with set_at <= T (both independent of union
-- arrival order), and a key with no
-- row has the default.
CREATE TABLE setting (
  id            TEXT PRIMARY KEY,        -- UUIDv7 chosen by the writer: globally unique, so a union is plain
  machine       TEXT NOT NULL,           -- the machine it applies to (as note.machine), or '*' for an archive-wide key;
                                         --   a union keeps every machine's rows
  key           TEXT NOT NULL,
  value         TEXT NOT NULL,
  author        TEXT NOT NULL,           -- 'operator' or the agent session that set it (user-data.md, who the author is)
  set_at        INTEGER NOT NULL,        -- the writer's clock, shown by `polylogue setting`
  local_seq     INTEGER NOT NULL         -- local order (Rules): publication only, never a key or an order
) STRICT;
CREATE INDEX setting_current ON setting(machine, key, set_at, id);

-- Exactly what was put into a model's context, kept because it cannot be
-- recomputed (architecture 4.3; old 37t.22). Prepared, delivered and used are
-- separate facts: a delivery row is what was prepared, with its exact bytes;
-- what happened to it is in delivery_event.
CREATE TABLE delivery (
  id                TEXT PRIMARY KEY,       -- UUIDv7 chosen when prepared; also in the injected reference line
  recipient_kind    TEXT NOT NULL,          -- session | operator
  recipient_id      TEXT,                   -- session id when recipient_kind = session (the session may arrive later)
  sender            TEXT NOT NULL,          -- hook | mcp | cli | ping | signal
  event             TEXT NOT NULL,          -- session_start | resume | compaction | takeover | recall | advisory | ping | signal
  cwd               TEXT,                   -- where the recipient was working, when known
  at                INTEGER NOT NULL,       -- prepared at
  content_hash      BLOB NOT NULL,          -- BLAKE3 of the delivered text
  bytes             INTEGER NOT NULL,
  body              BLOB NOT NULL,          -- the delivered text, zstd
  refs_json         TEXT NOT NULL,          -- references included, in order
  left_out_json     TEXT NOT NULL DEFAULT '[]',  -- references and sections left out, each with its reason
  local_seq         INTEGER NOT NULL
) STRICT;
CREATE INDEX delivery_recipient ON delivery(recipient_id, at);
CREATE INDEX delivery_content ON delivery(content_hash);

-- What happened to a prepared delivery: delivered | failed | not_sent | used.
CREATE TABLE delivery_event (
  delivery          TEXT NOT NULL REFERENCES delivery(id),
  what              TEXT NOT NULL CHECK (what IN ('delivered','failed','not_sent','used')),
  at                INTEGER NOT NULL,
  detail            TEXT,                   -- failed/not_sent: the reason; used: the evidence
  PRIMARY KEY (delivery, what, at)
) STRICT, WITHOUT ROWID;

-- Which change notices went where (old 1jc).
CREATE TABLE notice_delivery (
  pk                INTEGER PRIMARY KEY,
  notice            TEXT NOT NULL,
  recipient         TEXT NOT NULL,
  at                INTEGER NOT NULL,
  outcome           TEXT NOT NULL CHECK (outcome IN ('sent','seen','failed'))
) STRICT;

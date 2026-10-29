CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL) STRICT;
-- INDEX_VERSION, generation, cursor_key (the generation's secret that seals read
-- cursors), the write counter, building, facts_version,
-- preamble_set, boot_id (the boot of this generation). A changed boot
-- always starts a new generation, including after a clean shutdown.
-- No cursors for derived products: a product's work is the sessions whose
-- stored version differs.

-- Inputs as index.db sees them: a small local key for each archive.db input.
CREATE TABLE input_ref (
  pk                INTEGER PRIMARY KEY,
  id                TEXT NOT NULL UNIQUE,   -- archive.db input.id
  format            TEXT NOT NULL DEFAULT '' -- parser declaration used for this input
) STRICT;

CREATE TABLE session (
  pk                INTEGER PRIMARY KEY,
  id                TEXT NOT NULL UNIQUE,   -- '<origin>:<native id>'
  origin            TEXT NOT NULL,
  native_id         TEXT NOT NULL,
  kind              TEXT NOT NULL,          -- main | subagent | sidechain | compaction (continuations are links, not a kind)
  stub              INTEGER NOT NULL DEFAULT 0,  -- 1: created by a hook line naming a session that has no
                                            --   transcript yet, so its live state exists; cleared when its messages arrive
  title             TEXT,                   -- provider title only
  title_source      TEXT,                   -- title provenance; precedence is owned by model TitleSource
  derived_title     TEXT,                   -- derived (title): first line of the first typed prompt; shown when
                                            --   title is NULL, never exported as recorded
  cwd_json          TEXT,                   -- every working directory in order (model Session.cwd); `cwd` is the last
  started_time      BLOB NOT NULL DEFAULT X'5b6e756c6c2c7b227175616c697479223a22756e6b6e6f776e227d5d',
  started_at        INTEGER,                -- from the messages and events the session counts as its own
  last_activity_at  INTEGER,
  cwd               TEXT,
  repo              TEXT,                   -- name or URL, for display: record_repo, else the remote of the
                                            --   repo_lookup holding the session's cwd (derived: repository)
  mode              TEXT,                   -- harness mode as recorded, e.g. background
  record_repo       TEXT,                   -- the repository the session's own records name
  repo_root         TEXT,                   -- the repository's identity, its root commit hash, from repo_lookup (derived)
  branch            TEXT,
  git_commit        TEXT,
  model_provider    TEXT,
  instructions      TEXT,
  harness           TEXT,
  harness_version   TEXT,
  message_count INTEGER,
  human_prompt_count INTEGER,
  human_command_count INTEGER,
  word_count INTEGER,
  human_word_count INTEGER,
  model_word_count INTEGER,
  thinking_count INTEGER,
  compaction_count INTEGER,
  tool_call_count INTEGER,
  error_count INTEGER,
  -- Live facts: updated in the session's write transaction
  -- when a hook line for it is written, so they follow data, not
  -- the clock; the live state itself is computed at read time from these and now.
  live_last_event   TEXT,                   -- type of the latest hook event
  live_last_event_at INTEGER,
  live_open_tool_pk INTEGER,                -- tool_call with a pre-tool event and no post-tool event yet
  live_waiting_on   TEXT,                   -- permission | question | notification | NULL
  live_waiting_since INTEGER,
  live_todo         TEXT,                   -- the harness's in-progress todo item, as text (Claude Code: the last TodoWrite)
  live_task         TEXT,                   -- job ID or Bead ID
  live_task_evidence TEXT,                  -- job_env | claim_command
  changed           INTEGER NOT NULL,       -- write counter of the last change to its rows or derived values
                                            --   or derived products (not its live columns)
  live_reparse      INTEGER NOT NULL DEFAULT 0,
  live_changed      INTEGER                 -- write counter value of the last change to its live columns
) STRICT;
CREATE INDEX session_activity ON session(last_activity_at);   -- default order, since:
CREATE INDEX session_started  ON session(started_at);         -- started between, until:
CREATE INDEX session_changed  ON session(changed);            -- change feed; derive step
CREATE INDEX session_cwd      ON session(cwd);                -- cwd: prefix as a range; briefings; resume
CREATE INDEX session_repo     ON session(repo) WHERE repo IS NOT NULL;
CREATE INDEX session_live ON session(live_last_event_at) WHERE live_last_event_at IS NOT NULL;   -- running:, waiting:, the overview
CREATE INDEX session_repo_root ON session(repo_root) WHERE repo_root IS NOT NULL;   -- repo: by identity

CREATE TABLE message (
  pk                INTEGER PRIMARY KEY,
  id                TEXT NOT NULL,          -- unique: message_id
  session_pk        INTEGER NOT NULL REFERENCES session(pk),
  order_input_pk    INTEGER NOT NULL,       -- source of the ordering record; index-local input_ref key
  order_input_id    TEXT NOT NULL,          -- stable input identity in the order key, independent of insertion order
  seq               INTEGER NOT NULL,       -- byte offset of the first record in that input
  record_position   INTEGER NOT NULL DEFAULT 0, -- its place among the messages that record holds
  arrival_rank        INTEGER NOT NULL DEFAULT 0, -- the shown version's arrival: 0 live, 1 spool, 2 import
  role              TEXT NOT NULL,          -- user | assistant | system | tool (sources/ maps each provider's role)
  written_by        TEXT NOT NULL,          -- human | model | harness | unknown
  prompt_kind       TEXT,                   -- written_by human: prose | command
  model             TEXT,
  at_time           BLOB NOT NULL DEFAULT X'5b6e756c6c2c7b227175616c697479223a22756e6b6e6f776e227d5d',
  at                INTEGER,                -- provider time
  parent_id         TEXT,                   -- natural message id of the tree parent (may arrive later)
  input_pk          INTEGER NOT NULL REFERENCES input_ref(pk),  -- input the stored (shown) version came from
  source_chain      INTEGER,                -- the revision chain the byte range is in (archive.db revision.chain):
                                            --   offsets are stable only within a chain, which only appends
  source_start      INTEGER,                -- byte range of its records in that chain (export `sources[0]`):
  source_end        INTEGER,                --   raw view and outside anchors; for a JSON export, the conversation object
  in_progress       INTEGER NOT NULL DEFAULT 0, -- the provider marked this version as still being produced
  on_active_path    INTEGER,                -- 1 on the path, 0 off, NULL unknown (no marked leaf);
                                            --   always written, no default
  duplicate_of      TEXT,                   -- derived (duplicates): natural message id of another input's message the
                                            --   session shows for the same turn (no shared turn ID, matched by prompt text)
  duplicate_reason  TEXT,                   --   and why that input's message is the one shown
  attrs_json        TEXT,                   -- provider fields kept as given with no column of their own
                                            --   (attribution, effort, Codex phase, citations, speech or
                                            --   recap markers); one JSON object, export field `attrs`
  input_tokens      INTEGER,                -- NULL = not reported (never 0 for unknown)
  output_tokens     INTEGER,
  cache_read_tokens INTEGER,
  cache_write_tokens INTEGER,               -- all cache writes
  cache_write_5m_tokens INTEGER,            -- the part written with a 5-minute lifetime, when reported
  cache_write_1h_tokens INTEGER,            -- the part written with a 1-hour lifetime, when reported
  reasoning_tokens  INTEGER,
  speed             TEXT,                   -- e.g. 'fast', when the provider reports a mode priced differently
  service_tier      TEXT,                   -- batch | flex | priority | ..., when recorded
  provider_cost_nusd INTEGER,               -- the provider's own cost figure, when given
  cost_nusd         INTEGER,                -- the one per-message cost; NULL = unknown
  stop_reason       TEXT,
  duration_ms       INTEGER,
  words             INTEGER NOT NULL DEFAULT 0, -- words in its text blocks
  UNIQUE (pk, session_pk)                   -- lets children state their owner once
) STRICT;
-- A first build drops the three unique indexes on natural keys and builds
-- them once at its end; meanwhile the writer finds rows by key in memory.
CREATE UNIQUE INDEX message_id ON message(id);
CREATE INDEX message_session ON message(session_pk, order_input_id, seq, record_position, id);   -- reading in order; [last:N]
CREATE INDEX message_prose   ON message(session_pk) WHERE written_by = 'human' AND prompt_kind = 'prose';
CREATE INDEX message_branch  ON message(session_pk) WHERE on_active_path = 0;

-- Every stored version that contributed to a message: one row per input chain
-- whose records hold a version of it (the shown version's row repeats the
-- message's own columns). A citation names a row here, never just an input and
-- offsets, because a rewritten input's new chain puts other bytes at the same
-- offsets (an outside review, 2026-10-12). Rebuilt with the message.
CREATE TABLE message_source (
  message_pk        INTEGER NOT NULL REFERENCES message(pk),
  input_pk          INTEGER NOT NULL REFERENCES input_ref(pk),
  chain             INTEGER NOT NULL,       -- archive.db revision.chain
  source_start      INTEGER NOT NULL,
  source_end        INTEGER NOT NULL,       -- equal to source_start until parsers report record ends
  shown         INTEGER NOT NULL,       -- 1 for the version the message shows
  session_pk        INTEGER NOT NULL REFERENCES session(pk),
  main              INTEGER NOT NULL DEFAULT 1,
  arrival_rank      INTEGER NOT NULL DEFAULT 0,
  seq               INTEGER NOT NULL DEFAULT 0,
  record_position   INTEGER NOT NULL DEFAULT 0,
  complete          INTEGER NOT NULL DEFAULT 1, -- 0 while the source marks it in progress (no stop reason yet)
  version_hash      BLOB NOT NULL REFERENCES message_version(version_hash), -- BLAKE3 of the version's sorted-key JSON: the last tie-break
  PRIMARY KEY (message_pk, input_pk, chain)
) STRICT;
CREATE INDEX message_source_input ON message_source(input_pk, chain);   -- foreign-key check

-- Text is stored once per distinct content; `id` is the rowid of its search entries.
CREATE TABLE text (
  id                INTEGER PRIMARY KEY,
  hash              BLOB NOT NULL,          -- unique: text_hash. BLAKE3 of the UTF-8 text; embeddings are keyed by it
  -- Keep membership before the body: testing it must not read body overflow pages.
  indexed           INTEGER NOT NULL DEFAULT 0, -- bit 1: in text_fts (prose); bit 2: in output_fts (tool output)
  body              BLOB NOT NULL           -- the UTF-8 text, zstd-compressed (the search indexes are
                                            --   contentless, so this is the only copy in the index)
) STRICT;

-- What the provider wrote, one row per content item. Facts derived about a tool
-- call (tool, command, outcome, exit code, duration) belong to tool_call alone.
CREATE TABLE block (
  pk                INTEGER PRIMARY KEY,
  message_pk        INTEGER NOT NULL,
  session_pk        INTEGER NOT NULL,       -- the message's session, enforced by the composite key below
  idx               INTEGER,                -- compact position among the stored blocks, used for ordering
  kind              TEXT NOT NULL,          -- text | thinking | tool_call | tool_result | image | file | other
  text_id           INTEGER REFERENCES text(id),
  body              BLOB NOT NULL,          -- the model Block as zstd-compressed JSON, with its prose or output
                                            --   text moved to `text` (text_id): reading a block back is lossless
  call_id           TEXT,                   -- tool_call and tool_result: the provider's call id (unique only in a session)
  blob_hash         BLOB,                   -- image/file, large tool output: bytes in the blob store; export `blob`
  direction         TEXT,                   -- image/file: input | output | unknown
  media_type        TEXT,
  file_name         TEXT,                   -- export `name`
  size              INTEGER,                -- image/file
  url               TEXT,                   -- image/file downloaded from a link
  signature         TEXT,                   -- thinking
  encrypted         INTEGER NOT NULL DEFAULT 0,  -- thinking kept encrypted
  other_type        TEXT,                   -- other: the provider's content type name; export `type`
  data              BLOB,                   -- tool_result: its structured data as zstd-compressed JSON, kept
                                            --   out of body, so a text read never reads or decodes it
  ordinal           TEXT,                   -- original u64 source position; NULL means native dense idx
  FOREIGN KEY (message_pk, session_pk) REFERENCES message(pk, session_pk),
  UNIQUE (pk, session_pk)
) STRICT;
CREATE UNIQUE INDEX text_hash ON text(hash);
CREATE UNIQUE INDEX block_message ON block(message_pk, idx);
CREATE INDEX block_text   ON block(text_id) WHERE text_id IS NOT NULL;   -- search hit -> blocks

-- Full-text search: two contentless indexes keyed by text.id, holding folded
-- text. text_fts: prose (text and thinking blocks). output_fts: tool output
-- (tool_result blocks, and bodies written through file-writing and file-editing
-- tool calls). Rules below.
CREATE VIRTUAL TABLE text_fts USING fts5(body, content='', contentless_delete=1,
                                         tokenize='unicode61 remove_diacritics 2');
CREATE VIRTUAL TABLE output_fts USING fts5(body, content='', contentless_delete=1,
                                           tokenize='unicode61 remove_diacritics 2');
-- Document frequency of each word, for the search planner.
CREATE VIRTUAL TABLE fts_vocab_text USING fts5vocab(text_fts, 'row');
CREATE VIRTUAL TABLE fts_vocab_output USING fts5vocab(output_fts, 'row');
-- Merge segments less eagerly than the default (4), and buffer 64 MiB of
-- pending terms before writing a segment instead of the default 1 MiB: with
-- the default, flushing tiny segments and merging them was over half of the
-- writer's time on a first build (perf, 2026-10-19). The buffer is writer
-- memory, at most 64 MiB per index.
INSERT INTO text_fts(text_fts, rank) VALUES ('automerge', 16);
INSERT INTO output_fts(output_fts, rank) VALUES ('automerge', 16);
INSERT INTO text_fts(text_fts, rank) VALUES ('hashsize', 67108864);
INSERT INTO output_fts(output_fts, rank) VALUES ('hashsize', 67108864);

-- Search entries still to write. The writer records a
-- text here in the transaction that stores it, and the search thread writes
-- the entry shortly after, outside the writer's transaction; a flush writes
-- every waiting entry before it returns. A first build writes none: it
-- builds the search indexes in bulk when it ends.
CREATE TABLE search_pending (
  text_id           INTEGER NOT NULL,       -- text.id
  fts_table         INTEGER NOT NULL,       -- which index: 1 text_fts, 2 output_fts (the bits of text.indexed)
  PRIMARY KEY (text_id, fts_table)
) STRICT, WITHOUT ROWID;

-- One row per tool call, paired with its result inside the same session. The
-- one owner of every derived call fact.
CREATE TABLE tool_call (
  pk                INTEGER PRIMARY KEY,
  session_pk        INTEGER NOT NULL,       -- kept for index speed; the composite keys make it agree with its blocks
  call_block_pk     INTEGER NOT NULL UNIQUE,
  call_id           TEXT,                   -- the provider's call id: how a later result finds its call
  result_at         INTEGER,
  reported_duration_ms INTEGER,
  result_block_pk   INTEGER,                -- NULL until the result is seen
  at                INTEGER,
  tool              TEXT NOT NULL,
  action            TEXT,                   -- the parser's ToolAction: shell | code | file_read | file_write | file_edit | search | web | subagent | mcp | other
  command           TEXT,                   -- as given
  command_norm      TEXT,                   -- derived (tool_calls): 'cargo test', 'git commit'
  path              TEXT,                   -- file path argument, when the tool has one
  outcome           TEXT NOT NULL,          -- ok | error | unknown: facts of the records, never of the clock;
                                            --   `pending` is computed at read time from live state
  outcome_reason    TEXT,                   -- a reason code (no_result, background_never_reported, ...)
  exit_code         INTEGER,
  run_by_script     INTEGER NOT NULL DEFAULT 0, -- a command a script call ran, with no call of its own
  FOREIGN KEY (call_block_pk, session_pk) REFERENCES block(pk, session_pk),
  FOREIGN KEY (result_block_pk, session_pk) REFERENCES block(pk, session_pk)
) STRICT;
CREATE INDEX tool_call_session ON tool_call(session_pk, at);   -- exists, seq(...), the changes view
CREATE INDEX tool_call_pairing ON tool_call(session_pk, call_id) WHERE call_id IS NOT NULL;   -- result -> call
CREATE INDEX block_call ON block(session_pk, call_id) WHERE call_id IS NOT NULL;   -- call -> earlier result
CREATE INDEX tool_call_tool    ON tool_call(tool);
CREATE INDEX tool_call_action  ON tool_call(action, outcome);
CREATE INDEX tool_call_failed  ON tool_call(session_pk) WHERE outcome <> 'ok';
CREATE INDEX tool_call_path    ON tool_call(path) WHERE path IS NOT NULL;
CREATE INDEX tool_call_repo_command ON tool_call(command_norm, at) WHERE command_norm IS NOT NULL;   -- failing commands
CREATE INDEX tool_call_result  ON tool_call(result_block_pk) WHERE result_block_pk IS NOT NULL;   -- foreign-key check

-- Substring search on short fields (command:, path:) without a scan. Unfolded
-- text, so it can read its content from tool_call. The writer issues the FTS5
-- `delete` command with the row's old `command` and `path` before deleting or
-- changing a `tool_call` row, in the same transaction. A first build writes
-- no entries and rebuilds the index from tool_call when it ends.
CREATE VIRTUAL TABLE tool_call_trigram USING fts5(command, path,
    content='tool_call', content_rowid='pk', tokenize='trigram');
INSERT INTO tool_call_trigram(tool_call_trigram, rank) VALUES ('automerge', 16);
INSERT INTO tool_call_trigram(tool_call_trigram, rank) VALUES ('hashsize', 67108864);


CREATE TABLE link (
  pk                INTEGER PRIMARY KEY,
  parent_id         TEXT NOT NULL,          -- natural session id (may not be ingested yet)
  child_id          TEXT NOT NULL,          -- natural session id of the child (may not be ingested yet: a parent's
                                            --   thread_spawn_edges, continued-in or job children name it first)
  child_pk          INTEGER REFERENCES session(pk),  -- set once the child is stored; NULL while the link is pending
  kind              TEXT NOT NULL,          -- subagent | continuation | reference
  how               TEXT NOT NULL,          -- subagent: fresh | copied | summarized; continuation: resume | fork
                                            -- | after_compaction; reference: briefing | ping | mention
  branch_point      TEXT,                   -- natural message id where the child branches off
  spawning_call     TEXT,                   -- the spawning call's ID as the evidence names it (a tool-use ID)
  evidence          TEXT NOT NULL,          -- which input field established the parent
  source_chain      INTEGER NOT NULL DEFAULT 0,
  input_pk          INTEGER NOT NULL REFERENCES input_ref(pk),  -- the input holding that field
  UNIQUE (parent_id, child_id, kind, how)
) STRICT;
CREATE INDEX link_child  ON link(child_pk);                    -- parents
CREATE INDEX link_child_id ON link(child_id) WHERE child_pk IS NULL;   -- late children
CREATE INDEX link_branch_point ON link(branch_point) WHERE branch_point IS NOT NULL;
CREATE INDEX link_parent ON link(parent_id, kind);             -- children; late parents

CREATE TABLE event (
  pk                INTEGER PRIMARY KEY,
  id                TEXT NOT NULL,          -- unique: event_id. Computed by the parser from the raw record; an
                                            --   event that names no session is keyed by its input record instead
  session_pk        INTEGER REFERENCES session(pk),  -- NULL for an event about its input, not a session
  seq               INTEGER NOT NULL,       -- display position (shares numbering with message.seq); not a key
  after_message_id  TEXT,
  seq_from_message  INTEGER NOT NULL DEFAULT 0,
  after_message_pk  INTEGER REFERENCES message(pk),  -- the message it follows; export `after_message`
  kind              TEXT NOT NULL,          -- open set; each provider's kinds are listed in sources/
  at                INTEGER,
  input_pk          INTEGER NOT NULL REFERENCES input_ref(pk),  -- the input whose record it came from
  source_chain      INTEGER,                -- revision chain of the byte range (as message.source_chain)
  source_start      INTEGER,
  source_end        INTEGER,
  fields            BLOB NOT NULL,          -- canonical Event without its shared data payload
  data              BLOB NOT NULL           -- the provider's payload as given, zstd-compressed JSON; export `data`
) STRICT;
CREATE UNIQUE INDEX event_id ON event(id);
CREATE INDEX event_session ON event(session_pk, seq);
CREATE INDEX event_input   ON event(input_pk, source_chain);
CREATE INDEX event_after   ON event(after_message_pk) WHERE after_message_pk IS NOT NULL;
CREATE INDEX event_after_id ON event(after_message_id) WHERE after_message_id IS NOT NULL;
CREATE INDEX event_kind    ON event(kind, at);





-- Which inputs produced which sessions.
CREATE TABLE parsed (
  input_pk          INTEGER NOT NULL REFERENCES input_ref(pk),
  chain             INTEGER NOT NULL,       -- the revision chain parsed
  session_pk        INTEGER NOT NULL REFERENCES session(pk),
  PRIMARY KEY (input_pk, chain, session_pk)
) STRICT, WITHOUT ROWID;
CREATE INDEX parsed_session ON parsed(session_pk);             -- rebuild a session from all its inputs

-- How far each revision chain has been parsed, and how it ended. Every chain of every input has a row once the loop has seen it,
-- so nothing is silently skipped.
CREATE TABLE parse_state (
  input_pk          INTEGER NOT NULL REFERENCES input_ref(pk),
  chain             INTEGER NOT NULL,
  reading_version   TEXT NOT NULL DEFAULT '',
  status            TEXT NOT NULL,          -- parsing: more stored bytes need parsing; waiting: incomplete record at
                                            --   observed EOF; done: at the stored length
  parsed_upto       INTEGER NOT NULL,       -- byte offset in the chain after the last complete record parsed
  detail            TEXT,                   -- waiting: the stored length it waits at
  records           INTEGER NOT NULL DEFAULT 0,  -- every record of this chain read so far:
  used              INTEGER NOT NULL DEFAULT 0,  --   records = used + skipped + unreadable; an append adds
  skipped           INTEGER NOT NULL DEFAULT 0,  --   to the counts, a whole parse replaces them; written in the
  unreadable        INTEGER NOT NULL DEFAULT 0,  --   same transaction as the chain's last session
  commit_seq INTEGER CHECK(commit_seq > 0), -- actual committed parse checkpoint, never a read position
  PRIMARY KEY (input_pk, chain)
) STRICT, WITHOUT ROWID;
CREATE INDEX parse_state_status ON parse_state(status) WHERE status <> 'done';   -- status, waiting work

-- What a parser did not use, per input: records it skipped (known kinds it
-- counts by design, listed in sources/), records it could not read, and wire
-- fields it did not read. An append adds to the input's rows, a whole
-- re-parse replaces them.
CREATE TABLE unused (
  input_pk          INTEGER NOT NULL REFERENCES input_ref(pk),  -- also the example input status shows
  chain             INTEGER NOT NULL,       -- the revision chain counted; a whole re-parse of a chain replaces its rows
  format            TEXT NOT NULL,
  what              TEXT NOT NULL,          -- 'skipped:<kind>: <reason>', 'unreadable:<kind>: <reason>' or
                                            --   'field:<kind>.<json path>'
  count             INTEGER NOT NULL,
  counted_only      INTEGER NOT NULL DEFAULT 0,  -- 1: skipped, a known kind kept only as a count by design
  PRIMARY KEY (input_pk, chain, what)
) STRICT, WITHOUT ROWID;


-- Sessions removed without a surviving row (reading replacement), for the change
-- feed. Emptied only when a new generation starts.
CREATE TABLE removed_session (
  session_id        TEXT PRIMARY KEY,
  changed           INTEGER NOT NULL        -- write counter of the removing transaction
) STRICT;
CREATE INDEX removed_session_changed ON removed_session(changed);

-- Git correlation: commits and pull requests linked to sessions
-- by evidence in the session. Commit details come from stored git-lookup
-- inputs, never from reading a repository at derive time.
CREATE TABLE commit_link (
  session_pk        INTEGER NOT NULL REFERENCES session(pk),
  commit_hash       TEXT NOT NULL,          -- full hash; the one place the commit is a key, so it is named
                                            --   commit_hash here and `commit` everywhere else
  evidence_block_pk INTEGER REFERENCES block(pk) ON DELETE CASCADE,  -- the git commit tool call; NULL for a
                                            --   trailer. Derived: a rewrite of the call removes it, and the
                                            --   derive step makes it again
  evidence          TEXT NOT NULL,          -- commit-output | trailer | codex-git
  PRIMARY KEY (session_pk, commit_hash)
) STRICT, WITHOUT ROWID;
CREATE INDEX commit_link_hash ON commit_link(commit_hash);      -- commit:
CREATE INDEX commit_link_block ON commit_link(evidence_block_pk) WHERE evidence_block_pk IS NOT NULL;  -- the foreign-key check when blocks are deleted

CREATE TABLE pr_link (
  session_pk        INTEGER NOT NULL REFERENCES session(pk),
  repo              TEXT NOT NULL,
  number            INTEGER NOT NULL,
  url               TEXT,
  evidence          TEXT NOT NULL,          -- pr-link-record | gh-output | codex
  PRIMARY KEY (session_pk, repo, number)
) STRICT, WITHOUT ROWID;
CREATE INDEX pr_link_number ON pr_link(number);                 -- pr:


-- One per repository directory a session worked in, from the
-- polylogue-repo-lookup inputs the engine writes. A session belongs to the lookup whose path is the
-- longest one at or above its cwd; the writer copies the identity onto the
-- session (repo_root, and repo where no record names one).
CREATE TABLE repo_lookup (
  path              TEXT PRIMARY KEY,       -- the directory holding .git
  remote            TEXT,                   -- origin's URL without credentials; NULL when there is none
  root_commit       TEXT,                   -- NULL when it could not be read (shallow clone)
  input_pk          INTEGER NOT NULL REFERENCES input_ref(pk),
  chain             INTEGER NOT NULL
) STRICT, WITHOUT ROWID;




-- The work graph, derived from structure
-- only: goals are what a session declared it
-- would do (a Bead claimed by a `bd` command, a todo or plan item), attempts
-- are a session pursuing a goal, verifications are check commands with their
-- outcomes, artifacts are commits and pull requests. A claim, an observed
-- effect and an evaluation stay apart: a closed Bead is a claim; the checks
-- and commits beside it are the effects. Rebuilt per session by the derive
-- step; a Bead claimed by several sessions is one goal node.
CREATE TABLE work_node (
  pk                INTEGER PRIMARY KEY,
  key               TEXT NOT NULL UNIQUE,   -- bead:<id> | todo:<session>:<item> | attempt:<session>:<goal key>
                                            --   | check:<tool_call pk> | commit:<session>:<hash> | pr:<session>:<repo>#<n>
  kind              TEXT NOT NULL CHECK (kind IN ('goal','attempt','verification','artifact')),
  title             TEXT NOT NULL,
  session_pk        INTEGER REFERENCES session(pk),  -- the session that made it; NULL for a shared goal
  at                INTEGER,
  state             TEXT,                   -- attempt: open | closed | released | completed | pending;
                                            --   verification: the call's outcome
  tool_call_pk      INTEGER REFERENCES tool_call(pk) ON DELETE CASCADE,  -- the call that is its
                                            --   evidence; a rewrite of the call removes the node until the
                                            --   derive step makes it again, so derived rows never block a write
  attrs_json        TEXT                    -- attempt: checks passed and failed and commits while it was open
) STRICT;
CREATE INDEX work_node_session ON work_node(session_pk);
CREATE INDEX work_node_kind ON work_node(kind, at);
CREATE INDEX work_node_verification ON work_node(kind, title, at);   -- one verification's runs, for advice before a command runs
CREATE INDEX work_node_call ON work_node(tool_call_pk) WHERE tool_call_pk IS NOT NULL;   -- foreign-key check

CREATE TABLE work_edge (
  -- An edge goes with either node, so replacing a session's nodes
  -- by `session_pk` needs no rule of its own for edges.
  from_pk           INTEGER NOT NULL REFERENCES work_node(pk) ON DELETE CASCADE,
  to_pk             INTEGER NOT NULL REFERENCES work_node(pk) ON DELETE CASCADE,
  kind              TEXT NOT NULL CHECK (kind IN ('pursues','checks','produces')),
  PRIMARY KEY (from_pk, to_pk, kind)
) STRICT, WITHOUT ROWID;
CREATE INDEX work_edge_to ON work_edge(to_pk);

-- Which session state each session's work rows were derived from: the derive
-- step's work is the sessions whose version or `changed` differs.
-- What each input states about a session (model Session, as JSON), merged
-- within the input in file order. The session's own columns are derived
-- from all of them (derive/session.rs), never from arrival order.
CREATE TABLE session_record (
  session_pk        INTEGER NOT NULL REFERENCES session(pk) ON DELETE CASCADE,
  input_pk          INTEGER NOT NULL REFERENCES input_ref(pk),
  chain             INTEGER NOT NULL,
  facts             TEXT NOT NULL,
  admitted INTEGER CHECK(admitted > 0),
  PRIMARY KEY (session_pk, input_pk, chain)
) STRICT, WITHOUT ROWID;
CREATE INDEX session_record_input ON session_record(input_pk);   -- foreign-key check

-- Which derivation (derive.rs) made a session's derived rows, at which
-- version and from which state of the session; a session is derived again
-- when either moves.
CREATE TABLE derive_state (
  name              TEXT NOT NULL,
  session_pk        INTEGER NOT NULL REFERENCES session(pk) ON DELETE CASCADE,
  version           TEXT NOT NULL,
  from_changed      INTEGER NOT NULL,
  PRIMARY KEY (name, session_pk)
) STRICT, WITHOUT ROWID;
CREATE INDEX derive_state_session ON derive_state(session_pk);

-- Clocks and interval calculations over one session's parsed rows.
CREATE TABLE session_span (
  session_pk INTEGER PRIMARY KEY REFERENCES session(pk) ON DELETE CASCADE,
  first_at INTEGER,
  last_at INTEGER,
  time_source TEXT NOT NULL,
  wall_ms INTEGER,
  clustered_ms INTEGER,
  tool_sum_ms INTEGER,
  tool_union_ms INTEGER,
  tool_overlap_ms INTEGER,
  tool_untimed INTEGER NOT NULL,
  tool_no_result INTEGER NOT NULL,
  tool_same_timestamp INTEGER NOT NULL,
  tool_reversed INTEGER NOT NULL,
  timed_messages INTEGER NOT NULL,
  untimed_messages INTEGER NOT NULL,
  unordered_gaps INTEGER NOT NULL,
  first_failure_at INTEGER,
  prompts_to_failure INTEGER NOT NULL,
  prompted INTEGER NOT NULL,
  ends_with TEXT NOT NULL,
  ends_with_message TEXT,
  last_stop_reason TEXT
) STRICT;
CREATE INDEX session_span_wall ON session_span(wall_ms);
CREATE INDEX session_span_clustered ON session_span(clustered_ms);
CREATE INDEX session_span_ending ON session_span(ends_with);

CREATE TABLE session_time (
  session_pk INTEGER NOT NULL REFERENCES session(pk) ON DELETE CASCADE,
  lane TEXT NOT NULL CHECK(lane IN ('tool','model','human','unattributed')),
  key TEXT NOT NULL,
  ms INTEGER NOT NULL CHECK(ms >= 0),
  PRIMARY KEY(session_pk,lane,key)
) STRICT, WITHOUT ROWID;

-- Text that is not a conversation (claude.ai documents, for example), with the
-- same shown-version rule as messages and interned text.
CREATE TABLE document (
  pk INTEGER PRIMARY KEY,
  id TEXT NOT NULL UNIQUE,
  origin TEXT NOT NULL,
  native_id TEXT NOT NULL,
  purpose TEXT NOT NULL,
  title TEXT,
  source_time INTEGER,
  authorship TEXT NOT NULL,
  version_hash BLOB NOT NULL,
  input_pk INTEGER NOT NULL REFERENCES input_ref(pk),
  source_chain INTEGER NOT NULL,
  source_start INTEGER NOT NULL,
  source_end INTEGER NOT NULL,
  fields BLOB NOT NULL, -- the shown version without its blocks, zstd JSON
  changed INTEGER NOT NULL
) STRICT;
CREATE TABLE document_source (
  document_pk INTEGER NOT NULL REFERENCES document(pk) ON DELETE CASCADE,
  input_pk INTEGER NOT NULL REFERENCES input_ref(pk),
  chain INTEGER NOT NULL,
  source_start INTEGER NOT NULL,
  source_end INTEGER NOT NULL,
  arrival_rank INTEGER NOT NULL,
  source_time INTEGER,
  time BLOB NOT NULL DEFAULT X'5b6e756c6c2c7b227175616c697479223a22756e6b6e6f776e227d5d',
  version_hash BLOB NOT NULL REFERENCES document_version(version_hash),
  shown INTEGER NOT NULL DEFAULT 0,
  admitted INTEGER CHECK(admitted > 0),
  PRIMARY KEY(document_pk,input_pk,chain)
) STRICT, WITHOUT ROWID;
CREATE INDEX document_source_input ON document_source(input_pk,chain);
CREATE TABLE document_block (
  document_pk INTEGER NOT NULL REFERENCES document(pk) ON DELETE CASCADE,
  idx INTEGER NOT NULL,
  kind TEXT NOT NULL,
  body BLOB NOT NULL,
  text_id INTEGER REFERENCES text(id),
  blob_hash BLOB,
  data BLOB,                                -- as block.data
  PRIMARY KEY(document_pk,idx)
) STRICT, WITHOUT ROWID;
CREATE INDEX document_block_text ON document_block(text_id) WHERE text_id IS NOT NULL;
CREATE INDEX document_block_blob ON document_block(blob_hash) WHERE blob_hash IS NOT NULL;

-- Import membership is projected from the engine-authored observation details.
CREATE TABLE import_member (
  observation TEXT PRIMARY KEY,
  import_id TEXT NOT NULL,
  input TEXT NOT NULL,
  member TEXT NOT NULL,
  source_id TEXT,
  sidecar TEXT
) STRICT;
CREATE INDEX import_member_input ON import_member(input,import_id);
CREATE INDEX import_member_asset ON import_member(import_id,source_id);
CREATE INDEX import_member_sidecar ON import_member(input) WHERE sidecar IS NOT NULL;
CREATE TABLE import_asset_metadata (
  input TEXT NOT NULL,
  sidecar TEXT NOT NULL,
  ordinal INTEGER NOT NULL,
  source_id TEXT NOT NULL,
  name TEXT,
  media_type TEXT,
  size TEXT,
  PRIMARY KEY(input,sidecar,ordinal)
) STRICT, WITHOUT ROWID;
CREATE INDEX import_asset_metadata_id ON import_asset_metadata(input,sidecar,source_id);
CREATE TABLE import_asset_sidecar (
  input TEXT NOT NULL,
  sidecar TEXT NOT NULL,
  version INTEGER NOT NULL,
  source_version TEXT NOT NULL,
  error TEXT,
  PRIMARY KEY(input,sidecar)
) STRICT, WITHOUT ROWID;

-- How many rows the tables status counts hold, kept by the triggers below so
-- that status does not scan the tables.
CREATE TABLE row_count (
  name              TEXT PRIMARY KEY,       -- the table
  n                 INTEGER NOT NULL
) STRICT, WITHOUT ROWID;
INSERT INTO row_count(name, n) VALUES ('session', 0), ('message', 0), ('document', 0), ('tool_call', 0);
CREATE TRIGGER session_added AFTER INSERT ON session BEGIN UPDATE row_count SET n = n + 1 WHERE name = 'session'; END;
CREATE TRIGGER session_removed AFTER DELETE ON session BEGIN UPDATE row_count SET n = n - 1 WHERE name = 'session'; END;
CREATE TRIGGER message_added AFTER INSERT ON message BEGIN UPDATE row_count SET n = n + 1 WHERE name = 'message'; END;
CREATE TRIGGER message_removed AFTER DELETE ON message BEGIN UPDATE row_count SET n = n - 1 WHERE name = 'message'; END;
CREATE TRIGGER document_added AFTER INSERT ON document BEGIN UPDATE row_count SET n = n + 1 WHERE name = 'document'; END;
CREATE TRIGGER document_removed AFTER DELETE ON document BEGIN UPDATE row_count SET n = n - 1 WHERE name = 'document'; END;
CREATE TRIGGER tool_call_added AFTER INSERT ON tool_call BEGIN UPDATE row_count SET n = n + 1 WHERE name = 'tool_call'; END;
CREATE TRIGGER tool_call_removed AFTER DELETE ON tool_call BEGIN UPDATE row_count SET n = n - 1 WHERE name = 'tool_call'; END;

-- Packed readings are shared across input chains. Text stays interned separately.
CREATE TABLE message_version (
  version_hash BLOB PRIMARY KEY,
  body BLOB NOT NULL
) STRICT, WITHOUT ROWID;
CREATE TABLE message_projection (
  version_hash BLOB PRIMARY KEY REFERENCES message_version(version_hash) ON DELETE CASCADE,
  layout BLOB NOT NULL, -- original u64 block count and aligned original ordinals
  admitted INTEGER CHECK(admitted > 0) -- latest immutable admission contributing these facts
) STRICT, WITHOUT ROWID;
CREATE TABLE message_version_text (
  version_hash BLOB NOT NULL REFERENCES message_version(version_hash) ON DELETE CASCADE,
  idx INTEGER NOT NULL,
  text_id INTEGER NOT NULL REFERENCES text(id),
  PRIMARY KEY(version_hash, idx)
) STRICT, WITHOUT ROWID;
CREATE INDEX message_version_text_id ON message_version_text(text_id);
CREATE TABLE document_version (
  version_hash BLOB PRIMARY KEY,
  body BLOB NOT NULL
) STRICT, WITHOUT ROWID;
CREATE TABLE document_version_text (
  version_hash BLOB NOT NULL REFERENCES document_version(version_hash) ON DELETE CASCADE,
  idx INTEGER NOT NULL,
  text_id INTEGER NOT NULL REFERENCES text(id),
  PRIMARY KEY(version_hash, idx)
) STRICT, WITHOUT ROWID;
CREATE INDEX document_version_text_id ON document_version_text(text_id);

-- Membership rows use small value keys, independently of assertion payload size.
CREATE TABLE event_value (
  pk INTEGER PRIMARY KEY,
  hash BLOB NOT NULL UNIQUE,
  body BLOB NOT NULL
) STRICT;
CREATE TABLE event_source (
  event_pk INTEGER NOT NULL REFERENCES event(pk) ON DELETE CASCADE,
  input_pk INTEGER NOT NULL REFERENCES input_ref(pk),
  chain INTEGER NOT NULL,
  value_pk INTEGER NOT NULL REFERENCES event_value(pk),
  arrival_rank INTEGER NOT NULL,
  session_pk INTEGER REFERENCES session(pk),
  seq INTEGER,
  source_start INTEGER,
  source_end INTEGER,
  after_message_id TEXT,
  at INTEGER,
  fields BLOB NOT NULL,
  imported_provenance TEXT NOT NULL CHECK(json_valid(imported_provenance)),
  admitted INTEGER CHECK(admitted > 0),
  PRIMARY KEY(event_pk, input_pk, chain)
) STRICT, WITHOUT ROWID;
CREATE INDEX event_source_input ON event_source(input_pk, chain);
CREATE TABLE link_value (
  pk INTEGER PRIMARY KEY,
  hash BLOB NOT NULL UNIQUE,
  body BLOB NOT NULL
) STRICT;
CREATE TABLE link_source (
  link_pk INTEGER NOT NULL REFERENCES link(pk) ON DELETE CASCADE,
  input_pk INTEGER NOT NULL REFERENCES input_ref(pk),
  chain INTEGER NOT NULL,
  value_pk INTEGER NOT NULL REFERENCES link_value(pk),
  arrival_rank INTEGER NOT NULL,
  admitted INTEGER CHECK(admitted > 0),
  PRIMARY KEY(link_pk, input_pk, chain)
) STRICT, WITHOUT ROWID;
CREATE INDEX link_source_input ON link_source(input_pk, chain);

CREATE INDEX message_source_version ON message_source(version_hash);
CREATE INDEX document_source_version ON document_source(version_hash);
CREATE INDEX event_source_value ON event_source(value_pk);
CREATE INDEX link_source_value ON link_source(value_pk);
CREATE TABLE message_version_blob (
  version_hash BLOB NOT NULL REFERENCES message_version(version_hash) ON DELETE CASCADE,
  blob_hash BLOB NOT NULL,
  PRIMARY KEY(version_hash,blob_hash)
) STRICT, WITHOUT ROWID;
CREATE TABLE document_version_blob (
  version_hash BLOB NOT NULL REFERENCES document_version(version_hash) ON DELETE CASCADE,
  blob_hash BLOB NOT NULL,
  PRIMARY KEY(version_hash,blob_hash)
) STRICT, WITHOUT ROWID;
CREATE INDEX message_version_blob_hash ON message_version_blob(blob_hash);
CREATE INDEX document_version_blob_hash ON document_version_blob(blob_hash);

-- Reference changes wait for the round's cleanup, including after a restart.
CREATE TABLE changed_version (
  kind TEXT NOT NULL,
  version_hash BLOB NOT NULL,
  PRIMARY KEY(kind,version_hash)
) STRICT, WITHOUT ROWID;
CREATE TABLE changed_value (
  kind TEXT NOT NULL,
  value_pk INTEGER NOT NULL,
  PRIMARY KEY(kind,value_pk)
) STRICT, WITHOUT ROWID;
CREATE TABLE changed_text (
  text_id INTEGER PRIMARY KEY
) STRICT;
CREATE TRIGGER message_source_removed AFTER DELETE ON message_source BEGIN
  INSERT OR IGNORE INTO changed_version VALUES ('message',OLD.version_hash);
END;
CREATE TRIGGER message_source_changed AFTER UPDATE OF version_hash ON message_source
WHEN OLD.version_hash <> NEW.version_hash BEGIN
  INSERT OR IGNORE INTO changed_version VALUES ('message',OLD.version_hash);
END;
CREATE TRIGGER document_source_removed AFTER DELETE ON document_source BEGIN
  INSERT OR IGNORE INTO changed_version VALUES ('document',OLD.version_hash);
END;
CREATE TRIGGER document_source_changed AFTER UPDATE OF version_hash ON document_source
WHEN OLD.version_hash <> NEW.version_hash BEGIN
  INSERT OR IGNORE INTO changed_version VALUES ('document',OLD.version_hash);
END;
CREATE TRIGGER event_source_removed AFTER DELETE ON event_source BEGIN
  INSERT OR IGNORE INTO changed_value VALUES ('event',OLD.value_pk);
END;
CREATE TRIGGER event_source_changed AFTER UPDATE OF value_pk ON event_source
WHEN OLD.value_pk <> NEW.value_pk BEGIN
  INSERT OR IGNORE INTO changed_value VALUES ('event',OLD.value_pk);
END;
CREATE TRIGGER link_source_removed AFTER DELETE ON link_source BEGIN
  INSERT OR IGNORE INTO changed_value VALUES ('link',OLD.value_pk);
END;
CREATE TRIGGER link_source_changed AFTER UPDATE OF value_pk ON link_source
WHEN OLD.value_pk <> NEW.value_pk BEGIN
  INSERT OR IGNORE INTO changed_value VALUES ('link',OLD.value_pk);
END;
CREATE TRIGGER block_removed AFTER DELETE ON block
WHEN OLD.text_id IS NOT NULL BEGIN
  INSERT OR IGNORE INTO changed_text VALUES (OLD.text_id);
END;
CREATE TRIGGER block_changed AFTER UPDATE OF text_id ON block
WHEN OLD.text_id IS NOT NULL AND OLD.text_id IS NOT NEW.text_id BEGIN
  INSERT OR IGNORE INTO changed_text VALUES (OLD.text_id);
END;
CREATE TRIGGER document_block_removed AFTER DELETE ON document_block
WHEN OLD.text_id IS NOT NULL BEGIN
  INSERT OR IGNORE INTO changed_text VALUES (OLD.text_id);
END;
CREATE TRIGGER document_block_changed AFTER UPDATE OF text_id ON document_block
WHEN OLD.text_id IS NOT NULL AND OLD.text_id IS NOT NEW.text_id BEGIN
  INSERT OR IGNORE INTO changed_text VALUES (OLD.text_id);
END;
CREATE TRIGGER message_version_text_removed AFTER DELETE ON message_version_text
WHEN OLD.text_id IS NOT NULL BEGIN
  INSERT OR IGNORE INTO changed_text VALUES (OLD.text_id);
END;
CREATE TRIGGER message_version_text_changed AFTER UPDATE OF text_id ON message_version_text
WHEN OLD.text_id IS NOT NULL AND OLD.text_id IS NOT NEW.text_id BEGIN
  INSERT OR IGNORE INTO changed_text VALUES (OLD.text_id);
END;
CREATE TRIGGER document_version_text_removed AFTER DELETE ON document_version_text
WHEN OLD.text_id IS NOT NULL BEGIN
  INSERT OR IGNORE INTO changed_text VALUES (OLD.text_id);
END;
CREATE TRIGGER document_version_text_changed AFTER UPDATE OF text_id ON document_version_text
WHEN OLD.text_id IS NOT NULL AND OLD.text_id IS NOT NEW.text_id BEGIN
  INSERT OR IGNORE INTO changed_text VALUES (OLD.text_id);
END;

CREATE INDEX message_source_session ON message_source(session_pk);
CREATE INDEX event_source_session ON event_source(session_pk);
-- Recomputed comparison state for ordinary tool-database inputs.
CREATE TABLE tool_row (
    input_pk INTEGER NOT NULL REFERENCES input_ref(pk) ON DELETE CASCADE,
    tbl TEXT NOT NULL,
    key TEXT NOT NULL,
    hash BLOB NOT NULL,
    PRIMARY KEY(input_pk,tbl,key)
) WITHOUT ROWID;
CREATE TABLE tool_database (
    input_pk INTEGER PRIMARY KEY REFERENCES input_ref(pk) ON DELETE CASCADE,
    length INTEGER NOT NULL,
    schema TEXT NOT NULL
);

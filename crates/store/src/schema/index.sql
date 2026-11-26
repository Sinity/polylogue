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
  format            TEXT NOT NULL DEFAULT '', -- parser declaration used for this input
  route             TEXT NOT NULL DEFAULT 'live' CHECK (route IN (@names(Route)))
) STRICT;

CREATE TABLE session (
  pk                INTEGER PRIMARY KEY,
  id                TEXT NOT NULL UNIQUE,   -- '<origin>:<native id>'
  origin            TEXT NOT NULL,
  native_id         TEXT NOT NULL,
  kind              TEXT NOT NULL CHECK (kind IN (@names(SessionKind))),  -- continuations are links, not a kind
  stub              INTEGER NOT NULL DEFAULT 0,  -- 1: created by a hook line naming a session that has no
                                            --   transcript yet, so its live state exists; cleared when its messages arrive
  title             TEXT,                   -- provider title only
  title_source      TEXT,                   -- title source; precedence is owned by model TitleSource
  derived_title     TEXT,                   -- derived (title): first line of the first typed prompt; shown when
                                            --   title is NULL, never exported as recorded
  cwd_json          TEXT,                   -- every working directory in order (model Session.cwd); `cwd` is the last
  started_time      BLOB NOT NULL DEFAULT X'5b6e756c6c2c7b227175616c697479223a22756e6b6e6f776e227d5d',
  started_at        INTEGER,                -- from the messages and events the session counts as its own
  started_until     INTEGER,                --   with started_precision, as message.at_until and at_precision
  started_precision TEXT CHECK (started_precision IN (@names(Precision))),
  last_activity_at  INTEGER,                -- the latest start of its messages' times (session summary)
  last_active_until INTEGER,                -- the latest end of their periods, when one ends after
                                            --   last_activity_at (docs/query.md, times)
  last_active_precision TEXT CHECK (last_active_precision IN (@names(Precision))),
  cwd               TEXT,
  repo              TEXT,                   -- name or URL, for display: record_repo, else the remote of the
                                            --   repo_lookup holding the session's cwd (derived: repository)
  mode              TEXT,                   -- harness mode as recorded, e.g. background
  record_repo       TEXT,                   -- the repository the session's own records name
  repo_root         TEXT,                   -- the repository's identity: the first of its sorted root commits, from repo_lookup (derived)
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
  changed           INTEGER NOT NULL,       -- write counter of the last change to its own rows
                                            --   (not its derived values or live columns)
  derived_changed   INTEGER,                -- write counter of the summary transaction that last wrote its summaries
  live_changed      INTEGER,                -- write counter value of the last change to its live columns
  -- Live facts: the live summary of the session's live events (hook, process
  -- and registry lines, summary/live.rs); the live state is computed at read
  -- time from these and now (`pl_live`).
  live_facts TEXT,                          -- polylogue_model::live::LiveFacts as JSON
  live_last_at INTEGER,                     -- time of the latest live event included
  CHECK ((started_at IS NULL) = (started_precision IS NULL)),
  CHECK (started_until IS NULL OR (started_precision <> 'instant' AND started_until > started_at)),
  CHECK (started_until IS NOT NULL OR started_precision IS NULL OR started_precision IN ('instant','day','month','year')),
  CHECK ((last_activity_at IS NULL) = (last_active_precision IS NULL)),
  CHECK (last_active_until IS NULL OR (last_active_precision <> 'instant' AND last_active_until > last_activity_at)),
  CHECK (last_active_until IS NOT NULL OR last_active_precision IS NULL OR last_active_precision IN ('instant','day','month','year'))
) STRICT;
CREATE INDEX session_started  ON session(started_at);         -- started between, until:
CREATE INDEX session_changed  ON session(changed);            -- change feed; derive step
CREATE INDEX session_cwd      ON session(cwd);                -- cwd: prefix as a range; briefings; resume
CREATE INDEX session_repo     ON session(repo) WHERE repo IS NOT NULL;
CREATE INDEX session_live ON session(live_last_at) WHERE live_last_at IS NOT NULL;   -- running:, waiting:, the overview
CREATE INDEX session_repo_root ON session(repo_root) WHERE repo_root IS NOT NULL;   -- repo: by identity

CREATE TABLE message (
  pk                INTEGER PRIMARY KEY,
  id                TEXT NOT NULL,          -- unique: message_id
  session_pk        INTEGER NOT NULL REFERENCES session(pk),
  session_order     BLOB NOT NULL,          -- the message's place in its session (docs/sessions.md, messages):
                                            --   order time, input ID, position in that input, as bytes that
                                            --   sort in that order (encoding::session_order); ties by id
  role              TEXT NOT NULL CHECK (role IN (@names(Role))),  -- sources/ maps each provider's role
  written_by        TEXT NOT NULL CHECK (written_by IN (@names(WrittenBy))),
  prompt_kind       TEXT CHECK (prompt_kind IN (@names(PromptKind))),  -- a user message's category; harness
                                            --   also marks runtime notices under another role
  model             TEXT,
  at_time           BLOB NOT NULL DEFAULT X'5b6e756c6c2c7b227175616c697479223a22756e6b6e6f776e227d5d',
  at                INTEGER,                -- provider time (ms): the start of its period
  at_until          INTEGER,                -- the exclusive end (ms) of a time that is a period (encoding::
                                            --   TimeColumns); NULL for an instant, an unknown time, and a
                                            --   calendar grain the source gave no bound
  at_precision      TEXT CHECK (at_precision IN (@names(Precision))),  -- NULL exactly when `at` is
  parent_id         TEXT,                   -- natural message id of the tree parent (may arrive later)
  input_pk          INTEGER NOT NULL REFERENCES input_ref(pk),  -- the main route's input: its reading is the stored one
  version_hash      BLOB,                   -- the main reading's content hash: the message's version
  layout            BLOB,                   -- the main reading's block layout when it does not hold every block
  source_chain      INTEGER,                -- the revision chain the byte range is in (archive.db revision.chain):
                                            --   offsets are stable only within a chain, which only appends
  source_start      INTEGER,                -- byte range of its records in that chain (export `sources[0]`):
  source_end        INTEGER,                --   raw view and outside anchors; for a JSON export, the conversation object
  in_progress       INTEGER NOT NULL DEFAULT 0, -- the shown message's completion is in_progress (its main reading's completion)
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
  CHECK ((at IS NULL) = (at_precision IS NULL)),
  CHECK (at_until IS NULL OR (at_precision <> 'instant' AND at_until > at)),
  CHECK (at_until IS NOT NULL OR at_precision IS NULL OR at_precision IN ('instant','day','month','year'))
) STRICT;
-- A first build drops the three unique indexes on natural keys and builds
-- them once at its end; meanwhile the writer finds rows by key in memory.
CREATE UNIQUE INDEX message_id ON message(id);
CREATE INDEX message_session ON message(session_pk, session_order, id);   -- reading in order; [last:N]

-- Readings. Each chain's reading owns its rows: every row below carries the
-- input and chain whose records state it and the byte range of those
-- records, and holds what they say about one thing, taken together in byte
-- order. Replacing a reading deletes its rows and inserts the new ones; no
-- row is shared between chains, so nothing counts who else refers to it.
-- The shown rows (session, message, block, tool_call, event, link,
-- document) are computed from the readings present (writer/compose.rs).

-- What a chain states about a session (model Session, as JSON), merged in
-- byte order. A session exists while some reading states it.
CREATE TABLE session_record (
  input_pk          INTEGER NOT NULL REFERENCES input_ref(pk),
  chain             INTEGER NOT NULL,
  session_id        TEXT NOT NULL,          -- as the chain names it
  facts             TEXT NOT NULL,
  committed         INTEGER CHECK(committed > 0),
  PRIMARY KEY (input_pk, chain, session_id)
) STRICT, WITHOUT ROWID;
CREATE INDEX session_record_id ON session_record(session_id);

-- The IDs copies of a session carry when they have no native ID
-- (Session.aliases): an AI Studio prompt read from Drive names the content
-- key of its Takeout copies. An alias exactly one session records names that
-- session, decided from these rows each time the sessions are composed.
CREATE TABLE session_alias (
  input_pk          INTEGER NOT NULL REFERENCES input_ref(pk),
  chain             INTEGER NOT NULL,
  session_id        TEXT NOT NULL,          -- the recording session
  alias             TEXT NOT NULL,          -- the copy's session ID
  PRIMARY KEY (input_pk, chain, session_id, alias)
) STRICT, WITHOUT ROWID;
CREATE INDEX session_alias_name ON session_alias(alias, session_id);
CREATE INDEX session_alias_session ON session_alias(session_id);

-- One chain's reading of a message: where the chain first states it (a chain
-- that restates whole conversations: last), its place and order time, and
-- its content: the model Message with text moved to `text` (parse_text holds
-- the keys), stored once in a content group (content.rs) as its fields
-- without blocks, then each block and its structured value; the shown
-- message's blocks name those pieces. A reading of a sparse export also
-- keeps the block layout it selected.
CREATE TABLE message_parse (
  input_pk          INTEGER NOT NULL REFERENCES input_ref(pk),
  chain             INTEGER NOT NULL,
  message_id        TEXT NOT NULL,          -- as the chain names it
  session_id        TEXT NOT NULL,          -- as the chain names it
  source_start      INTEGER NOT NULL,
  source_end        INTEGER NOT NULL,
  seq               INTEGER NOT NULL,       -- its place in the input: the record's offset, or the
  record_position   INTEGER NOT NULL,       --   explicit reading order; then its position in the record
  order_time        INTEGER,                -- the latest source time (ms) of this reading and the readings
                                            --   before it in its chain and session; NULL before any time
  at                INTEGER,                -- this reading's own source time (ms)
  route             TEXT NOT NULL CHECK (route IN (@names(Route))),
  completion        TEXT NOT NULL CHECK (completion IN (@names(Completion))),
  id_kind           TEXT NOT NULL CHECK (id_kind IN (@names(IdKind))),
  hash              BLOB NOT NULL,          -- the message's content hash (its version)
  layout            BLOB NOT NULL,          -- original block count and selected ordinals
  response_key      TEXT,                   -- the key the source gives the records of one response
  content_name      TEXT,                   -- a content-named message's name without its number
  committed         INTEGER CHECK(committed > 0),
  content_group     INTEGER NOT NULL REFERENCES content_group(id),
  content_start     INTEGER NOT NULL,       -- its content's place in the group's decoded bytes
  content_length    INTEGER NOT NULL,
  PRIMARY KEY (input_pk, chain, message_id)
) STRICT;
CREATE INDEX message_parse_id ON message_parse(message_id);
CREATE INDEX message_parse_session ON message_parse(session_id);
CREATE INDEX message_parse_place ON message_parse(input_pk, chain, session_id, seq, record_position);
CREATE INDEX message_parse_start ON message_parse(input_pk, chain, source_start);
CREATE INDEX message_parse_response ON message_parse(input_pk, chain, response_key) WHERE response_key IS NOT NULL;
CREATE INDEX message_parse_content ON message_parse(input_pk, chain, content_name, source_start) WHERE content_name IS NOT NULL;
CREATE INDEX message_parse_hash ON message_parse(hash);
CREATE INDEX message_parse_group ON message_parse(content_group);   -- whether a group is still named
-- The message readings an input still holds: once a newer chain of the
-- input is read to its end, a content-named message it no longer holds is
-- gone from that input (an edit has a new ID). Composition reads the same.
CREATE VIEW held_message_parse AS
SELECT * FROM message_parse r
WHERE NOT (r.id_kind = 'content' AND EXISTS(SELECT 1 FROM parse_state p
  WHERE p.input_pk = r.input_pk AND p.chain > r.chain AND p.status IN ('done','waiting')));

-- The texts a reading's message and document bodies hold, by block. A text
-- leaves with its last row here, with its search entries.
CREATE TABLE parse_text (
  input_pk          INTEGER NOT NULL REFERENCES input_ref(pk),
  chain             INTEGER NOT NULL,
  document          INTEGER NOT NULL CHECK (document IN (0,1)),  -- 1: item names a document
  item              TEXT NOT NULL,          -- the message or document ID as the chain names it
  idx               INTEGER NOT NULL,       -- the block's position in the body
  text_id           INTEGER NOT NULL REFERENCES text(id),
  PRIMARY KEY (input_pk, chain, document, item, idx)
) STRICT, WITHOUT ROWID;
CREATE INDEX parse_text_id ON parse_text(text_id);

-- Records that are not messages but stand in a chain of parents: `name`
-- stands for `target` (NULL: no message), settled through the chain's earlier
-- aliases so one lookup resolves a name.
CREATE TABLE message_alias (
  input_pk          INTEGER NOT NULL REFERENCES input_ref(pk),
  chain             INTEGER NOT NULL,
  name              TEXT NOT NULL,
  target            TEXT,
  source_start      INTEGER NOT NULL,
  source_end        INTEGER NOT NULL,
  PRIMARY KEY (input_pk, chain, name)
) STRICT, WITHOUT ROWID;
CREATE INDEX message_alias_name ON message_alias(name);

-- One chain's reading of an event. Its content is the encoded Event without
-- its data, followed by its data as JSON, in a content group.
CREATE TABLE event_parse (
  input_pk          INTEGER NOT NULL REFERENCES input_ref(pk),
  chain             INTEGER NOT NULL,
  event_id          TEXT NOT NULL,          -- as the chain names it
  session_id        TEXT,                   -- NULL: an event about its input
  route             TEXT NOT NULL CHECK (route IN (@names(Route))),
  seq               INTEGER NOT NULL,
  source_start      INTEGER,
  source_end        INTEGER,
  after_message_id  TEXT,                   -- the message it follows, as the chain names it
  kind              TEXT NOT NULL,
  at                INTEGER,
  hash              BLOB NOT NULL,          -- the value's hash
  imported_sources  TEXT NOT NULL CHECK(json_valid(imported_sources)),
  committed         INTEGER CHECK(committed > 0),
  content_group     INTEGER NOT NULL REFERENCES content_group(id),
  content_start     INTEGER NOT NULL,       -- its fields' place in the group's decoded bytes,
  content_length    INTEGER NOT NULL,       --   and their length
  data_length       INTEGER NOT NULL,       -- the length of its data, which follows its fields
  PRIMARY KEY (input_pk, chain, event_id)
) STRICT;
CREATE INDEX event_parse_id ON event_parse(event_id);
CREATE INDEX event_parse_session ON event_parse(session_id) WHERE session_id IS NOT NULL;
CREATE INDEX event_parse_group ON event_parse(content_group);   -- whether a group is still named

-- One chain's statement of a link between sessions. `statement` is the
-- link's coordinates (parent, child, kind, how, branch point, spawning
-- call) as the chain names them.
CREATE TABLE link_parse (
  input_pk          INTEGER NOT NULL REFERENCES input_ref(pk),
  chain             INTEGER NOT NULL,
  statement         BLOB NOT NULL,
  parent_id         TEXT NOT NULL,
  child_id          TEXT NOT NULL,
  kind              TEXT NOT NULL CHECK (kind IN (@names(LinkKind))),
  how               TEXT CHECK (how IN (@names(ReferenceReason))),
  route             TEXT NOT NULL CHECK (route IN (@names(Route))),
  hash              BLOB NOT NULL,
  body              BLOB NOT NULL,          -- the model Link as JSON
  committed         INTEGER CHECK(committed > 0),
  PRIMARY KEY (input_pk, chain, statement)
) STRICT, WITHOUT ROWID;
CREATE INDEX link_parse_statement ON link_parse(statement);
CREATE INDEX link_parse_parent ON link_parse(parent_id);
CREATE INDEX link_parse_child ON link_parse(child_id);

-- One chain's reading of a document.
CREATE TABLE document_parse (
  input_pk          INTEGER NOT NULL REFERENCES input_ref(pk),
  chain             INTEGER NOT NULL,
  document_id       TEXT NOT NULL,
  route             TEXT NOT NULL CHECK (route IN (@names(Route))),
  source_start      INTEGER NOT NULL,
  source_end        INTEGER NOT NULL,
  source_time       INTEGER,
  time              BLOB NOT NULL,
  hash              BLOB NOT NULL,
  body              BLOB NOT NULL,          -- the model Document with text moved to `text`
  committed         INTEGER CHECK(committed > 0),
  PRIMARY KEY (input_pk, chain, document_id)
) STRICT, WITHOUT ROWID;
CREATE INDEX document_parse_id ON document_parse(document_id);

-- Outcomes a chain learned after a tool result was written, for calls that
-- ran in the background, as the chain states them.
CREATE TABLE call_outcome (
  input_pk          INTEGER NOT NULL REFERENCES input_ref(pk),
  chain             INTEGER NOT NULL,
  session_id        TEXT NOT NULL,
  call_id           TEXT NOT NULL,
  position          INTEGER NOT NULL,       -- its order among the chain's outcomes of this call
  body              TEXT NOT NULL,          -- the model BackgroundUpdate as JSON
  PRIMARY KEY (input_pk, chain, session_id, call_id)
) STRICT, WITHOUT ROWID;
CREATE INDEX call_outcome_call ON call_outcome(session_id, call_id);

-- The newest message time in each chain of an export that lists every
-- conversation of its account, by session origin. A session of that origin
-- the export does not hold, with no activity after that time, is gone
-- upstream (sessions `source: gone`); its rows stay.
CREATE TABLE export_reading (
  input_pk          INTEGER NOT NULL REFERENCES input_ref(pk),
  chain             INTEGER NOT NULL,
  origin            TEXT NOT NULL,
  latest_at         INTEGER NOT NULL,
  PRIMARY KEY (input_pk, chain, origin)
) STRICT, WITHOUT ROWID;
CREATE INDEX export_reading_origin ON export_reading(origin, latest_at);

-- The route each field group of a shown message came from, where it is not
-- the main route (the message's own input_pk, source_chain).
CREATE TABLE message_route (
  message_pk        INTEGER NOT NULL REFERENCES message(pk),
  field_group       TEXT NOT NULL CHECK (field_group IN (@names(RoutedFieldGroup))),
  input_pk          INTEGER NOT NULL REFERENCES input_ref(pk),
  chain             INTEGER NOT NULL,
  PRIMARY KEY (message_pk, field_group)
) STRICT, WITHOUT ROWID;

-- Content stored once (content.rs): each group is one zstd frame of pieces
-- laid one after another in the order a parse worker wrote them (about
-- 64 KiB before compression; a longer piece is a group alone). A group holds
-- texts, or message and event readings' content. The full-text files are
-- contentless and the shown rows name pieces of their readings, so this is
-- the only copy in the index. A group's key is never given to another group.
CREATE TABLE content_group (
  id                INTEGER PRIMARY KEY AUTOINCREMENT,
  hash              BLOB NOT NULL,          -- unique: content_group_hash. BLAKE3 of the compressed body
  body              BLOB NOT NULL
) STRICT;
CREATE UNIQUE INDEX content_group_hash ON content_group(hash);

-- Text is stored once per distinct content; `id` is the rowid of its search
-- entries, never given to another text after this one is removed.
CREATE TABLE text (
  id                INTEGER PRIMARY KEY AUTOINCREMENT,
  normalized_hash   BLOB NOT NULL,          -- BLAKE3 of NFC UTF-8; raw spellings may share this identity
  hash              BLOB NOT NULL,          -- unique: text_hash. BLAKE3 of the UTF-8 text; embeddings are keyed by it
  indexed           INTEGER NOT NULL DEFAULT 0, -- bit 1: a member of text_fts (prose); bit 2: of output_fts (tool output)
  group_id          INTEGER NOT NULL REFERENCES content_group(id), -- the group leaves once nothing names it
  start             INTEGER NOT NULL,       -- byte offset of the text in its group's decoded bytes
  length            INTEGER NOT NULL        -- its length in bytes
) STRICT;

-- What the provider wrote, one row per content item. Facts derived about a tool
-- call (tool, command, outcome, exit code, duration) belong to tool_call alone.
CREATE TABLE block (
  pk                INTEGER PRIMARY KEY,
  message_pk        INTEGER NOT NULL REFERENCES message(pk),
  session_pk        INTEGER NOT NULL,       -- its message's session, for block_call
  idx               INTEGER,                -- compact position among the stored blocks, used for ordering
  kind              TEXT NOT NULL CHECK (kind IN (@names(BlockKind))),
  text_id           INTEGER REFERENCES text(id),
  content_group     INTEGER NOT NULL,       -- the model Block as JSON, with its prose or output text moved to
  content_start     INTEGER NOT NULL,       --   `text` (text_id) and its structured value after it: a piece of
  content_length    INTEGER NOT NULL,       --   its message's main reading's content (message_parse), so
  data_length       INTEGER,                --   reading a block back is lossless. data_length: tool_result's
                                            --   structured data, kept apart so a text read never decodes it
  call_id           TEXT,                   -- tool_call and tool_result: the provider's call id (unique only in a session)
  blob_hash         BLOB,                   -- image/file, large tool output: bytes in the blob store; export `blob`
  direction         TEXT,                   -- image/file: input | output | unknown
  media_type        TEXT,
  file_name         TEXT,                   -- export `name`
  size              INTEGER,                -- image/file
  url               TEXT,                   -- image/file downloaded from a link
  encrypted         INTEGER NOT NULL DEFAULT 0,  -- thinking kept encrypted
  other_type        TEXT,                   -- other: the provider's content type name; export `type`
  ordinal           TEXT                    -- original u64 source position; NULL means the dense index the source implies
) STRICT;
CREATE UNIQUE INDEX text_hash ON text(hash);
CREATE INDEX text_group_member ON text(group_id);  -- whether a group still holds a text
CREATE UNIQUE INDEX block_message ON block(message_pk, idx);
CREATE INDEX block_text   ON block(text_id) WHERE text_id IS NOT NULL;   -- search hit -> blocks

-- Full-text entries are not in this file: they live in the full-text files
-- named by meta `search_files` (search.sql), split by text key.

-- Changes to full-text entries still to apply. A text is recorded here in the
-- transaction whose blocks make it a member of a search index or remove it
-- from one, and when the text itself is removed (search_triggers.sql); the
-- search thread reads the text and the writer brings the text's entry in its
-- full-text file to what the text is now: present and holding the text while
-- it is a member, absent otherwise. Applying a row twice changes nothing, so
-- the row leaves only after its file has committed. A flush applies every
-- waiting row before it returns. A first build queues nothing until its
-- composition ends: it fills the files while it composes.
CREATE TABLE search_pending (
  text_id           INTEGER NOT NULL,       -- text.id
  fts_table         INTEGER NOT NULL,       -- which index: 1 text_fts, 2 output_fts (the bits of text.indexed)
  PRIMARY KEY (text_id, fts_table)
) STRICT, WITHOUT ROWID;

-- One row per tool call, paired with its result inside the same session. The
-- one place every derived call fact is kept.
CREATE TABLE tool_call (
  pk                INTEGER PRIMARY KEY,
  session_pk        INTEGER NOT NULL,       -- its call block's session, for the indexes by session
  call_block_pk     INTEGER NOT NULL UNIQUE REFERENCES block(pk),
  call_id           TEXT,                   -- the provider's call id: how a later result finds its call
  result_at         INTEGER,
  reported_duration_ms INTEGER,
  result_message_id TEXT,                   -- the result's message and block position, NULL while no
  result_idx        INTEGER,                --   result is shown: the first tool_result block with the
                                            --   call's ID in session order
  at                INTEGER,
  tool              TEXT NOT NULL,
  action            TEXT CHECK (action IN (@names(ToolAction))),  -- the parser's ToolAction
  command           TEXT,                   -- as given
  command_norm      TEXT,                   -- derived (tool_calls): 'cargo test', 'git commit'
  path              TEXT,                   -- file path argument, when the tool has one
  outcome           TEXT NOT NULL CHECK (outcome IN (@names(CallOutcome))),  -- facts of the records, never of the clock;
                                            --   `pending` is computed at read time from live state
  outcome_reason    TEXT,                   -- a reason code (no_result, background_never_reported, ...)
  exit_code         INTEGER,
  run_by_script     INTEGER NOT NULL DEFAULT 0 -- a command a script call ran, with no call of its own
) STRICT;
CREATE INDEX tool_call_session ON tool_call(session_pk, at);   -- exists, seq(...), the changes view
CREATE INDEX tool_call_pairing ON tool_call(session_pk, call_id) WHERE call_id IS NOT NULL;   -- result -> call
CREATE INDEX block_call ON block(session_pk, call_id) WHERE call_id IS NOT NULL;   -- call -> earlier result
CREATE INDEX tool_call_action  ON tool_call(action, NULLIF(outcome, 'unknown'));   -- action: and outcome: as compiled
CREATE INDEX tool_call_repo_command ON tool_call(command_norm, at) WHERE command_norm IS NOT NULL;   -- failing commands
CREATE INDEX tool_call_result  ON tool_call(result_message_id) WHERE result_message_id IS NOT NULL;   -- result -> call

-- Substring search on short fields (command:, path:) without a scan. Unfolded
-- text, so it can read its content from tool_call. Triggers on tool_call keep
-- it (search_triggers.sql); a first build writes no entries and rebuilds the
-- index from tool_call when its composition ends.
CREATE VIRTUAL TABLE tool_call_trigram USING fts5(command, path,
    content='tool_call', content_rowid='pk', tokenize='trigram');
INSERT INTO tool_call_trigram(tool_call_trigram, rank) VALUES ('automerge', 16);
INSERT INTO tool_call_trigram(tool_call_trigram, rank) VALUES ('hashsize', 67108864);


CREATE TABLE link (
  pk                INTEGER PRIMARY KEY,
  parent_id         TEXT NOT NULL,          -- natural session id (may not be ingested yet)
  child_id          TEXT NOT NULL,          -- natural session id of the child (may not be ingested yet: a parent's
                                            --   thread_spawn_edges, continued-in or job children name it first)
  kind              TEXT NOT NULL CHECK (kind IN (@names(LinkKind))),
  how               TEXT CHECK (how IN (@names(ReferenceReason))),  -- a reference's reason
  copies_prefix     INTEGER NOT NULL DEFAULT 0 CHECK(copies_prefix IN (0,1)),
  branch_point      TEXT,                   -- natural message id where the child branches off
  spawning_call     TEXT,                   -- the spawning call's ID as the source record names it (a tool-use ID)
  evidence          TEXT NOT NULL,          -- which input field established the parent
  source_chain      INTEGER NOT NULL DEFAULT 0,
  input_pk          INTEGER NOT NULL REFERENCES input_ref(pk)  -- the input holding that field
) STRICT;
CREATE UNIQUE INDEX link_statement ON link(parent_id, child_id, kind, how IS NULL, coalesce(how,''), branch_point IS NULL, coalesce(branch_point,''), spawning_call IS NULL, coalesce(spawning_call,''));
CREATE INDEX link_child_id ON link(child_id);   -- parents by id; late children
CREATE INDEX link_parent ON link(parent_id, kind);             -- children; late parents

-- Copied-prefix facts an input states never create a relationship on their own.
CREATE TABLE prefix_copy (
  input_pk INTEGER NOT NULL REFERENCES input_ref(pk),
  chain INTEGER NOT NULL,
  parent_id TEXT NOT NULL,
  child_id TEXT NOT NULL,
  committed INTEGER CHECK(committed > 0),
  explicit_members INTEGER NOT NULL CHECK(explicit_members IN (0,1)),
  PRIMARY KEY(input_pk,chain,parent_id,child_id)
) STRICT, WITHOUT ROWID;
CREATE INDEX prefix_copy_edge ON prefix_copy(parent_id,child_id);
CREATE INDEX prefix_copy_child ON prefix_copy(child_id,parent_id);

-- Copied occurrences keep their original reading and order through exports.
-- The input/chain that states them controls retraction; source coordinates say where each was seen.
CREATE TABLE prefix_member (
  input_pk INTEGER NOT NULL REFERENCES input_ref(pk),
  chain INTEGER NOT NULL,
  parent_id TEXT NOT NULL,
  child_id TEXT NOT NULL,
  message_id TEXT NOT NULL,
  source_input TEXT NOT NULL,
  source_chain INTEGER NOT NULL,
  source_version TEXT NOT NULL,
  sequence TEXT NOT NULL,
  position TEXT NOT NULL,
  committed INTEGER CHECK(committed > 0),
  PRIMARY KEY(input_pk,chain,parent_id,child_id,message_id,source_input,source_chain,source_version,sequence,position)
) STRICT, WITHOUT ROWID;
CREATE INDEX prefix_member_child ON prefix_member(child_id,parent_id,message_id);
CREATE INDEX prefix_member_message ON prefix_member(message_id,parent_id);


-- Every consumer sees one relation when recorded coordinates agree. Distinct
-- known coordinates remain alternatives; partial markers still contribute
-- evidence and copied-prefix observations. All correlated lookups use the
-- affected edge's link_statement index, so a session read does not walk the whole graph.
CREATE VIEW session_link_identity AS
WITH statements AS NOT MATERIALIZED (
  SELECT l.pk,l.parent_id,l.child_id,l.kind,l.how,l.copies_prefix,
    l.branch_point,l.spawning_call,l.source_chain,l.input_pk,
    (SELECT count(DISTINCT x.branch_point) FROM link x
     WHERE x.parent_id=l.parent_id AND x.child_id=l.child_id AND x.kind=l.kind AND x.how IS l.how) AS branch_points,
    (SELECT count(DISTINCT x.spawning_call) FROM link x
     WHERE x.parent_id=l.parent_id AND x.child_id=l.child_id AND x.kind=l.kind AND x.how IS l.how) AS calls
  FROM link l
)
SELECT s.pk,s.parent_id,s.child_id,s.kind,s.how,
  ((SELECT max(x.copies_prefix) FROM link x
    WHERE x.parent_id=s.parent_id AND x.child_id=s.child_id AND x.kind=s.kind AND x.how IS s.how)
    OR EXISTS(SELECT 1 FROM prefix_copy p WHERE p.parent_id=s.parent_id AND p.child_id=s.child_id)) AS copies_prefix,
  CASE WHEN s.branch_points <= 1 THEN
    (SELECT min(x.branch_point) FROM link x
     WHERE x.parent_id=s.parent_id AND x.child_id=s.child_id AND x.kind=s.kind AND x.how IS s.how)
    ELSE s.branch_point END AS branch_point,
  CASE WHEN s.calls <= 1 THEN
    (SELECT min(x.spawning_call) FROM link x
     WHERE x.parent_id=s.parent_id AND x.child_id=s.child_id AND x.kind=s.kind AND x.how IS s.how)
    ELSE s.spawning_call END AS spawning_call,
  s.source_chain,s.input_pk
FROM statements s
WHERE
  (s.branch_points <= 1 AND s.calls <= 1 AND s.pk =
    (SELECT min(x.pk) FROM link x
     WHERE x.parent_id=s.parent_id AND x.child_id=s.child_id AND x.kind=s.kind AND x.how IS s.how))
  OR (s.branch_points > 1 AND s.calls <= 1 AND s.branch_point IS NOT NULL AND s.pk =
    (SELECT min(x.pk) FROM link x
     WHERE x.parent_id=s.parent_id AND x.child_id=s.child_id AND x.kind=s.kind AND x.how IS s.how
       AND x.branch_point=s.branch_point))
  OR (s.branch_points <= 1 AND s.calls > 1 AND s.spawning_call IS NOT NULL AND s.pk =
    (SELECT min(x.pk) FROM link x
     WHERE x.parent_id=s.parent_id AND x.child_id=s.child_id AND x.kind=s.kind AND x.how IS s.how
       AND x.spawning_call=s.spawning_call))
  OR (s.branch_points > 1 AND s.calls > 1 AND NOT EXISTS (
    SELECT 1 FROM link x
    WHERE x.parent_id=s.parent_id AND x.child_id=s.child_id AND x.kind=s.kind AND x.how IS s.how
      AND (s.branch_point IS NULL OR x.branch_point=s.branch_point)
      AND (s.spawning_call IS NULL OR x.spawning_call=s.spawning_call)
      AND ((s.branch_point IS NULL AND x.branch_point IS NOT NULL)
        OR (s.spawning_call IS NULL AND x.spawning_call IS NOT NULL))));

-- Evidence is loaded only by record readers, not membership/count queries.
CREATE VIEW session_link AS
SELECT s.*,(SELECT min(x.evidence) FROM link x
  WHERE x.parent_id=s.parent_id AND x.child_id=s.child_id AND x.kind=s.kind AND x.how IS s.how) AS evidence
FROM session_link_identity s;

-- Inherited history also exists in copied files with no provider
-- relationship. It never creates a logical-session edge.
CREATE VIEW session_prefix AS
SELECT DISTINCT parent_id,child_id,branch_point,copies_prefix
FROM session_link
WHERE kind <> 'reference' AND (kind IN ('fork','continuation','resume') OR copies_prefix=1)
UNION
SELECT DISTINCT p.parent_id,p.child_id,NULL,1
FROM prefix_copy p
WHERE NOT EXISTS (
  SELECT 1 FROM session_link l
  WHERE l.parent_id=p.parent_id AND l.child_id=p.child_id
    AND l.kind <> 'reference' AND (l.kind IN ('fork','continuation','resume') OR l.copies_prefix=1));

CREATE TABLE event (
  pk                INTEGER PRIMARY KEY,
  id                TEXT NOT NULL,          -- unique: event_id. Computed by the parser from the raw record; an
                                            --   event that names no session is keyed by its input record instead
  session_pk        INTEGER REFERENCES session(pk),  -- NULL for an event about its input, not a session
  seq               INTEGER NOT NULL,       -- display position: its reading's place in its input, or for an event
                                            --   placed after a message with no place of its own, that message's
                                            --   main reading's; not a key
  after_message_id  TEXT,                   -- the message it follows, joined by ID; export `after_message`
  kind              TEXT NOT NULL,          -- open set; each provider's kinds are listed in sources/
  at                INTEGER,                -- as message.at, at_until and at_precision
  at_until          INTEGER,
  at_precision      TEXT CHECK (at_precision IN (@names(Precision))),
  input_pk          INTEGER NOT NULL REFERENCES input_ref(pk),  -- the input whose record it came from
  source_chain      INTEGER,                -- revision chain of the byte range (as message.source_chain)
  source_start      INTEGER,
  source_end        INTEGER,
  content_group     INTEGER NOT NULL,       -- the shown reading's content (event_parse): the encoded Event
  content_start     INTEGER NOT NULL,       --   without its data, with its session and the message it
  content_length    INTEGER NOT NULL,       --   follows as stated (this row names them as shown), then
  data_length       INTEGER NOT NULL,       --   its normalized payload as JSON; export `data`
  session_order     BLOB,                   -- its place among its session's rows for the session's summaries:
                                            --   time, input ID, position, as message.session_order; NULL without
                                            --   a session
  CHECK ((at IS NULL) = (at_precision IS NULL)),
  CHECK (at_until IS NULL OR (at_precision <> 'instant' AND at_until > at)),
  CHECK (at_until IS NOT NULL OR at_precision IS NULL OR at_precision IN ('instant','day','month','year'))
) STRICT;
CREATE UNIQUE INDEX event_id ON event(id);
CREATE INDEX event_session_order ON event(session_pk, session_order, id) WHERE session_pk IS NOT NULL;  -- summary folds
CREATE INDEX event_path_marker ON event(session_pk)
  WHERE kind IN ('active_leaf','active_path_empty','active_path_unknown');   -- the active path summary
CREATE INDEX event_session ON event(session_pk, seq);
CREATE INDEX event_input   ON event(input_pk, source_chain, source_start);
CREATE INDEX event_after_id ON event(after_message_id) WHERE after_message_id IS NOT NULL;
CREATE INDEX event_placement ON event(session_pk, pk, seq, after_message_id, at, id);   -- where an unchanged event goes, without its row
CREATE INDEX event_kind    ON event(kind, at);





-- How far each revision chain has been parsed, and how it ended. Every chain of every input has a row once the loop has seen it,
-- so nothing is silently skipped.
CREATE TABLE parse_state (
  input_pk          INTEGER NOT NULL REFERENCES input_ref(pk),
  chain             INTEGER NOT NULL,
  parse_version   TEXT NOT NULL DEFAULT '',
  status            TEXT NOT NULL CHECK(status IN ('parsing','waiting','done','no_parser_yet','partial','refused','failed')),
                                            -- parsing: more stored bytes need parsing; waiting: incomplete record at
                                            --   observed EOF; done: at the stored length
  parsed_upto       INTEGER NOT NULL,       -- byte offset in the chain after the last complete record parsed
  at_length        INTEGER CHECK(at_length >= 0),  -- waiting, refused, failed: the stored length it was decided at
  records           INTEGER NOT NULL DEFAULT 0,  -- every record of this chain read so far:
  used              INTEGER NOT NULL DEFAULT 0,  --   records = used + skipped + unreadable; an append adds
  skipped           INTEGER NOT NULL DEFAULT 0,  --   to the counts, a whole parse replaces them; written in the
  unreadable        INTEGER NOT NULL DEFAULT 0,  --   same transaction as the chain's last session
  commit_seq INTEGER CHECK(commit_seq > 0), -- the committed parse checkpoint, never a read position
  checkpoint BLOB,                        -- versioned opaque parser state at this cursor
  reason            TEXT,                   -- refused: the reason; failed: the error
  stopped_attempt BLOB CHECK(stopped_attempt IS NULL OR status IN ('refused','failed')),
  PRIMARY KEY (input_pk, chain),
  CHECK((status IN ('waiting','refused','failed')) = (at_length IS NOT NULL)),
  CHECK((status IN ('refused','failed')) = (reason IS NOT NULL))
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
-- by what the session records. Commit details come from stored git-lookup
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
CREATE INDEX commit_link_block ON commit_link(evidence_block_pk) WHERE evidence_block_pk IS NOT NULL;  -- the foreign-key check when blocks are deleted

CREATE TABLE pr_link (
  session_pk        INTEGER NOT NULL REFERENCES session(pk),
  repo              TEXT NOT NULL,
  number            INTEGER NOT NULL,
  url               TEXT,
  evidence          TEXT NOT NULL,          -- pr-link-record | gh-output | codex
  PRIMARY KEY (session_pk, repo, number)
) STRICT, WITHOUT ROWID;


-- One per repository directory a session worked in, from the
-- polylogue-repo-lookup inputs the engine writes. A session belongs to the lookup whose path is the
-- longest one at or above its cwd; the writer copies the identity onto the
-- session (repo_root, and repo where no record names one).
CREATE TABLE repo_lookup (
  path              TEXT PRIMARY KEY,       -- the directory holding .git
  remote            TEXT,                   -- origin's URL without credentials; NULL when there is none
  root_commits      TEXT NOT NULL,          -- JSON array, sorted, shallow boundaries left out (common git-identity); [] when unread
  input_pk          INTEGER NOT NULL REFERENCES input_ref(pk),
  chain             INTEGER NOT NULL
) STRICT, WITHOUT ROWID;




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

CREATE TABLE session_time (
  session_pk INTEGER NOT NULL REFERENCES session(pk) ON DELETE CASCADE,
  party TEXT NOT NULL CHECK(party IN (@names(Party))),
  key TEXT NOT NULL,
  ms INTEGER NOT NULL CHECK(ms >= 0),
  PRIMARY KEY(session_pk,party,key)
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
  at INTEGER,                               -- its source time, as message.at, at_until and at_precision
  at_until INTEGER,
  at_precision TEXT CHECK (at_precision IN (@names(Precision))),
  authorship TEXT NOT NULL,
  version_hash BLOB NOT NULL,
  input_pk INTEGER NOT NULL REFERENCES input_ref(pk),
  source_chain INTEGER NOT NULL,
  source_start INTEGER NOT NULL,
  source_end INTEGER NOT NULL,
  fields BLOB NOT NULL, -- the shown version without its blocks, zstd JSON
  changed INTEGER NOT NULL,
  CHECK ((at IS NULL) = (at_precision IS NULL)),
  CHECK (at_until IS NULL OR (at_precision <> 'instant' AND at_until > at)),
  CHECK (at_until IS NOT NULL OR at_precision IS NULL OR at_precision IN ('instant','day','month','year'))
) STRICT;
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

-- One counter per import that moves whenever its members, asset metadata or
-- sidecars change; a read binds it instead of re-reading the members.
CREATE TABLE import_change (
  import_id TEXT PRIMARY KEY,
  changed INTEGER NOT NULL
) STRICT, WITHOUT ROWID;
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

-- First/last script-command transitions use this keyed presence lookup.
CREATE INDEX tool_call_script ON tool_call(session_pk) WHERE run_by_script = 1;
-- Session counts are applied once by the transaction's ChangeSummary after
-- messages, blocks, tool pairing and compaction events agree.

-- How many rows the tables status counts hold, kept by the triggers below so
-- that status does not scan the tables. A build's composition runs without
-- them and counts each table once when it ends.
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
INSERT INTO row_count(name, n) VALUES ('search_pending', 0);
CREATE TRIGGER search_pending_added AFTER INSERT ON search_pending BEGIN UPDATE row_count SET n = n + 1 WHERE name = 'search_pending'; END;
CREATE TRIGGER search_pending_removed AFTER DELETE ON search_pending BEGIN UPDATE row_count SET n = n - 1 WHERE name = 'search_pending'; END;
-- Input chains by parse status, kept the same way.
CREATE TABLE parse_status_count (
  status            TEXT PRIMARY KEY,
  n                 INTEGER NOT NULL
) STRICT, WITHOUT ROWID;
CREATE TRIGGER parse_state_added AFTER INSERT ON parse_state BEGIN
  INSERT INTO parse_status_count(status, n) VALUES (NEW.status, 1) ON CONFLICT(status) DO UPDATE SET n = n + 1;
END;
CREATE TRIGGER parse_state_removed AFTER DELETE ON parse_state BEGIN
  UPDATE parse_status_count SET n = n - 1 WHERE status = OLD.status;
END;
CREATE TRIGGER parse_state_changed AFTER UPDATE OF status ON parse_state WHEN OLD.status IS NOT NEW.status BEGIN
  UPDATE parse_status_count SET n = n - 1 WHERE status = OLD.status;
  INSERT INTO parse_status_count(status, n) VALUES (NEW.status, 1) ON CONFLICT(status) DO UPDATE SET n = n + 1;
END;
-- How far each stored revision chain is parsed, one row per chain of the
-- source, kept by the writer in the transaction of the change
-- (crates/store/src/chain_progress.rs), and counted below so that whether
-- the index has caught up is read without walking the chains. A group
-- without chains has no count row, so equal chains give equal rows however
-- they were reached.
CREATE TABLE chain_progress (
  input             TEXT NOT NULL,          -- archive.db input.id
  chain             INTEGER NOT NULL,
  origin            TEXT NOT NULL,          -- the format the chain is read as
  parse_version   TEXT NOT NULL,          -- of its parse state; '' without one
  state             TEXT NOT NULL CHECK(state IN ('no_checkpoint','current','no_parser','partial','behind')),
  PRIMARY KEY (input, chain)
) STRICT, WITHOUT ROWID;
CREATE TABLE chain_progress_count (
  origin            TEXT NOT NULL,
  parse_version   TEXT NOT NULL,
  state             TEXT NOT NULL,
  n                 INTEGER NOT NULL,
  PRIMARY KEY (origin, parse_version, state)
) STRICT, WITHOUT ROWID;
CREATE TRIGGER chain_progress_added AFTER INSERT ON chain_progress BEGIN
  INSERT INTO chain_progress_count(origin, parse_version, state, n) VALUES (NEW.origin, NEW.parse_version, NEW.state, 1)
    ON CONFLICT(origin, parse_version, state) DO UPDATE SET n = n + 1;
END;
CREATE TRIGGER chain_progress_removed AFTER DELETE ON chain_progress BEGIN
  UPDATE chain_progress_count SET n = n - 1 WHERE origin = OLD.origin AND parse_version = OLD.parse_version AND state = OLD.state;
  DELETE FROM chain_progress_count WHERE origin = OLD.origin AND parse_version = OLD.parse_version AND state = OLD.state AND n = 0;
END;
CREATE TRIGGER chain_progress_changed AFTER UPDATE OF origin, parse_version, state ON chain_progress BEGIN
  UPDATE chain_progress_count SET n = n - 1 WHERE origin = OLD.origin AND parse_version = OLD.parse_version AND state = OLD.state;
  DELETE FROM chain_progress_count WHERE origin = OLD.origin AND parse_version = OLD.parse_version AND state = OLD.state AND n = 0;
  INSERT INTO chain_progress_count(origin, parse_version, state, n) VALUES (NEW.origin, NEW.parse_version, NEW.state, 1)
    ON CONFLICT(origin, parse_version, state) DO UPDATE SET n = n + 1;
END;
-- Records parsers did not read, summed by what they were, kept the same way.
CREATE TABLE unused_count (
  what              TEXT PRIMARY KEY,
  n                 INTEGER NOT NULL
) STRICT, WITHOUT ROWID;
CREATE TRIGGER unused_added AFTER INSERT ON unused BEGIN
  INSERT INTO unused_count(what, n) VALUES (NEW.what, NEW.count) ON CONFLICT(what) DO UPDATE SET n = n + excluded.n;
END;
CREATE TRIGGER unused_removed AFTER DELETE ON unused BEGIN
  UPDATE unused_count SET n = n - OLD.count WHERE what = OLD.what;
END;
CREATE TRIGGER unused_changed AFTER UPDATE OF what, count ON unused BEGIN
  UPDATE unused_count SET n = n - OLD.count WHERE what = OLD.what;
  INSERT INTO unused_count(what, n) VALUES (NEW.what, NEW.count) ON CONFLICT(what) DO UPDATE SET n = n + excluded.n;
END;

-- Recomputed comparison state for tool-database inputs.
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

-- Closed intervals of a session's activity, wait or stall, computed from
-- its live events; an interval stays open (in the session's live facts) until
-- a live event closes it.
CREATE TABLE live_interval (
  session_pk INTEGER NOT NULL REFERENCES session(pk),
  aspect     TEXT NOT NULL,                 -- activity | waiting_on | stall
  value      TEXT NOT NULL,
  start_at   INTEGER NOT NULL,
  end_at     INTEGER NOT NULL,
  closed_by  TEXT NOT NULL,                 -- the kind of the live event that closed it
  closed     BLOB NOT NULL,                 -- that event's place in the live summary (summary/live.rs)
  PRIMARY KEY (session_pk, aspect, start_at)
) STRICT;
CREATE INDEX live_interval_end ON live_interval(end_at);
CREATE INDEX live_interval_closed ON live_interval(session_pk, closed);

-- Processes of live sessions from the process watcher's lines: the harness
-- root and its descendants, one row per process identity.
CREATE TABLE process (
  session_pk INTEGER NOT NULL REFERENCES session(pk),
  boot_id    TEXT NOT NULL,
  pid        INTEGER NOT NULL,
  start      INTEGER NOT NULL,              -- clock ticks after boot (field 22 of /proc/<pid>/stat)
  parent_pid INTEGER,
  comm       TEXT,
  argv       TEXT,                          -- JSON array
  cwd        TEXT,
  started_at INTEGER,
  ended_at   INTEGER,
  role       TEXT NOT NULL,                 -- harness | child
  opened     BLOB NOT NULL,                 -- the place in the live summary of the line that made the row
  ended      BLOB,                          --   and of the line that ended it
  PRIMARY KEY (session_pk, boot_id, pid, start)
) STRICT;

-- Every session ID readings state that a shown session is shown as: its own
-- ID, unless exactly one other session records it as an alias, and each
-- alias that it alone records (writer/identity.rs).
CREATE VIEW session_stated AS
SELECT s.pk AS session_pk, s.id AS stated_id FROM session s
WHERE (SELECT count(DISTINCT a.session_id) FROM session_alias a WHERE a.alias = s.id AND a.session_id <> s.id) <> 1
UNION ALL
SELECT s.pk, a.alias FROM session s JOIN session_alias a ON a.session_id = s.id AND a.alias <> s.id
WHERE (SELECT count(DISTINCT b.session_id) FROM session_alias b WHERE b.alias = a.alias AND b.session_id <> b.alias) = 1
GROUP BY s.pk, a.alias;

-- Each reading a shown message is composed from: those naming its ID, and
-- those of a copy its session is shown for, whose content-named messages
-- carry the copy's ID where the shown one carries the session's
-- (writer/identity.rs).
CREATE VIEW shown_message_parse AS
SELECT m.pk AS message_pk, r.* FROM message m JOIN held_message_parse r ON r.message_id = m.id
UNION ALL
SELECT m.pk, r.* FROM message m JOIN session s ON s.pk = m.session_pk
JOIN session_stated x ON x.session_pk = s.pk AND x.stated_id <> s.id
JOIN held_message_parse r ON r.session_id = x.stated_id
  AND r.message_id = x.stated_id || substr(m.id, length(s.id) + 1)
WHERE substr(m.id, 1, length(s.id) + 1) = s.id || '~';

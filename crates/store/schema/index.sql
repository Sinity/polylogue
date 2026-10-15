CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL) STRICT;
-- INDEX_VERSION, generation, parser_generation, the write counter, unsynced, facts_version,
-- the status counters count.sessions, count.messages, count.tool_calls,
-- count.tool_calls_failed, count.unrecognized (behavior.md, status totals),
-- preamble_set, boot_id (the boot the file was last opened in), clean_stop
-- (1 after an orderly stop, removed at start), last_stop (signal name or
-- `clean`). No cursors for derived products: a product's work is the sessions
-- whose stored version differs (convergence.md, Derive).

-- Inputs as index.db sees them: a small local key for each archive.db input.
CREATE TABLE input_ref (
  pk                INTEGER PRIMARY KEY,
  id                TEXT NOT NULL UNIQUE,   -- archive.db input.id
  excised           INTEGER NOT NULL DEFAULT 0  -- 1 once an input excision removed the input; the row stays (above)
) STRICT;

CREATE TABLE session (
  pk                INTEGER PRIMARY KEY,
  id                TEXT NOT NULL UNIQUE,   -- '<origin>:<native id>'
  origin            TEXT NOT NULL,
  native_id         TEXT NOT NULL,
  kind              TEXT NOT NULL,          -- main | subagent | chat (continuations are links, not a kind)
  stub              INTEGER NOT NULL DEFAULT 0,  -- 1: created by a hook or process line naming a session that has no
                                            --   transcript yet, so its live state exists; cleared when its messages arrive
  title             TEXT,                   -- provider title only
  title_at          INTEGER,                -- the provider time of the title (model.md, Session, rule 6 input)
  attrs_json        TEXT,                   -- provider fields with no column (account IDs, job facts: name, intent,
                                            --   state); never exported where the spec says so
  started_at        INTEGER,                -- from the messages and events the session counts as its own (model.md,
                                            --   Session: a copy counts while its source is absent)
  last_activity_at  INTEGER,
  cwd               TEXT,
  repo              TEXT,                   -- name or URL, for display
  checkout          TEXT,                   -- worktree path, when it differs from cwd's repository root
  collection_id     TEXT,                   -- provider grouping: ChatGPT project or custom GPT, claude.ai project
  collection_name   TEXT,
  branch            TEXT,
  "commit"          TEXT,
  harness           TEXT,
  harness_version   TEXT,
  machine           TEXT,                   -- machine.id the session ran on, when known (model.md, Session)
  team              TEXT,                   -- agent team (Claude Code teamName), when the session ran in one
  agent_name        TEXT,                   -- its name in the team or its subagent name, as given
  temporary         INTEGER NOT NULL DEFAULT 0,
  -- Live facts (model.md, Live state): updated in the session's write transaction
  -- when a hook line or process line for it is written, so they follow data, not
  -- the clock; the live state itself is computed at read time from these and now.
  live_last_event   TEXT,                   -- type of the latest hook event
  live_last_event_at INTEGER,
  live_root         TEXT,                   -- process identity 'boot_id:pid:start' of the current harness process
  live_root_alive   INTEGER,                -- 1 | 0 | NULL (no process evidence), from the latest process line
  live_open_tool_pk INTEGER,                -- tool_call with a pre-tool event and no post-tool event yet
  live_waiting_on   TEXT,                   -- permission | question | notification | NULL
  live_waiting_since INTEGER,
  live_todo         TEXT,                   -- the harness's in-progress todo item, as text (Claude Code: the last TodoWrite)
  live_task         TEXT,                   -- job ID or Bead ID (model.md, live task)
  live_task_evidence TEXT,                  -- job_env | claim_command
  changed           INTEGER NOT NULL,       -- write counter value of the last write to this session's records
                                            --   or derived products (not its live columns)
  live_changed      INTEGER                 -- write counter value of the last change to its live columns
) STRICT;
CREATE INDEX session_activity ON session(last_activity_at);   -- default order, since:
CREATE INDEX session_started  ON session(started_at);         -- started between, until:
CREATE INDEX session_changed  ON session(changed);            -- change feed; derive step
CREATE INDEX session_cwd      ON session(cwd);                -- cwd: prefix as a range; briefings; resume
CREATE INDEX session_repo     ON session(repo) WHERE repo IS NOT NULL;
CREATE INDEX session_collection ON session(collection_id) WHERE collection_id IS NOT NULL;
CREATE INDEX session_live ON session(live_last_event_at) WHERE live_root_alive = 1 OR live_waiting_on IS NOT NULL;   -- running:, waiting:, the overview
CREATE INDEX session_live_checkout ON session(checkout) WHERE live_root_alive = 1;   -- peers (repository from session_facts)

CREATE TABLE message (
  pk                INTEGER PRIMARY KEY,
  id                TEXT NOT NULL UNIQUE,   -- model.md Identity; inherited marking matches on it
  session_pk        INTEGER NOT NULL REFERENCES session(pk),
  seq               INTEGER NOT NULL,       -- display position (convergence.md, ordering key)
  role              TEXT NOT NULL,          -- user | assistant | system | tool (sources/ maps each provider's role)
  written_by        TEXT NOT NULL,          -- human | model | harness | unknown
  prompt_kind       TEXT,                   -- written_by human: prose | command (model.md, Message)
  model             TEXT,
  at                INTEGER,                -- provider time
  completed_at      INTEGER,                -- a message from several timed records: the last record's time
  at_quality        TEXT,                   -- time_quality of `at` (model.md, Time): intrinsic | file_mtime | unknown
  at_precision      TEXT,                   -- instant | second | minute | day | month | year | bounded
  at_until          INTEGER,                -- the end of a bounded `at`; NULL otherwise
  first_seen        INTEGER,                -- when Polylogue first had its bytes (convergence.md); NULL for at
                                            --   most a minute, until an observation covers them
  parent_id         TEXT,                   -- natural message id of the tree parent (may arrive later)
  input_pk          INTEGER NOT NULL REFERENCES input_ref(pk),  -- input the stored (preferred) version came from
  source_chain      INTEGER,                -- the revision chain the byte range is in (archive.db revision.chain):
                                            --   offsets are stable only within a chain, which only appends
  source_start      INTEGER,                -- byte range of its records in that chain (export `sources[0]`):
  source_end        INTEGER,                --   raw view and outside anchors; for a JSON export, the conversation object
  id_kind           TEXT,                   -- provider | provisional (model.md, provider and provisional IDs)
  native_message_id TEXT,                   -- the provider's own message ID as written (a ChatGPT node ID, a uuid), stored
                                            --   whole so the derive step can find a copy across sessions; never split from id
  response_id       TEXT,                   -- a model message: the provider's response ID (Claude Code message.id)
  request_id        TEXT,                   --   and request ID (requestId), so the derive step can match a response a
                                            --   fork re-emits under new uuids (decisions/lineage.md); NULL elsewhere
  version_hash      BLOB,                   -- BLAKE3 of the preferred version's records as canonical JSON without
                                            --   volatile fields: merge rule 2's last tie-break, decided the same way
                                            --   whichever input arrives first (law 4; sources/README.md, rule 2)
  in_progress       INTEGER NOT NULL DEFAULT 0, -- the provider marked this version as still being produced (model.md)
  text_origin_json TEXT,                   -- revision-bound evidence object and proxy metrics (model.md, Text-origin evidence)
  on_active_path    INTEGER,                -- 1 on the path, 0 off, NULL unknown (no marked leaf; model.md);
                                            --   always written, no default
  inherited_from    TEXT,                   -- natural message id of the copy's source as evidence names it, present or not
  copy_origin       TEXT,                   -- known | unknown for a copy (unknown: provider says copy, names no source)
  counted_as        TEXT,                   -- for a copy, the message counted in its place (model.md, Message)
  from_session      TEXT,                   -- agent-to-agent message: the sending session, when resolved
  from_name         TEXT,                   --   and the sender's name as the provider wrote it
  attrs_json        TEXT,                   -- provider fields kept as given with no column of their own
                                            --   (attribution, effort, Codex phase, citations, speech or
                                            --   recap markers); one JSON object, export field `attrs`
  usage_present     INTEGER NOT NULL DEFAULT 0, -- distinguishes a reported empty usage object from no usage
  input_tokens      INTEGER,                -- NULL = not reported (never 0 for unknown)
  output_tokens     INTEGER,
  cache_read_tokens INTEGER,
  cache_write_tokens INTEGER,               -- all cache writes
  cache_write_5m_tokens INTEGER,            -- the part written with a 5-minute lifetime, when reported
  cache_write_1h_tokens INTEGER,            -- the part written with a 1-hour lifetime, when reported
  reasoning_tokens  INTEGER,
  web_requests      INTEGER,                -- server-side web search and fetch requests
  speed             TEXT,                   -- e.g. 'fast', when the provider reports a mode priced differently
  service_tier      TEXT,                   -- batch | flex | priority | ..., when recorded
  provider_cost_nusd INTEGER,               -- the provider's own cost figure, when given
  cost_nusd         INTEGER,                -- the one per-message cost (analysis.md); NULL = unknown
  stop_reason       TEXT,
  duration_ms       INTEGER,
  words             INTEGER NOT NULL DEFAULT 0, -- words in its text blocks
  parser_version    INTEGER NOT NULL,       -- version of the parser that produced its winning version; a session
                                            --   rebuilt in place deletes, at its last step, the rows still at the
                                            --   old version of the bumped format (convergence.md)
  UNIQUE (pk, session_pk)                   -- lets children state their owner once (R73)
) STRICT;
CREATE INDEX message_session ON message(session_pk, seq);     -- reading; [last:N]; open at a position
CREATE INDEX message_prose   ON message(session_pk) WHERE written_by = 'human' AND prompt_kind = 'prose';
CREATE INDEX message_unseen  ON message(input_pk) WHERE first_seen IS NULL;   -- parse work: first_seen to fill
CREATE INDEX message_branch  ON message(session_pk) WHERE on_active_path = 0;
CREATE INDEX message_copy    ON message(inherited_from) WHERE inherited_from IS NOT NULL;   -- derive: copies of a changed source
CREATE INDEX message_response ON message(response_id, request_id) WHERE response_id IS NOT NULL;   -- derive: re-emitted responses
CREATE INDEX message_native ON message(native_message_id) WHERE native_message_id IS NOT NULL;      -- derive: copies by node ID

-- Every stored version that contributed to a message: one row per input chain
-- whose records hold a version of it (the preferred version's row repeats the
-- message's own columns). A citation names a row here, never just an input and
-- offsets, because a rewritten input's new chain puts other bytes at the same
-- offsets (an outside review, 2026-10-12). Rebuilt with the message.
CREATE TABLE message_source (
  message_pk        INTEGER NOT NULL REFERENCES message(pk),
  input_pk          INTEGER NOT NULL REFERENCES input_ref(pk),
  chain             INTEGER NOT NULL,       -- archive.db revision.chain
  source_start      INTEGER NOT NULL,
  source_end        INTEGER NOT NULL,
  preferred         INTEGER NOT NULL,       -- 1 for the version the message shows (model.md, preferred version)
  PRIMARY KEY (message_pk, input_pk, chain)
) STRICT;
CREATE INDEX message_source_input ON message_source(input_pk, chain);   -- foreign-key check; excision scope

-- Text is stored once per distinct content; `id` is the rowid of its search entries.
CREATE TABLE text (
  id                INTEGER PRIMARY KEY,
  hash              BLOB NOT NULL UNIQUE,   -- BLAKE3 of the UTF-8 text; embeddings are keyed by it
  body              TEXT NOT NULL
) STRICT;

-- What the provider wrote, one row per content item. Facts derived about a tool
-- call (tool, command, outcome, exit code, duration) belong to tool_call alone.
CREATE TABLE block (
  pk                INTEGER PRIMARY KEY,
  message_pk        INTEGER NOT NULL,
  session_pk        INTEGER NOT NULL,       -- the message's session, enforced by the composite key below
  idx               INTEGER,                -- position in the content list: Claude Code counts every item, thinking included;
                                            --   Codex counts non-reasoning items and reasoning in reasoning_idx; id = message id:idx
  reasoning_idx     INTEGER,                -- formats whose rewrites drop reasoning (Codex): position among its reasoning items; id = message id:r<reasoning_idx>
  attach_key        TEXT,                   -- attachment listed outside the content list: 'a:<id>' or 'a:~<hash>-<n>'
                                            --   (model.md Identity); exactly one of idx, reasoning_idx and attach_key is set
  kind              TEXT NOT NULL,          -- text | thinking | tool_call | tool_result | image | file | other
  text_id           INTEGER REFERENCES text(id),
  input_pk          INTEGER REFERENCES input_ref(pk),  -- explicit contributor or lazy value locator; otherwise all four NULL
  source_chain      INTEGER,                -- archive.db revision.chain of that contributor, never its current chain
  source_start      INTEGER,                -- byte range in that exact input and chain, including an empty value
  source_end        INTEGER,                --   field from another input or a value too large to index (blobs.md)
  call_id           TEXT,                   -- tool_call and tool_result: the provider's call id (unique only in a session)
  input_json        TEXT,                   -- tool_call: arguments as given (export field `input`)
  blob_hash         BLOB,                   -- image/file, large tool output: bytes in the blob store; export `blob`
  direction         TEXT,                   -- image/file: input | output | unknown
  member            TEXT,                   -- image/file inside an archive stored whole: 'zip:<input>/<member>', read in
                                            --   place (blobs.md); blob_hash is then NULL
  produced_by_pk    INTEGER,                -- image/file output: the block of the tool call or image request that made it
  media_type        TEXT,
  file_name         TEXT,                   -- export `name`
  size              INTEGER,                -- image/file
  url               TEXT,                   -- image/file downloaded from a link
  signature         TEXT,                   -- thinking
  encrypted         INTEGER NOT NULL DEFAULT 0,  -- thinking kept encrypted
  other_type        TEXT,                   -- other: the provider's content type name; export `type`
  data_json         TEXT,                   -- other: payload as given; encrypted thinking data; tool_call and
                                            --   tool_result: structured tool output kept as given (export `data`)
  result_outcome    TEXT,                   -- tool_result: provider fact, retained even without a paired call
  result_reason     TEXT,                   -- unknown/error reason in the exported vocabulary
  result_exit_code  INTEGER,
  result_duration_ms INTEGER,
  looks_like_secret INTEGER NOT NULL DEFAULT 0, -- has:secret (behavior.md, privacy)
  CHECK ((input_pk IS NULL AND source_chain IS NULL AND source_start IS NULL AND source_end IS NULL)
      OR (input_pk IS NOT NULL AND source_chain IS NOT NULL AND source_start IS NOT NULL AND source_end IS NOT NULL
          AND source_chain > 0 AND source_start >= 0 AND source_end >= source_start)),
  FOREIGN KEY (message_pk, session_pk) REFERENCES message(pk, session_pk),
  UNIQUE (pk, session_pk)
) STRICT;
CREATE UNIQUE INDEX block_message ON block(message_pk, idx);
CREATE UNIQUE INDEX block_attach ON block(message_pk, attach_key) WHERE attach_key IS NOT NULL;
CREATE INDEX block_text   ON block(text_id) WHERE text_id IS NOT NULL;   -- search hit -> blocks
CREATE INDEX block_secret ON block(message_pk) WHERE looks_like_secret = 1;

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

-- One row per tool call, paired with its result inside the same session. The
-- owner of derived call summaries; raw result facts remain on their block.
CREATE TABLE tool_call (
  pk                INTEGER PRIMARY KEY,
  session_pk        INTEGER NOT NULL,       -- kept for index speed; the composite keys make it agree with its blocks
  call_block_pk     INTEGER NOT NULL UNIQUE,
  result_block_pk   INTEGER,                -- NULL until the result is seen
  at                INTEGER,
  tool              TEXT NOT NULL,
  action            TEXT NOT NULL,          -- shell | code | file_read | file_write | file_edit | search | web | subagent | mcp | other
  command           TEXT,                   -- as given
  command_norm      TEXT,                   -- 'cargo test', 'git commit' (analysis.md)
  path              TEXT,                   -- file path argument, when the tool has one
  outcome           TEXT NOT NULL,          -- ok | error | unknown: facts of the records, never of the clock;
                                            --   `pending` is computed at read time from live state (model.md)
  outcome_reason    TEXT,                   -- a reason code from model.md (no_result, background_never_reported, ...)
  exit_code         INTEGER,
  duration_ms       INTEGER,                -- NULL (unknown) when the times are equal or reversed
  child_session_id  TEXT,                   -- natural id of a spawned subagent (may arrive later)
  inherited         INTEGER NOT NULL DEFAULT 0, -- part of an inherited message
  FOREIGN KEY (call_block_pk, session_pk) REFERENCES block(pk, session_pk),
  FOREIGN KEY (result_block_pk, session_pk) REFERENCES block(pk, session_pk)
) STRICT;
CREATE INDEX tool_call_session ON tool_call(session_pk, at);   -- exists, seq(...), the changes view
CREATE INDEX tool_call_tool    ON tool_call(tool);
CREATE INDEX tool_call_action  ON tool_call(action, outcome);
CREATE INDEX tool_call_failed  ON tool_call(session_pk) WHERE outcome <> 'ok';
CREATE INDEX tool_call_path    ON tool_call(path) WHERE path IS NOT NULL;
CREATE INDEX tool_call_repo_command ON tool_call(command_norm, at) WHERE command_norm IS NOT NULL;   -- repo_command_last recompute
CREATE INDEX tool_call_result  ON tool_call(result_block_pk) WHERE result_block_pk IS NOT NULL;   -- foreign-key check

-- Substring search on short fields (command:, path:) without a scan. Unfolded
-- text, so it can read its content from tool_call. The writer issues the FTS5
-- `delete` command with the row's old `command` and `path` before deleting or
-- changing a `tool_call` row, in the same transaction.
CREATE VIRTUAL TABLE tool_call_trigram USING fts5(command, path,
    content='tool_call', content_rowid='pk', tokenize='trigram');

-- Where two versions of one message disagree (sources/README.md, rule 2; model.md,
-- within-input conflicts; attachment conflicts): written by the writer in the
-- merge's transaction, rebuilt with the file. `status` counts them per input.
CREATE TABLE message_disagreement (
  message_pk        INTEGER NOT NULL REFERENCES message(pk),
  field             TEXT NOT NULL,          -- the field that differs (role, text, attachment:<id>, ...)
  other_input_pk    INTEGER NOT NULL REFERENCES input_ref(pk),  -- the input holding the version not kept
  other_value       TEXT,                   -- a short value (under 200 bytes) or the BLAKE3 of the longer one
  PRIMARY KEY (message_pk, field, other_input_pk)
) STRICT, WITHOUT ROWID;
CREATE INDEX message_disagreement_input ON message_disagreement(other_input_pk);   -- status; foreign-key check

CREATE TABLE link (
  pk                INTEGER PRIMARY KEY,
  parent_id         TEXT NOT NULL,          -- natural session id (may not be ingested yet)
  child_id          TEXT NOT NULL,          -- natural session id of the child (may not be ingested yet: a parent's
                                            --   thread_spawn_edges, continued-in or job children name it first)
  child_pk          INTEGER REFERENCES session(pk),  -- set once the child is stored; NULL while the link is pending
  kind              TEXT NOT NULL,          -- subagent | continuation | reference (model.md, Link)
  how               TEXT NOT NULL,          -- subagent: fresh | copied | summarized; continuation: resume | fork
                                            -- | after_compaction; reference: briefing | ping | mention
  branch_point      TEXT,                   -- natural message id where the child branches off
  spawned_by        TEXT,                   -- natural block id of the parent's spawning tool call; resolved by the
                                            --   derive step from spawning_call once the parent session is stored
  spawning_call     TEXT,                   -- the spawning call's ID as the evidence names it (a tool-use ID)
  spawn_evidence    TEXT,                   -- which input field named that call: meta.toolUseId, or the parent's
                                            --   toolUseResult.agentId (model.md, link evidence)
  evidence          TEXT NOT NULL,          -- which input field established the parent (model.md's closed list)
  input_pk          INTEGER NOT NULL REFERENCES input_ref(pk),  -- the input holding that field
  parser_version    INTEGER NOT NULL,       -- with input_pk: which parse produced it (convergence.md, rebuilding a session)
  attrs_json        TEXT,                   -- provider facts as given: agent type, description, role, depth,
                                            --   worktree, stopped by user, Codex edge status
  retry_of          TEXT,                   -- natural block id of the spawning call this one retries (harness evidence only)
  result_use        TEXT,                   -- returned | referenced | unknown (model.md, Link)
  UNIQUE (parent_id, child_id, kind, how)
) STRICT;
CREATE INDEX link_child  ON link(child_pk);                    -- lineage up
CREATE INDEX link_child_id ON link(child_id) WHERE child_pk IS NULL;   -- late children
CREATE INDEX link_parent ON link(parent_id, kind);             -- lineage down; late parents

CREATE TABLE event (
  pk                INTEGER PRIMARY KEY,
  id                TEXT NOT NULL UNIQUE,   -- computed by the parser from the raw record (model.md, Identity)
  session_pk        INTEGER NOT NULL REFERENCES session(pk),  -- every event names its session; a hook line naming a
                                            --   session with no row yet creates a stub session row (session.stub)
  seq               INTEGER NOT NULL,       -- display position (shares numbering with message.seq); not a key
  after_message_pk  INTEGER REFERENCES message(pk),  -- the message it follows; export `after_message`
  kind              TEXT NOT NULL,          -- open set; each provider's kinds are listed in sources/
                                            -- (compaction, rate_limit, hook_run, ...), plus marker_not_applied
  at                INTEGER,
  model_saw         INTEGER NOT NULL DEFAULT 0,  -- 1 if this content was part of the model's context
  input_pk          INTEGER NOT NULL REFERENCES input_ref(pk),
  parser_version    INTEGER NOT NULL,       -- with input_pk: which parse produced it (convergence.md, rebuilding a session)
  source_chain      INTEGER,                -- revision chain of the byte range (as message.source_chain)
  source_start      INTEGER,
  source_end        INTEGER,
  data_json         TEXT NOT NULL           -- the provider's payload as given; export `data`
) STRICT;
CREATE INDEX event_session ON event(session_pk, seq);
CREATE INDEX event_after   ON event(after_message_pk) WHERE after_message_pk IS NOT NULL;   -- foreign-key check
CREATE INDEX event_kind    ON event(kind, at);

-- Context Polylogue gave to sessions (model.md, Delivery), parsed from the
-- `delivery` lines the sender appends to the hook spool (sources/README.md). A line
-- written twice (a retried recording) has the same id and is one row.
CREATE TABLE delivery (
  pk                INTEGER PRIMARY KEY,
  id                TEXT NOT NULL UNIQUE,   -- UUIDv7 chosen at sending; also in the injected reference line
  recipient_kind    TEXT NOT NULL,          -- session | operator (a signal notice to status or the notifier)
  recipient_id      TEXT,                   -- session id when recipient_kind = session (the session may arrive later)
  sender            TEXT NOT NULL,          -- hook | mcp | ping | signal
  event             TEXT NOT NULL,          -- session_start | resume | compaction | prompt_reference | mcp_brief | ping | advisory | signal
  rule_id           TEXT,                   -- signal deliveries: the signal note
  rows_json         TEXT NOT NULL DEFAULT '[]',   -- signal deliveries: references of the rows that fired it
  author            TEXT,                   -- signal deliveries: the rule's author
  at                INTEGER NOT NULL,
  content_hash      BLOB NOT NULL,          -- BLAKE3 of the delivered text; context is in the recipient transcript, signal notices in the delivery spool
  bytes             INTEGER NOT NULL,
  refs_json         TEXT NOT NULL,          -- references included, in order
  signal_rules_json TEXT NOT NULL DEFAULT '[]',   -- Polylogue signal rules that filled the watching section (user-data.md)
  machine_rules_json TEXT NOT NULL DEFAULT '[]',  -- Sinex signal rules that filled the machine section (sources/README.md, delivery lines)
  left_out_json     TEXT NOT NULL,          -- references and sections left out, each with its reason (`budget` or
                                            --   `failed:<kind>`) and the command to get each group
  outcome           TEXT NOT NULL,          -- delivered | failed | unknown | not_sent
  reason            TEXT,                   -- for failed and not_sent
  input_pk          INTEGER NOT NULL REFERENCES input_ref(pk),
  parser_version    INTEGER NOT NULL        -- with input_pk: which parse produced it (convergence.md, rebuilding a session)
) STRICT;
CREATE INDEX delivery_recipient ON delivery(recipient_id, at);
CREATE INDEX delivery_content   ON delivery(content_hash);

-- Tasks and attempts a harness records (model.md, Run).
CREATE TABLE run (
  pk                INTEGER PRIMARY KEY,
  id                TEXT NOT NULL UNIQUE,   -- '<origin>:<native run id>'
  kind              TEXT NOT NULL,          -- task | attempt
  task_id           TEXT,                   -- for an attempt (may arrive later)
  instruction       TEXT,                   -- as given
  state             TEXT NOT NULL,          -- as the harness recorded: running | completed | failed | cancelled | unknown
  started_at        INTEGER,
  ended_at          INTEGER,
  evidence          TEXT NOT NULL,          -- the input field that established it (sources/)
  input_pk          INTEGER NOT NULL REFERENCES input_ref(pk),
  parser_version    INTEGER NOT NULL        -- with input_pk: which parse produced it (convergence.md, rebuilding a session)
) STRICT;
CREATE INDEX run_task ON run(task_id) WHERE task_id IS NOT NULL;

CREATE TABLE run_session (
  run_pk            INTEGER NOT NULL REFERENCES run(pk),
  session_pk        INTEGER NOT NULL REFERENCES session(pk),
  PRIMARY KEY (run_pk, session_pk)
) STRICT, WITHOUT ROWID;
CREATE INDEX run_session_session ON run_session(session_pk);

-- Harness processes and their descendants (model.md, Process), from the process
-- identity in hook lines and the daemon's process spool (sources/README.md).
CREATE TABLE process (
  pk                INTEGER PRIMARY KEY,
  boot_id           TEXT NOT NULL,
  pid               INTEGER NOT NULL,
  start_ticks       INTEGER NOT NULL,       -- /proc/<pid>/stat field 22
  ppid              INTEGER,
  role              TEXT NOT NULL,          -- harness | child
  comm              TEXT,
  argv_json         TEXT,                   -- as observed; flagged by has:secret like tool commands (behavior.md)
  cwd               TEXT,
  started_at        INTEGER,
  ended_at          INTEGER,                -- NULL while alive
  end_kind          TEXT,                   -- exited (seen gone between two samples) | gone_while_down (found gone at
                                            --   daemon start or after a reboot) | NULL while alive (sources/README.md, process spool)
  ended_after_ms    INTEGER,                -- for gone_while_down: the last time it was seen alive, a lower bound on its end
  cpu_ms            INTEGER,                -- user plus system CPU time at the last sample before it ended; NULL while alive or unknown (a lower bound)
  rss_peak_kb       INTEGER,                -- largest resident size seen by the samples (a lower bound)
  usage_partial     INTEGER NOT NULL DEFAULT 0, -- 1 when a daemon restart lost part of the samples
  tool_call_pk      INTEGER REFERENCES tool_call(pk),  -- only when timing and ancestry both match (slice 6)
  input_pk          INTEGER NOT NULL REFERENCES input_ref(pk),
  parser_version    INTEGER NOT NULL,       -- with input_pk: which parse produced it (convergence.md, rebuilding a session)
  UNIQUE (boot_id, pid, start_ticks)
) STRICT;
CREATE INDEX process_alive ON process(pk) WHERE ended_at IS NULL;   -- live view, process trees

-- Which sessions a harness process ran, and when: one process can run several
-- sessions (Claude Code's /clear starts a new session in the same process).
CREATE TABLE process_session (
  process_pk        INTEGER NOT NULL REFERENCES process(pk),
  session_pk        INTEGER NOT NULL REFERENCES session(pk),
  from_at           INTEGER,                -- the first hook line naming this session from this process
  until_at          INTEGER,                -- the next session's first line in the same process, or the process end
  PRIMARY KEY (process_pk, session_pk)
) STRICT, WITHOUT ROWID;
CREATE INDEX process_session_session ON process_session(session_pk);   -- live view; foreign-key check
CREATE INDEX process_tool_call ON process(tool_call_pk) WHERE tool_call_pk IS NOT NULL;   -- foreign-key check

-- Which inputs produced which sessions, and with which parser version.
CREATE TABLE parsed (
  input_pk          INTEGER NOT NULL REFERENCES input_ref(pk),
  chain             INTEGER NOT NULL,       -- the revision chain parsed (blobs.md)
  session_pk        INTEGER NOT NULL REFERENCES session(pk),
  parser_version    INTEGER NOT NULL,
  PRIMARY KEY (input_pk, chain, session_pk)
) STRICT, WITHOUT ROWID;
CREATE INDEX parsed_session ON parsed(session_pk);             -- rebuild a session from all its inputs

-- How far each revision chain has been parsed, with which parser version, and
-- how it ended. Every chain of every input has a row once the loop has seen it,
-- so nothing is silently skipped.
CREATE TABLE parse_state (
  input_pk          INTEGER NOT NULL REFERENCES input_ref(pk),
  chain             INTEGER NOT NULL,
  parser_version    INTEGER NOT NULL,
  status            TEXT NOT NULL,          -- parsing: byte interpretation runnable; waiting_tail: inspected partial suffix
                                            --   awaits acquisition/admission change; finalizing: row work remains;
                                            --   finalization_pending is independently runnable with either byte state;
                                            --   done: at the acquired end and row finalization committed; retired: invalidated, bounded cell cleanup owed;
                                            --   companion: read with its main input
                                            --   (detail: that input's id); no_sessions | refused | failed | no_parser_yet
  excisions_seen    INTEGER NOT NULL DEFAULT 0,  -- number of `excised` observations of the input when parsed;
                                            --   fewer than archive.db holds means the chain is parsed again
  companions_json   TEXT,                   -- [[companion input id, revision], ...] the companion revisions this
                                            --   chain was parsed with; a companion with a newer revision means the
                                            --   chain is parsed again (convergence.md, companion files)
  acquired_upto     INTEGER NOT NULL DEFAULT 0 CHECK (acquired_upto >= 0), -- pinned contiguous acquisition end; extending
                                            --   the same verified prefix does not complete its new suffix
  inspected_upto    INTEGER NOT NULL DEFAULT 0 CHECK (inspected_upto >= 0), -- examined end, including an owed partial tail
  parsed_upto       INTEGER NOT NULL CHECK (parsed_upto >= 0), -- complete-record checkpoint; rows/state through here committed
  basis_digest      BLOB CHECK (basis_digest IS NULL OR length(basis_digest) = 32), -- interpretation namespace: retained input/
                                            --   chain, parser, companions and excisions; acquired end is tracked separately
  state_generation  INTEGER NOT NULL DEFAULT 0 CHECK (state_generation >= 0), -- 0: legacy inline state; positive: parser_cell
  state_step        INTEGER NOT NULL DEFAULT 0 CHECK (state_step >= 0), -- predecessor checked atomically with each row/state step
  finalization_pending INTEGER NOT NULL DEFAULT 0 CHECK (finalization_pending IN (0, 1)), -- independent of an owed partial tail;
                                            --   only the owning writer's actual remaining obligations may clear it
  carry             BLOB,                   -- bounded resume state at parsed_upto, including references to
                                            --   addressable disposable parser state where maps grow; not a
                                            --   monolithic copy of all preceding IDs (convergence.md, Parse); NULL = start over
  detail            TEXT,                   -- the reason for refused, failed and no_sessions; for
                                            --   no_parser_yet, the slice that adds the parser, or 'none planned'
  framed            INTEGER NOT NULL DEFAULT 0,  -- law 120's accounting for this chain (sources/README.md): records
  used              INTEGER NOT NULL DEFAULT 0,  --   framed = used + ignored + unrecognized + undecodable; an append
  ignored           INTEGER NOT NULL DEFAULT 0,  --   adds to the counts, a whole parse replaces them; counts are
  unrecognized_n    INTEGER NOT NULL DEFAULT 0,  --   checkpointed with the last row-writing transaction of each bounded parse step
  undecodable       INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (input_pk, chain),
  CHECK (parsed_upto <= inspected_upto AND inspected_upto <= acquired_upto),
  CHECK (state_generation = 0 OR (basis_digest IS NOT NULL AND carry IS NOT NULL AND length(carry) = 16)),
  CHECK (framed >= 0 AND used >= 0 AND ignored >= 0 AND unrecognized_n >= 0 AND undecodable >= 0
         AND framed = used + ignored + unrecognized_n + undecodable)
) STRICT, WITHOUT ROWID;
CREATE INDEX parse_state_status ON parse_state(status) WHERE status NOT IN ('done', 'companion');   -- status, waiting work

-- Disposable provider state, addressed by typed provider-owned keys. A read view
-- pins a real SQLite read snapshot (or an independently qualified immutable
-- state version), never a mutable generation/key pair alone. Interpretation
-- holds no write transaction. Its touched delta, output and checkpoint commit
-- together only after checking the index generation, basis, state_generation
-- and state_step predecessor. Keyed carry stores bounded references, not these
-- cells (the keyed reference is two big-endian i64s: generation, then step).
-- Same-prefix acquisition may grow while an older step computes; that
-- step cannot certify the new suffix. Prepared output/finalization have their
-- own basis-bound state and are not complete merely because these cells exist.
CREATE TABLE parser_cell (
  input_pk          INTEGER NOT NULL REFERENCES input_ref(pk),
  chain             INTEGER NOT NULL,
  state_generation  INTEGER NOT NULL CHECK (state_generation > 0),
  namespace         TEXT NOT NULL,
  key               BLOB NOT NULL,          -- provider-owned ordered codec, not JSON lexical order
  value             BLOB NOT NULL,          -- one addressable cell; admission charges its full live representation
  PRIMARY KEY (input_pk, chain, state_generation, namespace, key)
) STRICT, WITHOUT ROWID;
-- Generations come from a monotonic per-index meta.parser_generation allocator,
-- surviving head/input removal; recreating either key never reuses an old token.
-- Retired generations are reclaimed in bounded keysets. Excision/input removal
-- must drain their cells before removing input_ref; there is no unbounded FK
-- cascade and no orphan state outside the owning index lifecycle.

-- Record types and fields a parser did not understand, and record types it
-- deliberately only counts (sources/ lists them), per input: an append adds to
-- the input's rows, a whole re-parse replaces them.
CREATE TABLE unrecognized (
  input_pk          INTEGER NOT NULL REFERENCES input_ref(pk),  -- also the example input status shows
  chain             INTEGER NOT NULL,       -- the revision chain counted; a whole re-parse of a chain replaces its rows
  format            TEXT NOT NULL,
  what              TEXT NOT NULL,          -- 'record:<type>'; 'field:<json path>'; 'user:<first tag>' (a user record
                                            --   whose content starts with an unknown tag); 'invalid_utf8';
                                            --   'time_unparsed'; 'undecodable' (a record that is not JSON)
  count             INTEGER NOT NULL,
  counted_only      INTEGER NOT NULL DEFAULT 0,  -- 1: a known type kept only as a count, by design
  first_at          INTEGER,                -- the record time (or, without one, the input's first observation) of the
  last_at           INTEGER,                --   first and last occurrence in this input; status shows the min and max
  first_offset      INTEGER,                -- byte offset of the first occurrence in that chain
  PRIMARY KEY (input_pk, chain, what)
) STRICT, WITHOUT ROWID;
CREATE INDEX unrecognized_what ON unrecognized(format, what);  -- status: counts per format

-- A cache of the latest observation per path in archive.db (blobs.md,
-- observing a file), plus scan times. Rebuilt from archive.db when absent; never
-- the authority for which input a path belongs to.
CREATE TABLE file_cursor (
  path              TEXT PRIMARY KEY,
  input_pk          INTEGER NOT NULL REFERENCES input_ref(pk),
  device            INTEGER NOT NULL,       -- the "unchanged" key: device, inode, size, mtime_ns, ctime_ns
  inode             INTEGER NOT NULL,       --   (for a database, the -wal file's size and times too)
  size              INTEGER NOT NULL,
  mtime_ns          INTEGER NOT NULL,
  ctime_ns          INTEGER NOT NULL,
  wal_key           TEXT,
  quiet_since       INTEGER,                -- when it last changed: joining, the full check, snapshots
  last_seen         INTEGER NOT NULL        -- the last scan that found the file unchanged
) STRICT;

-- A database snapshot's page hashes: a cache, recomputable from its blobs.
CREATE TABLE page_map (
  input_pk          INTEGER NOT NULL REFERENCES input_ref(pk),
  revision          INTEGER NOT NULL,
  page              INTEGER NOT NULL,
  hash              BLOB NOT NULL,          -- first 16 bytes of BLAKE3 of the page
  PRIMARY KEY (input_pk, revision, page)
) STRICT, WITHOUT ROWID;

-- Marker IDs the current parse extracts (user-data.md, inline markers): a marker
-- note is in force while its ID is here. Rebuilt with the messages.
CREATE TABLE marker (
  note_id           TEXT PRIMARY KEY,       -- archive.db note.id
  message_pk        INTEGER NOT NULL REFERENCES message(pk),
  block_idx         INTEGER NOT NULL CHECK (block_idx >= 0), -- text block's content-list position
  byte_offset       INTEGER NOT NULL CHECK (byte_offset >= 0), -- UTF-8 offset of :: within that block's text
  UNIQUE (message_pk, block_idx, byte_offset) -- one extracted marker at one position
) STRICT;
CREATE INDEX marker_message ON marker(message_pk);

-- Sessions removed without a surviving row (rebuild or excision), for the change
-- feed. Emptied only when a new generation starts.
CREATE TABLE removed_session (
  session_id        TEXT PRIMARY KEY,
  changed           INTEGER NOT NULL        -- write counter of the removing transaction
) STRICT;
CREATE INDEX removed_session_changed ON removed_session(changed);

-- Git correlation (analysis.md): commits and pull requests linked to sessions
-- by evidence in the session. Commit details come from stored git-lookup
-- inputs, never from reading a repository at derive time (convergence.md).
CREATE TABLE commit_link (
  session_pk        INTEGER NOT NULL REFERENCES session(pk),
  commit_hash       TEXT NOT NULL,          -- full hash; the one place the commit is a key, so it is named
                                            --   commit_hash here and `commit` everywhere else
  repo              TEXT,
  evidence_block_pk INTEGER REFERENCES block(pk),  -- the git commit tool call; NULL for a trailer
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

-- Per-session facts (field meanings: analysis.md, session facts). Costs here
-- are sums of message.cost_nusd, never computed separately.
CREATE TABLE session_facts (
  session_pk             INTEGER PRIMARY KEY REFERENCES session(pk),
  version                TEXT NOT NULL,     -- meta.facts_version it was computed at
  from_changed           INTEGER NOT NULL,  -- the session's `changed` it was computed from; stale when either differs
  repo_root              TEXT,              -- root commit hash, the repository's identity (model.md), derived by joining
                                            --   the session's cwd to repo-lookup records (sources/README.md)
  messages               INTEGER NOT NULL,  -- own, active path: counted_as IS NULL and not a copy of unknown origin
                                            --   (a non-copy, or the copy counted while its source is absent; model.md)
  inherited_messages     INTEGER NOT NULL,  -- copies not counted in this session (counted_as set)
  human_prompts          INTEGER NOT NULL,  -- human messages with prompt_kind prose (model.md)
  human_commands         INTEGER NOT NULL,  -- human messages with prompt_kind command
  words_human            INTEGER NOT NULL,
  words_model            INTEGER NOT NULL,
  tool_calls             INTEGER NOT NULL,  -- excludes a code-mode wrapper whose children are counted; a wrapper with
                                            --   no children counts itself (sources/codex.md, code mode; query.md, Defaults)
  tool_errors            INTEGER NOT NULL,  -- the same rule
  tool_no_result         INTEGER NOT NULL,  -- calls with no result record; split into pending and unknown at read
  tool_unknown           INTEGER NOT NULL,  -- other unknown outcomes
  input_tokens           INTEGER,           -- NULL when unknown for every message
  output_tokens          INTEGER,
  cache_read_tokens      INTEGER,
  cache_write_tokens     INTEGER,
  reasoning_tokens       INTEGER,
  unknown_usage_messages INTEGER NOT NULL,
  cost_nusd              INTEGER,           -- known cost; NULL when nothing is priced
  unpriced_messages      INTEGER NOT NULL,  -- messages whose usage or price is unknown
  wall_ms                INTEGER,
  engaged_ms             INTEGER,
  has_thinking           INTEGER NOT NULL,
  has_attachments        INTEGER NOT NULL,
  has_branches           INTEGER NOT NULL,
  compactions            INTEGER NOT NULL,
  children               INTEGER NOT NULL,  -- sessions this one spawned (incoming subagent links)
  tree_cost_nusd         INTEGER,           -- own cost plus every descendant's own cost, each once (analysis.md)
  tree_unpriced          INTEGER NOT NULL,  -- sessions in the tree with partial or unknown cost, plus spawning
                                            --   calls whose child session is not in the archive (analysis.md)
  depth                  INTEGER NOT NULL,  -- 0 for a top-level session; subagent depth below it
  display_title          TEXT,              -- the rule in model.md, Session (`title`), without notes; the current
                                            --   `title` note is joined at read time, so a note never makes this row stale
  title_source           TEXT NOT NULL,     -- note | provider | index | job | instruction | first_prompt | id
  root_session_id        TEXT               -- top of the subagent tree, for root: and lineage: without recursion
) STRICT;
CREATE INDEX session_facts_cost ON session_facts(cost_nusd);
CREATE INDEX session_facts_repo_root ON session_facts(repo_root) WHERE repo_root IS NOT NULL;   -- repo: by identity, peers

-- The last run of each normalized command per repository: what briefings list as
-- unresolved failures and failure statistics read. Kept by the writer in the same
-- transaction as the session it writes: for each command that session ran, a row is
-- replaced only by a later run, compared by (at, then tool_call id; a run with no
-- time ranks below every timed one), so a late import of an older session changes
-- nothing. When the writer deletes a session's rows (a rewrite, a rebuild of that
-- session, an excision), every row pointing at one of its tool calls is recomputed
-- in the same transaction from the remaining runs of that (repo_root, command_norm),
-- read through tool_call_repo_command; rebuilt with the file. A law checks it equals
-- a full recompute after shuffled arrival and deletions (law 141). A table because the answer spans every session of a repository, which a
-- briefing cannot rescan within its budget.
CREATE TABLE repo_command_last (
  repo_root         TEXT NOT NULL,          -- repository identity (model.md); sessions without one use 'cwd:' || cwd
  command_norm      TEXT NOT NULL,          -- analysis.md, normalized command
  tool_call_pk      INTEGER NOT NULL REFERENCES tool_call(pk),
  at                INTEGER,
  outcome           TEXT NOT NULL,          -- ok | error | unknown, with tool_call.outcome_reason
  PRIMARY KEY (repo_root, command_norm)
) STRICT, WITHOUT ROWID;
CREATE INDEX repo_command_last_failing ON repo_command_last(repo_root) WHERE outcome <> 'ok';
CREATE INDEX repo_command_last_call ON repo_command_last(tool_call_pk);   -- foreign-key check

CREATE TABLE session_model_usage (           -- per session and model
  session_pk        INTEGER NOT NULL REFERENCES session(pk),
  model             TEXT NOT NULL,
  input             INTEGER,
  output            INTEGER,
  cache_read        INTEGER,
  cache_write       INTEGER,
  reasoning         INTEGER,
  cost_nusd         INTEGER,
  PRIMARY KEY (session_pk, model)
) STRICT, WITHOUT ROWID;
CREATE INDEX session_model_usage_model ON session_model_usage(model);   -- model:X

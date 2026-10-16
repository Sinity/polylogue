# index.db

Everything parsed and derived from `archive.db` and the blobs: sessions,
messages, blocks, session-owned tool items, tool calls, the search indexes, facts and the daemon's
progress. Deleting it loses nothing; the loop rebuilds it.

## Rules

- **Versioned, never migrated.** The SQL below is the owning forward schema. The target pins and verifies its
  exact specification revision and content, with a version number (`INDEX_VERSION`).
  The inbox Rust comparison remains frozen at its declared index8 snapshot until
  replacement client/runtime acceptance qualifies; its green checks do not
  qualify this forward schema. When the stored version differs, the
  daemon builds a new `index.db` from scratch beside the old one and swaps it
  in when it has caught up (`convergence.md`, rebuilds). A code change that
  alters what parsing produces bumps that format's parser version instead,
  which re-parses only that format's inputs.
- **A fresh index is written without syncing.** A new `index.db`, whether
  the first build or one beside the old file, is created with
  `meta.unsynced = <boot ID>` and written with `synchronous=OFF`. When it
  becomes the serving index after catching up (at the swap, or for a first
  build when it first has no never-parsed input), the key is removed, the
  file switches to `synchronous=NORMAL`, and it is synced. At start, a file
  whose `unsynced` names another boot is deleted and rebuilt (only an
  operating-system crash can corrupt an unsynced file); one naming the
  current boot is continued. This follows from the file being rebuildable;
  it is not a separate mode. The new file is `index.db.new` (with its `-wal`
  and `-shm`); at start an `index.db.new` whose `INDEX_VERSION` is not the
  binary's is deleted.
- **The swap.** The daemon stops admitting reads to the old file and waits
  for reads in flight (each has a time limit, `blobs.md`, WAL), closes every
  connection to both files, checkpoints the new file with `TRUNCATE` and
  closes it, deletes its empty `-wal` and `-shm`, renames it over `index.db`,
  syncs the directory and reopens the pool. No connection is open across the
  rename; requests arriving meanwhile wait the few milliseconds it takes.
- **Keys.** Every entity table (session, message, block, tool_item, tool_call, link,
  event, delivery, run, process, input_ref) has an `INTEGER PRIMARY KEY`
  (`pk`; `text.id`) used only inside `index.db`: it keeps rows small at 20 M blocks and is never renumbered by
  `VACUUM`. Integer keys never leave this file (not to `archive.db`,
  `embeddings.db`, exports or any output), so a rebuild may renumber them.
  The natural IDs (`session.id`, `message.id`, the block ID in any of its
  three forms, `<message>:<n>`, `<message>:r<n>` and `<message>:a:<key>`
  (`export-format.md`), `event.id`, `tool_item.id`) are what every surface, export, note and reference
  uses, resolved through UNIQUE indexes. IDs are opaque and never split
  (`model.md`).
- **Block byte sources are revision addresses.** A block's explicit locator is
  `(input_pk, source_chain, source_start, source_end)`, all present or all
  absent. It names the contributor's retained chain, even when another input
  supplied the preferred message. It is also used for a value kept lazily in
  retained bytes (`blobs.md`, values too large for SQLite). When all four are
  absent, a reader may use a preferred-candidate span only when a format-owned
  witness identifies that exact span as this block's byte source. Being the
  first or only Message citation is not that proof; otherwise the byte source
  is unavailable. No reader substitutes an input's current chain or guesses among its
  `message_source` rows. A missing or excised retained range is reported as
  unavailable or excised, never resolved through another revision.
- **Contributor sets are not envelopes.** `message_source` pins every exact
  contributing raw span to its immutable candidate. One input/chain can supply
  several disjoint spans: older overlay fields, copied context and a timestamp
  can come from different records. A containing span or `[0,input_length)` is
  not a substitute. Replacing current parser metadata cannot redirect a pinned
  candidate's origins. Identical raw re-emission at another offset changes no
  content digest or merge rank; source coordinates are provenance, not version
  evidence. A whole-record citation also does not establish an exact lazy
  payload extent: that needs the format-owned selector/transform and its own
  byte-source contract. SQLite row identifiers are not byte offsets.
- **Owners stated once.** A child row names its session through a composite
  foreign key with its parent, so a block or tool call cannot name a
  different session from its message (R73). A session-owned ToolItem has no fabricated Message. Its version-pinned body associations and direct call anchors compose with the canonical selected-Message-body schema; they must enforce same-session ownership, not an unversioned latest-Message join.
- **Foreign keys are on.** Natural IDs that may name rows not yet ingested
  (`link.parent_id`, `tool_call.child_session_id`, `delivery.recipient_id`, `run.task_id`) have
  no foreign key.
- **Notes outlive index rows.** Notes in `archive.db` refer to sessions,
  messages, blocks, tool items, tool calls and events by natural ID. Proved operation reference refinement must preserve older block/provisional citations through the direct anchor contract (`model.md`, ToolItem). If a target is not in the index
  (not yet ingested, or excised), the note is kept and shown pointing at a
  missing target. Nothing in the index can make a note invalid, so there is
  nothing to check.

**ToolItem qualification boundary.** The independent source-fact tables below define scoped identity, result presence and immutable contributor versions. They do not implement the runtime projection. Body-dependent association/anchor and selected call-summary DDL is owned with the canonical stable-Message-head/body selection change, not a second old-block compatibility schema. An association must pin the selected ToolItem version and exact selected Message body incarnation before it becomes visible. Until that companion and the target arrival-order fixture qualify, the prototype reports incomplete native operation accounting (`plan/compare-old.md`). Ordinary block ownership below is not weakened to pretend an unowned item is a Message child.

## Message preparation and selection

`message` is an identity and visibility head. Public Message/detail/block/export/citation/fact reads follow its `selected_body_pk`, require `hidden=0`, and resolve immutable payloads through that body's roots. A reserved identity with no selected body is absent from public populations. Retained-history reads explicitly name their body/candidate; they never resolve old facts through today's mutable head. Directory rank derives public ordinary/reasoning block positions, while attachment keys remain explicit. Equal local ordinals in different contributions do not prove semantic field correspondence.

A provider contributes addressable groups, blocks and source-field facts. `message_candidate` retains pre-fill mapper facts, not scores inferred from normalized defaults. Reduce eligible revisions within each chain, chain representatives within each input, then effective input representatives, as sources rules2/5 require. `message_chain_choice` and `message_input_choice` retain those representatives and field roots. Their indexed scalar rank summaries are copied from the selected candidate in the same transaction, never independently rescored. Within one retained archive revision, a ready replacement of the same format-owned logical candidate retires the earlier intermediate state from current eligibility and missing-field donors. Scheduling cuts create neither additional source witnesses nor historical input authority. This does not collapse distinct competing versions deliberately represented by a format, and actual historical revisions/chains retain the declared lossy/edit policy. Finish ranks positively in-progress, unclassified, positively finished in ascending order. Edits first exclude superseded content and omissions; unresolved edit-head authority remains unresolved. A detached independent import keeps its own input contribution.

A source candidate retains its exact parser-state generation, committed step and source-basis digest independently of its full preparation digest. Alongside its input/chain, archive revision, logical candidate key, evidence revision and index incarnation, those coordinates reconstruct the preparation predecessor even after disposable parser cells retire. The typed writer checks that predecessor against the retained preparation digest before using it as authority. A newly admitted parser generation may atomically publish the replacement and withdraw only the identified eligible predecessor for the same Message/input/chain/archive revision/logical candidate when its verified generation is strictly older. An absent current-generation receipt alone grants no authority: a same-generation missing receipt, inconsistent coordinates/digest, or unmatched candidate refuses. Other archive revisions and deliberately distinct logical candidates are not retired by this transition. Parser-version numbers, candidate PKs, arrival order and payload-hash order do not choose the predecessor. The old selected body and pinned readers remain coherent until ordinary per-Message activation and bounded retirement; this transition neither switches a whole Session nor qualifies a general index rebuild. These coordinates have no fabricated defaults and survive source-body detachment with the immutable certificate.

A conservative complete-body compatibility certificate may avoid a general fold only for eligible immutable imported snapshots with the same interpretation context, parser, source-evidence digest and source rank, whose entire normalized bodies and explicitly retained source sequence agree. Its domain-separated digest frames normalized scalar values, complete ordered block structure and semantic field values (including suppliedness and time) using integrity-ready value codec/length/hash summaries. Source coordinates and origin descriptors remain owned by each candidate; equal evidence hashes alone do not certify equality. Unknown comparisons, conflicting values, operation shapes not covered by the certificate and ambiguous multiple candidates within one input remain unresolved. The carrier among equal bodies uses the validated content-derived imported InputId as a provenance tie only, so import arrival never changes the preferred origin. Every contributing input retains its own exact citations and source projection. A body owns at most one disposable `message_body_compatibility` row; deleting that body removes the row without retaining payload. Readiness and exact immutable body identity fence certificate publication. The Message's contributor revision advances on each eligible-membership change, including withdrawal and re-addition; a streamed selection checks that revision, index incarnation and removal epoch before installing a carrier. Removal invalidates reads; reparse requires fresh eligible source certificates. This narrow equality path neither ranks nonidentical versions nor implements missing-field fill.

The first native realization may rebuild a known predecessor disposable index during stopped daemon startup under its existing writer ownership. Before touching the old index it preserves and syncs its exact main/WAL/SHM file set with readback verification. A synced, versioned transition marker authorizes restart to discard only residual old or partial replacement index files. Fresh schema initialization and ordinary retained-input recovery rederive the index; deleting the marker follows the schema commit and precedes actor work. Unknown/newer versions and unsafe or busy archives refuse. Source bytes, durable archive state and recovery qualification remain authoritative. Catch-up must remain unavailable or degraded until the requested obligations actually settle. This offline transition does not qualify concurrent serving or a side-by-side generation swap.

`message_build` owns the pinned contributor/removal basis and scalar phase/cursors. Its bounded frontier leases nodes while preparing copy-on-write paths. Its active-value reference is a durable producer lease while normalized bytes are appended/verified; readiness transfers that reference to its field/text owner, and abort releases it before bounded payload retirement. A build may also own one `active_block_pk`, with the same Message/session ownership as the build. This block is private: before installation no body directory or tool summary may reference it. Its text/value and field-root leases are durable while individual fields are prepared; only this exclusive preparation may change its field-root pointer. Installation atomically transfers the block into the prepared body directory, clears the active pointer and commits the exact source cursor. It does not copy all fields. A block reached through an immutable directory is immutable. Completion refuses an active block or active value. Abort first releases the active value and active block ownership, then drains their known field/text/node/value references through bounded retirement; collection checks active-block ownership through its index. Source block/text/field summaries come from mapper witnesses, with block-local field counts included exactly once; an empty Text contributes zero to every criterion. Thinking encryption is nullable: unknown never becomes false. Outer group directories reference local block directories; a replacement copies at most the affected radix16 paths, without shifting later block payload rows. Field directories retain source-slot donors independently of filled output. Block-local semantic field keys omit the final public block ordinal: the selected block root establishes correspondence. Distinct blocks of one candidate may supply the same local key. The typed writer enforces at most one fact for that key in each root before insertion; fact identity is not globally unique by candidate and key. A donor from another block still requires explicit semantic correspondence. A selected field root may instead contain an occupied conflict marker for a semantic slot: ordinary scalar output is unknown, and lower-authority fill cannot replace it. The marker enters a shared immutable witness directory retaining its exact fact references, so a pinned historical body never resolves the conflict through today's mutable input choices. The marker is derived resolution, never an additional source-supplied fact or rank contribution. Typed readiness verifies at least two unequal canonical typed values at equal top authority/version; an empty or agreeing witness set cannot certify a conflict. An immutable node owns at most16 edges; internal height strictly decreases, group leaves enter block roots, field conflict leaves enter witness roots, and final leaves reference blocks or field facts. Witness leaves cannot contain conflicts or other directory kinds, so this additional shared subtree introduces no cycle. Cached counts are computed from those children by the typed writer before sealing the node. No ready node or ready body is mutated.

Bodies, representatives, frontier rows and incoming edges acquire/release node references in their small owning transaction. A zero-reference node is claimed as `retiring` before draining at most the admitted edge count; acquisition cannot attach a retiring node. Retirement releases children by stable keysets, then removes the empty owner. Payload/fact/value cleanup likewise checks remaining references and drains bounded chunks. No cascade, giant ID list or whole-body map is used. SQLite readers pinned before retirement retain their old complete graph through WAL. Index collection cannot delete archive-owned source bytes.

**Field origins.** An immutable field fact may also lease `origin_value_pk`, a
ready JSON `body_value` containing its source-owned, versioned interpretation
descriptor: adapter/parser and selector/transform versions plus direct immutable
typed raw origins. Byte-addressed origins name exact input/chain spans; a
SQLite row origin instead pins the retained immutable snapshot and its
schema/table/native key, never a rowid disguised as a byte offset. These are extraction instructions and locators,
not another copy of source payload. Copied context and overlaid fields retain
the records actually selected, never a pointer to a mutable parser context
cell. The candidate's `message_source` union cannot replace this per-field
binding. Historical and conflict-witness facts retain their own descriptor;
current parser replacement cannot redirect them. Missing origin stays unknown,
and unsupported interpretation remains unavailable rather than guessed from a
first citation. A record origin alone is not a lazy payload extent. Provenance
bytes and their integrity hash do not enter content-version evidence or rank.
The typed writer validates a supported descriptor against the pinned candidate
basis before attaching it, including membership of byte spans in that
candidate's contributor set or validation of a native selector against its
retained snapshot. A ready JSON value alone is not that validation.

Origin preparation uses a separate `message_build.active_origin_value_pk`
producer lease while the normalized active value remains owned. The existing
bounded chunk, cancellation and predecessor rules apply to both. Installing a
field transfers its normalized and ready-origin leases together with the exact
source cursor; neither descriptor nor partial fact becomes public early.
Completion refuses either active producer. Abort releases both before bounded
retirement, and value collection checks both build indexes and both fact
references. Deduplicated equal value PKs still represent two counted leases.
Descriptors obey the same explicit admission/streaming bounds as other values;
one field lookup must not decode a whole-Message provenance map. This schema
does not qualify descriptor codecs, lifecycle or historical provenance at runtime.

Activation checks the exact index incarnation, build identity/basis, head predecessor, contributor revision and removal epoch, then selects one ready body and removes only fixed build/root references. It never adopts/copies every descendant. Candidate/body preparation readiness is committed together; the deferred preferred-candidate constraint pins an immutable certificate and digest. Current eligibility and normalized payload retention are independent of that certificate. A true edit/excision can make a candidate ineligible without rewriting or deleting every historical body that cites its completed preparation. An ineligible candidate may detach its `body_pk`/`body_ready` pair when that unselected body retires; its immutable digest/rank certificate and still-leased facts remain. An eligible candidate must retain a ready source body. Detachment never changes a selected Message head or redirects another body's preferred-candidate reference. A changed contributor or privacy epoch rejects stale activation and leaves newer work owed. Current privacy removal hides the identity immediately where required; preparation cannot resurrect its removed payload. A pinned content snapshot may keep its old complete incarnation across ordinary replacement, but each resumed public read also checks the current hidden/removal epoch; privacy invalidation revokes that old view.

Ready-body retirement first proves the exact index/body incarnation is unselected, is not an active build's body, and no eligible candidate requires it. Separate build-frontier node leases remain counted and are never released with this body. Each turn detaches at most one known ineligible candidate's body pointer through the body index, or deletes the fixed body row and releases its at most three root leases in one transaction. It never unseals a ready certificate or edits ready-body roots. Zero-lease node collection then releases at most16 known edges per turn, including final block/text/fact references, while pinned SQLite readers retain their original view. A field fact remains owned while its candidate is eligible, still retains a source body, or a directory edge leases that fact; final reference release may reclaim it only after those indexed ownership checks. Certificate/provenance cleanup follows its actual remaining body/choice/fact/source references, one known row per turn. Cleanup cannot race a new lease or erase newer eligibility: each detach/release/delete shares the owning predecessor/liveness checks and transaction. These are required writer transitions, not a claim that the schema alone performs collection.

Selected ToolItem versions may have known operation existence/classification while required payload evidence remains pending. Such proved operations still count with incomplete payload/history metadata. Unknown classification/correspondence remains unknown coverage. Immutable association leaves are shared by persistent per-body operation roots. Reads require selected-root membership and the selected item version; unowned operations require no fabricated Message. Block-derived call summaries become visible through selected directory leaves, and operation-key deduplication follows proved correspondence. Direct call anchors refine tool lookup without changing a block citation's meaning.

**Qualification boundary.** This is the forward representation, not a claim that its runtime or resource acceptance has passed. Whole provider cells, decoded JSON tokens and evidence-hash updates still require their explicit admission/scheduling bounds. Variable normalized bytes have one canonical location: body_value chunks. Message/block metadata resolves through typed field roots and their fact/value references; text.id is a deduplicated hash/value reference, never another text body. Large command/input/data/attrs values are not copied into a final whole-cell column at readiness. The payload preparation/streaming implementation must earn those bounds; the schema alone does not. `ready` qualifies a complete Message representation; it does not certify search or all input finalization. Unfinished indexing/hash/derivation remains owed in the owning convergence state. Staged content must not enter serving FTS/trigram populations: relational filtering cannot isolate their ranking or vocabulary statistics. The search implementation decision and full actual-client/restart acceptance remain open. The frozen inbox comparison cannot qualify these changed tables.

## Tables

**Every foreign key has an index on its child column.** Reconciliation and
retirement use bounded keyset steps. SQLite checks each foreign key on a delete
by looking up the child column; without an index that is a scan of the child
table per deleted row. The same holds for the rebuild's own
queries: a session's blocks are found through `message` (`message_session`,
then `block_message`), never by `block.session_pk`, which has no index; a real
first build that filtered on it went quadratic (`evidence/real-converge.md`).
A partial index whose only condition is
`<column> IS NOT NULL` serves the lookup; one with any other condition does not.
An excised input retains its `input_ref` identity with `excised` set until
its references have been drained. Input-scoped indexes also support that
closure; no cascade or whole-session deletion stands in for bounded retirement.

```sql
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

-- Stable identity/order. Content is selected only through a ready immutable body.
CREATE TABLE message (
  pk INTEGER PRIMARY KEY AUTOINCREMENT,
  id TEXT NOT NULL UNIQUE,
  session_pk INTEGER NOT NULL REFERENCES session(pk),
  seq INTEGER NOT NULL,
  selected_body_pk INTEGER,
  selected_ready INTEGER NOT NULL DEFAULT 1 CHECK(selected_ready=1),
  contributor_revision INTEGER NOT NULL DEFAULT 0 CHECK(contributor_revision>=0),
  removal_epoch INTEGER NOT NULL DEFAULT 0 CHECK(removal_epoch>=0),
  hidden INTEGER NOT NULL DEFAULT 0 CHECK(hidden IN(0,1)),
  UNIQUE(pk,session_pk),
  FOREIGN KEY(selected_body_pk,pk,session_pk,selected_ready)
    REFERENCES message_body(pk,message_pk,session_pk,ready)
) STRICT;
CREATE INDEX message_session ON message(session_pk,seq);
CREATE INDEX message_selected_body ON message(selected_body_pk) WHERE selected_body_pk IS NOT NULL;

CREATE TABLE message_body (
  pk                INTEGER PRIMARY KEY AUTOINCREMENT,
  message_pk        INTEGER NOT NULL,
  session_pk        INTEGER NOT NULL,
  ready             INTEGER NOT NULL DEFAULT 0 CHECK(ready IN(0,1)),
  basis_digest      BLOB NOT NULL CHECK(length(basis_digest)=32),
  contributor_revision INTEGER NOT NULL CHECK(contributor_revision>=0),
  removal_epoch     INTEGER NOT NULL CHECK(removal_epoch>=0),
  preferred_candidate_pk INTEGER,
  preferred_ready INTEGER NOT NULL DEFAULT 1 CHECK(preferred_ready=1),
  block_root_pk     INTEGER,
  block_root_kind   TEXT NOT NULL DEFAULT 'groups' CHECK(block_root_kind='groups'),
  operation_root_pk INTEGER,
  operation_root_kind TEXT NOT NULL DEFAULT 'operations' CHECK(operation_root_kind='operations'),
  field_root_pk     INTEGER,
  field_root_kind   TEXT NOT NULL DEFAULT 'fields' CHECK(field_root_kind='fields'),
  root_state        TEXT NOT NULL DEFAULT 'ready' CHECK(root_state='ready'),
  layout            TEXT NOT NULL CHECK(layout IN('flat','separate_reasoning')),
  role TEXT NOT NULL CHECK(role IN('user','assistant','system','tool')),
  written_by TEXT NOT NULL CHECK(written_by IN('human','model','harness','unknown')),
  prompt_kind       TEXT,                   -- written_by human: prose | command (model.md, Message)
  at                INTEGER,                -- provider time
  completed_at      INTEGER,                -- a message from several timed records: the last record's time
  at_until          INTEGER,                -- the end of a bounded `at`; NULL otherwise
  first_seen        INTEGER,                -- when Polylogue first had its bytes (convergence.md); NULL for at
                                            --   most a minute, until an observation covers them
  input_pk          INTEGER NOT NULL REFERENCES input_ref(pk),  -- input the stored (preferred) version came from
  source_chain      INTEGER,                -- the revision chain the byte range is in (archive.db revision.chain):
                                            --   offsets are stable only within a chain, which only appends
  source_start      INTEGER,                -- byte range of its records in that chain (export `sources[0]`):
  source_end        INTEGER,                --   raw view and outside anchors; for a JSON export, the conversation object
  id_kind           TEXT,                   -- provider | provisional (model.md, provider and provisional IDs)
  version_hash      BLOB CHECK(version_hash IS NULL OR length(version_hash)=32), -- BLAKE3 of the preferred version's records as canonical JSON without
                                            --   volatile fields: merge rule 2's last tie-break, decided the same way
                                            --   whichever input arrives first (law 4; sources/README.md, rule 2)
  in_progress       INTEGER CHECK(in_progress IN(0,1)), -- NULL unclassified; never infer false from framing/EOF
  on_active_path    INTEGER,                -- 1 on the path, 0 off, NULL unknown (no marked leaf; model.md);
                                            --   always written, no default
  copy_origin       TEXT,                   -- known | unknown for a copy (unknown: provider says copy, names no source)
  usage_present     INTEGER NOT NULL DEFAULT 0, -- distinguishes a reported empty usage object from no usage
  input_tokens      INTEGER,                -- NULL = not reported (never 0 for unknown)
  output_tokens     INTEGER,
  cache_read_tokens INTEGER,
  cache_write_tokens INTEGER,               -- all cache writes
  cache_write_5m_tokens INTEGER,            -- the part written with a 5-minute lifetime, when reported
  cache_write_1h_tokens INTEGER,            -- the part written with a 1-hour lifetime, when reported
  reasoning_tokens  INTEGER,
  web_requests      INTEGER,                -- server-side web search and fetch requests
  provider_cost_nusd INTEGER,               -- the provider's own cost figure, when given
  cost_nusd         INTEGER,                -- the one per-message cost (analysis.md); NULL = unknown
  duration_ms       INTEGER,
  words             INTEGER NOT NULL DEFAULT 0, -- words in its text blocks
  parser_version    INTEGER NOT NULL,       -- exact source interpretation basis; not a deletion/completion cursor
  source_sequence   INTEGER CHECK(source_sequence>=0), -- candidate's supplied Session order; NULL unrecorded, never inferred from the current head
  CHECK(ready=0 OR (preferred_candidate_pk IS NOT NULL AND version_hash IS NOT NULL)),
  FOREIGN KEY(message_pk,session_pk) REFERENCES message(pk,session_pk),
  FOREIGN KEY(preferred_candidate_pk,message_pk,session_pk,input_pk,version_hash,preferred_ready)
    REFERENCES message_candidate(pk,message_pk,session_pk,input_pk,version_hash,ready) DEFERRABLE INITIALLY DEFERRED,
  FOREIGN KEY(block_root_pk,message_pk,session_pk,block_root_kind,root_state)
    REFERENCES message_body_node(pk,message_pk,session_pk,kind,state),
  FOREIGN KEY(operation_root_pk,message_pk,session_pk,operation_root_kind,root_state)
    REFERENCES message_body_node(pk,message_pk,session_pk,kind,state),
  FOREIGN KEY(field_root_pk,message_pk,session_pk,field_root_kind,root_state)
    REFERENCES message_body_node(pk,message_pk,session_pk,kind,state),
  UNIQUE(pk,message_pk,session_pk),
  UNIQUE(pk,message_pk,session_pk,ready),
  UNIQUE(pk,ready)
) STRICT;
-- Derived normalized-body equality, never raw source version evidence. The typed
-- producer visits the complete immutable body before issuing this certificate.
-- One fixed child of a body owns no payload lease; body retirement removes it.
CREATE TABLE message_body_compatibility (
  body_pk INTEGER PRIMARY KEY,
  body_ready INTEGER NOT NULL DEFAULT 1 CHECK(body_ready=1),
  digest BLOB NOT NULL CHECK(length(digest)=32),
  FOREIGN KEY(body_pk,body_ready) REFERENCES message_body(pk,ready) ON DELETE CASCADE
) STRICT;

CREATE INDEX message_body_owner ON message_body(message_pk,session_pk);
CREATE INDEX message_body_input ON message_body(input_pk);
CREATE INDEX message_body_preferred ON message_body(preferred_candidate_pk);
CREATE INDEX message_body_blocks ON message_body(block_root_pk);
CREATE INDEX message_body_fields ON message_body(field_root_pk);
CREATE INDEX message_body_operations ON message_body(operation_root_pk);
CREATE INDEX message_body_prose   ON message_body(session_pk) WHERE written_by = 'human' AND prompt_kind = 'prose';
CREATE INDEX message_body_unseen  ON message_body(input_pk) WHERE first_seen IS NULL;   -- parse work: first_seen to fill
CREATE INDEX message_body_branch  ON message_body(session_pk) WHERE on_active_path = 0;


-- A typed source candidate is not a global winner. Reduce versions per chain,
-- chain representatives per input, then input representatives (sources rules2/5).
-- Unresolved facts/hash/head remain retained and are not fabricated eligibility.
CREATE TABLE message_candidate (
  pk INTEGER PRIMARY KEY AUTOINCREMENT,
  message_pk INTEGER NOT NULL,
  session_pk INTEGER NOT NULL,
  input_pk INTEGER NOT NULL REFERENCES input_ref(pk),
  chain INTEGER NOT NULL CHECK(chain>0),
  revision INTEGER NOT NULL CHECK(revision>0),
  revision_length INTEGER NOT NULL CHECK(revision_length>=0),
  evidence_revision INTEGER NOT NULL CHECK(evidence_revision>=0),
  logical_key BLOB NOT NULL CHECK(length(logical_key)=32), -- format-owned candidate identity, never a scheduling cut
  parser_version INTEGER NOT NULL CHECK(parser_version>=0),
  source_state_generation INTEGER NOT NULL CHECK(source_state_generation>0),
  source_state_step INTEGER NOT NULL CHECK(source_state_step>=0),
  source_basis_digest BLOB NOT NULL CHECK(length(source_basis_digest)=32),
  basis_digest BLOB NOT NULL CHECK(length(basis_digest)=32),
  body_pk INTEGER,
  body_ready INTEGER CHECK(body_ready=1),
  finish INTEGER NOT NULL CHECK(finish BETWEEN 0 AND 2), -- in-progress / unclassified / finished
  route INTEGER NOT NULL CHECK(route BETWEEN 0 AND 2), -- DOM/export/native
  source_blocks INTEGER CHECK(source_blocks>=0),
  source_text_blocks INTEGER CHECK(source_text_blocks BETWEEN 0 AND source_blocks),
  source_fields INTEGER CHECK(source_fields>=0),
  version_hash BLOB CHECK(version_hash IS NULL OR length(version_hash)=32),
  facts_ready INTEGER NOT NULL DEFAULT 0 CHECK(facts_ready IN(0,1)),
  ready INTEGER NOT NULL DEFAULT 0 CHECK(ready IN(0,1)), -- immutable preparation certificate
  eligible INTEGER NOT NULL DEFAULT 0 CHECK(eligible IN(0,1)), -- mutable source-basis eligibility
  CHECK(facts_ready=0 OR (source_blocks IS NOT NULL AND source_text_blocks IS NOT NULL AND source_fields IS NOT NULL)),
  CHECK(body_pk IS NOT NULL OR body_ready IS NULL),
  CHECK(eligible=0 OR (ready=1 AND body_pk IS NOT NULL AND body_ready IS NOT NULL)),
  CHECK(ready=0 OR ((body_pk IS NULL)=(body_ready IS NULL))),
  CHECK(ready=0 OR (facts_ready=1 AND source_blocks IS NOT NULL
        AND source_text_blocks IS NOT NULL AND source_fields IS NOT NULL
        AND version_hash IS NOT NULL)),
  FOREIGN KEY(message_pk,session_pk) REFERENCES message(pk,session_pk),
  FOREIGN KEY(body_pk,message_pk,session_pk,body_ready) REFERENCES message_body(pk,message_pk,session_pk,ready),
  UNIQUE(pk,message_pk,session_pk,input_pk,version_hash,ready),
  UNIQUE(pk,message_pk,session_pk),
  UNIQUE(pk,message_pk,session_pk,input_pk),
  UNIQUE(pk,message_pk,session_pk,input_pk,chain)
) STRICT;
-- One currently eligible effective state of one provider-owned candidate within
-- a retained revision. Prior scheduling cuts are retained history, not donors.
CREATE UNIQUE INDEX message_candidate_effective ON message_candidate(
  message_pk,input_pk,chain,revision,logical_key) WHERE eligible=1;
CREATE INDEX message_candidate_input ON message_candidate(input_pk,chain,revision);
CREATE INDEX message_candidate_owner ON message_candidate(message_pk,session_pk);
CREATE INDEX message_candidate_eligible_cursor ON message_candidate(message_pk,pk) WHERE eligible=1;
CREATE INDEX message_candidate_body ON message_candidate(body_pk) WHERE body_pk IS NOT NULL;
CREATE INDEX message_candidate_required_body ON message_candidate(body_pk) WHERE eligible=1; -- fixed liveness probe without scanning retired pointers
CREATE INDEX message_candidate_chain_rank ON message_candidate(message_pk,input_pk,chain,
  finish DESC,route DESC,source_blocks DESC,source_text_blocks DESC,source_fields DESC,
  revision_length DESC,version_hash DESC) WHERE eligible=1;

-- Fence every eligible-membership change, including withdraw/re-add ABA.
CREATE TRIGGER message_candidate_membership_insert AFTER INSERT ON message_candidate
WHEN NEW.eligible=1 BEGIN
  UPDATE message SET contributor_revision=contributor_revision+1 WHERE pk=NEW.message_pk;
END;
CREATE TRIGGER message_candidate_membership_update AFTER UPDATE OF eligible ON message_candidate
WHEN OLD.eligible<>NEW.eligible BEGIN
  UPDATE message SET contributor_revision=contributor_revision+1 WHERE pk=NEW.message_pk;
END;
CREATE TRIGGER message_candidate_membership_delete AFTER DELETE ON message_candidate
WHEN OLD.eligible=1 BEGIN
  UPDATE message SET contributor_revision=contributor_revision+1 WHERE pk=OLD.message_pk;
END;

-- Representatives retain roots, including eligible missing-field donors, not
-- merely the previous global winner. Edits publish a proved current head or an
-- explicit unresolved state; a hash/mtime/arrival never supplies that proof.
CREATE TABLE message_chain_choice (
  message_pk INTEGER NOT NULL,
  session_pk INTEGER NOT NULL,
  input_pk INTEGER NOT NULL,
  chain INTEGER NOT NULL,
  candidate_pk INTEGER,
  finish INTEGER CHECK(finish BETWEEN 0 AND 2),
  route INTEGER CHECK(route BETWEEN 0 AND 2),
  source_blocks INTEGER CHECK(source_blocks>=0),
  source_text_blocks INTEGER CHECK(source_text_blocks BETWEEN 0 AND source_blocks),
  source_fields INTEGER CHECK(source_fields>=0),
  version_hash BLOB CHECK(version_hash IS NULL OR length(version_hash)=32),
  field_root_pk INTEGER,
  field_root_kind TEXT NOT NULL DEFAULT 'fields' CHECK(field_root_kind='fields'),
  root_state TEXT NOT NULL DEFAULT 'ready' CHECK(root_state='ready'),
  revision INTEGER NOT NULL CHECK(revision>=0),
  is_current INTEGER NOT NULL CHECK(is_current IN(0,1)),
  CHECK(candidate_pk IS NULL OR (finish IS NOT NULL AND route IS NOT NULL
    AND source_blocks IS NOT NULL AND source_text_blocks IS NOT NULL
    AND source_fields IS NOT NULL AND version_hash IS NOT NULL)),
  FOREIGN KEY(message_pk,session_pk) REFERENCES message(pk,session_pk),
  FOREIGN KEY(candidate_pk,message_pk,session_pk,input_pk,chain)
    REFERENCES message_candidate(pk,message_pk,session_pk,input_pk,chain),
  FOREIGN KEY(input_pk) REFERENCES input_ref(pk),
  FOREIGN KEY(field_root_pk,message_pk,session_pk,field_root_kind,root_state)
    REFERENCES message_body_node(pk,message_pk,session_pk,kind,state),
  PRIMARY KEY(message_pk,input_pk,chain)
) STRICT, WITHOUT ROWID;
CREATE INDEX message_chain_choice_input ON message_chain_choice(input_pk);
CREATE INDEX message_chain_choice_candidate ON message_chain_choice(candidate_pk);
CREATE INDEX message_chain_choice_fields ON message_chain_choice(field_root_pk);
CREATE INDEX message_chain_choice_rank ON message_chain_choice(message_pk,input_pk,
  finish DESC,route DESC,source_blocks DESC,source_text_blocks DESC,source_fields DESC,
  is_current DESC,version_hash DESC) WHERE candidate_pk IS NOT NULL;
CREATE TABLE message_input_choice (
  message_pk INTEGER NOT NULL,
  session_pk INTEGER NOT NULL,
  input_pk INTEGER NOT NULL REFERENCES input_ref(pk),
  candidate_pk INTEGER,
  finish INTEGER CHECK(finish BETWEEN 0 AND 2),
  route INTEGER CHECK(route BETWEEN 0 AND 2),
  source_blocks INTEGER CHECK(source_blocks>=0),
  source_text_blocks INTEGER CHECK(source_text_blocks BETWEEN 0 AND source_blocks),
  source_fields INTEGER CHECK(source_fields>=0),
  version_hash BLOB CHECK(version_hash IS NULL OR length(version_hash)=32),
  field_root_pk INTEGER,
  field_root_kind TEXT NOT NULL DEFAULT 'fields' CHECK(field_root_kind='fields'),
  root_state TEXT NOT NULL DEFAULT 'ready' CHECK(root_state='ready'),
  revision INTEGER NOT NULL CHECK(revision>=0),
  state TEXT NOT NULL CHECK(state IN('ready','none','unresolved_edit_head','pending')),
  CHECK(state!='ready' OR candidate_pk IS NOT NULL),
  CHECK(candidate_pk IS NULL OR (finish IS NOT NULL AND route IS NOT NULL
    AND source_blocks IS NOT NULL AND source_text_blocks IS NOT NULL
    AND source_fields IS NOT NULL AND version_hash IS NOT NULL)),
  FOREIGN KEY(message_pk,session_pk) REFERENCES message(pk,session_pk),
  FOREIGN KEY(candidate_pk,message_pk,session_pk,input_pk)
    REFERENCES message_candidate(pk,message_pk,session_pk,input_pk),
  FOREIGN KEY(field_root_pk,message_pk,session_pk,field_root_kind,root_state)
    REFERENCES message_body_node(pk,message_pk,session_pk,kind,state),
  PRIMARY KEY(message_pk,input_pk)
) STRICT, WITHOUT ROWID;
CREATE INDEX message_input_choice_input ON message_input_choice(input_pk);
CREATE INDEX message_input_choice_candidate ON message_input_choice(candidate_pk);
CREATE INDEX message_input_choice_fields ON message_input_choice(field_root_pk);
CREATE INDEX message_input_choice_rank ON message_input_choice(message_pk,
  finish DESC,route DESC,source_blocks DESC,source_text_blocks DESC,source_fields DESC,
  version_hash DESC) WHERE state='ready';

-- Canonical normalized value bytes, streamed in admitted chunks. Their content
-- hash is payload integrity, never merge version evidence. Raw lazy source
-- locators remain revision addresses; they do not masquerade as normalized bytes.
CREATE TABLE body_value (
  pk INTEGER PRIMARY KEY AUTOINCREMENT,
  codec TEXT NOT NULL CHECK(codec IN('utf8','json','bytes')),
  length INTEGER NOT NULL CHECK(length>=0),
  hash BLOB CHECK(hash IS NULL OR length(hash)=32),
  state TEXT NOT NULL CHECK(state IN('building','ready','retiring')),
  incoming INTEGER NOT NULL CHECK(incoming>=0),
  CHECK(state!='ready' OR hash IS NOT NULL),
  CHECK(state!='retiring' OR incoming=0),
  UNIQUE(pk,state),
  UNIQUE(pk,state,codec),
  UNIQUE(pk,state,codec,hash)
) STRICT;
CREATE INDEX body_value_reclaim ON body_value(pk) WHERE incoming=0;
CREATE INDEX body_value_retiring ON body_value(pk) WHERE state='retiring';
CREATE UNIQUE INDEX body_value_content ON body_value(codec,hash) WHERE hash IS NOT NULL AND state='ready';
CREATE TABLE body_value_chunk (
  value_pk INTEGER NOT NULL REFERENCES body_value(pk),
  offset INTEGER NOT NULL CHECK(offset>=0),
  bytes BLOB NOT NULL CHECK(length(bytes) BETWEEN 1 AND 65536),
  PRIMARY KEY(value_pk,offset)
) STRICT, WITHOUT ROWID;
CREATE TABLE message_field_slot (
  pk INTEGER PRIMARY KEY AUTOINCREMENT,
  message_pk INTEGER NOT NULL,
  session_pk INTEGER NOT NULL,
  semantic_key BLOB NOT NULL, -- typed/proved correspondence, never coincident body ordinal alone
  FOREIGN KEY(message_pk,session_pk) REFERENCES message(pk,session_pk),
  UNIQUE(message_pk,semantic_key),
  UNIQUE(pk,message_pk,session_pk)
) STRICT;
CREATE TABLE message_field_fact (
  pk INTEGER PRIMARY KEY AUTOINCREMENT,
  message_pk INTEGER NOT NULL,
  session_pk INTEGER NOT NULL,
  candidate_pk INTEGER NOT NULL,
  slot_pk INTEGER NOT NULL,
  value_pk INTEGER NOT NULL,
  value_state TEXT NOT NULL DEFAULT 'ready' CHECK(value_state='ready'),
  origin_value_pk INTEGER, -- immutable source-owned selector/transform + direct raw origins; NULL means unknown
  origin_value_state TEXT NOT NULL DEFAULT 'ready' CHECK(origin_value_state='ready'),
  origin_value_codec TEXT NOT NULL DEFAULT 'json' CHECK(origin_value_codec='json'),
  source_time INTEGER,
  supplied INTEGER NOT NULL CHECK(supplied IN(0,1)), -- mapper-owned before fill/enrichment
  FOREIGN KEY(candidate_pk,message_pk,session_pk)
    REFERENCES message_candidate(pk,message_pk,session_pk),
  FOREIGN KEY(slot_pk,message_pk,session_pk)
    REFERENCES message_field_slot(pk,message_pk,session_pk),
  FOREIGN KEY(value_pk,value_state) REFERENCES body_value(pk,state),
  FOREIGN KEY(origin_value_pk,origin_value_state,origin_value_codec)
    REFERENCES body_value(pk,state,codec),
  UNIQUE(pk,message_pk,session_pk)
) STRICT;
CREATE INDEX message_field_fact_owner ON message_field_fact(message_pk,session_pk);
CREATE INDEX message_field_fact_candidate ON message_field_fact(candidate_pk,slot_pk,pk);
CREATE INDEX message_field_fact_slot ON message_field_fact(slot_pk);
CREATE INDEX message_field_fact_value ON message_field_fact(value_pk);
CREATE INDEX message_field_fact_origin ON message_field_fact(origin_value_pk);

-- Message-local immutable directories. Outer groups reference block roots;
-- separate field roots retain donor facts without serializing/copying a map.
-- Each sparse radix16 node has <=16 edges; u64 ordinals require <=16 levels.
CREATE TABLE message_body_node (
  pk INTEGER PRIMARY KEY AUTOINCREMENT,
  message_pk INTEGER NOT NULL,
  session_pk INTEGER NOT NULL,
  kind TEXT NOT NULL CHECK(kind IN('groups','blocks','fields','operations','witnesses')),
  height INTEGER NOT NULL CHECK(height BETWEEN 0 AND 15),
  state TEXT NOT NULL CHECK(state IN('building','ready','retiring')),
  incoming INTEGER NOT NULL CHECK(incoming>=0),
  ordinary_count INTEGER NOT NULL CHECK(ordinary_count>=0),
  reasoning_count INTEGER NOT NULL CHECK(reasoning_count>=0),
  source_blocks INTEGER NOT NULL CHECK(source_blocks>=0),
  source_text_blocks INTEGER NOT NULL CHECK(source_text_blocks BETWEEN 0 AND source_blocks),
  source_fields INTEGER NOT NULL CHECK(source_fields>=0),
  CHECK(state!='retiring' OR incoming=0),
  FOREIGN KEY(message_pk,session_pk) REFERENCES message(pk,session_pk),
  UNIQUE(pk,message_pk,session_pk),
  UNIQUE(pk,message_pk,session_pk,kind),
  UNIQUE(pk,message_pk,session_pk,kind,state),
  UNIQUE(pk,message_pk,session_pk,kind,height)
) STRICT;
CREATE INDEX message_body_node_owner ON message_body_node(message_pk,session_pk);
CREATE INDEX message_body_node_reclaim ON message_body_node(pk) WHERE incoming=0;
CREATE INDEX message_body_node_owner_reclaim ON message_body_node(message_pk,pk) WHERE incoming=0;
CREATE TABLE message_body_edge (
  parent_pk INTEGER NOT NULL,
  message_pk INTEGER NOT NULL,
  session_pk INTEGER NOT NULL,
  parent_kind TEXT NOT NULL,
  parent_height INTEGER NOT NULL,
  slot INTEGER NOT NULL CHECK(slot BETWEEN 0 AND 15),
  child_node_pk INTEGER,
  child_kind TEXT,
  child_height INTEGER,
  child_state TEXT NOT NULL DEFAULT 'ready' CHECK(child_state='ready'),
  block_pk INTEGER,
  field_fact_pk INTEGER,
  conflict_slot_pk INTEGER, -- occupied unresolved scalar; witnesses remain in contributing roots
  tool_call_pk INTEGER,
  association_pk INTEGER,
  ordinary_count INTEGER NOT NULL CHECK(ordinary_count>=0),
  reasoning_count INTEGER NOT NULL CHECK(reasoning_count>=0),
  source_blocks INTEGER NOT NULL CHECK(source_blocks>=0),
  source_text_blocks INTEGER NOT NULL CHECK(source_text_blocks BETWEEN 0 AND source_blocks),
  source_fields INTEGER NOT NULL CHECK(source_fields>=0),
  CHECK((parent_height>0 AND child_node_pk IS NOT NULL AND child_kind=parent_kind
         AND child_kind IS NOT NULL AND child_height IS NOT NULL AND child_height=parent_height-1 AND block_pk IS NULL AND field_fact_pk IS NULL AND conflict_slot_pk IS NULL AND tool_call_pk IS NULL AND association_pk IS NULL)
     OR (parent_height=0 AND parent_kind='groups' AND child_node_pk IS NOT NULL
         AND child_kind='blocks' AND child_kind IS NOT NULL AND child_height IS NOT NULL AND block_pk IS NULL AND field_fact_pk IS NULL AND conflict_slot_pk IS NULL AND tool_call_pk IS NULL AND association_pk IS NULL)
     OR (parent_height=0 AND parent_kind='blocks' AND child_node_pk IS NULL AND child_kind IS NULL
         AND child_height IS NULL AND block_pk IS NOT NULL AND field_fact_pk IS NULL AND conflict_slot_pk IS NULL AND association_pk IS NULL)
     OR (parent_height=0 AND parent_kind IN('fields','witnesses') AND child_node_pk IS NULL AND child_kind IS NULL
         AND child_height IS NULL AND block_pk IS NULL AND field_fact_pk IS NOT NULL
         AND conflict_slot_pk IS NULL AND tool_call_pk IS NULL AND association_pk IS NULL)
     OR (parent_height=0 AND parent_kind='fields' AND child_node_pk IS NOT NULL
         AND child_kind='witnesses' AND child_kind IS NOT NULL AND child_height IS NOT NULL AND block_pk IS NULL
         AND field_fact_pk IS NULL AND conflict_slot_pk IS NOT NULL AND tool_call_pk IS NULL
         AND association_pk IS NULL AND ordinary_count=0 AND reasoning_count=0
         AND source_blocks=0 AND source_text_blocks=0 AND source_fields=0)
     OR (parent_height=0 AND parent_kind='operations' AND child_node_pk IS NULL AND child_kind IS NULL
         AND child_height IS NULL AND block_pk IS NULL AND field_fact_pk IS NULL AND conflict_slot_pk IS NULL AND tool_call_pk IS NULL AND association_pk IS NOT NULL)),
  FOREIGN KEY(parent_pk,message_pk,session_pk,parent_kind,parent_height)
    REFERENCES message_body_node(pk,message_pk,session_pk,kind,height),
  FOREIGN KEY(child_node_pk,message_pk,session_pk,child_kind,child_height)
    REFERENCES message_body_node(pk,message_pk,session_pk,kind,height),
  FOREIGN KEY(child_node_pk,message_pk,session_pk,child_kind,child_state)
    REFERENCES message_body_node(pk,message_pk,session_pk,kind,state),
  FOREIGN KEY(block_pk,message_pk,session_pk) REFERENCES block(pk,message_pk,session_pk),
  FOREIGN KEY(field_fact_pk,message_pk,session_pk) REFERENCES message_field_fact(pk,message_pk,session_pk),
  FOREIGN KEY(conflict_slot_pk,message_pk,session_pk) REFERENCES message_field_slot(pk,message_pk,session_pk),
  FOREIGN KEY(association_pk,message_pk,session_pk) REFERENCES tool_item_association(pk,message_pk,session_pk),
  FOREIGN KEY(tool_call_pk,block_pk,message_pk,session_pk)
    REFERENCES tool_call(pk,call_block_pk,message_pk,session_pk),
  PRIMARY KEY(parent_pk,slot)
) STRICT, WITHOUT ROWID;
CREATE INDEX message_body_edge_child ON message_body_edge(child_node_pk);
CREATE INDEX message_body_edge_block ON message_body_edge(block_pk);
CREATE INDEX message_body_edge_fact ON message_body_edge(field_fact_pk);
CREATE INDEX message_body_edge_conflict ON message_body_edge(conflict_slot_pk);
CREATE INDEX message_body_edge_call ON message_body_edge(tool_call_pk);
CREATE INDEX message_body_edge_association ON message_body_edge(association_pk);

-- One owned resumable build per Message. This is disposable computation over
-- a pinned basis, not an independent queue. Cursor scalars never contain a
-- growing list of IDs, blocks, fields or parser state. AUTOINCREMENT prevents a
-- stale process from naming a later build after this row has been retired.
CREATE TABLE message_build (
  pk INTEGER PRIMARY KEY AUTOINCREMENT,
  message_pk INTEGER NOT NULL,
  session_pk INTEGER NOT NULL,
  body_pk INTEGER NOT NULL,
  basis_digest BLOB NOT NULL CHECK(length(basis_digest)=32),
  contributor_revision INTEGER NOT NULL CHECK(contributor_revision>=0),
  removal_epoch INTEGER NOT NULL CHECK(removal_epoch>=0),
  phase TEXT NOT NULL CHECK(phase IN('project','rank','fold','hash','layout','ready','retiring')),
  active_value_pk INTEGER REFERENCES body_value(pk), -- durable producer lease, transferred/released atomically
  active_origin_value_pk INTEGER REFERENCES body_value(pk), -- independent producer lease for the pending field's origin descriptor
  active_block_pk INTEGER, -- one private mutable block; transferred to its directory atomically
  source_cursor INTEGER NOT NULL DEFAULT 0 CHECK(source_cursor>=0),
  group_cursor INTEGER NOT NULL DEFAULT 0 CHECK(group_cursor>=0),
  local_cursor INTEGER NOT NULL DEFAULT 0 CHECK(local_cursor>=0),
  field_cursor INTEGER NOT NULL DEFAULT 0 CHECK(field_cursor>=0),
  hash_offset INTEGER NOT NULL DEFAULT 0 CHECK(hash_offset>=0),
  FOREIGN KEY(body_pk,message_pk,session_pk) REFERENCES message_body(pk,message_pk,session_pk),
  FOREIGN KEY(active_block_pk,message_pk,session_pk) REFERENCES block(pk,message_pk,session_pk),
  UNIQUE(message_pk),
  UNIQUE(pk,message_pk,session_pk)
) STRICT;
CREATE INDEX message_build_body ON message_build(body_pk);
CREATE INDEX message_build_value ON message_build(active_value_pk);
CREATE INDEX message_build_origin_value ON message_build(active_origin_value_pk);
CREATE INDEX message_build_block ON message_build(active_block_pk) WHERE active_block_pk IS NOT NULL;
-- Fixed frontier slots lease ready nodes while paths are copied. A path through
-- an outer group/block path or field/conflict-witness path has at most 32 radix16 nodes. Root
-- leases owned by bodies/representatives are separate incoming references.
CREATE TABLE message_build_frontier (
  build_pk INTEGER NOT NULL,
  message_pk INTEGER NOT NULL,
  session_pk INTEGER NOT NULL,
  depth INTEGER NOT NULL CHECK(depth BETWEEN 0 AND 31),
  node_pk INTEGER NOT NULL,
  next_slot INTEGER NOT NULL CHECK(next_slot BETWEEN 0 AND 16),
  FOREIGN KEY(build_pk,message_pk,session_pk) REFERENCES message_build(pk,message_pk,session_pk),
  FOREIGN KEY(node_pk,message_pk,session_pk) REFERENCES message_body_node(pk,message_pk,session_pk),
  PRIMARY KEY(build_pk,depth)
) STRICT, WITHOUT ROWID;
CREATE INDEX message_build_frontier_node ON message_build_frontier(node_pk);

-- Exact contributor spans, pinned to source candidate/input/chain. Several
-- disjoint records may supply one candidate; no bounding-envelope fallback.
-- A citation never substitutes a current chain for this retained chain.
CREATE TABLE message_source (
  candidate_pk      INTEGER NOT NULL,
  message_pk        INTEGER NOT NULL,
  session_pk        INTEGER NOT NULL,
  input_pk          INTEGER NOT NULL,
  chain             INTEGER NOT NULL CHECK(chain>0), -- archive.db revision.chain
  source_start      INTEGER NOT NULL CHECK(source_start>=0),
  source_end        INTEGER NOT NULL CHECK(source_end>=source_start),
  FOREIGN KEY(candidate_pk,message_pk,session_pk,input_pk,chain)
    REFERENCES message_candidate(pk,message_pk,session_pk,input_pk,chain),
  PRIMARY KEY (candidate_pk, input_pk, chain, source_start, source_end)
) STRICT;
CREATE INDEX message_source_message ON message_source(message_pk);
CREATE INDEX message_source_input ON message_source(input_pk, chain);   -- foreign-key check; excision scope

-- Deduplicated UTF-8 payload references; bytes live only in body_value_chunk.
-- `id` is the rowid of serving search entries, not a second text payload.
CREATE TABLE text (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  hash BLOB NOT NULL UNIQUE CHECK(length(hash)=32),
  value_pk INTEGER NOT NULL UNIQUE,
  value_state TEXT NOT NULL DEFAULT 'ready' CHECK(value_state='ready'),
  value_codec TEXT NOT NULL DEFAULT 'utf8' CHECK(value_codec='utf8'),
  FOREIGN KEY(value_pk,value_state,value_codec,hash) REFERENCES body_value(pk,state,codec,hash)
) STRICT;

-- What the provider wrote, one row per content item. Facts derived about a tool
-- call (tool, command, outcome, exit code, duration) belong to tool_call alone.
CREATE TABLE block (
  pk                INTEGER PRIMARY KEY AUTOINCREMENT,
  message_pk        INTEGER NOT NULL,
  session_pk        INTEGER NOT NULL,       -- the message's session, enforced by the composite key below
  group_ordinal     INTEGER NOT NULL CHECK(group_ordinal>=0),
  local_ordinal     INTEGER NOT NULL CHECK(local_ordinal>=0),
  slot_kind         TEXT NOT NULL CHECK(slot_kind IN('ordinary','reasoning','attachment')),
  attach_key        TEXT,
  kind              TEXT NOT NULL CHECK(kind IN('text','thinking','tool_call','tool_result','image','file','other')),
  source_block INTEGER NOT NULL DEFAULT 0 CHECK(source_block IN(0,1)),
  source_text INTEGER NOT NULL DEFAULT 0 CHECK(source_text IN(0,1) AND source_text<=source_block),
  field_root_pk INTEGER,
  field_root_kind TEXT NOT NULL DEFAULT 'fields' CHECK(field_root_kind='fields'),
  field_root_state TEXT NOT NULL DEFAULT 'ready' CHECK(field_root_state='ready'),
  text_id           INTEGER REFERENCES text(id),
  input_pk          INTEGER REFERENCES input_ref(pk),  -- explicit contributor or lazy value locator; otherwise all four NULL
  source_chain      INTEGER,                -- archive.db revision.chain of that contributor, never its current chain
  source_start      INTEGER,                -- byte range in that exact input and chain, including an empty value
  source_end        INTEGER,                --   field from another input or a value too large to index (blobs.md)
  blob_hash         BLOB,                   -- image/file, large tool output: bytes in the blob store; export `blob`
  direction         TEXT,                   -- image/file: input | output | unknown
  produced_by_pk    INTEGER,                -- image/file output: the block of the tool call or image request that made it
  size              INTEGER,                -- image/file
  encrypted         INTEGER CHECK(encrypted IS NULL OR encrypted IN(0,1)), -- unknown remains unknown
  result_outcome    TEXT,                   -- tool_result: provider fact, retained even without a paired call
  result_reason     TEXT,                   -- unknown/error reason in the exported vocabulary
  result_exit_code  INTEGER,
  result_duration_ms INTEGER,
  looks_like_secret INTEGER NOT NULL DEFAULT 0, -- has:secret (behavior.md, privacy)
  CHECK((slot_kind='attachment')=(attach_key IS NOT NULL)),
  CHECK ((input_pk IS NULL AND source_chain IS NULL AND source_start IS NULL AND source_end IS NULL)
      OR (input_pk IS NOT NULL AND source_chain IS NOT NULL AND source_start IS NOT NULL AND source_end IS NOT NULL
          AND source_chain > 0 AND source_start >= 0 AND source_end >= source_start)),
  FOREIGN KEY(field_root_pk,message_pk,session_pk,field_root_kind,field_root_state)
    REFERENCES message_body_node(pk,message_pk,session_pk,kind,state),
  FOREIGN KEY (message_pk, session_pk) REFERENCES message(pk, session_pk),
  UNIQUE (pk, session_pk),
  UNIQUE (pk, message_pk, session_pk)
) STRICT;
CREATE INDEX block_owner ON block(message_pk,session_pk);
CREATE INDEX block_fields ON block(field_root_pk);
CREATE INDEX block_source ON block(input_pk) WHERE input_pk IS NOT NULL;
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

-- Stable native/session-owned operation identity. Its selected version is a
-- immutable effective facts candidate. It may have pending payload evidence:
-- known operation existence/classification still counts, while complete-payload
-- reads require ready=1. Unknown Message ownership is independently valid.
CREATE TABLE tool_item (
  pk                INTEGER PRIMARY KEY AUTOINCREMENT,
  id                TEXT NOT NULL UNIQUE,
  session_pk        INTEGER NOT NULL REFERENCES session(pk),
  provider_namespace TEXT NOT NULL,
  identity_schema   TEXT NOT NULL,
  identity_kind     TEXT NOT NULL CHECK (identity_kind IN ('native', 'legacy', 'provisional')),
  native_turn_id    TEXT,
  native_item_id    TEXT,
  selected_version_pk INTEGER,              -- effective fact version; may have pending required payload
  CHECK ((identity_kind = 'native' AND native_turn_id IS NOT NULL AND length(native_turn_id) > 0
      AND native_item_id IS NOT NULL AND length(native_item_id) > 0)
      OR (identity_kind <> 'native' AND native_turn_id IS NULL AND native_item_id IS NULL)),
  UNIQUE (pk, session_pk),
  UNIQUE (pk, session_pk, id),
  FOREIGN KEY (selected_version_pk, pk, session_pk)
    REFERENCES tool_item_version(pk, tool_item_pk, session_pk)
) STRICT;
CREATE INDEX tool_item_session ON tool_item(session_pk);
CREATE UNIQUE INDEX tool_item_native ON tool_item(session_pk, provider_namespace, native_turn_id, native_item_id)
  WHERE identity_kind = 'native';
CREATE INDEX tool_item_selected ON tool_item(selected_version_pk) WHERE selected_version_pk IS NOT NULL;

-- Candidate identity is independent of its eventual source-evidence digest.
-- A building candidate can retain known call/result facts and exact source
-- locators while required bytes are unavailable. Such facts are inspectable as
-- pending evidence. Proven operation existence is independent of payload
-- readiness; incomplete classification/correspondence remains unknown coverage.
-- New versions use new rows;
-- ready rows are immutable under the typed writer. Pinned/historical body reads
-- name this version, never mutable payload columns on the identity head.
CREATE TABLE tool_item_version (
  pk                INTEGER PRIMARY KEY AUTOINCREMENT,
  tool_item_pk      INTEGER NOT NULL,
  session_pk        INTEGER NOT NULL,
  evidence_revision INTEGER NOT NULL CHECK (evidence_revision >= 0),
  ready             INTEGER NOT NULL DEFAULT 0 CHECK (ready IN (0,1)),
  version_hash      BLOB CHECK (version_hash IS NULL OR length(version_hash) = 32),
  kind              TEXT NOT NULL,          -- provider's operation kind, not a fabricated block kind
  agent_class       TEXT NOT NULL CHECK (agent_class IN ('agent', 'non_agent', 'unknown')),
  correspondence    TEXT NOT NULL CHECK (correspondence IN ('established', 'unresolved', 'conflict')),
  native_order      INTEGER,                -- provider-established order within its scope; not global adjacency
  at                INTEGER,
  call_present      INTEGER NOT NULL CHECK (call_present IN (0, 1)),
  call_id           TEXT,                   -- source invocation/pairing ID, not a Message or process ID
  input_value_pk INTEGER REFERENCES body_value(pk), -- JSON value, assembled in bounded chunks
  result_present    INTEGER NOT NULL CHECK (result_present IN (0, 1)),
  output_present    INTEGER NOT NULL CHECK (output_present IN (0, 1)),
  result_text_id    INTEGER REFERENCES text(id),
  result_blob_hash  BLOB,
  result_data_value_pk INTEGER REFERENCES body_value(pk),
  result_outcome    TEXT CHECK (result_outcome IN ('ok', 'error', 'unknown')),
  result_reason     TEXT,
  result_exit_code  INTEGER,
  result_duration_ms INTEGER,
  CHECK (result_present = 1 OR (output_present = 0 AND result_text_id IS NULL
      AND result_blob_hash IS NULL AND result_data_value_pk IS NULL AND result_outcome IS NULL
      AND result_reason IS NULL AND result_exit_code IS NULL AND result_duration_ms IS NULL)),
  CHECK (output_present = 1 OR (result_text_id IS NULL AND result_blob_hash IS NULL)),
  CHECK (result_text_id IS NULL OR result_blob_hash IS NULL),
  CHECK (call_present = 1 OR input_value_pk IS NULL),
  CHECK (ready = 0 OR version_hash IS NOT NULL),
  FOREIGN KEY (tool_item_pk, session_pk) REFERENCES tool_item(pk, session_pk),
  UNIQUE (pk, tool_item_pk, session_pk)
) STRICT;
CREATE INDEX tool_item_version_owner ON tool_item_version(tool_item_pk, session_pk);
CREATE INDEX tool_item_version_input_value ON tool_item_version(input_value_pk);
CREATE INDEX tool_item_version_data_value ON tool_item_version(result_data_value_pk);
CREATE INDEX tool_item_version_result_text ON tool_item_version(result_text_id) WHERE result_text_id IS NOT NULL;

-- Immutable contributors/field locators of a candidate, including old versions
-- and UI transformations. The stable version key works before a digest exists.
-- Identical distinct chunks keep different slots; replacement uses a new owning
-- candidate version. Body-dependent associations pin that exact candidate, not
-- only whichever digest the identity head currently selects.
CREATE TABLE tool_item_source (
  tool_item_version_pk INTEGER NOT NULL,
  tool_item_pk      INTEGER NOT NULL,
  session_pk        INTEGER NOT NULL,
  input_pk          INTEGER NOT NULL REFERENCES input_ref(pk),
  source_chain      INTEGER NOT NULL CHECK (source_chain > 0),
  source_start      INTEGER NOT NULL CHECK (source_start >= 0),
  source_end        INTEGER NOT NULL CHECK (source_end >= source_start),
  field             TEXT NOT NULL,
  slot              INTEGER NOT NULL CHECK (slot >= 0),
  constructor       TEXT NOT NULL,
  PRIMARY KEY (tool_item_version_pk, input_pk, source_chain, source_start, source_end, field, slot),
  FOREIGN KEY (tool_item_version_pk, tool_item_pk, session_pk)
    REFERENCES tool_item_version(pk, tool_item_pk, session_pk)
) STRICT;
CREATE INDEX tool_item_source_owner ON tool_item_source(tool_item_pk, session_pk);
CREATE INDEX tool_item_source_input ON tool_item_source(input_pk); -- FK/excision lookup, never a full source scan

-- Shared immutable ownership proofs. Visibility requires BOTH selected operation
-- root membership and item.selected_version_pk. An append path-copies the root
-- directory, not every unchanged proof row. Unknown ownership has no proof leaf.
CREATE TABLE tool_item_association (
  pk INTEGER PRIMARY KEY AUTOINCREMENT,
  tool_item_version_pk INTEGER NOT NULL,
  tool_item_pk INTEGER NOT NULL,
  message_pk INTEGER NOT NULL,
  session_pk INTEGER NOT NULL,
  kind TEXT NOT NULL CHECK(kind IN('message_owner','wrapper_parent','wrapper_overlap')),
  input_pk INTEGER NOT NULL REFERENCES input_ref(pk),
  source_chain INTEGER NOT NULL CHECK(source_chain>0),
  source_start INTEGER NOT NULL CHECK(source_start>=0),
  source_end INTEGER NOT NULL CHECK(source_end>=source_start),
  constructor TEXT NOT NULL,
  FOREIGN KEY(tool_item_version_pk,tool_item_pk,session_pk)
    REFERENCES tool_item_version(pk,tool_item_pk,session_pk),
  FOREIGN KEY(message_pk,session_pk) REFERENCES message(pk,session_pk),
  UNIQUE(pk,message_pk,session_pk),
  UNIQUE(tool_item_version_pk,message_pk,kind,input_pk,source_chain,source_start,source_end)
) STRICT;
CREATE INDEX tool_item_association_item ON tool_item_association(tool_item_pk);
CREATE INDEX tool_item_association_message ON tool_item_association(message_pk);
CREATE INDEX tool_item_association_input ON tool_item_association(input_pk,source_chain);

-- Direct operation lookup refinement. A block citation remains a block citation;
-- only tool-call lookup follows this proof. Retained proof addresses outlive a
-- disposable body; no old body graph is leased solely to keep an anchor alive.
CREATE TABLE tool_call_anchor (
  anchor TEXT NOT NULL PRIMARY KEY,
  tool_item_pk INTEGER NOT NULL REFERENCES tool_item(pk),
  tool_item_version_pk INTEGER NOT NULL,
  session_pk INTEGER NOT NULL,
  input_pk INTEGER NOT NULL REFERENCES input_ref(pk),
  source_chain INTEGER NOT NULL CHECK(source_chain>0),
  source_start INTEGER NOT NULL CHECK(source_start>=0),
  source_end INTEGER NOT NULL CHECK(source_end>=source_start),
  constructor TEXT NOT NULL,
  FOREIGN KEY(tool_item_version_pk,tool_item_pk,session_pk)
    REFERENCES tool_item_version(pk,tool_item_pk,session_pk)
) STRICT, WITHOUT ROWID;
CREATE INDEX tool_call_anchor_item ON tool_call_anchor(tool_item_pk);
CREATE INDEX tool_call_anchor_version ON tool_call_anchor(tool_item_version_pk);
CREATE INDEX tool_call_anchor_input ON tool_call_anchor(input_pk,source_chain);

-- One row per tool call, paired with its result inside the same session. The
-- owner of derived call summaries; raw result facts remain on their block.
CREATE TABLE tool_call (
  pk                INTEGER PRIMARY KEY AUTOINCREMENT,
  session_pk        INTEGER NOT NULL,       -- kept for index speed; the composite keys make it agree with its blocks
  message_pk        INTEGER,
  operation_id      TEXT NOT NULL,
  tool_item_pk      INTEGER,
  tool_item_version_pk INTEGER,
  call_block_pk     INTEGER,
  result_block_pk   INTEGER,                -- NULL until the result is seen
  at                INTEGER,
  tool_value_pk INTEGER REFERENCES body_value(pk),
  action TEXT CHECK(action IN('shell','code','file_read','file_write','file_edit','search','web','subagent','mcp','other')),
  command_value_pk INTEGER REFERENCES body_value(pk), -- exact given UTF-8, chunked
  command_norm_value_pk INTEGER REFERENCES body_value(pk),
  path_value_pk INTEGER REFERENCES body_value(pk),
  outcome           TEXT NOT NULL CHECK(outcome IN('ok','error','unknown')), -- facts of the records, never of the clock;
                                            --   `pending` is computed at read time from live state (model.md)
  outcome_reason    TEXT,                   -- a reason code from model.md (no_result, background_never_reported, ...)
  exit_code         INTEGER,
  duration_ms       INTEGER,                -- NULL (unknown) when the times are equal or reversed
  child_session_id  TEXT,                   -- natural id of a spawned subagent (may arrive later)
  inherited         INTEGER NOT NULL DEFAULT 0, -- part of an inherited message
  CHECK((call_block_pk IS NOT NULL AND message_pk IS NOT NULL) OR tool_item_version_pk IS NOT NULL),
  CHECK((call_block_pk IS NULL)=(message_pk IS NULL)),
  CHECK((tool_item_pk IS NULL)=(tool_item_version_pk IS NULL)),
  FOREIGN KEY(call_block_pk,message_pk,session_pk) REFERENCES block(pk,message_pk,session_pk),
  FOREIGN KEY(tool_item_version_pk,tool_item_pk,session_pk)
    REFERENCES tool_item_version(pk,tool_item_pk,session_pk),
  FOREIGN KEY(tool_item_pk,session_pk,operation_id)
    REFERENCES tool_item(pk,session_pk,id),
  FOREIGN KEY (result_block_pk, session_pk) REFERENCES block(pk, session_pk),
  UNIQUE(pk,call_block_pk,message_pk,session_pk)
) STRICT;
-- One summary per immutable call-block incarnation; NULL permits session-owned items.
-- Different body-specific facts or pairings require distinct call blocks while an
-- older body still leases its summary. Operation IDs need not be unique here.
CREATE UNIQUE INDEX tool_call_block ON tool_call(call_block_pk);
CREATE INDEX tool_call_item ON tool_call(tool_item_pk);
CREATE INDEX tool_call_item_version ON tool_call(tool_item_version_pk);
CREATE INDEX tool_call_message ON tool_call(message_pk);
CREATE INDEX tool_call_operation ON tool_call(operation_id);
CREATE INDEX tool_call_session ON tool_call(session_pk, at);   -- exists, seq(...), the changes view
CREATE INDEX tool_call_tool ON tool_call(tool_value_pk);
CREATE INDEX tool_call_action  ON tool_call(action, outcome);
CREATE INDEX tool_call_failed  ON tool_call(session_pk) WHERE outcome <> 'ok';
CREATE INDEX tool_call_path ON tool_call(path_value_pk);
CREATE INDEX tool_call_command ON tool_call(command_value_pk);
CREATE INDEX tool_call_repo_command ON tool_call(command_norm_value_pk, at);   -- repo_command_last recompute
CREATE INDEX tool_call_result  ON tool_call(result_block_pk) WHERE result_block_pk IS NOT NULL;   -- foreign-key check

-- Serving substring corpus only. Canonical command/path bytes live in body_value.
-- No prepared/unselected summary enters this index; bounded selected-corpus
-- publication and retirement remain part of the open search implementation.
CREATE VIRTUAL TABLE tool_call_trigram USING fts5(command, path,
    content='', contentless_delete=1, tokenize='trigram');

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
-- holds no write transaction. A small delta may commit with its checkpoint;
-- a larger provider delta uses the parse_work admission barrier below. Logical
-- state publication checks the index generation, basis, state_generation
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

-- A provider-state preparation barrier, not a canonical Message publication.
-- Admission pins index incarnation plus the exact parser predecessor. New
-- source snapshots are refused while work exists; already pinned SQLite views
-- remain coherent. No provider interpretation runs under a write transaction.
-- Each admitted batch stages reserved @work:<work_id> parser_cell members before
-- seal, then applies and deletes a bounded keyset. Final publication changes
-- only this control row and the parser head/resume/counts/checkpoint, coupled
-- to the last required canonical publication by its owner. It never
-- adopts or copies an unbounded delta. Public canonical Messages are untouched.
-- Same-prefix growth stays owed. Basis replacement/retirement invalidates this
-- row; old-generation members remain subject to bounded parser_cell collection.
-- Interrupted unsealed staging may instead discard its members in bounded
-- keysets without changing published state. Sealed application must complete or
-- be invalidated; it cannot replay interpretation from a partly changed map.
CREATE TABLE parse_work (
  input_pk          INTEGER NOT NULL REFERENCES input_ref(pk),
  chain             INTEGER NOT NULL,
  work_id           INTEGER NOT NULL CHECK (work_id > 0), -- same monotonic allocator as state generations
  state_generation  INTEGER NOT NULL CHECK (state_generation > 0),
  state_step        INTEGER NOT NULL CHECK (state_step >= 0),
  basis_digest      BLOB NOT NULL CHECK (length(basis_digest) = 32),
  acquired_upto     INTEGER NOT NULL CHECK (acquired_upto >= 0),
  phase             TEXT NOT NULL CHECK (phase IN ('staging','applying','discarding')),
  staged            INTEGER NOT NULL DEFAULT 0 CHECK (staged >= 0),
  applied           INTEGER NOT NULL DEFAULT 0 CHECK (applied BETWEEN 0 AND staged),
  inspected_upto    INTEGER NOT NULL CHECK (inspected_upto >= 0),
  parsed_upto       INTEGER NOT NULL CHECK (parsed_upto BETWEEN 0 AND inspected_upto),
  next_status       TEXT NOT NULL CHECK (next_status IN ('parsing','waiting_tail','finalizing','done')),
  finalization_pending INTEGER NOT NULL CHECK (finalization_pending IN (0,1)),
  framed            INTEGER NOT NULL CHECK (framed >= 0),
  used              INTEGER NOT NULL CHECK (used >= 0),
  ignored           INTEGER NOT NULL CHECK (ignored >= 0),
  unrecognized_n    INTEGER NOT NULL CHECK (unrecognized_n >= 0),
  undecodable       INTEGER NOT NULL CHECK (undecodable >= 0),
  PRIMARY KEY (input_pk, chain),
  CHECK (framed = used + ignored + unrecognized_n + undecodable),
  CHECK (inspected_upto <= acquired_upto)
) STRICT, WITHOUT ROWID;

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
```

**Deliveries.** A second delivery line with the same `id` and the same content
is the same row. With different content (a `not_sent` line followed by the
proxy's later report, say), the line with the later `observed_at_ms` gives
the outcome, and equal times are broken by the higher BLAKE3 of the line's
bytes; the rule depends only on the lines, never on arrival order
(`sources/README.md`, delivery lines).

## Full-text search

- **Two indexes.** Plain search uses `text_fts` (prose); `output:` uses
  `output_fts` (tool output, which is 68% of all text and would otherwise
  drown prose hits), and plain search shows a count of tool-output hits as a
  hint. A text used by both kinds is in both. Membership follows surviving block and ToolItem result references, not arrival order. A later proven association reuses the same result/text, rather than inserting another output contribution. For block-only content: when a block uses an existing text,
  the writer adds the text to that block's index if no earlier block of the
  same kind had (a subagent's final message, first indexed as prose in the
  subagent, is added to `output_fts` when the parent's tool result uses
  it), and when the last block of a kind using a text is deleted, the text
  leaves that index. The prototype's first writer missed this and found 4 of
  5 hits for one word (`prototype/README.md`).
- **Contentless, folded.** Both indexes are contentless (`content=''`) with
  deletes enabled, keyed by `text.id`. The writer inserts each text's folded
  form and deletes by rowid, in the same transaction as `text`. Nothing reads
  text back from them, so the folded index and the unfolded UTF-8 value referenced by `text.value_pk` cannot
  disagree.
  **Statistics qualification remains open.** Removing a row from matches is
  not proof that native BM25 now describes only surviving texts. In SQLite
  3.53.2, the contentless-delete storage path installs a tombstone without
  the row/token-total decrements used by text-aware deletion
  ([pinned storage implementation](https://github.com/sqlite/sqlite/blob/version-3.53.2/ext/fts5/fts5_storage.c#L579-L608)).
  The DDL above therefore does not by itself qualify selected-corpus ranking
  or rebuild-equivalent scores after retirement. The chosen search
  representation must pass the fresh-corpus and cleanup comparisons in law 6;
  filtering deleted hits or periodically merging segments is not that proof.
- **Folding.** `remove_diacritics 2` folds only letters that decompose into a
  base letter plus combining marks; `ł Ł ø Ø đ Đ ħ Ħ ı ß` have none. One Rust
  function folds them (`ł→l, ø→o, đ→d, ħ→h, ı→i, ß→ss`, capitals alike) both
  for the text the writer indexes and for the words the query compiler
  matches, so index and query cannot disagree. Changing its map changes
  `INDEX_VERSION`.
- **Snippets** are cut by Polylogue from the chunked UTF-8 value referenced by `text.value_pk`, at the positions of the
  query words found by matching their folded forms.
- **Choosing the driving side.** For each query the planner compares the
  scope's size (text blocks after the non-text filters) with each search
  word's document frequency (`fts_vocab_text`, `fts_vocab_output`). A
  candidate from the scope costs one index seek, far more than one posting
  read from the index, so the scope drives only when it is small: at most
  256 text blocks, or 64 times its text blocks still below the word's
  document frequency (the prototype measured a scope a third of the archive
  wide at 5.1 s driving, against 0.09–0.23 s for the other plans;
  `prototype/README.md`). A session's blocks are read through
  `message_session` and `block_message`, so `block` needs no session index.
  When the scope drives (a session, a repository, a short time range): candidate texts come from the scoped sessions' blocks and are
  matched with `MATCH … AND rowid IN (temporary table)`. Otherwise the index
  drives: matches are ranked inside the index (`ORDER BY rank` on
  `text_fts` alone), then filtered by scope in rank order, fetching further
  ranks until the page is full; a page is never cut short. Rows tied on
  rank are read through to the end of the tie before the page is final (the
  prototype doubles its batch each round), so ties never split differently
  between plans. For a recency
  order (`sort by`, or the session default when no relevance is asked for)
  and a frequent word, the newest sessions drive instead: their blocks are
  walked newest first and each text is checked with `MATCH … AND rowid =`,
  stopping when the page is full. Both paths must return the same rows
  for the same query (`plan/testing/laws.md`, law 82); `query.md` owns the
  query semantics.
- **Never join every hit.** Rows are joined to blocks, messages and
  sessions only for the page being returned (and the candidates needed to
  fill it). Joining every match first and sorting afterwards took 166 s for
  a word in 488,000 texts on the 3-million-message benchmark; ranking inside
  the index and joining the page took 0.4 s (`evidence/ingest-benchmark.md`).
- **Measured costs** (warm, 20.3 GB of Claude Code data, 1.65 million
  texts): relevance ranking inside the index costs about 0.9 µs per matching
  text (32 ms for "error" in 39,000 texts, 107 ms for "test" in 130,000,
  425 ms for "the" in 488,000; two words together, 23 ms, since the rarer
  drives). Walking newest sessions for a frequent word: 1–31 ms. So the
  200 ms first-page target holds for any word in fewer than about 200,000
  texts by relevance and for every word by recency; above that, a relevance
  query's `notes` say how many texts matched.

## Change counter and generation

`session.changed` comes from a counter in `meta` that the single writer
increments once per transaction; it is a sequence number, not a clock. Every
write that changes a session's records or derived products sets it,
including a change that moves neither its message count nor its sort key (a
corrected role or `written_by`). Live columns set `live_changed` instead, so
the clock-driven trickle of hook and process lines does not flood consumers
of `changed`. When a session is written, the writer also bumps `changed` on
every session whose `link` names it as parent, so a late parent's children
get their lineage marks recomputed, and, on each subagent ancestor whose
`tree_cost_nusd` the write changes, applies the difference and sets that
ancestor's `changed` and `session_facts.from_changed` together, so the
ancestor is not derive's work.

A rebuilt `index.db` starts the counter again, so `meta` also holds a
`generation`, a random ID chosen when the file is created. The daemon also
starts a new generation after an unclean stop followed by a new boot (a
power loss or operating-system crash, the only case in which
`synchronous=NORMAL` can lose a committed transaction), so a counter value is
never reused within a generation. At start, a file whose `boot_id` differs and
has no `clean_stop` gets a new generation.

**The change feed** (`polylogue changes`). A position is
`GEN:COUNTER:ARCHIVE`: the index generation and counter cover session rows;
`ARCHIVE` is `archive.db`'s write counter, and notes and excisions are
reported in `local_seq` order (`archive-db.md`, local order), so marker notes
and notes from a union are reported when they arrive here, whatever their ID.
Sessions removed without an excision are reported from `removed_session`.
After a "generation changed, read everything again" line, the feed lists
every excision in `archive.db`, so a consumer that replaces its state also
applies any excision it missed and never keeps content the operator removed.

## Read-only SQL views

The tables above are internal and change with `INDEX_VERSION`. For people and
tools that want plain SQL (Datasette, DuckDB, a quick script), a small set of
documented views is the public, versioned surface: `v1_sessions`,
`v1_messages`, `v1_tool_calls`, `v1_notes` and `v1_facts`, with the field
names of the query language's units (`query.md`) and of the export format.
`polylogue sql 'SELECT …'` (`surfaces/cli.md`) runs in the daemon on a read-only
connection with `archive.db` attached, creates these as temporary views on
that connection (so no file's schema changes), and runs the statement in one
snapshot under the read time limit (`blobs.md`, WAL). The views
apply the query language's defaults only where their name says so:
`v1_sessions` leaves out subagents and hidden sessions, `v1_sessions_all`
includes them. A change of meaning or a removed column makes `v2_…`; `v1_…`
stays until the next major release, the one sanctioned case of an old and a
new form side by side, because outside readers cannot be updated in the same
change. (Decided by the spec authors: versioned views give a public contract
without exposing internal tables; the operator may overrule.)

Evidence: `evidence/architecture-review.md` (path authority, contentless
indexes, the merged parse table, the planner rule, unsynced fresh builds);
`evidence/deep-schema.md` (no pending-work tables, so staleness is a
comparison); `evidence/conceptual-concurrency.md` (swap, unsynced marker,
feed); `evidence/conceptual-idempotency.md` (feed positions, version walk); `polylogue-013x` (tool bodies searchable),
`polylogue-g0s6k.6`, `polylogue-mp5xq` (SQL views); audit P29-A5, P30, P33-A6.

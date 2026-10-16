# archive.db

What cannot be recomputed: inputs and their revisions, where they were found,
notes, kept results, excisions, variants, settings. How bytes are stored is in
`blobs.md`.

## Rules

- **Global keys.** Every row has a globally unique key: a content hash, a
  UUIDv7, or a number within such a key (a revision within its input, an
  alignment within its variant), never a name or an archive-wide local
  sequence. So the archives of two machines combine as a plain union of rows.
  A union inserts input bytes through the store step, which applies every
  excision to them as to new bytes, and resets every text excision's
  `applied_upto`.
- **Only gains rows.** No row is updated or deleted, except:
  `meta` (the write counter and archive integrity qualification below); `blob` and `extent`, the physical
  storage of bytes, which the store rewrites when it joins pieces, finishes a
  blob after a crash or re-encodes a delta as plain (the revisions they
  describe do not change); `revision.blake3`, set once from NULL when the
  revision is complete (`blobs.md`, the full check) and never changed; and
  excision, which deletes or overwrites what it names and advances its own
  `applied_upto`. Excision is the only thing that
  changes a revision's bytes, only in the excised spans, or for a
  whole-document format by a structural rewrite that removes only the excised
  records; an `excised` observation records which. Every other later state (an edited note, a
  withdrawal, a renamed saved query, a file seen again or changed) is a new
  row referring to the old one. The archive as it was at any time T is the
  rows created by T, read with the physical storage as it is now and a
  revision's `blake3` as unknown if it was set after T.
- **Local order.** `note.local_seq`, `excision.local_seq`, `revision.local_seq` and `setting.local_seq` are this
  archive's write counter at the transaction that inserted the row here. They
  are assigned by the local writer, never copied in a union, and never a key;
  the change feed and the read view use them (`index-db.md`, change counter).
  Restoring the same archive from Sinex preserves its own published local
  positions and continues above the verified retained high-water mark. A union
  assigns destination-local order to imported rows; it never copies another
  archive's counter. Imports advance this archive's write counter before any
  subsequent local write. Original publication metadata remains scoped to its
  registered producer/archive lineage in Sinex. A copied archive running
  independently registers a new lineage; resuming the original lineage requires
  fencing its previous writer. Contribution completion and the designated
  Polylogue resolver's completed union publication are separate chains, not a
  single counter across archives (`polylogue-sinex-joint/topics/03-transport.md`,
  Archive publication and union resolution; joint #29). Record-only writes also
  close a publication, so the last note never waits for a future input revision.
- **Migrations.** `archive.db` cannot be recomputed from anything local (in
  backed mode it can be restored from Sinex, but that is recovery, not a
  schema path), so its schema changes by
  migration: an ordered list of SQL steps in the `store` crate, applied at
  daemon start after preserving the pre-migration main/WAL/SHM set under archive ownership, with the step number in
  `PRAGMA user_version`. Before the first release there are none. A step
  runs in bounded batches (one transaction each, so no step holds the writer
  long), and the step number moves only with its last batch, so a stop
  mid-step resumes that step; a fresh archive and an upgraded one must be
  identical (law 124).
  `archive.db`'s tables all exist from slice 1; later slices only start
  writing them.
- **A newer archive refuses older writers.** A binary that finds an archive
  newer than it knows (a higher migration step here or in `embeddings.db`)
  refuses to start, naming both versions. `index.db` is different: it is
  rebuildable, so an `INDEX_VERSION` other than the binary's, higher or
  lower, makes the daemon build a new one (`index-db.md`). An older
  binary reading a newer value keeps it as it is (R68).
- **Writer.** Only the daemon writes here, and it increments `meta`'s write
  counter once per transaction. Notes, tags, settings, excisions and imports
  arrive as client requests over its socket (`decisions/process.md`); each
  request carries an ID chosen by the client, and the rows it creates take
  their IDs from it, so a resent request writes nothing new. All access goes through one module,
  `store::inputs` (with `store::archive_meta` for the archive's own rows), the only place another storage backend could be added
  (`decisions/sinex.md`).

## Recovery qualification

The controlled-copy recovery path in `storage/README.md` persists one typed
`meta` value, key `recovery_qualification`. Its versioned value contains
`latest_acknowledged_head: unknown`, the uncertain-generation reason, affected
durable tiers/frame positions and the BLAKE3 digest of the preserved-source
manifest in the recovery receipt. It records archive integrity scope, not a user
setting, annotation, parser state or index flag. Ordinary writes, successful
parsing and index rebuild never clear it. The first path has no clearing command.

Every read pins this value with its archive snapshot and exposes it as `recovery`
(`query.md`, Results); status and zero-row answers carry it too. Complete local
processing does not erase uncertainty about the recovered historical head.

An older binary must refuse this format before it can ignore the qualification.
The first native implementation therefore introduces a recovery-aware migration
step through `PRAGMA user_version`, and tests the actual predecessor binary's
refusal. An extra meta key under an old format version is insufficient. Normal
older-format archives retain the owning migration rules above; recovery updates
only its separately owned destination. This contract is planned, not evidence
that a migration or recovery command has shipped.

## Tables

```sql
-- The write counter (key 'write_counter'), incremented once per transaction,
-- and typed archive integrity qualification (key 'recovery_qualification').
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
  id            TEXT PRIMARY KEY,        -- immutable import key (model.md, immutable import identity),
                                         -- or UUIDv7 for a location that can grow or change
  format        TEXT NOT NULL,           -- 'claude-code-jsonl', 'codex-rollout', 'codex-state-sqlite', 'chatgpt-export', ...
  original_hash BLOB,                    -- immutable import's original 32-byte digest; survives byte excision
  context_json  TEXT,                    -- version-1 string array (model.md); [] is self-identifying
                                         -- both NULL for ordinary observed/location-backed UUID inputs
  CHECK ((original_hash IS NULL AND context_json IS NULL) OR
         (original_hash IS NOT NULL AND length(original_hash) = 32 AND
          context_json IS NOT NULL AND json_valid(context_json) AND json_type(context_json) = 'array'))
) STRICT;

-- Every distinct content of an input. Never deleted except by excision. The
-- observation that first saw a revision is a query (the earliest by seen_at, then
-- id), not a column, so the rows of an immutable tuple-named input are identical on
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
  local_seq     INTEGER NOT NULL,        -- destination-local insertion order (Rules), never a global key
                                         -- backed publication metadata is scoped by registered archive and
                                         -- publication kind; union never imports another archive's counter
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
  stands_for    TEXT,                    -- target location from the container's retained mapping (sources/README.md);
                                         -- 'zip:' and 'copy:' observations: the location path these bytes are an
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
                                         --   owning session/message IDs, complete content and occurrence (user-data.md)
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
  author        TEXT NOT NULL,           -- 'operator', or an agent session id
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
```

**Current revision, withdrawal, conflict**: `user-data.md`, Notes, owns
these rules. Withdrawn revisions do not supersede their predecessors; a
withdrawn root withdraws its family. Different current values of a
single-valued kind conflict across families too; equal values agree.
All states are computed from notes and relations, never stored as flags.

**Inline-marker notes** live here, not in `index.db`, although they come from
inputs: people and agents revise and withdraw them like any note, and a
revision needs a durable target. Their IDs come from the record's owning
session/message IDs, complete marker content and occurrence, never its
position (`user-data.md`, Inline markers), so re-parsing never
adds a second copy. Whether one is *in force* is derived: `index.db` holds the
set of marker IDs the current parse extracts (`marker`). Marker extraction is
the one write the parse step makes to `archive.db`: it adds only notes whose
IDs do not exist and are not named by a `note:` excision.

**Saved queries** are notes of kind `saved_query` (key: the name, value: the
query text). `query:<name>` resolves to the current one with that name; two
current families with the same name are an ambiguity, listed in the error.

**Pages** are notes too: a page, its slots and its members are notes with
relations, and need no tables of their own.

**Attachment conflicts.** An attachment ID seen with two different contents is
a conflict recorded as a `message_disagreement` row (`index-db.md`). The version kept is chosen by the merge
rules (`sources/README.md`, rule 2) like any other field, never by arrival,
and never resolved to no value.

## How excision works

Excision stays in the design: retention is forever, so it is the one way
content that must go is removed (decided by the operator). Only the operator
excises (`user-data.md`, the opening ruling on agents).

An `excision` row is a standing rule: in every round, and after any union of
archives, the loop removes whatever an excision names that still exists
(`convergence.md`, the excise row of the loop's table).

**Usage survives excision.** Before a session's rows go, the excision row
records its usage as numbers only, in `usage_json`: per provider account,
model and subscription window, the request-bearing message count and tokens
by kind. No text, ID or time finer than the window survives. Window usage,
the usage outlook and archive totals add these numbers and say so
("includes 1 excised session"), because the provider counted that usage
(`analysis.md`, subscription windows); the session itself no longer exists
anywhere.

**Scopes.**

- `input:<id>`: the input, its revisions, extents and observations are
  deleted, and its blobs go to `deleting` unless another input's extent uses
  them (a blob that is another input's delta base is first re-encoded plain
  for that delta). Before the rows go, the excision row records the input's
  locations, immutable import identity tuple when present, and the BLAKE3 of
  its original bytes, revisions and blobs in `refused_json`.
- `session:<id>`: an input that fed only that session is excised as above.
  An input shared with other sessions is rewritten without the session's
  records:
  - JSONL (a hook spool file, `history.jsonl`): the session's lines become
    spaces of the same length, so offsets and change classification are
    unaffected;
  - a JSON document (an export): rewritten in the format's own structure; the
    input keeps its ID, its revision's extents point at the rewritten blob,
    the old blob's hash goes into `refused_json`, and the `excised`
    observation's `spans_json` is `null` with `ids_json` listing the removed
    records (a structural rewrite has no byte spans);
  - a database snapshot: the session's rows are deleted on a copy of each
    snapshot with `secure_delete=ON` and without `VACUUM`, so freed cells are
    zeroed and page numbers stay; free pages holding the session's IDs or
    text are zeroed too. The rewritten newest base is stored plain and each
    earlier snapshot as a page patch over its own rewritten base;
  - an archive stored whole (an export ZIP): the member holding the session
    is rewritten by the rule for its own format (a JSON document as above),
    and the archive is rewritten with that member replaced and every other
    member byte-identical; the input keeps its ID, and the old blob's hash
    goes into `refused_json`.

  Copies of the session's messages in other sessions (resumes, forks, copied
  subagents) are excised with it: in their records only the text-bearing
  strings are overwritten, as a text excision overwrites a string; the
  structural fields (`type`, `uuid`, `sessionId`, `parentUuid`, usage
  numbers and the like) stay, so the copies still parse and keep their
  copy evidence, and their blocks render `[excised]`. The excision row
  lists the excised session's message IDs (`message_ids_json`), and the
  derive step marks every copy of one of them as inherited from an excised
  session: counted in no session, so its usage never counts again (the
  excision's `usage_json` covers it), even though the source is now absent,
  and the same after an index rebuild. The preview lists every session that
  inherits from it.
- `text:<blake3>`: names a string. Its occurrences are every JSON string
  value or SQLite text value in any stored revision that equals it or
  contains it: prose, tool arguments, structured output, commands, process
  arguments. They are found by scanning the stored bytes for the string's
  encodings (raw UTF-8 and its JSON-escaped forms), in input order, and each
  occurrence's bytes become `*` of the same length. An occurrence that cannot
  be overwritten in place (a deflated ZIP member, a SQLite overflow chain) is
  handled the way session excision handles that format. The vectors of the
  string and of every text value that contained it are deleted (step 4).
- `note:<id>`: `key` and the targets are deleted and `value` becomes
  `[excised]`. The row keeps `id`, `root`, `kind`, `author` and
  `created_at`, and its relations stay, so revision families, withdrawals
  and answers keep their meaning. A marker note named here is never extracted
  again.

**Order.** Every step finds its work from state, so a crash between any two
resumes where it stopped.

1. The `excision` row commits, with `refused_json` for input scopes, and for
   session scopes only the inputs that fed that session alone (shared inputs
   are rewritten, and only the hashes of their replaced revisions and blobs
   are refused, never their locations). From that commit, reads exclude what
   it names (rows of an excised input or session, texts equal to an excised
   string, the note) by joining the attached `excision` table, and render
   `[excised]`. Texts that merely contain the string are excluded once step 3
   rebuilds their sessions, before `excise` returns. Also at the commit the
   daemon empties every warm cache and drops the excised texts' vectors from
   memory.
2. Stored bytes, one input per `archive.db` transaction: the input's blobs
   are rewritten as new blobs with the extents repointed (blobs are
   immutable; a delta whose base changed is re-encoded), and an `excised`
   observation records the spans, the content-derived IDs of the records
   they touch and the hashes of the text values that held them. For a text scope the same transaction advances `applied_upto`.
3. Index: every session fed by an input with an `excised` observation that
   its `parse_state` has not seen is rebuilt; rows of excised inputs and
   sessions are deleted, and each removed session is recorded for the change
   feed (`index-db.md`). While a new `index.db` is being built beside the old
   one, these deletions are applied to both.
   This includes session-owned ToolItems and all their candidate versions,
   source locators, proved anchors, output-search references and derived call
   facts, including items with no Message owner. A Message or text removal
   follows its exact retained ranges and proved operation correspondence;
   it neither skips an unowned representation of those bytes nor guesses
   ownership for unrelated items. A surviving operation may retain its known
   structural facts while its removed payload is explicitly excised. Pending
   payload completion and later association must recheck the removal basis;
   they cannot restore an old body, select a pre-excision version or refill
   it from another representation. Complete this closure before claiming the
   ToolItem runtime projection qualified.
4. Every vector whose `text_hash` is in an `excised` observation's
   `text_hashes_json`, or belongs to a text only the excised session used, is
   deleted from `embeddings.md`'s `embedded`, and `near:` skips any vector
   whose hash is named there.
5. Bounded FTS `merge` steps rewrite every segment holding deleted entries,
   then each file is checkpointed with `TRUNCATE` (retried without waiting).
   `secure_delete` has already zeroed freed pages, so no `VACUUM` runs.
6. Sinex-backed mode: send the explicit tombstone/privacy order and shared
   replacements defined by `decisions/sinex.md`, Deletion. Byte purge is the
   fallback there, not the default excision mechanism; status keeps remote
   content removal and byte/key completion pending until each is confirmed.

`excise` returns when steps 1 to 4 are done, showing progress for a long
text scan; `status` shows steps 5 and 6 as pending until they finish.
Step 5's work is found from SQLite: the loop runs FTS `merge` steps until one
reports no change (`total_changes()` unchanged) after the latest excision's
`local_seq` the daemon has seen since start, so after a restart it runs them
once more; `status` shows step 5 pending until that point and step 6 while
Sinex reports holding the content.

**What excision cannot reach.** Excision removes the content from the
archive's files, not from copies Polylogue does not control: filesystem
snapshots and backups taken before it (the operator's btrfs snapshots,
borg), and the provider's own file while it still exists (a Claude Code
transcript keeps the text until Claude Code deletes it). `excise` says so
in its output, naming the provider files that still hold the content and a
pre-migration copy of `archive.db` if one is still present (`README.md`,
retention), and `status` lists them as "excised location, still present". Backups age out
on their own schedule; Polylogue does not touch them. In Sinex-backed mode
the same limit holds for Sinex's backups, bounded by its key-table retention
(crypto-shredding takes effect in every backup once its retention window,
30 days by default, has passed; `sinex-spec/decisions/privacy.md`).
Per-session sealing in Polylogue's own blob store, which would give the
same bound without Sinex, is set aside (`SET-ASIDE.md`).

**Never stored again.**

- **Collect** treats a location in any `refused_json` as excised while the
  same file (device and inode) is there: it records nothing and stores
  nothing from it, and `status` lists it as "excised location, still
  present". A new file at that path is a new input.
- **Store** refuses any revision or blob whose hash is in a `refused_json`
  list, and applies every text and session excision to new bytes before
  storing them and before classifying the change. The engine does this, not
  `store`: it asks the format's parser for the byte spans of the excised
  sessions' records in the new bytes (`sources` crate, `records_of`) and
  passes the overwritten bytes and spans to `store`. Each store that
  overwrites bytes this way writes, in the same transaction, an `excised`
  observation for the input naming the excision, the spans and the
  pre-overwrite IDs, exactly as step 2 does. An imported file whose original byte digest is
  refused, under any interpretation context, is stored in its excised form (which usually exists already); for
  an input scope, nothing is stored.
- **Change classification** of an input with an `excised` observation masks
  the recorded spans (`storage/blobs.md`, excised inputs).
- **Parse** reads a string as excised when its bytes, mapped through the
  chain's extents to blob ranges, lie inside a recorded span: a block with no text, rendered `[excised]`, with no search entry
  and no vector. No string is read as excised because of its content.
  Content-derived IDs of records touching a span come from `ids_json`, so
  excision never changes an ID.
- **Embeddings** are inserted only after checking that the text is not
  excised (`embeddings.md`), so a request in flight cannot write a vector
  back.

**Follow-on removals.** Inline-marker notes go with the message they came
from (`note.source`), by any scope. Notes whose every target lies inside the
scope are excised with it; a note with other targets keeps them and loses
the excised ones. In a kept result, each row inside the scope keeps its
reference, and its displayed values and match rows become `[excised]` (a
rewrite of `rows_json` under the excision exception). Variants of excised
content go with it. A delivery whose recipient session is excised goes with
that session's spool lines; deliveries that merely referenced it keep their
row, with the reference shown as excised. The `carry` of every chain whose
bytes changed is discarded, so its parse restarts from the chain's start.

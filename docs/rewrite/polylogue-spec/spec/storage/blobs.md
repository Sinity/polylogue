# Blobs, inputs and revisions

How Polylogue keeps the exact bytes of every input: blobs, inputs, revisions,
extents, observations, provider databases, and crash safety. The tables are
in `archive-db.md`.

## Blobs

A **blob** is a byte string named by the BLAKE3 of its uncompressed content
and compressed with zstd (level 3).

Every blob is a whole stored input or a piece of one; attachments and side
files are either inputs of their own (a download, a companion file) or
members of an archive stored whole and read in place (a ZIP member is
addressed `zip:<input>/<member>`). The one exception is an archive member
that becomes an older chain of an input at a location (`sources/README.md`,
archives of provider files): a chain's bytes are extents, which are blob
ranges, so that member's uncompressed bytes are stored once as a blob of
their own (deduplicated by hash), while the archive stays stored whole for
provenance. `extent` is the only reference that keeps a blob.

**BLAKE3** is the one hash function for blob names, input IDs, text hashes
(the key for embeddings) and the fallback message-ID hash. Its tree structure
lets a reader verify a byte range of a large blob without hashing all of it,
which fits seekable range reads, and Sinex's store already uses it. `b3sum`
replaces `sha256sum` for checking by hand. (Decided with the Sinex session,
joint #14; the old purchased vectors are re-keyed by BLAKE3 at import,
`plan/migration.md`.)

Where a blob lives depends only on its size; the size never decides whether
it is kept:

- **under 128 KiB**: inline in `archive.db` (`blob.data`), written in the same
  transaction as the rows that refer to it. Most inputs by count are small;
- **128 KiB and over**: a file `blobs/<first byte hex>/<blake3>.zst` in the
  zstd seekable format: independent 4 MiB frames plus a seek table, so one
  record of a 1.6 GB input is read by decompressing about 4 MiB. Plain `zstd`
  still decompresses these files.

Blob files are written to `blobs/tmp/`, synced, renamed into place, and every
directory created on the way synced. Every full read checks the BLAKE3; a
partial read is covered by the frames' checksums, so every frame is written
with zstd's checksum flag on: incompressible data is stored in raw blocks,
where a flipped byte decodes cleanly into wrong bytes and only the checksum
catches it (found by the joint `hashed-blobs` prototype's tests).

**Batched durability: new inputs share one sync.** Syncing each blob file
and its directory costs more than writing it: storing a real day of 6,880 media
files took 173 s with three syncs per blob and 3.6 s in one batch (the joint
`hashed-blobs` prototype), and a 120 MB first build took 275 s with a sync per
file against 4.7 s on tmpfs (`evidence/real-converge.md`). So the one rule is:
a round's **first observations** (inputs not seen before, whether found by
collection, a rebuild's re-collection or `import`) first insert their
`blob` rows as `writing` in one transaction, then write their files to
`blobs/tmp/`, `syncfs` once, rename every file into place, `syncfs` again,
and only then commit, in one transaction, the rows `present` with the
inputs, first revisions, extents and observations that refer to them. A
batch holds at most 512 files or 256 MiB of bytes, whichever comes first;
a round with more new paths stores several batches, so a first build of
hundreds of thousands of files (slice 1's restored spool files are about
944,000) keeps its memory and its transactions bounded (the prototype's
`FIRST_BATCH_FILES` and `FIRST_BATCH_BYTES`; a batch of 512 files still
pays three durable steps for all of them). It is the per-file protocol below
done for a batch: a row exists before its file, so a crash leaves only
`writing` rows the comparison cleans up, and no row is `present` before its
bytes are durable. An
append, rewrite or piece of an input already known syncs its file and
directory on its own, because those arrive a few at a time during live work
and `syncfs` also flushes other programs' dirty pages on the same
filesystem.

**Already-compressed bytes** (images, audio, video, and archive members
already deflated, recognised by their magic bytes) are stored at zstd level
1: higher levels gain nothing on them (Opus and WebP stayed at 94% at
level 3, for 25% more CPU), and the frames are still needed for checksums
and seeking. A mismatch is an error
naming the blob, shown in `status`, never passed to a parser.

**The file layer is the `hashed-blobs` crate**, shared with Sinex (joint #11,
#12). It holds bytes by hash, plain or as a delta, and reads extents as one
stream; it holds no policy (when to snapshot, when to use a delta, page maps
and joining stay in Polylogue's input layer, and so does the choice to keep a
small blob inline). The archive needs one storage root; a second root can be
added and is searched in order when reading, with the first used for
writing. Nothing tiers or moves data between roots. The file mode is a
parameter (Polylogue uses 0600).

```rust
// The skeleton's hashed-blobs crate (skeleton/crates/hashed-blobs) holds these signatures.
pub fn put(&self, bytes: impl Read, level: i32) -> Result<Hash>;                  // plain, seekable
pub fn put_delta(&self, base: &Hash, bytes: impl Read, level: i32) -> Result<Hash>; // zstd patch-from: base as prefix
pub fn base_of(&self, blob: &Hash) -> Result<Option<Hash>>;                       // None for plain blobs
pub fn read(&self, blob: &Hash) -> Result<impl Read>;                             // resolves a delta
pub fn read_range(&self, blob: &Hash, offset: u64, len: u64) -> Result<Vec<u8>>;
pub struct Extent { pub blob: Hash, pub offset: u64, pub len: u64 }
pub fn assemble(&self, extents: &[Extent]) -> Result<impl Read + Seek>;           // a revision's bytes
pub fn verify(&self, blob: &Hash) -> Result<Verified>;                            // reports; never repairs
pub fn delete(&self, blob: &Hash) -> Result<()>;                                  // caller checks references
```

## Inputs, revisions and extents

An **input** is one logical file: the history of one canonical location, or
one imported file. Its **revisions** are the distinct contents Polylogue has
observed, in order. Every revision is kept: providers rewrite their files and
drop information, so an older revision can be the only record of something.

A revision's bytes are described by **extents**: ranges `(blob, offset in
blob, length)` placed at offsets in the revision. An acquisition can retain
only an append window or changed pages; its blob is not thereby the full
revision. Reconstruction uses every required earlier extent and delta base
by immutable blob identity, with the input, chain and byte-window coordinates
that were captured. After storage, parsing and recovery read those bytes even
if the provider path was deleted or changed; missing retained ranges are
reported, never filled from the current path. A normalized session also keeps
its contributing input references, including companions and attachments;
the newest acquisition alone cannot stand for its complete evidence.

A logical SQLite row export and a physical database snapshot are different
representations. Store and label the actual acquired representation: the
former cannot prove retention of database pages or unexported columns.
Provider database collection follows the snapshot policy below; this
distinction does not replace it with logical exports. Unknown acquisition
coverage stays unknown rather than claiming a full revision from a tail or
omitted inputs (`sources/README.md`, parsing and merging).

Three kinds of change use this one extent mechanism:

- **Append** (Claude Code and Codex session logs while they only grow, hook
  spool files, `history.jsonl`): the new revision is the previous one plus
  one extent, a **piece**, holding only the new bytes, cut at the last
  complete line. Storing an append costs the bytes appended. Successive
  appends form a **revision chain**: revisions that are each a prefix of the
  next and share one set of extents.
  - **Head rewrites.** In a format with a mutable first record (Claude Code
    session files), a changed first line starts a new chain whose extents
    are one new extent for the first line followed by the previous chain's
    blob ranges after its first line, shifted by the change in the first
    line's length (extents are ranges into blobs, so nothing is copied), plus
    a piece for any appended bytes. The previous chain is **subsumed**: its
    bytes after the first line are a prefix of the new chain's, so it adds
    nothing to the session but its first record, which the new chain's
    replaces. A head rewrite writes its revision row at once, since it starts
    a chain (the extents name it), and records the chain it subsumes
    (`revision.subsumes`). Parsing and rebuilds never read a subsumed chain,
    so a rebuild and the incremental path give the same rows: citations
    name the newest chain, every message has one `message_source` row for
    it, `first_seen` still comes from observations and is unaffected, and
    the new chain's `parse_state` counts are its own.
  - **Pieces as they arrive, revisions once a minute.** Each file
    notification stores the new complete lines as a piece on the current
    chain, and parse reads them at once, so a new record is searchable within
    seconds. A `revision` row and its `observation` are written at most once
    per minute per file, covering every piece stored since the last one. A
    message whose bytes no revision covers yet has no `first_seen` until the
    covering observation is written (`convergence.md`, Parse); the evidence cutoff keeps
    one-minute resolution. After a crash, extents past the latest revision's
    length are pieces the next observation covers.
  - **Joining.** When the file has been quiet for the **quiet time** (15
    minutes), the small pieces after the last large one (1 MiB or more) are
    joined into one blob, so each byte is written at most twice. The
    revisions' bytes do not change.
- **Rewrite** (anything else that changed while present): a new revision
  chain starting with one extent over a new blob.
  - *When it is a delta.* If a zstd delta against the input's newest
    plainly stored blob is under half the plain size, and that base is at
    most 640 MiB, the blob is stored as a delta; otherwise plain and
    seekable. A rewrite under 128 KiB is stored inline whole, never as a
    delta. The plain size is estimated as the base's stored size scaled by
    the length ratio, so the new bytes are compressed once
    (`put_delta_within`, which writes the delta only when it fits, else
    nothing) and a second time only when the delta is refused, to store it
    plain. The estimate was within 0.1% at the median on 1,439 real Codex
    rewrite pairs and erred low by at most 36%, which only makes the rule
    stricter.
  - *What a delta is.* zstd's patch-from form (the base as a prefix, a
    window covering base and content, long-distance matching), recording
    the base's hash. A dictionary delta saved nothing from 4 MiB up, while
    patch-from gives 29 KB for a 100-byte change on a 256 MiB base and
    123 KB on 1 GiB (the joint `hashed-blobs` prototype). A delta's base is
    always plain, so reading a revision decompresses at most two blobs, and
    a base is never deleted while a delta uses it. Delta blobs are not
    seekable; only the raw view of an old revision reads one.
  - *Why 640 MiB.* It is a memory bound, not a cap on what is kept.
    Decoding a delta takes about twice its base (526 MB at 256 MiB, 2.1 GB
    at 1 GiB, measured), so 640 MiB stays near 1.3 GB, inside the build's
    2 GB target (`behavior.md`), and covers every Codex rollout on this
    machine (the largest is 0.60 GB). Writing takes about three times the
    base (the base, the new bytes and the compressor's tables: about
    1.26 GB at a 409 MB base on real Codex pairs,
    `evidence/codex-rewrite-deltas.md`), so a rewrite near the bound reaches
    about 2 GB, and the writer stores one such rewrite at a time.
- **Database snapshot** (below): changed pages overlaid on a base snapshot.

**Whole-file formats** (JSON documents, protobuf, single-file captures) have
no append: any change is a rewrite, stored once the file has been quiet for
30 seconds and at least every 5 minutes while it keeps changing. **Export
ZIPs** are stored whole as one input with one revision; members are
addressed as `zip:<input>/<member>`.

**IDs.** An imported file that cannot change (an export) is named by the hex
BLAKE3 of its bytes as first stored, and keeps that name after an excision;
storing the same bytes again only adds an observation. An input at a location
that can grow or change gets a UUIDv7 when first seen and keeps it for life.
Each revision records the BLAKE3 of its full bytes once it is complete
(for a growing input, at the full check, below), computed from the **stored**
bytes (the earlier revision's extents plus the new piece), never by
re-reading the live file: a live re-read would fold an edit the sampled
windows missed into the stored hash, and the edit would never be found
again ("absorption", found by Sinex's fuzzing of this design). Excision
never changes it, and a revision with an `excised` observation is verified
by its blobs' hashes instead.

**Order of chains.** Within a chain, revisions are ordered by length. Between
chains, the **current chain** is the one holding the latest observation at
the input's location, and only it is parsed incrementally. Revisions added
from elsewhere (a migration, an old copy, an archive member) are observed at
a `copy:` or `zip:` path of their own, never at the live location, so they
are older chains; a `zip:` or `copy:` observation is never a location, so an
import never moves the current chain. Their mtime is recorded and shown,
never used to order them; where older chains need an order (the writer's,
below the current chain's), it is by content, the BLAKE3 of each chain's
newest revision, never by arrival or chain number.

**Every revision is parsed.** An older chain is parsed once, unless a head
rewrite subsumed it. A record present
only in an older revision is merged into its session by the ordinary merge
rules (`sources/README.md`, rules 2 and 5, which also define "older").

## Observing a file

**`archive.db` decides which input a path belongs to.** Collect compares each
file with the latest `observation` for its `(machine, path)`. An input's
**location** is the path of its latest observation, so after a rename only
the new path is compared and only it can be observed deleted. "Latest" is
local insertion order (`archive-db.md`), since a rename and the append found
in the same round share a `seen_at`. The **stored length** a file is compared
with is the end of the current chain's last extent, including pieces no
revision covers yet; comparing with the newest revision's length would store
those pieces again (after an `index.db` rebuild, for example). `file_cursor`
in `index.db` is only a cache of that lookup (plus the last-seen time and
quiet time), rebuilt from `archive.db` when missing, so deleting or rebuilding
`index.db` never creates a second input, revision or observation for a known
path.

A file is compared with its last observation by the key `(device, inode,
size, mtime_ns, ctime_ns)`. The change time cannot be reset by `touch` or
`rsync -t`, so a rewrite that restores the modification time is still seen.
A device number alone is not identity: when inode, size, `mtime_ns` and
`ctime_ns` are equal and only the device differs (a remount; btrfs
subvolumes), the file is unchanged and the cache takes the new device.
'Replaced with more' and 'renamed' compare inodes within the same device or
this equivalence.

| Class | Rule | Cost | Stored |
|---|---|---|---|
| first | a path with no observation | — | a new input and its first revision; when an archive member or copy already stands for this path, a new chain of that input instead (an observation only, if a chain holds these bytes), which becomes its current chain |
| unchanged | key equal, and the last observation, or the last byte comparison that found the file unchanged (the cache's last-seen time, taken from the `stat` before that read), was at least 50 ms after the file's modification and change times; without the comparison time, a file once found racy would be re-read in every round until its next observation (otherwise a same-length edit within one timestamp tick would look unchanged: "racy unchanged", found by Sinex's fuzzing of this design; such a file is compared by its windows and, if they match, by a full hash) | a `stat` | nothing (the cache's last-seen time) |
| appended | same inode, larger, and the sampled windows of the stored length equal the stored bytes | 8–40 KiB read | a piece; an observation |
| head rewritten, rest appended | only for formats that declare a mutable first record (Claude Code session files): same inode, the first line differs, and the windows after the first line, shifted by its change in length, equal the stored bytes; it takes precedence over appended when both hold (a changed first line with more bytes after it) | 8–40 KiB plus the first line | a new chain (Head rewrites, above) whose revision is written at once; an observation. The new chain subsumes the previous one, so the session still comes from one chain: its parse starts from the previous chain's carry with every offset shifted by the change in the first line's length, re-reads only the first record (replacing the stored first message's rows) and continues incrementally |
| replaced with more | a different inode or device, size at least the stored length, and its first stored-length bytes equal the stored revision's (compared by hash, or by reading the stored extents when the hash is not yet set) | one read of the stored length (rare) | a piece; an observation with the new inode |
| touched | same inode and size, different times (`touch`, an `rsync -t` restore), and the file's bytes equal the stored bytes | one read of the file | nothing in `archive.db`; the cache takes the new key. Storing a revision identical to the stored one would add a chain that says nothing |
| rewritten | anything else with the file still present (smaller, or a window differs, or same size with different bytes) | one read of the file | a new revision chain; an observation |
| renamed | a new path whose device, inode and content match a known location, and the old path gone | `stat`s | an observation at the new path. Nothing is parsed again: a parser's path context is the input's first path (`convergence.md`, Parse) |
| deleted | a known path is gone | `stat` | an observation with the time it was noticed; the bytes stay |
| back | a path whose latest observation is `deleted` reappears | `stat` | an observation (then classified as above) |
| imported | `polylogue import` | one read | an input named by hash, or an observation if it exists. An archive member that stands for a location becomes an older chain of that location's input, or starts that input when no file has been seen there; one some chain already holds adds nothing (`sources/README.md`, archives) |

**Sampled windows.** For stored length L (the current chain's full length):
4 KiB at offset 0 (for a format with a mutable first record, at the end of
the first line, compared shifted by its change in length), the 4 KiB before
L, and up to 8 more 4 KiB windows at offsets `hash(L, input id, i) mod L`, so
the windows move at each growth and an edit in the middle is found within a
few appends. At most 40 KiB per observation.

**Excised inputs.** For an input with an `excised` observation, the recorded
spans are masked in both the live and the stored bytes when windows are
compared, and the full check hashes the live bytes with those spans replaced
by the stored filler and compares the result with the stored bytes. A live
file still holding excised text therefore classifies as it would without the
excision (`archive-db.md`, how excision works).

**Full check when quiet, and every 6 hours.** When an append-only file has
been quiet for the quiet time, the loop reads it once and compares the live
file's hash, up to the stored length, with the stored revision's hash (from
the stored bytes, above). A file that never goes quiet (a log appended to
all day) gets the same check every 6 hours. When it is due is read from
stored state, not from a recorded last check: the current chain's newest
revision has no `blake3`, and either the file has been quiet for the quiet
time or the newest revision of the chain with a `blake3` was first observed
at least 6 hours ago. A passing check sets the newest revision's `blake3`.
First and rewritten revisions get theirs when stored, since the whole file
was read. A mismatch is a rewrite (for a
format with a mutable first record, a mismatch confined to the first line
is a head rewrite), so an edit the windows missed is still caught (the 1.6
GB file: about 3 s). Sinex's fuzzing of this design (509 edits in 4 MiB
files, `sinex-spec/prototype/change-classifier/`) found no wrong class with
these rules; 3 edits were first read as appends and all were caught by
later windows.

**Edge cases.** A partial last line waits for the next observation. At the
quiet-time check, a partial last line still there (a writer that crashed
mid-line) is stored as a piece and the parser counts it as a bad record;
otherwise those bytes would never be stored, since every other store is cut
at the last newline. `parsed_upto` stays at the line's start, so if the line
is ever finished the record is parsed whole. A file is
read only up to the size its `stat` reported, so growth during the read is
picked up next time. That `stat` is taken on the open file the bytes are
read from, never on the path before opening it, so a file swapped between
listing and reading (another file, or a symlink to an unrelated one) is
observed as what was actually read, never recorded under the old
observation (`polylogue-lssl6`). An editor that saves through a new file with an
identical prefix is "replaced with more". Zero-filled regions left by a
crashed writer are kept as bytes and the parser counts the bad line. A file
that shrinks and regrows past its old size between observations is found by
the windows or the full check and stored as a rewrite.

### What an observation records

Every observation that found something is an insert-only `observation` row:
machine, path, device, inode, size, modification, change and birth times in
nanoseconds (birth time where the filesystem has it), mode, owner, group, the
symlink target if the path or a directory on it was a symlink (followed as
`behavior.md`, untrusted input, says), the class, the revision it produced or
confirmed and that revision's length, Polylogue's clock and version, and the
provider's version when the content states it (Claude Code `version`, Codex
`cli_version`, a database's migration table). Unchanged scans update only the
cache. Extended attributes are not recorded (provider files carry none).

Per revision: input, number, how it was produced (`first`, `imported`,
`appended`, `head_rewritten`, `rewritten`, `snapshot`; `archive-db.md`,
`revision.produced`), its chain or base, its length and the BLAKE3 of its
full bytes once complete; the observation that first saw it is the earliest
naming it. A deletion is an observation of class `deleted`; nothing else
changes. An excision that changed an input's stored bytes writes an
observation of class `excised` with the overwritten spans.

`read --view raw` and `input cat ID[@REV]` reproduce any revision's exact
bytes from its extents and show its observation history beside it.

## Provider databases

Provider databases (SQLite: Codex's state, goal, memory and thread-history
databases, Hermes `state.db`, Antigravity's stores; `sources/` lists them per provider)
are read only through SQLite, never by copying their files: a copy of the
`.sqlite` file without its `-wal` file is inconsistent and can be corrupt.

- **Taking a snapshot.** A read-only connection (`mode=ro`, never
  `immutable=1` on a live file). For a database in WAL mode, one read
  transaction reads every page through the `sqlite_dbpage` table (SQLite
  built with `SQLITE_ENABLE_DBPAGE_VTAB`: the bundled SQLite is compiled with
  that flag through `LIBSQLITE3_FLAGS`, and a test fails if the table is
  missing, since common builds such as Python's lack it), which returns each page as that
  transaction sees it, committed WAL frames included. Each page is hashed
  against the previous page map as it streams, and only the pages that differ
  are written, so no temporary copy is made: Codex's 10.5 GB thread-history
  database would otherwise cost a 10 GB write per snapshot
  (`evidence/sqlite-readonly-probe.md`; hashing all its 2.56M pages took 40 s).
  A WAL-mode read transaction never blocks the provider's writer; while it is
  open, the provider's checkpoints cannot pass it, so the provider's WAL can
  grow for those seconds. For a database in rollback-journal mode, a read
  transaction would hold a shared lock and block the writer for the whole
  read, so the backup API copies a few thousand pages per step instead and
  restarts if the database changes. `VACUUM INTO` is not used: it renumbers
  pages.
- **Exclusive locks.** Where an application holds its database under an
  exclusive lock (Chromium does) and no read transaction can start, after
  a short timeout the collector copies the database file and its journal,
  runs `PRAGMA quick_check` on the copy, and stores it only if the check
  passes; otherwise it records the failure and retries later.
- **When.** A database is snapshotted when its key, including its `-wal`
  file's size and times, has changed and it has been quiet for 5 minutes, and
  at most once an hour while it keeps changing. An unchanged database costs a
  `stat`.
- **Successive snapshots.** The first snapshot is a plain blob in the
  seekable format. A later snapshot is hashed page by page against the
  previous one's page map, and every page that differs from the base snapshot
  is stored in one **page patch** blob (page numbers plus page bytes,
  compressed), overlaid on the base: reading any snapshot needs the base plus
  one patch. A new plain base is written after a `VACUUM` (which moves every
  page) or when the patches stored since the current base add up to more than
  the source's rebase threshold, by default the base's stored size, so each
  base cycle stores at most about twice the base. The page size comes from the database header. Page maps are a
  rebuildable cache in `index.db`. Page 1 holds the header, whose file
  change counter moves with every write, so it is stored in every patch
  rather than compared (Sinex found it always differs from a backup-API
  copy; `sinex-spec/prototype/sqlite-pages/`).
- **Fidelity.** A snapshot is the database's consistent content at a moment,
  page for page, including free pages (which may hold deleted rows); not a
  copy of the transient `-wal` and `-shm` files. It records the page size, the
  schema version from the header and, where present, the application's
  migration table. Row views are the parser's job; logical dumps are not
  stored, because they lose exact pages and free-page content. (Decided by
  the spec authors from principle 5, which keeps exact bytes; the Python
  version stored logical exports, `polylogue-jbcln`, `polylogue-8t81f`. The
  operator may overrule.)
- **Readers open read-only.** Every reader opens a database read-only
  (`mode=ro`), and the database file itself is never written: after a
  snapshot its size, modification time and content are unchanged (law 117).
  `mode=ro` does not mean nothing beside the database is touched: on a WAL
  database whose `-wal` and `-shm` files are absent (the provider closed it
  cleanly), SQLite creates both and leaves them, and when they exist, a
  reader updates the `-shm` shared-memory index, so its modification time
  changes (seen on Hermes's `state.db` and Antigravity's databases;
  `evidence/gemini-hermes-real-survey.md`). SQLite and the provider treat
  both as normal. In a
  directory Polylogue cannot write, that open fails; no writer can create a
  WAL there either, so while no `-wal` exists the file alone is consistent and
  is opened with `immutable=1`, recorded on the snapshot. `immutable=1` is
  never used otherwise.
- **Never a second descriptor on a live database.** SQLite's locks are POSIX
  advisory locks owned by the process: opening and closing another file
  descriptor on a database the process has open through SQLite releases
  SQLite's locks. A database is hashed and stored only through the one
  connection that reads it (its `sqlite_dbpage` pages, or the backup API's
  copy for a rollback-journal database), never by opening the live file a
  second time.
- Diagnostic log databases (Codex `logs_2.sqlite`) are not collected;
  `sources/codex.md` says why.

## Polylogue's own SQLite files

**WAL and checkpoints.** A reader's open read transaction stops checkpoints
from passing it, so queries hold none between pages (R52). The daemon owns
checkpointing: automatic checkpoints are off on its connection
(`wal_autocheckpoint=0`); the loop runs a `PASSIVE` checkpoint as a bounded
step after each commit batch and a `RESTART` checkpoint when idle, and
`journal_size_limit` truncates the WAL to [160 MiB] after a checkpoint.
`RESTART` and `TRUNCATE` run with no busy wait: they are tried once and give
up if a reader is behind, so a checkpoint never makes the writer wait for a
reader. Every read request has a time limit, enforced with SQLite's progress
handler ([30 s] for queries and `sql`); `export` reads a session in steps,
each its own snapshot. The WAL is then bounded by the write rate times the
longest read limit. `status` shows the WAL size, warns above [256 MiB], and
names the longest current read. Checkpoints name the
schema (`PRAGMA main.wal_checkpoint`): an unqualified one also walks attached
databases and fails on one attached read-only.

A `RESTART` need not erase older-generation frames beyond the reused prefix.
After a crash these can be legitimate retained bytes. The cold-start WAL guard
(`README.md`, retained durable WAL) nevertheless refuses uncertain generation
or partial tails before SQLite opens them. Checkpoint scheduling does not prove
that an acknowledged commit survived, and this intermediate guard may require
controlled recovery after a legitimate crash. Active writer-owned reads use the
existing archive authority; they do not apply a cold scan to an append in flight.

**Values too large for SQLite.** A single string or blob over SQLite's maximum
length (1,000,000,000 bytes by default) is never truncated: it stays in the
stored input. The block records the value's complete byte locator
(`input_pk`, `source_chain`, `source_start`, `source_end`; `index-db.md`, block
byte sources), including when the value came from another input than the
preferred message. A read uses those retained chain extents, never the
contributor's current chain. An absent or excised range is reported as such,
without substituting new bytes. The value is not searchable, and `status`
counts it as too large to index.

**A blob found missing.** When a read finds a blob missing and the input's
location still holds the same bytes (same observation key and hash), the next
collect step stores them again; otherwise the blob stays missing and `status`
names it. This is the loop comparing states, not a repair command.

## Crash safety without repair

Every file Polylogue writes (blob files, spool files, exports) goes to a
temporary name, is synced and renamed into place, and then every directory
created on the way is synced, not only the last one.

Every blob row has a `state`. Storing a file blob: (1) insert the `blob` row
as `writing`, alone; (2) write, sync and rename the file; (3) in one
transaction set `present` and insert the extents and revision rows that refer
to it. An inline blob is one transaction. Deleting runs the other way:
references removed and the row set to `deleting` in one transaction, then the
file, then the row. After a crash,
leftover rows are part of the loop's ordinary comparison of states
(`convergence.md`):

| Found | Done |
|---|---|
| `writing`, file present with the right hash | set `present` if a pending piece or import still needs it, otherwise delete file and row |
| `writing`, the file missing or wrong | delete the file if present, then the row |
| `deleting` | delete the file if present, then the row |
| files in `blobs/tmp/` | deleted at daemon start |

A file is only ever written after its row exists, so there is no directory
scan, garbage collection pass or fsck. "Is this blob still used" is a `NOT
EXISTS` on `extent` and on delta bases (`blob.base`). A `writing` row younger
than the daemon's last round is left alone (an import may still be writing it).

**Safety checks fail closed.** If "is this blob still referenced" or "is this
input excised" cannot be answered (a read error, a missing table), nothing is
deleted or stored, and the error is shown.

A blob is deleted only when no extent refers to it and no delta uses it as its
base: after joining (its bytes are in the joined blob) or by excision.

Evidence: measurements in `evidence/blobstore-v2.md` and
`evidence/ideas-storage.md` (4 MiB frames cost 3% of compression ratio, 1 MiB
13%; Codex rewrites store as deltas at 0.94% of plain zstd at the median, 6.7% at most, 2.2% overall over 1,431 real pairs, `evidence/codex-rewrite-deltas.md`; two snapshots of
`state_5.sqlite` 41 s apart differed in 7 of 53,122 pages); `polylogue-ps055`
(Claude Code's mutable first record, 194 files); `polylogue-p6tgn`,
`polylogue-ryp2j` (second descriptors); `polylogue-b8of0`, `polylogue-fhikb`
(directory sync); `polylogue-9jelj` (fail closed); audit P22-A11, P27-A4,
P31-A1, P32-K2.

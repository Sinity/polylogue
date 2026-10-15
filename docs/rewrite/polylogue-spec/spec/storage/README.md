# Storage

Three SQLite files and a blob directory under `$XDG_DATA_HOME/polylogue/`
(normally `~/.local/share/polylogue/`), or the directory `--archive DIR`
names (`behavior.md`, the archive location). Why three and not six: `decisions/storage-files.md`. On the
operator's machine the root disk is wear-limited and `/realm` holds heavy
data, so the Sinnix module sets the archive under `/realm/state/polylogue/`
(the blob store and a first build write tens of GB); the hook spool stays
in `$XDG_DATA_HOME`, where its synced appends are small
(`sources/README.md`, durability against the time budget).

| Path | Holds | If deleted | Details |
|---|---|---|---|
| `blobs/` | the bytes of large inputs and attachments, by BLAKE3 | **loss**: back it up | `blobs.md` |
| `archive.db` | what cannot be recomputed: inputs (small ones inline) and where they were found, notes, kept results, excisions, settings | **loss**: back it up | `archive-db.md` |
| `index.db` | everything parsed and derived: sessions, messages, blocks, tool calls, search, facts, daemon progress | rebuilt from `archive.db` and `blobs/` | `index-db.md` |
| `embeddings.db` | purchased embedding vectors, keyed by text hash | vectors must be bought again: back it up | `embeddings.md` |

## Backup

Backup is the filesystem's job (btrfs snapshots, borg, whatever the operator
uses); Polylogue has no backup command and keeps the files consistent at
every commit. A backup takes `archive.db`, `blobs/` and `embeddings.db`
together. Only an atomic snapshot of the whole directory is fully
consistent. A two-step copy should take `archive.db` first and `blobs/`
after it: no extent refers to a blob before its file is complete, so the copy misses
only blobs that joining, re-encoding or excision deleted during the copy. A
restored archive with a missing blob fails loudly: reading names the blob,
`status` counts it, and the loop stores it again when its location still
holds the bytes (`blobs.md`, a blob found missing). The directory is 0700 and every file in it 0600 (`behavior.md`,
privacy).

## Access

- All three files use WAL mode and `secure_delete`.
- **On btrfs, the archive directory is not copy-on-write.** SQLite rewrites
  pages in place, which copy-on-write turns into a new extent per write: the
  benchmark's 9.15 GB `index.db` ended in 283,712 extents, and reading its
  tool-output index cold ran at 29 MB/s against 71 MB/s for the same file
  written without copy-on-write (758 extents), with a cold search taking
  463 ms against 271 ms (`evidence/ingest-benchmark.md`). So Polylogue sets
  the No_COW attribute (`FS_NOCOW_FL`) on the archive directory when it
  creates it, before any file exists, and every file inherits it; on other
  filesystems nothing is done. Copy-on-write would only add btrfs checksums
  and compression, and neither is needed: blobs are already zstd-compressed,
  written once, and checked against their BLAKE3 on every full read (frame
  checksums cover partial reads); `archive.db` gets its own page checksums
  from SQLite's checksum VFS (`cksumvfs`, 8 bytes per page, verified on every
  read, a mismatch reported as corruption naming the page), and so does
  `embeddings.db`, whose vectors were paid for and are restored from backup,
  never recomputed for free; only `index.db` is rebuilt, never restored. Snapshots still work: after
  one, the next write to each page is copied once.
  `cksumvfs` is not in SQLite's amalgamation, which is what the bundled
  `libsqlite3-sys` compiles: it is SQLite's `ext/misc/cksumvfs.c`, public
  domain, vendored into the `store` crate at the bundled SQLite's version,
  compiled by its `build.rs` and registered with `sqlite3_register_cksumvfs`
  before any connection opens. A new `archive.db` or `embeddings.db` gets the
  8 reserve bytes per page when it is created (`SQLITE_FCNTL_RESERVE_BYTES`
  before the first write; existing files cannot gain them in place, and
  there are none before the first release). A test opens a file, flips one
  byte of a page, and expects the read of that page to fail naming it (law
  123); `index.db` uses no checksum VFS: damage SQLite reports on a read
  ("database disk image is malformed" and the like) makes the daemon
  rebuild it, and a flip SQLite does not notice goes undetected, which is
  accepted because the file is rebuilt from the other two at any version
  change.
- **Retained durable WAL is checked before cold recovery.** The checksum VFS
  above protects main-database page reads; it does not wrap WAL reads. SQLite's
  own WAL recovery stops at an invalid frame and may successfully expose an
  earlier commit, including after damage to an acknowledged transaction. Native
  WAL checksums alone do not establish that an acknowledged history survived.
  Before opening any SQLite connection on a cold archive, acquire its existing
  archive coordination lock without rewriting it. A missing lock on retained
  state requires controlled recovery; busy means unavailable, never an empty
  archive. A fresh empty archive may create its lock. Validate `archive.db-wal`
  and `embeddings.db-wal` while quiescent, then open SQLite and inspect schemas
  before any database mutation. No endpoint serves requests while this check is
  pending. An active writer creates read handles through its already-owned
  authority; those handles do not scan a concurrently appended WAL. Cold
  standalone read handles acquire the same noncreating exclusive authority and
  retain it for their lifetime.

  Validation reads the pinned SQLite WAL format, header checksum, frame salts,
  positive page numbers and rolling frame checksums with one page buffer. Total
  IO is proportional to the retained WAL length, not constant. Cancellation is
  checked before and between page reads; cancellation is an explicit non-success
  and does not change retained files. Complete validated uncommitted frames are
  allowed: SQLite may ignore them. An invalid header, bad current-generation
  frame, incomplete frame, orphan WAL, or uncertain generation tail returns
  **recovery required**, naming the file and reason (and frame when known),
  before SQLite can recover, checkpoint or discard bytes. A damaged frame's
  page number is not trusted enough to invent a named-page corruption error.
  Nothing truncates, deletes, repairs, or automatically adopts an earlier
  committed prefix on this refusal.

  This is an intermediate integrity-first availability tradeoff, not complete
  crash recovery. A legitimate unfinished write can leave a partial frame; a
  `RESTART` can leave old-salt frames after the current generation. Both may
  require controlled recovery and block startup. This explicitly qualifies the
  ordinary logical-step restart behavior (`behavior.md`, failures); it is not a
  claim that every crash resumes automatically. Nor does the guard prove all
  history survived: aligned truncation to an earlier valid commit, replacement
  by an older internally consistent main/WAL pair, salt/header/history changes
  that leave a valid prefix, and checksum collisions cannot be ruled out from
  these files alone without an independent expected durable head. There is no
  external commit ledger by default. Preserve the originals for a separately
  controlled recovery decision; do not label ambiguity as proven media
  corruption or silently weaken this boundary to improve startup availability.
- Only the daemon opens the database files and is their only writer
  (`decisions/process.md`). The daemon's read connections open `index.db`
  read-only with `archive.db` attached (`ATTACH … AS archive`), so a query
  that filters by notes (`tag:`, `hidden`, `outcome:`) or excisions is one
  SQL statement.
- **One view per read.** SQLite takes a snapshot per file. The daemon
  serializes the short acquisition of a read's view with its existing writer
  commits, index swaps and publication of an immutable in-memory vector set.
  During that acquisition it begins the read transaction, reads `main.meta`
  and `archive.meta` to establish both database snapshots, and retains the
  vector-set reference. No commit, swap or vector publication can intervene
  between those operations. It then releases serialization before executing
  the query; the read retains its snapshots and vector reference, never the
  writer's turn, for its lifetime. Every acquired view is a state the daemon
  actually held. Committing `archive.db` before `index.db` alone does not
  provide this guarantee. A failed acquisition closes its partial read;
  an index swap retries acquisition or returns an explicit unavailable result
  within the request's deadline (`query.md`, results).
  The read's **view** remains (index generation, index counter,
  `archive.db` write counter, the highest `embedded.id`); caches are keyed by
  it (`behavior.md`, warm state).
  Acceptance forces this interleaving: a reader pins index counter 0, then
  the writer commits archive counter 1, index counter 1, and archive counter
  2 before the reader tries to pin the archive. Independent acquisition can
  return (0, 2), although the actual states were (0, 0), (0, 1), (1, 1),
  (1, 2). The production read must return an actual state or refuse/retry;
  the same check includes index swap and vector publication. A SQLite witness
  of the invalid pair establishes the need for this contract, not that the
  daemon has implemented it.
- **One writer** (rule R35). The daemon takes an exclusive `flock` on a lock
  file in the archive directory, opened through the directory's resolved
  path, so a second daemon reaching the same archive through a symlink or
  another spelling of the path finds the lock held and exits naming the
  holder. Write access is also a type only the daemon obtains
  (`convergence.md`, Parse). At start the daemon takes the lock before
  touching the socket path; only the lock holder removes a leftover socket
  and binds, and at exit it removes the socket only if the path still has the
  inode it bound. The socket path must fit a Unix socket address (108 bytes
  on Linux), counted in encoded bytes, not characters: a runtime directory
  with multibyte characters that would overflow it makes the daemon refuse
  to start, naming the path and its length in bytes (`polylogue-bwuz0`).
- **Local filesystem only.** The archive directory is on a local filesystem
  and is never synced live between machines; archives combine through Sinex
  or through export and import.
- All access to `archive.db` and the blob store goes through one module,
  `store::inputs` (with `store::archive_meta` for the archive's own rows), the only place another storage backend could be added
  (`decisions/sinex.md`).

## Retention

Inputs, revisions, notes and vectors are kept forever (`principles.md` #5).
Every other store states its bound where it is defined: spool files are
deleted only by the daemon, under the conditions in `sources/README.md`;
`blobs/tmp/` is emptied at daemon start; the pre-migration copy of
`archive.db` is deleted once the migrated file has opened; the old
`index.db` is deleted when a rebuilt one is swapped in (`index-db.md`).

## Conventions for all three files

- **Units.** Times are INTEGER Unix milliseconds, UTC. Durations are
  milliseconds. Money is INTEGER nano-USD. Nothing is stored as a float.
- **Column names** match the model and export field names (`model.md`,
  `export-format.md`), with suffixes the export drops: `_json` for a column
  holding JSON (`attrs_json` is the export's `attrs`), `_pk` for an integer
  key local to `index.db` (`after_message_pk` is exported as the message ID
  `after_message`), `_tokens` and `_nusd` for token and money columns. Other
  differences (`blob_hash`, `file_name`, `other_type`, `source_start` and
  `source_end`) name their export field in the column comment, and
  `provider_cost_nusd` is exported as `provider_cost_usd` (decimal USD).
- **No `CHECK` lists of allowed values.** Allowed values are Rust enums,
  checked by the `store` crate on every write. Values read back keep unknown
  variants as they are, so an older binary never fails on a newer value; a
  `CHECK` would turn that value into an error for it. (Decided by the spec
  authors from one checking place, the `store` crate, and old binaries
  reading new archives; the operator may overrule.)
- **No variable-length `IN (…)` lists.** A set of IDs is passed as one
  parameter through `json_each(?)` or a temporary table, so no query depends
  on SQLite's limit on parameters (`plan/testing/laws.md` has the law).

Evidence: `polylogue-mdtrf` (the Python backup copied rows without their
blobs); `polylogue-1oa7o` and `evidence/beads/closed-5.md`, class J (one
writer); `polylogue-l5bdr`, narrowed by the 2026-10-11 retention ruling;
`polylogue-o21` (generated `CHECK` constraints, deliberately not kept);
`polylogue-1tyq5.187` (excision failed past the parameter limit).

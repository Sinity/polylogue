# polylogue-store

Every SQLite connection and every write in Polylogue goes through this crate.
Other crates get typed handles and typed operations; raw connections and SQL
never leave it (compile-fail doctests in `lib.rs` check this).

## An archive directory

| File | Contents | Rebuildable |
|---|---|---|
| `archive.db` | inputs, blobs, notes, client requests (`schema/archive.sql`, `schema/observed-context.sql`) | no |
| `blobs/` | content-addressed bytes, managed by [`hashed-blobs`](../hashed-blobs/README.md) | no |
| `index.db` | parsed sessions, messages, blocks, tool calls, parser state (`schema/index.sql`) | yes, from the two above |
| `embeddings.db` | text embeddings; the schema exists, nothing writes it yet (`schema/embeddings.sql`) | no |
| `writer.lock` | the exclusive lock file | |

Versions: archive format 4, index 20, embeddings 0 (`ARCHIVE_VERSION`,
`INDEX_VERSION`, `EMBEDDINGS_VERSION` in `lib.rs`). Directories are created
`0700`, files `0600`, and a fresh archive directory gets No_COW on btrfs.

## Handles

- `Writer::open(dir)` takes the exclusive lock on `writer.lock` and holds it
  until dropped. Only the daemon opens a writer. Read handles made from it
  (`writer.read_handle()`) share the lock and see the current state.
- `ReadHandle::open(dir)` opens a stopped archive read-only. It fails with
  `WriterHeld` while a writer runs.
- `ReadHandle::fork` gives another handle on the same lock, for a separate
  request or thread.
- `BlobCapture` (from `writer.blob_capture()`) hashes and stores bytes on a
  worker thread without holding a SQLite connection; the writer then records
  the result in one short transaction.

## Startup checks

Before any database is opened for writing, `Writer::open` and
`ReadHandle::open`:

1. check every WAL file's header, frame salts and checksums without letting
   SQLite touch it. A damaged, partial or orphaned WAL returns
   `RecoveryRequired` and leaves all files unchanged. This check can also
   refuse some harmless torn tails; those need `recovery::recover_copy` too.
2. check schema versions. A newer archive returns `NewerSchema`. An index of a
   different version returns `IndexRebuildRequired`; older index versions
   (18, 19) are refused because their IDs were computed differently, and there
   is no index rebuild yet.
3. upgrade archive formats 1 to 3 to format 4 (writer only). The upgrade
   first copies every database file into `migrations/v<N>-to-v4-*/` with a
   manifest, then adds the `observed_context` table.
4. compare each database's schema text with the expected SQL and refuse any
   difference.

SQLite is the bundled 3.53.2, checked at build time and startup. `archive.db`
and `embeddings.db` use SQLite's checksum VFS (`vendor/cksumvfs.c`): a page
that fails its checksum returns `CorruptPage` with the file, page and offset.
`index.db` uses the plain Unix VFS.

## Recovery and copies

- `recovery::recover_copy(source, destination)` makes a new, verified copy of
  a stopped archive whose WAL ends in a torn previous-generation tail. It
  never modifies the source. The copy carries a permanent `RecoveryRecord`
  (archive `meta` key `recovery_qualification`), and every read reports it,
  because the last committed state before the damage is unknown.
- `copy_preparation::Preparation` prepares a stopped archive to become a
  restored copy: it checks every stored blob and revision, then records
  completion in `client_request`. A pending preparation blocks the writer.
  A preparation left pending in a format-3 archive also blocks the upgrade to
  format 4; finish it with a format-3 build (commit `5ddb6975dafa`) first.

## What is stored, by module

- **Inputs** (`imports`, `observed`, `capture`, `claude_bodies`,
  `history_pages`, `codex_history`): every imported source file is stored
  once, immutably, with its byte digest and format context. A retried import
  with the same client request replays the stored result. `observed` reads
  stored source bytes by range and verifies them.
- **Discovery** (`configured_claude`, `configured_codex`): walk the
  configured Claude projects directory and Codex home in resumable turns and
  hand each found file to acquisition.
- **Parser state** (`parser_state`): disposable per-input parser checkpoints in
  `index.db`. Parsing reads a pinned snapshot; its output and checkpoint
  commit in one transaction. Large outputs are staged in batches and applied
  in steps that survive a restart.
- **Values** (`values`): normalized text and JSON payloads, appended in chunks
  of at most 64 KiB and sealed with a BLAKE3 hash computed outside write
  transactions.
- **Messages** (`contributions`): each source contributes a candidate body
  for a message, built field by field and block by block in a radix-16
  directory of immutable nodes, so a change copies one path rather than the
  whole message. One candidate is selected; readers pin one SQLite snapshot
  and see one complete body. Abandoned builds and old bodies are removed in
  small steps.
- **Tool items** (`tool_items`): session-owned tool operations and their
  versions, linked to message blocks.
- **Notes** (`authored`): attributed notes, corrections and withdrawals, kept
  in `archive.db` so they survive an index rebuild.
- **Search** (`prose_search`): a per-query, in-memory FTS5 index over
  selected message bodies. There is no persistent search index yet.

## Not built yet

Persistent search, index rebuild and migration from older index versions,
excision, export, and recovery from general corruption.

## Tests

`cargo test -p polylogue-store`. Tests use synthetic data in temporary
directories, including killed child processes for crash cases, corrupted
pages and WAL tails, and lock contention.

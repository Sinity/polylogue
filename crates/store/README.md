# Foundation store

The daemon acquires `Writer::open` and holds it until shutdown. An exclusive
file lock at the canonical archive path rejects competing writers, including
symlink aliases. Callers receive typed read handles and status, never SQLite
connections. `put_file_blob` is a foundation byte-storage diagnostic, not provider
ingestion, revisions, parsing or M1 support.

All three schemas are exact SQL-fence extractions from the owning copied
specification under `docs/rewrite/polylogue-spec/spec/storage/`; a drift test
compares the extracted text. Every table exists initially. Archive and
embeddings migration versions are zero because there are no pre-release
migration steps. The index has version two, adding required parser incarnation metadata. Startup checks every existing
schema/version before target database mutations. A newer durable file is
refused unchanged; an index mismatch returns `IndexRebuildRequired`, preserving
the old index. A missing or empty main file with retained nonempty WAL or
shared-memory bytes requires recovery and is never initialized as fresh.
An orphan nonempty WAL without its shared-memory file requires
explicit recovery before non-mutating inspection; no WAL is silently ignored.
An existing live WAL is inspected with its real SQLite view; SQLite may update
coordination read marks in its existing shared-memory file. Durable database
and WAL bytes are not changed by inspection. Rebuild and migration machinery remain later work.

Bundled SQLite 3.53.2 is checked at build time and runtime. The unchanged
matching upstream checksum VFS is statically linked and registered before
store connections open. Durable files reserve eight bytes before creation;
index uses the underlying Unix VFS. Page-corruption tests require
`SQLITE_IOERR_DATA` and identify the database path. Upstream logs the corrupt
offset; propagating an exact page number in the public error remains open.

Fresh archive directories get No_COW on btrfs before any children exist.
Directories use 0700 and owned files 0600. Tests cover real writes, read-only
mutation denial, byte-range corruption, schema refusal without sibling
creation or chmod, alias locks and privacy. Compile-fail examples verify that
read handles have no writer method and connections remain private. These tests
do not qualify cross-file product snapshots, ingestion, excision or index
rebuilds.


Disposable parser state uses typed Writer admission and read-only pinned
snapshots, with no connection or SQL escape. State cells, supplied output and
checkpoint share the final step transaction; the store primitive does not infer
canonical completion from EOF. A per-index monotonic allocator survives head
and input-key removal. Snapshot, stale-result, bounded retirement and actual
child-process crash tests qualify this component, including the real archive
writer lock. Provider ingestion, canonical message finalization and public
client stepping are still unregistered.

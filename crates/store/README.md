# Foundation store

The daemon acquires `Writer::open` and holds it until shutdown. The canonical
archive's exclusive file lock rejects competing writers and cold readers,
including aliases. Writer-created read handles retain that authority until
their connections close; they use the active SQLite view without scanning a WAL
being appended. Cold read handles acquire the existing lock without creating or
rewriting it. Callers receive typed handles, never raw SQLite connections.
`put_file_blob` is a foundation byte-storage diagnostic, not provider ingestion,
revisions, parsing or M1 support.

All three schemas are exact SQL-fence extractions from the owning snapshot;
a drift test compares their text. Archive/embeddings versions are zero because
there are no pre-release migrations; index version is three, adding provider preparation. Existing schemas
and versions are checked before database mutations. A newer durable file is
refused unchanged; an index mismatch returns `IndexRebuildRequired`, preserving
it. Retained state without its coordination lock requires controlled recovery;
raw headers may explain a newer-version refusal without opening SQLite.

Before any cold SQLite open, the lock holder checks durable WAL headers,
current frame salts, page numbers and rolling checksums. Memory is bounded by
one page (at most 64 KiB); total IO scales with WAL length. Cancellation is
checked before and between frames. Valid complete uncommitted frames are
allowed. A bad header/frame, partial frame, orphan WAL or uncertain generation
tail returns `RecoveryRequired` before SQLite can discard or checkpoint bytes.
Refusal and scan cancellation preserve retained files. After successful
validation, SQLite may update SHM coordination state during inspection.

This intermediate guard may refuse legitimate torn uncommitted tails and stale
frames from `RESTART` reuse. It needs controlled recovery and does not qualify
full startup availability. Aligned truncation to an earlier valid commit,
older consistent main/WAL pairs, valid-prefix salt/header/history changes and
checksum collisions cannot be excluded without an independent expected head.
There is no external commit ledger. Runtime live-WAL corruption and complete
recovery remain unqualified.

Bundled SQLite 3.53.2 is checked at build time and runtime. Its matching
checksum VFS has one documented local fault callback. Main-database checksum
failures retain `SQLITE_IOERR_DATA` and report the VFS's actual file, page,
offset and page size through operation-scoped thread-local evidence. The VFS
does not wrap WAL reads; the cold guard above is a separate boundary. Index
uses the underlying Unix VFS and is rebuildable.

Fresh archive directories get No_COW on btrfs before children exist. Directories
use 0700 and owned files 0600. Synthetic tests cover actual writes, read-only
mutation denial, range/page corruption, killed-process retained-WAL faults,
legitimate/ambiguous tails, cancellation, lock lifetimes, schema refusal and
privacy. Compile-fail examples deny writer methods and raw connections. These
checks do not qualify coupled cross-file views, ingestion, excision, index
rebuild/swap, full recovery or M0 completion.

Disposable parser state uses typed Writer admission and read-only pinned
snapshots, with no connection or SQL escape. Small state deltas, supplied output and
checkpoint share the final step transaction; the store primitive does not infer
canonical completion from EOF. A per-index monotonic allocator survives head
and input-key removal. Snapshot, stale-result, bounded retirement and actual
child-process crash tests qualify this component, including the real archive
writer lock. Provider ingestion, canonical message finalization and public
client stepping are still unregistered.

Larger provider deltas can be staged in admitted encoded-cell batches, sealed,
and applied in bounded cell/byte steps. New parser snapshots wait while this
work exists; already pinned readers retain the old SQLite view. Interrupted
unsealed preparation can be discarded in bounded steps. Sealed application
resumes after process exit and publishes its head/counts/checkpoint only after
all staged members have been consumed. This protocol changes no canonical
Message, block or search rows. The eventual canonical writer must couple its
message-local selection to final publication. Whole-cell encoding, decoding and
SQLite page work remain explicitly unqualified for arbitrarily large cells.


The forward index9 SQL is extracted from the pinned owning specification revision
`INDEX_SPEC_COMMIT` and checked against `INDEX_SQL_BLAKE3`, copied Markdown, and
actual SQLite application. Its Message head/body/directory schema replaces the
old mutable Message columns. The inbox index8 executable comparison remains
separate and cannot qualify this forward representation. These schema checks do
not establish canonical publication, public-client restart or giant-value/search
resource acceptance.

Canonical normalized payloads now use `values`: at most64KiB per immutable
append/range read, UTF8 validation and bounded ordinary-BLAKE3 proof steps outside
write transactions. Process restart keeps appended bytes and restarts hashing at
zero; sealing performs fixed metadata/digest work. JSON typed encoding remains
its producer contract. No complete payload is hidden in a final readiness row.

Message directory primitives copy a fixed radix16 path and retain unchanged
children through counted references. Collection drains one at-most16-edge node
per transaction, with pinned readers retaining their historical WAL snapshot.
These primitives are still being connected to canonical Message activation and
public reads; they do not certify those unfinished runtime operations.

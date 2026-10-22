# Excision

A text excision scans every retained revision chain, including earlier rewrites and bytes split across stored pieces. Plain UTF-8 inputs are matched as text. In JSON inputs, matching uses decoded string values, including mixed Unicode escapes, surrogate pairs and JSON payloads embedded in string values. A matching JSON object key refuses the operation: the store cannot decide whether it is structural syntax or arbitrary payload. Keys and identities remain unchanged. Matching preserves byte lengths, line breaks, revision IDs and offsets; it does not normalize or reserialize other records.

When a blob supplies slices to several inputs, each referring extent is copied with only its own chain's matches overwritten. An unrelated input's shared fragments remain exact. Preview and execution use the same retained-chain scan. Index text bodies are inspected one row at a time before the affected inputs are parsed again from their stored bytes.

Text excision also overwrites matching authored payloads: note keys, values and checks; variant bodies; kept-result names, queries and result values; delivered text and event details; and excision reasons. Row IDs, revision families, targets and relations remain intact. Counts distinguish redacted notes from removed notes.

Retained ZIP containers, SQLite page images, PDF and gzip streams, non-UTF-8 representations, non-plain blob encodings, and uninterpreted export companions require format-specific decoding and rewriting. Encountering one, or a matching JSON key, returns `unsupported_excision`, naming the input and format, before committing changes. A corrupt blob remains a storage failure, not a format refusal. Imported plain archive members and captured logical JSON database rows are different from their original containers/page images.

Provider files remain untouched. `on_disk` identifies locations whose stored history held a match; scrub those original files separately before a future full read could collect their text again. Earlier external backups and filesystem snapshots are outside this operation. Successful synthetic checks of this path do not qualify interrupted cross-database commits or the full session/primary-mode purge protocol.

A scan keeps one verified copy of each unique plain blob in an unlinked, mode-0600 temporary file under the process's temporary directory. File blobs use the shared sequential reader: one whole-name verification pass and one forward copy, including oversized historical frames. Inline blobs use incremental SQLite BLOB I/O and a streaming decoder. Logical extents and the rewrite reuse those verified copies, so shared or reordered slices do not trigger another decode. Memory holds fixed transfer buffers, decoder state, match ranges and blob/extent metadata rather than complete input bodies. Temporary disk use can reach the total decoded size of the unique blobs visited; it is released when the attempt ends. This is a decode-work guarantee, not a general latency or peak-RSS claim. The excision API currently has no caller cancellation token; errors from the logical visitor stop that visit.

## Sessions in shared exports

Removing a session from a retained JSON export erases its complete records and the necessary array separators. Other bytes, line breaks and offsets stay unchanged, so an index rebuilt from the retained export can still read the surviving sessions. Preview performs the same boundary and overlap checks as execution.

This path requires complete, independently decodable JSON pieces. Citations that cut through scalar values, object-member removals, gaps or split records, and bytes shared with an input outside the rewrite return `unsupported_excision`. Removal also refuses overlaps with surviving sessions' message or event citations. A refusal leaves stored bytes, indexed records and authored notes unchanged.

Session-record planning currently reads one complete blob at a time. The streaming text-scan memory description above does not apply to this path. Interrupted cross-database commits and physical retirement remain separate qualification work.

## Interrupted commits

The archive tracks a numbered, uniquely tagged excision revision in its metadata; SQLite triggers change it whenever an excision row changes. The index acknowledges that revision in its own transaction. A reader reserves a slot without pinning either database; its first database access checks both revisions inside the transaction and returns `ExcisionPending` on mismatch. Reads that already established a snapshot keep their consistent earlier view. A commit-phase failure stops the writer and returns `ExcisionInterrupted`; it cannot accept more writes until reopening reconciles the committed state. A refused reader cannot serve later queries and discards its connection when released.

On reopening the archive, an index behind the archive's excision revision is replaced and rebuilt from canonical bytes under the standing excision rules. Before any rebuild, including a schema change or interrupted first build, the archive must still contain each previously applied removal rule unchanged. The index stores a BLAKE3 hash of its ID and a BLAKE3 hash of scope, refused hashes/locations and message IDs; reason text and local progress are excluded. A higher counter alone does not prove inclusion. Missing or changed rules, an older archive revision, or different tags at the same revision refuse recovery for reconciliation. Pre-revision indexes can adopt the current archive's rules during rebuild; they cannot establish unknown earlier removal history. Restore checks stable policy inclusion inside its target transaction, before committing. A rejected view preserves the old policy rows, revision, checkpoint and unrelated selected rows. Successful installation then prepares the index. Reinstalling identical excision rows preserves the revision, so an unrelated restore update keeps a correct index. This local archive-format migration makes older writers refuse the archive.

New excisions record exact removed text hashes before committing. Startup repeats pending vector and provider-failure deletions before exposing reads. Each vector mutation holds the embeddings writer lock while checking its reader's excision revision, so held provider responses cannot write across an excision. Unrelated purchased vectors survive. Existing index rules are checked before cleanup; conflicting history refuses without deleting paid data.

The canonical excision transaction uses at least SQLite FULL synchronization before old source files can be removed. Ordinary ingestion keeps its prior synchronization mode, restored after success or refusal; previews leave it unchanged.

Local blob-file deletions are recorded in the same canonical transaction. Startup and normal completion replay that exact intent in groups of up to 500 hashes, using only the primary archive store. Missing-file retries sync the nearest existing local parent before clearing intent; other errors retain intent and stop the writer. Cataloged blobs, retained delta bases and current replacement/source references refuse deletion. A retained delta that needs rewriting refuses excision before commit. Fresh reads remain unavailable while either cleanup queue is pending.

Startup and successful excision remove recognized obsolete local indexes:
`index.db.v<N>`, `index.db.unsynced` and `index.db.excision-<UUID>`, including
their WAL and shared-memory files. All copies' applied removal rules must be
included in the canonical archive before any is deleted or purchased-vector
cleanup begins. SQLite checkpointing preserves those rules in each main file
before sidecar deletion; directory syncs make unlink retries durable. A busy
reader, malformed database, orphan sidecar or nonregular file refuses cleanup.
Conflicting history is preserved for reconciliation. Previews preserve the
copies. Unknown filenames, original provider files and external snapshots are
outside this cleanup.

Historical excisions without exact cleanup hashes, separately restored old
embedding files and unrelated orphan files remain separate work. Process-interruption recovery does not establish secure disk erasure or full power-loss recovery.

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

The archive tracks a numbered, uniquely tagged excision revision in its metadata; SQLite triggers change it whenever an excision row changes. The index acknowledges that revision in its own transaction. Each new reader checks both revisions inside its existing snapshot and returns `ExcisionPending` on mismatch. Already admitted reads keep their consistent earlier view. A commit-phase failure stops the writer and returns `ExcisionInterrupted`; it cannot accept more writes until reopening reconciles the committed state. A failed check releases its connection rather than returning an unusable one to the pool.

On reopening the archive, an index behind the archive's excision revision is retained as `index.db.excision-<id>` and rebuilt from canonical bytes under the standing excision rules. An older archive revision, or different tags at the same revision, refuses recovery for reconciliation. Successful restore installation uses the same preparation. Reinstalling identical excision rows preserves the revision, so an unrelated restore update keeps a correct index. This local archive-format migration makes older writers refuse the archive.

Recovery preserves authored data, retained inputs and embedding data. Stale index copies can contain removed text; their physical retirement, interrupted embedding cleanup and orphan-file cleanup remain separate work. This mechanism qualifies derived-index validity, not completion of every physical deletion.

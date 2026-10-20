# Text excision

A text excision scans every retained revision chain, including earlier rewrites and bytes split across stored pieces. Plain UTF-8 inputs are matched as text. In JSON inputs, matching uses decoded string values, including mixed Unicode escapes, surrogate pairs and JSON payloads embedded in string values. JSON keys stay unchanged. Matching preserves byte lengths, line breaks, revision IDs and offsets; it does not normalize or reserialize other records.

When a blob supplies slices to several inputs, each referring extent is copied with only its own chain's matches overwritten. An unrelated input's shared fragments remain exact. Preview and execution use the same retained-chain scan. Index text bodies are inspected one row at a time before the affected inputs are parsed again from their stored bytes.

Text excision also overwrites matching authored payloads: note keys, values and checks; variant bodies; kept-result names, queries and result values; delivered text and event details; and excision reasons. Row IDs, revision families, targets and relations remain intact. Counts distinguish redacted notes from removed notes.

Retained ZIP containers, SQLite page images, PDF and gzip streams, non-UTF-8 representations, non-plain blob encodings, and uninterpreted export companions require format-specific decoding and rewriting. Encountering one returns `unsupported_excision`, naming the input and format, before committing changes. A corrupt blob remains a storage failure, not a format refusal. Imported plain archive members and captured logical JSON database rows are different from their original containers/page images.

Provider files remain untouched. `on_disk` identifies locations whose stored history held a match; scrub those original files separately before a future full read could collect their text again. Earlier external backups and filesystem snapshots are outside this operation. Successful synthetic checks of this path do not qualify interrupted cross-database commits or the full session/primary-mode purge protocol.

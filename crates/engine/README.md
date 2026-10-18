# polylogue-engine

Parses provider files outside the writer and hands the results to the store.
Parsing and encoding run on worker threads against pinned store snapshots; the
writer only applies finished results.

## Modules

- `owner`: the single writer actor. It accepts Codex rollout imports, Claude
  Code imports with a declared context, and configured locations. Capture,
  parsing and Message preparation run on one capture worker and one compute
  worker; the actor applies their results between peer requests. Inputs are
  processed one at a time. On restart the actor lists stored inputs and
  resumes or discards prepared state before parsing again.
- `source_turn`: parses one window of raw bytes with the provider cores from
  `polylogue-sources`. It reads typed parser state only: no file IO and no SQL
  writes. The framing continuation is released only when the store's
  publication receipt matches (`CommitGuard`).
- `state::{claude,codex}`: provider state codecs over a pinned parser-state
  snapshot. Reads are charged against cell and encoded-byte budgets. Claude
  body references come from the same snapshot, never from the filesystem.
- `converge`: `PendingInstall` writes one parsed delta in transactions of at
  most 64 cells and 2 MiB (stage, seal, apply, publish). After a crash,
  `recover_turn` discards an unsealed delta so the input is parsed again, or
  finishes applying a sealed one. Installing parser state never marks an input
  done; reaching end of file does not either.
- `canonical_message`: builds Codex and Claude Messages from parsed state. The
  worker computes one action (a value, field, block, cursor or source
  certificate) and the actor applies it in one transaction. Before a field
  origin is attached, its selector is evaluated against the stored source
  bytes and the normalized digest and length must match. Unsupported mappings
  stop with a pending diagnostic.
- `version_digest`: hashes the Codex version evidence of one Message in slices,
  outside writer transactions. Missing evidence, count mismatches and stale
  revisions yield no digest. Progress is not stored; a restart hashes from zero.
- `source_rows`: checks that an input's parsed rows are complete (step results,
  effects and the selected source certificates). The result is recorded in
  parser state and survives restart; an unfinished check restarts from the
  beginning.
- `session_status`: for one Session read, walks the stored-input catalog to
  find the contributing inputs and reports whether they are processed:
  complete, pending, unavailable or unsupported. Membership comes from the
  provider sources, not from matching Message rows. Record error counts use
  unique record starts, so replays cannot double-count.
- `session_query`: typed Session listing and prose search over the same pinned
  catalog. Prose search builds a per-query FTS index from selected human and
  model text blocks; every candidate is visited before sorting and limits.
- `authored`: hashes the current text of an authored-note target.
- `codex_discovery`: Session references from Codex prompt-history snapshots.
- `claude_bodies`, `claude_capture`: tool-result files that a configured Claude
  main file names explicitly. No directory scan.

Implementation notes for the batching rules are in
[provider installation](../../docs/rewrite/provider-installation.md) and
[Message preparation](../../docs/rewrite/message-preparation.md).

## Limits

Budgets count parser cells and encoded bytes. Decoded allocations, RSS, SQLite
page work and worker CPU time are not bounded. Selector checks decode whole
source records.

Not implemented yet: fair per-input scheduling, global tool-call pairing,
cross-input election, a persistent search index, linked subagent promotion,
snippets, and retirement of old source bodies after excision.

## Tests

Unit tests that need private access live in `src/**/tests.rs` or
`*_tests.rs`; the rest are in `tests/`. All inputs are synthetic and use
temporary directories. Restart tests reopen the archive or exit the owning
process.

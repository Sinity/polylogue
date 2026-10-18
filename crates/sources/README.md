# polylogue-sources

Parsers for provider session files: Claude Code JSONL and Codex rollouts, plus
Codex prompt history and thread-history rows. The crate does no IO. The engine
hands it bytes and a read-only view of the stored parser state, and writes the
returned changes.

## Modules

- `frame`: splits input into records (JSONL lines, JSON array elements or
  SQLite rows), decodes each one, keeps its byte range and counts what the
  provider did with it. `Counts::balanced` checks that every framed record was
  used, ignored, unrecognized or undecodable.
- `raw`, `time`: byte-exact JSON field access and timestamp parsing shared by
  both providers.
- `selection`: field descriptors. A descriptor names the original record range
  and JSON path a normalized value came from, plus a versioned transform, so a
  reader can recompute the value from the stored bytes.
- `claude_code`, `codex`: the provider parsers.

## Parsing in steps

`frame::JsonlDriver` feeds complete lines to a provider's `RecordParser`, within
a byte and record budget. A partial last line is kept in a buffer that is moved,
never copied, to the next step. `checkpoint` ends a step;
`PendingJsonlStep::commit_with` hands back the continuation only after the
writer has committed the parser state, its output and the record position
together. If the parser or writer fails, the step is discarded and the next
attempt starts from the last committed state. A record larger than the budget's
`record_bytes` is reported as `NeedsCapacity`; it is never skipped or cut.
`drive_jsonl` runs a whole input through the same driver.

Records decode to `RawJson`, which keeps number tokens, UTF-16 strings and
object keys exactly as written. Text repairs are counted in
`Counts.text_repairs`, separately from record outcomes. Codex still refuses
non-scalar strings and keys.

## Provider state

Both parsers read stored cells through a `ReadView` (point lookups, plus ordered
neighbors for Codex) and return only the cells they changed, with events and
links. There is no whole-state snapshot and no second parser path;
`MemoryReadView` is an in-memory view for tests.

Claude Code groups records into responses by message ID, overlays repeated
records with the same UUID (later members replace earlier ones; `message`
members merge), resolves parents through non-Message records, splits sidechains
and subagents into their own Sessions, and attaches background-task
completions and required tool-result bodies to their calls. Published Messages
are assembled by the engine from the per-record contribution cells.

Codex groups items into model requests, replaces reasoning snapshots, pairs
calls with outputs (including running commands and code-mode wrappers), and
recognizes prompt echoes. A repeated identical item is a re-emission; a changed
snapshot replaces its blocks, keeps fields missing from the newer copy and
reports the conflict. Each changed Message has evidence cells, an
`EvidenceHead` and a `pending_versions` entry. The engine walks the evidence
slots with `EvidenceHead::next_slot`, computes the version digest outside its
write transaction and checks the evidence revision again before publishing.
`Message.version_hash` stays empty until then. Continuation output is stored as
`OutputChunk` cells with `pending_results` entries. Code-mode operations without
an owner and non-Agent commands become Session events and mark normalization
incomplete; unmapped content becomes an `Other` block. See
[the Codex notes](../../docs/rewrite/codex-native-representations.md) for
supported records and limits.

## Tests

Integration tests in `tests/` use synthetic fixtures. `tests/fixtures/python/`
holds files copied from the Python Polylogue; see
[python-port.md](../../docs/rewrite/python-port.md).
`tests/fixtures/morrow-main.README.md` explains how the Morrow fixture was
generated.

# Codex rollouts and thread-history databases

Import a rollout or a thread-history database into an archive while the daemon
runs:

```sh
polylogue --archive DIR import rollout.jsonl --format codex-rollout --wait
polylogue --archive DIR import history.sqlite --format codex-history-sqlite --wait
polylogue --archive DIR 'tool_items where session:codex:S' --json
polylogue --archive DIR 'tool_calls where session:codex:S text:needle' --json
polylogue --archive DIR 'operation:HASH@VERSION_HASH' read --budget 20000
```

MCP `search` takes the same query and `read` takes the returned `ref`.
`tool_items` lists every item, including unresolved and non-agent ones.
`tool_calls` lists known Agent operations once per scoped item, without
overlapping code wrappers. `text:` is an exact, case-sensitive substring of the
output, never of the command; empty output matches nothing. These queries need
one exact Session and refuse other operators.

## Supported format

The adapter follows OpenAI Codex `rust-v0.162.1`
(`092d3acd6bec3e3a14bdc7e7a2810ab628ab759d`): thread-history migrations 1
through 8 and the `CommandExecution` conversion in
`codex-rs/app-server-protocol/src/protocol/v2/item.rs`. Before decoding, the
store checks the `thread_items` columns and keys, the migration ledger and the
migration-8 turn field. An unsupported schema is stored but not decoded. Only
the store opens SQLite; the sources crate maps typed rows.

Capture opens the database read-only inside one read transaction and streams
every page through bundled SQLite `sqlite_dbpage`, so committed WAL pages and
page one are included consistently. The stored bytes are that read view, not a
copy of the main file. Logical row exports are not accepted. Citations name the
input, its full range, the schema, table, native row key and item-JSON hash.
Version evidence also covers the constructor, the pinned schema and every typed
row fact, so equal item JSON with different row facts gets a different version.
Later reads never open the original path.

## Operation identity

An operation ID frames the public Session, native turn and native item ID. The
public `invocation` is `CommandExecution.id`, scoped by Session and turn; for a
direct tool it equals the rollout call ID, so a rollout call can match when its
turn is known. A code wrapper's call ID differs from its child command's ID.
Native items have no tool name, so `tool` is null unless a rollout call supplies
it. Equal commands, generated `item-N` IDs and legacy Session or call IDs never
establish a match; generated IDs stay unresolved.

A rollout call without a turn gets a provisional identity from its block. A
completion event in the same source with matching `thread_id`, `turn_id` and
item ID refines it; the old unpinned reference then follows the new anchor, and
the old version-pinned reference still reads its own body. A completion without
its own turn stays provisional and never borrows the current turn context.
`CommandExecution` has no parent-call field: overlapping wrappers only suppress
double counting and never create a Message or parent.

`associations` distinguish `message_owner` from `wrapper_overlap` and carry the
request-call source ranges, the selected Message body key and hash, and the
index generation. An item is visible when its selected version is in that
Message body's operation root. Publishing an association never edits a ready
body or its version hash. Historical ToolItem reads return `selected: false` and
no association. Native UI items never create Messages.

## Results and counts

Each version keeps its own body, presence facts, constructor and citations.
`representations` previews up to 16 alternative citations, ordered by input,
version and source range; `representations_complete` says whether the preview
covers the history, and a truncated preview adds
`operation_representation_history` to `missing`. Exact version references stay
readable regardless.

An explicit empty core `aggregated_output` is `output: ""`; the UI conversion
drops it, giving `output: null` with a present result. A call without a
completed result has `result_present: false`. UserShell items are not agent
calls. Status labels without execution evidence do not make a pre-execution
item a completed Agent call. Conflicting classifications, results or output
bodies are excluded from proved call counts. Core argv stays a vector and
display text stays text; the provider's quoted, redacted display text is not
treated as argv.

Counts are reported separately: stored representations, decoded items, row
dispositions and correspondence. `total_items` is null because the full
population is unknown. While correspondence is unresolved, `tool_calls` and
`failed` are null and `known_agent_items` counts the known subset; an unknown
result also keeps `failed` null, and `outcomes_complete` shows which. Unknown
rows keep coverage degraded, and general Session coverage is always reported
incomplete. A contested Message merge stays unavailable while independently
known operations still publish; `import --wait` reports that.

Current-row selection prefers a present result, present output and a known
Message association. Every read checks the selected body's sources: each cited
rollout record must match its canonical hash (ignoring only top-level
`timestamp` and `ordinal`), and history evidence must match the stored snapshot
digest. Unavailable selected bytes are an error even if a normalized copy
exists; an unavailable alternative does not affect the selected one.

## Rollout records the parser reads

- Unified-command lifecycle records: a direct call and output Message stays
  readable when the operation's agent attribution is unknown, provided the
  native constructor, full session/turn/invocation scope, a valid duration and
  the Message association are all present. Lifecycle status is not copied into
  the Message, and a separately recorded output keeps its own result, status
  and origins in either arrival order. A small keyed record binds the
  constructor to its full operation ID and status-record hash; a replacement
  constructor removes it first.
- `AgentMessage` and `Reasoning` completion snapshots may arrive before their
  response items. The snapshot waits, keyed by thread, turn and item, until a
  matching response arrives; missing or conflicting counterparts leave
  normalization incomplete. Agent text and phase must agree, and an empty
  native reasoning body never replaces recorded encrypted content. A redundant
  snapshot adds no Message or evidence.
- A typed `UserMessage` echo pairs with the first unpaired equal prompt in the
  same turn and keeps that Message's identity. A duplicate echo cannot pair
  with another prompt, and a later response correction removes the pairing,
  retracting only its own Human attribution and echo evidence. An echo recorded
  before its response stays pending. Malformed turn metadata never falls back
  to the current turn. Unknown constructor members and raw reasoning content
  are refused.

## Limits

One snapshot: 16 MiB, 4096 typed rows and 16 MiB of item JSON. One operation:
2 MiB of metadata and output, 16 wrapper overlaps, 16 direct anchors, 17
association leaves per publication. Population reads: 4096 items, 4096
citations per item, 16 preview citations per item, 32 MiB of source
verification and 4 MiB of encoded data. Read budgets are 1 to 1048576 estimated
tokens and must fit the whole operation; larger output continuations are not
available. A rollout record is limited to 1 MiB. Exceeding a limit leaves the
work pending and never reports an empty population.

These records use the existing ToolItem, version, source, association and anchor
tables; no schema version changes. Live history collection, catalog
reconciliation, other item schemas and complete Codex Session export are not
supported here. [Configured discovery](codex-configured-discovery.md) covers
following a Codex home.

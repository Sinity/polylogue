# Explicit Codex operation representations

Import an explicitly supplied rollout or physical thread-history database into
an isolated archive with the ordinary daemon running:

```sh
polylogue --archive DIR import rollout.jsonl --format codex-rollout --wait
polylogue --archive DIR import history.sqlite --format codex-history-sqlite --wait
polylogue --archive DIR 'tool_items where session:codex:S' --json
polylogue --archive DIR 'tool_calls where session:codex:S text:needle' --json
polylogue --archive DIR 'operation:HASH@VERSION_HASH' read --budget 20000
```

The existing MCP `search` accepts the same query and `read` accepts the returned
`ref`. `tool_items` includes unresolved and non-agent items; `tool_calls` contains
proved Agent operations once per scoped item, excluding overlapping code
wrappers. `text:` is an exact, case-sensitive output substring. It never searches
command text. Empty output has no searchable term. These bounded queries require
one exact Session; other query operators are refused.

## Pinned interpretation

The adapter binds OpenAI Codex revision
`092d3acd6bec3e3a14bdc7e7a2810ab628ab759d` (`rust-v0.162.1`), thread-history
migrations 1 through 8, and the `CommandExecution` core-to-app-server conversion
in `codex-rs/app-server-protocol/src/protocol/v2/item.rs`. Store checks the
`thread_items` column/key shape, the migration ledger and the migration-8 turn
field before decoding rows. Unsupported schemas remain retained but cannot
produce a successful decoded result. Only store opens SQLite. Sources receive
typed rows and map the UI `commandExecution` payload without SQL or source IO.

Capture opens the supplied regular database read-only, establishes a read
transaction before SQLite backup, and retains the resulting physical snapshot.
It therefore includes committed WAL pages consistently; the retained bytes are
the backup image, not a purported copy of the original main file. Citations name
that immutable input, its full physical range, the schema, table, exact native
row key and canonical item-JSON hash. Logical row exports are not accepted as
physical snapshots. The original path is never consulted by subsequent reads.

Operation identity length-frames the opaque public Session, native turn and
native item ID under the Codex namespace. The public `invocation` is the
provider's `CommandExecution.id`, scoped by Session and turn. In the pinned
core constructor the direct tool's `ctx.call_id` becomes that item ID; a rollout
response call can therefore correspond when its turn is independently known.
A code wrapper's call ID is a different invocation from its child command ID.
Equal commands, generated legacy UI `item-N` IDs and legacy Session/call IDs
supply no correspondence proof. Generated UI IDs remain unresolved.

A direct rollout call lacking turn scope has a provisional identity derived
from its independently proved block. An exact same-source call followed by the
pinned completion event, with matching `thread_id`, native `turn_id` and item ID,
can refine it. The old unpinned reference follows one direct anchor; the old
version-pinned reference still reads its immutable body and citations. A
standalone native row cannot assign a turn to an unrelated unscoped call.
`CommandExecution` has no parent-call field in this schema. Open wrapper overlap
supports counting suppression, never an invented Message or wrapper parent.
Native UI items do not manufacture Messages.

## Presence and qualification

Each version retains its own body, presence facts, constructor and source
citations. `representations` additionally lists alternative immutable witnesses;
those alternatives are not claimed to contribute to the selected body. A core
explicit empty `aggregated_output` is `output: ""`. The pinned UI conversion
filters that empty display text, producing `output: null` while retaining a
present result. Neither means missing result. A call without a completed result
has `result_present: false`. UserShell is non-agent. Source labels or terminal
status without execution evidence cannot upgrade a pre-execution item to a
completed Agent call. Conflicting known classifications, results or output
bodies qualify correspondence and are excluded from a proved call count.

The population reports retained representations, decoded selected items, source
row dispositions and correspondence independently. A source with unknown rows
keeps coverage degraded. The query always reports incomplete general Session
coverage: a physical paginated projection supplies neither request boundaries
nor proof that the entire Session was acquired. Current-row selection prefers a
present result, present output and independently proved Message association;
all historical versions remain addressable. Selected-body source availability
is checked on every read. Unavailable selected bytes produce an error with no
shown body, even if a normalized copy exists. An unavailable alternative does
not invalidate an independent selected representation.

## Admission and remaining scope

This first path admits a physical snapshot up to 16 MiB, 4096 typed rows and
16 MiB of item JSON; one operation's normalized metadata plus output is limited
to 2 MiB. Existing chunked immutable values hold admitted bodies. Population
reads admit 4096 selected items, 4096 citations per item and 4 MiB encoded data.
Read display budgets are 1 through 1048576 estimated tokens and must fit the
whole operation; operation-output continuations and resource qualification for
larger bodies remain unavailable. Admission failures retain owed work and never
claim a complete empty population.

The existing archive/index schema owns these records; this change uses its
provisioned ToolItem/version/source/association/anchor tables and adds no durable
schema version or reset. Healthy predecessor archives follow their ordinary
migration owner. Live SQLite collection, catalog discovery, source-default
replacement, page-patch scheduling, other item schemas and complete Codex
Session/export support remain outside this path.

`scripts/check-codex-native.py --binary PATH --output NEW_DIR` exercises the
ordinary packaged CLI and actual SDK MCP process against synthetic fixtures.
The declared `native_representation_demo` operation runs it and retains command
arguments, replies, daemon logs and its report outside Git. It covers both
arrival orders, wrapper ambiguity, turn/legacy/classification controls,
provisional references, restart, same-request retry and unavailable bytes.

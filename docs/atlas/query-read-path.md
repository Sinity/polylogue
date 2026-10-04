# Query and Read Path

## Area boundary

The query layer turns explicit user intent into a typed SQL plan and bounded
pages. Read surfaces expose stable archive objects and projections through
the CLI, API, MCP, and daemon without reimplementing substrate semantics
(`polylogue/archive/query/expression.py:1-80`; `polylogue/archive/query/transaction.py:1-100`).

## Route

```text
CLI/API/MCP request → filters + expression DSL → query plan → page/viewport
                                      ↓
                              index/read models
```

The CLI is query-first: root filters precede `find`, and query intent must be
signalled by `find`, a quoted expression, or field syntax. The query DSL is
lowered to SQL; it is not a grep-like post-filter. Pagination and cancellation
are part of the route contract (`polylogue/archive/query/transaction.py:1-100`).

## Read identities

Use generated session, message, and block identities for exact reads. Use
public `origin` filters, not provider-wire names. Lineage-aware reads compose
parent prefixes and report cycle or dangling-branch-point status rather
than silently claiming completeness
(`_composed_transcript_plan` in `polylogue/storage/sqlite/archive_tiers/write.py:3399`).

Ordinary block reads preserve the stored language and media type through typed records and domain hydration. Message streams hydrate blocks in bounded batches on their held read connection; closing a public stream closes its nested reader before releasing that connection.

## Scoped ranked reads

Semantic and hybrid session requests qualify the canonical SQL and residual scope before ranking. The held ArchiveStore connection supplies current prose and occurrence identity to the retained-vector TEMP projection. Exact L2 scoring covers every eligible stored output; near takes the minimum over all stored seed outputs. Session witness selection, complete hybrid lane ranks and n-ary RRF precede the final session window (`polylogue/archive/query/archive_execution.py`, `polylogue/storage/search_providers/sqlite_vec_queries.py`, `polylogue/storage/sqlite/archive_tiers/archive.py`). TEMP relations use FILE storage selected at connection acquisition. Successful full lane settlement is explicit in `completed_lanes`; unavailable or failed lanes remain named gaps.

Session-level existential DSL filters stay independent of the semantic witness. Semantic under a correlated unit predicate or a non-session terminal source remains a typed compilation refusal (`polylogue/archive/query/expression.py`).

## Terminal outcome

Every row-bearing envelope carries one typed outcome -- `ok`, `empty`,
`degraded`, or `error` -- decided at the operation boundary
(`polylogue/surfaces/outcome.py:1-60`). `empty` means the declared scope
completed and holds zero rows; `degraded` means named gaps shaped the answer,
so absent rows may be the gap rather than the archive; `error` means no valid
answer. Surfaces serialize that decision and map it to their transport: the
CLI exit table is `OUTCOME_EXIT_CODES`, and the daemon's readiness chips are
projections of it. A surface that re-derives readiness from `not rows` has
reintroduced the ambiguity the type removes.

## Controlled-read boundary

A read may open the archive itself only through one of the two declared
read-boundary owners; every other direct open is an explicit writer-lease
open. The invariant, its owners and its change procedure are one stanza in
`doctrine.md`. No gate censuses direct opens any longer;
`tests/unit/daemon/test_surface_data_boundary.py` refuses connection openers in
the surface packages.

A read view that computes its own answer in-process is a different question
from this one: those are declared, and shrink-only, in
`polylogue/cli/read_view_registry.py:95-110`.

## Surface ownership

- `polylogue/cli/` owns command grammar and human/machine presentation.
- `polylogue/api/` owns the Python facade and typed result payloads.
- `polylogue/mcp/` owns capability-gated operation dispatch.
- `polylogue/daemon/` owns transport, lifecycle, and serialized mutation
  admission; its HTTP/UDS readers use the same product routes.
- `polylogue/analysis/` owns descriptor-driven derived projections.

Surfaces should call shared operations, query planners, and insight
descriptors. A surface-specific SQL query is a design smell unless its
read-model contract is explicitly owned there.

## Gotchas

New Click parameters on query verbs go last: positional shifts silently
reroute arguments. Stable JSON output requires schema and parity checks across
CLI/API/MCP. A successful page is not proof that derived models are current;
readiness and staleness are explicit fields. Never turn unavailable data into
an exact zero or infer tool failure from prose.

## Verification route

Begin with the focused query or surface test through `devtools test`. For a
cross-surface change, run the relevant CLI/API/MCP parity tests, pagination and
cancellation coverage, then the generated surface check. Use `devtools why` to inspect a managed verification refusal or
failure before interpreting a receipt.

## Large session evidence

`file-edits` and `web-content` share the byte-bounded evidence owner in
`polylogue/operations/evidence_window.py` and `evidence_payloads.py`.
Small rows remain ordinary objects in `rows`. An oversized row is delivered
as `row_fragment` instead, with its row offset and ordered field fragments.
Each field declares its name, `encoding` (`utf-8` or `json`), byte `offset`,
`total_bytes`, and `data_base64`. Concatenate decoded bytes in offset order;
decode UTF-8 only when the entire field is present, then parse JSON for
`encoding=json`. Empty strings, nulls, booleans, and structured patches keep
their normal projected values. The fragment's `complete` completes that row,
not necessarily the relation.

`returned`, `offset`, `next_offset`, and `total` count completed rows, never
fragments. A page can therefore have `rows=[]`, `returned=0`, and an advancing
continuation while delivering part of a row. Continue until the window's
`complete` is true. The token binds both row and field-byte coordinates to
the original result, snapshot, and expiry; changing the transport byte budget
does not change result identity. SQLite reads only bounded field slices on
resume. Evidence insert/update/delete advances the archive frame so callers
cannot silently assemble one row from two revisions. CLI, Python API, and MCP
preserve this contract; MCP supplies its smaller delivery budget to the owner.

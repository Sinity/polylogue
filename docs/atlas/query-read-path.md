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
parent prefixes and report depth-limit or dangling-branch-point status rather
than silently claiming completeness
(`polylogue/storage/sqlite/archive_tiers/write.py:1488-1555`).

## Surface ownership

- `polylogue/cli/` owns command grammar and human/machine presentation.
- `polylogue/api/` owns the Python facade and typed result payloads.
- `polylogue/mcp/` owns capability-gated operation dispatch.
- `polylogue/daemon/` owns transport, lifecycle, and serialized mutation
  admission; its HTTP/UDS readers use the same product routes.
- `polylogue/insights/` owns descriptor-driven derived projections.

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
cancellation coverage, then `devtools verify doc-commands` and the generated
surface check. Use `devtools why` to inspect a managed verification refusal or
failure before interpreting a receipt.

verified: 24be873c0 2026-08-27

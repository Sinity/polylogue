# Execution Plan

Living sequencing plan for Polylogue subsystem maturation.
Updated per-PR. See `docs/architecture-spine.md` for the target shape.

## Landed

| Subsystem | Status | Closed by |
|-----------|--------|-----------|
| Archive substrate (acquisition, parsing, persistence, FTS, blob store) | Done | Multiple PRs |
| Content hash idempotency | Done | #838, #421 |
| Session insights (profiles, work events, phases, threads) | Done | Multiple PRs |
| Cost/subscription tracking and outlook | Done | #938, #943 |
| Daemon convergence (named stages) | Done | Multiple PRs |
| Browser extension + receiver | Done | #937, #940 |
| Identity ledger (stable IDs across re-ingestion) | Done | #775, #963 |
| MCP server (35+ tools, read/write/admin roles) | Done | Multiple PRs |
| CLI context-pack | Done | #968 |
| Paste detection | Done | #947, #966 |
| Root artifact cleanup | Done | #954, #966 |
| Verification baseline (format, lint, mypy, topology, manifests) | Done | #944 |
| Type tightening (SubjectRef.kind enum) | Done | #421, #968 |
| CLI mutation flags (--add-tag/--remove-tag) | Done | #862, #967 |
| Manifest-only conversation prevention | Done | #945 |
| MCP error sanitization and metadata validation | Done | #948 |
| CLI format matrix coverage | Done | #949 |

## In Flight

| Substream | Blocked by | Next artifact |
|-----------|-----------|---------------|
| Embedding pipeline activation (#828) | VOYAGE_API_KEY + daemon opt-in | Daemon-owned post-ingest embedding convergence |
| Web reader realtime (#957) | — | SSE streaming channel from daemon to reader |

## Queued

Dependency order (items in each tier are parallelizable):

1. **Storage hardening**: blob GC (#818), FTS bloat reduction (#817), daemon safety (#771)
2. **Archive semantics**: context/protocol artifact storage (#839), provider_meta graduation (#864)
3. **Read surfaces**: search explainability (#873), reader marks (#867), session lineage (#866)
4. **Operational surfaces**: daemon validation failure visibility (#844), maintenance planner (#871), read-surface SLOs (#872)
5. **Verification depth**: systematic test architecture (#807), structural simplification (#805)
6. **Product polish**: CLI polish (#958), docs revamp (#952), broad distribution (#953)

## Frozen / Parked

| Subsystem | Freeze reason | Unfreeze condition |
|-----------|--------------|-------------------|
| TUI dashboard | Lower priority than web reader | Web reader reaches stable MK2 |
| Site publication | Lower priority than web reader | Reader docs completion |

# Execution Plan

Living sequencing plan for Polylogue subsystem maturation. This document is a
coordination map, not a replacement for issue acceptance criteria. See
`docs/architecture-spine.md` for the target shape.

## Landed

| Subsystem | Status | Closed by |
|-----------|--------|-----------|
| Archive substrate (acquisition, parsing, persistence, FTS, blob store) | Done | Multiple PRs |
| Content hash idempotency | Done | #838, #421 |
| Session insights baseline (profiles, work events, phases, threads) | Landed; rigor hardening remains | Multiple PRs, #1019 |
| Cost/subscription tracking and outlook slice | Landed; product forecasting remains | #938, #943, #995 |
| Daemon convergence architecture | Landed; production proof and residual workload remain | #847, #854, #845, #1036 |
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
| Daemon convergence proof and residual workload (#845, #1036) | Deployment via Sinnix for latest merged daemon changes | Production-corpus convergence report and packaged rollout |
| Source vocabulary and local-agent sources (#1022) | — | Public source-family contract and filter/completion parity |
| Insight rigor and downstream contracts (#1019) | — | Product-by-product evidence/inference/readiness matrix |
| Web reader realtime (#957) | — | SSE streaming channel from daemon to reader |
| Reconciliation ledger (#944) | Open residual owners stay precise | Closeout matrix mapping stale closure claims to owner issues |

## Queued

Dependency order (items in each tier are parallelizable):

1. **Storage hardening**: blob GC (#818), FTS bloat reduction (#817), daemon safety (#771)
2. **Archive semantics**: context/protocol artifact storage (#839), provider_meta graduation (#864), source vocabulary (#1022)
3. **Read surfaces**: search explainability (#873), reader marks (#867), session lineage (#866), insight rigor (#1019)
4. **Operational surfaces**: maintenance/replay planner (#996), daemon health notifications (#999), read-surface SLOs (#872)
5. **Verification depth**: systematic test architecture (#997), verifiability dashboard/traceability (#998), evidence quality (#594/#590)
6. **Product polish**: CLI polish (#958), docs revamp (#952), broad distribution (#953), webui advanced functionality (#993)

## Frozen / Parked

| Subsystem | Freeze reason | Unfreeze condition |
|-----------|--------------|-------------------|
| TUI dashboard | Lower priority than web reader | Web reader reaches stable MK2 |
| Site publication | Lower priority than web reader | Reader docs completion |

# Current issue execution control plane

This document is the current top-down execution layer for the open Polylogue issue set. Generated-pack version names are historical context, not dispatch truth. Use concrete issue bodies, later comments, and this document for coding-agent handoff.

## Dispatch doctrine

Issues should make coding agents instantiate decisions, not rediscover them. Every implementation PR states issue slice, inspected files, contracts or tests added, old surface removed or explicitly retained, verification commands and result, and any mismatch between issue intent and code reality.

Do not preserve stale public aliases. Do not add ceremonial proof, witness, QA, or detached-manifest layers. Prefer code-coupled contracts, generated tests, fixture worlds, typed registries, and explicit negative tests.

## Current critical path

1. #1873: stop-line correctness for query/action behavior.
2. #1818: machine-output/action-contract hub, gating #1842, #1849, #1825, and #1847.
3. #1842: public command-floor prune. Ship the familiar `find QUERY then ACTION` floor and remove the old command zoo. Do not absorb the full #2006 ceiling.
4. #2006: full query DSL substrate.
5. #1880: transform-first recovery/digest views for agent sessions.
6. #1883, #1882, #1881: KV/assertions, agent identity/context/delivery, and Beads/GitHub/Polylogue boundary.
7. Demo, README, release, web workbench, and work packets follow the import/query/action/output spine.

## Issue notes

#1818 needs an executable action contract beside command/action registration. The contract should describe path, effect, input kind, supported formats, default format, machine schema, cardinality, daemon requirement, guards, and completion context. JSON means one finite document on stdout. NDJSON means one object per line. Human diagnostics go to stderr.

#1842 owns the public command floor: `find`, `import`, `config`, `ops`, and terminal actions `read`, `mark`, `analyze`, `remove`, and `continue` if it truly resumes work. `read` absorbs show/open/messages/raw/export. `mark` absorbs tags/user-state/blackboard/feedback. `analyze` absorbs stats/facets/cost/neighbors/insights/diagnostics/correlate.

#2006 owns the full query DSL: a typed AST and lowering pipeline over SQLite, FTS5, vector search, recursive CTEs, EXISTS subqueries, and existing read/action contracts. It is not a custom database engine. It includes Boolean predicates, structural message/action predicates, lineage/run/event/assertion traversal, aggregation, terminal report/action stages, and explain output. The Lark grammar is the query grammar; old flat-expression behavior is only parity evidence while the AST lowerer absorbs and replaces it.

#1815 owns public import/demo/source vocabulary. Public command is `import`; internal ingestion terminology may remain only for daemon or pipeline mechanics. Remaining work is demo fixture worlds, parser regression coverage, and operation/status truthfulness.

#1810 owns terminology cleanup: public vocabulary is session/origin, not conversation/provider. Raw third-party export terms stay only inside parser/raw-fixture boundaries.

#1844 completion must consume the same grammar/AST as CLI, daemon, MCP, and web. It suggests fields, values, and actions from the archive and contracts. Completion performs no writes.

#1880 makes compiled recovery/digest views the default read surface for coding-agent sessions. Initial outputs: resume packet, digest, typed timeline, subagent reports, tool summaries, decisions/KV candidates, forensic index. Every extracted claim links back to raw evidence.

#1883 unifies user and agent overlays as typed KV/assertions: marks, annotations, highlights, corrections, lessons, decisions, blockers, handoffs, RunState, prompt evaluations, and transform candidates. Raw evidence and deterministic indexes are not KV.

#1882 separates RoleSpec, AgentPath, AgentInstance, AgentRun, ContextEnvelope, CommunicationEvent, and DeliveryEvent so work packets can say which session ran, what context it had, and what outside review material was actually seen.

#1881 keeps the boundary clear: Beads owns internal work graph state; GitHub owns public collaboration state; Polylogue owns evidence, traces, context, KV assertions, work packets, and reports.

#1838/#1840/#1845 work packets are forensic bundles around attempts and outcomes, not a task tracker. They link sessions/runs, identity, context, branch/worktree, PR/issues/Beads, checks, touched files, KV assertions, caveats, and outcome.

#1846/#1847 web workbench should inspect the same objects as CLI/MCP and must not create a second query/action/DTO model.

#1848/#1849 replace stale public mental models only when their useful behavior is absorbed by read/analyze/web/tests. Replace proof/QA/showcase confidence layers with code-coupled contracts, fixture worlds, and generated tests.

#1830/#1832/#1851 status/readiness/performance must expose real archive health and import/backpressure state. Performance fixes are profile-first and produce artifacts.

## Review discipline

Required-green is not enough when bot or human review findings arrive after checks settle. Merge decisions should record whether review material was read or injected into the agent context.

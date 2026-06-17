# Current issue execution control plane

This document is the current top-down execution layer for the open Polylogue issue set. Generated-pack version names are historical context, not dispatch truth. Use concrete issue bodies, later comments, and this document for coding-agent handoff.

## Dispatch doctrine

Issues should make coding agents instantiate decisions, not rediscover them. Every implementation PR states issue slice, inspected files, contracts or tests added, old surface removed or explicitly retained, verification commands and result, and any mismatch between issue intent and code reality.

Do not preserve stale public aliases. Do not add ceremonial confidence scores,
witness ledgers, QA badges, or detached-manifest layers. Prefer code-coupled
contracts, generated tests, fixture worlds, typed registries, and explicit
negative tests.

## Current critical path

The old command/output floor is largely landed. The active path is no longer
"make the archive usable at all"; it is making the surviving query, recovery,
assertion, API, and web surfaces coherent enough that a new agent or user can
inspect real AI work without reading chat logs.

1. #2006: full query DSL substrate. Keep extending the one Lark-backed grammar,
   typed AST, lowerers, explain output, and completion/query-builder metadata.
2. #1880: on-demand recovery views for coding-agent sessions. Recovery is a
   deterministic transform/read surface, not a stored compile step.
3. #1883 + #1845: assertion and ref substrate. User overlays, transform
   candidates, caveats, decisions, and lessons must point at evidence refs.
4. #1882 + #1840 + #1838: run/context/event projection into recovery/work
   packets. Build fixture-backed projections that improve handoff views.
5. #1847 + #1846 + #1824: stable local web/API/capture route contracts, then a
   demonstrable workbench flow over the same query/read/recovery/assertion
   contracts.
6. #1825 + #1827 + #1849: cross-surface parity, release gate truthfulness, and
   deletion/folding of any detached documentation or evaluation surface that is
   not coupled to executable behavior.

## Issue notes

#1816 remains the action-contract source for mutation/read command behavior.
Do not recreate parallel machine-output registries. JSON means one finite
document on stdout. NDJSON means one object per line. Human diagnostics go to
stderr.

#2006 owns the full query DSL: a typed AST and lowering pipeline over SQLite, FTS5, vector search, recursive CTEs, EXISTS subqueries, and existing read/action contracts. It is not a custom database engine. The Lark grammar is the query grammar today; compact field/text clauses and explicit Boolean predicates are entry shapes in that grammar, not a legacy/future split. Current execution covers Boolean session predicates, message/action/block `exists`, action sequences, FTS, lineage, and semantic seed plus residual filters. Remaining work is broader unit coverage (runs/events/assertions/context/bundles), traversal, aggregation, terminal report/action stages, explain metadata, and shared completion/query-builder metadata.

#1880 makes on-demand recovery/digest views the default handoff surface for
coding-agent sessions. Initial outputs include resume packet, digest, typed
timeline, subagent reports, tool summaries, decision/assertion candidates,
forensic index, and work packet views. Every extracted claim links back to raw
evidence.

#1883 unifies user and agent overlays as typed KV/assertions: marks, annotations, highlights, corrections, lessons, decisions, blockers, handoffs, RunState, prompt evaluations, and transform candidates. Raw evidence and deterministic indexes are not KV.

#1882 owns the bounded Run, ContextSnapshot, and ObservedEvent projection. Use
these to say which concrete execution acted, what context it had, what it
spawned, and which review/check/comment evidence was actually seen or injected.
Do not add a broader agent ontology unless fixtures prove the primitive set
cannot represent the behavior.

The Beads/GitHub/Polylogue boundary is now expressed by #1807 and child issue
bodies rather than a separate issue: Beads owns internal work graph state;
GitHub owns public collaboration state; Polylogue owns evidence, traces,
context, assertions, work packets, and reports.

#1838/#1840/#1845 work packets are forensic bundles around attempts and outcomes, not a task tracker. They link sessions/runs, identity, context, branch/worktree, PR/issues/Beads, checks, touched files, KV assertions, caveats, and outcome.

#1846/#1847 web workbench should inspect the same objects as CLI/MCP and must not create a second query/action/DTO model.

#1849 replaces stale public mental models only when their useful behavior is
absorbed by read/analyze/web/tests. Replace detached confidence, QA, and
showcase layers with code-coupled contracts, fixture worlds, generated tests,
and benchmark artifacts that fail on real drift.

## Review discipline

Required-green is not enough when bot or human review findings arrive after checks settle. Merge decisions should record whether review material was read or injected into the agent context.

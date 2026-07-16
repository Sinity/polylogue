# 188. polylogue-fs1 — Hermes bridge: state.db + runtime spans -> canonical evidence -> forensics/eval export

Priority/type/status: **P2 / epic / open**. Lane: **11-interoperability-origin**. Release: **K-interop-origin**. Readiness: **epic-needs-child-closure**.

## What the bead says

Positioning: Hermes acts; Polylogue remembers and explains what the agent did; Sinex knows what was happening on the machine around it. Hermes already HAS observability hooks (observer layer, middleware, Langfuse, NeMo Relay ATOF/ATIF export) — so 'add observability' and 'trajectory export' are not the wedge. The wedge: Hermes traces are product-local operational data; Polylogue turns them into a durable, cross-provider, longitudinal evidence corpus with provenance, cost semantics, git correlation, and machine-context windows.

Key correction (hermes.md): the current parse_hermes path reads optional ~/.hermes/sessions/session_*.json snapshots — a compatibility adapter, likely off by default upstream. Hermes's real durable state is state.db (sessions, messages, tool calls, token/cache/reasoning counters, costs, parent sessions, compaction/archive/rewind state, FTS). This epic subsumes the old gh#2460 'parse_hermes work-structure fidelity' scope — delegation/tasks/lineage land from state.db + observer spans, not from deeper snapshot parsing.

Strategic role: a flagship external demo ('2-minute demo: polylogue forensics hermes'). Operator does not use Hermes yet — value is the external artifact plus the fully-local-stack story. Priority rises when an artifact from this program is scheduled as a deliverable. A hermes-forensics.zip prototype was built in an earlier session (fixtures, loop/stall/reasoning-burn detection, Atropos JSONL round-tripped through Nous jsonl2html.py) — treat as prototype record, design from scratch against current Hermes. VERIFY current Hermes internals before building; the analysis snapshot is 2026-07 (v0.18.0) and Hermes ships ~1,700 commits per minor release.

## Existing design note

Bridge Hermes's durable state.db (sessions, messages, tool calls, token/cache/reasoning/cost counters, parent sessions, compaction/archive/rewind state, FTS) plus observer/runtime spans into canonical Polylogue evidence — NOT the optional ~/.hermes/sessions/session_*.json snapshots (a compatibility adapter, likely off by default). The wedge is turning product-local operational traces into a durable, cross-provider, longitudinal evidence corpus with provenance, cost semantics, git correlation, and machine-context windows. Subsumes the old gh#2460 parse_hermes work-structure scope (delegation/tasks/lineage come from state.db + spans). Flagship deliverable: a '2-minute polylogue forensics of Hermes' export artifact. VERIFY current Hermes internals first (analysis snapshot is v0.18.0; ~1,700 commits per minor release); an earlier hermes-forensics.zip prototype is a record only, design from scratch.

## Acceptance criteria

- Current Hermes internals are verified against the live build before building — the state.db schema (sessions, messages, tool calls, token/cache/reasoning counters, costs, parent sessions, compaction/archive/rewind, FTS) is confirmed and the verification note is recorded on the bead.
- The bridge reads Hermes state.db + observer/runtime spans (not the ~/.hermes/sessions snapshots) into canonical Polylogue evidence rows carrying provenance, cost semantics, git correlation, and machine-context windows.
- The flagship deliverable is defined and reproducible from a fixture: a '2-minute polylogue forensics of Hermes' forensics/eval export (e.g. Atropos JSONL round-trip).
- gh#2460 delegation/tasks/lineage scope is covered from state.db + spans; child beads track the write path; the epic advances only when an artifact is scheduled as a deliverable (operator gate).

## Static mechanism / likely defect

Issue description localizes the mechanism: Positioning: Hermes acts; Polylogue remembers and explains what the agent did; Sinex knows what was happening on the machine around it. Hermes already HAS observability hooks (observer layer, middleware, Langfuse, NeMo Relay ATOF/ATIF export) — so 'add observability' and 'trajectory export' are not the wedge. The wedge: Hermes traces are product-local operational data; Polylogue turns them into a durable, cross-provider, longitudinal evidence corpus with provenance, cost semantics, git correlation, and machine-con… Design direction: Bridge Hermes's durable state.db (sessions, messages, tool calls, token/cache/reasoning/cost counters, parent sessions, compaction/archive/rewind state, FTS) plus observer/runtime spans into canonical Polylogue evidence — NOT the optional ~/.hermes/sessions/session_*.json snapshots (a compatibility adapter, likely off by default). The wedge is turning product-local operational traces into a durable, cross-provider, …

## Source anchors to inspect first

- `polylogue/sources/dispatch.py` — Current origin/source dispatch logic; target for OriginSpec consolidation.
- `polylogue/sources/import_preflight.py` — Preflight/readiness should report origin strictness and ambiguity.
- `polylogue/sources/provider_completeness.py` — Provider completeness is adjacent to OriginSpec readiness.
- `polylogue/sources/parsers/base.py` — Parser base contracts should be folded into OriginSpec.
- `polylogue/core/refs.py` — Existing ref model should be extended, not bypassed.
- `polylogue/storage/sqlite/archive_tiers/user_write.py:901` — Findings should reuse assertion/candidate lifecycle.
- `polylogue/surfaces/payloads.py:747` — Surface payloads currently expose branch/message refs; expand consistently.
- `polylogue/mcp/payloads.py:377` — MCP message payloads carry variant/branch metadata today.

## Implementation plan

1. Bridge Hermes's durable state.db (sessions, messages, tool calls, token/cache/reasoning/cost counters, parent sessions, compaction/archive/rewind state, FTS) plus observer/runtime spans into canonical Polylogue evidence — NOT the optional ~/.hermes/sessions/session_*.json snapshots (a compatibility adapter, likely off by default).
2. The wedge is turning product-local operational traces into a durable, cross-provider, longitudinal evidence corpus with provenance, cost semantics, git correlation, and machine-context windows.
3. Subsumes the old gh#2460 parse_hermes work-structure scope (delegation/tasks/lineage come from state.db + spans).
4. Flagship deliverable: a '2-minute polylogue forensics of Hermes' export artifact.
5. VERIFY current Hermes internals first (analysis snapshot is v0.18.0
6. ~1,700 commits per minor release)
7. an earlier hermes-forensics.zip prototype is a record only, design from scratch.

## Tests to add

- Acceptance proof: Current Hermes internals are verified against the live build before building — the state.db schema (sessions, messages, tool calls, token/cache/reasoning counters, costs, parent sessions, compaction/archive/rewind, FTS) is confirmed and the verification note is recorded on the bead.
- Acceptance proof: The bridge reads Hermes state.db + observer/runtime spans (not the ~/.hermes/sessions snapshots) into canonical Polylogue evidence rows carrying provenance, cost semantics, git correlation, and machine-context windows.
- Acceptance proof: The flagship deliverable is defined and reproducible from a fixture: a '2-minute polylogue forensics of Hermes' forensics/eval export (e.g.
- Acceptance proof: Atropos JSONL round-trip).
- Acceptance proof: gh#2460 delegation/tasks/lineage scope is covered from state.db + spans
- Acceptance proof: child beads track the write path
- Acceptance proof: the epic advances only when an artifact is scheduled as a deliverable (operator gate).

## Verification commands

- `devtools test <focused tests added for this bead>`

## Pitfalls

- Do not implement the epic as a single broad PR; use it to close/split child work and verify the terminal invariant.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.

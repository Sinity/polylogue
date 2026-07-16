# 174. polylogue-60i5 — Durable-tier batch coordination: one user v4->v5 and one source v2->v3 migration window

Priority/type/status: **P2 / task / open**. Lane: **06-agent-context-coordination**. Release: **D-agent-context-coordination**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

Cross-cutting operational constraint (the single biggest insight across the R&D specs): MANY pending designs each want a durable-tier bump — user v5: recursive-safety columns, content-variants tables, s7ae coordination messages, config-engine settings, queries/result-sets/analyses (rxdo.2, rxdo.8); source v3: compaction snapshots, ingest-fidelity fingerprints, secret-redaction tombstones, zstd blob placement. Durable migrations run behind verified backup manifests one user_version step at a time — landing them piecemeal means repeated backup+migrate ceremonies on the live 38GB archive. This bead is the COORDINATION POINT: collect the ready durable-tier changes, cut ONE numbered migration per tier per window, land together. Blocks nothing by itself; each schema:user-v5 / schema:source-v3 labeled bead should reference it and hold its migration file until a window is declared.

## Existing design note

Process: when >=2 labeled beads are execution-ready, declare a window, write migrations NNN as one coherent set per tier, verify backup manifest, apply one step, run parity tests. The z7rv migration framework (closed 2026-07-04) is the runner substrate. Index-tier (v24->v25) batching is separate and cheaper (blue-green b5l removes downtime) but the same batching discipline applies — coordinate via the schema-bumps bd memory.

## Acceptance criteria

First batch window executed with a single user-tier migration covering all ready v5 consumers; no durable migration lands outside a declared window. Verify: migration chain contiguity test + backup manifest check.

## Static mechanism / likely defect

Issue description localizes the mechanism: Cross-cutting operational constraint (the single biggest insight across the R&D specs): MANY pending designs each want a durable-tier bump — user v5: recursive-safety columns, content-variants tables, s7ae coordination messages, config-engine settings, queries/result-sets/analyses (rxdo.2, rxdo.8); source v3: compaction snapshots, ingest-fidelity fingerprints, secret-redaction tombstones, zstd blob placement. Durable migrations run behind verified backup manifests one user_version step at a time — landing them pie… Design direction: Process: when >=2 labeled beads are execution-ready, declare a window, write migrations NNN as one coherent set per tier, verify backup manifest, apply one step, run parity tests. The z7rv migration framework (closed 2026-07-04) is the runner substrate. Index-tier (v24->v25) batching is separate and cheaper (blue-green b5l removes downtime) but the same batching discipline applies — coordinate via the schema-bumps b…

## Source anchors to inspect first

- No precise source anchor was localized in this static pass. Start from the bead description and repository search.

## Implementation plan

1. Process: when >=2 labeled beads are execution-ready, declare a window, write migrations NNN as one coherent set per tier, verify backup manifest, apply one step, run parity tests.
2. The z7rv migration framework (closed 2026-07-04) is the runner substrate.
3. Index-tier (v24->v25) batching is separate and cheaper (blue-green b5l removes downtime) but the same batching discipline applies — coordinate via the schema-bumps bd memory.

## Tests to add

- Acceptance proof: First batch window executed with a single user-tier migration covering all ready v5 consumers
- Acceptance proof: no durable migration lands outside a declared window.
- Acceptance proof: Verify: migration chain contiguity test + backup manifest check.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools test tests/unit/coordination tests/unit/mcp -q`

## Pitfalls

- Do not add direct context injection or trusted memory writes outside the scheduler/candidate policy.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.

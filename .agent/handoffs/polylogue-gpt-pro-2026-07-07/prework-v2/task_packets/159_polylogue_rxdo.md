# 159. polylogue-rxdo — Analysis provenance: queries, result-sets, findings, analyses as first-class objects

Priority/type/status: **P2 / epic / open**. Lane: **05-analysis-provenance-citations**. Release: **C-read-contract**. Readiness: **epic-needs-child-closure**.

## What the bead says

THE convergent frontier from the 2026-07-05 R&D program (hit independently by swarm waves 2/4 and multiple GPT-Pro review branches). Today a query is a transient execution: nothing addressable survives it, so analyses cannot be iterative, citable, annotatable, or composable, and Polylogue cannot observe its own use. Target object graph: query:<hash> (content-addressed canonical plan) -> query_run (execution event) -> result_set (relation snapshot with grain+corpus epoch) -> finding (assertion-kind claim, judge lifecycle) -> analysis_run/report, plus annotation batches as the provenance container for external-agent labeling. TIER PLACEMENT (synthesized from two competing corpus designs, verified against reset semantics): query identity + promoted manifests + query_edges are DURABLE user.db (v5 additive migration); every committed run + routine result fingerprints are ops.db telemetry; index.db holds only a rebuildable member cache. Rationale: result snapshots are functions of (source x query x time), NOT derivable from source alone, so the index-tier rebuild contract cannot hold them — a reset --index must not destroy cited evidence. Previews are never persisted. Enables: standing queries as change detectors, findings-as-tests (CI invariants), self-observed usage analytics, external annotation loops, citable reports, cohort set-algebra operands.

## Acceptance criteria

A committed query returns stable query/query-run/result-set refs on every surface; assertions can target them; reset --index cannot destroy promoted/cited result sets; findings live in the existing candidate->judge lifecycle; the twelve recursive-loop failure modes recorded in child beads have guards. Grain mismatch in composition fails closed.

## Static mechanism / likely defect

Issue description localizes the mechanism: THE convergent frontier from the 2026-07-05 R&D program (hit independently by swarm waves 2/4 and multiple GPT-Pro review branches). Today a query is a transient execution: nothing addressable survives it, so analyses cannot be iterative, citable, annotatable, or composable, and Polylogue cannot observe its own use. Target object graph: query:<hash> (content-addressed canonical plan) -> query_run (execution event) -> result_set (relation snapshot with grain+corpus epoch) -> finding (assertion-kind claim, judge lif…

## Source anchors to inspect first

- `polylogue/core/refs.py` — Existing ref model should be extended, not bypassed.
- `polylogue/storage/sqlite/archive_tiers/user_write.py:901` — Findings should reuse assertion/candidate lifecycle.
- `polylogue/surfaces/payloads.py:747` — Surface payloads currently expose branch/message refs; expand consistently.
- `polylogue/mcp/payloads.py:377` — MCP message payloads carry variant/branch metadata today.
- `polylogue/archive/query/spec.py:440` — SessionQuerySpec is the core read selection object.
- `polylogue/archive/query/spec.py:498` — from_params centralizes parameter lowering.
- `polylogue/cli/query_contracts.py:70` — QueryOutputSpec is current output/render contract seam.
- `polylogue/cli/archive_query.py:164` — CLI archive query execution remains a dense orchestration point.
- `polylogue/daemon/http.py:1734` — Daemon split-archive route already has from_params path for some cases.
- `polylogue/daemon/http.py:3296` — Another daemon path builds SessionQuerySpec from raw query params.
- `polylogue/mcp/query_contracts.py:54` — MCP query normalization has a distinct lowering path.
- `README.md` — Public claims should be grounded through the claims ledger.
- `docs/agent-forensics.md` — Existing forensics docs are a pattern for proof artifacts.
- `docs/demo.md` — Demo docs should depend on evidence/citation machinery.

## Implementation plan

1. Inventory open child beads and map them to the invariant named by the epic.
2. Add/verify a terminal acceptance checklist for the epic rather than landing broad code.
3. Close only after child beads are closed or explicitly split out with new blockers.

## Tests to add

- Acceptance proof: A committed query returns stable query/query-run/result-set refs on every surface
- Acceptance proof: assertions can target them
- Acceptance proof: reset --index cannot destroy promoted/cited result sets
- Acceptance proof: findings live in the existing candidate->judge lifecycle
- Acceptance proof: the twelve recursive-loop failure modes recorded in child beads have guards.
- Acceptance proof: Grain mismatch in composition fails closed.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools test tests/unit tests/integration -k "query or contract or citation or insight"`

## Pitfalls

- Do not implement the epic as a single broad PR; use it to close/split child work and verify the terminal invariant.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.

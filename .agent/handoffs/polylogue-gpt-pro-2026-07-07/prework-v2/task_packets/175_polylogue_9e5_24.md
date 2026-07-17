# 175. polylogue-9e5.24 — Sink MCP analysis primitives into insights/ + api facade; delete surface-side math

Priority/type/status: **P2 / task / open**. Lane: **06-agent-context-coordination**. Release: **D-agent-context-coordination**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

_No description in export._

## Existing design note

server_insight_tools.py implements analysis math directly in the MCP surface, unreachable from CLI/library: correlate_sessions (:826 Pearson r + metric-name->field map), find_similar_sessions metadata lane (:652 weighted heuristic), aggregate_sessions/workflow_shape_distribution/find_abandoned_sessions (:510/:233/:298 GROUP-BY + severity-rank + ISO-week), tool_call_latency_distribution (:131 nearest-rank percentile), compare_sessions (:559 per-key set diff). Move each into insights/ (archive_rollups.py owns aggregate reducers; portfolio.py _distribution/DistributionStat is the canonical percentile; metadata similarity beside SessionNeighborCandidate) and expose via api/insights.py so MCP, CLI, and the library share one definition. This is the read/execution half split out from the 9e5.16 parity AUDIT (which stays read-only per the 9e5 rule).

## Acceptance criteria

correlate/find_similar-metadata/aggregate/workflow_shape/find_abandoned/tool_call_latency/compare have api facade methods and the MCP tools call them (grep shows no math/GROUP-BY left in server_insight_tools.py); the severity map, similarity weights, and week-bucketing are defined once in insights/; a CLI or library caller produces byte-identical aggregates to the MCP tool for a fixture archive; devtools verify green. Cross-refs polylogue-9e5.16.

## Static mechanism / likely defect

Design direction: server_insight_tools.py implements analysis math directly in the MCP surface, unreachable from CLI/library: correlate_sessions (:826 Pearson r + metric-name->field map), find_similar_sessions metadata lane (:652 weighted heuristic), aggregate_sessions/workflow_shape_distribution/find_abandoned_sessions (:510/:233/:298 GROUP-BY + severity-rank + ISO-week), tool_call_latency_distribution (:131 nearest-rank percentile)…

## Source anchors to inspect first

- `polylogue/insights/audit.py:173` — build_insight_rigor_audit_report is the audit entry point.
- `polylogue/insights/audit.py:194` — Current code iterates list_rigor_contracts, not the product registry.
- `polylogue/insights/audit.py:216` — Registry lookup is secondary and skipped for products without contracts.
- `polylogue/insights/rigor.py:85` — _RIGOR_MATRIX declares only a subset of registered products.
- `polylogue/insights/registry.py:294` — INSIGHT_REGISTRY is the universe the audit should iterate.
- `polylogue/archive/actions/followup.py` — Action/followup classification is a real structural analytics input.
- `polylogue/archive/actions/fields.py` — Action fields determine what can be measured without prose heuristics.
- `polylogue/insights/registry.py:294` — Insight registry should become measure/product registry input.
- `scripts/agent_forensics.py` — Existing forensics script is a proof artifact and candidate product surface.

## Implementation plan

1. server_insight_tools.py implements analysis math directly in the MCP surface, unreachable from CLI/library: correlate_sessions (:826 Pearson r + metric-name->field map), find_similar_sessions metadata lane (:652 weighted heuristic), aggregate_sessions/workflow_shape_distribution/find_abandoned_sessions (:510/:233/:298 GROUP-BY + severity-rank + ISO-week), tool_call_latency_distribution (:131 nearest-rank percentil…
2. Move each into insights/ (archive_rollups.py owns aggregate reducers
3. portfolio.py _distribution/DistributionStat is the canonical percentile
4. metadata similarity beside SessionNeighborCandidate) and expose via api/insights.py so MCP, CLI, and the library share one definition.
5. This is the read/execution half split out from the 9e5.16 parity AUDIT (which stays read-only per the 9e5 rule).

## Tests to add

- Acceptance proof: correlate/find_similar-metadata/aggregate/workflow_shape/find_abandoned/tool_call_latency/compare have api facade methods and the MCP tools call them (grep shows no math/GROUP-BY left in server_insight_tools.py)
- Acceptance proof: the severity map, similarity weights, and week-bucketing are defined once in insights/
- Acceptance proof: a CLI or library caller produces byte-identical aggregates to the MCP tool for a fixture archive
- Acceptance proof: devtools verify green.
- Acceptance proof: Cross-refs polylogue-9e5.16.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools test tests/unit/coordination tests/unit/mcp -q`

## Pitfalls

- Do not add direct context injection or trusted memory writes outside the scheduler/candidate policy.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.

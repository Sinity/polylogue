# 182. polylogue-9l5.6 — tool-episodes projection: call + result + outcome + context + next action

Priority/type/status: **P2 / feature / open**. Lane: **10-analytics-experiments**. Release: **I-analytics-experiments**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

Sidecar research (Sartre): affordance-usage and analyze tools stop at aggregate evidence. A first-class tool-episodes projection — tool call, paired result, outcome status, surrounding context, what the agent did next, caveats — supports Serena/codebase-memory utility evaluation and is the natural drill-down unit under every aggregate. Likely reuses the action outcome fields + followup_class machinery from the campaign.

## Existing design note

New derived read model `tool_episodes` (rebuildable; registry pattern under polylogue/storage/insights/session/, registered in insights/registry.py so CLI+MCP inherit it). Each episode joins a tool_use block to its paired tool_result via the existing `actions` view and carries: the keystone structural outcome fields (tool_result_is_error, tool_result_exit_code, index schema v16), a bounded surrounding-context window (prev/next K messages), followup_class (from the closed sru.1 keystone), and per-episode caveats (unknown-outcome NULL vs structural). Surfaces: (a) an `analyze` drill-down projection, (b) a DSL `tool-episodes` unit that is the natural drill-down under affordance-usage / analyze-tools aggregates, (c) an MCP tool. Aggregates OVER episodes register as MeasureSpecs via 9l5.7; the projection itself is a unit, not a measure. Pitfall: a tool_use with no paired result (interrupted/streamed) must still yield exactly one episode with outcome=unknown — never dropped.

## Acceptance criteria

1. On the seeded/demo corpus `tool-episodes` is queryable and each row carries call + paired result + structural outcome (is_error/exit_code) + surrounding-context window + next-action + caveat. 2. A drill-down from an affordance-usage (or analyze-tools) aggregate cell returns exactly the underlying episodes for that cell. 3. Property test: every tool_use block resolves to exactly one episode (paired or unknown-outcome), zero dropped. 4. Aggregates over episodes register through 9l5.7 (so tier footnotes render). Verify: `polylogue analyze tools` drill-down and a DSL `... | tool-episodes` query both render on the demo archive; the MCP tool returns the same rows; `devtools test` selection over the new insight + registry passes.

## Static mechanism / likely defect

Issue description localizes the mechanism: Sidecar research (Sartre): affordance-usage and analyze tools stop at aggregate evidence. A first-class tool-episodes projection — tool call, paired result, outcome status, surrounding context, what the agent did next, caveats — supports Serena/codebase-memory utility evaluation and is the natural drill-down unit under every aggregate. Likely reuses the action outcome fields + followup_class machinery from the campaign. Design direction: New derived read model `tool_episodes` (rebuildable; registry pattern under polylogue/storage/insights/session/, registered in insights/registry.py so CLI+MCP inherit it). Each episode joins a tool_use block to its paired tool_result via the existing `actions` view and carries: the keystone structural outcome fields (tool_result_is_error, tool_result_exit_code, index schema v16), a bounded surrounding-context window…

## Source anchors to inspect first

- `polylogue/archive/actions/followup.py` — Action/followup classification is a real structural analytics input.
- `polylogue/archive/actions/fields.py` — Action fields determine what can be measured without prose heuristics.
- `polylogue/insights/registry.py:294` — Insight registry should become measure/product registry input.
- `scripts/agent_forensics.py` — Existing forensics script is a proof artifact and candidate product surface.
- `polylogue/core/refs.py` — Existing ref model should be extended, not bypassed.
- `polylogue/storage/sqlite/archive_tiers/user_write.py:901` — Findings should reuse assertion/candidate lifecycle.
- `polylogue/surfaces/payloads.py:747` — Surface payloads currently expose branch/message refs; expand consistently.
- `polylogue/mcp/payloads.py:377` — MCP message payloads carry variant/branch metadata today.

## Implementation plan

1. New derived read model `tool_episodes` (rebuildable
2. registry pattern under polylogue/storage/insights/session/, registered in insights/registry.py so CLI+MCP inherit it).
3. Each episode joins a tool_use block to its paired tool_result via the existing `actions` view and carries: the keystone structural outcome fields (tool_result_is_error, tool_result_exit_code, index schema v16), a bounded surrounding-context window (prev/next K messages), followup_class (from the closed sru.1 keystone), and per-episode caveats (unknown-outcome NULL vs structural).
4. Surfaces: (a) an `analyze` drill-down projection, (b) a DSL `tool-episodes` unit that is the natural drill-down under affordance-usage / analyze-tools aggregates, (c) an MCP tool.
5. Aggregates OVER episodes register as MeasureSpecs via 9l5.7
6. the projection itself is a unit, not a measure.
7. Pitfall: a tool_use with no paired result (interrupted/streamed) must still yield exactly one episode with outcome=unknown — never dropped.

## Tests to add

- Acceptance proof: 1.
- Acceptance proof: On the seeded/demo corpus `tool-episodes` is queryable and each row carries call + paired result + structural outcome (is_error/exit_code) + surrounding-context window + next-action + caveat.
- Acceptance proof: 2.
- Acceptance proof: A drill-down from an affordance-usage (or analyze-tools) aggregate cell returns exactly the underlying episodes for that cell.
- Acceptance proof: 3.
- Acceptance proof: Property test: every tool_use block resolves to exactly one episode (paired or unknown-outcome), zero dropped.
- Acceptance proof: 4.
- Acceptance proof: Aggregates over episodes register through 9l5.7 (so tier footnotes render).

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools test tests/unit tests/integration -k "query or contract or citation or insight"`

## Pitfalls

- Do not publish a number or success claim until evidence tier, denominator, and caveat rendering exist.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.

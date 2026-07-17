# 180. polylogue-9l5.1 — Outcome-conditioned analytics: cost/duration/retries/tools by structural success

Priority/type/status: **P2 / feature / open**. Lane: **10-analytics-experiments**. Release: **I-analytics-experiments**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

Group cost, duration, retry chains, and tool usage by structural outcome (exit_code/is_error terminal state), with per-origin coverage caveats. The includes High-Value backlog names this directly. Consumes the action outcome fields; surfaces through analyze projections + DSL aggregates + MCP insight tools — one relation, three surfaces.

## Existing design note

Anchored examples (all one step from existing substrate): cost of failed vs clean sessions; failure-rate by model VERSION; retry cascade depth; 'sessions where >30% of tool calls errored' (needs the child-count DSL predicate or the relation directly). Keystone fields tool_result_is_error/exit_code + the actions view are the ground truth; outcomes are captured today but analytics still mostly counts and sums. This is the highest-leverage analytics move precisely because it is the construct-valid one: success measured from provider-reported structure, never assistant prose. Per-origin coverage caveats from the column-honesty audit bead; surfaces = analyze projections + DSL aggregates + MCP insight tools over ONE shared relation.

## Acceptance criteria

1. One shared relation groups cost, duration, retry-chain depth, and tool-mix by structural outcome (terminal tool_result_is_error / exit_code from the actions view — never assistant prose), reachable identically through the analyze projection, a DSL aggregate, and an MCP insight tool (one relation, three surfaces returning the same numbers). 2. A per-origin coverage caveat (from the 9e5.3 column-honesty audit) renders on every grouped row. 3. The predicate 'sessions where >30% of tool calls errored' resolves on the seeded corpus. Verify: `polylogue analyze <outcome-view>`, the equivalent DSL aggregate, and the MCP call each return identical figures on the demo archive; a snapshot test pins the coverage caveat and the >30%-errored predicate result.

## Static mechanism / likely defect

Issue description localizes the mechanism: Group cost, duration, retry chains, and tool usage by structural outcome (exit_code/is_error terminal state), with per-origin coverage caveats. The includes High-Value backlog names this directly. Consumes the action outcome fields; surfaces through analyze projections + DSL aggregates + MCP insight tools — one relation, three surfaces. Design direction: Anchored examples (all one step from existing substrate): cost of failed vs clean sessions; failure-rate by model VERSION; retry cascade depth; 'sessions where >30% of tool calls errored' (needs the child-count DSL predicate or the relation directly). Keystone fields tool_result_is_error/exit_code + the actions view are the ground truth; outcomes are captured today but analytics still mostly counts and sums. This is…

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

1. Anchored examples (all one step from existing substrate): cost of failed vs clean sessions
2. failure-rate by model VERSION
3. retry cascade depth
4. 'sessions where >30% of tool calls errored' (needs the child-count DSL predicate or the relation directly).
5. Keystone fields tool_result_is_error/exit_code + the actions view are the ground truth
6. outcomes are captured today but analytics still mostly counts and sums.
7. This is the highest-leverage analytics move precisely because it is the construct-valid one: success measured from provider-reported structure, never assistant prose.

## Tests to add

- Acceptance proof: 1.
- Acceptance proof: One shared relation groups cost, duration, retry-chain depth, and tool-mix by structural outcome (terminal tool_result_is_error / exit_code from the actions view — never assistant prose), reachable identically through the analyze projection, a DSL aggregate, and an MCP insight tool (one relation, three surfaces returning the same numbers).
- Acceptance proof: 2.
- Acceptance proof: A per-origin coverage caveat (from the 9e5.3 column-honesty audit) renders on every grouped row.
- Acceptance proof: 3.
- Acceptance proof: The predicate 'sessions where >30% of tool calls errored' resolves on the seeded corpus.
- Acceptance proof: Verify: `polylogue analyze <outcome-view>`, the equivalent DSL aggregate, and the MCP call each return identical figures on the demo archive
- Acceptance proof: a snapshot test pins the coverage caveat and the >30%-errored predicate result.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools test tests/unit tests/integration -k "query or contract or citation or insight"`

## Pitfalls

- Do not publish a number or success claim until evidence tier, denominator, and caveat rendering exist.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.

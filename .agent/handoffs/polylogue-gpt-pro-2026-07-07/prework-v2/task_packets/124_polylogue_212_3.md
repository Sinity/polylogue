# 124. polylogue-212.3 — D2 'Where did the money actually go': cost by outcome

Priority/type/status: **P2 / task / open**. Lane: **12-external-legibility-demos**. Release: **L-legibility-demos**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

Five-axis cost basis shown honestly (provider-reported exact vs catalog-priced with stated coverage), then the pivot nobody else can do: cost by outcome — '$N this month; X% spent in sessions that ended abandoned or with a failing final action; five most expensive failures, click through to the exact turn.' Needs the outcome-conditioned join (action outcome fields bead); instruments otherwise exist (cost_rollups, session_costs, terminal-state profiles, per-origin exact/estimate labels rendered as footnotes).

## Existing design note

A demo: show the five-axis cost basis honestly (provider-reported exact vs catalog-priced with stated coverage), then the pivot no chat UI can do, cost by outcome: total monthly spend, the % spent in sessions that ended abandoned or with a failing final action, and the five most expensive failures each drillable to the exact turn. Needs the outcome-conditioned join (action outcome fields bead); cost instruments exist (cost_rollups, session_costs, terminal-state profiles, per-origin exact/estimate labels rendered as footnotes).

## Acceptance criteria

1. The demo renders a five-axis cost basis with provider-reported-exact vs catalog-priced values clearly labeled and coverage stated (per-origin exact/estimate footnotes). 2. Cost-by-outcome pivot: total monthly spend, the fraction spent in abandoned or failing-final-action sessions, and the five most expensive failures, each drillable to the exact turn via the outcome-conditioned join. Verify: the demo runs via cost_rollups/session_costs against the seeded corpus (recorded output); depends on the action-outcome join bead (note dependency); `devtools test` selection covers the join query if new.

## Static mechanism / likely defect

Issue description localizes the mechanism: Five-axis cost basis shown honestly (provider-reported exact vs catalog-priced with stated coverage), then the pivot nobody else can do: cost by outcome — '$N this month; X% spent in sessions that ended abandoned or with a failing final action; five most expensive failures, click through to the exact turn.' Needs the outcome-conditioned join (action outcome fields bead); instruments otherwise exist (cost_rollups, session_costs, terminal-state profiles, per-origin exact/estimate labels rendered as footnotes). Design direction: A demo: show the five-axis cost basis honestly (provider-reported exact vs catalog-priced with stated coverage), then the pivot no chat UI can do, cost by outcome: total monthly spend, the % spent in sessions that ended abandoned or with a failing final action, and the five most expensive failures each drillable to the exact turn. Needs the outcome-conditioned join (action outcome fields bead); cost instruments exis…

## Source anchors to inspect first

- `README.md` — Public claims should be grounded through the claims ledger.
- `docs/agent-forensics.md` — Existing forensics docs are a pattern for proof artifacts.
- `docs/demo.md` — Demo docs should depend on evidence/citation machinery.
- `polylogue/core/refs.py` — Existing ref model should be extended, not bypassed.
- `polylogue/storage/sqlite/archive_tiers/user_write.py:901` — Findings should reuse assertion/candidate lifecycle.
- `polylogue/surfaces/payloads.py:747` — Surface payloads currently expose branch/message refs; expand consistently.
- `polylogue/mcp/payloads.py:377` — MCP message payloads carry variant/branch metadata today.
- `polylogue/archive/actions/followup.py` — Action/followup classification is a real structural analytics input.
- `polylogue/archive/actions/fields.py` — Action fields determine what can be measured without prose heuristics.
- `polylogue/insights/registry.py:294` — Insight registry should become measure/product registry input.
- `scripts/agent_forensics.py` — Existing forensics script is a proof artifact and candidate product surface.

## Implementation plan

1. A demo: show the five-axis cost basis honestly (provider-reported exact vs catalog-priced with stated coverage), then the pivot no chat UI can do, cost by outcome: total monthly spend, the % spent in sessions that ended abandoned or with a failing final action, and the five most expensive failures each drillable to the exact turn.
2. Needs the outcome-conditioned join (action outcome fields bead)
3. cost instruments exist (cost_rollups, session_costs, terminal-state profiles, per-origin exact/estimate labels rendered as footnotes).

## Tests to add

- Acceptance proof: 1.
- Acceptance proof: The demo renders a five-axis cost basis with provider-reported-exact vs catalog-priced values clearly labeled and coverage stated (per-origin exact/estimate footnotes).
- Acceptance proof: 2.
- Acceptance proof: Cost-by-outcome pivot: total monthly spend, the fraction spent in abandoned or failing-final-action sessions, and the five most expensive failures, each drillable to the exact turn via the outcome-conditioned join.
- Acceptance proof: Verify: the demo runs via cost_rollups/session_costs against the seeded corpus (recorded output)
- Acceptance proof: depends on the action-outcome join bead (note dependency)
- Acceptance proof: `devtools test` selection covers the join query if new.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools verify --docs --demo`

## Pitfalls

- Do not publish a number or success claim until evidence tier, denominator, and caveat rendering exist.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.

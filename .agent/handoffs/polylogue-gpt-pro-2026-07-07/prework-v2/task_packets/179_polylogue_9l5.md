# 179. polylogue-9l5 — Outcome-grounded analytics: the archive answers 'so what' questions

Priority/type/status: **P2 / epic / open**. Lane: **10-analytics-experiments**. Release: **I-analytics-experiments**. Readiness: **epic-needs-child-closure**.

## What the bead says

The archive answers 'so what' questions. Tower map (2026-07-03 design pass): LAYER 0 substrate (exists) — profiles, work events, phases, threads, cost rollups with five-axis accounting, structural pathologies, followup_class, run projection, topology/logical sessions, tool timing, workflow shapes. LAYER 1 descriptive (children .1-.6): outcome-conditioned, cross-provider, epidemiology, token economy, saved views, tool episodes. LAYER 2 statistical honesty (.7): uncertainty primitives + the measure registry with construct-validity metadata — the keystone every higher layer composes through. LAYER 3 temporal (.8): trends, baselines, changepoints. LAYER 4 duration & sequence (.9 survival, .10 process mining). LAYER 5 causal (experiment hosting bead): declared arms, prereg, paired analysis. LAYER 6 predictive (.11): calibrated classical models as advisories. Plus cross-cutting measures (.12 information-theoretic + graph) and the semantic layer (mhx.5 topics/novelty). COMPOSITION RULE: every layer lands as registered measures over the query algebra (fnm/4p1) — measure x grouping x window x comparison x uncertainty — never as bespoke analyze modes; construct validity is enforced by the registry (evidence tier + sample frame + confounds declared per measure, coverage preconditions checked at composition, tier footnotes rendered in every output).

## Existing design note

Epic: the archive answers 'so what' questions, layered over the Layer-0 substrate (profiles, work events, phases, threads, five-axis cost rollups, structural pathologies, followup_class, run projection, topology/logical sessions, tool timing, workflow shapes). Layer 1 descriptive (children .1-.6), Layer 2 statistical honesty (.7 uncertainty primitives + measure registry with construct-validity metadata, the keystone), Layer 3 temporal (.8), Layer 4 duration/sequence (.9 survival, .10 process mining), Layer 5 causal (experiment hosting), Layer 6 predictive (.11), plus cross-cutting measures (.12) and the semantic layer (mhx.5). Composition rule: every layer lands as registered measures over the query algebra (fnm/4p1), measure x grouping x window x comparison x uncertainty, never as bespoke analyze modes; construct validity is enforced by the registry.

## Acceptance criteria

1. All child beads (9l5.1-.12 and folded-in measures) are closed (`bd show polylogue-9l5 --json` shows no open children). 2. Every delivered analytic lands as a registered measure over the query algebra (fnm/4p1), not a bespoke analyze mode; the measure registry (.7) enforces evidence tier + sample frame + confounds per measure and renders tier footnotes in every output. 3. The keystone statistical-honesty layer (.7) is in place before higher layers compose through it, and coverage preconditions are checked at composition time. Verify: `bd show polylogue-9l5 --json` children closed; `devtools test` selection on the measure registry asserts construct-validity metadata is required per measure and that outputs carry tier footnotes.

## Static mechanism / likely defect

Issue description localizes the mechanism: The archive answers 'so what' questions. Tower map (2026-07-03 design pass): LAYER 0 substrate (exists) — profiles, work events, phases, threads, cost rollups with five-axis accounting, structural pathologies, followup_class, run projection, topology/logical sessions, tool timing, workflow shapes. LAYER 1 descriptive (children .1-.6): outcome-conditioned, cross-provider, epidemiology, token economy, saved views, tool episodes. LAYER 2 statistical honesty (.7): uncertainty primitives + the measure registry with con… Design direction: Epic: the archive answers 'so what' questions, layered over the Layer-0 substrate (profiles, work events, phases, threads, five-axis cost rollups, structural pathologies, followup_class, run projection, topology/logical sessions, tool timing, workflow shapes). Layer 1 descriptive (children .1-.6), Layer 2 statistical honesty (.7 uncertainty primitives + measure registry with construct-validity metadata, the keystone…

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

1. Epic: the archive answers 'so what' questions, layered over the Layer-0 substrate (profiles, work events, phases, threads, five-axis cost rollups, structural pathologies, followup_class, run projection, topology/logical sessions, tool timing, workflow shapes).
2. Layer 1 descriptive (children .1-.6), Layer 2 statistical honesty (.7 uncertainty primitives + measure registry with construct-validity metadata, the keystone), Layer 3 temporal (.8), Layer 4 duration/sequence (.9 survival, .10 process mining), Layer 5 causal (experiment hosting), Layer 6 predictive (.11), plus cross-cutting measures (.12) and the semantic layer (mhx.5).
3. Composition rule: every layer lands as registered measures over the query algebra (fnm/4p1), measure x grouping x window x comparison x uncertainty, never as bespoke analyze modes
4. construct validity is enforced by the registry.

## Tests to add

- Acceptance proof: 1.
- Acceptance proof: All child beads (9l5.1-.12 and folded-in measures) are closed (`bd show polylogue-9l5 --json` shows no open children).
- Acceptance proof: 2.
- Acceptance proof: Every delivered analytic lands as a registered measure over the query algebra (fnm/4p1), not a bespoke analyze mode
- Acceptance proof: the measure registry (.7) enforces evidence tier + sample frame + confounds per measure and renders tier footnotes in every output.
- Acceptance proof: 3.
- Acceptance proof: The keystone statistical-honesty layer (.7) is in place before higher layers compose through it, and coverage preconditions are checked at composition time.
- Acceptance proof: Verify: `bd show polylogue-9l5 --json` children closed

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools test tests/unit tests/integration -k "query or contract or citation or insight"`

## Pitfalls

- Do not implement the epic as a single broad PR; use it to close/split child work and verify the terminal invariant.
- Do not publish a number or success claim until evidence tier, denominator, and caveat rendering exist.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.

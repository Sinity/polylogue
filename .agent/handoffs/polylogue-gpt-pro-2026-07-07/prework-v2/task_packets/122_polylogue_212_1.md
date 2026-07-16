# 122. polylogue-212.1 — Post-hoc forensic Q&A demo: questions a tracer cannot answer

Priority/type/status: **P2 / task / open**. Lane: **12-external-legibility-demos**. Release: **L-legibility-demos**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

The category-separation demo: take one completed multi-hour coding-agent session and answer post-hoc questions live — when did the bad assumption first enter; which file churned before the regression; what evidence did the agent cite for a design choice; which prior failed attempts resemble today's failure. Composes existing reads (postmortem bundle, work events, phases, neighbor candidates, git correlation); packaging is the work, plus one honest 'we cannot answer X' slide (construct validity).

## Existing design note

A category-separation demo: take one completed multi-hour coding-agent session and answer post-hoc questions live, when the bad assumption first entered; which file churned before the regression; what evidence the agent cited for a design choice; which prior failed attempts resemble today's. Composes existing reads (postmortem bundle, work events, phases, neighbor candidates, git correlation); packaging is the work, plus one honest 'we cannot answer X' slide for construct validity.

## Acceptance criteria

1. Against one completed multi-hour session, the demo answers each forensic question live using existing reads (get_postmortem_bundle, session_work_events, session_phases, neighbor_candidates, git correlation): first-bad-assumption entry, file churned before the regression, cited evidence for a design choice, and resembling prior failed attempts. 2. One explicit 'we cannot answer X' slide is included (construct-validity honesty). Verify: the demo runs end-to-end against a chosen archived session (recorded output/artifact) using only existing reads (no new query machinery).

## Static mechanism / likely defect

Issue description localizes the mechanism: The category-separation demo: take one completed multi-hour coding-agent session and answer post-hoc questions live — when did the bad assumption first enter; which file churned before the regression; what evidence did the agent cite for a design choice; which prior failed attempts resemble today's failure. Composes existing reads (postmortem bundle, work events, phases, neighbor candidates, git correlation); packaging is the work, plus one honest 'we cannot answer X' slide (construct validity). Design direction: A category-separation demo: take one completed multi-hour coding-agent session and answer post-hoc questions live, when the bad assumption first entered; which file churned before the regression; what evidence the agent cited for a design choice; which prior failed attempts resemble today's. Composes existing reads (postmortem bundle, work events, phases, neighbor candidates, git correlation); packaging is the work,…

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

1. A category-separation demo: take one completed multi-hour coding-agent session and answer post-hoc questions live, when the bad assumption first entered
2. which file churned before the regression
3. what evidence the agent cited for a design choice
4. which prior failed attempts resemble today's.
5. Composes existing reads (postmortem bundle, work events, phases, neighbor candidates, git correlation)
6. packaging is the work, plus one honest 'we cannot answer X' slide for construct validity.

## Tests to add

- Acceptance proof: 1.
- Acceptance proof: Against one completed multi-hour session, the demo answers each forensic question live using existing reads (get_postmortem_bundle, session_work_events, session_phases, neighbor_candidates, git correlation): first-bad-assumption entry, file churned before the regression, cited evidence for a design choice, and resembling prior failed attempts.
- Acceptance proof: 2.
- Acceptance proof: One explicit 'we cannot answer X' slide is included (construct-validity honesty).
- Acceptance proof: Verify: the demo runs end-to-end against a chosen archived session (recorded output/artifact) using only existing reads (no new query machinery).

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools verify --docs --demo`

## Pitfalls

- Do not publish a number or success claim until evidence tier, denominator, and caveat rendering exist.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.

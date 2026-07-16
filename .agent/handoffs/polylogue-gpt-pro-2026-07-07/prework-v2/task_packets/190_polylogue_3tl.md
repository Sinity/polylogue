# 190. polylogue-3tl — External legibility: a stranger can understand, run, and cite Polylogue

Priority/type/status: **P1 / epic / open**. Lane: **12-external-legibility-demos**. Release: **L-legibility-demos**. Readiness: **epic-needs-child-closure**.

## What the bead says

Every finished artifact proves the substrate is honest; this program makes the project legible to someone with no context. The gap is weeks, not months (fables positioning analysis): the value exists but is illegible from outside. Core diagnosis: the problem is category anchoring, not absence of explanation — name the category ('the system of record for AI work') rather than borrowing chat-viewer/observability/memory/QS buckets that all mis-frame it. Deliverable set: README rewrite around the named category and four verbs (search/analyze/audit/remember), one-command demo, two published evidence artifacts, two recordings, findings with URLs. Discipline: capability-phrased memory claims until the uplift re-run (polylogue-cfk) reports; it, published with its data, is the natural launch post.

## Acceptance criteria

Terminal state: a stranger can (1) understand from the README's first screen, (2) run the one-command demo successfully, (3) cite a published finding URL. All three verified by a cold-reader pass from someone/something with no project context.

## Static mechanism / likely defect

Issue description localizes the mechanism: Every finished artifact proves the substrate is honest; this program makes the project legible to someone with no context. The gap is weeks, not months (fables positioning analysis): the value exists but is illegible from outside. Core diagnosis: the problem is category anchoring, not absence of explanation — name the category ('the system of record for AI work') rather than borrowing chat-viewer/observability/memory/QS buckets that all mis-frame it. Deliverable set: README rewrite around the named category and fo…

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

1. Inventory open child beads and map them to the invariant named by the epic.
2. Add/verify a terminal acceptance checklist for the epic rather than landing broad code.
3. Close only after child beads are closed or explicitly split out with new blockers.

## Tests to add

- Acceptance proof: Terminal state: a stranger can (1) understand from the README's first screen, (2) run the one-command demo successfully, (3) cite a published finding URL.
- Acceptance proof: All three verified by a cold-reader pass from someone/something with no project context.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools verify --docs --demo`

## Pitfalls

- Do not implement the epic as a single broad PR; use it to close/split child work and verify the terminal invariant.
- Do not publish a number or success claim until evidence tier, denominator, and caveat rendering exist.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.

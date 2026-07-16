# 127. polylogue-3tl.13 — Reconcile schema-versioning docs + retire superseded execution-plan.md

Priority/type/status: **P2 / task / open**. Lane: **12-external-legibility-demos**. Release: **L-legibility-demos**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

_No description in export._

## Existing design note

architecture-spine.md:34-37 lists 'in-place upgrade chains' as Rejected with no durable-additive carve-out, contradicting the shipped migrations and internals.md's own two-regime text (internals.md:284-289 is internally inconsistent). docs/execution-plan.md is fully superseded (dropped #1807 umbrella; every issue re-encoded as a bead) yet README.md:14 still calls it 'current sequencing plan'. Fix the spine section, reconcile internals.md, retire execution-plan.md with a pointer to Beads, and repoint README:14.

## Acceptance criteria

architecture-spine + internals schema-versioning sections describe the two-regime model consistently; execution-plan.md is archived/removed and no doc calls it current; README points at Beads. Verify: render docs-surface --check + grep 'execution-plan' docs README.

## Static mechanism / likely defect

Design direction: architecture-spine.md:34-37 lists 'in-place upgrade chains' as Rejected with no durable-additive carve-out, contradicting the shipped migrations and internals.md's own two-regime text (internals.md:284-289 is internally inconsistent). docs/execution-plan.md is fully superseded (dropped #1807 umbrella; every issue re-encoded as a bead) yet README.md:14 still calls it 'current sequencing plan'. Fix the spine section, …

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

1. architecture-spine.md:34-37 lists 'in-place upgrade chains' as Rejected with no durable-additive carve-out, contradicting the shipped migrations and internals.md's own two-regime text (internals.md:284-289 is internally inconsistent).
2. docs/execution-plan.md is fully superseded (dropped #1807 umbrella
3. every issue re-encoded as a bead) yet README.md:14 still calls it 'current sequencing plan'.
4. Fix the spine section, reconcile internals.md, retire execution-plan.md with a pointer to Beads, and repoint README:14.

## Tests to add

- Acceptance proof: architecture-spine + internals schema-versioning sections describe the two-regime model consistently
- Acceptance proof: execution-plan.md is archived/removed and no doc calls it current
- Acceptance proof: README points at Beads.
- Acceptance proof: Verify: render docs-surface --check + grep 'execution-plan' docs README.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools verify --docs --demo`

## Pitfalls

- Do not publish a number or success claim until evidence tier, denominator, and caveat rendering exist.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.

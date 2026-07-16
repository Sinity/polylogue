# 126. polylogue-3tl.12 — README de-meta / de-persuasion pass with reproducible capability claims

Priority/type/status: **P2 / task / open**. Lane: **12-external-legibility-demos**. Release: **L-legibility-demos**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

_No description in export._

## Existing design note

Raw-log 2026-07-04 18:16-18:21 (post-dates the closed 3tl.1 skim-ladder rewrite): strip the meta/persuasion register from the README, define agent-coined terms (judged notes, work phases, logical session) on first use, and make each capability claim reproducible on the operator's own archive (a command the reader can run). Distinct axis from 3tl.1's structure work.

## Acceptance criteria

README first screen names the category and four verbs without persuasion register; every coined term is defined at first use; each capability claim links a runnable `polylogue`/`devtools` command; a fresh no-context reader can reproduce >=2 claims. Verify: docs-commands lint green + cold-reader pass.

## Static mechanism / likely defect

Design direction: Raw-log 2026-07-04 18:16-18:21 (post-dates the closed 3tl.1 skim-ladder rewrite): strip the meta/persuasion register from the README, define agent-coined terms (judged notes, work phases, logical session) on first use, and make each capability claim reproducible on the operator's own archive (a command the reader can run). Distinct axis from 3tl.1's structure work.

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

1. Raw-log 2026-07-04 18:16-18:21 (post-dates the closed 3tl.1 skim-ladder rewrite): strip the meta/persuasion register from the README, define agent-coined terms (judged notes, work phases, logical session) on first use, and make each capability claim reproducible on the operator's own archive (a command the reader can run).
2. Distinct axis from 3tl.1's structure work.

## Tests to add

- Acceptance proof: README first screen names the category and four verbs without persuasion register
- Acceptance proof: every coined term is defined at first use
- Acceptance proof: each capability claim links a runnable `polylogue`/`devtools` command
- Acceptance proof: a fresh no-context reader can reproduce >=2 claims.
- Acceptance proof: Verify: docs-commands lint green + cold-reader pass.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools verify --docs --demo`

## Pitfalls

- Do not publish a number or success claim until evidence tier, denominator, and caveat rendering exist.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.

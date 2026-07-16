# 136. polylogue-9e5.27 — Speed up live affordance usage surface inventory

Priority/type/status: **P2 / task / open**. Lane: **02-usage-cost-honesty**. Release: **A-trust-floor**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

After switching the default family report and inventory counts away from action-row materialization, the live .agent/demos/agent-affordance-usage regeneration still took roughly 88 seconds on the full archive. The artifact is usable, but this is too slow for a polished demo/workspace command.

## Existing design note

Profile devtools workspace affordance-usage on the live archive with query-plan evidence. Likely targets: CLI command/path matching over generic tool-use rows, missing expression indexes for generated command/path fields, or repeated direct scans that should be a reusable product query primitive.

## Acceptance criteria

1. Capture query-plan/timing evidence for each major affordance-usage phase on /home/sinity/.local/share/polylogue. 2. Reduce default live regeneration to a materially faster target or document the exact storage/index bead needed. 3. Keep detail-pattern scans explicit and avoid reintroducing tool_input body scans into default reports.

## Static mechanism / likely defect

Issue description localizes the mechanism: After switching the default family report and inventory counts away from action-row materialization, the live .agent/demos/agent-affordance-usage regeneration still took roughly 88 seconds on the full archive. The artifact is usable, but this is too slow for a polished demo/workspace command. Design direction: Profile devtools workspace affordance-usage on the live archive with query-plan evidence. Likely targets: CLI command/path matching over generic tool-use rows, missing expression indexes for generated command/path fields, or repeated direct scans that should be a reusable product query primitive.

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

1. Profile devtools workspace affordance-usage on the live archive with query-plan evidence.
2. Likely targets: CLI command/path matching over generic tool-use rows, missing expression indexes for generated command/path fields, or repeated direct scans that should be a reusable product query primitive.

## Tests to add

- Acceptance proof: 1.
- Acceptance proof: Capture query-plan/timing evidence for each major affordance-usage phase on /home/sinity/.local/share/polylogue.
- Acceptance proof: 2.
- Acceptance proof: Reduce default live regeneration to a materially faster target or document the exact storage/index bead needed.
- Acceptance proof: 3.
- Acceptance proof: Keep detail-pattern scans explicit and avoid reintroducing tool_input body scans into default reports.

## Verification commands

- `devtools test <focused tests added for this bead>`

## Pitfalls

- Do not broaden scope beyond the bead acceptance criteria; make a failing test first, then patch the smallest shared seam.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.

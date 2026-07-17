# 134. polylogue-9e5.25 — Review zero-use MCP surfaces from affordance usage artifact

Priority/type/status: **P2 / task / open**. Lane: **02-usage-cost-honesty**. Release: **A-trust-floor**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

The current agent-affordance-usage demo classifies 59 MCP tools as zero captured agent use and non-operator surfaces. This is review input, not automatic deletion: use .agent/demos/agent-affordance-usage/surface-inventory.csv and affordance-usage.report.json to decide which surfaces should collapse into query/surface algebra, which need docs/examples, and which should be removed.

## Existing design note

Batch the review through contracts/surface-algebra rather than deleting isolated tools. Preserve operator-only caveats; verify each proposed removal or merge against the registered MCP tool set and actual consumers.

## Acceptance criteria

1. Every MCP kill-candidate row from .agent/demos/agent-affordance-usage/surface-inventory.csv is classified as remove / merge / keep / needs-demo with rationale. 2. Removal or merge work is split into executable beads with exact tool names and surface contracts. 3. No MCP surface is removed solely because this archive has zero captured agent use.

## Static mechanism / likely defect

Issue description localizes the mechanism: The current agent-affordance-usage demo classifies 59 MCP tools as zero captured agent use and non-operator surfaces. This is review input, not automatic deletion: use .agent/demos/agent-affordance-usage/surface-inventory.csv and affordance-usage.report.json to decide which surfaces should collapse into query/surface algebra, which need docs/examples, and which should be removed. Design direction: Batch the review through contracts/surface-algebra rather than deleting isolated tools. Preserve operator-only caveats; verify each proposed removal or merge against the registered MCP tool set and actual consumers.

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

1. Batch the review through contracts/surface-algebra rather than deleting isolated tools.
2. Preserve operator-only caveats
3. verify each proposed removal or merge against the registered MCP tool set and actual consumers.

## Tests to add

- Acceptance proof: 1.
- Acceptance proof: Every MCP kill-candidate row from .agent/demos/agent-affordance-usage/surface-inventory.csv is classified as remove / merge / keep / needs-demo with rationale.
- Acceptance proof: 2.
- Acceptance proof: Removal or merge work is split into executable beads with exact tool names and surface contracts.
- Acceptance proof: 3.
- Acceptance proof: No MCP surface is removed solely because this archive has zero captured agent use.

## Verification commands

- `devtools test <focused tests added for this bead>`

## Pitfalls

- Do not broaden scope beyond the bead acceptance criteria; make a failing test first, then patch the smallest shared seam.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.

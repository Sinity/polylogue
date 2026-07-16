# 008. polylogue-9e5.29 — Number-over-empty gates: quantitative fields need field-level evidence contracts

Priority/type/status: **P1 / bug / open**. Lane: **00-trust-floor**. Release: **A-trust-floor**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

Products can emit 0.0 (a number) when backing rows are empty/NULL — a rendered zero is a claim, and absent evidence must render as None/uncovered, never zero. Add field-level RigorFieldContract for number fields: provenance class, nullable_when_ungrounded, denominator/unit-frame, evidence tier. Deliberate byte-compat break for consumers expecting numeric zero — gate behind materializer-version bump. Distinguish three states everywhere: absent evidence / true zero / not-applicable.

## Existing design note

Anchor files: polylogue/insights/rigor.py (RigorContract ~L45, RigorVersionField ~L37), polylogue/insights/audit.py (insight_rigor_audit surface), polylogue/insights/confidence.py. Add a field-level RigorFieldContract: for each quantitative field of an insight payload declare provenance class (counted/derived/estimated), the evidence query or reducer that grounds it, and nullable_when_ungrounded=True so an empty backing frame renders None/uncovered — never 0.0. Wire: registry descriptors (insights/registry.py) declare field contracts; the rigor audit enumerates fields lacking contracts; renderers treat None as uncovered, not zero. Start with the worst offenders: any field the audit currently shows emitting 0.0 over empty rows. Pitfall from notes: field paths must resolve to block+json-path+reducer+denominator or the bytes-resolution product promise narrows to block granularity — fold that dimension into the contract design.

## Acceptance criteria

Property tests generate all-NULL rows and assert None/uncovered, never 0.0; every number-bearing contract declares denominator+provenance; a rendered insight cannot carry a quantitative claim over empty backing rows. Verify: hypothesis tests + audit report.

## Static mechanism / likely defect

Aggregate/report paths can render missing backing rows as zero. In this system every numeric field is a claim, so absence, not-applicable, uncovered, and true zero need distinct payload states.

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

1. Introduce a small `EvidenceNumber`/field-contract shape or equivalent metadata: value, unit, denominator, evidence_state, evidence_tier, provenance, nullable_when_ungrounded.
2. Apply to public insight products first: coverage, usage/cost, action outcomes, archive debt, rollups.
3. Change renderers to show unknown/uncovered/not-applicable explicitly; do not coerce null to `0` except where field contract proves true zero.
4. Add a registry audit that identifies numeric fields without evidence contracts.

## Tests to add

- Empty backing table report renders unknown/uncovered, not 0.
- Covered empty sample renders true zero with denominator and evidence_state=covered_zero.
- CLI/web/MCP JSON preserve the distinction.

## Verification commands

- ``devtools test tests/unit/insights/test_rigor_audit.py tests/unit/storage/test_usage*.py tests/unit/cli/test_*usage* -k 'empty or zero or uncovered or field_contract'``

## Pitfalls

- Do not broaden scope beyond the bead acceptance criteria; make a failing test first, then patch the smallest shared seam.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.

# 007. polylogue-9e5.28 — Rigor audit iterates contracts, not the registry: uncovered number-bearing products vanish from audit

Priority/type/status: **P1 / bug / open**. Lane: **00-trust-floor**. Release: **A-trust-floor**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

_RIGOR_MATRIX covers ~5 of 11 number-bearing insight products; audit.py iterates declared contracts, so a product with NO contract silently disappears from the audit instead of showing as uncovered — cost/coverage/tool/debt surfaces escape entirely. Fix: iterate INSIGHT_REGISTRY, emit coverage_status=uncovered rows for contract-less products, add RIGOR_EXEMPT with inline justification for genuinely non-number products, and make devtools lab policy insight-honesty fail on an uncovered number-bearing product.

## Acceptance criteria

One audit row per registered product or a justified exemption; monkeypatching a contract out yields uncovered, not omission; policy gate fails on uncovered number products. Verify: focused audit tests.

## Static mechanism / likely defect

The rigor audit iterates only `list_rigor_contracts()`, while the real product universe is `INSIGHT_REGISTRY`; number-bearing products without registered contracts vanish from the audit.

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

1. Make `INSIGHT_REGISTRY` the outer loop. For each product, join optional rigor contract, exemptions, and last materialization state.
2. Emit `uncovered` rows for products without a contract, with severity based on whether the product exposes numeric/user-visible claims.
3. Add an explicit exemption registry with owner, reason, expiry/review date.
4. Update CLI/docs to show covered/uncovered/exempt totals and fail the strict gate on uncovered number-bearing products.

## Tests to add

- Register a fake number-bearing insight with no contract; audit emits uncovered and strict mode fails.
- Register a fake non-claim product with exemption; audit shows exempt, does not fail until expiry.
- Existing five contracted products remain covered.

## Verification commands

- ``devtools test tests/unit/insights/test_rigor_audit.py -k 'registry or uncovered or exemption or rigor'``

## Pitfalls

- Do not broaden scope beyond the bead acceptance criteria; make a failing test first, then patch the smallest shared seam.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.

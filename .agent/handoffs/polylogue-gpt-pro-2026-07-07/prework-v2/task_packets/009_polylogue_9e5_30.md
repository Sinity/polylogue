# 009. polylogue-9e5.30 — Prose-mined forensic fields must carry text_derived provenance in the payload model

Priority/type/status: **P1 / bug / open**. Lane: **00-trust-floor**. Release: **A-trust-floor**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

transforms.py mines commit SHAs / decisions / caveats / test-pass counts from prose into forensic bundles while the no-regex-over-prose rule only structurally holds for the exit-code axis; the recovery-digest incident (#2482) fixed one renderer, not the type system. Add text_derived_fields / evidence_class markers to ToolSummary, DecisionCandidate, ForensicIndexEntry and successors — payload MODELS carry the tag, not just bundle prose; renderers show caveats; machine promotion without evidence refs is blocked. This is the type-system version of the unverified-candidate discipline, and the claim-kind compatibility registry (37t.16) consumes it.

## Acceptance criteria

Digest from prose containing SHA+decision marks those fields text_derived while exit-code outcome stays raw_evidence; policy test fails on a forensic conclusion rendered from text-derived fields without caveat. Verify: transforms payload tests.

## Static mechanism / likely defect

Forensic transforms mine SHAs, decisions, caveats, and counts from prose. These are useful hints but not the same as structured evidence; payloads need text-derived provenance.

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

1. Find prose-mined fields in tool summaries, decision candidates, forensic index entries, and report builders.
2. Add `evidence_class`/`text_derived_fields` metadata that names which fields were mined from prose.
3. Render caveats wherever those fields appear; block promotion to machine-trusted finding unless concrete evidence refs are attached.
4. Update export/report schemas so downstream consumers cannot mistake mined text for observed fact.

## Tests to add

- A prose-only SHA/decision fixture renders with text-derived caveat.
- A structured evidence-ref fixture renders as observed/grounded.
- Finding promotion rejects text-derived-only claims unless policy explicitly allows candidate-only.

## Verification commands

- ``devtools test tests/unit/insights/test_transforms.py -k 'text_derived or prose or forensic or evidence_class'` plus renderer tests for the affected report surface.`

## Pitfalls

- Do not broaden scope beyond the bead acceptance criteria; make a failing test first, then patch the smallest shared seam.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.

# 07. polylogue-9e5.29 — Distinguish absent evidence from true numeric zero at field level

Priority: **P1**  
Lane: **evidence-honesty**  
Readiness: **needs-small-spec then code**

Depends on packet(s): polylogue-9e5.28

## Why this is urgent / critical-path

Counts, costs, rates, durations, and scores are user-visible claims. `0` must mean true zero, not “no backing rows were available.”

## Static diagnosis / likely mechanism

Mechanism: aggregate SQL and Pydantic payloads use numeric identity values (`COALESCE(SUM(...), 0)`, default `0.0`, default count fields) without denominator/evidence metadata. Product-level rigor contracts cannot detect a specific field whose denominator is absent. Likely hot spots include usage/cost rollups in `polylogue/storage/usage.py`, archive-tier cost aggregations, and summary/forensics payloads.

## Implementation plan

Implementation shape:
1. Add `RigorFieldContract` with fields such as `field_path`, `value_kind`, `denominator_path`, `unit`, `provenance_class`, `evidence_tier`, `nullable_when_ungrounded`, and `zero_semantics`.
2. Add `field_contracts` to `RigorContract`.
3. Teach the rigor audit to evaluate field contracts: if denominator is absent/zero and `nullable_when_ungrounded=true`, a stored/rendered numeric `0` is a defect unless the field’s zero semantics says true zero.
4. Convert worst public offenders first: provider/cost rollups and any forensics/report fields that can render `0.0` over empty backing rows.
5. SQL pattern: use `CASE WHEN COUNT(backing.id)=0 THEN NULL ELSE SUM(...) END`, not blanket `COALESCE`.
6. Render `None`/unknown as `uncovered`, `unknown`, or `not applicable`, not as zero.

## Test plan

Tests:
- empty provider-usage backing rows produce `None`/uncovered for costs/tokens, not `0.0`.
- a real row with zero cost/tokens still renders zero.
- field-level audit catches a fixture product where denominator=0 but value=0.0.
- renderer/surface test shows `uncovered` or equivalent marker.

## Verification command / proof

`devtools test tests/unit/insights/test_rigor_audit.py tests/unit/storage/test_usage*.py tests/unit/cli/test_*usage* -k 'empty or zero or uncovered or field_contract'`

## Pitfalls

Avoid a huge repo-wide migration in one PR. First add the contract/audit machinery and convert the highest-risk public numbers. Then file follow-ups for remaining numeric products revealed by the audit.

## Files/functions to inspect or touch

- `polylogue/insights/rigor.py`
- `polylogue/insights/audit.py`
- `polylogue/storage/usage.py`
- `polylogue/storage/sqlite/archive_tiers/archive.py`
- `scripts/agent_forensics.py or promoted report surface`

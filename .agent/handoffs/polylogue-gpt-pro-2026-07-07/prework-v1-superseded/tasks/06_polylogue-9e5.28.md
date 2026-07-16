# 06. polylogue-9e5.28 — Make the rigor audit iterate the full insight registry, not only existing contracts

Priority: **P1**  
Lane: **evidence-honesty**  
Readiness: **ready-now / code-local**

Depends on packet(s): polylogue-s7ae.6

## Why this is urgent / critical-path

An audit that only loops declared contracts cannot find missing contracts. Product claims then look audited even when entire insight products are invisible to the audit.

## Static diagnosis / likely mechanism

Root cause: `build_insight_rigor_audit_report` explicitly says products without registered rigor contracts are skipped and loops `for contract in list_rigor_contracts()` (`polylogue/insights/audit.py:173-201`). `_RIGOR_MATRIX` only covers a small subset of registry products (`polylogue/insights/rigor.py:85+`), while `INSIGHT_REGISTRY` is the product universe (`polylogue/insights/registry.py:294+`).

## Implementation plan

Implementation shape:
1. Add `coverage_status` to `InsightRigorAuditEntry`: `covered`, `uncovered`, `exempt`.
2. Add an explicit exemption map with reason strings for products that intentionally have no rigor contract.
3. Change audit iteration to sorted registry names. For each registry product:
   - if contract exists: run current audit, `coverage_status=covered`;
   - if exemption exists: emit a row, `coverage_status=exempt`, with reason;
   - else: emit a row, `coverage_status=uncovered`, with an error like `missing_rigor_contract`.
4. Fail the `insight-honesty` policy/lab lane if any non-exempt product is uncovered.
5. Update docs/renderers so missing-contract rows are loud and countable.

## Test plan

Tests:
- monkeypatched registry product without contract appears as `uncovered`, not omitted.
- exemption appears as `exempt` with reason.
- known contract product still audits normally.
- policy gate fails on uncovered product.
- old “all contract names exist in registry” test remains but no longer pretends coverage completeness.

## Verification command / proof

`devtools test tests/unit/insights/test_rigor_audit.py -k 'registry or uncovered or exemption or rigor'`

## Pitfalls

Do not just add more entries to `_RIGOR_MATRIX`. That helps today but preserves the same blindness for the next product.

## Files/functions to inspect or touch

- `polylogue/insights/audit.py:173-201`
- `polylogue/insights/rigor.py:45-85`
- `polylogue/insights/registry.py:294+`
- `devtools/lab or policy lane for insight-honesty`

# 029. polylogue-9e5.19 — Storage-layer correctness scenario family

Priority/type/status: **P2 / task / open**. Lane: **00-trust-floor**. Release: **A-trust-floor**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

_No description in export._

## Existing design note

scenario-coverage.yaml gap 'storage-correctness' orphaned on gh#590. Build a scenario family (devtools lab projections / scenarios) exercising split-tier writes, content-hash idempotency, FTS trigger integrity, blob-lease GC, and lineage composition against seeded archives.

## Acceptance criteria

A storage-correctness scenario family exists and runs via devtools lab lanes; it covers idempotent re-ingest, FTS trigger drift, and lineage composition; scenario-coverage.yaml references this bead, not gh#590. Verify: devtools lab projections + lab lanes.

## Static mechanism / likely defect

Bead design: create scenario family covering split-tier writes, content-hash idempotency, FTS trigger integrity, blob-lease GC, and lineage composition against seeded archives.

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
- `polylogue/core/dates.py:10` — parse_date has no injected clock parameter.
- `polylogue/core/dates.py:37` — RELATIVE_BASE uses ambient datetime.now.
- `polylogue/archive/query/expression.py:2440` — Query grammar recognizes relative-date literals.
- `polylogue/archive/query/spec.py:498` — SessionQuerySpec.from_params is the central query-spec constructor.
- `polylogue/insights/temporal_source.py:66` — classify_profile_hwm_source promotes any updated_at to provider_ts.
- `polylogue/insights/temporal_source.py:97` — classify_aggregate_hwm_source currently collapses all non-empty source updates to provider_ts.

## Implementation plan

1. Implementation shape:
2. 1. Add `storage-correctness` to scenario-coverage configuration, referencing this bead.
3. 2. Build seeded archive fixtures for: idempotent re-ingest, split-tier write/read, FTS mutation drift, leased blob GC, lineage composition snapshot.
4. 3. Wire into devtools lab/projections lanes.
5. 4. The scenario should aggregate existing focused tests where possible rather than duplicate all logic.
6. 5. Emit a compact scenario report with pass/fail and fixture paths.

## Tests to add

- Acceptance proof: A storage-correctness scenario family exists and runs via devtools lab lanes
- Acceptance proof: it covers idempotent re-ingest, FTS trigger drift, and lineage composition
- Acceptance proof: scenario-coverage.yaml references this bead, not gh#590.
- Acceptance proof: Verify: devtools lab projections + lab lanes.

## Verification commands

- ``devtools lab projections --scenario storage-correctness` or the project’s current lab command; exact command should be documented in the PR.`

## Pitfalls

- Do not broaden scope beyond the bead acceptance criteria; make a failing test first, then patch the smallest shared seam.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.

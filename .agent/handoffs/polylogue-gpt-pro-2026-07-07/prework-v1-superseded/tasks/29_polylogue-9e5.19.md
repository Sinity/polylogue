# 29. polylogue-9e5.19 — Storage-layer correctness scenario family in devtools lab

Priority: **P2**  
Lane: **storage-correctness**  
Readiness: **ready-now after focused bugs**

Depends on packet(s): polylogue-8jg9.4 + polylogue-8jg9.2, polylogue-1xc.12, polylogue-4ts.4

## Why this is urgent / critical-path

Several storage correctness invariants are spread across tests. A scenario lane gives future agents one place to prove split-tier/idempotency/FTS/blob/lineage basics.

## Static diagnosis / likely mechanism

Bead design: create scenario family covering split-tier writes, content-hash idempotency, FTS trigger integrity, blob-lease GC, and lineage composition against seeded archives.

## Implementation plan

Implementation shape:
1. Add `storage-correctness` to scenario-coverage configuration, referencing this bead.
2. Build seeded archive fixtures for: idempotent re-ingest, split-tier write/read, FTS mutation drift, leased blob GC, lineage composition snapshot.
3. Wire into devtools lab/projections lanes.
4. The scenario should aggregate existing focused tests where possible rather than duplicate all logic.
5. Emit a compact scenario report with pass/fail and fixture paths.

## Test plan

Tests are the scenario: it must fail if one invariant is intentionally broken in fixture/code. Unit test the scenario registration if devtools has registry tests.

## Verification command / proof

`devtools lab projections --scenario storage-correctness` or the project’s current lab command; exact command should be documented in the PR.

## Pitfalls

Do this after the highest-risk concrete fixes so the scenario lane locks corrected behavior instead of snapshotting known-bad behavior.

## Files/functions to inspect or touch

- `devtools/lab*`
- `scenario-coverage.yaml`
- `tests/fixtures archive builders`
- `FTS/blob/lineage tests`

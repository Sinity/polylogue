Summary

- Make the 20d.14 interactive catalog explicitly name daemon query, completion, cold CLI, and ingest-to-searchable budget families.
- Add the ingest-to-searchable benchmark row and describe the shared catalog as the budget authority for sibling performance work.
- Extend catalog tests so removal of any interactive budget family fails verification.
- Update `devtools bench slo` command guidance to include the interactive surfaces.

Problem

The catalog had daemon query and completion measurements but no ingest-to-searchable row, so the parent interactive-performance contract did not enumerate all four required end-to-end budget families. The existing live run also shows that cold/warm status are above their declared targets and that the 5k seeded fixture times out during branch-point witness refresh; those results are not represented as green.

Solution

Added the `ingest_to_searchable` row with the existing live convergence benchmark and the 5s/10s p50/p95 budget. Added deterministic anti-vacuity coverage for daemon query, daemon completion, cold CLI, and ingest-to-searchable entries. Existing informational gates remain informational until their owning production routes and seeded-corpus measurements are stable.

Verification

- `nix develop --command devtools test tests/unit/devtools/test_slo_catalog.py` — 17 passed.
- `nix develop --command devtools verify --quick` — exit 0; all listed quick checks passed.
- `nix develop --command python -m devtools bench slo --include-lab` — exit 1: daemon query/status/completion/cancellation/concurrency passed; cold status measured p50 1766.6ms and warm status p50 4369.4ms against 500/400ms targets; 5k seeded fixture timed out in `write._refresh_stable_branch_point_witnesses`, leaving query/reader/facets without benchmark results.

Residual risk

The parent acceptance is not fully green on this branch: cold/warm status remain over budget, ingest-to-searchable is cataloged but informational, and the seeded 5k fixture timeout blocks the ordinary query/reader/facets benchmark rows. No live daemon or operator archive state was changed.

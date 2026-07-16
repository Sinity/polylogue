# 16. polylogue-20d.4 — Mirror daemon structured-query routing in CLI so non-FTS filters skip the FTS readiness gate

Priority: **P2**  
Lane: **query-correctness**  
Readiness: **ready-now / code-local**

## Why this is urgent / critical-path

Structured-only queries should work even when full-text-search is stale or rebuilding. Paying the FTS gate for origin/date/path filters is both slow and incorrect.

## Static diagnosis / likely mechanism

Mechanism from bead: daemon route discriminates structured-only vs FTS queries, while CLI search calls the search path unconditionally. Static source confirms daemon has a route split in `polylogue/daemon/http.py`; archive query/lowering uses `SessionQuerySpec` and query terms.

## Implementation plan

Implementation shape:
1. Inspect the single CLI archive search/list execution site, likely in `polylogue/cli/archive_query.py`.
2. Construct `SessionQuerySpec` exactly once.
3. Branch like daemon: if `spec.query_terms` or `spec.contains_terms` or semantic/similar text is present, use search/FTS/vector path; otherwise use list/facade structured path.
4. Keep output rendering identical so users do not see a behavior split except the missing FTS gate.
5. Add a helper shared with daemon if duplication is small enough; otherwise add a comment tying both discriminators together.

## Test plan

Tests:
- archive with deliberately absent/stale FTS still answers `--origin X --since DATE` or equivalent structured-only CLI query.
- query with text terms still uses FTS readiness and fails/repairs as before.
- daemon and CLI return same session ids for the same structured filter fixture.
- regression name references `#1860`/bead id.

## Verification command / proof

`devtools test tests/unit/cli/test_archive_query*.py tests/unit/daemon/test_daemon_http*.py -k 'structured or fts or query_routing'`

## Pitfalls

Verify current v23 shape first; recent freshness work may have moved the discriminator. Do not make structured queries accidentally bypass semantic/vector queries.

## Files/functions to inspect or touch

- `polylogue/cli/archive_query.py`
- `polylogue/archive/query/*`
- `polylogue/daemon/http.py`
- `polylogue/storage/sqlite/archive_tiers/archive.py`

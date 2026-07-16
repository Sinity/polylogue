# 044. polylogue-t46.6 — Fix referenced_path OR-vs-AND filter divergence and delete dead CLI stats aggregators

Priority/type/status: **P2 / task / open**. Lane: **04-read-contract-query-render**. Release: **C-read-contract**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

_No description in export._

## Existing design note

cli/query_semantic.py:63 referenced_path_matches_slice uses any(term...) (OR-of-terms) for multi-term referenced_path while the substrate archive/query/runtime_matching.py:35 uses all(term...) (AND-of-terms), so the semantic-stats surface selects different sessions than the actual query filter -- a live correctness divergence. Separately, cli/query_stats.py (origin/date grouping :63-78/:353/:399, semantic grouping :446/:514, profile work-kind grouping :628 via auto_tags 'kind:' scan) and query_semantic.py:151 re-derive aggregation that ArchiveStore.stats_by (SQL, api get_stats_by) already owns via workflow_shape/sort_key_ms, and they have no live CLI dispatch caller (only re-exports + tests). Fix: route query_semantic path/action matching through the shared predicate/SQL params (delete the CLI copies), and delete the dead query_stats/query_semantic in-memory aggregators in favor of stats_by, removing the tests that pin the dead shape.

## Acceptance criteria

A two-term referenced_path query returns the same session set from the semantic-stats surface and from the query filter (regression test); referenced_path_matches_slice/action_matches_slice and the dead query_stats aggregators are deleted (grep confirms callers gone); origin/date/tool/work-kind grouping goes through stats_by; devtools verify green.

## Static mechanism / likely defect

Design direction: cli/query_semantic.py:63 referenced_path_matches_slice uses any(term...) (OR-of-terms) for multi-term referenced_path while the substrate archive/query/runtime_matching.py:35 uses all(term...) (AND-of-terms), so the semantic-stats surface selects different sessions than the actual query filter -- a live correctness divergence. Separately, cli/query_stats.py (origin/date grouping :63-78/:353/:399, semantic grouping :…

## Source anchors to inspect first

- `polylogue/archive/query/spec.py:440` — SessionQuerySpec is the core read selection object.
- `polylogue/archive/query/spec.py:498` — from_params centralizes parameter lowering.
- `polylogue/cli/query_contracts.py:70` — QueryOutputSpec is current output/render contract seam.
- `polylogue/cli/archive_query.py:164` — CLI archive query execution remains a dense orchestration point.
- `polylogue/daemon/http.py:1734` — Daemon split-archive route already has from_params path for some cases.
- `polylogue/daemon/http.py:3296` — Another daemon path builds SessionQuerySpec from raw query params.
- `polylogue/mcp/query_contracts.py:54` — MCP query normalization has a distinct lowering path.

## Implementation plan

1. cli/query_semantic.py:63 referenced_path_matches_slice uses any(term...) (OR-of-terms) for multi-term referenced_path while the substrate archive/query/runtime_matching.py:35 uses all(term...) (AND-of-terms), so the semantic-stats surface selects different sessions than the actual query filter -- a live correctness divergence.
2. Separately, cli/query_stats.py (origin/date grouping :63-78/:353/:399, semantic grouping :446/:514, profile work-kind grouping :628 via auto_tags 'kind:' scan) and query_semantic.py:151 re-derive aggregation that ArchiveStore.stats_by (SQL, api get_stats_by) already owns via workflow_shape/sort_key_ms, and they have no live CLI dispatch caller (only re-exports + tests).
3. Fix: route query_semantic path/action matching through the shared predicate/SQL params (delete the CLI copies), and delete the dead query_stats/query_semantic in-memory aggregators in favor of stats_by, removing the tests that pin the dead shape.

## Tests to add

- Acceptance proof: A two-term referenced_path query returns the same session set from the semantic-stats surface and from the query filter (regression test)
- Acceptance proof: referenced_path_matches_slice/action_matches_slice and the dead query_stats aggregators are deleted (grep confirms callers gone)
- Acceptance proof: origin/date/tool/work-kind grouping goes through stats_by
- Acceptance proof: devtools verify green.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools test tests/unit tests/integration -k "query or contract or citation or insight"`

## Pitfalls

- Do not broaden scope beyond the bead acceptance criteria; make a failing test first, then patch the smallest shared seam.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.

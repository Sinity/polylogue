# 020. polylogue-20d.4 — CLI structured-query routing parity with daemon (#1860): no FTS gate for non-FTS queries

Priority/type/status: **P2 / bug / open**. Lane: **08-scale-performance-live**. Release: **B/G-storage-live-performance**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

The daemon discriminates structured-only queries from FTS queries (http.py ~:1789-1793); the CLI calls the search path unconditionally, so structured filters pay the FTS readiness gate. Port the discriminator at the single CLI search-vs-list site (branch on spec.query_terms/contains_terms). Regression: structured-only query on an archive with deliberately-stale FTS must succeed. Verify current state first — v23 + recent work may have changed the shape.

## Existing design note

The daemon discriminates structured-only queries from FTS queries (polylogue/daemon/http.py ~:1789-1793); the CLI calls the search path unconditionally so structured filters pay the FTS readiness gate. Port the discriminator to the single CLI search-vs-list site, branching on spec.query_terms/contains_terms so structured-only queries skip the FTS gate. Verify current shape first — v23 freshness work may have changed it.

## Acceptance criteria

- The CLI search-vs-list site branches on structured-only vs FTS (spec.query_terms/contains_terms), mirroring the daemon http.py discriminator; structured-only queries no longer pass through the FTS readiness gate.
- Regression test: a structured-only query (filter by origin/date, no query terms) against an archive with deliberately-stale/absent FTS returns results and does not raise or deny on FTS readiness; `devtools test <cli query routing test>` green.
- The current (post-v23) routing shape is verified and documented in the PR before the change.

## Static mechanism / likely defect

Some CLI structured queries still route through FTS readiness gates even when the query can be satisfied structurally. Daemon paths already use `SessionQuerySpec.from_params` in places, creating parity pressure.

## Source anchors to inspect first

- `polylogue/storage/sqlite/archive_tiers/index.py:296` — messages_fts table DDL lives in archive_tiers/index.py.
- `polylogue/storage/fts/sql.py:11` — messages_fts DDL copy also exists in storage/fts/sql.py.
- `polylogue/storage/fts/fts_lifecycle.py:292` — ensure_fts_triggers_sync owns runtime repair/recreation.
- `polylogue/storage/fts/fts_lifecycle.py:512` — Thread FTS rebuild logic is duplicated in lifecycle repair path.
- `polylogue/daemon/convergence_stages.py:988` — daemon convergence has another repair/readiness path for archive messages_fts.
- `polylogue/archive/query/spec.py:440` — SessionQuerySpec is the core read selection object.
- `polylogue/archive/query/spec.py:498` — from_params centralizes parameter lowering.
- `polylogue/cli/query_contracts.py:70` — QueryOutputSpec is current output/render contract seam.
- `polylogue/cli/archive_query.py:164` — CLI archive query execution remains a dense orchestration point.
- `polylogue/daemon/http.py:1734` — Daemon split-archive route already has from_params path for some cases.
- `polylogue/daemon/http.py:3296` — Another daemon path builds SessionQuerySpec from raw query params.
- `polylogue/mcp/query_contracts.py:54` — MCP query normalization has a distinct lowering path.

## Implementation plan

1. Classify query params into structural-only vs FTS-required before readiness checks.
2. Route CLI query construction through the same `SessionQuerySpec.from_params` path as daemon/API.
3. Move FTS readiness failures into only the branches that need MATCH/text search.
4. Add a parity fixture comparing CLI and daemon envelopes for structured-only filters.

## Tests to add

- `origin=codex status=...` style query works with missing/broken messages_fts.
- A text MATCH query still fails loudly when FTS is unavailable.
- CLI/daemon/MCP row counts match for structural-only queries.

## Verification commands

- ``devtools test tests/unit/cli/test_archive_query*.py tests/unit/daemon/test_daemon_http*.py -k 'structured or fts or query_routing'``

## Pitfalls

- Do not optimize by hiding degraded/partial state; status must get more truthful, not prettier.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.

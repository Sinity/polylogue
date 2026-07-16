# 041. polylogue-t46.3 — Unify list/search query-spec->ArchiveStore execution across CLI, MCP, and daemon web

Priority/type/status: **P2 / task / open**. Lane: **04-read-contract-query-render**. Release: **C-read-contract**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

_No description in export._

## Existing design note

Three surfaces re-map the query DSL/params to ArchiveStore filter args and each own pagination/total/cursor: daemon http.py:1902 _do_archive_list_sessions (+ the _do_archive_* fast-path family), mcp/archive_support.py:254-379 archive_session_list_payload/archive_search_payload, and cli/archive_query.py:674/787/815 _query_hits/_paginate_rows/_build_cursor. The http.py:1970 comment admits it 'must mirror those public params here' and it re-fixed bugs #1873/#1860 in the parallel path; MCP has two internal list surfaces with different total semantics (archive_support estimate vs server_tools.py:363 poly.archive_count_sessions). Fix: route every surface through SessionQuerySpec.from_params + a single archive execution helper in archive/query/archive_execution.py that returns (rows, total, cursor); surfaces differ only in payload projection (build_search_envelope is already shared). Collapse the _web_reader_archive_root dual path so the facade is the single execution owner.

## Acceptance criteria

CLI find, MCP archive_list_sessions/archive_search_sessions, and daemon /api/sessions return the same total and page boundaries for identical filters (parity test across the three surfaces); the per-surface spec->filter mapping and total/cursor logic is deleted in favor of one execution helper (grep shows no second query_terms/contains merge); the two MCP list surfaces converge to one total semantic; devtools verify green.

## Static mechanism / likely defect

Design direction: Three surfaces re-map the query DSL/params to ArchiveStore filter args and each own pagination/total/cursor: daemon http.py:1902 _do_archive_list_sessions (+ the _do_archive_* fast-path family), mcp/archive_support.py:254-379 archive_session_list_payload/archive_search_payload, and cli/archive_query.py:674/787/815 _query_hits/_paginate_rows/_build_cursor. The http.py:1970 comment admits it 'must mirror those public …

## Source anchors to inspect first

- `polylogue/archive/query/spec.py:440` — SessionQuerySpec is the core read selection object.
- `polylogue/archive/query/spec.py:498` — from_params centralizes parameter lowering.
- `polylogue/cli/query_contracts.py:70` — QueryOutputSpec is current output/render contract seam.
- `polylogue/cli/archive_query.py:164` — CLI archive query execution remains a dense orchestration point.
- `polylogue/daemon/http.py:1734` — Daemon split-archive route already has from_params path for some cases.
- `polylogue/daemon/http.py:3296` — Another daemon path builds SessionQuerySpec from raw query params.
- `polylogue/mcp/query_contracts.py:54` — MCP query normalization has a distinct lowering path.

## Implementation plan

1. Three surfaces re-map the query DSL/params to ArchiveStore filter args and each own pagination/total/cursor: daemon http.py:1902 _do_archive_list_sessions (+ the _do_archive_* fast-path family), mcp/archive_support.py:254-379 archive_session_list_payload/archive_search_payload, and cli/archive_query.py:674/787/815 _query_hits/_paginate_rows/_build_cursor.
2. The http.py:1970 comment admits it 'must mirror those public params here' and it re-fixed bugs #1873/#1860 in the parallel path
3. MCP has two internal list surfaces with different total semantics (archive_support estimate vs server_tools.py:363 poly.archive_count_sessions).
4. Fix: route every surface through SessionQuerySpec.from_params + a single archive execution helper in archive/query/archive_execution.py that returns (rows, total, cursor)
5. surfaces differ only in payload projection (build_search_envelope is already shared).
6. Collapse the _web_reader_archive_root dual path so the facade is the single execution owner.

## Tests to add

- Acceptance proof: CLI find, MCP archive_list_sessions/archive_search_sessions, and daemon /api/sessions return the same total and page boundaries for identical filters (parity test across the three surfaces)
- Acceptance proof: the per-surface spec->filter mapping and total/cursor logic is deleted in favor of one execution helper (grep shows no second query_terms/contains merge)
- Acceptance proof: the two MCP list surfaces converge to one total semantic
- Acceptance proof: devtools verify green.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools test tests/unit tests/integration -k "query or contract or citation or insight"`

## Pitfalls

- Do not broaden scope beyond the bead acceptance criteria; make a failing test first, then patch the smallest shared seam.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.

# 048. polylogue-x7d — Unify root query row rendering contracts

Priority/type/status: **P2 / task / open**. Lane: **04-read-contract-query-render**. Release: **C-read-contract**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

The bounded find fix had to patch three projection/rendering paths: archive_query root rows, query_output deterministic rows, and select rows. This duplication let --limit bound row count while multiline titles/snippets still exploded output in the live archive. Collapse list/search/select row rendering onto one projection contract for title normalization, snippet bounds, machine payload shape, and plain text rendering, then keep archive_query/query_output/select as thin adapters.

## Existing design note

Target shape: define a small shared row projection helper or value object for session list rows and search-hit rows, with explicit budgets (title 96 or table budget, snippet 320), single-line normalization, and separate full-read expansion. archive_query._summary_payload/_hit_payload/_summary_line/_hit_line, cli.query_output format_summary_list/format_search_hit_list, and cli.select select_row_from_result should call that shared contract rather than each carrying its own truncation rules. Preserve existing JSON schemas; change only overlong values. Add parity tests proving the three surfaces produce bounded titles/snippets for the same giant title/search hit.

## Acceptance criteria

- A shared row-projection helper/value object exists for session-list rows and search-hit rows with explicit budgets (title 96 / table budget, snippet 320), single-line normalization, and separate full-read expansion.
- archive_query._summary_payload/_hit_payload/_summary_line/_hit_line, cli.query_output.format_summary_list/format_search_hit_list, and cli.select.select_row_from_result all call the shared contract (grep shows no per-surface truncation rules remaining).
- Existing JSON schemas are preserved; only overlong values change.
- Parity tests prove the three surfaces produce bounded titles/snippets for the same giant title/search hit (`devtools test <parity test>` green).
- Informativeness: unified rows carry, beyond title/origin/date, an outcome badge (structural terminal state: completed/failed/abandoned/unknown), cost when priced provenance exists, relative time, repo/cwd basename, and message count; the column set is consistent across find results, `read --all` listings, and select pickers; `--format json` carries the same fields under the same names (schema-checked). Display-title synthesis (30h) feeds the title cell.

## Static mechanism / likely defect

Issue description localizes the mechanism: The bounded find fix had to patch three projection/rendering paths: archive_query root rows, query_output deterministic rows, and select rows. This duplication let --limit bound row count while multiline titles/snippets still exploded output in the live archive. Collapse list/search/select row rendering onto one projection contract for title normalization, snippet bounds, machine payload shape, and plain text rendering, then keep archive_query/query_output/select as thin adapters. Design direction: Target shape: define a small shared row projection helper or value object for session list rows and search-hit rows, with explicit budgets (title 96 or table budget, snippet 320), single-line normalization, and separate full-read expansion. archive_query._summary_payload/_hit_payload/_summary_line/_hit_line, cli.query_output format_summary_list/format_search_hit_list, and cli.select select_row_from_result should cal…

## Source anchors to inspect first

- `polylogue/archive/query/spec.py:440` — SessionQuerySpec is the core read selection object.
- `polylogue/archive/query/spec.py:498` — from_params centralizes parameter lowering.
- `polylogue/cli/query_contracts.py:70` — QueryOutputSpec is current output/render contract seam.
- `polylogue/cli/archive_query.py:164` — CLI archive query execution remains a dense orchestration point.
- `polylogue/daemon/http.py:1734` — Daemon split-archive route already has from_params path for some cases.
- `polylogue/daemon/http.py:3296` — Another daemon path builds SessionQuerySpec from raw query params.
- `polylogue/mcp/query_contracts.py:54` — MCP query normalization has a distinct lowering path.

## Implementation plan

1. Target shape: define a small shared row projection helper or value object for session list rows and search-hit rows, with explicit budgets (title 96 or table budget, snippet 320), single-line normalization, and separate full-read expansion.
2. archive_query._summary_payload/_hit_payload/_summary_line/_hit_line, cli.query_output format_summary_list/format_search_hit_list, and cli.select select_row_from_result should call that shared contract rather than each carrying its own truncation rules.
3. Preserve existing JSON schemas
4. change only overlong values.
5. Add parity tests proving the three surfaces produce bounded titles/snippets for the same giant title/search hit.

## Tests to add

- Acceptance proof: A shared row-projection helper/value object exists for session-list rows and search-hit rows with explicit budgets (title 96 / table budget, snippet 320), single-line normalization, and separate full-read expansion.
- Acceptance proof: archive_query._summary_payload/_hit_payload/_summary_line/_hit_line, cli.query_output.format_summary_list/format_search_hit_list, and cli.select.select_row_from_result all call the shared contract (grep shows no per-surface truncation rules remaining).
- Acceptance proof: Existing JSON schemas are preserved
- Acceptance proof: only overlong values change.
- Acceptance proof: Parity tests prove the three surfaces produce bounded titles/snippets for the same giant title/search hit (`devtools test <parity test>` green).
- Acceptance proof: Informativeness: unified rows carry, beyond title/origin/date, an outcome badge (structural terminal state: completed/failed/abandoned/unknown), cost when priced provenance exists, relative time, repo/cwd basename, and message count
- Acceptance proof: the column set is consistent across find results, `read --all` listings, and select pickers
- Acceptance proof: `--format json` carries the same fields under the same names (schema-checked).

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools test tests/unit tests/integration -k "query or contract or citation or insight"`

## Pitfalls

- Do not broaden scope beyond the bead acceptance criteria; make a failing test first, then patch the smallest shared seam.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.

# 049. polylogue-fnm.11 — Pipeline/clause parity across units + generated support matrix

Priority/type/status: **P2 / task / open**. Lane: **04-read-contract-query-render**. Release: **C-read-contract**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

Live evidence: `sessions where origin:claude-code-session | count` fails with 'pipeline terminal stage must be an executable <unit>s where ... query' while `observed-events where kind:tool_finished | group by handler | count` works — the sessions unit does not support the count pipeline the docs/memory present as canonical. `after:2026-07-01` parses in bare find mode but is 'invalid query expression near column 43' inside `sessions where ...` — the compact-clause vocabulary differs between find-mode and unit-where-mode with no documentation of the split. The one bright spot: unsupported group-by fields produce a helpful error listing supported fields (keep that pattern).

## Existing design note

(1) Build the support matrix FROM the registries (query_units + stage lowerers + clause grammar), not by hand: a generated docs/query-support-matrix.md (devtools render, drift-checked) showing units x pipeline stages x compact clauses. The generator doubles as the gap list. (2) Close the two gaps the evidence hit: `| count` (and group-by) on the sessions unit; date clauses (after:/before:) inside unit-where expressions. Both lower onto existing SQL (sessions has created_at; count is trivial) — the gap is grammar wiring, not storage. Memory note applies: pipeline stages are hand-parsed outside the Lark grammar (split on |), so new stage support per unit is lowerer work; terminal priorities pitfall for any new ':' token. (3) Error rendering: every unsupported-combination error follows the group-by pattern — name the unit, the stage/clause, and the nearest supported alternative; parse errors gain a caret line under the query text (column number already computed, 'near column 43' is user-hostile without one). Feeds fnm.1 (aggregates) — do the matrix first so fnm.1 lands against known gaps.

## Acceptance criteria

docs/query-support-matrix.md is generated from registries and drift-checked by render all --check. 'sessions where origin:X | count' and group-by on sessions work. after:/before: clauses parse inside unit-where expressions. Every unsupported unit/stage/clause combination errors with the unit, the construct, and the nearest supported alternative; parse errors render a caret line.

## Static mechanism / likely defect

Issue description localizes the mechanism: Live evidence: `sessions where origin:claude-code-session | count` fails with 'pipeline terminal stage must be an executable <unit>s where ... query' while `observed-events where kind:tool_finished | group by handler | count` works — the sessions unit does not support the count pipeline the docs/memory present as canonical. `after:2026-07-01` parses in bare find mode but is 'invalid query expression near column 43' inside `sessions where ...` — the compact-clause vocabulary differs between find-mode and unit-where… Design direction: (1) Build the support matrix FROM the registries (query_units + stage lowerers + clause grammar), not by hand: a generated docs/query-support-matrix.md (devtools render, drift-checked) showing units x pipeline stages x compact clauses. The generator doubles as the gap list. (2) Close the two gaps the evidence hit: `| count` (and group-by) on the sessions unit; date clauses (after:/before:) inside unit-where expressi…

## Source anchors to inspect first

- `polylogue/archive/query/spec.py:440` — SessionQuerySpec is the core read selection object.
- `polylogue/archive/query/spec.py:498` — from_params centralizes parameter lowering.
- `polylogue/cli/query_contracts.py:70` — QueryOutputSpec is current output/render contract seam.
- `polylogue/cli/archive_query.py:164` — CLI archive query execution remains a dense orchestration point.
- `polylogue/daemon/http.py:1734` — Daemon split-archive route already has from_params path for some cases.
- `polylogue/daemon/http.py:3296` — Another daemon path builds SessionQuerySpec from raw query params.
- `polylogue/mcp/query_contracts.py:54` — MCP query normalization has a distinct lowering path.

## Implementation plan

1. (1) Build the support matrix FROM the registries (query_units + stage lowerers + clause grammar), not by hand: a generated docs/query-support-matrix.md (devtools render, drift-checked) showing units x pipeline stages x compact clauses.
2. The generator doubles as the gap list.
3. (2) Close the two gaps the evidence hit: `| count` (and group-by) on the sessions unit
4. date clauses (after:/before:) inside unit-where expressions.
5. Both lower onto existing SQL (sessions has created_at
6. count is trivial) — the gap is grammar wiring, not storage.
7. Memory note applies: pipeline stages are hand-parsed outside the Lark grammar (split on |), so new stage support per unit is lowerer work

## Tests to add

- Acceptance proof: docs/query-support-matrix.md is generated from registries and drift-checked by render all --check.
- Acceptance proof: 'sessions where origin:X | count' and group-by on sessions work.
- Acceptance proof: after:/before: clauses parse inside unit-where expressions.
- Acceptance proof: Every unsupported unit/stage/clause combination errors with the unit, the construct, and the nearest supported alternative
- Acceptance proof: parse errors render a caret line.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools test tests/unit tests/integration -k "query or contract or citation or insight"`

## Pitfalls

- Do not broaden scope beyond the bead acceptance criteria; make a failing test first, then patch the smallest shared seam.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.

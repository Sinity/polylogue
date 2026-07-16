# 152. polylogue-fnm.12 — User-defined query macros: named, composable DSL shorthands in user.db

Priority/type/status: **P2 / feature / open**. Lane: **04-read-contract-query-render**. Release: **C-read-contract**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

The highest-leverage runtime configurable found in the preference design pass: operators (and agents) repeat the same filter combinations constantly — 'my real coding sessions' = origin:claude-code-session + repo-scope + exclude-subagents + trailing-90d. Today that is retyped or shell-aliased outside the product. Named macros stored in user.db make the DSL personal: define once, compose anywhere the grammar accepts a predicate, share with agents automatically (they resolve server-side, so MCP/webui/CLI all understand them).

## Existing design note

(1) Definition: 'polylogue config macro set mine "origin:claude-code-session exclude:subagents after:-90d"' (and MCP/webui equivalents); stored as typed user.db rows (the y4c settings registry), validated at definition time by compiling against the grammar — a macro that does not parse is refused with the caret error. (2) Reference syntax: @mine inside any query position where a predicate group is valid ('@mine "WAL contention" | group by model | count'). Expansion happens in the compiler BEFORE lowering (textual-hygienic: expanded predicates carry their macro provenance for error messages and explain output — 'explain' shows the expansion). (3) Composability rules: macros may reference macros (depth-capped, cycle-checked at definition); macros are predicate-groups only in v1 — no pipeline stages inside macros (keeps semantics local; revisit with evidence). (4) Surfaces: completions offer @-macros (fnm.4 registry projection); saved views can be macro-defined; the query-support matrix documents them. (5) Agent leverage: agents see the operator's macros via completions/explain — shared vocabulary between operator and agents for free; agents may define their own under a namespace (agent:@retry-storms) kept visually distinct.

## Acceptance criteria

Define/list/delete macros via CLI+MCP; @macro composes inside find, unit-where, and pipeline queries on the live archive; invalid macro refused at definition with caret; explain shows expansion with provenance; cycle/depth guards tested; completions surface @-macros.

## Static mechanism / likely defect

Issue description localizes the mechanism: The highest-leverage runtime configurable found in the preference design pass: operators (and agents) repeat the same filter combinations constantly — 'my real coding sessions' = origin:claude-code-session + repo-scope + exclude-subagents + trailing-90d. Today that is retyped or shell-aliased outside the product. Named macros stored in user.db make the DSL personal: define once, compose anywhere the grammar accepts a predicate, share with agents automatically (they resolve server-side, so MCP/webui/CLI all underst… Design direction: (1) Definition: 'polylogue config macro set mine "origin:claude-code-session exclude:subagents after:-90d"' (and MCP/webui equivalents); stored as typed user.db rows (the y4c settings registry), validated at definition time by compiling against the grammar — a macro that does not parse is refused with the caret error. (2) Reference syntax: @mine inside any query position where a predicate group is valid ('@mine "WAL…

## Source anchors to inspect first

- `polylogue/archive/query/spec.py:440` — SessionQuerySpec is the core read selection object.
- `polylogue/archive/query/spec.py:498` — from_params centralizes parameter lowering.
- `polylogue/cli/query_contracts.py:70` — QueryOutputSpec is current output/render contract seam.
- `polylogue/cli/archive_query.py:164` — CLI archive query execution remains a dense orchestration point.
- `polylogue/daemon/http.py:1734` — Daemon split-archive route already has from_params path for some cases.
- `polylogue/daemon/http.py:3296` — Another daemon path builds SessionQuerySpec from raw query params.
- `polylogue/mcp/query_contracts.py:54` — MCP query normalization has a distinct lowering path.

## Implementation plan

1. (1) Definition: 'polylogue config macro set mine "origin:claude-code-session exclude:subagents after:-90d"' (and MCP/webui equivalents)
2. stored as typed user.db rows (the y4c settings registry), validated at definition time by compiling against the grammar — a macro that does not parse is refused with the caret error.
3. (2) Reference syntax: @mine inside any query position where a predicate group is valid ('@mine "WAL contention" | group by model | count').
4. Expansion happens in the compiler BEFORE lowering (textual-hygienic: expanded predicates carry their macro provenance for error messages and explain output — 'explain' shows the expansion).
5. (3) Composability rules: macros may reference macros (depth-capped, cycle-checked at definition)
6. macros are predicate-groups only in v1 — no pipeline stages inside macros (keeps semantics local
7. revisit with evidence).

## Tests to add

- Acceptance proof: Define/list/delete macros via CLI+MCP
- Acceptance proof: @macro composes inside find, unit-where, and pipeline queries on the live archive
- Acceptance proof: invalid macro refused at definition with caret
- Acceptance proof: explain shows expansion with provenance
- Acceptance proof: cycle/depth guards tested
- Acceptance proof: completions surface @-macros.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools test tests/unit tests/integration -k "query or contract or citation or insight"`

## Pitfalls

- Do not broaden scope beyond the bead acceptance criteria; make a failing test first, then patch the smallest shared seam.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.

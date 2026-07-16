# 150. polylogue-fnm — Query DSL: one grammar owns query semantics; compose instead of multiplying verbs

Priority/type/status: **P2 / epic / open**. Lane: **04-read-contract-query-render**. Release: **C-read-contract**. Readiness: **epic-needs-child-closure**.

## What the bead says

The Lark grammar in polylogue/archive/query/expression.py is THE query language; extend in place. Landed since the GH issue: with-projection for all units, field selection for attached units, projection-unit completions. GH issue thread (body + comments) is input, not authority; this bead's scope statement wins where they conflict.

## Existing design note

The Lark grammar in polylogue/archive/query/expression.py IS the query language — extend it in place, never as a parallel verb/flag path. Baseline already landed: with-projection for all units, field selection for attached units, projection-unit completions. Treat the GH issue thread as input, not authority; this bead's scope statement wins where they conflict. Coordinates with t46 (the DSL becomes the sole owner of query semantics).

## Acceptance criteria

- New query semantics are added to the Lark grammar in polylogue/archive/query/expression.py (grep shows the grammar rule) rather than as a parallel verb or flag.
- The landed-since baseline (with-projection all units, field selection for attached units, projection-unit completions) stays green; `explain_query_expression` / `query_units` reflect the grammar.
- `devtools verify` is green on DSL tests; `devtools render all --check` is clean for any generated query-surface docs/schemas.
- Individual grammar extensions are tracked as child beads; the epic closes when the DSL is the sole owner of query semantics (t46 coordination).

## Static mechanism / likely defect

Issue description localizes the mechanism: The Lark grammar in polylogue/archive/query/expression.py is THE query language; extend in place. Landed since the GH issue: with-projection for all units, field selection for attached units, projection-unit completions. GH issue thread (body + comments) is input, not authority; this bead's scope statement wins where they conflict. Design direction: The Lark grammar in polylogue/archive/query/expression.py IS the query language — extend it in place, never as a parallel verb/flag path. Baseline already landed: with-projection for all units, field selection for attached units, projection-unit completions. Treat the GH issue thread as input, not authority; this bead's scope statement wins where they conflict. Coordinates with t46 (the DSL becomes the sole owner of…

## Source anchors to inspect first

- `polylogue/archive/query/spec.py:440` — SessionQuerySpec is the core read selection object.
- `polylogue/archive/query/spec.py:498` — from_params centralizes parameter lowering.
- `polylogue/cli/query_contracts.py:70` — QueryOutputSpec is current output/render contract seam.
- `polylogue/cli/archive_query.py:164` — CLI archive query execution remains a dense orchestration point.
- `polylogue/daemon/http.py:1734` — Daemon split-archive route already has from_params path for some cases.
- `polylogue/daemon/http.py:3296` — Another daemon path builds SessionQuerySpec from raw query params.
- `polylogue/mcp/query_contracts.py:54` — MCP query normalization has a distinct lowering path.

## Implementation plan

1. The Lark grammar in polylogue/archive/query/expression.py IS the query language — extend it in place, never as a parallel verb/flag path.
2. Baseline already landed: with-projection for all units, field selection for attached units, projection-unit completions.
3. Treat the GH issue thread as input, not authority
4. this bead's scope statement wins where they conflict.
5. Coordinates with t46 (the DSL becomes the sole owner of query semantics).

## Tests to add

- Acceptance proof: New query semantics are added to the Lark grammar in polylogue/archive/query/expression.py (grep shows the grammar rule) rather than as a parallel verb or flag.
- Acceptance proof: The landed-since baseline (with-projection all units, field selection for attached units, projection-unit completions) stays green
- Acceptance proof: `explain_query_expression` / `query_units` reflect the grammar.
- Acceptance proof: `devtools verify` is green on DSL tests
- Acceptance proof: `devtools render all --check` is clean for any generated query-surface docs/schemas.
- Acceptance proof: Individual grammar extensions are tracked as child beads
- Acceptance proof: the epic closes when the DSL is the sole owner of query semantics (t46 coordination).

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools test tests/unit tests/integration -k "query or contract or citation or insight"`

## Pitfalls

- Do not implement the epic as a single broad PR; use it to close/split child work and verify the terminal invariant.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.

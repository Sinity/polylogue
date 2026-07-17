# 052. polylogue-fnm.13 — Set-algebra over query results: union/intersect/except between queries

Priority/type/status: **P2 / feature / open**. Lane: **04-read-contract-query-render**. Release: **C-read-contract**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

_No description in export._

## Existing design note

BRAINSTORM (2026-07-05, operator asked to explore syntax incl. changing the pipeline operator; grain = message/unit AND session).

CORE REFRAME: this is relational algebra. Every construct is relation->relation:
- query (and/or/not predicates) = base relation (row-set; grain = session OR message/unit)
- set-ops union/intersect/except = binary relation->relation
- pipeline stages (group by, fields, read, context-image) = unary relation->relation
So we do NOT need three bespoke mechanisms; we need one relation algebra.

DESIGN A (RECOMMENDED — set-ops as subquery-taking pipeline stages; NO pipeline-syntax change):
  find auth | intersect (test) | except (draft) | group by model | read
  - `| intersect (SUBQUERY)` / `| union (SUBQUERY)` / `| except (SUBQUERY)` are binary stages.
  - Operand SUBQUERY is a full query (can nest set-ops/pipelines) -> this IS fnm.9 generalized.
  - Works at the CURRENT relation's grain (message-set or session-set); grain flows through the pipe.
  - Left-to-right, no precedence rules, parens delimit operands, existing `|` unchanged (non-breaking).
  - Cross-lane free: `semantic:"db migration" | except (~keyword)` combines a vector set and an FTS set.
  - Macros compose: `@cohortA | intersect (@cohortB)`.

DESIGN B (infix keyword sugar at top level): `auth intersect week:W01`, `(A intersect B) except C`.
  - Reads better for the simple 2-operand case; needs precedence vs the pipeline and vs and/or/not.
  - Can be added later as SUGAR that lowers to Design A. Not the primitive.

DESIGN C (operator repurpose — the "change pipeline syntax" option): make `|`=union, `&`=intersect,
  `-`=except (set-convention), and move pipeline to `then`/`>>`:
     (auth | test) - draft then group by model then read
  - Most set-algebra-forward and terse, but BREAKING: `|` means pipeline everywhere today (docs, tests,
    muscle memory). Design A already achieves full set-algebra without this break, so C's cost isn't
    justified unless we independently want `then`/`>>` for readability. Recommend NOT breaking `|`.

GRAIN: session-set is the clean base; message/unit-set works because the stage operates on the current
relation. Identity = session_id (session grain) or (session_id, message_id[, variant_index]) (message
grain). except/intersect/union on that key; ORDER for message grain follows the left operand's order
(document that union is left-then-new-right, dedup by key).

EXECUTION: reuse plan_execution to materialize each operand to a keyed set; a SetOpStage in the pipeline
(_split_pipeline_stages at expression.py:1436 already splits stages; add intersect/union/except as stage
verbs whose argument is a parenthesized subquery parsed by the same _QUERY_PARSER). Fail closed on
grain-mismatch between operands. EXPLAIN shows two sub-plans, not a cross join.

CROSS-SURFACE: CLI/MCP/API/daemon route the same string through one parser (fnm.11 matrix). Completions
(fnm.4) offer the set-op stage verbs after `|`.

RECOMMENDATION: build Design A (subquery-taking pipeline stages, session+message grain), add Design B
infix sugar as a fast-follow. Leave `|` = pipeline. This is the minimal-surface, maximal-power path and
it makes fnm.9 concrete.

## Acceptance criteria

`polylogue find 'auth intersect week:2026-W01'` returns exactly the session_ids in both operand sets; `except` subtracts; `union` dedups. Cross-lane composition works: `semantic:"X" except ~keyword` combines a vector set and an FTS set. Macros compose: `@a intersect @b` (with fnm.12). The combined set flows into the pipeline (`| read`, aggregates). Parenthesized nesting parses. Mixed-grain or unsupported operand fails with a typed error, never silently broadens. Parity: CLI/MCP/API/daemon route the same string through one parser (fnm.11 matrix). Verify: a parametrized test over union/intersect/except x {fts,semantic,structural,macro} operands asserting exact set identity + EXPLAIN shows two sub-plans, not a cross join.

## Static mechanism / likely defect

Design direction: BRAINSTORM (2026-07-05, operator asked to explore syntax incl. changing the pipeline operator; grain = message/unit AND session). CORE REFRAME: this is relational algebra. Every construct is relation->relation: - query (and/or/not predicates) = base relation (row-set; grain = session OR message/unit) - set-ops union/intersect/except = binary relation->relation - pipeline stages (group by, fields, read, context-image…

## Source anchors to inspect first

- `polylogue/archive/query/spec.py:440` — SessionQuerySpec is the core read selection object.
- `polylogue/archive/query/spec.py:498` — from_params centralizes parameter lowering.
- `polylogue/cli/query_contracts.py:70` — QueryOutputSpec is current output/render contract seam.
- `polylogue/cli/archive_query.py:164` — CLI archive query execution remains a dense orchestration point.
- `polylogue/daemon/http.py:1734` — Daemon split-archive route already has from_params path for some cases.
- `polylogue/daemon/http.py:3296` — Another daemon path builds SessionQuerySpec from raw query params.
- `polylogue/mcp/query_contracts.py:54` — MCP query normalization has a distinct lowering path.

## Implementation plan

1. BRAINSTORM (2026-07-05, operator asked to explore syntax incl.
2. changing the pipeline operator
3. grain = message/unit AND session).
4. CORE REFRAME: this is relational algebra.
5. Every construct is relation->relation:
6. query (and/or/not predicates) = base relation (row-set
7. grain = session OR message/unit)

## Tests to add

- Acceptance proof: `polylogue find 'auth intersect week:2026-W01'` returns exactly the session_ids in both operand sets
- Acceptance proof: `except` subtracts
- Acceptance proof: `union` dedups.
- Acceptance proof: Cross-lane composition works: `semantic:"X" except ~keyword` combines a vector set and an FTS set.
- Acceptance proof: Macros compose: `@a intersect @b` (with fnm.12).
- Acceptance proof: The combined set flows into the pipeline (`| read`, aggregates).
- Acceptance proof: Parenthesized nesting parses.
- Acceptance proof: Mixed-grain or unsupported operand fails with a typed error, never silently broadens.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools test tests/unit tests/integration -k "query or contract or citation or insight"`

## Pitfalls

- Do not broaden scope beyond the bead acceptance criteria; make a failing test first, then patch the smallest shared seam.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.

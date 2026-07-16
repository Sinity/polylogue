# 050. polylogue-fnm.1 — Aggregates beyond count (sum/avg/min/max/percentiles)

Priority/type/status: **P2 / feature / open**. Lane: **04-read-contract-query-render**. Release: **C-read-contract**. Readiness: **blocked-hard**.

Hard blockers: polylogue-fnm.11

## What the bead says

`group by X | count` is the only aggregate; cost/duration/token questions need sum/avg/percentiles to compose instead of spawning bespoke analyze modes.

## Existing design note

Full target shape (fables ladder item 4): `| group by tool, session.origin | agg count, avg:duration_ms, p90:duration_ms, sum:tokens` — multi-field group by AND named aggregate list AND time bucketing `group by bucket:day(time)` (temporal-bucket machinery already exists in the temporal read view; reuse its bucket functions in the lowering). SQLite computes sum/avg/min/max natively; percentiles via nearest-rank in Python over grouped rows (pattern insights/portfolio.py:107-128). Pipeline stages are hand-parsed OUTSIDE the Lark grammar (~expression.py:1574/:2777) — no grammar change for the stage itself. Chain: stage parser -> AST dataclass + to_payload (pattern :312-478) -> QueryUnitPipelineStage union + assembly (:511-540; aggregate is currently Literal['count']) -> executor (unit_results.py/plan_execution.py) -> SQL SELECT-list on per-unit sql_query_method -> metadata.py aggregate_metrics + multi-field aggregate_group_fields -> shell_completion_values.py -> render openapi/cli-output-schemas/cli-reference. This is what converts the DSL from counting console to the analytics engine the web aggregate view and saved-view defaults sit on. Line refs pre-07-03; re-locate.

## Acceptance criteria

- On the live archive `messages where ... | group by tool | agg count, avg:duration_ms, p90:duration_ms` returns per-group rows with each named metric column; sum/avg/min/max lower to native SQLite aggregates and percentiles compute via nearest-rank in Python over grouped rows. Verify: pytest over a seeded corpus asserts column presence and computed values.
- Multi-field group-by (`group by tool, session.origin`) and time bucketing (`group by bucket:day(time)`) reuse the temporal read-view bucket functions.
- Unsupported agg names/fields error naming the unit, the metric, and the supported set (the fnm.11 group-by error pattern).
- The QueryUnitPipelineStageSpec aggregate union is widened from Literal['count'] and round-trips through to_payload; explain_query_expression shows the new aggregate. Verify: `devtools render openapi && devtools render cli-output-schemas && devtools render cli-reference` regen and `devtools render all --check` pass.

## Static mechanism / likely defect

Issue description localizes the mechanism: `group by X | count` is the only aggregate; cost/duration/token questions need sum/avg/percentiles to compose instead of spawning bespoke analyze modes. Design direction: Full target shape (fables ladder item 4): `| group by tool, session.origin | agg count, avg:duration_ms, p90:duration_ms, sum:tokens` — multi-field group by AND named aggregate list AND time bucketing `group by bucket:day(time)` (temporal-bucket machinery already exists in the temporal read view; reuse its bucket functions in the lowering). SQLite computes sum/avg/min/max natively; percentiles via nearest-rank in Py…

## Source anchors to inspect first

- `polylogue/archive/query/spec.py:440` — SessionQuerySpec is the core read selection object.
- `polylogue/archive/query/spec.py:498` — from_params centralizes parameter lowering.
- `polylogue/cli/query_contracts.py:70` — QueryOutputSpec is current output/render contract seam.
- `polylogue/cli/archive_query.py:164` — CLI archive query execution remains a dense orchestration point.
- `polylogue/daemon/http.py:1734` — Daemon split-archive route already has from_params path for some cases.
- `polylogue/daemon/http.py:3296` — Another daemon path builds SessionQuerySpec from raw query params.
- `polylogue/mcp/query_contracts.py:54` — MCP query normalization has a distinct lowering path.

## Implementation plan

1. Full target shape (fables ladder item 4): `| group by tool, session.origin | agg count, avg:duration_ms, p90:duration_ms, sum:tokens` — multi-field group by AND named aggregate list AND time bucketing `group by bucket:day(time)` (temporal-bucket machinery already exists in the temporal read view
2. reuse its bucket functions in the lowering).
3. SQLite computes sum/avg/min/max natively
4. percentiles via nearest-rank in Python over grouped rows (pattern insights/portfolio.py:107-128).
5. Pipeline stages are hand-parsed OUTSIDE the Lark grammar (~expression.py:1574/:2777) — no grammar change for the stage itself.
6. Chain: stage parser -> AST dataclass + to_payload (pattern :312-478) -> QueryUnitPipelineStage union + assembly (:511-540
7. aggregate is currently Literal['count']) -> executor (unit_results.py/plan_execution.py) -> SQL SELECT-list on per-unit sql_query_method -> metadata.py aggregate_metrics + multi-field aggregate_group_fields -> shell_completion_values.py -> render openapi/cli-output-schemas/cli-reference.

## Tests to add

- Acceptance proof: On the live archive `messages where ...
- Acceptance proof: | group by tool | agg count, avg:duration_ms, p90:duration_ms` returns per-group rows with each named metric column
- Acceptance proof: sum/avg/min/max lower to native SQLite aggregates and percentiles compute via nearest-rank in Python over grouped rows.
- Acceptance proof: Verify: pytest over a seeded corpus asserts column presence and computed values.
- Acceptance proof: Multi-field group-by (`group by tool, session.origin`) and time bucketing (`group by bucket:day(time)`) reuse the temporal read-view bucket functions.
- Acceptance proof: Unsupported agg names/fields error naming the unit, the metric, and the supported set (the fnm.11 group-by error pattern).
- Acceptance proof: The QueryUnitPipelineStageSpec aggregate union is widened from Literal['count'] and round-trips through to_payload
- Acceptance proof: explain_query_expression shows the new aggregate.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools test tests/unit tests/integration -k "query or contract or citation or insight"`

## Pitfalls

- Do not broaden scope beyond the bead acceptance criteria; make a failing test first, then patch the smallest shared seam.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.

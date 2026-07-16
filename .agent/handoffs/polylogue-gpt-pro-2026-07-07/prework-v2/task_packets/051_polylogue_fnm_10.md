# 051. polylogue-fnm.10 — fields/select stage with parent-field projection (first real Transform)

Priority/type/status: **P2 / feature / open**. Lane: **04-read-contract-query-render**. Release: **C-read-contract**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

The upward-access ceiling: session.* fields work for FILTERING on every unit (~25 scoped fields, metadata.py:620-658) and for whitelisted group-by, but output shapes are frozen Pydantic payloads that hardcode exactly two parent fields (MessageQueryRowPayload carries origin+title, payloads.py:~1265-1280). `messages where session.repo:polylogue AND text:timeout | fields text, occurred_at, session.title, session.repo` — the sessions join is already paid at filter time; projection means emitting columns the lowering already touches.

## Existing design note

Land it as the first real Transform, fulfilling the QueryUnitTransformStage reservation (expression.py:376-386, 'never produced by the current parser'). Chain: hand-parsed stage keyword 'fields'/'select' -> Transform(name='select', args=[field list validated against the unit's field registry + session.* scoped family] ) -> lowering appends the requested columns to the SELECT list (parent columns via the existing sessions join) -> output becomes a generic row payload (dict-shaped, field-name keyed) emitted ALONGSIDE the typed default payloads, not replacing them — existing consumers keep their frozen shapes, `fields` opts into the generic one. Registry: mark projectable fields per unit in metadata.py so completions + validation share one source. Note partial overlap: field selection for ATTACHED units landed (867b1d094 era); this bead is projection on the PRIMARY unit rows. Regen: render openapi + cli-output-schemas (new generic payload model), completions, cli-reference.

## Acceptance criteria

- `messages where session.repo:polylogue AND text:timeout | fields text, occurred_at, session.title, session.repo` returns generic dict-shaped rows keyed by requested field name, emitted alongside (not replacing) the typed MessageQueryRowPayload. Verify: pytest asserts row shape and that the frozen typed default payload is unchanged.
- Requested parent fields resolve through the already-paid sessions join; field names are validated against the unit's field registry + the session.* scoped family (metadata.py); an unknown field errors listing supported fields.
- The QueryUnitTransformStage reservation (expression.py:377) is now actually produced by the parser for the `fields`/`select` keyword; explain shows the transform stage.
- Regen: `devtools render openapi && devtools render cli-output-schemas` emit the new generic payload model and `devtools render all --check` passes.

## Static mechanism / likely defect

Issue description localizes the mechanism: The upward-access ceiling: session.* fields work for FILTERING on every unit (~25 scoped fields, metadata.py:620-658) and for whitelisted group-by, but output shapes are frozen Pydantic payloads that hardcode exactly two parent fields (MessageQueryRowPayload carries origin+title, payloads.py:~1265-1280). `messages where session.repo:polylogue AND text:timeout | fields text, occurred_at, session.title, session.repo` — the sessions join is already paid at filter time; projection means emitting columns the lowering a… Design direction: Land it as the first real Transform, fulfilling the QueryUnitTransformStage reservation (expression.py:376-386, 'never produced by the current parser'). Chain: hand-parsed stage keyword 'fields'/'select' -> Transform(name='select', args=[field list validated against the unit's field registry + session.* scoped family] ) -> lowering appends the requested columns to the SELECT list (parent columns via the existing ses…

## Source anchors to inspect first

- `polylogue/archive/query/spec.py:440` — SessionQuerySpec is the core read selection object.
- `polylogue/archive/query/spec.py:498` — from_params centralizes parameter lowering.
- `polylogue/cli/query_contracts.py:70` — QueryOutputSpec is current output/render contract seam.
- `polylogue/cli/archive_query.py:164` — CLI archive query execution remains a dense orchestration point.
- `polylogue/daemon/http.py:1734` — Daemon split-archive route already has from_params path for some cases.
- `polylogue/daemon/http.py:3296` — Another daemon path builds SessionQuerySpec from raw query params.
- `polylogue/mcp/query_contracts.py:54` — MCP query normalization has a distinct lowering path.

## Implementation plan

1. Land it as the first real Transform, fulfilling the QueryUnitTransformStage reservation (expression.py:376-386, 'never produced by the current parser').
2. Chain: hand-parsed stage keyword 'fields'/'select' -> Transform(name='select', args=[field list validated against the unit's field registry + session.* scoped family] ) -> lowering appends the requested columns to the SELECT list (parent columns via the existing sessions join) -> output becomes a generic row payload (dict-shaped, field-name keyed) emitted ALONGSIDE the typed default payloads, not replacing them — …
3. Registry: mark projectable fields per unit in metadata.py so completions + validation share one source.
4. Note partial overlap: field selection for ATTACHED units landed (867b1d094 era)
5. this bead is projection on the PRIMARY unit rows.
6. Regen: render openapi + cli-output-schemas (new generic payload model), completions, cli-reference.

## Tests to add

- Acceptance proof: `messages where session.repo:polylogue AND text:timeout | fields text, occurred_at, session.title, session.repo` returns generic dict-shaped rows keyed by requested field name, emitted alongside (not replacing) the typed MessageQueryRowPayload.
- Acceptance proof: Verify: pytest asserts row shape and that the frozen typed default payload is unchanged.
- Acceptance proof: Requested parent fields resolve through the already-paid sessions join
- Acceptance proof: field names are validated against the unit's field registry + the session.* scoped family (metadata.py)
- Acceptance proof: an unknown field errors listing supported fields.
- Acceptance proof: The QueryUnitTransformStage reservation (expression.py:377) is now actually produced by the parser for the `fields`/`select` keyword
- Acceptance proof: explain shows the transform stage.
- Acceptance proof: Regen: `devtools render openapi && devtools render cli-output-schemas` emit the new generic payload model and `devtools render all --check` passes.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools test tests/unit tests/integration -k "query or contract or citation or insight"`

## Pitfalls

- Do not broaden scope beyond the bead acceptance criteria; make a failing test first, then patch the smallest shared seam.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.

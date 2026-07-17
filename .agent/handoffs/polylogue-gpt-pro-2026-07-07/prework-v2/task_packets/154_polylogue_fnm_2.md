# 154. polylogue-fnm.2 — Projection predicates/windows + render/layout stages on attached units

Priority/type/status: **P2 / feature / open**. Lane: **04-read-contract-query-render**. Release: **C-read-contract**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

Declared predicates/windows on attached units (e.g. with messages[role:user, last:20]) and render/layout stages so read packages and demos are declarable in the query rather than per-view flags. Direction: .agent/includes/fables-poly-findings.md.

## Existing design note

Two layers: (1) predicates/windows on attached units — extend the with-stage parse (hand-parsed pipeline region, expression.py ~:1484-1601 where WITH_PROJECTION_SUPPORTED_UNITS is enforced) to accept per-unit bracket args (messages[role:user, last:20]); lower onto the existing exact-session-id fetch in attached_units.py (caps exist: _MAX_ROWS_PER_SESSION=200; field selection landed 867b1d048 — extend that payload, don't fork it). (2) render/layout stages — new pipeline stage kind (same touchpoint chain as aggregates: stage parser -> AST/to_payload -> executor -> registry -> completions -> render regen) that binds a read-package/render profile to the query result. Keep grammar untouched (stages are hand-parsed); regenerate openapi/cli-output-schemas/cli-reference; explain payloads pick stages up via to_payload.

## Acceptance criteria

- `... with messages[role:user, last:20]` parses per-unit bracket predicates/windows in the hand-parsed with-stage region and lowers onto the existing exact-session-id fetch in attached_units.py, respecting _MAX_ROWS_PER_SESSION and extending the landed field-selection payload rather than forking it. Verify: pytest asserts filtered/windowed attached rows and cap enforcement.
- A new render/layout pipeline stage binds a read-package/render profile to the result and is picked up by explain via to_payload.
- The Lark grammar file is unchanged (stages stay hand-parsed). Verify: grammar-file diff is empty.
- openapi/cli-output-schemas/cli-reference regens pass `devtools render all --check`.

## Static mechanism / likely defect

Issue description localizes the mechanism: Declared predicates/windows on attached units (e.g. with messages[role:user, last:20]) and render/layout stages so read packages and demos are declarable in the query rather than per-view flags. Direction: .agent/includes/fables-poly-findings.md. Design direction: Two layers: (1) predicates/windows on attached units — extend the with-stage parse (hand-parsed pipeline region, expression.py ~:1484-1601 where WITH_PROJECTION_SUPPORTED_UNITS is enforced) to accept per-unit bracket args (messages[role:user, last:20]); lower onto the existing exact-session-id fetch in attached_units.py (caps exist: _MAX_ROWS_PER_SESSION=200; field selection landed 867b1d048 — extend that payload, d…

## Source anchors to inspect first

- `polylogue/archive/query/spec.py:440` — SessionQuerySpec is the core read selection object.
- `polylogue/archive/query/spec.py:498` — from_params centralizes parameter lowering.
- `polylogue/cli/query_contracts.py:70` — QueryOutputSpec is current output/render contract seam.
- `polylogue/cli/archive_query.py:164` — CLI archive query execution remains a dense orchestration point.
- `polylogue/daemon/http.py:1734` — Daemon split-archive route already has from_params path for some cases.
- `polylogue/daemon/http.py:3296` — Another daemon path builds SessionQuerySpec from raw query params.
- `polylogue/mcp/query_contracts.py:54` — MCP query normalization has a distinct lowering path.

## Implementation plan

1. Two layers: (1) predicates/windows on attached units — extend the with-stage parse (hand-parsed pipeline region, expression.py ~:1484-1601 where WITH_PROJECTION_SUPPORTED_UNITS is enforced) to accept per-unit bracket args (messages[role:user, last:20])
2. lower onto the existing exact-session-id fetch in attached_units.py (caps exist: _MAX_ROWS_PER_SESSION=200
3. field selection landed 867b1d048 — extend that payload, don't fork it).
4. (2) render/layout stages — new pipeline stage kind (same touchpoint chain as aggregates: stage parser -> AST/to_payload -> executor -> registry -> completions -> render regen) that binds a read-package/render profile to the query result.
5. Keep grammar untouched (stages are hand-parsed)
6. regenerate openapi/cli-output-schemas/cli-reference
7. explain payloads pick stages up via to_payload.

## Tests to add

- Acceptance proof: `...
- Acceptance proof: with messages[role:user, last:20]` parses per-unit bracket predicates/windows in the hand-parsed with-stage region and lowers onto the existing exact-session-id fetch in attached_units.py, respecting _MAX_ROWS_PER_SESSION and extending the landed field-selection payload rather than forking it.
- Acceptance proof: Verify: pytest asserts filtered/windowed attached rows and cap enforcement.
- Acceptance proof: A new render/layout pipeline stage binds a read-package/render profile to the result and is picked up by explain via to_payload.
- Acceptance proof: The Lark grammar file is unchanged (stages stay hand-parsed).
- Acceptance proof: Verify: grammar-file diff is empty.
- Acceptance proof: openapi/cli-output-schemas/cli-reference regens pass `devtools render all --check`.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools test tests/unit tests/integration -k "query or contract or citation or insight"`

## Pitfalls

- Do not broaden scope beyond the bead acceptance criteria; make a failing test first, then patch the smallest shared seam.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.

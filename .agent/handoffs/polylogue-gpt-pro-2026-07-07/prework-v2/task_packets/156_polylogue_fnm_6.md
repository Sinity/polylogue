# 156. polylogue-fnm.6 — Wire the terminal stage to projections: | read / | context-image

Priority/type/status: **P2 / feature / open**. Lane: **04-read-contract-query-render**. Release: **C-read-contract**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

QueryUnitTransformStage is reserved-never-parsed and terminal args are reserved for future actions. Wire the terminal stage so a pipeline can end in a projection: `sessions where ... | read view:dialogue` / `| context-image budget:4000` — queries become complete read/context programs. Same hand-parsed stage chain as aggregates; the read/context compilers already accept the needed specs.

## Existing design note

The terminal args slot is explicitly reserved for this (expression.py:397-407 'future actions (read view, analyze mode, bundle kind)'). Target: `messages where session.repo:x AND text:timeout | limit 40 | context-image budget:4000` and `sessions where ... | read view:temporal` and `| bundle:handoff` — the DSL becomes the single language from selection through composition to rendering. Altitude: terminal keywords in the hand-parsed stage region + an executor registry dispatching to the existing compilers — compile_context already accepts seed queries; read views resolve via read_view_registry. Payload: the terminal's output replaces the row payload with the projection artifact envelope (typed per terminal kind). This is the convergence point the demo radar kept circling (query + projection + renderer).

## Acceptance criteria

- `sessions where ... | read view:temporal`, `messages where ... | limit 40 | context-image budget:4000`, and `... | bundle:handoff` execute end-to-end: the terminal stage dispatches through an executor registry to the existing compile_context / read_view_registry compilers, and the terminal's output replaces the row payload with a typed projection-artifact envelope per terminal kind. Verify: pytest asserts each of the three forms returns its envelope type.
- Unknown terminal keywords/args error naming the terminal and the supported kinds.
- explain shows the terminal stage via to_payload; completions offer the terminal keywords and their read-view/bundle argument values from the same registries as fnm.4.
- Regen passes `devtools render all --check`.

## Static mechanism / likely defect

Issue description localizes the mechanism: QueryUnitTransformStage is reserved-never-parsed and terminal args are reserved for future actions. Wire the terminal stage so a pipeline can end in a projection: `sessions where ... | read view:dialogue` / `| context-image budget:4000` — queries become complete read/context programs. Same hand-parsed stage chain as aggregates; the read/context compilers already accept the needed specs. Design direction: The terminal args slot is explicitly reserved for this (expression.py:397-407 'future actions (read view, analyze mode, bundle kind)'). Target: `messages where session.repo:x AND text:timeout | limit 40 | context-image budget:4000` and `sessions where ... | read view:temporal` and `| bundle:handoff` — the DSL becomes the single language from selection through composition to rendering. Altitude: terminal keywords in …

## Source anchors to inspect first

- `polylogue/archive/query/spec.py:440` — SessionQuerySpec is the core read selection object.
- `polylogue/archive/query/spec.py:498` — from_params centralizes parameter lowering.
- `polylogue/cli/query_contracts.py:70` — QueryOutputSpec is current output/render contract seam.
- `polylogue/cli/archive_query.py:164` — CLI archive query execution remains a dense orchestration point.
- `polylogue/daemon/http.py:1734` — Daemon split-archive route already has from_params path for some cases.
- `polylogue/daemon/http.py:3296` — Another daemon path builds SessionQuerySpec from raw query params.
- `polylogue/mcp/query_contracts.py:54` — MCP query normalization has a distinct lowering path.

## Implementation plan

1. The terminal args slot is explicitly reserved for this (expression.py:397-407 'future actions (read view, analyze mode, bundle kind)').
2. Target: `messages where session.repo:x AND text:timeout | limit 40 | context-image budget:4000` and `sessions where ...
3. | read view:temporal` and `| bundle:handoff` — the DSL becomes the single language from selection through composition to rendering.
4. Altitude: terminal keywords in the hand-parsed stage region + an executor registry dispatching to the existing compilers — compile_context already accepts seed queries
5. read views resolve via read_view_registry.
6. Payload: the terminal's output replaces the row payload with the projection artifact envelope (typed per terminal kind).
7. This is the convergence point the demo radar kept circling (query + projection + renderer).

## Tests to add

- Acceptance proof: `sessions where ...
- Acceptance proof: | read view:temporal`, `messages where ...
- Acceptance proof: | limit 40 | context-image budget:4000`, and `...
- Acceptance proof: | bundle:handoff` execute end-to-end: the terminal stage dispatches through an executor registry to the existing compile_context / read_view_registry compilers, and the terminal's output replaces the row payload with a typed projection-artifact envelope per terminal kind.
- Acceptance proof: Verify: pytest asserts each of the three forms returns its envelope type.
- Acceptance proof: Unknown terminal keywords/args error naming the terminal and the supported kinds.
- Acceptance proof: explain shows the terminal stage via to_payload
- Acceptance proof: completions offer the terminal keywords and their read-view/bundle argument values from the same registries as fnm.4.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools test tests/unit tests/integration -k "query or contract or citation or insight"`

## Pitfalls

- Do not broaden scope beyond the bead acceptance criteria; make a failing test first, then patch the smallest shared seam.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.

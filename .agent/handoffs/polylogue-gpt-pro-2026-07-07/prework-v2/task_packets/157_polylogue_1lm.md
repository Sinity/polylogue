# 157. polylogue-1lm — Composable transcript views: selector x transform x budget algebra

Priority/type/status: **P2 / task / open**. Lane: **04-read-contract-query-render**. Release: **C-read-contract**. Readiness: **blocked-hard**.

Hard blockers: polylogue-jnj.1

## What the bead says

'Prose-only' is one point in a space the operator keeps requesting by example: user messages plus directly-adjacent agent replies (raw-log 07-02 — what the agent intended to report, minus the toil); tool outputs truncated from the middle beyond N lines (raw-log 06-23); decisions-only; tool-skeleton (calls + outcomes, no bodies); failure-slices; reboot-with-refs (37t.3); compact recaps for mass export ('every sinex-related chatlog in compact form for gptpro', 06-18). One algebra: SELECTOR (role, material-origin, block type, adjacency, outcome class, topic) x TRANSFORM per class (verbatim | refify | truncate-middle(n) | fold-to-line | recap) x BUDGET (per-class allowances, tail/head bias). Prose-only itself conflates authored prose, protocol chatter, and generated packs — material_origin already distinguishes them; the algebra should too.

## Existing design note

(1) Extend ProjectionSpec (jnj.1) with typed selector predicates (reuse the DSL block-predicate grammar — no second filter language) and per-class TransformSpec; compile_context and renderers consume the same spec (4p1's Projection axis, deepened). (2) Adjacency selectors are the novel primitive: adjacent-to(role:user, distance<=1, after) via window functions over position. (3) Transforms compose with ap7 semantic renderers; truncate-middle keeps first/last K lines with an omission marker carrying the block ref (expandable, jgp). (4) Named presets as registry entries: prose, dialogue, skeleton, decisions, forensic, reboot, compact-export — uniform across read --view, MCP detail levels, export profiles, context compilation. (5) Acceptance driven by the raw-log examples: each expressible as a one-line spec, no code.

## Acceptance criteria

The three raw-log examples work as presets/inline specs on the live archive; compile_context and read share the machinery; presets visible to completions; omission markers always carry resolvable refs.

## Static mechanism / likely defect

Issue description localizes the mechanism: 'Prose-only' is one point in a space the operator keeps requesting by example: user messages plus directly-adjacent agent replies (raw-log 07-02 — what the agent intended to report, minus the toil); tool outputs truncated from the middle beyond N lines (raw-log 06-23); decisions-only; tool-skeleton (calls + outcomes, no bodies); failure-slices; reboot-with-refs (37t.3); compact recaps for mass export ('every sinex-related chatlog in compact form for gptpro', 06-18). One algebra: SELECTOR (role, material-origin, bl… Design direction: (1) Extend ProjectionSpec (jnj.1) with typed selector predicates (reuse the DSL block-predicate grammar — no second filter language) and per-class TransformSpec; compile_context and renderers consume the same spec (4p1's Projection axis, deepened). (2) Adjacency selectors are the novel primitive: adjacent-to(role:user, distance<=1, after) via window functions over position. (3) Transforms compose with ap7 semantic r…

## Source anchors to inspect first

- `polylogue/archive/query/spec.py:440` — SessionQuerySpec is the core read selection object.
- `polylogue/archive/query/spec.py:498` — from_params centralizes parameter lowering.
- `polylogue/cli/query_contracts.py:70` — QueryOutputSpec is current output/render contract seam.
- `polylogue/cli/archive_query.py:164` — CLI archive query execution remains a dense orchestration point.
- `polylogue/daemon/http.py:1734` — Daemon split-archive route already has from_params path for some cases.
- `polylogue/daemon/http.py:3296` — Another daemon path builds SessionQuerySpec from raw query params.
- `polylogue/mcp/query_contracts.py:54` — MCP query normalization has a distinct lowering path.

## Implementation plan

1. (1) Extend ProjectionSpec (jnj.1) with typed selector predicates (reuse the DSL block-predicate grammar — no second filter language) and per-class TransformSpec
2. compile_context and renderers consume the same spec (4p1's Projection axis, deepened).
3. (2) Adjacency selectors are the novel primitive: adjacent-to(role:user, distance<=1, after) via window functions over position.
4. (3) Transforms compose with ap7 semantic renderers
5. truncate-middle keeps first/last K lines with an omission marker carrying the block ref (expandable, jgp).
6. (4) Named presets as registry entries: prose, dialogue, skeleton, decisions, forensic, reboot, compact-export — uniform across read --view, MCP detail levels, export profiles, context compilation.
7. (5) Acceptance driven by the raw-log examples: each expressible as a one-line spec, no code.

## Tests to add

- Acceptance proof: The three raw-log examples work as presets/inline specs on the live archive
- Acceptance proof: compile_context and read share the machinery
- Acceptance proof: presets visible to completions
- Acceptance proof: omission markers always carry resolvable refs.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools test tests/unit tests/integration -k "query or contract or citation or insight"`

## Pitfalls

- Do not broaden scope beyond the bead acceptance criteria; make a failing test first, then patch the smallest shared seam.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.

# 039. polylogue-4p1 — Decision: one read algebra — Query x Projection x Render as the only read contract

Priority/type/status: **P2 / task / open**. Lane: **04-read-contract-query-render**. Release: **C-read-contract**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

The read side has N surfaces multiplying independently: CLI verbs with per-view flags, analyze boolean modes, MCP tools (~61 read tools, many being named parameterizations of the same underlying read), web routes, read-view profiles, read-package layouts. jnj collapses CLI flags into ProjectionSpec; fnm.6 wires DSL terminal stages to projections; t46 deletes parallel dispatch. This bead records the target those programs converge on, so future surface work has a stated invariant instead of rediscovering the direction: EVERY read surface lowers to (QuerySpec: which units) x (ProjectionSpec: which shape/fields/views) x (RenderSpec: which format/layout/budget) — and a new surface capability is a new spec value, never a new code path.

## Existing design note

Doctrine bead, deliverable is the recorded contract + a conformance inventory, not a rewrite: (1) Write the algebra down in docs/architecture-spine.md — the three spec types, their composition law, and the rule that MCP tools / analyze modes / CLI views / web routes are NAMED PRESETS over the algebra (a preset = a (Q,P,R) triple with defaults, registered via the declare-once machinery). (2) Conformance inventory: classify every existing read surface as conformant / preset-expressible / algebra-hole (needs a new spec capability) / genuinely-other; the holes become the priority list for fnm/jnj slices. Include the current projection/render duplication explicitly: ContentProjectionSpec vs ProjectionSpec body policy; RenderFormat vs session output formats; RenderDestination vs read invocation delivery; RenderSpec.layout vs read-view/profile metadata. (3) The payoff criterion for future expansion, stated as policy: adding a read affordance (new MCP tool, new analyze mode, new web panel, static HTML export, variant/translation view) must be expressible as a preset/spec value; if it requires a new bespoke read path, the algebra gains or reuses the capability FIRST. (4) Explicit scope guard: writes, ops, and maintenance flows are NOT in this algebra — do not force-fit them.

## Acceptance criteria

docs/architecture-spine.md gains a 'One read algebra' entry under Major Decisions naming SelectionSpec x ProjectionSpec x RenderSpec (QueryProjectionSpec) as the sole read contract and presets as named (S,P,R) triples, with projection_spec.py cited as the existing realization. A conformance inventory (table in the doc or a linked docs/plans/*.yaml manifest checked by devtools verify manifests) lists every read surface with its classification; every algebra-hole row links to the fnm or jnj bead that closes it. The expansion policy is stated as doctrine: a new read affordance (new MCP tool, analyze mode, web panel, static HTML/variant/translation view) must be expressible as a preset/spec value, and if it needs a bespoke read path the algebra gains or reuses the capability FIRST. The writes/ops/maintenance scope guard is recorded. This bead requires no code rewrite; it references the existing spec types and hands the enumerated holes to fnm/jnj/rlsb. devtools render all --check passes after the doc edit.

## Static mechanism / likely defect

Issue description localizes the mechanism: The read side has N surfaces multiplying independently: CLI verbs with per-view flags, analyze boolean modes, MCP tools (~61 read tools, many being named parameterizations of the same underlying read), web routes, read-view profiles, read-package layouts. jnj collapses CLI flags into ProjectionSpec; fnm.6 wires DSL terminal stages to projections; t46 deletes parallel dispatch. This bead records the target those programs converge on, so future surface work has a stated invariant instead of rediscovering the directi… Design direction: Doctrine bead, deliverable is the recorded contract + a conformance inventory, not a rewrite: (1) Write the algebra down in docs/architecture-spine.md — the three spec types, their composition law, and the rule that MCP tools / analyze modes / CLI views / web routes are NAMED PRESETS over the algebra (a preset = a (Q,P,R) triple with defaults, registered via the declare-once machinery). (2) Conformance inventory: cl…

## Source anchors to inspect first

- `polylogue/archive/query/spec.py:440` — SessionQuerySpec is the core read selection object.
- `polylogue/archive/query/spec.py:498` — from_params centralizes parameter lowering.
- `polylogue/cli/query_contracts.py:70` — QueryOutputSpec is current output/render contract seam.
- `polylogue/cli/archive_query.py:164` — CLI archive query execution remains a dense orchestration point.
- `polylogue/daemon/http.py:1734` — Daemon split-archive route already has from_params path for some cases.
- `polylogue/daemon/http.py:3296` — Another daemon path builds SessionQuerySpec from raw query params.
- `polylogue/mcp/query_contracts.py:54` — MCP query normalization has a distinct lowering path.

## Implementation plan

1. Doctrine bead, deliverable is the recorded contract + a conformance inventory, not a rewrite: (1) Write the algebra down in docs/architecture-spine.md — the three spec types, their composition law, and the rule that MCP tools / analyze modes / CLI views / web routes are NAMED PRESETS over the algebra (a preset = a (Q,P,R) triple with defaults, registered via the declare-once machinery).
2. (2) Conformance inventory: classify every existing read surface as conformant / preset-expressible / algebra-hole (needs a new spec capability) / genuinely-other
3. the holes become the priority list for fnm/jnj slices.
4. Include the current projection/render duplication explicitly: ContentProjectionSpec vs ProjectionSpec body policy
5. RenderFormat vs session output formats
6. RenderDestination vs read invocation delivery
7. RenderSpec.layout vs read-view/profile metadata.

## Tests to add

- Acceptance proof: docs/architecture-spine.md gains a 'One read algebra' entry under Major Decisions naming SelectionSpec x ProjectionSpec x RenderSpec (QueryProjectionSpec) as the sole read contract and presets as named (S,P,R) triples, with projection_spec.py cited as the existing realization.
- Acceptance proof: A conformance inventory (table in the doc or a linked docs/plans/*.yaml manifest checked by devtools verify manifests) lists every read surface with its classification
- Acceptance proof: every algebra-hole row links to the fnm or jnj bead that closes it.
- Acceptance proof: The expansion policy is stated as doctrine: a new read affordance (new MCP tool, analyze mode, web panel, static HTML/variant/translation view) must be expressible as a preset/spec value, and if it needs a bespoke read path the algebra gains or reuses the capability FIRST.
- Acceptance proof: The writes/ops/maintenance scope guard is recorded.
- Acceptance proof: This bead requires no code rewrite
- Acceptance proof: it references the existing spec types and hands the enumerated holes to fnm/jnj/rlsb.
- Acceptance proof: devtools render all --check passes after the doc edit.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools test tests/unit tests/integration -k "query or contract or citation or insight"`

## Pitfalls

- Do not broaden scope beyond the bead acceptance criteria; make a failing test first, then patch the smallest shared seam.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.

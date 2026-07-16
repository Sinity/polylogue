# 046. polylogue-jnj.1 — Collapse read per-view flags into ProjectionSpec/RenderSpec algebra

Priority/type/status: **P2 / task / open**. Lane: **04-read-contract-query-render**. Release: **C-read-contract**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

read exposes compact algebra (--projection/--render/--spec) alongside per-view flag clusters (--window-hours, --repo-path, --since-hours, --related-limit...). Extend ProjectionSpec/RenderSpec to cover neighbor/correlation/context options FIRST, then remove the aliases. Broad public CLI change, not deletion-only.

## Existing design note

Collapse read per-view flags into the existing Query x Projection x Render algebra, and use this bead to converge the duplication that would otherwise make export/variant work accrete another surface. Current known overlap to resolve or explicitly boundary-test: ProjectionSpec.body_policy/exclude_block_kinds vs ContentProjectionSpec; RenderFormat vs SESSION_OUTPUT_FORMATS/format_session; RenderDestination vs ReadViewInvocation.destination/deliver_content; RenderSpec.layout free strings vs read-view/profile metadata; READ_VIEW_PROJECTION_FAMILIES vs READ_VIEW_PROFILES vs executable handlers.

Implementation should extend ProjectionSpec/RenderSpec only after deciding whether an existing abstraction already owns the concern. Semantic inclusion belongs to ProjectionSpec or ContentProjectionSpec. Rendering owns encoding, destination, timestamp policy, and a named visual/profile/layout choice. HTML/static exports and web reader profiles should consume the same projected payloads rather than inventing export-specific flags. Extend ProjectionSpec/RenderSpec to cover neighbor/correlation/context options first, route existing handlers through the spec, then remove aliases/per-view flags where the algebra is expressive enough.

## Acceptance criteria

read --spec remains the visible contract for composed selection/projection/render state. Existing per-view options for neighbor/correlation/context are represented in ProjectionSpec/RenderSpec or an explicitly named profile contract. At least one duplication pair is removed or converted into a single source of truth with tests; any remaining pair has a documented boundary and drift check. HTML/static export work can select a reader render profile over QueryProjectionSpec without a bespoke export command family. CLI reference, projection docs, read-view profile payloads, and generated schemas are refreshed.

## Static mechanism / likely defect

Issue description localizes the mechanism: read exposes compact algebra (--projection/--render/--spec) alongside per-view flag clusters (--window-hours, --repo-path, --since-hours, --related-limit...). Extend ProjectionSpec/RenderSpec to cover neighbor/correlation/context options FIRST, then remove the aliases. Broad public CLI change, not deletion-only. Design direction: Collapse read per-view flags into the existing Query x Projection x Render algebra, and use this bead to converge the duplication that would otherwise make export/variant work accrete another surface. Current known overlap to resolve or explicitly boundary-test: ProjectionSpec.body_policy/exclude_block_kinds vs ContentProjectionSpec; RenderFormat vs SESSION_OUTPUT_FORMATS/format_session; RenderDestination vs ReadVie…

## Source anchors to inspect first

- `polylogue/archive/query/spec.py:440` — SessionQuerySpec is the core read selection object.
- `polylogue/archive/query/spec.py:498` — from_params centralizes parameter lowering.
- `polylogue/cli/query_contracts.py:70` — QueryOutputSpec is current output/render contract seam.
- `polylogue/cli/archive_query.py:164` — CLI archive query execution remains a dense orchestration point.
- `polylogue/daemon/http.py:1734` — Daemon split-archive route already has from_params path for some cases.
- `polylogue/daemon/http.py:3296` — Another daemon path builds SessionQuerySpec from raw query params.
- `polylogue/mcp/query_contracts.py:54` — MCP query normalization has a distinct lowering path.

## Implementation plan

1. Collapse read per-view flags into the existing Query x Projection x Render algebra, and use this bead to converge the duplication that would otherwise make export/variant work accrete another surface.
2. Current known overlap to resolve or explicitly boundary-test: ProjectionSpec.body_policy/exclude_block_kinds vs ContentProjectionSpec
3. RenderFormat vs SESSION_OUTPUT_FORMATS/format_session
4. RenderDestination vs ReadViewInvocation.destination/deliver_content
5. RenderSpec.layout free strings vs read-view/profile metadata
6. READ_VIEW_PROJECTION_FAMILIES vs READ_VIEW_PROFILES vs executable handlers.
7. Implementation should extend ProjectionSpec/RenderSpec only after deciding whether an existing abstraction already owns the concern.

## Tests to add

- Acceptance proof: read --spec remains the visible contract for composed selection/projection/render state.
- Acceptance proof: Existing per-view options for neighbor/correlation/context are represented in ProjectionSpec/RenderSpec or an explicitly named profile contract.
- Acceptance proof: At least one duplication pair is removed or converted into a single source of truth with tests
- Acceptance proof: any remaining pair has a documented boundary and drift check.
- Acceptance proof: HTML/static export work can select a reader render profile over QueryProjectionSpec without a bespoke export command family.
- Acceptance proof: CLI reference, projection docs, read-view profile payloads, and generated schemas are refreshed.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools test tests/unit tests/integration -k "query or contract or citation or insight"`

## Pitfalls

- Do not broaden scope beyond the bead acceptance criteria; make a failing test first, then patch the smallest shared seam.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.

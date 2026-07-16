# 077. polylogue-rlsb — Variant-aware projection, query, and reader render profiles

Priority/type/status: **P1 / feature / open**. Lane: **07-content-variants**. Release: **E-content-variants**. Readiness: **blocked-hard**.

Hard blockers: polylogue-arso

## What the bead says

Why: translated or simplified content should be selectable, readable, exported, and queried through the existing read algebra, not through bespoke translation flags or export modes. Query results and renderers must label source-vs-variant text so downstream analysis does not lie about the underlying archive.

## Existing design note

Extend the existing Query x Projection x Render algebra without adding a new overlay abstraction. Variant inclusion is semantic projection: add a ProjectionSpec variant_policy (include none/exact/inherited/composed, kinds, target_language, status_policy, coverage_policy, alignment_policy) and an EvidenceFamily/terminal unit surface for variants where needed. Do NOT put source-vs-variant inclusion decisions in RenderSpec. RenderSpec remains delivery/encoding/profile only: format, destination, timestamp policy, out path, and a renderer/profile/layout id that chooses visual arrangement such as original-only, variant-only, dual, interleaved, or hover-source when the projected payload already contains variant lanes.

Before or while implementing variants, audit and converge the existing selection/projection/render duplication: ProjectionSpec.body_policy and exclude_block_kinds overlap with ContentProjectionSpec; RenderFormat overlaps SESSION_OUTPUT_FORMATS; RenderDestination overlaps ReadViewInvocation.destination/deliver_content; RenderSpec.layout is a free string while read-view/profile metadata already exists. Prefer nudging these into one coherent registry/contract over adding another RendererProfile/overlay silo. Export/read commands should become QueryProjectionSpec programs over shared reader/render profiles, not bespoke translation/export paths. Coordinate with fnm.2/fnm.6/fnm.10, jnj.1, bby.11, and 4p1.

## Acceptance criteria

CLI/API/MCP/daemon query/read paths can request variants through ProjectionSpec and existing query/projection stages. JSON payloads label original source text and variant text distinctly, including exact/inherited/composed coverage and aligned refs. Queries can find variants by target, kind, language, status, and alignment, and source rows can report variant coverage without treating translated text as original evidence. Markdown/HTML renderers support original, variant, and dual/interleaved visual profiles by consuming projected variant lanes, not by deciding semantic inclusion themselves. The implementation includes a concrete convergence audit/fix for the current projection/render overlap: no duplicated new variant-layout abstraction, and any retained RenderSpec/read-view/profile/format/content-projection split has a documented boundary and tests. Generated CLI reference, OpenAPI/output schemas, projection docs, and relevant read-view profile metadata are updated.

## Static mechanism / likely defect

Issue description localizes the mechanism: Why: translated or simplified content should be selectable, readable, exported, and queried through the existing read algebra, not through bespoke translation flags or export modes. Query results and renderers must label source-vs-variant text so downstream analysis does not lie about the underlying archive. Design direction: Extend the existing Query x Projection x Render algebra without adding a new overlay abstraction. Variant inclusion is semantic projection: add a ProjectionSpec variant_policy (include none/exact/inherited/composed, kinds, target_language, status_policy, coverage_policy, alignment_policy) and an EvidenceFamily/terminal unit surface for variants where needed. Do NOT put source-vs-variant inclusion decisions in Render…

## Source anchors to inspect first

- `polylogue/core/identity_law.py:42` — Current identity includes variant_index only for provider sibling messages, not transformed content variants.
- `polylogue/storage/sqlite/queries/message_query_reads.py:34` — Read model projects message variant_index as branch_index.
- `polylogue/surfaces/payloads.py:747` — Reader payload maps message variant_index to branch_index.
- `polylogue/daemon/compare.py:220` — Compare/alignment semantics exist for message diffing, not content-transform alignment.
- `polylogue/core/refs.py` — Existing ref model should be extended, not bypassed.
- `polylogue/storage/sqlite/archive_tiers/user_write.py:901` — Findings should reuse assertion/candidate lifecycle.
- `polylogue/surfaces/payloads.py:747` — Surface payloads currently expose branch/message refs; expand consistently.
- `polylogue/mcp/payloads.py:377` — MCP message payloads carry variant/branch metadata today.
- `polylogue/archive/query/spec.py:440` — SessionQuerySpec is the core read selection object.
- `polylogue/archive/query/spec.py:498` — from_params centralizes parameter lowering.
- `polylogue/cli/query_contracts.py:70` — QueryOutputSpec is current output/render contract seam.
- `polylogue/cli/archive_query.py:164` — CLI archive query execution remains a dense orchestration point.
- `polylogue/daemon/http.py:1734` — Daemon split-archive route already has from_params path for some cases.
- `polylogue/daemon/http.py:3296` — Another daemon path builds SessionQuerySpec from raw query params.
- `polylogue/mcp/query_contracts.py:54` — MCP query normalization has a distinct lowering path.

## Implementation plan

1. Extend the existing Query x Projection x Render algebra without adding a new overlay abstraction.
2. Variant inclusion is semantic projection: add a ProjectionSpec variant_policy (include none/exact/inherited/composed, kinds, target_language, status_policy, coverage_policy, alignment_policy) and an EvidenceFamily/terminal unit surface for variants where needed.
3. Do NOT put source-vs-variant inclusion decisions in RenderSpec.
4. RenderSpec remains delivery/encoding/profile only: format, destination, timestamp policy, out path, and a renderer/profile/layout id that chooses visual arrangement such as original-only, variant-only, dual, interleaved, or hover-source when the projected payload already contains variant lanes.
5. Before or while implementing variants, audit and converge the existing selection/projection/render duplication: ProjectionSpec.body_policy and exclude_block_kinds overlap with ContentProjectionSpec
6. RenderFormat overlaps SESSION_OUTPUT_FORMATS
7. RenderDestination overlaps ReadViewInvocation.destination/deliver_content

## Tests to add

- Acceptance proof: CLI/API/MCP/daemon query/read paths can request variants through ProjectionSpec and existing query/projection stages.
- Acceptance proof: JSON payloads label original source text and variant text distinctly, including exact/inherited/composed coverage and aligned refs.
- Acceptance proof: Queries can find variants by target, kind, language, status, and alignment, and source rows can report variant coverage without treating translated text as original evidence.
- Acceptance proof: Markdown/HTML renderers support original, variant, and dual/interleaved visual profiles by consuming projected variant lanes, not by deciding semantic inclusion themselves.
- Acceptance proof: The implementation includes a concrete convergence audit/fix for the current projection/render overlap: no duplicated new variant-layout abstraction, and any retained RenderSpec/read-view/profile/format/content-projection split has a documented boundary and tests.
- Acceptance proof: Generated CLI reference, OpenAPI/output schemas, projection docs, and relevant read-view profile metadata are updated.

## Verification commands

- `devtools test <focused tests added for this bead>`

## Pitfalls

- Do not overwrite original message/block text; variants are separate evidence-linked objects.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.

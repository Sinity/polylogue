# 155. polylogue-fnm.4 — Shell completion + fuzzy selection as read-only projections of the grammar registries

Priority/type/status: **P2 / feature / open**. Lane: **04-read-contract-query-render**. Release: **C-read-contract**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

Completion/query-builder metadata built on the same grammar+registries used by CLI/MCP/daemon/web — not a second parser. Substantial substrate exists (query_completions tool, projection-unit completions landed 07-03); remaining scope per issue. GH issue thread (body + comments) is input, not authority; this bead's scope statement wins where they conflict.

## Existing design note

Scope per gh#1844 minus what landed: query_completions MCP tool, projection-unit completions, and dynamic shell completions (polylogue config completions --shell) exist. Remaining: completion coverage for pipeline stages/operators/read-view names sourced from the SAME registries (metadata.py descriptors, read_view_registry, action contracts — no second vocabulary); bounded archive-backed value providers (origins, repos, tags) with latency caps; fzf-style fuzzy selection beyond the select verb. Acceptance: a completion snapshot test asserts every grammar-reachable field/unit/stage/view appears in the completion payload (registry-diff test, so new DSL work cannot silently miss completions).

## Acceptance criteria

- A registry-diff snapshot test asserts that every grammar-reachable field/unit/pipeline-stage/read-view name (enumerated from metadata.py descriptors, read_view_registry, and operations.action_contracts.ACTION_CONTRACTS) appears in the completion payload, so new DSL work cannot silently miss completions. Verify: pytest registry-diff test.
- Completions for pipeline stages, operators, and read-view names are sourced from those same registries (no second vocabulary — completions.py already imports query_unit_descriptor/terminal_query_pipeline_stage_infos/ACTION_CONTRACTS at completions.py:14-32).
- Archive-backed value providers (origins, repos, tags) return under a stated latency cap. Verify: test measures provider latency against the cap.
- Both `polylogue config completions --shell` and the query_completions MCP tool resolve pipeline-stage and read-view completions. Verify: a test asserts parity across the two surfaces.

## Static mechanism / likely defect

Issue description localizes the mechanism: Completion/query-builder metadata built on the same grammar+registries used by CLI/MCP/daemon/web — not a second parser. Substantial substrate exists (query_completions tool, projection-unit completions landed 07-03); remaining scope per issue. GH issue thread (body + comments) is input, not authority; this bead's scope statement wins where they conflict. Design direction: Scope per gh#1844 minus what landed: query_completions MCP tool, projection-unit completions, and dynamic shell completions (polylogue config completions --shell) exist. Remaining: completion coverage for pipeline stages/operators/read-view names sourced from the SAME registries (metadata.py descriptors, read_view_registry, action contracts — no second vocabulary); bounded archive-backed value providers (origins, re…

## Source anchors to inspect first

- `polylogue/archive/query/spec.py:440` — SessionQuerySpec is the core read selection object.
- `polylogue/archive/query/spec.py:498` — from_params centralizes parameter lowering.
- `polylogue/cli/query_contracts.py:70` — QueryOutputSpec is current output/render contract seam.
- `polylogue/cli/archive_query.py:164` — CLI archive query execution remains a dense orchestration point.
- `polylogue/daemon/http.py:1734` — Daemon split-archive route already has from_params path for some cases.
- `polylogue/daemon/http.py:3296` — Another daemon path builds SessionQuerySpec from raw query params.
- `polylogue/mcp/query_contracts.py:54` — MCP query normalization has a distinct lowering path.

## Implementation plan

1. Scope per gh#1844 minus what landed: query_completions MCP tool, projection-unit completions, and dynamic shell completions (polylogue config completions --shell) exist.
2. Remaining: completion coverage for pipeline stages/operators/read-view names sourced from the SAME registries (metadata.py descriptors, read_view_registry, action contracts — no second vocabulary)
3. bounded archive-backed value providers (origins, repos, tags) with latency caps
4. fzf-style fuzzy selection beyond the select verb.
5. Acceptance: a completion snapshot test asserts every grammar-reachable field/unit/stage/view appears in the completion payload (registry-diff test, so new DSL work cannot silently miss completions).

## Tests to add

- Acceptance proof: A registry-diff snapshot test asserts that every grammar-reachable field/unit/pipeline-stage/read-view name (enumerated from metadata.py descriptors, read_view_registry, and operations.action_contracts.ACTION_CONTRACTS) appears in the completion payload, so new DSL work cannot silently miss completions.
- Acceptance proof: Verify: pytest registry-diff test.
- Acceptance proof: Completions for pipeline stages, operators, and read-view names are sourced from those same registries (no second vocabulary — completions.py already imports query_unit_descriptor/terminal_query_pipeline_stage_infos/ACTION_CONTRACTS at completions.py:14-32).
- Acceptance proof: Archive-backed value providers (origins, repos, tags) return under a stated latency cap.
- Acceptance proof: Verify: test measures provider latency against the cap.
- Acceptance proof: Both `polylogue config completions --shell` and the query_completions MCP tool resolve pipeline-stage and read-view completions.
- Acceptance proof: Verify: a test asserts parity across the two surfaces.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools test tests/unit tests/integration -k "query or contract or citation or insight"`

## Pitfalls

- Do not broaden scope beyond the bead acceptance criteria; make a failing test first, then patch the smallest shared seam.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.

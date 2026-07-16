# 151. polylogue-t46 — Contracts own surfaces: delete parallel dispatch and the QA middle layer

Priority/type/status: **P2 / epic / open**. Lane: **04-read-contract-query-render**. Release: **C-read-contract**. Readiness: **epic-needs-child-closure**.

## What the bead says

Make existing contracts (query DSL, terminal units, refs, read-view profiles, action/route contracts, generated docs/schemas) the actual owners of behavior; delete hand-written parallel surfaces. GH issue thread (body + comments) is input, not authority; this bead's scope statement wins where they conflict.

## Existing design note

First slices (read-only audit 2026-07-02, Herschel): config aliases (daemon_host/daemon_port, top-level observability), hidden/help + historical CLI aliases (find_help, -n, --full, demo-shelf --bundle), status JSON compatibility aliases, origin->provider projection bridges in outputs, browser-capture old synthetic-ID recovery. Rule per slice: the contract (DSL/registry/generated schema) becomes the owner, the parallel surface is deleted in the same PR — replacement-first, no compatibility fronts. Regenerate render surfaces after each (openapi, cli-output-schemas, cli-reference).

## Acceptance criteria

- For each listed first slice — config aliases (daemon_host/daemon_port, top-level observability); hidden/help + historical CLI aliases (find_help, -n, --full, demo-shelf --bundle); status JSON compatibility aliases; origin->provider projection bridges in outputs; browser-capture old synthetic-ID recovery — the parallel hand-written surface is DELETED in the same PR that makes the contract (DSL/registry/generated schema) the sole owner (grep confirms the alias/bridge is gone, no compatibility front left).
- After each slice, `devtools render openapi && devtools render cli-output-schemas && devtools render cli-reference` are regenerated and committed; `devtools render all --check` is clean.
- `devtools verify` is green after each slice; grep for each deleted alias name returns nothing (or only removal-asserting tests).
- The epic closes when all listed first slices are landed or explicitly re-scoped into child beads.

## Static mechanism / likely defect

Issue description localizes the mechanism: Make existing contracts (query DSL, terminal units, refs, read-view profiles, action/route contracts, generated docs/schemas) the actual owners of behavior; delete hand-written parallel surfaces. GH issue thread (body + comments) is input, not authority; this bead's scope statement wins where they conflict. Design direction: First slices (read-only audit 2026-07-02, Herschel): config aliases (daemon_host/daemon_port, top-level observability), hidden/help + historical CLI aliases (find_help, -n, --full, demo-shelf --bundle), status JSON compatibility aliases, origin->provider projection bridges in outputs, browser-capture old synthetic-ID recovery. Rule per slice: the contract (DSL/registry/generated schema) becomes the owner, the parall…

## Source anchors to inspect first

- `polylogue/archive/query/spec.py:440` — SessionQuerySpec is the core read selection object.
- `polylogue/archive/query/spec.py:498` — from_params centralizes parameter lowering.
- `polylogue/cli/query_contracts.py:70` — QueryOutputSpec is current output/render contract seam.
- `polylogue/cli/archive_query.py:164` — CLI archive query execution remains a dense orchestration point.
- `polylogue/daemon/http.py:1734` — Daemon split-archive route already has from_params path for some cases.
- `polylogue/daemon/http.py:3296` — Another daemon path builds SessionQuerySpec from raw query params.
- `polylogue/mcp/query_contracts.py:54` — MCP query normalization has a distinct lowering path.

## Implementation plan

1. First slices (read-only audit 2026-07-02, Herschel): config aliases (daemon_host/daemon_port, top-level observability), hidden/help + historical CLI aliases (find_help, -n, --full, demo-shelf --bundle), status JSON compatibility aliases, origin->provider projection bridges in outputs, browser-capture old synthetic-ID recovery.
2. Rule per slice: the contract (DSL/registry/generated schema) becomes the owner, the parallel surface is deleted in the same PR — replacement-first, no compatibility fronts.
3. Regenerate render surfaces after each (openapi, cli-output-schemas, cli-reference).

## Tests to add

- Acceptance proof: For each listed first slice — config aliases (daemon_host/daemon_port, top-level observability)
- Acceptance proof: hidden/help + historical CLI aliases (find_help, -n, --full, demo-shelf --bundle)
- Acceptance proof: status JSON compatibility aliases
- Acceptance proof: origin->provider projection bridges in outputs
- Acceptance proof: browser-capture old synthetic-ID recovery — the parallel hand-written surface is DELETED in the same PR that makes the contract (DSL/registry/generated schema) the sole owner (grep confirms the alias/bridge is gone, no compatibility front left).
- Acceptance proof: After each slice, `devtools render openapi && devtools render cli-output-schemas && devtools render cli-reference` are regenerated and committed
- Acceptance proof: `devtools render all --check` is clean.
- Acceptance proof: `devtools verify` is green after each slice

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools test tests/unit tests/integration -k "query or contract or citation or insight"`

## Pitfalls

- Do not implement the epic as a single broad PR; use it to close/split child work and verify the terminal invariant.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.

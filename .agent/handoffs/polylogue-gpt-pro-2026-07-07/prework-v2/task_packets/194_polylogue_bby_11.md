# 194. polylogue-bby.11 — Webui architecture v2: the stack that can carry the ambition

Priority/type/status: **P1 / feature / open**. Lane: **99-horizon-or-general**. Release: **N-horizon**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

The roadmap now on the reader (mission control, timeline+firehose, replay, pinboard, day page, command palette, semantic renderers, SSE-live everything) cannot be built in JS-in-Python-strings, and shouldn't be built three views deep before the foundation is chosen. This bead decides and scaffolds the stack, sized for CODING AGENTS as the builders: maximum training-data familiarity, typed end-to-end, componentized, testable, self-contained (strict no-CDN/offline posture preserved).

## Existing design note

(1) STACK DECISION with rationale: TypeScript + Preact + Vite. Preact because React idioms are the deepest vein of agent training data at 4KB runtime cost (React itself rejected for size; Svelte/Solid rejected for thinner agent familiarity; no-build HTM rejected because losing TypeScript forfeits the mypy-equivalent net the whole codebase strategy relies on). Vite dev server proxies to the daemon (the dev-loop bead 5en integrates). (2) PACKAGING: built assets committed to polylogue/daemon/static/dist/ by a devtools render webui command (CI verifies build reproducibility; wheel/nix ship the committed dist — no node in the deploy chain, node only in the dev/CI chain). (3) STRUCTURE: webui/src/{lib,components,views}: lib/api.ts (typed client GENERATED from the OpenAPI render — payload types stay contract-true by construction), lib/live.ts (SSE subscription + cursor-keyed cache: the bby.8 semantics as ONE module every view gets for free), lib/tokens.css (lu1 design tokens); components/ implements the ap7 renderer specs (shared spec files, snapshot-tested against the Python renderer structure); views/ = list, reader, mission-control, timeline, judge-queue, settings. (4) CORE INTERACTIONS as foundation, not features: command palette (Ctrl-K: navigation, DSL/macro input, actions — the query-first philosophy as muscle memory), deep-link routing for every ref (scd contract), keyboard-first throughout, virtualized lists (bby.8). (5) MIGRATION: strangler pattern — v2 mounts at /app serving new views against the same API; old SPA remains until view parity, then dies in one PR (no long dual maintenance); bby.6's extraction is superseded for the JS (CSS tokens still shared) — re-note bby.6. (6) TESTS: vitest for components (extension suite precedent exists), the CDP smoke lane drives real flows (bby.7 parity walk runs against v2 too). (7) VIEW ROADMAP (each its own bead, this bead ships the foundation + ported list/reader): mission control (bby.9), timeline/firehose (bby.10), replay, day page, pinboard, judge queue (p5g web sibling), compare view, cost drill-anywhere (evidence-resolution rule applied to money — every cost figure expands to its usage events).

## Acceptance criteria

Scaffold merged: typed generated API client, SSE/cache module, tokens, palette, routing; list + reader views reach parity with the old SPA on the seeded corpus (including the bby.7 ref walk) and the old SPA's list/reader are retired; devtools render webui reproduces byte-identical committed dist in CI; a coding agent added one new view (the judge queue) purely against the scaffold docs — the agent-buildability proof.

## Static mechanism / likely defect

Issue description localizes the mechanism: The roadmap now on the reader (mission control, timeline+firehose, replay, pinboard, day page, command palette, semantic renderers, SSE-live everything) cannot be built in JS-in-Python-strings, and shouldn't be built three views deep before the foundation is chosen. This bead decides and scaffolds the stack, sized for CODING AGENTS as the builders: maximum training-data familiarity, typed end-to-end, componentized, testable, self-contained (strict no-CDN/offline posture preserved). Design direction: (1) STACK DECISION with rationale: TypeScript + Preact + Vite. Preact because React idioms are the deepest vein of agent training data at 4KB runtime cost (React itself rejected for size; Svelte/Solid rejected for thinner agent familiarity; no-build HTM rejected because losing TypeScript forfeits the mypy-equivalent net the whole codebase strategy relies on). Vite dev server proxies to the daemon (the dev-loop bead …

## Source anchors to inspect first

- `polylogue/archive/query/spec.py:440` — SessionQuerySpec is the core read selection object.
- `polylogue/archive/query/spec.py:498` — from_params centralizes parameter lowering.
- `polylogue/cli/query_contracts.py:70` — QueryOutputSpec is current output/render contract seam.
- `polylogue/cli/archive_query.py:164` — CLI archive query execution remains a dense orchestration point.
- `polylogue/daemon/http.py:1734` — Daemon split-archive route already has from_params path for some cases.
- `polylogue/daemon/http.py:3296` — Another daemon path builds SessionQuerySpec from raw query params.
- `polylogue/mcp/query_contracts.py:54` — MCP query normalization has a distinct lowering path.

## Implementation plan

1. (1) STACK DECISION with rationale: TypeScript + Preact + Vite.
2. Preact because React idioms are the deepest vein of agent training data at 4KB runtime cost (React itself rejected for size
3. Svelte/Solid rejected for thinner agent familiarity
4. no-build HTM rejected because losing TypeScript forfeits the mypy-equivalent net the whole codebase strategy relies on).
5. Vite dev server proxies to the daemon (the dev-loop bead 5en integrates).
6. (2) PACKAGING: built assets committed to polylogue/daemon/static/dist/ by a devtools render webui command (CI verifies build reproducibility
7. wheel/nix ship the committed dist — no node in the deploy chain, node only in the dev/CI chain).

## Tests to add

- Acceptance proof: Scaffold merged: typed generated API client, SSE/cache module, tokens, palette, routing
- Acceptance proof: list + reader views reach parity with the old SPA on the seeded corpus (including the bby.7 ref walk) and the old SPA's list/reader are retired
- Acceptance proof: devtools render webui reproduces byte-identical committed dist in CI
- Acceptance proof: a coding agent added one new view (the judge queue) purely against the scaffold docs — the agent-buildability proof.

## Verification commands

- `devtools test <focused tests added for this bead>`

## Pitfalls

- Do not broaden scope beyond the bead acceptance criteria; make a failing test first, then patch the smallest shared seam.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.

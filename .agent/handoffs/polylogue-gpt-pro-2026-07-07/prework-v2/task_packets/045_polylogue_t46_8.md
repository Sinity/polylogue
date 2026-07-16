# 045. polylogue-t46.8 — MCP surface collapse: ~96 tools -> verb algebra (query/get/explain/context/assert/maintenance...)

Priority/type/status: **P2 / task / open**. Lane: **04-read-contract-query-render**. Release: **C-read-contract**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

The MCP surface (96 tools live) is a discovery burden and a maintenance trap (every tool = contract + names + regen). Collapse to a small verb algebra: one query(expression) over the DSL absorbs ~40 read tools; get/explain/context/correlate/coordinate/assert/retract/maintenance cover the rest; sessions/messages/blocks/evidence-packs become MCP RESOURCES (URI-addressed, subscriptions/list_changed) and recall packs/saved views become MCP PROMPTS — the protocol-native primitive split instead of tools-for-everything. HARD SAFETY RULE: per-tool equivalence goldens BEFORE deletion (silent capability loss is the failure mode); resources/prompts flow through the same resolver + context compiler + recursive-safety inject gates as CLI (prompt expansion must not become an injection path). Wave spec: bundles/rnd-bundle-5-of-6.md L2286.

## Acceptance criteria

Verb set + resources + prompts cover every retired tool proven by goldens; EXPECTED_TOOL_NAMES shrinks with equivalence evidence per deletion; discovery tests + contracts regenerated; no capability regression reported by the golden suite. SHADOW TELEMETRY GATE (added 2026-07-06): before any tool deletion, a shadow-mode window records per-tool called-count by client/harness, mapped replacement verb/resource/prompt, golden parity status, and last-seen timestamp — deletion order follows observed compatibility, not design purity alone (MCP clients may have prompts/learned behavior keyed to old tool names). Verify: golden equivalence suite + the shadow-usage report artifact.

## Static mechanism / likely defect

Issue description localizes the mechanism: The MCP surface (96 tools live) is a discovery burden and a maintenance trap (every tool = contract + names + regen). Collapse to a small verb algebra: one query(expression) over the DSL absorbs ~40 read tools; get/explain/context/correlate/coordinate/assert/retract/maintenance cover the rest; sessions/messages/blocks/evidence-packs become MCP RESOURCES (URI-addressed, subscriptions/list_changed) and recall packs/saved views become MCP PROMPTS — the protocol-native primitive split instead of tools-for-everything. …

## Source anchors to inspect first

- `polylogue/archive/query/spec.py:440` — SessionQuerySpec is the core read selection object.
- `polylogue/archive/query/spec.py:498` — from_params centralizes parameter lowering.
- `polylogue/cli/query_contracts.py:70` — QueryOutputSpec is current output/render contract seam.
- `polylogue/cli/archive_query.py:164` — CLI archive query execution remains a dense orchestration point.
- `polylogue/daemon/http.py:1734` — Daemon split-archive route already has from_params path for some cases.
- `polylogue/daemon/http.py:3296` — Another daemon path builds SessionQuerySpec from raw query params.
- `polylogue/mcp/query_contracts.py:54` — MCP query normalization has a distinct lowering path.

## Implementation plan

1. Identify the currently duplicated surface paths for this behavior.
2. Create/extend the shared contract object and route one surface at a time through it.
3. Add parity tests across CLI, daemon/API, MCP, and Python facade.
4. Delete dead surface-side code after parity is green.

## Tests to add

- Acceptance proof: Verb set + resources + prompts cover every retired tool proven by goldens
- Acceptance proof: EXPECTED_TOOL_NAMES shrinks with equivalence evidence per deletion
- Acceptance proof: discovery tests + contracts regenerated
- Acceptance proof: no capability regression reported by the golden suite.
- Acceptance proof: SHADOW TELEMETRY GATE (added 2026-07-06): before any tool deletion, a shadow-mode window records per-tool called-count by client/harness, mapped replacement verb/resource/prompt, golden parity status, and last-seen timestamp — deletion order follows observed compatibility, not design purity alone (MCP clients may have prompts/learned behavior keyed to old tool names).
- Acceptance proof: Verify: golden equivalence suite + the shadow-usage report artifact.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools test tests/unit tests/integration -k "query or contract or citation or insight"`

## Pitfalls

- Do not broaden scope beyond the bead acceptance criteria; make a failing test first, then patch the smallest shared seam.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.

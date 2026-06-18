# Execution Plan

Living sequencing plan for the remaining Polylogue backlog. This document is a
coordination map, not a substitute for issue acceptance criteria. Issue bodies
and PR discussion own scope; this file names the order that minimizes rework.

## Current Backlog Shape

As of this plan, the active public backlog is small:

| Issue | Scope | Primary owner surface |
| --- | --- | --- |
| #1807 | Present Polylogue as a local AI-work evidence cockpit. | Epic / product integration |
| #2006 | Finish the full query DSL substrate. | `polylogue/archive/query/`, storage lowerers, CLI/API/MCP/web query routes |
| #1847 | Stabilize daemon/web API DTOs and local auth boundary. | `polylogue/daemon/`, `polylogue/surfaces/`, API/MCP parity |
| #1846 | Build the web workbench over archive contracts. | `polylogue/daemon/web_shell.py`, visual/daemon tests |
| #2126 | Collapse devtools command sprawl into owner groups. | `devtools/`, generated devtools docs, workflows/hooks |

Closed design or campaign notes should not live here as dispatch truth. If an
old plan is still useful, fold the relevant decision into the owning issue body
or current reference docs.

## Execution Order

1. **Finish devtools grouping (#2126).**
   - Already landed: release, workspace, lab, and benchmark groupings.
   - Remaining decisions: whether archive-facing diagnostics such as workload
     and space probes stay in `devtools` or move under `polylogue ops
     diagnostics`; whether any flat aliases remain because capability was not
     absorbed.
   - Evidence: generated `docs/devtools.md`, workflow references, hooks, and
     focused devtools tests all use the grouped commands.

2. **Finish query substrate (#2006).**
   - The grammar is already Lark-backed; do not describe a separate floor
     grammar or compatibility compiler.
   - Expand only through real lowerers, typed errors, explain metadata, and
     cross-surface tests.
   - Evidence: CLI, daemon, API, MCP, completion, and web query routes share the
     same parser/AST behavior; unsupported fields fail closed instead of
     broadening results.

3. **Stabilize route and DTO boundaries (#1847).**
   - Promote stable daemon routes only when they have typed payloads, auth
     posture, generated docs/schema where appropriate, and parity tests.
   - Keep shell-supported routes honest when they are intentionally local
     workbench internals.
   - Evidence: route contracts, OpenAPI/docs decisions, auth/redaction tests,
     and API/MCP parity where the route advertises shared behavior.

4. **Complete the web workbench verticals (#1846).**
   - Build on the existing daemon shell, not a separate app or marketing page.
   - Use shared query/read/recovery/assertion/ref DTOs; do not invent browser
     vocabularies.
   - Evidence: fixture/demo-backed flow for search -> open -> read/recovery ->
     assertions/evidence -> explicit raw drilldown.

5. **Close the epic (#1807) only after the product story is truthful.**
   - The README, CLI help, docs site, daemon shell, and release gate must
     describe what exists now.
   - Claims about query, web, recovery, assertions, and work packets must be
     backed by executable routes/tests or explicitly scoped out.

## Verification Policy

Use focused checks while editing and a broad gate at publication boundaries.

- Docs-only cleanup: `devtools render docs-surface --check`, `devtools render
  all --check`, and `devtools verify-doc-commands` when command examples move.
- Devtools command moves: focused `tests/unit/devtools` selections, `devtools
  render devtools-reference --check`, then `devtools verify --quick`.
- Query/route/workbench changes: focused behavior tests for the touched surface,
  generated schema/doc checks, then `devtools verify`.

Do not add tests that only prove an old name stayed deleted. Behavior, route
payloads, command examples, generated docs, and parser semantics are the
contracts.

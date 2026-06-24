# Execution Plan

Living sequencing plan for the remaining Polylogue backlog. This document is a
coordination map, not a substitute for issue acceptance criteria. Issue bodies
and PR discussion own scope; this file names the order that minimizes rework.

## Current Backlog Shape

As of this plan, the active public backlog is intentionally small but no longer
only substrate-shaped. Live deployed probing on 2026-06-21/22 added a product
and operations layer over the older query/route/workbench spine:

| Issue | Scope | Primary owner surface |
| --- | --- | --- |
| #1807 | Present Polylogue as a local AI-work evidence cockpit. | Epic / product integration |
| #2006 | Finish the full query DSL substrate. | `polylogue/archive/query/`, storage lowerers, CLI/API/MCP/web query routes |
| #1847 | Stabilize daemon/web API DTOs and local auth boundary. | `polylogue/daemon/`, `polylogue/surfaces/`, API/MCP parity |
| #1846 | Build the web workbench over archive contracts. | `polylogue/daemon/web_shell.py`, visual/daemon tests |
| #2304 | Keep the web workbench truthful and responsive when routes are slow, missing, or stale. | `polylogue/daemon/http.py`, `polylogue/daemon/web_shell.py`, facet/read models, browser smoke |
| #2305 | Define and verify executable `find QUERY then ACTION` workflows. | CLI verbs, JSON envelopes, completions, HTTP/MCP affordances, web action rendering |
| #2306 | Make recovery/context/evidence packs trustworthy handoff artifacts. | read views, evidence windows, assertion/candidate review, context/recovery renderers |
| #2307 | Polish docs, theming, packaging, release proof, and local control-plane probes. | README/docs, renderers, Nix package/runtime proof, dev-loop/browser diagnostics |
| #2308 | Make deployed state trustworthy and browser captures queryable. | deployment smoke, browser-capture receiver/archive flow, daemon status/resource behavior |
| #2309 | Make runtime and deployment configuration intentional. | `polylogue/config.py`, config docs, HM/NixOS module options, runtime/user/ops state boundaries |

Closed design or campaign notes should not live here as dispatch truth. If an
old plan is still useful, fold the relevant decision into the owning issue body
or current reference docs.

#2126, the devtools command-sprawl cleanup, is closed. Treat it as maintenance
history, not as a queued implementation lane. Reopen command-surface work only
when a concrete command has the wrong owner, generated docs are stale, or a
workflow/hook points at a removed command.

## Execution Order

1. **Keep deployed-state trust green while product work continues (#2308).**
   - Do not let the live service drift back to an unprovable state while
     feature work proceeds.
   - Deployment smoke is now the installed-artifact witness for versions,
     completions, core daemon routes, browser-capture spool/raw/index
     materialization, receiver archive-state, and optional browser first paint.
   - Evidence: `devtools workspace deployment-smoke --json` passes against the
     systemwide deployment; `polylogue` and `polylogued` versions match the
     intended build; the latest browser capture is queryable from archive rows,
     not only present in the receiver spool.

2. **Finish query substrate (#2006) together with query-action UX (#2305).**
   - The grammar is already Lark-backed; do not describe a separate floor
     grammar or compatibility compiler.
   - Expand only through real lowerers, typed errors, explain metadata, and
     cross-surface tests.
   - Treat `find QUERY then ACTION` as the product contract over the query
     substrate: selection envelope, cardinality, exactness, action
     affordances, rendering, completions, and JSON/HTTP/MCP/web parity.
   - Evidence: CLI, daemon, API, MCP, completion, and web query routes share the
     same parser/AST behavior; unsupported fields fail closed instead of
     broadening results; exact refs do not degrade into broad FTS; query-result
     envelopes expose the same action affordance DTO across surfaces.

3. **Stabilize route and DTO boundaries (#1847) through the new product
   surfaces (#2304/#2305/#2306).**
   - Promote stable daemon routes only when they have typed payloads, auth
     posture, generated docs/schema where appropriate, and parity tests.
   - Keep shell-supported routes honest when they are intentionally local
     workbench internals.
   - Slow or partial routes need typed degraded responses, not silent UI
     failure. Facets/status/read-view profiles and context/recovery routes must
     advertise completeness, staleness, omissions, and fallback actions when
     applicable.
   - Evidence: route contracts, OpenAPI/docs decisions, auth/redaction tests,
     API/MCP parity where the route advertises shared behavior, and smoke/DOM
     tests for degraded web states.

4. **Complete the web workbench verticals (#1846/#2304).**
   - Build on the existing daemon shell, not a separate app or marketing page.
   - Use shared query/read/recovery/assertion/ref DTOs; do not invent browser
     vocabularies.
   - First paint must not depend on expensive optional route families. Results
     should remain visible when facets, status, read-view profiles, or session
     detail routes degrade.
   - Evidence: fixture/demo-backed flow for search -> open -> read/recovery ->
     assertions/evidence -> explicit raw drilldown; live browser smoke produces
     DOM and screenshot artifacts; degraded-route tests prevent populated data
     coexisting with unexplained zero counters or permanent `Loading...`.

5. **Make context/evidence packs usable for real handoff (#2306).**
   - Recovery/context packs must state selection strategy, evidence refs,
     omissions, redaction policy, caveats, and token/size estimates.
   - Candidate extraction must avoid treating large instruction dumps as
     durable decisions unless evidence supports the claim.
   - Evidence: demo/live fixtures with instruction dumps and real decisions;
     JSON and Markdown handoff outputs distinguish quoted evidence, inferred
     summary, accepted/rejected/deferred candidates, unavailable source
     material, and disabled review actions with reasons. Default handoffs redact
     raw absolute paths unless an explicit raw/no-redact path is requested.

6. **Make configuration explicit before adding more knobs (#2309).**
   - Classify settings as startup config, deployment policy, mutable user
     preference, provider/cost control, presentation preference, or disposable
     ops state before adding them.
   - Config inspection must show effective values and source layers while
     redacting secrets. Nix/HM options should map to the same semantics as
     non-Nix installs, with deployment-only resource policy called out.
   - Evidence: config inventory tests, docs that explain static vs runtime vs
     ops state, redaction/security tests, deployment smoke effective-config
     evidence, and Sinnix module checks where Nix options are touched.

7. **Keep repo/package/dev-loop polish tied to shipped workflows (#2307/#2248).**
   - Docs, theming, release proof, and private-browser/dev-loop probes should
     make real workflows easier to run and verify; avoid decorative evidence layers and
     absence memorials.
   - Evidence: README/install commands match shipped behavior; package checks
     prove runtime dependency closure; rendering respects plain/no-color modes;
     branch-local daemon/web/receiver runs expose ports, archive roots, logs,
     browser executable paths, and capture artifacts.

8. **Close the epic (#1807) only after the product story is truthful.**
   - The README, CLI help, docs site, daemon shell, and release gate must
     describe what exists now.
   - Claims about query, web, recovery, assertions, and work packets must be
     backed by executable routes/tests or explicitly scoped out.

## Rawlog Coverage Map

The 2026-06-18 rawlog points should have one durable owner each. If a future
agent cannot map a raw idea to one of these owners, update the relevant issue
body before implementing from chat context.

| Rawlog theme | Durable owner | Implementation posture |
| --- | --- | --- |
| `find QUERY then ACTION` UX, CLI vs interactive vs web/TUI, JSON automation, exact refs, affordances | #2305 | Product contract first, then CLI/HTTP/MCP/web parity and golden paths. |
| Markdown/format/rendering aesthetics, pleasant terminal output, no-color/plain modes, pywal/theme responsiveness | #2307 | Renderer/theme tokens and docs/release proof; do not hard-code a one-off palette. |
| Web UI rethink, route failures, slow facets, first-paint truthfulness | #2304 | Bounded route contracts, degraded UI states, lazy/stale facets, DOM/browser smoke. |
| Browser capture fully functional, provider-native payloads, realtime/live-session state, archive-state lifecycle | #2308 for deployed correctness; #2248 for branch-local extension/dev loop | Keep current capture queryable first; widen to fidelity/lifecycle/sync once the deployed invariant is stable. |
| Recovery/context packs for feeding stronger agents and avoiding missing context | #2306 | Evidence-backed bundles with omissions, caveats, redaction, and candidate-review affordances. |
| Configurability without pointless knobs, Nix/NixOS/runtime config clarity | #2309 | Classify each setting before adding it; distinguish startup config, deployment policy, mutable user state, presentation preferences, provider spend, and disposable ops state. |
| Codebase incoherence, stale names, schema/concept review | #2177 plus #2307 | Cleanup belongs with contract ownership or repo-polish work, not standalone rename ceremony. |
| Agent/process/desktop/browser control and ambient capabilities | #2248, with Polylogue only owning branch-local URLs/logs/receiver config | Do not make Polylogue a general desktop automation framework; expose inspectable control points for existing local tools. |
| OTLP/observability relevance | #2183 for projection, #2248/#2308 for local daemon/dev-loop resource evidence | OTel-style export is a projection, not internal authority. |

## Verification Policy

Use focused checks while editing and a broad gate at publication boundaries.

- Docs-only cleanup: `devtools render docs-surface --check`, `devtools render
  all --check`, and `devtools verify doc-commands` when command examples move.
- Devtools command moves: focused `tests/unit/devtools` selections, `devtools
  render devtools-reference --check`, then `devtools verify --quick`.
- Query/route/workbench changes: focused behavior tests for the touched surface,
  generated schema/doc checks, then `devtools verify`.

Do not add tests that only prove an old name stayed deleted. Behavior, route
payloads, command examples, generated docs, and parser semantics are the
contracts.

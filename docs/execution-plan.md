# Execution Plan

Living sequencing plan for the remaining Polylogue backlog. This document is a
coordination map, not a substitute for issue acceptance criteria. Issue bodies
and PR discussion own scope; this file names the order that minimizes rework.

## Current Backlog Shape

As of the 2026-06-24 issue export, the active public backlog is a residual map
around #1807 rather than a single linear implementation queue. Closed campaign
issues are baseline contracts; open issues below own the remaining execution
lanes.

| Issue | Residual scope | Primary owner surface |
| --- | --- | --- |
| #1807 | Integrate the remaining work into a truthful local AI-work evidence cockpit story. | Epic / product integration |
| #2006 | Finish the full query DSL substrate on the existing parser, typed AST, query-unit metadata, lowerers, pipeline stages, docs, and cross-surface parity. | `polylogue/archive/query/`, storage lowerers, CLI/API/MCP/web query routes |
| #1844 | Build completion, fuzzy selection, and query-builder metadata from shared query grammar/action/read-view registries. | `polylogue/archive/query/metadata.py`, `polylogue/cli/shell_completion_values.py`, completion tests |
| #1883 | Finish assertion lifecycle and policy over the landed assertion-backed user-overlay substrate. | `polylogue/storage/sqlite/archive_tiers/user.py`, assertion read/write/review surfaces |
| #2177 | Make shared contracts own surfaces and delete remaining parallel dispatch/facade/string-vocabulary copies. | CLI/MCP/API/daemon/read-view/action contract metadata and generated-surface inputs |
| #2196 | Replace showcase-era QA with demo-driven CLI, DOM, visual, and ordinary behavior tests. | `demo/`, `tests/infra/`, visual tests, validation-lane/devtools cleanup |
| #2246 | Extend archive schema/read-path hygiene only where landed evidence-cockpit surfaces prove a storage need. | `polylogue/storage/sqlite/archive_tiers/`, schema docs, self-verification |
| #2248 | Make daemon, web-shell, browser-capture, and extension debugging branch-local and isolated from deployed state. | `devtools/dev_loop.py`, `docs/dev-loop.md`, `browser-extension/`, daemon/browser-capture docs |
| #2304 | Keep the web workbench truthful and responsive when routes are slow, missing, stale, or expensive. | `polylogue/daemon/http.py`, `polylogue/daemon/web_shell.py`, facet/read models, DOM/browser smoke |
| #2307 | Polish docs, theming, package proof, and local control-plane probes that support shipped workflows. | README/docs map, renderers, Nix/package checks, dev-loop/browser diagnostics |
| #2308 | Make deployed state trustworthy and browser captures queryable from archive rows, not only receiver spool files. | deployment smoke, browser-capture receiver/archive flow, daemon status/resource behavior |
| #2309 | Make runtime and deployment configuration intentional and inspectable. | `polylogue/config.py`, config docs, HM/NixOS options, runtime/user/ops state boundaries |
| #2316 | Make provider usage accounting complete, auditable, and explicit about coverage and cache semantics. | parser usage events, usage ledger/rollups, diagnostics, CLI/API/MCP usage surfaces |
| #2317 | Rationalize root onboarding, reader/dashboard promises, status discoverability, mark ownership, and facet usefulness. | root CLI/help/tutorial/dashboard/facet surfaces and demo-backed tests |

Closed issues that this plan can reference as baseline but should not present as
active execution lanes: #1846 web workbench contract, #1847 daemon/API DTO and
auth boundary, #2182 candidate assertion promotion, #2183 OTLP projection,
#2253 ops/product command placement, #2305 query-action workflows, and #2306
recovery/context/evidence handoff packs. Reopen those only when new source
evidence shows their contracts regressed; otherwise route follow-up through the
open residual owner above.

#2126, the devtools command-sprawl cleanup, is closed. Treat it as maintenance
history, not as a queued implementation lane. Reopen command-surface work only
when a concrete command has the wrong owner, generated docs are stale, or a
workflow/hook points at a removed command.

## Residual Map

| Residual class | Current owner | Where it belongs | Locality / agent posture |
| --- | --- | --- | --- |
| Live/deployment local-only | #2308 | Deployment smoke, installed `polylogue`/`polylogued` version checks, browser-capture receiver/archive consistency, daemon status/resource probes. | Requires the operator's live deployment, archive, browser-capture spool, and Sinnix/NixOS wrapper state for closeout. Cloud agents may patch source probes and docs, but cannot certify deployed truth. |
| Live/deployment local-only | #2248 | Branch-local daemon/web/receiver/extension loop, dev archive roots, isolated ports, run logs, copied-profile safety boundary. | Local-only for copied profiles, real browser control, screenshots, extension service workers, and production-service coexistence. Cloud-safe work is limited to deterministic synthetic smoke, docs, and helper-source changes. |
| Source-only implementation | #2006, #1844, #2177 | Shared query grammar/metadata/lowerers, completion metadata, normalized request/scope DTOs, read-view/action metadata, generated-surface inputs. | Cloud-safe when backed by unit/integration fixtures and generated docs/schema checks. Do not fork parser, completion, or route vocabularies to work around local data. |
| Source-only implementation | #2316 | Provider usage extraction, `session_provider_usage_events`, rollups, coverage diagnostics, and usage-facing command/API/MCP payloads. | Initial Codex token-count preservation and usage-event storage exist in source; residual work is coverage semantics, rollups, provider matrix, docs, and rebuild guidance. Local archive probes are evidence, not a substitute for fixtures. |
| Product/UX design | #2304, #2317, #1883 | Web degraded states, first-paint budget, root help/onboarding/dashboard/status behavior, facet canonicalization, assertion lifecycle/review policy. | Design decisions must land as observable contracts, help text, payload states, DOM/CLI tests, and docs. Avoid unverified tutorial/dashboard claims or decorative registries. |
| Schema/runtime migration | #2246, #1883, #2316 | Fresh-first tier DDL, assertion indexes and lifecycle, usage ledger/rollups, self-verify, schema docs, explicit rebuild/reset paths. | Source changes are cloud-safe; closeout needs fresh-archive/rebuild evidence. `ops.db` may keep documented disposable compatibility helpers; durable tiers should not grow chained migrations. |
| Docs/navigation | #2196, #2307, #2248 | Docs index, execution plan, dev-loop/browser-capture docs, generated docs surface, demo/visual verification docs. | Safe docs cleanup should point at shipped workflows and current architecture pages. Remove showcase-era or closed-campaign language rather than adding absence memorials. |
| Stale or misframed | #1807 and any docs/issue references to closed #1846/#1847/#2182/#2183/#2253/#2305/#2306 as active lanes | Issue bodies, execution docs, generated docs descriptions, release/readiness notes. | Treat those closed issues as baseline contracts. Patch repo docs immediately; use issue comment/body drafts only when the stale surface is GitHub issue text rather than repository docs. |

## Execution Order

1. **Keep deployed-state trust green while product work continues (#2308).**
   - Do not let the live service drift back to an unprovable state while
     feature work proceeds.
   - Deployment smoke is the installed-artifact witness for versions,
     completions, core daemon routes, browser-capture spool/raw/index
     materialization, receiver archive-state, and optional browser first paint.
   - Source patches can improve smoke payloads and receiver/archive diagnostics,
     but closeout requires the operator's live deployment and archive.
   - Evidence: `devtools workspace deployment-smoke --json` passes against the
     systemwide deployment; `polylogue` and `polylogued` versions match the
     intended build; the latest browser capture is queryable from archive rows,
     not only present in the receiver spool.

2. **Finish query substrate and discovery on shared metadata (#2006/#1844/#2177).**
   - The grammar is already Lark-backed; do not describe a separate floor
     grammar or compatibility compiler.
   - Expand only through real lowerers, typed errors, explain metadata,
     shared query-unit descriptors, and cross-surface tests.
   - Completion, fuzzy selection, and query-builder help should introspect the
     same query/action/read-view registries rather than carrying parallel field
     lists.
   - Evidence: CLI, daemon, API, MCP, completion, and web query routes share the
     same parser/AST behavior; unsupported fields fail closed instead of
     broadening results; exact refs do not degrade into broad FTS; generated
     docs come from the same contract inputs as runtime surfaces.

3. **Make product-facing entrypoints and facets coherent (#2317, with #2305 as baseline).**
   - Treat the query-action workflow contract as baseline, not as an active
     issue lane. Follow-up belongs in root help, onboarding, dashboard/reader,
     status discoverability, mark ownership, and facet usefulness.
   - `dashboard`, `tutorial`, `status`, root query-action verbs, and `mark`
     should either have explicit product ownership or be moved/renamed/deleted
     with behavior tests.
   - Facets should be canonical, bounded, and useful; path fragments, agent ids,
     archive folders, and one-off scratch names must not masquerade as top repo
     facets.
   - Evidence: root help groups commands by role; `polylogue status` has an
     intentional behavior or targeted `polylogue ops status` diagnostic;
     tutorial/dashboard output proves what happened; facet JSON reports family,
     canonicalization, omission/noise, freshness, and budget/degraded state.

4. **Keep the web workbench truthful under degradation (#2304, with #1846/#1847 as baseline).**
   - The daemon web shell and local HTTP/auth DTO baseline already exists; do not
     reopen closed web/API contract issues unless source evidence shows a
     regression.
   - First paint must not depend on expensive optional route families. Results
     should remain visible when facets, status, read-view profiles, or session
     detail routes degrade.
   - Slow or partial routes need typed degraded responses, visible retry/fallback
     actions, and DOM/browser smoke coverage rather than silent UI failure.
   - Evidence: populated lists cannot coexist with unexplained zero counters or
     permanent `Loading...`; degraded-route tests cover timed-out facets,
     missing read-view profiles, failed session detail, stale data, retry, and
     fallback command rendering.

5. **Finish assertion, schema, and usage runtime semantics (#1883/#2246/#2316).**
   - User-overlay storage is already assertion-backed in the source tree; the
     residual is lifecycle/read policy, index/read hardening, export/reset
     semantics, and review surfaces, not a second overlay store.
   - Schema work should follow fresh-first tier DDL and add durable tables only
     when landed read/query/ref/usage surfaces prove they need identity,
     freshness, or performance.
   - Provider usage has an initial event substrate in source; finish provider
     coverage semantics, rollups, coverage states, current-vs-cumulative token
     handling, cache/pricing labels, diagnostics, and rebuild guidance.
   - Evidence: schema docs match tier DDL; self-verify covers cheap high-value
     invariants; usage surfaces distinguish exact provider telemetry, text
     estimates, unsupported/missing data, acquired-not-materialized rows, and
     stale rollups.

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

7. **Keep demo/docs/dev-loop polish tied to shipped workflows (#2196/#2307/#2248).**
   - Replace showcase-era verification with demo-backed CLI/DOM/visual tests and
     generated docs that describe current behavior.
   - Docs, theming, package proof, and private-browser/dev-loop probes should
     make real workflows easier to run and verify; avoid decorative evidence layers and
     absence memorials.
   - Branch-local daemon/web/receiver runs must expose ports, archive roots,
     logs, browser executable paths, and capture artifacts without depending on
     systemwide deployment or committed browser profile state.
   - Evidence: docs index links the current architecture/execution/dev-loop
     pages; rendering respects plain/no-color modes; branch-local synthetic
     receiver/extension smokes pass; live browser/copied-profile proof remains a
     local operator-only lane.

8. **Treat #1807 as ready for final review only after the product story is truthful.**
   - The README, CLI help, docs site, daemon shell, and release gate must
     describe what exists now.
   - Claims about query, web, recovery, assertions, usage accounting, work
     packets, browser capture, and deployment trust must be backed by executable
     routes/tests or explicitly scoped out.

## Rawlog Coverage Map

The 2026-06-18 rawlog points should have one durable owner each. If a future
agent cannot map a raw idea to one of these owners, update the relevant issue
body before implementing from chat context.

| Rawlog theme | Durable owner | Implementation posture |
| --- | --- | --- |
| `find QUERY then ACTION` UX, CLI vs interactive vs web/TUI, JSON automation, exact refs, affordances | #2006 and #2317; #2305 is the baseline contract | Keep query/action semantics on shared substrate, then repair root/help/onboarding/facet UX where the product surface still misleads. |
| Markdown/format/rendering aesthetics, pleasant terminal output, no-color/plain modes, pywal/theme responsiveness | #2307 plus #2196 | Renderer/theme tokens and demo-backed visual tests; do not hard-code a one-off palette or keep showcase-era proof vocabulary. |
| Web UI rethink, route failures, slow facets, first-paint truthfulness | #2304; #1846/#1847 are baseline contracts | Bounded route contracts, degraded UI states, lazy/stale facets, DOM/browser smoke. |
| Browser capture fully functional, provider-native payloads, realtime/live-session state, archive-state lifecycle | #2308 for deployed correctness; #2248 for branch-local extension/dev loop | Keep current capture queryable first; widen to fidelity/lifecycle/sync once the deployed invariant is stable. |
| Recovery/context packs for feeding stronger agents and avoiding missing context | #1883 and #2246 for residual assertion/schema policy; #2306 is the handoff baseline | Treat packs as evidence-backed outputs with omissions, caveats, redaction, and candidate-review affordances; add storage only when read/query/ref surfaces need it. |
| Configurability without pointless knobs, Nix/NixOS/runtime config clarity | #2309 | Classify each setting before adding it; distinguish startup config, deployment policy, mutable user state, presentation preferences, provider spend, and disposable ops state. |
| Provider usage, pricing, cache semantics, and provider UI reconciliation | #2316 | Preserve provider-reported usage as evidence, roll it up with coverage states, and keep text-volume estimates separate from billable/provider counters. |
| Codebase incoherence, stale names, schema/concept review | #2177 plus #2246/#2307 | Cleanup belongs with contract ownership, schema doctrine, or repo-polish work, not standalone rename ceremony. |
| Agent/process/desktop/browser control and ambient capabilities | #2248, with Polylogue only owning branch-local URLs/logs/receiver config | Do not make Polylogue a general desktop automation framework; expose inspectable control points for existing local tools. |
| OTLP/observability relevance | #2183 is the projection baseline; #2248/#2308 own local daemon/dev-loop resource evidence | OTel-style export is a projection, not internal authority. |

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

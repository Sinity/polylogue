# WebUI v2 session list/read — revision 02 evidence and adjudication

## Evidence order

The implementation used the following authority order: current snapshot source; repository instructions; complete relevant Beads records and later notes; current tests and generated contracts; reachable history; then the mission prompt. Older descriptions were not allowed to override landed source.

## Snapshot and repository identity

Observed from the supplied archive manifest and Git bundle:

- Branch `master` at `536a53efac0cbe4a2473ad379e4db49ef3fce74d`.
- Commit subject `fix(repair): harden raw authority convergence (#3046)`.
- Commit timestamp `2026-07-17T18:55:47+02:00`.
- Snapshot generated `2026-07-17T180950Z` from `/realm/project/polylogue`.
- Supplied archive SHA-256 `ec3fa193b2f99a6daee0bc620af4f69b5728834e99f8196792a17c3fbae11155`.
- Working-tree archive SHA-256 `9b0664c982b58a980e52f47af7a7466f6f5f3b3b3cf4914c16dba232639bc8bf`.
- All-refs bundle SHA-256 `b8595e2392d5a0587e083fe7c9510dce20b71dbdfeccfab39dcba14fb484e13a`.
- Manifest says `dirty: true`, but the exported branch-delta patch, file list, and log are all empty.

Adjudication: the extracted working tree is the current source authority. The archive does not preserve a separate dirty delta, so this package does not invent one.

Repository instructions inspected:

- `CLAUDE.md` and linked `AGENTS.md` for architecture, source ownership, generated surfaces, quality policy, and required topology accounting.
- `TESTING.md` and `CONTRIBUTING.md` for focused verification and test anti-vacuity expectations.
- `pyproject.toml`, package-data rules, `.gitignore`, `webui/package.json`, `webui/playwright.config.ts`, and existing browser harness for build/distribution behavior.
- `devtools render` ownership for OpenAPI, topology status, and site-source consistency.

## Current source findings

### Shared query transaction is the paging protocol authority

`polylogue/archive/query/transaction.py` defines the vocabulary this vertical consumes:

- `QueryTransactionRequest`: `operation`, `arguments`, `page_size`, `offset`, `projection`, `stable_order`.
- `query_ref`: hashes operation, logical arguments, projection, and stable order. Page size and offset are intentionally excluded.
- `QueryContinuation`: an opaque versioned `q1.` token carrying the advancing request and result ref.
- `QueryResultPage`: derives `next_offset` from current offset plus returned rows and emits a continuation only when `has_more` is true.
- `QueryTransaction.result_ref`: hashes query ref, projection, and stable order into a stable logical result-family ref.
- `archive_read_context`/execution-control integration: bounded, interruptible archive execution uses the same request vocabulary.

Decision: use these exact symbols and names. Do not introduce a WebUI cursor schema, browser-generated offset, or parallel result identity.

Important limitation observed in source: `result_ref` is deterministic logical-family identity. It does not pin a database snapshot or include archive generation. The implementation can reject family drift but cannot promise mutation isolation between pages.

### Canonical read semantics already have owners

- `polylogue/core/enums.py`: canonical `Role`, `MessageType`, and `MaterialOrigin` tokens.
- `polylogue/archive/actions/parsing.py:tool_result_outcome`: provider-error and exit-code outcome calculation, including exit-code authority.
- Existing archive/session/message payload builders in `polylogue/daemon/http.py`: session summaries, composed messages, structured blocks, attachment fields, degraded route states, and stable reader anchors.
- `polylogue/daemon/topology_http.py:build_topology_envelope`: bounded fork/resume family evidence.

Decision: project those facts directly. The browser declares exact token unions for type safety but owns no synonym or reinterpretation table.

### HTTP, access, and distribution boundaries

Inspected in `polylogue/daemon/http.py`, `route_contracts.py`, `web_auth.py`, and current tests:

- Shell HTML has an existing unauthenticated-loopback bootstrap policy and token requirement off loopback.
- JSON reads use configured bearer auth or first-party cookie credentials.
- `/api/web-auth/session` already establishes a scoped, exact-origin, HttpOnly cookie; the vertical should consume it rather than invent auth.
- Existing static and shell paths use explicit dispatch and security headers.
- Route contracts generate the current OpenAPI route catalog.
- The wheel already packages Python package data; committed daemon assets need no CDN or Node server.

Decision: `/app` uses the existing shell bootstrap policy. Islands bootstrap the existing first-party credential before JSON. Assets are fixed names under the daemon package and requests remain same-origin.

Security implication: local unauthenticated HTML can contain archive content under current loopback architecture. This is a conscious cutover review point, not hidden as a new security model.

### Existing WebUI state is not a v2 scaffold

At the named commit, the tracked `webui/` root contains the package/lock, Playwright configuration, and `tests/first-party-auth.spec.ts`. No Preact source tree, Vite configuration, `webui-01` interface, or scaffold component exists.

Reachable WebUI history inspected includes first-party daemon credentials (`0e0cddaee`) and bounded cockpit aggregates (`c1f7704fa`), but no `webui-01` commit or path.

Decision: preserve the existing package and browser-auth lane; add only the minimum ratified Preact/Vite lane needed for this vertical.

### Legacy list/read ownership is interleaved

`polylogue/daemon/web_shell.py` owns the current session list/read navigation and offset controls. `polylogue/daemon/web_shell_reader.py` owns transcript rendering but is composed by other old-shell features.

Decision: land a strangler route and enumerate symbol-level deletion candidates. Deleting the old reader module wholesale would remove shared behavior before call-site migration.

## Second-pass findings and repairs

### Prior client trusted compile-time types at an untrusted boundary

Observed in the first implementation: the continuation utility parsed successful JSON with a TypeScript cast and accepted a loose token. Components appended returned rows if the HTTP call succeeded. The server validates tokens, but a server/proxy/regression response with changed refs or malformed pagination metadata could still be merged into the current UI.

Repair:

- Add runtime envelope validation for current SSR state and fetched pages.
- Validate field types and internal pagination relationships.
- Pass the complete current page into `fetchContinuationPage`.
- Require stable `query_ref`, stable `result_ref`, exact prior `next_offset`, stable limit, and changed continuation before merge.
- Reject cross-origin endpoints before credential bootstrap.
- Add seven focused tests that mutate one invariant at a time.

Falsification evidence: removing any one check causes its corresponding test to accept unrelated or malformed rows.

### Route-level tests did not prove production composition

Observed in the first implementation: focused route tests isolated archive-return values, which proved rendering and continuation adapters but not seeded archive → production daemon → SSR/API/assets composition.

Repair: add `TestWebUIV2Vertical` to `tests/unit/daemon/test_web_reader.py`. It runs the real HTTP server and split archive, verifies semantic HTML before bootstrap, token-only page two, SSR/API role/material parity, and committed assets.

Falsification evidence: empty client mounts, archive bypasses, caller-reconstructed paging, role/material drift, or missing assets fail the test.

### Stored-content and asset-path security deserved executable assertions

Observed in current source: renderer helpers use context-specific escaping and the asset path uses an allowlist, but the first package relied mainly on source inspection.

Repair:

- Add a hostile synthetic title route test that proves visible HTML escaping and script-safe JSON while decoding the exact intended value.
- Add encoded traversal request `/app/assets/..%2Fhttp.py` and require 404.

Falsification evidence: raw interpolation or path joining makes these tests fail.

## Tests and runtime evidence inspected

Existing test areas inspected before implementation:

- `tests/unit/daemon/test_daemon_http_contracts.py`
- `tests/unit/daemon/test_daemon_http_security.py`
- `tests/unit/daemon/test_route_contracts.py`
- `tests/unit/daemon/test_web_reader.py`
- `tests/unit/test_read_surface_coherence.py`
- `tests/browser/web_auth_server.py`
- `webui/tests/first-party-auth.spec.ts`

Observed final verification evidence:

- 661 daemon route/security/contract/vertical tests pass.
- Nine directly affected reader tests pass.
- All 50 read-surface coherence tests pass.
- Ten frontend tests pass, seven of them aimed specifically at continuation integrity.
- Strict TypeScript, Ruff, Ruff formatting, strict mypy, Python byte compilation, generated-surface checks, Vite build, offline npm audit, wheel build, and patch application all pass.
- Real production HTTP over a deterministic 15-session/62-message demo archive returns semantic list/read pages, first-party credentials, stable token-only continuation, canonical message facts, and exact committed assets.
- The production bundle hydrates both SSR documents in jsdom without DOM changes or console errors; a live-daemon filter updates the hydrated list correctly.

The full reader file has one failure. The identical test fails on a pristine named-snapshot checkout with the same missing-`index.db` provider-usage error. This is observed baseline evidence, not an inference.

Native Playwright navigation remains unavailable because managed Chromium blocks all URLs with `URLBlocklist: ["*"]`. The server starts and discovery succeeds; the first `page.goto()` is blocked before product content. Browser layout/accessibility evidence is therefore unresolved.

## Relevant Beads records

### `polylogue-bby.11` — WebUI architecture v2

Observed intent:

- TypeScript + Preact + Vite is ratified.
- Assets are committed, offline, and no-CDN.
- Daemon routes must serve semantic HTML plus typed JSON.
- Preact is progressive enhancement, not an SPA-only replacement.
- List/reader parity precedes broader graph/report work.

Decision: implement the requested list/read vertical only. Do not import broader basket/report/evidence-graph product models into this slice.

### `polylogue-1ilk` — WebUI v2 test stack

Observed intent: Vitest component tests are the per-change lane; Playwright and visual evidence are separate browser lanes.

Decision: make component and route tests mandatory now, attempt Playwright, and report the managed-browser block instead of representing jsdom as browser proof.

### `polylogue-3utv` — declare-once RouteSpec/OpenAPI/TypeScript client

Observed state: the desired generated route registry/client remains open and is not current daemon architecture.

Decision: extend the current route contract table and regenerate current OpenAPI. Keep one localized exact TypeScript payload contract; do not invent a competing route registry or claim generated-client ownership.

### `polylogue-bby.7` — stable list/detail refs

Observed intent/history: list-emitted refs must resolve on detail routes; encoded IDs remain stable.

Decision: emit `session_id` on archive and live list paths, encode it in `/app/s/...`, preserve message anchors, and test live fallback compatibility.

### `polylogue-bby.8` — paging/performance

Observed correction: current lists are server-paged and clients must consume server semantics rather than reconstruct paging locally.

Decision: initial filters are ordinary query input; every advance is opaque-token-only. Revision 02 additionally refuses to merge a page that is not demonstrably the next member of the same family.

### `polylogue-9xuk` — evidence-honest visual semantics

Observed intent: unknown, unavailable, partial, and degraded evidence must not appear as blank success or zero.

Decision: expose `total_relation`, preserve route-state evidence, and make unknown-empty states explicit on SSR and hydrated paths.

### `polylogue-2n39` — stored-content XSS hardening

Observed state: old-shell hardening is closed, but every new renderer must retain context-aware escaping.

Decision: separate HTML text/attribute escaping from script-state escaping and add an executable hostile-content regression.

### `polylogue-z9gh.3` and shared query work

Observed direction: query discovery and every surface should share executable transaction/declaration semantics, explicit result identity, and continuation behavior.

Decision: consume the landed transaction substrate and avoid making this WebUI vertical the owner of a new query catalog.

## Relevant history inspected

- Query transaction and bounded execution work from 2026-07-17, including `9163d0134` and `fd7b35492`.
- Semantic transcript renderer history, including `fc770dbd9` and `0c251b600`.
- Daemon list/auth/cockpit history, including `d0bc0a927`, `0e0cddaee`, and `c1f7704fa`.
- Role synonym consolidation `5623c2ab9`.
- Full reachable history search for `webui-01`: no result.

History supports using current shared owners rather than restoring stale plans or creating a new client-only model.

## Contradictions and resolutions

### Mission allows a possible `webui-01`; source has none

Resolution: define the minimum scaffold assumption explicitly and build on existing package/auth infrastructure only.

### Architecture wants a generated client; route-registry work is still open

Resolution: use the current executable `ROUTE_CONTRACTS` owner and localized exact payload declarations. Do not fake a generator that does not exist.

### Architecture eventually retires old list/read; mission forbids deleting integration candidates

Resolution: land `/app` as a strangler path, preserve all legacy tests/files, and identify exact later deletion candidates.

### Snapshot reports dirty; branch delta is empty

Resolution: disclose both. The current extracted tree is authoritative; no unnamed patch identity is claimed.

### Generated topology output changes rows outside this vertical

Resolution: keep exact generator output. A new Python module changes source accounting; leaving generated files stale would fail `render all --check`.

### Existing clients still use offset/legacy fields

Resolution: preserve all established fields and first-request behavior. Add continuation metadata additively. Only the v2 islands are token-only on advance.

### Runtime page identity is stable but not snapshot isolation

Resolution: enforce all identity the substrate actually provides and state the remaining mutation limitation. Do not describe `result_ref` as a snapshot.

### Browser test failure could be mistaken for product failure

Resolution: inspect the browser policy and distinguish an administrative URL block from route behavior. HTTP/jsdom evidence is retained; native browser acceptance remains unverified.

### Full reader failure could be mistaken for a regression

Resolution: reproduce the exact failing case on a pristine checkout. It fails identically before the patch, so it is reported as baseline debt rather than waived silently.

## Source-supported inferences

- Because generated OpenAPI reads `ROUTE_CONTRACTS`, new `/app` routes require regenerating `docs/openapi/search.yaml`; the final generated check confirms this.
- Because package data includes the daemon module tree and the wheel contains both committed assets, this vertical needs no runtime CDN or Node service.
- Because `web_shell_reader.py` has multiple legacy callers, deleting the file at initial list/read cutover is unsafe; symbol/call-site retirement must happen first.
- Because split-archive read composition currently owns lineage/inherited-prefix/attachment/structured-block semantics as a whole, simply paging raw SQL rows would risk semantic drift. The current adapter composes then slices; a more scalable solution belongs in a shared paged semantic transcript substrate.
- Because first-party credential bootstrap is exact-origin and loopback-oriented, `/app` is primarily a local daemon surface unless the broader auth architecture changes.

## Unresolved uncertainty and falsification evidence

Unresolved:

- Native browser rendering, accessibility tree, focus order, and responsive layout.
- Archive mutation behavior during a multi-page walk, because no snapshot pin exists.
- Performance of full transcript composition for exceptionally large sessions.
- The unnamed dirty state referenced by the snapshot manifest but omitted from its zero-byte branch delta.
- How a future generated RouteSpec/TypeScript client should replace the localized contract after its owning Bead lands.

Evidence that would falsify or require revising this design:

- A current source owner showing that `result_ref` pins archive generation rather than logical query family.
- A landed generated route/client registry superseding `ROUTE_CONTRACTS` and localized `contracts.ts`.
- A remaining old-shell caller proving one of the listed deletion candidates is still required after route cutover.
- Browser evidence showing hydration mismatch, inaccessible disclosure controls, broken focus, or unusable narrow layout.
- Scale evidence showing compose-then-slice message reads violate declared latency/memory budgets.
- Security review rejecting archive-bearing unauthenticated loopback HTML under the existing shell policy.

## Recommended ownership and next decisions

A small certification repair belongs to the WebUI/browser evidence lane: run the provided build under an unblocked browser, add `/app` journeys, and fix only observed accessibility/layout/focus defects.

A legacy cutover belongs to daemon WebUI ownership after certification: redirect `/` and `/s/:session_id`, then delete only dominated symbols after call-site proof.

Snapshot-pinned continuation and paged semantic transcripts belong to the shared archive query/read substrate, not to Preact components. A local WebUI workaround would create a second product contract and should be rejected.

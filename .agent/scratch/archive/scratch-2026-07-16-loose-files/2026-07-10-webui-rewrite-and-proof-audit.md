---
created: 2026-07-10
purpose: Evidence-anchored audit of the current Polylogue web UI, daemon HTTP backend, tests, and Web-related Beads
status: complete-read-only-audit
project: polylogue
---

# Web UI rewrite and proof audit

## Scope and evidence boundary

This is a read-only source, test, history, Beads, and committed-media audit. I
did not edit production code or Beads, run the test suite, start a daemon, or
touch the live archive. At audit time `polylogue-index-rebuild.service` was
`activating` (`MainPID=582902`, started 2026-07-10 08:15:16 CEST) and
`polylogued.service` was inactive, so browser/runtime work remains an explicit
post-rebuild gate.

The operator's claim that the current Web UI is structurally broken and the
backend needs a rewrite is supported. The highest-risk findings are not matters
of taste: the recently closed concurrency bead leaves the normal split-archive
read path outside its executor; authenticated deployments serve a shell that
cannot authenticate its own requests; large reader routes materialize whole
sessions or the whole archive; and the supposed visual lane never executes the
JavaScript or a browser.

Inventory at this revision:

- 71 implemented HTTP route patterns (54 GET, 12 POST, 5 DELETE) and 71 separate
  descriptive `RouteContract` entries.
- `polylogue/daemon/http.py`: 3,827 lines.
- 11 `web_shell*.py` files: 9,918 lines of mostly embedded HTML/CSS/JavaScript.
- Entire `polylogue/daemon/` ring: 30,094 Python lines.
- 126 commits touched `daemon/http.py` and 54 touched the web-shell files in the
  available six-month history (the repository history shown is concentrated in
  May-July 2026). This is a hot product surface, not stable incidental plumbing.

## Current architecture and data flow

### Browser surface

1. `WEB_SHELL_HTML` is one Python raw string assembled at import time by chained
   `.replace()` injection of ten CSS/JS fragments
   (`polylogue/daemon/web_shell.py:21-28`, `2404-2424`). The root, session
   deep-link, and workspace paths all return this same bootstrap
   (`polylogue/daemon/http.py:1309-1318`). `/p` and `/a` are separate embedded
   pages (`http.py:1320-1337`). There is no `webui/`, TypeScript, component
   tree, frontend build, or static `dist/` yet.
2. One global mutable `state` object owns query, selection, panels, route
   readiness, caches, and coordination state (`web_shell.py:350-409`). The
   bootstrap launches sessions, facets, read profiles, user state, status, and
   coordination independently (`web_shell.py:2386-2398`).
3. Search/facet/session data is fetched with hand-written `fetchJSON` calls.
   Sessions have an 8 s client timeout and detail has 10 s
   (`web_shell.py:667-705`, `768-788`). Facets alone use abort/stale-request
   suppression (`web_shell.py:805-837`); session search does not.
4. The selected-session response already contains every message and is rendered
   by assigning generated strings to `innerHTML` (`web_shell.py:1291-1377`).
   Specialized fragments independently render reader cards, lineage,
   attachments, provenance, selection, workspace, and coordination.
5. Realtime uses `EventSource` with a polling fallback
   (`web_shell_realtime.py:197-270`). The server implements SSE as a one-second
   SQLite long poll on a request thread, capped at 30 seconds before reconnect
   (`polylogue/daemon/events_http.py:108-153`).

### HTTP/backend surface

1. `DaemonAPIHTTPServer` is still `ThreadingHTTPServer`: every accepted
   connection gets a request thread (`polylogue/daemon/http.py:3793-3815`).
   Dispatch is a mixture of static tables, parameter matching, and separate
   workspace/user-state dispatchers (`http.py:1304-1402`). Auth, Host/Origin,
   body parsing, async bridging, errors, and headers are manual middleware.
2. Async product operations are bridged by constructing a fresh `Polylogue`
   facade and calling `asyncio.run()` inside an eight-worker executor
   (`http.py:1249-1295`). Admission allows 8 running plus 16 queued tasks and
   rejects after saturation (`http.py:3775-3815`). A 30 s timeout stops the
   request from waiting; it does not stop a running query.
3. Normal live reads take a different path. Whenever current `source.db` and
   `index.db` exist, `_web_reader_archive_root()` succeeds
   (`http.py:361-377`) and list/detail/messages/stack/compare open
   `ArchiveStore` directly on the request thread. Only missing/mismatched
   archives fall back to the facade/executor path
   (`http.py:1827-1836`, `2333-2351`, `3310-3332`, `3338-3398`).
4. The direct list route compiles and re-lowers HTTP query fields itself, then
   does list/search plus a count on fresh read connections
   (`http.py:1955-2279`). Its SQLite cancellation helper is invoked with
   `deadline_s=None`, so it only notices socket disconnects, not a server-side
   deadline (`http.py:2158-2168`, `2249-2270`, `3450-3497`).
5. Route metadata and runtime dispatch are still parallel declarations.
   `route_contracts.py` explicitly says it is descriptive and dispatch remains
   elsewhere (`polylogue/daemon/route_contracts.py:1-7`). OpenAPI is another
   hand-maintained renderer over those contracts (`devtools/render_openapi.py:1-17`).

## Verified risks and their Beads

Severity here is about product correctness and proof, not only crash severity.

### P0/P1: backend correctness and operability

#### 1. The closed concurrency fix does not cover the primary live read path

`polylogue-0hqs` was closed as if all archive-query handlers now ran through a
bounded executor. The test itself makes that same claim
(`tests/unit/daemon/test_daemon_http_contracts.py:906-915`). In source, only
`_sync_run` uses the executor (`http.py:1255-1295`), while a healthy split
archive sends `/api/sessions`, `/api/sessions/:id`, `/messages`, stack, and
compare through direct synchronous `ArchiveStore` calls on unbounded
`ThreadingHTTPServer` request threads. The list route's per-SQL helper has no
deadline. `server_close()` also acknowledges that timed-out running workers
cannot be cancelled and may survive until SIGKILL (`http.py:3817-3825`).

Impact: the mechanism observed in 0hqs (request-thread accumulation and stuck
SQLite reads) is bounded only for facade-backed routes such as facets. Direct
session traffic can still accumulate request threads, and timed-out executor
work still occupies an admission slot indefinitely.

Tracker mapping:

- **Misclosed:** `polylogue-0hqs`. Its close reason accepted a structural proxy
  for the requested live concurrency proof and explicitly deferred the soak.
  The executor bypass is stronger than a missing confidence soak.
- **Direction exists:** `polylogue-dx1` (ratified ASGI migration) and
  `polylogue-3utv` (typed Starlette route registry).
- **Missing durable scope:** either reopen 0hqs for the bypass plus production
  soak, or add one narrowly linked residual bead. Do not create a second backend
  program.

#### 2. Token-authenticated deployments break the web shell by construction

With an API token, every API route requires `Authorization: Bearer ...`
(`http.py:1085-1109`), while loopback still serves the shell without auth
(`http.py:1309-1318`, `1431-1436`). The JS request builder sends only
`X-Request-ID` plus caller headers (`web_shell.py:451-478`); there is no token
bootstrap, Authorization header, or session credential anywhere in the shell.
Tests separately prove that `/` returns 200 and `/api/sessions` returns 401
under a configured token, but never execute the browser flow
(`tests/unit/daemon/test_web_reader.py:2361-2395`).

Impact: the server advertises a supported configuration whose first-party UI
cannot use any data route. Remote binding requires exactly this token
configuration (`polylogue/daemon/cli.py:836-847`).

Tracker mapping:

- `polylogue-kwsb.1` and `polylogue-2n39` cover adjacent daemon/web security,
  not first-party credential delivery.
- **Missing bead:** an auth-bootstrap/session-credential contract spanning
  shell, generated client, SSE, and Playwright. It should be a child/relation of
  dx1/3utv/bby.11, not a standalone security program.

#### 3. Two execution engines return materially different reader envelopes

The facade detail path returns repo/cwd/flags/summary and domain-derived
lineage fields (`http.py:2353-2418`). The primary direct-archive detail path
hardcodes `branch_type`, `parent_id`, `repo`, `cwd_display`, `model`, `flags`,
and `summary` to `None` (`http.py:2420-2477`). List/search likewise has a large
independent lowering/serialization implementation. This is correctness drift,
not only duplication.

Tracker mapping: `polylogue-t46`, `polylogue-4p1`, `polylogue-7le`, and
`polylogue-3utv` point toward one contract. `polylogue-bby.7` fixed one emitted
ID/detail-route break, but cannot structurally prevent another while two engines
remain. Sequence these contract beads before porting dozens of v2 panels.

#### 4. Several reader APIs scale as whole-archive or whole-session walks

- Paste browser: list every summary, fetch every session, walk every message
  until the requested output page fills (`http.py:1549-1598`). Offset does not
  reduce the upstream work.
- Attachment library: same N+1 whole-archive/session walk
  (`http.py:1625-1676`).
- `/messages`: `read_session()` materializes the entire session and only then
  slices the Python tuple (`http.py:3310-3332`).
- Detail returns every message and every attachment in one JSON body
  (`http.py:2420-2478`). Stack and compare repeat that full-detail operation per
  selected session (`http.py:3365-3398`).
- User-state bootstrap serially fetches all marks, annotations, saved views, and
  workspaces (`web_shell.py:734-766`); marks are unpaginated at the handler
  boundary (`user_state_http.py:179-199`).

Tracker mapping:

- `polylogue-bby.5` and `polylogue-37km` cover long-session navigation and
  rendering, not storage-level keyset/chunk reads.
- `polylogue-bby.8` covers perceived client performance but its premise that
  the current list renders 16k rows is stale: current state requests 100 rows
  (`web_shell.py:352-355`, `667-670`). Its useful parts are cancellation,
  cache/revalidation, and virtualization for genuinely unbounded panes.
- `polylogue-duti` observes slow facets in CLI but does not own these web API
  algorithms.
- **Missing bead:** one backend read-shape/scaling bead under dx1/3utv or the
  interactive-performance lane: SQL/keyset message pages, aggregate attachment
  and paste queries, bounded user overlays, and stack/compare projections.

### P1/P2: UI structure, usability, and security

#### 5. The UI is an untyped, unlinted private framework

Nearly 10k lines of HTML/CSS/JS live inside Python string constants and are
composed with textual replacements (`web_shell.py:21-28`, `2404-2424`). Runtime
rendering relies on `innerHTML` and inline event-handler strings throughout
(`web_shell.py:1291-1377`; `web_shell_reader.py:208-220`, `331-357`). There is
no frontend parser/typecheck/component test to prove the shipped JS even
executes.

Tracker mapping:

- **Correct direction:** `polylogue-bby.11`, ratified TS + Preact + Vite with
  committed offline assets, SSR semantic HTML, and progressive-enhancement
  islands.
- `polylogue-bby.6` correctly says not to extract the doomed old SPA; retain
  only its modal/token residuals.
- `polylogue-7le` and `polylogue-ap7` own renderer convergence and semantic
  transcript cards.
- `polylogue-1ilk` is no longer waiting for a stack decision: bby.11 is
  ratified and explicitly makes 1ilk binding scaffold acceptance. Treat it as
  part of the first v2 vertical slice, not a later test enhancement.

#### 6. Search and failure state can discard useful data or race

Facets cancel prior requests and ignore stale completions
(`web_shell.py:805-837`). Session search does neither; multiple debounced calls
can complete out of order, and any failure deletes the current rows even though
the route state initially claimed stale data was available
(`web_shell.py:667-705`). This is a direct explanation for flicker/false
emptiness under a struggling daemon.

Tracker mapping: `polylogue-bby.1` and the still-valid cancellation/cache parts
of `polylogue-bby.8`. Their AC must be proven with deterministic delayed and
out-of-order responses, not only ready demo data.

#### 7. Desktop-only layout and weak accessibility are untracked

The main shell fixes columns at `300px 1fr 320px`, locks the viewport to
`100vh`, and hides body overflow (`web_shell.py:49-56`). No `@media` rules exist
in any `web_shell*.py`. Many interactions are clickable `div`/`span` elements
or inline handlers, tabs lack tab roles/state, and the selection dialog has no
focus management despite its `role=dialog` marker
(`web_shell_reader.py:331-357`, `web_shell.py:304-313`,
`web_shell_selection.py:63-90`).

Tracker mapping:

- bby.11 notes mention phone/curl compatibility, but the scaffold AC does not
  require responsive viewports or accessibility.
- 1ilk specifies Playwright and screenshots, but not axe, keyboard-only
  journeys, focus, or reduced motion.
- **Missing bead/AC:** make WCAG-oriented semantics and keyboard/focus behavior
  a bby.11/1ilk acceptance slice. Do not file a parallel accessibility epic.

#### 8. The inline-script architecture remains a recurring XSS/CSP liability

`polylogue-2n39` correctly fixed a stored-content breakout and added contextual
escaping. The underlying architecture still generates inline JavaScript event
handlers from archive values, requiring three hand-maintained escape contexts
(`web_shell.py:411-429`). `_send_html` sets no CSP or other hardening headers
(`http.py:1174-1181`), and the inline handlers would prevent a strict no-inline
CSP. This is a residual structural risk, not a claim that a current exploit was
found.

Tracker mapping: bby.11 removes this architecture; bby.6/7le remove renderer
drift. The v2 definition of done should include CSP-compatible built assets and
no archive-data interpolation into executable contexts.

#### 9. Product hierarchy and visual direction are specified but not landed

The current shell exposes a dense three-pane inspector but no responsive
composition, full DSL discovery, aggregate view, or long-session structure.
`polylogue-bby` and children `.1-.15` map the missing cockpit workflows.
`polylogue-tjx1` is correctly closed as a design decision, with implementation
in `polylogue-9xuk` (generated tokens), `polylogue-bkzv` (provenance
vocabulary), and `polylogue-37km` (transcript measure/rail/tool outcomes).
Those should feed the v2 scaffold rather than produce another styling pass over
the old shell.

## What the current tests actually prove

### Strengths worth retaining

- Large endpoint/envelope suites drive the production HTTP server against
  synthetic SQLite archives and cover auth, errors, query shapes, mutations,
  privacy, XSS escaping, and many route contracts.
- `tests/visual` has useful deterministic fixtures for ready, empty,
  no-results, degraded FTS, overlays, workspaces, attachments, paste spans,
  costs, and insights.
- The PR CI runs `tests/visual` in `demo-visual-verify`
  (`.github/workflows/ci.yml:87-99`) even though the full coverage suite is
  intentionally post-merge only (`ci.yml:36-50`).

### Proof gaps

1. **“Visual” is browserless structure/API smoke.** Its DOM parser records only
   IDs, classes, script/style counts, viewport presence, and text
   (`tests/visual/conftest.py:39-77`). The harness uses `urllib` and
   `HTMLParser`, not a JS engine (`conftest.py:429-467`; documented honestly at
   `docs/visual-evidence.md:73-87`). Tests often prove only that function names
   and route strings occur in the shell (`test_reader_dom_smoke.py:34-111`,
   `393-427`). They do not click, render, focus, execute, or observe layout.
2. **The committed “visual tapes” do not show the Web UI.** I inspected frames
   from `reader-evidence-tour.gif` and `browser-capture-tour.gif`; both are
   terminal recordings of commands/report JSON. The documented specs also say
   the reader tape runs browserless smoke and prints report output
   (`docs/visual-evidence.md:35-50`). They are reproducible command evidence,
   not screenshots or an externally presentable UI demo.
3. **Tiny fixtures and loose latency cannot expose scale failures.** The visual
   corpus is three tiny sessions; the only reader route budget is “under 10 s”
   on that corpus (`tests/unit/daemon/test_web_reader.py:3170-3182`). No test
   combines a real concurrent convergence writer with HTTP load.
4. **Reader benchmarks do not benchmark HTTP endpoints.** Despite endpoint
   names in docstrings, `tests/benchmarks/test_reader_api.py` calls repository
   methods directly (`:39-65`, `:73-92`, `:100-113`); context and cost are
   explicit placeholders (`:137-186`). The SLO catalog nevertheless labels
   them reader/facets endpoint measurements (`docs/plans/slo-catalog.yaml:43-56`).
5. **The anti-vacuity lane is absent.** `devtools bench mutation status` reports
   `daemon-http` as `missing`. Worse, that campaign mutates all of 3,827-line
   `http.py` but runs only the five auth/origin helper tests in
   `tests/unit/daemon/test_daemon_http.py`
   (`devtools/mutation_scenario_catalog.py:258-266`; test file `:1-50`). It
   cannot validate the handler surface even if run.
6. **Contract declarations are checked against route names, not browser
   behavior.** `route_contracts.py` is descriptive, and current census tests
   cannot ensure that a generated client, runtime handler, response schema, and
   rendered view agree. `polylogue-stzx` and `polylogue-yeq` correctly identify
   this.

## Compact test architecture for v2 and the backend rewrite

Use one generated matrix, not separate ad hoc suites. Rows are user journeys;
columns are state, API contract, UI assertion, and retained proof artifact.

| Journey | Required states | API proof | Browser/UI proof | Retained artifact |
|---|---|---|---|---|
| Boot -> search -> open -> back | ready, empty, no-results, 401, 503, slow, out-of-order | route registry + OpenAPI response validation + emitted-ref walk | first useful paint, stale rows retained, latest query wins, focus/URL restored | trace + key screenshot + route ledger |
| Read a long session | 0, 1, 1k, 10k messages; tool/error/thinking/attachment | keyset pages, stable cursors, bounded payload | incremental transcript, constant DOM window, phase jump, copy/open ref | DOM-count/perf metrics + screenshots |
| Mark/annotate/save workspace | create/update/delete/conflict/auth expiry | typed mutation envelope and no unintended writes | optimistic state, rollback/error, modal focus, reload persistence | API log + final DB assertion |
| Live tail | append burst, reconnect, coalesced snapshot, duplicate/out-of-order event | monotonic cursor and cache invalidation | no flicker, one appended row/message, stale indicator on disconnect | event trace + video/trace |
| Evidence -> basket -> report/export | missing/degraded refs, re-ingest drift | content-hash/citation verifier and ref resolution | select, cite, export, reopen citation | verifier report + portable HTML screenshot |
| Stack/compare/lineage | missing member, cycle/quarantine, huge branch | bounded projection and honest degraded nodes | responsive compare, keyboard traversal, quarantined edge visible | snapshot + ref-walk report |

Layers and cadence:

1. **Vitest component/state lane, per PR.** Preact Testing Library against
   generated-client fixture builders. Test reducers/cache/state transitions,
   semantic transcript components, focus, and loading/stale/error rendering.
   Never hand-copy DTO interfaces.
2. **Python API/contract lane, per PR.** Starlette in-process client generated
   from `RouteSpec`; Pydantic/OpenAPI validation; emitted-ref walk; read requests
   leave durable tiers unchanged. `polylogue-3utv`, `stzx`, and `yeq` already own
   this direction.
3. **Playwright journey lane, per PR for a small Chromium smoke.** Start the
   ASGI app on an ephemeral port with the existing synthetic archive. Execute
   boot/search/open, one mutation, one auth-token session, and one live-tail
   journey. Record trace only on failure.
4. **Visual/responsive/a11y lane, master/nightly.** Screenshot canonical states
   at 390x844, 1440x900, and 2560x1440 in dark/light, with fonts and motion
   deterministic. Run axe plus keyboard-only focus-order/dialog/tab checks and
   reduced-motion. Rebaseline only in explicit visual changes.
5. **Fault lane, master/nightly.** A deterministic ASGI middleware/test adapter
   injects delay, 401/409/503, truncated JSON, dropped SSE, reconnect, and
   out-of-order completion. Assert truthful stale/error state and cancellation,
   not only eventual recovery.
6. **Scale/load/soak lane, scheduled/manual.** Synthetic 20k-session archive,
   10k-message session, large overlays/attachments, concurrent convergence-like
   writer, and multiple HTTP/SSE clients. Measure route p50/p95/p99, first paint,
   RSS, threads/tasks, file descriptors, SQLite busy time, response bytes, and
   DOM node count. A multi-hour live soak is the closure proof for 0hqs.
7. **External presentation gate, release/demo.** A clean-profile browser journey
   produces actual screenshots/video of the UI plus a machine manifest tying
   them to archive fixture, commit, routes, a11y result, and citation verifier.
   Terminal JSON tapes remain useful but are not labeled UI proof.

### Anti-vacuity requirements

Every new layer must demonstrate at least one seeded regression it catches,
then restore the code. Keep the demonstrations as mutation operators or fixture
faults, not committed broken branches:

- Change an emitted session ID so detail 404s -> ref-walk and Playwright fail.
- Remove the auth credential from generated fetch/SSE -> auth journey fails.
- Let an older search response overwrite the newest -> delayed-response state
  test fails.
- Clear stale rows on 503 -> state and screenshot fail.
- Bypass route admission/deadline on one direct read -> concurrent-load test
  fails and thread/task bound exceeds budget.
- Replace keyset paging with full `read_session()` -> response-byte/query-count
  budget fails.
- Remove a dialog label/focus trap or make a clickable div -> axe/keyboard test
  fails.
- Remove virtualization -> 10k-message DOM-count/scroll budget fails.
- Change a response field without regenerating the client -> registry/OpenAPI
  compile or schema test fails.
- Alter a provenance/unknown token -> component semantic snapshot and visual
  screenshot fail.

## Bead corrections, sequencing, and drop order

This stays inside the existing Beads program. No second tracker is needed.

### Corrections before implementation

1. Reopen or narrowly supersede `polylogue-0hqs` for the direct-archive executor
   bypass, cancellation, and live soak. Its current closure overclaims the
   runtime boundary.
2. Add the missing first-party token-auth journey as a linked dx1/3utv/bby.11
   bead or acceptance item.
3. Add one backend read-shape/scale bead for message keyset reads, attachment and
   paste set queries, bounded overlays, and stack/compare projections.
4. Add responsive/a11y acceptance to bby.11/1ilk rather than creating a second
   UX epic.
5. Reframe `bby.8`: the 16k list premise is stale under the current 100-row
   pagination. Keep cancellation, cache/cursor invalidation, optimistic
   navigation, and virtualization requirements for truly long panes.
6. Update `bby`/`37km` proof wording: current `tests/visual` cannot satisfy a
   screenshot or visual-regression AC. `1ilk` is the browser proof owner.
7. `dx1` is already ratified in notes but still titled as an open decision. Its
   remaining work is the blocker-hunt ASGI probe and measured migration ramp;
   record it that way so agents do not reopen the settled stack choice.
8. `3utv` says “every migrated route”; require the registry before the first new
   v2 route and a route census that shrinks the old dispatcher to zero.

### Minimal execution sequence

1. **Stop the live failure mechanisms:** cover direct reads with bounded
   cancellation/admission, fix token-auth bootstrap, and add the focused
   concurrent synthetic regression. This is a stabilization slice, not a reason
   to renovate the old UI.
2. **Finish the backend decision/ramp:** execute dx1's one-family Starlette probe
   with latency/RSS/extension compatibility measurements. Put the new family in
   `RouteSpec` from day one (`3utv`).
3. **Establish the contract spine:** land 4p1/t46 conformance decisions needed to
   make one read execution path; generate OpenAPI and the TS client from the
   runtime registry. Add stzx/yeq fast schema/ref checks.
4. **Ship one v2 proof vertical:** bby.11 + 1ilk together: SSR list and reader,
   Preact hydration only where interactive, generated tokens, generated client,
   token auth, responsive/a11y semantics, list/open/back Playwright, and an
   actual screenshot. Use existing APIs only where their bounded read shape is
   already proven.
5. **Migrate route families and core journeys:** sessions/messages first, then
   overlays/workspaces, insights/evidence, stack/compare, attachments/paste,
   events, operational endpoints. Preserve byte contracts where required; use
   registry census to show old-route count monotonically falling.
6. **Reach old-reader parity and delete it:** only after the matrix covers every
   old journey/state. Remove `web_shell*.py` application JS, its inline-handler
   escaping machinery, and old shell routes in the same PR that flips `/` to
   v2. Do not maintain `/app` and the old SPA indefinitely.
7. **Build cockpit features:** bby.1/.2/.5/.8 and evidence basket/report before
   timeline/replay/pinboard breadth, following bby.11's ratified evidence-first
   order. Every feature adds a matrix row/state, not a private test harness.
8. **Delete `BaseHTTPRequestHandler`:** after the route census reaches zero,
   remove old dispatch, `_sync_run`, duplicated route contracts, and the
   compatibility server. The ASGI app becomes the daemon HTTP owner.

Explicit non-work: do not implement bby.6's old-JS extraction; do not add more
old-shell panels; do not build another hand-typed route layer; do not call DOM
string checks visual proof; do not postpone component/browser tests until after
the scaffold.

## What still must be manually run and seen

After the index rebuild releases the permit:

1. Start only the normal daemon and open the current UI in a clean private
   browser. Capture desktop, 4K, and mobile screenshots; keyboard-walk search,
   list, transcript, every inspector tab, workspace/compare, and dialogs.
2. Repeat with `api_auth_token` enabled. The expected current result from source
   is shell 200 plus API 401; record it as the auth repro before fixing.
3. Reproduce direct-route pressure, not only facets: concurrent
   `/api/sessions`, a huge `/api/sessions/:id`, `/messages`, stack/compare, and
   one convergence cycle. Record thread/RSS/FD counts and p95/p99 through client
   aborts, then repeat after the stabilization slice.
4. Inspect a real 5k+ message session for response size, first contentful paint,
   DOM count, scroll/jump behavior, and memory. The tiny synthetic corpus cannot
   answer this.
5. Run axe and a full keyboard/focus audit in Chromium. Verify screen-reader
   names for tabs, dialog focus/return, dynamic route status, and non-color
   provenance cues.
6. Run the first Playwright matrix in Chromium and at least one second engine
   (Firefox or WebKit) before calling the rewrite externally presentable.
7. Produce one real UI screenshot/video artifact. The current committed GIFs
   show terminal workflows, not the reader.

## Bottom line

The existing backend is not an acceptable foundation for further cockpit
breadth. It has useful domain contracts and extensive endpoint fixtures, but
runtime dispatch, read execution, generated contracts, and the first-party UI
are not one system. The fastest path to a correct and presentable product is the
already-ratified ASGI + RouteSpec + TS/Preact direction, sequenced through a
small stabilization slice and a fully tested vertical reader slice. The key
bar is not “the scaffold builds”; it is that real browser journeys, failure
states, bounded load, auth, accessibility, and evidence export all have
non-vacuous proof before the old shell is deleted.

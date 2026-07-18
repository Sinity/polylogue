# WebUI v2 session list/read vertical — revision 02 handoff

## Mission, result, and revision delta

This cohesive revision implements the first production WebUI v2 list/read vertical against the attached Polylogue snapshot. It adds a semantic server-rendered session list and session reader, then hydrates only the controls that need client behavior: list filtering, continuation paging, transcript paging, and block expand/collapse. JavaScript-disabled readers receive the complete first page as ordinary form/list/article HTML. The daemon remains the owner of archive semantics, request identity, count evidence, route state, lineage, tool outcomes, and continuation state.

Revision 02 preserves the first implementation and repairs its highest-risk boundary rather than replacing it. The browser continuation utility no longer trusts a TypeScript cast or blindly appends any successful JSON page. It now validates both the server-rendered current page and each fetched page at runtime, rejects cross-origin endpoints before credential bootstrap, and refuses to merge pages when query identity, result-family identity, page size, starting offset, token advancement, or envelope invariants drift. Seven focused Vitest cases cover those rejections. A new production HTTP integration test boots the real daemon over a seeded split archive and proves SSR, token-only paging, archive/API/SSR semantic parity, first-party assets, and stable query/result refs in one composition path. Two Python security cases now exercise stored-content bootstrap escaping and encoded asset traversal.

The vertical lands under `/app` as a strangler slice. It does not remove the current shell, workspace, paste, or attachment routes. Existing `/api/sessions` and `/api/sessions/:id/messages` fields remain available; the patch adds the shared continuation vocabulary additively. `PATCH.diff` changes 29 files with 8,113 insertions and 72 deletions. No `FILES/` directory is included because the unified diff fully disambiguates every change.

## Snapshot identity and authority

- Supplied archive: `polylogue-all.tar(103).gz`
- Supplied archive SHA-256: `ec3fa193b2f99a6daee0bc620af4f69b5728834e99f8196792a17c3fbae11155`
- Snapshot generated: `2026-07-17T180950Z`
- Snapshot source recorded by the manifest: `/realm/project/polylogue`
- Branch: `master`
- Commit: `536a53efac0cbe4a2473ad379e4db49ef3fce74d`
- Commit subject: `fix(repair): harden raw authority convergence (#3046)`
- Commit timestamp: `2026-07-17T18:55:47+02:00`
- Snapshot manifest: `dirty: true`
- Exported branch-delta patch, file list, and log: all zero bytes
- All-refs bundle SHA-256: `b8595e2392d5a0587e083fe7c9510dce20b71dbdfeccfab39dcba14fb484e13a`
- Working-tree archive SHA-256: `9b0664c982b58a980e52f47af7a7466f6f5f3b3b3cf4914c16dba232639bc8bf`

The extracted current working-tree archive is therefore the source authority used for inspection, but the archive does not contain a recoverable named dirty patch. This handoff reports both facts and does not invent a dirty-patch identity.

No `webui-01` implementation or interface exists in the snapshot or reachable history. At `HEAD`, `webui/` contains the package/lock files, Playwright configuration, and first-party-auth journey only. The minimal scaffold assumption is: retain that package and authentication lane, add the ratified TypeScript + Preact + Vite component lane, emit fixed daemon-served assets, and integrate through the existing HTTP/auth/archive boundaries rather than introducing a parallel route framework.

## Inspected evidence

The implementation followed dependencies beyond the two visible routes. The inspected production owners include:

- Repository instructions in `CLAUDE.md`/`AGENTS.md`, `TESTING.md`, `CONTRIBUTING.md`, packaging configuration, generated-surface commands, and ignore/package-data rules.
- `polylogue/archive/query/transaction.py` for `QueryTransactionRequest`, `QueryContinuation`, `QueryResultPage`, `query_ref`, `result_ref`, and bounded execution vocabulary.
- `polylogue/core/enums.py` for canonical role, message-type, and material-origin tokens.
- `polylogue/archive/actions/parsing.py:tool_result_outcome` for provider-error/exit-code precedence.
- `polylogue/daemon/http.py`, `route_contracts.py`, `topology_http.py`, `web_auth.py`, `web_shell.py`, and `web_shell_reader.py` for dispatch, access policy, archive/live composition, topology, credentials, stable anchors, and exact legacy deletion candidates.
- Existing daemon route/security/reader/coherence tests and the browser credential harness.
- Relevant Beads records: `polylogue-bby.11`, `polylogue-1ilk`, `polylogue-3utv`, `polylogue-bby.7`, `polylogue-bby.8`, `polylogue-9xuk`, `polylogue-2n39`, and `polylogue-z9gh.3`.
- Relevant history for query transactions, semantic transcript rendering, first-party credentials, bounded cockpit routes, role consolidation, and the absence of `webui-01`.

Detailed findings and contradiction adjudication are in `EVIDENCE.md`.

## Delivered mechanism

`polylogue/daemon/webui_v2.py` is a narrow adapter around current production owners. It builds `QueryTransactionRequest` values using the exact shared vocabulary, validates daemon-issued continuations, projects only fields used by this vertical, emits context-safe JSON bootstrap state, and renders semantic first-page HTML. It does not reclassify archive records.

`polylogue/daemon/http.py` adds `/app` dispatch, allowlisted same-origin assets, SSR list/read composition, and continuation support on the existing JSON routes. Split-archive reads execute with the operation, logical arguments, page size, offset, projection, and stable order carried by the transaction request. The live-domain fallback preserves the same page contract.

`webui/src/` contains two Preact islands. The list island owns filters and session paging. The reader island owns message paging and disclosure controls. Neither island normalizes roles, material origin, message type, lineage, attachment state, or tool outcomes. The client obtains the existing first-party web credential and uses same-origin JSON only.

The committed offline assets are:

- `polylogue/daemon/static/dist/webui-v2.css`: 7,405 bytes; SHA-256 `71489101cc30bed895222f38392908db254ef5537727add1fbcc2a308f305125`
- `polylogue/daemon/static/dist/webui-v2.js`: 28,914 bytes; SHA-256 `58f06ff517e8590f6d6ec35bda61acc2875bce6420b70f0e90b9d5751566fba7`

## Route map

| Route | Method | Access | Response and purpose |
|---|---:|---|---|
| `/app` | GET | Existing unauthenticated-loopback shell bootstrap policy; token required when non-loopback | Semantic HTML for the first 25 session rows, filter form, evidence state, stable refs, and island state |
| `/app/s/:session_id` | GET | Same shell bootstrap policy | Semantic HTML for the session header, first 40 messages, lineage banner, structured blocks, attachments, evidence state, and island state |
| `/app/assets/webui-v2.css` | GET | Same shell bootstrap policy | Allowlisted committed CSS bytes |
| `/app/assets/webui-v2.js` | GET | Same shell bootstrap policy | Allowlisted committed ES-module bytes |
| `/api/sessions` | GET | Existing credential policy | Existing list/search behavior; plain list pages also expose continuation identity and evidence-total fields |
| `/api/sessions/:id/messages` | GET | Existing credential policy | Message page with canonical role/material-origin fields, structured blocks, attachments, and continuation identity |
| `/api/web-auth/session` | POST | Existing first-party credential bootstrap | Used by islands before JSON reads; behavior is not reimplemented by this patch |

The asset route accepts only the two compiled names above. Encoded traversal is rejected. Assets use fixed content types plus `nosniff`, no-referrer, and no-cache headers.

## JSON contracts consumed

### Shared continuation page envelope

The daemon emits these fields on plain session and message pages:

- `mode: string`
- `items: T[]`
- `messages: SessionMessage[]` on message pages as a compatibility alias
- `total: number | null`
- `total_relation: "exact" | "at_least" | "unknown"`
- `limit: number`
- `offset: number`
- `page_count: number`
- `has_more: boolean`
- `next_offset: number | null`
- `query_ref: string`
- `result_ref: string`
- `continuation: string | null`
- `route_state: RouteState`

Existing payload fields remain present. Ranked FTS/vector calls still return the established `SearchEnvelope`; this vertical does not issue ranked-search requests and does not reinterpret that envelope.

`RouteState` fields read by the islands are exactly:

- `state: string`
- `route: string`
- `reason?: string | null`
- `component?: string | null`
- `stale_available?: boolean | null`

### Session list requests and fields

First page:

```text
GET /api/sessions?origin=<token>&since=<date>&until=<date>&repo=<value>&limit=<page-size>
```

Only nonempty filters are sent. SSR uses the same four logical filters with page size 25.

Every advance:

```text
GET /api/sessions?continuation=<daemon-issued-q1-token>
```

No origin, date, repository, limit, offset, cursor, projection, or order is reconstructed by the browser.

List item fields consumed by `SessionList.tsx` are exactly:

- `id: string`
- `session_id: string`
- `title: string`
- `origin: string`
- `anchor: string`
- `created_at: string | null`
- `updated_at: string | null`
- `date: string | null`
- `message_count: number | null`
- `word_count: number | null`
- `repo: string | null`
- `cwd_display: string | null`
- `tags: string[]`
- `summary: string | null`

The live-domain path now emits `session_id` explicitly alongside legacy `id`; otherwise SSR could fall back to `id` while hydrated links encoded an absent field.

### Session reader requests and fields

First page:

```text
GET /api/sessions/<encoded-session_id>/messages?limit=<page-size>
```

SSR uses page size 40. Every subsequent page sends only the opaque continuation. The daemon rejects a token when its operation/projection/order does not match the route, its result ref does not recompute from its request, or its embedded stable session ref differs from the URL session ref.

Session header fields consumed are exactly:

- `id`, `session_id`, `title`, `display_title`, `origin`, `anchor`
- `created_at`, `updated_at`
- `message_count`, `word_count`
- `repo`, `cwd_display`, `tags`
- `branch_type`, `parent_id`
- `lineage_complete`, `lineage_truncation_reason`

Message fields consumed are exactly:

- `id: string`
- `role: "user" | "assistant" | "system" | "tool" | "unknown"`
- `text: string | null`
- `anchor: string`
- `timestamp: string | null`
- `message_type: "message" | "summary" | "tool_use" | "tool_result" | "thinking" | "context" | "protocol"`
- `material_origin: "human_authored" | "assistant_authored" | "operator_command" | "runtime_protocol" | "runtime_context" | "tool_result" | "generated_context_pack" | "generated_analysis_pack" | "unknown"`
- `duration_ms: number | null`
- `parent_message_id: string | null`
- `variant_index: number`
- optional `is_active_path`, `is_active_leaf`, `source_session_id`, `inherited_prefix`
- `word_count: number | null`
- `blocks: MessageBlock[]`
- `attachments: Attachment[]`

Block fields consumed are exactly:

- `id`, `type`, `text`
- `tool_name`, `tool_id`, `semantic_type`, `tool_input`
- `metadata`, `language`
- `tool_result_is_error`, `tool_result_exit_code`, `tool_result_outcome`

`tool_result_outcome` is projected through production `tool_result_outcome()`. Exit code remains authoritative over a provider error flag, matching current archive semantics.

Attachment fields consumed are exactly:

- `attachment_id`, `session_id`, `message_id`
- `name`, `mime_type`, `size_bytes`
- `path`, `state`

The UI displays name, MIME type, bounded size text, and state. It never displays the stored path.

Topology fields consumed are exactly:

- `target_id`, `root_id`
- `nodes[]` with `session_id`, `origin`, `title`, `depth`, `is_root`
- `edges[]` with `child_id`, `parent_id`, `parent_native_id`, `kind`, `resolved`
- `node_count`, `total_node_count`, `truncated_count`, `unresolved_edge_count`
- `cycle_detected`, `readiness`, `node_limit`

Missing topology counters render as unknown rather than zero. Fork/resume edge kinds are shown verbatim.

## Shared continuation utility and integrity boundary

`webui/src/lib/continuation.ts` is the shared client utility introduced by this vertical.

1. `sameOriginUrl()` resolves against `window.location.origin` and rejects a different origin before any network request.
2. `ensureWebCredential()` performs the existing first-party `POST /api/web-auth/session` once per browser lifecycle and resets after a failed bootstrap.
3. `parsePage()` performs runtime validation of the current SSR page and every fetched JSON page. It validates field types, nonnegative counts, positive limits, `page_count === items.length`, bounded page count, route-state shape, query/result-ref prefixes, and the relationships among `has_more`, `continuation`, `next_offset`, offset, and page count.
4. `fetchFirstPage()` accepts explicit initial filters, obtains the credential, and performs an authenticated same-origin GET.
5. `fetchContinuationPage()` accepts the complete current page rather than a loose token. It validates the current page before credential bootstrap, sends only its opaque continuation, validates the next envelope, then requires unchanged `query_ref`, unchanged `result_ref`, next offset equal to the daemon-issued prior `next_offset`, unchanged page size, and an advanced token.
6. Failed JSON calls surface daemon-provided `detail`/`error` strings when available; otherwise they retain a status-derived error.

Server-side validation lives in `decode_list_continuation()` and `decode_message_continuation()`. It mirrors `QueryContinuation`, `QueryTransactionRequest`, `QueryResultPage`, `query_ref`, `result_ref`, `next_offset`, and `continuation` from `archive/query/transaction.py` rather than creating a second cursor format.

The current shared substrate’s `result_ref` identifies a logical result family; it is not a database snapshot pin. This patch prevents client-side family drift but does not claim mutation isolation that the substrate does not provide.

## Evidence honesty and semantic decisions

- Exact totals render as exact counts.
- Lower bounds render as `At least N` with an explanation that the number is not exact.
- Unknown totals render as unknown. An empty unknown page explicitly says zero was not established.
- Provider-marked tool failures render visibly, failed details are open by default, `data-outcome="failed"` is present, and exit-code or provider-flag evidence is exposed.
- Roles, message types, and material-origin tokens are projected and rendered verbatim. The browser has no normalization table beyond exact TypeScript token declarations copied from current enums.
- Stable session refs are visible in list/read cards. Session links encode `session_id`; message articles use archive anchors, display `message_id`, and provide permalinks.
- The lineage banner uses the current bounded topology envelope and does not infer missing ancestry.
- Bootstrap JSON escapes `&`, `<`, `>`, U+2028, and U+2029 before insertion into `application/json`. Visible text/attributes use HTML-context escaping.
- Stored attachment paths are not rendered.

## Sanitized fixture inventory

- `webui/src/fixtures/session-list-page.json`: one synthetic first-page session, exact total two, and an opaque continuation token.
- `webui/src/fixtures/session-read-page.json`: a synthetic fork child, inherited-prefix evidence, tool use, a failed tool result with exit code 2, an attachment, and a family containing fork and resume edges.
- `webui/src/lib/continuation.test.ts`: synthetic next-page variants used to induce query-ref drift, result-ref drift, offset drift, page-size drift, malformed envelopes, malformed SSR state, and cross-origin endpoint rejection.
- `tests/unit/daemon/test_webui_v2_routes.py`: synthetic session/message/topology rows, temporary archive roots, a hostile title used only for escaping assertions, and encoded traversal input.
- `tests/unit/daemon/test_web_reader.py::TestWebUIV2Vertical`: the repository’s deterministic synthetic archive fixture and real production daemon.

No operator conversation content is present in the patch or package.

## Changed files

Production and generated contract surfaces:

- `.gitignore`
- `polylogue/daemon/http.py`
- `polylogue/daemon/route_contracts.py`
- `polylogue/daemon/webui_v2.py`
- `polylogue/daemon/static/dist/webui-v2.css`
- `polylogue/daemon/static/dist/webui-v2.js`
- `docs/openapi/search.yaml`
- `docs/plans/topology-target.yaml`
- `docs/topology-status.md`

WebUI source and tooling:

- `webui/package.json`
- `webui/package-lock.json`
- `webui/tsconfig.json`
- `webui/vite.config.ts`
- `webui/src/main.tsx`
- `webui/src/styles.css`
- `webui/src/lib/contracts.ts`
- `webui/src/lib/continuation.ts`
- `webui/src/lib/continuation.test.ts`
- `webui/src/lib/format.ts`
- `webui/src/components/SessionList.tsx`
- `webui/src/components/SessionList.test.tsx`
- `webui/src/components/SessionRead.tsx`
- `webui/src/components/SessionRead.test.tsx`
- `webui/src/tests/setup.ts`
- `webui/src/fixtures/session-list-page.json`
- `webui/src/fixtures/session-read-page.json`

Python tests:

- `tests/unit/daemon/test_webui_v2_routes.py`
- `tests/unit/daemon/test_route_contracts.py`
- `tests/unit/daemon/test_web_reader.py`

Topology generation refreshed pre-existing line-count rows outside this vertical because a new `polylogue/` module changes generated source accounting. Those source files are not changed by this patch. Retaining generated output is required for `devtools render all --check` to remain clean.

## Acceptance matrix

| Requirement | Implementation | Verification | Status |
|---|---|---|---|
| Semantic first-page SSR for list/read | `/app` and `/app/s/:session_id` emit forms, lists, articles, refs, lineage, blocks, attachments, and evidence before bootstrap state | Seven focused route tests plus real-daemon HTTP composition test | Met |
| JS-disabled readability | Meaningful page content precedes JSON state and does not require hydration | HTTP integration asserts semantic list/transcript before bootstrap; manual production HTTP read confirms 15 list cards and 4 message cards | Met |
| Islands for filtering, paging, disclosure | Preact list/read components hydrate server markup | Component tests, production build hydration in jsdom, and live-daemon hydrated filter exercise | Met at component/DOM level; real browser layout unverified |
| Continuation-only advances | Client sends only the daemon-issued token | Component URL assertion, runtime utility tests, Python token restoration, and live HTTP page 2 | Met |
| Page-family integrity | Runtime envelope and cross-page identity checks before row merge | Seven continuation tests | Met |
| Shared transaction vocabulary | Server request/token/page identity uses current query substrate | Stable refs, result-ref recomputation, route/session boundary tests | Met |
| Exact/qualified/unknown totals | `total_relation` controls count and empty-state text | SSR/component unknown-state assertions | Met |
| Visible structured failures | Canonical outcome, exit/error evidence, open failed details | Reader fixture/component and SSR route tests | Met |
| Stable session/message deep links | Encoded session route, archive anchor, visible IDs, permalinks | SSR/component/integration assertions | Met |
| Canonical role/material-origin semantics | Direct production projection; no browser reinterpretation | Fixture and real HTTP SSR/API parity assertions | Met |
| Sanitized fixtures | Synthetic refs/content only | Fixture and package inspection | Met |
| Same-origin, zero CDN | URL guard and fixed local asset allowlist | Utility test, asset traversal test, build/source/package scan | Met |
| Stored-content XSS boundary | Contextual HTML escaping and script-safe JSON | Synthetic route test decodes exact state while proving no raw executable markup | Met |
| Vitest list/read tests | Components plus shared continuation utility | 3 files, 10 tests passed | Met |
| Python SSR/route tests | Seven focused tests and one real-daemon vertical integration | Included in 661-test daemon selection | Met |
| Legacy list/read retirement | Exact candidates identified below; no deletion in this patch | Integrator action remains | Deliberately deferred |
| Native browser/accessibility/responsive proof | Existing Playwright lane invoked | Chromium policy blocked loopback navigation before Polylogue | Unverified |

## Legacy list/read supersession and deletion candidates

This vertical supersedes the behavior of the legacy `/` list landing and `/s/:session_id` reader, but remains at `/app` and `/app/s/:session_id` for certification before route cutover.

Deletion candidates after cutover and browser certification:

- In `polylogue/daemon/web_shell.py`: `loadSessions`, `loadSession`, `loadMoreSessionMessages`, `renderSessions`, `selectSession`, and the session-list/session-detail branches and event plumbing inside `renderMain` that serve `/` and `/s/:session_id`.
- In `polylogue/daemon/web_shell_reader.py`: `READER_CSS`, `READER_JS`, `READER_HELP_HTML`, and especially `renderMessageBlocks`, but only after all remaining old-shell callers migrate.
- Offset previous/next controls and list facet/search DOM handlers used solely by the old list/read vertical.

Not deletion candidates from this package:

- `/w/:mode`, `/p`, and `/a`.
- `_serve_web_shell` as a whole.
- Workspace, paste, attachment-library, semantic-card, or selection features owned by the old shell.
- `web_shell_reader.py` as a whole before call-site retirement; other legacy modules still compose it.

No existing test/helper or legacy production file is deleted by `PATCH.diff`.

## Apply and integration order

1. Check out `536a53efac0cbe4a2473ad379e4db49ef3fce74d` on `master`.
2. Run `git apply --check PATCH.diff` and then `git apply PATCH.diff`.
3. In `webui/`, run `npm ci`, `npm run check`, `npm test -- --run`, and `npm run build`.
4. At repository root, run the Ruff, format, strict mypy, byte-compilation, focused pytest, and coherence commands in `TESTS.md`.
5. Run `python -m devtools render all --check` and confirm all generated surfaces remain synchronized.
6. Build a wheel with `UV_OFFLINE=1 UV_PROJECT_ENVIRONMENT="$PWD/.venv" uv build --wheel .` and confirm `webui_v2.py` plus both assets are present.
7. Run `/app` Playwright journeys in a browser without the managed `URLBlocklist` policy. Certify JS-disabled SSR, credential bootstrap, filtering, continuation, failed-tool disclosure, stable deep links, keyboard/focus order, accessibility tree, and narrow/wide layout.
8. Only after browser certification should an integrator redirect `/` and `/s/:session_id` and delete dominated legacy symbols.

## Verification performed

The exact commands, production dependencies, anti-vacuity mutations, and results are recorded in `TESTS.md`. Final evidence includes:

- Strict TypeScript check: passed.
- Vitest: 3 files and 10 tests passed, including seven continuation-boundary tests.
- Vite production build: passed; 9 modules transformed; committed CSS 7,405 bytes and JS 28,914 bytes.
- Offline npm audit: zero vulnerabilities. A later registry-backed audit call returned HTTP 400 `Invalid package tree`; `npm ls --all` and the offline audit were clean, so this is recorded as an external registry endpoint discrepancy rather than concealed.
- Ruff check and Ruff format check across six changed Python files: passed.
- Strict mypy across six changed Python files: no issues.
- Python byte compilation for the three changed production modules: passed.
- Daemon route/security/contract/vertical selection: 661 passed in 6.37 seconds.
- Directly affected reader selection: 9 passed, 140 deselected in 5.58 seconds.
- Read-surface coherence: 50 passed in 0.80 seconds.
- Generated surfaces and local links: synchronized.
- Real production daemon over deterministic demo archive: `/app` 200 with 15 semantic cards; first-party credential 201 with one HttpOnly cookie; token-only page 2 preserved query/result refs and advanced offset 0→2; read page 200 with four semantic message cards and visible failed-tool state; both assets returned exact committed bytes.
- Production-bundle hydration in jsdom: list and read island DOM remained byte-for-byte unchanged and emitted no console errors. A hydrated live-daemon filter reduced 15 cards to seven `codex-session` cards through credential bootstrap and same-origin JSON.
- Wheel: 4,194,666 bytes, SHA-256 `3170516e40f22c77ae5af8048f4da9604b132ebe020803d1e48c3fe35dbe9f7a`; includes `polylogue/daemon/webui_v2.py` and both exact assets.
- Patch application: clean `git apply --check`; all 29 resulting file SHA-256 values match the implementation tree; `git diff --check` is clean.

The full `tests/unit/daemon/test_web_reader.py` run reached 148 passed and one failure. The failing privacy case deletes `index.db` and then receives HTTP 500 from provider usage. The identical single test fails on a pristine checkout of the named snapshot with the same `sqlite3.OperationalError: unable to open database file`; it is a pre-existing baseline defect, not introduced by this patch.

Playwright did start the production server, but the first `page.goto()` failed with `net::ERR_BLOCKED_BY_ADMINISTRATOR`. The container’s managed Chromium policy is `/etc/chromium/policies/managed/000_policy_merge.json` with `"URLBlocklist": ["*"]`. Two remaining tests did not run. This does not certify browser behavior and is not represented as a product pass.

## Risks, limitations, and remaining value

The main remaining gap is native-browser certification. SSR, real HTTP composition, the production bundle, hydration parity, and live filtering have been exercised, but rendered layout, computed accessibility tree, keyboard focus behavior, responsive screenshots, and actual browser disclosure/paging interactions remain unverified because Chromium blocks every URL in this environment.

The current `QueryTransaction.result_ref` is stable logical result-family identity, not a pinned database snapshot. Inserts or changes during a multi-page walk may affect page membership according to current storage/query behavior. This patch rejects identity drift but intentionally does not invent snapshot semantics.

The split-archive session reader composes the full semantic transcript and then slices the requested page. That preserves lineage, inherited-prefix, attachment, and semantic-block correctness using current owners, but may be expensive for exceptionally large transcripts. A substrate-level paged semantic transcript API would be a substantial separate project.

`/app` inherits the current shell’s unauthenticated-loopback bootstrap policy and therefore exposes archive content to local processes that can reach the daemon. That is current product architecture, not a new credential model; it remains an explicit security tradeoff for route cutover review. Interactive JSON remains protected by the first-party cookie flow.

Another small iteration has meaningful value only as browser certification in an unblocked environment and, if defects appear, focused accessibility/layout repairs. A substantial second pass is justified only for one of three larger changes: legacy route cutover/removal, a snapshot-pinned continuation substrate, or a genuinely paged semantic transcript owner. Reworking the current implementation without one of those goals is unlikely to add much value.

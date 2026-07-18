# WebUI v2 session list/read — revision 02 test design and results

## Test strategy

The suite is split at the production boundaries that can invalidate this vertical: daemon dispatch and semantic SSR, archive continuation identity, runtime client envelope validation, live-domain compatibility, canonical reader projection, first-party credential composition, client rendering, generated route surfaces, and wheel packaging. Each test below names the production dependency it exercises and a representative mutation/removal that should make it fail.

All fixtures are synthetic. No operator archive or conversation content was used.

## Python route and security tests

`tests/unit/daemon/test_webui_v2_routes.py` contains seven tests.

### `test_session_list_route_ssr_emits_semantic_first_page`

Production dependency: `DaemonAPIHandler.do_GET`, `_serve_webui_v2_session_list`, `_do_archive_list_sessions`, `continuation_page_payload`, and `render_session_list_document`.

Assertions: semantic filter form/list/article markup, stable session refs and links, committed asset URLs, bootstrap JSON, exact total relation, and query/result refs.

Anti-vacuity mutation: replace the renderer with an empty mount, bypass the archive list adapter, omit transaction identity, or move all session content behind JavaScript.

### `test_session_list_ssr_escapes_visible_content_and_json_bootstrap`

Production dependency: `_a`, `_e`, `_safe_json`, and the real list renderer.

Assertions: hostile synthetic title content is escaped in visible HTML, `<` is neutralized inside bootstrap JSON, the state script still parses as JSON, and parsing reconstructs the exact original synthetic title.

Anti-vacuity mutation: use plain interpolation in text/attributes, call `json.dumps()` without script-safe escaping, or test only for absence without decoding the intended value.

### `test_webui_v2_asset_route_rejects_encoded_traversal`

Production dependency: route decoding, `webui_v2_asset_path`, and the fixed asset allowlist.

Assertions: `/app/assets/..%2Fhttp.py` returns 404 rather than reading a neighboring Python module.

Anti-vacuity mutation: join the decoded asset suffix directly onto the static directory or use suffix/extension-only admission.

### `test_session_read_route_ssr_renders_failure_attachment_and_fork_family`

Production dependency: `_serve_webui_v2_session_read`, `_do_archive_webui_v2_session_read`, canonical block projection, `tool_result_outcome`, attachment projection, topology envelope, and reader renderer.

Assertions: role/material-origin attributes, visible failed outcome and exit code, failed details open by default, attachment state, lineage link/kind, and stable message ref.

Anti-vacuity mutation: flatten blocks to text, omit provider outcome fields, ignore exit codes, drop attachments/topology, or return only a client mount.

### `test_list_api_continuation_restores_filters_and_stable_result_ref`

Production dependency: `QueryResultPage.continuation`, `decode_list_continuation`, `_handle_list_sessions`, and the shared request identity.

Assertions: a token-only second request restores origin, page size, and offset and preserves `query_ref` and `result_ref`.

Anti-vacuity mutation: reconstruct filters/offset from second-request query keys, expose only `next_offset`, or recompute a different logical request.

### `test_live_list_payload_keeps_session_id_for_hydrated_links`

Production dependency: `_do_list` fallback payload.

Assertions: `session_id` is present after hydration alongside legacy `id`.

Anti-vacuity mutation: remove explicit `session_id`; server markup may still fall back to `id`, but the typed island link breaks.

### `test_message_continuation_cannot_cross_session_boundaries`

Production dependency: `decode_message_continuation` and `_handle_get_messages`.

Assertions: a structurally valid token for session A cannot page session B.

Anti-vacuity mutation: validate operation/projection/order only and ignore the embedded stable session ref.

## Real production-daemon composition test

`tests/unit/daemon/test_web_reader.py::TestWebUIV2Vertical::test_semantic_ssr_and_continuation_identity_use_the_real_daemon` boots `DaemonAPIHTTPServer` over the repository’s deterministic split-archive fixture. It does not mock the renderer, transaction, HTTP server, static asset reader, list query, or message query.

Production dependencies: `DaemonAPIHTTPServer`, `DaemonAPIHandler`, archive store/query composition, both SSR renderers, `QueryTransaction` continuations, message canonicalization, and committed assets.

Assertions:

- `/app` returns HTML with semantic form/list content before bootstrap state.
- Bootstrap state carries exact total evidence and query/result refs.
- A first list page with limit one produces a continuation.
- A second request containing only that continuation starts at the exact issued next offset, keeps query/result refs, and returns a different session row.
- `/app/s/:id` returns semantic message flow before bootstrap state.
- SSR and the JSON message route agree on stable message ID, role, and material origin.
- Both first-party assets return with expected content types and nonempty bodies; the JavaScript contains no `https://` dependency.

Anti-vacuity mutation: replace either SSR route with an empty mount, bypass the archive, reconstruct an offset on page two, alter canonical role/material-origin projection, or remove packaged assets.

`tests/unit/daemon/test_route_contracts.py` also extends route discovery and loopback/non-loopback access matrices for `/app`, `/app/s/:session_id`, and `/app/assets/:asset`. Removing those route declarations or changing their shell-access classification makes the generated-contract tests fail.

## Frontend tests

### Shared continuation utility

`webui/src/lib/continuation.test.ts` contains seven tests:

1. Query-ref drift is rejected before rows can be merged.
2. Result-ref drift is rejected before rows can be merged.
3. A page starting anywhere except the prior daemon-issued `next_offset` is rejected.
4. Page-size drift is rejected.
5. A malformed envelope (`page_count !== items.length`) is rejected instead of trusted through a TypeScript cast.
6. Malformed server-rendered current state is rejected before credential bootstrap or paging.
7. A cross-origin endpoint is rejected before any network call.

Production dependency: `parsePage`, `sameOriginUrl`, `fetchContinuationPage`, credential bootstrap ordering, and `validateAdvance`.

Representative anti-vacuity mutations: return `response.json() as ContinuationPage<T>`, pass only a loose token into the utility, skip current-state validation, remove one identity comparison, use caller-computed offsets, or resolve endpoints without comparing origins.

### Session list component

`webui/src/components/SessionList.test.tsx` contains two tests.

- Paging mocks credential bootstrap and one next page, then inspects the actual URL. The only second-page query key must be `continuation`; origin and offset must be absent. It also proves row append and terminal-button state.
- Unknown-empty evidence proves `total: null` plus `total_relation: unknown` renders an unknown total and explicitly refuses to claim zero results.

Production dependency: `SessionList`, filter/page state, and the shared continuation utility.

Representative anti-vacuity mutations: reconstruct filters/offsets on advance, replace page state without appending rows, or coerce null total to zero.

### Session reader component

`webui/src/components/SessionRead.test.tsx` renders the sanitized fork/resume fixture.

Assertions: verbatim role/material-origin/message-type tokens, failed tool-result labeling, exit code 2, failure output, attachment metadata/state, stable message permalink, fork/resume family evidence, failed block initially open, and collapse/expand behavior.

Production dependency: `SessionRead`, structured block renderer, topology banner, attachment renderer, and disclosure controls.

Representative anti-vacuity mutations: normalize canonical tokens, flatten structured blocks, ignore exit code, close failed results by default, omit family links, or remove stable anchors.

## Commands and successful results

### Frontend type checking, components, and production build

Run from `webui/`:

```text
npm run check
npm test -- --run
npm run build
npm audit --offline --audit-level=low
npm ls --all
```

Results:

- TypeScript `tsc --noEmit`: passed.
- Vitest 3.2.7: 3 files passed, 10 tests passed in 11.56 seconds.
  - `src/lib/continuation.test.ts`: 7 passed.
  - `src/components/SessionRead.test.tsx`: 1 passed.
  - `src/components/SessionList.test.tsx`: 2 passed.
- Vite 7.3.6: 9 modules transformed; build passed in 213 ms.
- CSS: 7,405 bytes, gzip report 2.14 kB, SHA-256 `71489101cc30bed895222f38392908db254ef5537727add1fbcc2a308f305125`.
- JavaScript: 28,914 bytes, gzip report 10.24 kB, SHA-256 `58f06ff517e8590f6d6ec35bda61acc2875bce6420b70f0e90b9d5751566fba7`.
- Offline npm audit: `found 0 vulnerabilities`.
- `npm ls --all`: exit status 0.

A later registry-backed `npm audit --audit-level=low` call returned HTTP 400 with `Invalid package tree, run npm install to rebuild your package-lock.json`. The exact lock resolves cleanly under `npm ls`, the offline advisory check reports zero, and an earlier registry-backed audit succeeded. The external registry response is retained here as a limitation rather than represented as a clean final online audit.

### Changed Python production and tests

```text
.venv/bin/python -m ruff check \
  polylogue/daemon/webui_v2.py \
  polylogue/daemon/http.py \
  polylogue/daemon/route_contracts.py \
  tests/unit/daemon/test_webui_v2_routes.py \
  tests/unit/daemon/test_route_contracts.py \
  tests/unit/daemon/test_web_reader.py

.venv/bin/python -m ruff format --check \
  polylogue/daemon/webui_v2.py \
  polylogue/daemon/http.py \
  polylogue/daemon/route_contracts.py \
  tests/unit/daemon/test_webui_v2_routes.py \
  tests/unit/daemon/test_route_contracts.py \
  tests/unit/daemon/test_web_reader.py

.venv/bin/python -m mypy \
  polylogue/daemon/webui_v2.py \
  polylogue/daemon/http.py \
  polylogue/daemon/route_contracts.py \
  tests/unit/daemon/test_webui_v2_routes.py \
  tests/unit/daemon/test_route_contracts.py \
  tests/unit/daemon/test_web_reader.py

.venv/bin/python -m py_compile \
  polylogue/daemon/webui_v2.py \
  polylogue/daemon/http.py \
  polylogue/daemon/route_contracts.py
```

Results:

- Ruff: all checks passed.
- Ruff format: six files already formatted.
- Strict mypy: no issues in six source files.
- Byte compilation: passed.

### Daemon route, contract, security, and real vertical integration selection

```text
.venv/bin/python -m pytest -q -n 0 \
  -p no:randomly -p no:random-order \
  tests/unit/daemon/test_webui_v2_routes.py \
  tests/unit/daemon/test_route_contracts.py \
  tests/unit/daemon/test_daemon_http_contracts.py \
  tests/unit/daemon/test_daemon_http_security.py \
  tests/unit/daemon/test_web_reader.py::TestWebUIV2Vertical
```

Result: `661 passed in 6.37s`.

The explicit plugin disables match repository verification behavior. They prevent unrelated random-order plugin seed failures from obscuring product assertions.

### Directly affected legacy reader selection

```text
.venv/bin/python -m pytest -q -n 0 \
  -p no:randomly -p no:random-order \
  tests/unit/daemon/test_web_reader.py \
  -k 'TestWebUIV2Vertical or archive_session_list_route_uses_bounded_sql_helper or session_detail_returns_header_and_messages or session_messages_envelope_carries_messages_and_total or archive_file_set_session_detail_and_messages_from_archive_tiers or browser_capture_reader_boundary_keeps_text_and_escapes_shell or empty_archive_returns_zero_envelope or degraded_search_index_returns_route_state_not_zero_results or message_endpoint_clamps_oversized_pages'
```

Result: `9 passed, 140 deselected in 5.58s`.

### Cross-surface read coherence

```text
.venv/bin/python -m pytest -q -n 0 \
  -p no:randomly -p no:random-order \
  tests/unit/test_read_surface_coherence.py
```

Result: `50 passed in 0.80s`.

### Generated surfaces

```text
.venv/bin/python -m devtools render all --check
```

Result: all generated CLI schemas, OpenAPI, devtools reference, demo datasheet, quality/workflow docs, docs surface, MCP equivalence/index, and local site links synchronized. The generated OpenAPI and topology files in the patch are the exact synchronized outputs.

### Wheel build and content

```text
UV_OFFLINE=1 UV_PROJECT_ENVIRONMENT="$PWD/.venv" \
  uv build --wheel . --out-dir /tmp/polylogue-wheel-r02
```

Result:

- Wheel: `polylogue-0.2.0-py3-none-any.whl`
- Size: 4,194,666 bytes
- SHA-256: `3170516e40f22c77ae5af8048f4da9604b132ebe020803d1e48c3fe35dbe9f7a`
- Included `polylogue/daemon/webui_v2.py`: 34,510 bytes
- Included CSS: 7,405 bytes
- Included JavaScript: 28,914 bytes
- `uv.lock` remained unchanged.

### Real daemon and deterministic demo archive

A one-shot local harness launched the production `tests/browser/web_auth_server.py` with `.venv/bin/python`, a temporary synthetic archive, and a 60-second credential TTL. It then used an HTTP cookie jar against the reported loopback URL.

Observed results:

- Demo seed: 15 sessions and 62 messages.
- `GET /app`: 200, `text/html`, 25,718 bytes, 15 semantic session cards, filter form, JSON bootstrap, and only local CSS/JS references.
- `POST /api/web-auth/session` with the existing first-party headers: 201, state `ready`, one HttpOnly cookie.
- `GET /api/sessions?limit=2`: exact total 15, offset 0, `q1.` continuation.
- Page two requested with only `continuation`: 200, offset 2, stable query ref, stable result ref.
- `GET /app/s/codex-session%3Ademo-terminal-error`: 200, 30,406 bytes, four semantic message cards, stable message refs, and visible failed-tool state.
- Message JSON: four items, canonical user/assistant/tool role and human/assistant/tool-result material-origin pairs, and at least one `tool_result_outcome: failed` block.
- CSS and JS: 200 with exact committed sizes and hashes.

The demo’s failed result did not contain exit code 2; that exit-code branch is covered by the sanitized reader fixture and Python SSR test. The live check therefore claims visible failed state, not a live exit-code assertion.

### Production-bundle hydration without a native browser

The exact production SSR documents and committed JavaScript were loaded into jsdom with browser globals installed. Before/after island `outerHTML` and console errors were compared.

Results:

- Session list island: 12,192 bytes before and after; unchanged; no console errors.
- Session reader island: 5,226 bytes before and after; unchanged; no console errors.

A separate hydrated list run against the real daemon completed credential bootstrap and `GET /api/sessions?origin=codex-session&limit=25`; the UI changed from 15 cards to seven, all with origin `codex-session`, with no alert or console error. This proves production-bundle/daemon composition at DOM level, not visual-browser behavior.

### Patch application and content integrity

```text
git diff --cached --check
git diff --cached --binary --full-index HEAD > PATCH.diff
git apply --check PATCH.diff
git apply PATCH.diff
```

The patch applies cleanly to a fresh clone at the named commit. All 29 patched file SHA-256 values match the implementation tree exactly. `git diff --check` is clean. Patch statistics are 29 files, 8,113 insertions, and 72 deletions. The patch is 366,129 bytes before ZIP compression.

Human-authored additions were scanned for `TODO`, `FIXME`, `TBD`, `PLACEHOLDER`, `PSEUDOCODE`, `INSERT HERE`, and template markers. No implementation placeholder was found. Literal HTML `placeholder="owner/repo or URL"` attributes and explanatory uses of the word “placeholder” are not unfinished code. The patch contains no supplied archive filename, prompt filename, `/mnt/data` path, or copied project-state archive.

## Full reader file and baseline comparison

Command:

```text
.venv/bin/python -m pytest -q -n 8 \
  -p no:randomly -p no:random-order \
  tests/unit/daemon/test_web_reader.py
```

Result: 148 passed and one failed in 32.58 seconds.

Failure: `TestReaderAssertionEndpoint::test_operational_web_payloads_redact_configured_archive_paths`. The test deletes the configured archive `index.db`, then `/api/provider-usage` returns HTTP 500 because `ArchiveStore.open_existing()` raises `sqlite3.OperationalError: unable to open database file`.

Baseline falsification command, run from a pristine clone of `536a53efac0cbe4a2473ad379e4db49ef3fce74d` with the same environment:

```text
<shared-venv-python> -m pytest -q -n 0 \
  -p no:randomly -p no:random-order \
  tests/unit/daemon/test_web_reader.py::TestReaderAssertionEndpoint::test_operational_web_payloads_redact_configured_archive_paths
```

Baseline result: the same test fails with the same HTTP 500 and `unable to open database file`. This is a pre-existing snapshot defect, not a regression introduced by the vertical.

## Native-browser attempt and unverified evidence

Command from `webui/`:

```text
UV_OFFLINE=1 UV_PROJECT_ENVIRONMENT="$PWD/../.venv" \
  npm run test:e2e -- --reporter=line
```

The production test server started and Playwright discovered three tests. The first navigation failed before Polylogue content was reached:

```text
page.goto: net::ERR_BLOCKED_BY_ADMINISTRATOR at http://127.0.0.1:38089/
```

The container’s managed Chromium policy file contains `"URLBlocklist": ["*"]`. One test failed at the first `page.goto()` and two did not run. Therefore the following remain unverified in a native browser:

- JavaScript-disabled browser rendering rather than raw HTTP semantics.
- Computed accessibility tree and screen-reader naming.
- Keyboard and focus behavior after filtering/paging/disclosure changes.
- Responsive layout and screenshot stability at narrow/wide viewports.
- Browser-native credential/cookie behavior beyond HTTP and jsdom composition.

No browser pass is claimed.

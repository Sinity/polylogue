# WebUI v2 Search Vertical — Test Evidence

## Test strategy

The tests exercise the real owners touched by the implementation rather than mock-only replicas:

- shared transaction request, identity, coverage, and continuation decoding;
- production Lark expression parser and `SessionQuerySpec` lowering;
- actual daemon GET dispatch for `/app/search` and `/api/web-search`;
- real manifest-based WebUI asset discovery;
- real SQLite archive/FTS storage for distinct-session paging and batched evidence;
- generated OpenAPI and generated TypeScript client synchronization;
- strict browser response decoding;
- Preact interaction behavior in jsdom;
- complete Vite production builds.

## Production-route tests

`tests/unit/daemon/test_web_search_vertical.py` contains 18 tests.

### DSL source integrity

`test_dsl_examples_are_copied_from_parser_tests_and_round_trip` reads the declared source file and exact line for each teaching expression, then runs the production parser. It fails if an example is invented, its provenance drifts, or the parser no longer accepts it.

Representative falsification: change an expression or its line number, or remove the corresponding grammar support.

### Closed response and paging contract

`test_typed_wire_contract_rejects_shape_drift_and_incoherent_paging` exercises Pydantic response models. It rejects extra fields and a continuation flag/token mismatch.

Representative falsification: loosen models to accept arbitrary dictionaries or allow `has_more=true` with no token.

### Complete continuation replay

`test_continuation_replays_complete_request_and_rejects_overrides` builds a request with expression, repeated facets, time bucket, lane, page size, execution frame, projection, stable order, and refs. It round-trips through the shared opaque continuation and proves semantic overrides are rejected.

Representative falsification: encode only offset, accept `q` beside a continuation, or omit result-identity validation.

### Relative-time stability

`test_relative_time_is_anchored_in_complete_request` proves `since:7d` is compiled against the continuation-owned `as_of` frame rather than the wall clock of each page.

Representative falsification: call `datetime.now()` while compiling a replayed request.

### Exact versus qualified coverage

The semantic, lexical, disproven-count, and overshot-page tests exercise `_page_contract` through production types:

- semantic results use lower-bound or unknown coverage and conservative continuation;
- lexical results use exact coverage when count/page agree;
- a fetched extra row downgrades a stale exact count;
- an empty overshot page does not treat its offset as observed coverage.

Representative falsification: label all totals exact, derive a lower bound from offset alone, or drop extra-row probing.

### SSR and real daemon dispatch

`test_ssr_contains_semantic_skeleton_contract_and_safe_json` checks the no-JS form, search landmarks, result/state markup, bootstrap data, and escaping against script-breaking values.

`test_real_routes_emit_ssr_and_json_contract_fields` invokes the actual daemon dispatcher. It exercises route recognition, shell bootstrap access, manifest asset resolution, semantic HTML, API authentication policy, response status, and every client-critical JSON family.

Representative falsification: remove the route from dispatcher metadata, bypass manifest discovery, change a client-required field, or render SSR from a different shape.

### Filter-only snippets and provenance

`test_filter_only_route_projects_bounded_message_evidence` uses a real temporary archive and checks useful bounded evidence, message identity, session identity, and canonical reader anchor.

Representative falsification: return only session metadata, read messages one session at a time, omit message refs, or emit a noncanonical reader link.

### Honest degraded states

Dedicated route tests cover:

- parser diagnostics rather than a silent empty response;
- FTS probe failures without exposing archive paths;
- FTS convergence withholding rows;
- missing embeddings as a distinct state;
- unexpected execution failures contained in the typed search response.

Representative falsification: catch every failure as an empty list, leak `str(exception)` with local paths, or run lexical/semantic search despite failed readiness.

### Continuation and facets

`test_real_route_continuation_advances_complete_request_without_duplicates` stores several matching sessions, requests one row, follows the opaque token, checks identity/offset advancement, and verifies distinct sessions.

`test_selected_zero_count_facets_remain_available_to_undo` proves a selected value is projected even when its current result count is zero.

Representative falsification: page FTS blocks instead of sessions, regenerate a different query ref, or drop selected zero-count values.

## Shared transaction tests

`tests/unit/archive/query/test_transaction.py` covers:

- round-trip of complete request state;
- query identity independent of page offset;
- result identity adoption by query-unit envelopes;
- rejection of values that only satisfy annotations through Python coercion;
- malformed Base64, UTF-8, JSON, shape, and protocol tokens;
- coercive boolean/integer/float fields;
- oversized continuation tokens;
- non-advancing empty continuation pages;
- exact, lower-bound, and unknown coverage invariants.

Representative falsification: replace strict field checks with `int(value)`/`bool(value)`, remove the token cap, omit operation/projection/stable-order fields, or allow a token on a page with no rows.

## Storage tests

`test_archive_tiers_search_pages_distinct_sessions_before_applying_offset` inserts several matching blocks into one session plus another matching session. It proves search grouping happens before offset/limit and that each session appears once.

Representative falsification: restore block-level `LIMIT/OFFSET` before grouping; the first session consumes multiple positions and the test fails.

`test_archive_tiers_selects_representative_messages_for_many_sessions_in_one_projection` checks one bounded, windowed message query for a page of summaries and verifies the preferred role/fallback behavior.

Representative falsification: issue one query per session or remove the role-order window.

## Browser contract tests

`webui/src/contracts/web-search.test.ts` verifies:

- acceptance of a complete generated-server projection;
- rejection of incoherent continuation, rank, provenance, reader-link, coverage, and facet selection;
- emission of the complete transaction-shaped request;
- complete first-page parameters and cursor-only continuation parameters.

Representative falsification: remove cross-field validation, allow a mismatched session/message ref, or include expression/facets with a continuation.

## Preact island tests

`webui/src/islands/web-search.test.tsx` verifies:

- append of exactly the server continuation page without local sorting;
- parser diagnostic and corrected-example rendering;
- current rows remain visible while a facet request is pending, then are replaced by the newest server response;
- a failed continuation preserves all completed rows.

Representative falsification: sort hits in the component, mutate visible rows immediately on input, apply a stale response, or clear rows on transport failure.

## Commands and results

All commands below were run in `/mnt/data/repo140` against the final staged implementation unless a different working directory is shown.

### Complete WebUI check

```bash
cd webui
npm run check
```

Result: passed.

This command performed:

- generated design-system check;
- generated TypeScript-client check;
- design-system lint;
- strict `tsc --noEmit`;
- Vitest: 8 files, 23 tests passed;
- generated-client Node contract tests: 8 passed;
- production Vite application build;
- client design-system build;
- SSR design-system build.

Production application outputs:

- `web-search-DLJgDHLi.js`: 9.18 kB, 3.43 kB gzip;
- `web-search-CCDEqyB0.css`: 29.63 kB, 5.53 kB gzip;
- shared `api-17V4jNyH.js`: 32.10 kB, 11.62 kB gzip.

### Python production/query/security/generated/storage suite

```bash
.venv/bin/python -m pytest -q \
  tests/unit/archive/query/test_transaction.py \
  tests/unit/daemon/test_web_search_vertical.py \
  tests/unit/daemon/test_route_contracts.py \
  tests/unit/daemon/test_daemon_http_security.py \
  tests/unit/devtools/test_render_openapi.py \
  tests/unit/devtools/test_render_webui_client.py \
  tests/unit/devtools/test_render_webui_design_system.py \
  tests/unit/storage/test_archive_tiers_archive.py \
  -k 'not test_exact_session_action_count_bounds_pairing_before_global_ranking'
```

Result: `705 passed, 1 deselected in 14.94s`.

The deselected test is a pre-existing SQLite VM-step canary. On an untouched worktree at the exact synthetic baseline it fails with the same assertion as the implementation state: `assert 0 >= 50000`. It does not exercise the WebUI search changes, but it is explicitly recorded instead of being counted as green.

An attempted rerun with the global `/opt/pyvenv` interpreter failed before collection because that unrelated environment lacks Hypothesis. An attempted path using `tests/unit/daemon/test_daemon_security.py` also failed before collection because the current file is named `test_daemon_http_security.py`. Neither failed invocation executed implementation tests; the successful command above uses the snapshot’s dependency-complete `.venv` and current path.

### Python lint, formatting, and types

```bash
mapfile -t pyfiles < <(git diff --cached --name-only --diff-filter=ACMR -- '*.py')
.venv/bin/python -m ruff check "${pyfiles[@]}"
.venv/bin/python -m ruff format --check "${pyfiles[@]}"
.venv/bin/python -m mypy "${pyfiles[@]}"
```

Result:

- Ruff: all checks passed;
- formatting: 13 files already formatted;
- mypy: no issues in 13 changed Python files.

### Generated surfaces and architecture

```bash
.venv/bin/python -m devtools render all --check
.venv/bin/python -m devtools verify topology
.venv/bin/python -m devtools verify layering
.venv/bin/python -m devtools verify degrade-loudly
```

Result:

- every generated surface in `render all` synchronized;
- topology: 1,037 realized, 1,037 declared, `blocking=False`;
- nine pre-existing storage modules remain marked as unresolved topology targets;
- no layering violations;
- 71 broad exception handlers scanned, all allowlisted, zero new silent soft-fails.

### Dependency audits

```bash
cd webui
npm ci --offline --ignore-scripts
npm audit --omit=dev --audit-level=low
npm audit --audit-level=low
```

Result:

- offline lockfile installation succeeded;
- production audit: zero vulnerabilities;
- complete audit: zero vulnerabilities.

### Patch checks

```bash
git diff --cached --check
git diff --cached --binary > PATCH.diff
git apply --check PATCH.diff
```

The patch was also applied to a detached copy of the exact synthetic baseline. All 32 changed paths were compared byte-for-byte with the implementation worktree and matched. Aggregate result hash: `2c3556d520e0ac1488f6cba3bd1ab37e759cbf7c26aa7c098dad401d4daeb4ea`.

## Unverified checks

The following were not claimed:

- Playwright interaction, accessibility, or visual-regression run in a real browser;
- live operator daemon or archive;
- live FTS convergence during concurrent import;
- real Voyage embedding acquisition or an operator embeddings database;
- first-party credential lifecycle in a deployed browser;
- NixOS, wheel, or installed-package static-asset smoke;
- production latency, memory, cancellation, or large-archive benchmark;
- mutation-stable continuation under concurrent archive changes.

These require deployment/browser/operator resources absent from the supplied snapshot environment.

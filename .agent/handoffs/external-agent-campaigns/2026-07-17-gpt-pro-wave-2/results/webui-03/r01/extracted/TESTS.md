# WebUI v2 search vertical — verification record

## Testing strategy

The tests exercise production routes and shared owners rather than private stand-ins. Each high-value assertion is paired with a representative mutation that would make it fail.

| Behavior | Production dependency exercised | Anti-vacuity mutation/removal |
|---|---|---|
| DSL examples parse and retain provenance | Lark parser plus current parser test corpus | change expression or source line; parser/provenance test fails |
| Complete continuation replay | `QueryContinuation`, request identity, route parser | omit a facet/as-of/order field or accept semantic overrides; transaction/route tests fail |
| Strict token decoding | shared transaction decoder | restore `int()`/`str()` coercion or stop catching malformed Base64; decoder tests fail |
| Continuing pages make progress | `QueryResultPage` | allow `has_more=True` with zero items; non-advancing-page test fails |
| Session-level lexical paging | real FTS tables and `ArchiveStore.search_summaries` | page matching blocks before sessions; duplicate-session paging test fails |
| Exact/qualified/unknown coverage | shared coverage model and web page contract | mark semantic count exact, count an empty overshot offset as observed, or allow incoherent values; tests fail |
| Filter-only evidence batching | real archive SQL projection | return to per-session `query_session_messages`; route test monkeypatch raises |
| Canonical provenance | reader anchors and stable ref vocabulary | emit `message:<session>:<message>` or mismatched refs; Python/TS contract tests fail |
| FTS lagging and missing embeddings | existing readiness probes | silently execute/fallback; degraded-state route tests fail |
| Parser diagnostic | real route plus expression compiler | translate failure to empty 200; route test fails |
| SSR first page | real `/search` handler and renderer | remove form/results/initial JSON/asset reference or fail escaping; SSR test fails |
| Server-authoritative facets | Preact component against typed responses | mutate visible rows before response; component test fails |
| Lossless append order | continuation-only request and response append | add local sort/dedup/filter or reconstruct query parameters; component test fails |
| Newest response wins | AbortController/request serial | allow stale response to commit; out-of-order test fails |
| Browser history restoration | `/search` to `/api/search` mapping and `popstate` | replay local state or push duplicate entry; component test fails |
| Runtime browser decoder | TypeScript boundary guard | accept malformed page/ref/time/page-size shape; decoder tests fail |
| Route registry/security | production route catalog and auth parameterization | omit route declaration/auth metadata; route/security suites fail |
| Generated declarations | OpenAPI/topology renderers | leave generated drift or undeclared new modules; render/verify fails |

Fixtures are synthetic and sanitized. No operator archive, credentials, or live daemon was used.

## Executed Python tests

### Route, transaction, route-contract, and security suite

```bash
PYTHONPATH=$PWD /tmp/polylogue-testvenv/bin/python -m pytest -q -p no:randomly \
  tests/unit/archive/query/test_transaction.py \
  tests/unit/daemon/test_web_search_vertical.py \
  tests/unit/daemon/test_route_contracts.py \
  tests/unit/daemon/test_daemon_http_security.py
```

Result: **654 passed in 4.34 seconds**.

This includes real `DaemonAPIHandler.do_GET()` coverage for `/search` and `/api/search`, JSON fields consumed by the island, no-JS semantic HTML, parse error, FTS convergence, missing embeddings, filter-only evidence, and two-page distinct-session continuation.

### Date execution-frame regression suite

```bash
PYTHONPATH=$PWD /tmp/polylogue-testvenv/bin/python -m pytest -q -p no:randomly \
  tests/unit/core/test_dates.py \
  tests/unit/core/test_timestamp_guards.py
```

Result: **101 passed in 5.66 seconds**.

### Archive-tier suite excluding one reproduced baseline canary

```bash
PYTHONPATH=$PWD /tmp/polylogue-testvenv/bin/python -m pytest -q -p no:randomly \
  tests/unit/storage/test_archive_tiers_archive.py \
  -k 'not exact_session_action_count_bounds_pairing_before_global_ranking'
```

Result: **33 passed, 1 deselected in 9.97 seconds**.

The deselected VM-step assertion was executed separately in both worktrees:

```text
implementation: assert 0 >= 50000 — failed in 1.24s
clean snapshot: assert 0 >= 50000 — failed in 1.29s
```

The same assertion and value fail at the untouched base commit, so it is an environment-sensitive baseline canary rather than a regression hidden by this patch.

### Broader query/search execution suite

```bash
PYTHONPATH=$PWD /tmp/polylogue-testvenv/bin/python -m pytest -q -p no:randomly \
  tests/unit/archive/query/test_evaluator.py \
  tests/unit/archive/query/test_production_evaluator.py \
  tests/unit/archive/query/test_read_surface_control.py \
  tests/unit/storage/test_archive_search_contracts.py \
  tests/unit/storage/test_archive_tiers_search_guard.py \
  tests/unit/storage/test_search_explanation_wiring.py \
  tests/unit/storage/test_search_misc.py \
  tests/unit/storage/test_search_text_write_tool_coverage.py \
  tests/unit/storage/test_search_timeless_since_filter.py
```

Result: **86 passed in 5.60 seconds**.

One earlier combined shell invocation reached its outer command timeout during this group after the preceding suites; rerunning the identical group independently produced the result above.

## Python static checks

Changed Python source and tests:

```bash
python -m ruff check <11 changed Python files>
python -m ruff format --check <11 changed Python files>
python -m mypy <11 changed Python files>
```

Results:

- Ruff lint: **passed**.
- Ruff format: **11 files already formatted**.
- Targeted mypy: **no issues in 11 files**.

The files were the shared transaction/date/storage owners, daemon route/search modules, and their changed tests.

## Browser unit/type/build checks

```bash
cd webui
npm run typecheck
npm run test:unit
npm ci --ignore-scripts --dry-run
npm audit --omit=dev --audit-level=low
npm audit --audit-level=low
```

Results:

- TypeScript strict compilation: **passed**.
- Vitest: **3 files, 19 tests passed**.
  - query request builders: 6;
  - runtime response contract: 5;
  - Preact search island: 8.
- Lockfile dry run: **up to date**.
- Production-only audit: **0 vulnerabilities**.
- Complete audit: **0 vulnerabilities**.

A temporary, uncommitted Vite library config built the real production entry:

```text
entry: webui/src/verticals/search/entry.tsx
output: web-search.js
modules transformed: 8
bundle: 31,590 bytes
level-9 gzip: 10,229 bytes
source map: 75,198 bytes
SHA-256: 67c5994a314ad24c46928b80ca4605d75fdf6ba540b941e7ac267ce754abb402
```

The built JavaScript contains no URL for unpkg, jsDelivr, cdnjs, esm.sh, or Skypack. Standards namespace URLs embedded by the DOM runtime are not external dependencies.

## Generated and architecture checks

```bash
python -m devtools render openapi --check
python -m devtools render topology-status --check
python -m devtools verify topology
python -m devtools verify layering
python -m devtools verify degrade-loudly
python -m devtools render all --check
```

Results:

- OpenAPI synchronized.
- Topology: **1,026 realized / 1,026 declared**, blocking false.
- Nine pre-existing storage modules remain marked TBD; no new topology mismatch.
- Layering: **no violations**.
- Degrade-loudly: **71 broad handlers scanned, all allowlisted, zero new silent soft-fails**.
- Aggregate generated-surface check: **passed**; site sources and generated local links resolve.

## Patch and package checks

The final patch was emitted with:

```bash
git diff --binary --full-index > PATCH.diff
git diff --check
```

Validation against a fresh detached checkout at the named commit:

```text
git apply --check: passed
git apply: passed
changed files compared byte-for-byte: 26
aggregate file hash: b69d1daee4a19fb0cdc9be771b45067c48fe6b7da03040dfa9a7b6c25e4f8c59
PATCH.diff: 224,495 bytes; 6,839 lines
```

The patch contains no copied input archive and no implementation placeholders. The only literal `placeholder` occurrences are the intended HTML/JS query-input attribute and a local SQL bind-variable name. Ellipses are Python protocol stubs/sentinel values, TypeScript spread syntax, and the FTS snippet delimiter—not unfinished code.

ZIP validation is recorded in the final chat response after reopening the archive.

## Not executed or not claimed

- full repository pytest suite;
- operator archive or production-scale corpus;
- live daemon or browser credential flow;
- Playwright interaction/accessibility/visual tests;
- committed asset reproduction through `devtools render webui` (owner absent);
- Nix, wheel, or deployed static-asset inclusion;
- mutation-concurrent continuation proof;
- one-snapshot page/count/facet coherence.

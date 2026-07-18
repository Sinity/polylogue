# WebUI v2 search vertical — implementation handoff

## Delivery status

This package is an apply-ready, cohesive implementation of the WebUI v2 search vertical. It adds semantic first-page HTML, a typed daemon JSON projection, parser-proven DSL examples, server-computed facets, snippets and provenance, qualified coverage, opaque continuation replay, and a Preact island that enhances paging and facets without acquiring query semantics.

This revision also refines the first implementation rather than merely repackaging it. The search adapter is split into immutable wire models, canonical execution, and semantic HTML rendering. Shared lexical paging now operates on distinct sessions before applying `LIMIT/OFFSET`; filter-only evidence is selected for the whole page in one windowed SQL read; canonical message refs are enforced; browser history is restored through the server; and both Python and TypeScript boundaries reject incoherent continuations and projections.

The snapshot does not contain the WebUI v2 production asset/theme/generated-client scaffold owned by `polylogue-bby.11`. This patch therefore defines the narrow scaffold interface it needs and does not build a competing scaffold.

## Snapshot identity

| Field | Value |
|---|---|
| Snapshot ref | `master` |
| Extracted worktree state | detached HEAD reachable from `master` and `origin/master` |
| Commit | `536a53efac0cbe4a2473ad379e4db49ef3fce74d` |
| Subject | `fix(repair): harden raw authority convergence (#3046)` |
| Commit time | `2026-07-17T18:55:47+02:00` |
| Patch base | exactly the commit above |
| Patch shape | 26 files; 6,385 insertions and 83 deletions |
| Patch size | 224,495 bytes; 6,839 lines |
| Clean-apply aggregate | `b69d1daee4a19fb0cdc9be771b45067c48fe6b7da03040dfa9a7b6c25e4f8c59` |

The supplied project-state archive was used only as source authority. It is not present in this package.

## Mechanism and ownership

### One logical request

Both `GET /search` and `GET /api/search` lower to `QueryTransactionRequest`. A new request owns:

```json
{
  "operation": "query",
  "arguments": {
    "expression": "repo:polylogue since:7d has:paste",
    "facets": {
      "origin": ["codex-session"],
      "repo": ["polylogue"],
      "time": "30d"
    },
    "retrieval_lane": "dialogue",
    "as_of": "2026-07-17T12:00:00Z"
  },
  "page_size": 20,
  "offset": 0,
  "projection": "web-search.v1",
  "stable_order": "rank-score-session-id"
}
```

`as_of` is part of request identity. Relative expressions such as `since:7d` and the time facet therefore retain the same execution frame on every continuation page.

The server computes `query_ref` and `result_ref`. The continuation is the shared opaque `q1` envelope and carries the complete advancing request. A continuation request must be replayed alone; semantic overrides are rejected. The decoder rejects malformed Base64/JSON, coercive booleans or floats, empty protocol identifiers, wrong operation/projection/order, excessive page size, and query/result identity mismatch.

### Canonical semantics

The vertical does not parse or implement query meaning independently. It composes the existing owners:

- Lark expression parser/lowerer: `polylogue/archive/query/expression.py`.
- Immutable selection request: `SessionQuerySpec` and its canonical plan.
- Ranked execution: `Polylogue.search_session_hits`.
- Filter-only selection and count: `SessionQuerySpec.list_summaries` and `.count`.
- Facets and miss diagnostics: `Polylogue.facets` and `Polylogue.diagnose_query_miss`.
- Read admission/cancellation identity: `polylogue/archive/query/transaction.py`.
- FTS and embedding readiness: existing daemon readiness probes.
- Reader anchors: the existing reader anchor helper and canonical `session:<id>` / `message:<id>` refs.

The TypeScript island sends requests and renders returned state. It never parses the DSL, filters, ranks, computes facets, invents totals, deduplicates hits, or reconstructs cursor state.

### Three explicit server layers

- `polylogue/daemon/web_search_contract.py` owns immutable, closed Pydantic wire models and protocol constants.
- `polylogue/daemon/web_search.py` owns transport decoding, canonical lowering/execution, readiness decisions, and projection construction.
- `polylogue/daemon/web_search_page.py` owns escaped semantic HTML for no-JavaScript first-page operation.

JSON and HTML consume the same `WebSearchResponse`; there is no second dictionary-shaped SSR contract.

### Session-level lexical paging

The original `ArchiveStore.search_summaries` counted distinct sessions but applied `LIMIT/OFFSET` to matching blocks. A session with two matching messages could therefore occupy two rows and displace another session. The revised query:

1. groups FTS matches by session and computes the best rank;
2. pages distinct sessions in deterministic order;
3. selects one best evidence block for each paged session;
4. returns page ranks beginning at the logical offset.

This aligns row cardinality, exact counts, coverage, and continuation offsets on a static archive.

### Bounded filter-only evidence

A filter-only result still needs a useful snippet and message anchor. `ArchiveStore.query_session_message_evidence` performs one bounded windowed query for the complete returned page. It selects one representative normalized message per session, preferring user/assistant roles and falling back to the first message. It avoids per-session reads and returns only message id plus bounded text evidence.

### Coverage

The shared `QueryCoverage` vocabulary is:

- `exact(n)`: the current lexical/filter count is proven for the controlled reads;
- `at_least(n)`: directly observed rows or a bounded ranked provider prove only a lower bound;
- `unknown`: no honest integer is available.

An offset contributes to an observed lower bound only when the returned page contains evidence. An empty overshot or altered continuation cannot manufacture an inflated `at_least` total. A page marked as continuing must contain at least one item, preventing non-advancing cursors.

Semantic/hybrid pages remain qualified because current ranked providers can be bounded. The UI renders the qualifier verbatim.

## Browser JSON contract

Every response contains exactly these top-level fields:

```text
schema
  generated_at
  request
  state
  coverage
  hits
  facets
  page
  resolved_retrieval_lane
  diagnostics
  examples
```

Important invariants enforced in Python and revalidated before hydration in TypeScript:

- schema/projection is `web-search.v1`;
- stable order is `rank-score-session-id`;
- page size is an integer from 1 through 100;
- `as_of`, `generated_at`, and hit timestamps are timezone-aware ISO-8601 values;
- origin/repository facet arguments are non-empty and unique;
- time is `24h`, `7d`, `30d`, `all`, or null as appropriate;
- exact/qualified coverage has a non-negative integer; unknown coverage has null;
- ranks are positive, unique, and increasing;
- `session_ref` equals `session:<session_id>`;
- `message_ref` is present exactly with `message_id` and equals `message:<message_id>`;
- reader links begin with `/s/`;
- `has_more` is true exactly when a `q1` continuation is present;
- an empty state cannot carry hits;
- stated coverage cannot be smaller than an observed non-empty page.

### States and HTTP behavior

| Condition | HTTP | State | Result rows |
|---|---:|---|---|
| Ready or no matches | 200 | `ready` / `empty` | complete page / none |
| Parser or query compilation failure | 400 | `parse_error` | none; diagnostic and corrected parser-proven example |
| Invalid request/continuation | 400 | `invalid_request` | none |
| FTS exists but is converging | 409 | `fts_lagging` | withheld rather than presented as empty |
| Embeddings absent/unusable | 409 | `embeddings_absent` | none; no lexical masquerade |
| FTS missing/probe unavailable | 503 | `fts_unavailable` | none |
| Archive tier unavailable | 503 | `archive_unavailable` | none |
| Controlled timeout/storage failure | 503 | `failed` | no partial page |
| Unexpected implementation failure | 500 | `failed` | sanitized typed response |

Filesystem paths and exception text remain in daemon logs; they are not projected into the browser contract.

## SSR and island behavior

`GET /search` renders a complete HTML document with query form, lane selector, origin/repository/time facets, state notice, qualified total, result rows, snippets, provenance, reader deep links, and an ordinary continuation link. The first page remains readable and navigable without JavaScript.

The island then provides:

- same-origin credential bootstrap through the scaffold hooks;
- server-authoritative facet/query replacement;
- opaque continuation append in exact server order;
- query/result identity validation before append;
- cancellation and newest-response-wins for rapid requests;
- preservation of the completed page when a later continuation fails;
- browser history push for new logical searches;
- `popstate` restoration by re-fetching `/api/search`, without pushing a duplicate entry.

The production bundle has no CDN/package-host URL and is built from the locked local Preact/Vite dependencies.

## DSL teaching corpus

Each example is copied from the current parser test at the declared source line and round-tripped through the real parser:

| Expression | Provenance |
|---|---|
| `repo:polylogue` | `tests/unit/cli/test_query_expression.py:5345` |
| `repo:polylogue since:7d has:paste` | `tests/unit/cli/test_query_expression.py:5584` |
| `exists message(role:assistant AND text:timeout)` | `tests/unit/cli/test_query_expression.py:1660` |
| `near:"semantic search test"` | `tests/unit/cli/test_query_expression.py:5433` |

The browser offers only the canonical retrieval lanes `auto`, `dialogue`, `actions`, and `hybrid`. Pure semantic intent is taught through `near:` rather than an unsupported client-only lane.

## Shared browser utilities

`webui/src/lib/queryRequest.ts` provides reusable transport helpers:

- canonical projection/order/lane/time/page-size constants;
- runtime lane, time-bucket, and timestamp guards;
- a transport-neutral `buildQueryTransactionRequest` shape compatible with the shared query transaction and future generated HTTP/MCP clients;
- repeated-facet URL construction;
- continuation-only URL construction;
- SSR form normalization;
- symmetric `/search` ↔ `/api/search` history mapping.

## Changed files

### Shared substrate

- `polylogue/archive/query/transaction.py`
- `polylogue/core/dates.py`
- `polylogue/storage/sqlite/archive_tiers/archive.py`

### Daemon surface

- `polylogue/daemon/http.py`
- `polylogue/daemon/route_contracts.py`
- `polylogue/daemon/web_search.py`
- `polylogue/daemon/web_search_contract.py`
- `polylogue/daemon/web_search_page.py`

### WebUI source and locked toolchain

- `webui/package.json`
- `webui/package-lock.json`
- `webui/tsconfig.json`
- `webui/vitest.config.ts`
- `webui/src/test-setup.ts`
- `webui/src/lib/queryRequest.ts`
- `webui/src/lib/queryRequest.test.ts`
- `webui/src/verticals/search/types.ts`
- `webui/src/verticals/search/types.test.ts`
- `webui/src/verticals/search/SearchIsland.tsx`
- `webui/src/verticals/search/entry.tsx`
- `webui/src/verticals/search/__tests__/SearchIsland.test.tsx`

### Tests and generated declarations

- `tests/unit/archive/query/test_transaction.py`
- `tests/unit/storage/test_archive_tiers_archive.py`
- `tests/unit/daemon/test_web_search_vertical.py`
- `docs/openapi/search.yaml`
- `docs/plans/topology-target.yaml`
- `docs/topology-status.md`

## Minimal scaffold interface

The absent canonical scaffold must:

1. build `webui/src/verticals/search/entry.tsx` as `/assets/web-search.js` with local Preact/Vite dependencies;
2. serve the committed reproducible asset through the canonical daemon static owner;
3. expose the existing first-party `window.ensureWebCredential` and `window.bootstrapWebCredential` hooks before protected API reads;
4. replace the deliberately self-contained fallback SSR CSS with generated design tokens when the token owner lands;
5. generate shared browser types/client calls from the route-registry-derived OpenAPI without changing the JSON shape documented here;
6. add Playwright interaction, accessibility, history, credential, degraded-state, and visual coverage against the live daemon/demo archive.

No built asset is committed here because the owning `devtools render webui` path and static distribution directory are absent from the snapshot.

## Legacy shell deletion candidates

Delete only after independent parity certification:

- search/facet markup and CSS around the current shell search box and facet bar in `polylogue/daemon/web_shell.py`;
- shell-local query/origin/offset state and `loadSessions()` search orchestration;
- shell-local `loadFacets()` orchestration;
- sidebar result/facet renderers, debounce, facet-click, and offset previous/next handlers;
- search-box augmentation in `polylogue/daemon/web_shell_reader.py` once reader integration is owned by v2.

Do not delete `web_shell_reader.py` wholesale. Do not delete general `/api/sessions` or `/api/facets` routes merely because this vertical adds a composed `/api/search` projection.

## Apply order

1. Check out `536a53efac0cbe4a2473ad379e4db49ef3fce74d` with no local changes.
2. Run `git apply --check PATCH.diff` from the repository root.
3. Apply with `git apply PATCH.diff`.
4. Install the locked WebUI dependencies through the repository’s normal Node workflow.
5. Run the commands and compare the results in `TESTS.md`.
6. Integrate the entry point through the canonical WebUI scaffold seam above.
7. Run the unverified live browser, packaging, and deployment checks before deleting legacy shell behavior.

The packaged patch was checked and applied to a fresh detached worktree at the named commit. All 26 resulting files, including generated files represented as Git binary deltas, matched the implementation worktree byte-for-byte.

## Acceptance matrix

| Requirement | Result |
|---|---|
| Parser-valid progressive examples | implemented and source-line proven |
| Bounded snippets and provenance | implemented for ranked and filter-only results |
| Reader message anchors | implemented with canonical refs and `/s/<id>#<anchor>` links |
| Server-computed origin/time/repo facets | implemented; selected zero-count options remain undoable |
| Continuation only | implemented; no load-all mode and no client cursor reconstruction |
| Exact versus qualified totals | implemented through shared `QueryCoverage` |
| Distinct FTS convergence/absence, embedding absence, parse error | implemented as typed states |
| First-page SSR | implemented and route-tested |
| Preact paging/facet islands | implemented and component-tested |
| No client filtering/ranking | enforced by design and tests |
| HTTP/MCP request-shape convergence seam | implemented as transport-neutral builder |
| Zero CDN; synthetic fixtures | implemented |
| Static-archive session-lossless paging | implemented, including duplicate-block regression |
| Mutation-stable every-row-exactly-once paging | not complete; remains shared transaction work |
| Live scaffold/browser/package integration | unverified because scaffold/deployment are absent |

## Important limitations and risks

### Offset continuation is not mutation-stable

The current shared `q1` token is complete and identity-checked but advances by offset. On a static archive, the corrected session-level lexical query yields each logical session once. Concurrent insertion, deletion, edit, FTS rebuild, or reranking can still duplicate or skip rows. Keyset/snapshot/spool behavior belongs to open Bead `polylogue-z9gh.9.1`; a private WebUI cursor would violate the single query-owner rule.

### Reads are controlled but not one archive snapshot

Page retrieval, count, origin/repository facets, four time counts, and filter-only evidence are separate bounded reads. Concurrent writes can briefly skew page/count/facet values. Coverage downgrades when returned evidence disproves a separately read count, but the vertical cannot create transaction-wide snapshot coherence.

### Ranked provider completeness remains qualified

Semantic/hybrid providers may be bounded. The vertical therefore does not claim exact totals or terminality beyond available evidence. Exhaustive ranked continuation remains shared query work.

### Facet cost

Time facets currently perform four canonical count reads. This is bounded and correct for the current contract, but a shared snapshot-aware multi-bucket facet projection would reduce repeated work at larger scale.

### Scaffold and deployment

Asset serving, generated design tokens/client, live credential timing, Playwright, accessibility/visual regression, Nix/wheel inclusion, and deployed-daemon behavior were not available and are not claimed.

## Value of another iteration

No packaging repair remains once this ZIP validates. A small follow-up could only polish naming or reduce the four time-count reads. A substantial second pass is valuable only if it lands shared mutation-stable continuation/snapshot semantics, the canonical WebUI asset/client scaffold, and live browser/deployment proof. Those are cross-cutting owners rather than further leaf-adapter elegance work.

# WebUI v2 Search Vertical — Implementation Handoff

## Delivery status

This package is a cohesive implementation of the WebUI v2 search vertical against the supplied Polylogue project-state snapshot. It replaces the previous revision rather than layering a supplemental patch on top of it.

The implementation provides:

- semantic SSR at `GET /app/search`;
- the matching typed JSON projection at `GET /api/web-search`;
- parser-backed DSL teaching examples;
- bounded snippets and canonical provenance links;
- server-computed origin, repository, and time facets;
- opaque continuation-only paging;
- exact, lower-bound, and unknown coverage labels;
- distinct parser, FTS, embeddings, archive, and unexpected-failure states;
- Preact hydration that never filters or ranks locally;
- generated OpenAPI and TypeScript-client integration;
- packaged, manifest-resolved Vite assets using the existing WebUI v2 design system and credential bootstrap.

## Snapshot identity and patch base

The attached source authority resolves to:

| Item | Value |
|---|---|
| Upstream commit | `bf8191b3f56aa40da8f271df7f3385c712825497` |
| Upstream subject | `feat: land WebUI v2 scaffold, design system, and generated client (#3074)` |
| Upstream branch containment | `master`, `origin/master` |
| Upstream parent | `4b574ce66533ac114961a9533ba2f2c4a0c45b83` |
| Supplied archive | `/mnt/data/polylogue-all.tar(140).gz` |
| Supplied dirty files | `polylogue/archive/query/unit_results.py`, `polylogue/daemon/http.py`, `polylogue/hooks/__init__.py` |
| Supplied dirty-patch SHA-256 | `f30102453e3576c9049eb65dcea7265fa99ac694c45531b06b0a34dd85279e9f` |
| Synthetic exact-snapshot baseline | `be2c26f90e3cfdba3a2015cb045ccf7f42cf7ce8` |

`PATCH.diff` applies to the exact supplied state, represented by the synthetic baseline commit. Applying it directly to a clean checkout of the upstream commit is not sufficient because the supplied archive already contained three uncommitted changes. To recreate the exact base from upstream, apply the supplied dirty patch first, then apply this package’s `PATCH.diff`.

The implementation patch contains 32 changed paths, 4,276 insertions, and 65 deletions. Its SHA-256 is `ddc49958fba929a8c68e00c663b15fc7be894a2a82f5345e15524b69c7f71bb5`.

## Evidence inspected

The implementation was designed from current source rather than the earlier package’s assumptions. The following authority was inspected before and during the port:

- repository instructions and architecture: `AGENTS.md`, `polylogue-overview.md`, `docs/architecture-spine.md`, `docs/search.md`;
- query semantics: `polylogue/archive/query/expression.py`, query parser tests, `SessionQuerySpec`, plans, execution controls, search envelopes, FTS lifecycle/readiness, semantic-search paths, and the shared transaction module;
- storage: `polylogue/storage/sqlite/archive_tiers/archive.py`, FTS tables, summary listing/counting, message projections, and representative evidence selection;
- daemon: `polylogue/daemon/http.py`, `route_contracts.py`, `webui.py`, current SSR route handling, credential bootstrap, static asset serving, CSP, and legacy `web_shell_*` modules;
- WebUI v2: generated client/runtime, design-system components and tokens, Vite entrypoint ownership, manifest contract, existing archive-overview island, test setup, and client contract tests;
- generated surfaces: `devtools/render_openapi.py`, `docs/openapi/search.yaml`, topology projection/status, and design-system/client generators;
- Beads: `polylogue-z9gh.9.1`, `polylogue-4p1`, and `polylogue-t46.8` in full from `.beads/issues.jsonl`;
- current history around `bf8191b3…`, especially the landed WebUI v2 scaffold commit.

The decisive source change from the prior package is that the canonical WebUI v2 scaffold now exists. This revision therefore removes the previous “minimal scaffold interface” assumption and integrates with the real generated client, auth bootstrap, design system, manifest, and Vite build.

## Architecture and mechanism

### One semantic owner

The query DSL remains server-owned. The browser emits form intent or replays an opaque continuation; it does not compile the expression, apply facets, rank rows, estimate totals, or infer degraded readiness.

The server flow is:

1. Parse transport parameters or decode a `q1` continuation.
2. Validate the complete `QueryTransactionRequest` and its logical/result identity.
3. Lower the request through the real Lark query parser into the existing `SessionQuerySpec` path.
4. Inspect required FTS and embedding readiness before reading rows.
5. Execute canonical lexical, semantic, hybrid, or filter-only queries.
6. Project bounded result rows, snippets, provenance, reader links, facets, and qualified coverage.
7. Serialize one immutable `WebSearchResponse` used by both JSON and semantic SSR.

### Server modules

`polylogue/daemon/web_search_contract.py` owns the closed, immutable Pydantic wire models and invariants. Unknown fields are rejected. It defines the request projection, states, coverage, facets, hits, provenance, continuation page, examples, and complete response.

`polylogue/daemon/web_search.py` owns transport decoding, continuation replay, relative-time anchoring, request lowering, readiness decisions, execution, facets, evidence projection, coverage, and error containment.

`polylogue/daemon/web_search_page.py` renders semantic HTML from the same typed `WebSearchResponse` used by `/api/web-search`. SSR is not assembled from an independent dictionary contract.

`polylogue/archive/query/transaction.py` now provides strict transaction request/continuation/coverage primitives shared beyond this vertical. It rejects coercive or malformed continuation payloads, preserves complete request state, supplies stable query/result references, and prevents a non-advancing empty continuation page.

### Browser modules

`webui/src/contracts/web-search.ts` supplies:

- the complete transport-neutral `SearchQueryTransactionRequest` shape intended to converge with the protocol-native MCP `query` verb;
- initial-parameter and continuation-only builders;
- form extraction and URL conversion;
- a strict runtime decoder for the complete `WebSearchResponse`;
- cross-field validation for coverage, ranks, facets, provenance, reader links, states, request identity, and continuation coherence.

`webui/src/islands/web-search.tsx` renders server order verbatim. It replaces rows only after an authoritative facet/search response, appends only a validated continuation page, aborts superseded requests, ignores stale responses, preserves completed rows after a continuation failure, and re-fetches server state during browser history restoration.

`webui/src/lib/api.ts` uses the generated `PolylogueClient.projectWebSearch` method through the first-party credential bootstrap. Typed degraded/error responses are parsed from daemon error payloads rather than collapsed into generic transport failures.

`webui/src/entrypoints/web-search.tsx` hydrates the SSR island. `webui/src/web-search.css` composes the existing generated design-system tokens and components rather than creating a second theme system.

## Routes and authentication

| Route | Role | Authentication | Contract |
|---|---|---|---|
| `GET /app/search` | semantic SSR shell and no-JS search form | loopback shell bootstrap | HTML rendered from `WebSearchResponse` |
| `GET /api/web-search` | authoritative query projection | credential if configured | JSON `WebSearchResponse` |
| `GET /app/assets/:name` | hashed Vite assets named by manifest | loopback shell bootstrap | JS/CSS bytes |
| `POST /api/web-auth/session` | first-party credential bootstrap | WebUI bootstrap policy | credential session |

The page’s script and stylesheet names are resolved from the packaged Vite manifest. The CSP permits forms only to the same origin, retaining no-JS search without broadening script, style, frame, or external-network policy.

## Exact request contract

An initial browser request uses normal query parameters:

- `q`: DSL expression;
- repeated `origin` values;
- repeated `repo` values;
- `time`: `24h`, `7d`, `30d`, or `all`;
- `retrieval_lane`: `auto`, `dialogue`, `actions`, or `hybrid`;
- `page_size`: 1 through 100.

The daemon normalizes these into the complete logical request:

```json
{
  "operation": "query",
  "arguments": {
    "expression": "repo:polylogue since:7d has:paste",
    "facets": {
      "origin": ["codex-session"],
      "repo": ["polylogue"],
      "time": "7d"
    },
    "retrieval_lane": "auto",
    "as_of": "2026-07-18T09:00:00Z"
  },
  "page_size": 20,
  "offset": 0,
  "projection": "web-search.v1",
  "stable_order": "rank-score-session-id"
}
```

The response echoes this request with `query_ref` and `result_ref`. `as_of` is request-owned so relative dates such as `since:7d` retain one execution frame across continuation requests.

A continuation request contains exactly one semantic parameter:

```text
GET /api/web-search?continuation=q1.…
```

Supplying `q`, facets, lane, page size, or other semantic overrides beside a continuation is rejected. The continuation embeds the complete request plus the expected result identity and is validated before execution.

## Exact response fields

The top-level `WebSearchResponse` is:

```json
{
  "schema": "web-search.v1",
  "generated_at": "timezone-aware ISO-8601 timestamp",
  "request": {
    "operation": "query",
    "arguments": {
      "expression": "string",
      "facets": {
        "origin": ["string"],
        "repo": ["string"],
        "time": "24h | 7d | 30d | all | null"
      },
      "retrieval_lane": "auto | dialogue | actions | hybrid",
      "as_of": "timezone-aware ISO-8601 timestamp"
    },
    "page_size": 20,
    "offset": 0,
    "projection": "web-search.v1",
    "stable_order": "rank-score-session-id",
    "query_ref": "query:<24 lowercase hex>",
    "result_ref": "result:<24 lowercase hex>"
  },
  "state": {
    "kind": "ready | empty | parse_error | invalid_request | archive_unavailable | fts_lagging | fts_unavailable | embeddings_absent | failed",
    "message": "string",
    "detail": "string or null",
    "field": "string or null",
    "corrected_example": "string or null",
    "stale_available": false
  },
  "coverage": {
    "kind": "exact | at_least | unknown",
    "value": "non-negative integer or null"
  },
  "hits": [],
  "facets": {
    "origin": [],
    "repo": [],
    "time": []
  },
  "page": {
    "has_more": false,
    "continuation": "q1.… or null"
  },
  "resolved_retrieval_lane": "string",
  "diagnostics": "object or null",
  "examples": []
}
```

Each hit contains:

- positive monotonic `rank`;
- non-empty `title`;
- bounded `snippet` or null;
- actual `retrieval_lane`;
- timezone-aware `occurred_at` or null;
- `repo` or null;
- canonical `/s/<percent-encoded-session>#<percent-encoded-anchor>` reader link;
- provenance with `origin`, raw IDs, canonical `session:<id>` and optional `message:<id>` refs, and the message/session anchor.

Facet entries contain `value`, non-negative `count`, and `selected`; time entries also contain a display `label`. A selected facet remains represented even when its server-computed count is zero, making the empty selection reversible.

## Coverage and continuation semantics

Lexical and filter-only queries can report exact coverage when the archive count and fetched page agree. If a fetched extra row disproves a separately read count, the response downgrades to `at_least` rather than presenting a false exact total.

Semantic and hybrid providers can be bounded. Their coverage is therefore `at_least` when a lower bound exists and `unknown` when no positive lower bound is available. Full semantic pages conservatively retain continuation because a bounded provider cannot prove exhaustion merely from a local count.

The current shared `q1` mechanism advances by offset. It is complete and deterministic on an unchanged archive, and the storage fix ensures lexical rows are paged by distinct session rather than matching block. It does not yet provide mutation-stable exactly-once enumeration under concurrent insertion, deletion, editing, FTS rebuild, or reranking. Keyset, snapshot, or owned-spool continuation remains shared transaction work under `polylogue-z9gh.9.1`; the WebUI does not invent a private cursor.

## Search-row and storage corrections

`ArchiveStore.search_summaries` previously counted distinct sessions while paging matching blocks. A session with multiple matching messages could therefore consume several result positions, contradict the count, and duplicate across continuation pages.

The patch now:

1. groups matches by session;
2. computes one best rank per session;
3. applies stable ordering and offset/limit to distinct sessions;
4. chooses one best evidence block for each paged session.

The result unit is consistently “session” across exact count, rank, offset, and page cardinality.

Filter-only searches now fetch representative message evidence for an entire page in one bounded windowed query. They prefer user or assistant messages and otherwise select the first normalized message. This avoids an N+1 message-read pattern while still supplying a useful snippet and message anchor.

## DSL teaching corpus

Every expression is copied from the parser-test corpus at the declared line and is re-parsed by the production parser in tests:

| Expression | Provenance |
|---|---|
| `repo:polylogue` | `tests/unit/cli/test_query_expression.py:5345` |
| `repo:polylogue since:7d has:paste` | `tests/unit/cli/test_query_expression.py:5584` |
| `exists message(role:assistant AND text:timeout)` | `tests/unit/cli/test_query_expression.py:1660` |
| `near:"semantic search test"` | `tests/unit/cli/test_query_expression.py:5433` |

The parser-error state carries the actual parser/compiler diagnostic, the implicated field where available, and a known-valid corrected example. It never silently renders as a zero-result state.

## Degraded-state matrix

| Condition | State kind | HTTP status | Rows | User-visible behavior |
|---|---|---:|---:|---|
| Successful non-empty query | `ready` | 200 | yes | Hits, facets, qualified total, continuation when available |
| Successful query with no matches | `empty` | 200 | no | Explicit empty-state message and usable facets/examples |
| DSL compile/parse failure | `parse_error` | 400 | no | Parser diagnostic, field when known, corrected valid example |
| Malformed request or continuation | `invalid_request` | 400 | no | Typed contract error; no generic daemon body |
| Archive root/config unavailable | `archive_unavailable` | 503 | no | Distinct archive availability state |
| FTS exists but is not converged | `fts_lagging` | 409 | no | Rows withheld to avoid partial lexical truth |
| FTS cannot be used | `fts_unavailable` | 503 | no | Distinct index-unavailable state |
| Semantic/hybrid request without embeddings | `embeddings_absent` | 409 | no | Distinct semantic capability state |
| Unexpected execution failure | `failed` | 500 or 503 | no | Typed generic failure without sensitive exception paths |

## Generated and packaged WebUI integration

`devtools/render_openapi.py` generates the search operation and Pydantic-derived response schemas into `docs/openapi/search.yaml`. The WebUI client generator then owns `ProjectWebSearchParameters`, `WebSearchResponse`, and `PolylogueClient.projectWebSearch` in `webui/src/api/generated.ts`.

The Vite build emits and commits:

| Asset | Bytes | SHA-256 |
|---|---:|---|
| `api-17V4jNyH.js` | 32,102 | `a1dd7f1da9f6726f08ffb9c5110fb7693c1c5532f001e3de8c0efd3010ef71f3` |
| `archive-overview-B-5sp0Wj.js` | 2,478 | `68ac5d284d7ee2fc5a83d2bb263920db867dfe820bef5346e7693675034ed46e` |
| `archive-overview-Ce5VissO.css` | 3,372 | `6ccc25451ac6466539ab63006bcf519c9acf2e702b0ba0fbd4f18f10f0dae627` |
| `web-search-DLJgDHLi.js` | 9,178 | `85aa992ae8a91de652a998e0d3c055e923c22075765b70935d16319b13c589ba` |
| `web-search-CCDEqyB0.css` | 29,633 | `107d5e640b008b22baa38cf95a6ddeba53a6249f59404ace6125d1b369777b5d` |
| `manifest.json` | 656 | `9a06edfa473e74771e74416f31f91c93714499f0bb3e99bb14707bef069a1a47` |

The shared API chunk changed because the generated client gained the search operation; the archive-overview entry hash consequently changed even though the overview source did not acquire search semantics.

## Changed files by ownership

### Shared query and date substrate

- `polylogue/archive/query/transaction.py`
- `polylogue/core/dates.py`
- `tests/unit/archive/query/test_transaction.py`

### Search projection and daemon routes

- `polylogue/daemon/web_search_contract.py`
- `polylogue/daemon/web_search.py`
- `polylogue/daemon/web_search_page.py`
- `polylogue/daemon/http.py`
- `polylogue/daemon/route_contracts.py`
- `polylogue/daemon/webui.py`
- `tests/unit/daemon/test_web_search_vertical.py`

### Storage semantics

- `polylogue/storage/sqlite/archive_tiers/archive.py`
- `tests/unit/storage/test_archive_tiers_archive.py`

### WebUI source and tests

- `webui/src/contracts/web-search.ts`
- `webui/src/contracts/web-search.test.ts`
- `webui/src/islands/web-search.tsx`
- `webui/src/islands/web-search.test.tsx`
- `webui/src/entrypoints/web-search.tsx`
- `webui/src/test/web-search-fixture.ts`
- `webui/src/web-search.css`
- `webui/src/lib/api.ts`
- `webui/vite.config.ts`

### Generated contracts and assets

- `devtools/render_openapi.py`
- `docs/openapi/search.yaml`
- `webui/src/api/generated.ts`
- `polylogue/daemon/static/dist/*`
- `docs/plans/topology-target.yaml`
- `docs/topology-status.md`

## Acceptance matrix

| Mission requirement | Result | Evidence |
|---|---|---|
| DSL-teaching query input | Complete | Four parser-test expressions with exact source lines and real-parser round-trip test |
| Snippets and provenance | Complete | Typed hit projection, bounded lexical/filter-only evidence, origin/session/message refs |
| Reader deep link | Complete | Canonical percent-encoded `/s/<id>#<anchor>` link validated server and browser side |
| Origin, repository, time facets | Complete | Server computes every family; selected zero-count values retained |
| Continuation only | Complete for static archive | No offset/page controls in client protocol; opaque `q1` replay only after first page |
| Exact versus qualified totals | Complete | Shared `QueryCoverage`; lexical exact, semantic/hybrid lower-bound or unknown |
| FTS lagging state | Complete | Separate `fts_lagging` state and 409 response; no partial rows |
| Embeddings absent state | Complete | Separate `embeddings_absent` state and 409 response |
| Parser diagnostic and correction | Complete | Typed 400 state with real diagnostic and parser-backed example |
| Semantic SSR | Complete | `/app/search` contains form, facets, rows/states, continuation link, bootstrap data |
| Preact islands | Complete | Facets and paging hydrate over generated client; server order preserved |
| Server-owned semantics | Complete | Browser has no parser, filter, ranker, or facet counter |
| Vitest paging test | Complete | Appends one validated server page without local reranking |
| Vitest parse-error test | Complete | Renders daemon diagnostic and corrected example |
| Vitest facet-selection test | Complete | Keeps current rows pending and applies newest authoritative response |
| Python real-route test | Complete | Exercises actual daemon dispatch, manifest asset resolution, SSR, and JSON fields |
| Zero CDN | Complete | All runtime JS/CSS is packaged under `/app/assets` |
| Sanitized fixtures | Complete | Synthetic IDs, titles, snippets, origins, and repositories only |
| Mutation-stable exactly-once paging | Not yet complete | Shared cursor is offset-based; requires z9gh.9.1 keyset/snapshot/spool work |
| Live browser/deployment proof | Unverified | No operator daemon, archive, credentials, browser, NixOS, or deployed package used |

## Apply order

1. Check out upstream `bf8191b3f56aa40da8f271df7f3385c712825497`.
2. Reproduce the supplied three-file dirty state. Its binary diff must hash to `f30102453e3576c9049eb65dcea7265fa99ac694c45531b06b0a34dd85279e9f`.
3. From that exact state, run `git apply --check PATCH.diff`.
4. Apply with `git apply PATCH.diff`.
5. Install the existing Python and WebUI dependencies using the repository’s normal locked workflow.
6. Run the commands in `TESTS.md`.
7. Rebuild generated surfaces only when modifying their owners; the committed generated files already match this patch.

The patch was clean-applied to a detached copy of the synthetic exact-snapshot baseline. Every one of the 32 changed paths was compared byte-for-byte with the implementation worktree. The aggregate SHA-256 over path names and resulting bytes is `2c3556d520e0ac1488f6cba3bd1ab37e759cbf7c26aa7c098dad401d4daeb4ea`.

## Legacy deletion candidates, not deleted

The new vertical supersedes the discovery/search behavior embedded in the legacy shell, but this patch deliberately does not delete shared reader or workspace code.

Candidates for a separately certified deletion/migration are:

- the search input and local state in `polylogue/daemon/web_shell.py`;
- `loadSessions()` search requests to `GET /api/sessions?query=…&origin=…&offset=…&limit=…`;
- `loadFacets()` requests to `GET /api/facets` for the shell search sidebar;
- offset keyboard paging (`n`/`p`) and exact-total assumptions in that shell flow;
- legacy root `/` discovery UI after all navigation entry points target `/app/search` and the dedicated read vertical;
- legacy `/s/:id` reader shell only after the WebUI v2 read page proves feature parity with `web_shell_reader.py` and its attachment, paste, lineage, provenance, similar-session, realtime, selection, workspace, and semantic-card integrations.

Do not delete `web_shell_reader.py` or sibling modules merely because search now deep-links to `/s/:id`; they still own the current reader implementation in this snapshot.

## Important limitations and risks

The strongest remaining correctness gap is mutation-stable continuation. Offset replay can duplicate or skip logical rows when data or ranking changes between pages. A substantial next pass should land keyset, snapshot, or spool semantics in the shared query transaction and then consume them unchanged here.

Page rows, exact count, origin/repository facets, four time-bucket counts, and representative evidence are separate controlled reads, not one SQLite snapshot. Concurrent writes can briefly produce page/count/facet skew. The implementation downgrades exact coverage when page evidence directly disproves a count, but it cannot promise transaction-wide snapshot coherence.

Semantic and hybrid retrieval are provider-bounded. Qualified coverage is intentional and must not be “simplified” into exact totals without stronger provider/executor evidence.

The typed SSR bootstrap embeds escaped JSON in HTML. Tests cover script-breaking content and route output, but no live browser content-security-policy or accessibility scan was run.

No Playwright interaction/visual test, operator archive, live embeddings database, deployed daemon, credential lifecycle against a real browser, Nix/wheel inclusion smoke, or production memory/latency benchmark was available. Those checks remain unverified rather than implied by unit and jsdom coverage.

One pre-existing SQLite VM-step canary fails on both the untouched supplied baseline and the implementation state with the same assertion (`0 >= 50000`). It is excluded from the green focused suite and recorded in `TESTS.md`; it is not hidden as an implementation pass.

## Verification summary

Completed verification includes:

- 705 focused Python tests passed, with the identical baseline canary explicitly deselected;
- 23 Vitest tests passed across all WebUI test files;
- 8 generated-client contract tests passed;
- complete `npm run check` passed, including generated checks, design-system lint, strict TypeScript, tests, production build, and client/SSR design-system builds;
- Ruff lint and formatting passed for all 13 changed Python files;
- targeted mypy passed for all 13 changed Python files;
- `devtools render all --check` passed;
- topology reports 1,037 realized and 1,037 declared modules with no blocking mismatch;
- layering verification found no violations;
- degrade-loudly verification found no new unallowlisted broad exception fallback;
- production and complete npm audits reported zero vulnerabilities;
- staged diff and patch whitespace checks passed;
- clean apply and byte comparison passed for all changed files;
- ZIP and embedded patch validation are described in the final delivery report.

A small further polish pass would add little value. The next meaningful iteration is substantial: shared mutation-stable continuation, snapshot-aware facets/evidence, and real Playwright plus deployment/package verification should be landed together rather than as leaf-level WebUI workarounds.

# WebUI v2 Search Vertical — Source and Decision Evidence

## Authority order used

1. Current source in the supplied project-state archive.
2. Repository instructions and architecture documentation.
3. Complete relevant Beads records.
4. Current history around the supplied commit.
5. Earlier implementation package only where it remained consistent with the changed source.

The current source overruled the earlier package’s assertion that the WebUI v2 scaffold was absent.

## Snapshot observations

The supplied repository points at `bf8191b3f56aa40da8f271df7f3385c712825497`, contained by `master` and `origin/master`. The commit subject is `feat: land WebUI v2 scaffold, design system, and generated client (#3074)`.

The archive was not clean. It contained modifications to:

- `polylogue/archive/query/unit_results.py`;
- `polylogue/daemon/http.py`;
- `polylogue/hooks/__init__.py`.

Their binary patch SHA-256 is `f30102453e3576c9049eb65dcea7265fa99ac694c45531b06b0a34dd85279e9f`. These changes were preserved in synthetic baseline commit `be2c26f90e3cfdba3a2015cb045ccf7f42cf7ce8` before staging the search implementation.

This matters because `PATCH.diff` is intentionally based on the actual supplied state, not a guessed clean upstream checkout.

## Current scaffold evidence

The current snapshot contains:

- Vite entrypoint ownership in `webui/vite.config.ts`;
- generated design-system source and tests under `webui/src/design-system`;
- generated API runtime/client under `webui/src/api`;
- first-party credential bootstrap in `webui/src/lib/api.ts`;
- manifest-based asset discovery in `polylogue/daemon/webui.py`;
- packaged hashed assets under `polylogue/daemon/static/dist`;
- a semantic archive-overview SSR/island path that provides the integration pattern.

Therefore, building a second scaffold or hard-coding a standalone `/assets/web-search.js` contract would contradict current source. The implementation adds a named `web-search` Vite entry and consumes the existing manifest, client, auth, and design system.

## Search and parser evidence

`docs/search.md` and the query modules establish layered retrieval:

- lexical FTS over archive search text;
- semantic/vector retrieval when embeddings exist;
- the Lark query DSL as the expression owner;
- `SessionQuerySpec` and existing archive APIs as the execution path.

The browser receives no parser or plan. `polylogue/daemon/web_search.py` lowers `q` through the real parser/compiler and existing spec. This preserves the architecture-spine rule that UI surfaces project substrate behavior rather than reproduce it.

The exact teaching examples are present in `tests/unit/cli/test_query_expression.py` at lines 5345, 5584, 1660, and 5433. A production-parser test ties every response example to that source line.

## Shared transaction evidence

`polylogue/archive/query/transaction.py` existed in the changed snapshot and already established the beginning of the request/page/continuation vocabulary. The implementation extends that owner rather than defining a WebUI-only cursor.

Added invariants include:

- closed request state with operation, arguments, page size, offset, projection, and stable order;
- logical query identity independent of page offset;
- result identity derived from the complete logical request;
- exact/lower-bound/unknown coverage;
- strict continuation payload decoding and size bound;
- no continuation on an empty page.

The WebUI response embeds the transaction request plus refs, and its `q1` token replays the whole request. The current token still advances by offset, which does not satisfy the full mutation-stability objective in the Bead; the limitation is explicit.

## Bead findings

### `polylogue-z9gh.9.1`

Status: in progress, priority 0. Title: “Land the shared query transaction across every read surface.”

The record requires the transaction to be the sole production read boundary, to own compact typed pages, complete opaque continuation state, qualified totals, useful evidence, stable refs, and eventually exactly-once resumability through keyset, deterministic snapshot replay, or an owned spool.

Implementation consequence: this vertical uses and strengthens the shared transaction vocabulary, but does not falsely claim that its current offset continuation completes the Bead’s mutation-stable acceptance criteria.

### `polylogue-4p1`

Status: open, priority 1. Title: “Make Query × Projection × Render the sole executable read algebra.”

The record says surfaces must be adapters to one read meaning rather than independent query/render paths.

Implementation consequence: query selection/lowering remains in existing server query types; `WebSearchResponse` is a named projection; SSR and Preact both render that projection. The browser does not create an alternate filter or ranker.

### `polylogue-t46.8`

Status: open, priority 1. Title: “Replace MCP tool sprawl with a protocol-native verb algebra.”

The record says MCP must reuse the shared query transaction and may not create a parallel parser or query engine.

Implementation consequence: `buildQueryTransactionRequest` emits a transport-neutral shape with `operation: "query"`, complete arguments, projection, page size, offset, and stable order. HTTP still sends compact URL parameters, but the builder can converge with the declared MCP verb without changing semantic fields.

## Storage contradiction found and resolved

Observed source behavior before the patch:

- search count was based on distinct sessions;
- lexical paging operated over matching blocks;
- one session with multiple matching messages could occupy multiple rows.

This contradicted the response unit and made continuation non-lossless even on an unchanged archive.

Resolution: group/rank by session before offset and limit, then choose one best evidence block per selected session. A direct SQLite regression test plants multiple matching blocks in one session and fails if block-level pagination returns.

## Relative-date contradiction found and resolved

A continuation must preserve complete request meaning, but relative dates can drift if `since:7d` is evaluated against each request’s current wall clock.

Resolution: the initial request captures timezone-aware `as_of`; compilation uses it as the relative-date base on every page. The continuation carries it unchanged.

## SSR/asset contradiction found and resolved

The earlier package used a fallback interface because its snapshot lacked the WebUI scaffold. The changed source now has a canonical hashed-asset manifest and generated client.

Resolution:

- route moved to the scaffold namespace: `/app/search`;
- API uses `/api/web-search` to avoid colliding with legacy/general search naming;
- the manifest resolves the actual JS and CSS filenames;
- the generated client owns the operation;
- the existing credential bootstrap owns API access;
- the design system owns components and tokens.

No hard-coded asset filename or parallel credential mechanism remains.

## Legacy shell evidence

`polylogue/daemon/http.py` still serves the legacy shell for `/`, `/s/:id`, and workspace routes. `polylogue/daemon/web_shell.py` contains:

- `state.query`, `state.origin`, `state.offset`, `state.limit`, and `state.total`;
- `loadSessions()` calling `/api/sessions` with `query`, `origin`, `offset`, and `limit`;
- `loadFacets()` calling `/api/facets`;
- local keyboard next/previous offset paging;
- a search input that triggers both requests.

The new search vertical supersedes that discovery flow. It does not yet supersede the legacy reader’s broader attachment, paste, lineage, provenance, similar-session, realtime, selection, and workspace behavior, so those modules are deletion candidates only after independent parity certification.

## Route and security evidence

`polylogue/daemon/route_contracts.py` declares:

- `/app/search` as a loopback semantic shell route;
- `/api/web-search` as a credential-protected read-query route.

`polylogue/daemon/http.py` dispatches both through the existing route machinery and serves hashed assets via `/app/assets/:name`. The no-JS form submits only to the same origin; the CSP was narrowed to `form-action 'self'` rather than relaxed generally.

The implementation’s unexpected-error path returns a typed generic `failed` state and does not expose archive paths or exception internals. A security test checks the FTS probe-failure case specifically.

## Generated-contract evidence

`devtools/render_openapi.py` imports the Pydantic `WebSearchResponse` and defines `projectWebSearch`. It generates:

- `/api/web-search` parameters and response statuses;
- component schemas for the typed projection;
- route metadata for `/app/search` and `/api/web-search`.

`webui/src/api/generated.ts` is generated from that OpenAPI document. The full render check and explicit WebUI-client check pass, so hand-edited client drift is not being masked.

## Browser contract evidence

The runtime decoder is intentionally stricter than TypeScript’s compile-time interface. It verifies:

- exact top-level and nested keys;
- timezone-aware timestamps;
- request operation/projection/order/page bounds;
- unique non-empty facet arguments and options;
- selected facet options against request identity;
- canonical query/result refs;
- monotonically expected ranks;
- coverage not below observed rows;
- rowless degraded states;
- canonical session/message refs;
- percent-encoded reader path and fragment;
- continuation presence, prefix, and non-empty page evidence.

This prevents hydration from silently accepting a server/client contract split.

## Patch and reproduction evidence

The staged binary patch is 278,765 bytes and 4,960 lines. SHA-256:

`ddc49958fba929a8c68e00c663b15fc7be894a2a82f5345e15524b69c7f71bb5`

It passes `git diff --cached --check`. It was applied to a detached copy of the exact synthetic baseline. All 32 changed paths matched the implementation worktree byte-for-byte, including generated binary-diff paths. Aggregate path/content hash:

`2c3556d520e0ac1488f6cba3bd1ab37e759cbf7c26aa7c098dad401d4daeb4ea`

## Unresolved uncertainty

The source proves static-archive continuation and identity behavior; it does not prove behavior during concurrent mutation. Offset-based replay is therefore a known limitation, not a resolved uncertainty.

Separate SQL reads for rows, counts, facets, time buckets, and evidence can observe slightly different archive moments. No transaction-wide snapshot was added.

The tests prove semantic SSR text and hydration behavior under jsdom, not a live browser’s accessibility tree, layout, cookie timing, or deployed CSP.

The environment did not include operator secrets, a live archive, a live embeddings provider/database, or a deployed Nix/wheel artifact. No claims rely on them.

## Decision summary

- Use the landed WebUI v2 scaffold, never a second one.
- Keep parsing, filtering, ranking, facets, readiness, and coverage server-owned.
- Use one closed response model for JSON, SSR, and hydration.
- Page distinct sessions, not FTS blocks.
- Replay continuation alone and preserve the complete request frame.
- Qualify semantic totals rather than manufacture exactness.
- Withhold rows for FTS convergence and missing required embeddings.
- Preserve selected zero-count facets.
- Keep legacy reader modules until the read vertical proves parity.
- Defer mutation-stable continuation and snapshot coherence to the shared transaction owner.

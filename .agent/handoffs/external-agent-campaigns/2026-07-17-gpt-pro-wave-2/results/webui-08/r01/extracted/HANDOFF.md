# WebUI v2 generated client contracts — implementation handoff

## Mission and delivered outcome

This package implements the generated-client layer requested by `webui-08-client-contracts` against the attached Polylogue snapshot. The implementation makes the generated daemon OpenAPI document the single client-generation input, commits deterministic TypeScript output, adds a dependency-free same-origin runtime, and gives the two currently declared paged reads real async iterators.

The end-to-end path is now:

```text
Pydantic surface payloads + daemon route declarations
  -> devtools render openapi
  -> docs/openapi/search.yaml
  -> devtools render webui-client
  -> webui/src/api/generated.ts
  -> PolylogueClient + FetchTransport + Page<T>
```

The implementation also repairs a production mismatch discovered while tracing that path. `/api/query-units` already returned `QueryTransaction` continuation tokens, but the HTTP handler could not consume them. The handler now accepts the opaque token, reconstructs the complete server-owned request, validates its operation/projection/order/result identity, rejects parameter overrides, and advances the same bounded query. This makes `for await (const page of client.query(...))` a real production behavior rather than a client-side offset convention.

The current generated client covers every operation in the snapshot's generated OpenAPI artifact:

- `bootstrapWebCredential()`
- `revokeWebCredential()`
- `searchSessions()`
- `search()` async iterator for ranked search
- `queryUnits()`
- `query()` async iterator for terminal query units
- `readSessionView()`
- `listAssertionClaims()`

No runtime package was added. The client runtime uses only TypeScript/DOM platform APIs already available to the WebUI scaffold.

## Snapshot identity and patch base

The snapshot overview identifies the source as `/realm/project/polylogue`, generated at `2026-07-17T180950Z`, on branch `master` at:

```text
536a53efac0cbe4a2473ad379e4db49ef3fce74d
fix(repair): harden raw authority convergence (#3046)
```

The overview records `dirty=true`. After reconstructing the repository from the supplied all-refs bundle and tracked working-tree archive, the tracked checkout had no diff against that commit and the snapshot's branch-delta patch was empty. The dirty bit therefore is not an apply prerequisite represented in this package; it may describe ignored or untracked local state that the snapshot intentionally did not make part of a tracked patch. `PATCH.diff` is based exactly on commit `536a53efac0cbe4a2473ad379e4db49ef3fce74d`.

The working branch used to produce this package was `work/webui-08-client-contracts`; it adds no commit and changes no Git history.

## Evidence inspected before implementation

### Repository instructions and architecture

- `CLAUDE.md` / `AGENTS.md`: substrate owns meaning; surfaces are adapters; generated surfaces and focused verification are required.
- `docs/architecture.md` and the architecture guidance embedded in `CLAUDE.md`: read semantics must not be reimplemented in WebUI code.
- `devtools/generated_surfaces.py`, `devtools/render_support.py`, `devtools/command_catalog.py`, and the `render all --check` orchestration.
- `devtools/render_openapi.py` and committed `docs/openapi/search.yaml`.
- CLI output schema generation and its committed/check-mode pattern.

### Production read and continuation paths

- `polylogue/archive/query/transaction.py`: `QueryTransactionRequest`, logical `query_ref`, `QueryContinuation`, `QueryResultPage`, and complete opaque continuation state.
- `polylogue/archive/query/unit_results.py`: `query_unit_request()` and `query_unit_envelope()`; the latter already creates the continuation, `query_ref`, and `result_ref` from the canonical request.
- `polylogue/daemon/http.py`: route registration, `/api/sessions`, `/api/query-units`, session read-view routing, error envelopes, web credential admission, and same-origin behavior.
- `polylogue/surfaces/payloads.py`: `SearchEnvelope`, `SessionListResponse`, query-unit envelopes, session read-view envelope, assertion claims, and daemon error payloads.
- Ranked-search cursor code and tests: the cursor is merged into the original filters while unstable offset state is removed.

### Declaration direction

- `.beads/issues.jsonl` records:
  - `polylogue-o21.1`: storage-free declare-once kernel and deterministic derivation.
  - `polylogue-t46.8.1`: declarative MCP tool algebra, generated contracts/discovery/inventories, explicit paging/ref behavior, and no legacy deletion in the pilot.
  - `polylogue-z9gh.9.1`: shared bounded read transaction, opaque complete continuation, exact/qualified coverage, and query/result refs.
  - `polylogue-4p1`: Query × Projection × Render as the sole executable read algebra.
- Current source under `polylogue/declarations/` and `polylogue/mcp/declarations/`. This is newer than the mission wording that describes the declaration kernel as forthcoming. The kernel and MCP pilot are present, but the daemon HTTP OpenAPI renderer is not yet derived from those registries.

### Relevant history

The following commits were inspected to understand why the present contracts look as they do:

- `9163d0134f3d334960e4c249c96c5671919a9a06` — bounded agent-facing archive reads and shared query transaction.
- `ed44be18f448c31f9fa5b9289c75da7eee99b131` — current MCP declaration algebra.
- `c1f7704fa4e6723fb1c3edd8eda0ad72bcc06d6b` — bounded Web cockpit aggregate routes.
- `0e0cddaeee82b17d1a1c0edd562f26fe26e6896b` — first-party daemon web credential bootstrap.
- `7efde00ec572f81a8c3b365bce69a4151e81ecca` — generated surfaces bound to contract owners.
- `86281ffe5bbd38addd522939f93671ef0edf1211` — query-unit daemon OpenAPI contract.
- `02a1bb485a39f19edeff6826b3c49cbc30091276` — ranked-search stable cursor/keyset paging.
- `20d7955a1b58998aefcc1cae713897092205121f` — typed ranked-search envelope and OpenAPI emission.

## Generator design

`devtools/render_webui_client.py` is a small deterministic OpenAPI 3.1 renderer. It intentionally consumes the JSON Schema subset emitted by this repository's Pydantic models rather than creating a second hand-maintained schema inventory or introducing a runtime generator dependency.

It reads:

1. standard OpenAPI operations, operation IDs, path/query parameters, JSON success/error responses;
2. standard component schemas using `$ref`, `const`, `enum`, `anyOf`, `oneOf`, `allOf`, object properties, arrays, scalars, nullable unions, and `additionalProperties`;
3. one narrow operation-level extension, `x-polylogue-page`, which describes how an already-declared operation advances its server-owned continuation.

It emits:

- one TypeScript alias per component schema;
- parameter, success response, and error union aliases per operation;
- one `PolylogueClient` method per operation;
- one async page iterator for each operation carrying `x-polylogue-page`;
- page envelope guards sufficient to reject the non-page branch of a union operation before reading its item field.

Output is deterministic because component names, paths, operations, parameters, status codes, and object properties are rendered in stable order. The committed file begins with its generator and source path, and `--check` compares exact bytes.

The generator fails closed instead of silently dropping future contract shape. It rejects malformed/duplicate operation IDs, unknown or external schema references, unsupported parameter references/locations, operations without a 2xx response, undeclared iterator parameters, unknown page envelopes/properties, unsupported continuation modes, request bodies, unsupported JSON Schema types, unsupported evidence qualifications, and exact/qualified coverage value overlap. A future body-bearing daemon operation therefore breaks generation until body support is implemented rather than producing an incomplete client.

### Why `x-polylogue-page` is not a parallel schema source

The extension does not redeclare request or response fields. All field types still come from the generated OpenAPI operation and component schemas. It supplies transport semantics OpenAPI cannot infer from a cursor-looking string alone:

- iterator method name;
- which first-page parameters are required;
- which declared response branch is paged;
- item/cursor/ref property names;
- whether continuation is cursor-only or merged with original filters;
- which unstable parameters must be removed;
- how the server qualifies totals.

This metadata lives beside the route declaration in `render_openapi.py`, is committed in the generated OpenAPI artifact, and is tested as part of that artifact. When the daemon renderer is retargeted to `DeclarationSpec`, the declaration derivation should emit the same standard operation/schema plus this paging metadata.

## Exact schema sources and server-side declaration additions

The sole generator input is `docs/openapi/search.yaml`, itself generated by `devtools/render_openapi.py` from existing Pydantic surface models and route declarations.

This patch adds the smallest missing declarations needed for sound current client behavior:

1. Publishes existing `SessionListResponse`, which pulls its existing dependent row/flag/action/ref schemas into OpenAPI.
2. Corrects `GET /api/sessions` success schema from a false `SearchEnvelope`-only declaration to `oneOf: SearchEnvelope | SessionListResponse`.
3. Corrects the route description: ranked search has `next_cursor`; current plain list mode remains offset-based.
4. Declares ranked-search paging metadata: `hits`, `next_cursor`, merged continuation, `offset` reset, and dynamic exact/capped/sampled/estimate coverage.
5. Makes query-unit `expression` first-page-only rather than unconditionally required.
6. Declares the `continuation` query parameter and its no-overrides rule.
7. Declares query-unit paging metadata: `items`, opaque cursor-only continuation, `query_ref`, `result_ref`, and page-qualified totals.

No new response model was invented. No daemon route was created solely for the WebUI. The only production behavior addition is consumption and validation of the continuation already emitted by the shared query transaction.

## Runtime client contract

`webui/src/api/runtime.ts` defines the shared contract once:

```ts
export type ExactCoverage = {
  readonly kind: "exact";
  readonly total: number;
};

export type QualifiedCoverage = {
  readonly kind: "qualified";
  readonly total: number | null;
  readonly qualification: "page" | "capped" | "sampled" | "estimate" | "unknown";
};

export type Coverage = ExactCoverage | QualifiedCoverage;

export type Page<T, TEnvelope = unknown> = {
  readonly items: ReadonlyArray<T>;
  readonly cursor: string | null;
  readonly coverage: Coverage;
  readonly queryRef: string | null;
  readonly resultRef: string | null;
  readonly envelope: TEnvelope;
};
```

The envelope remains available because verticals sometimes need server-projected metadata beyond the common page fields. The common fields prevent every vertical from reinterpreting cursor, coverage, and refs.

### `FetchTransport`

The runtime transport:

- accepts only origin-relative generated paths and rejects absolute/protocol-relative escapes;
- normalizes an explicit SSR/test base URL to its origin and disallows URL credentials;
- uses `credentials: "same-origin"`, `cache: "no-store"`, `redirect: "error"`, and `referrerPolicy: "no-referrer"`;
- emits `Accept: application/json`, `X-Polylogue-Web-Client: 1`, and a request ID when absent;
- serializes optional JSON bodies at runtime, while generation currently rejects body-bearing operations;
- maps declared daemon error payloads to `DaemonHttpError<TPayload>` while preserving status, status text, code, detail, field, request ID, and web credential state;
- distinguishes invalid successful JSON (`DaemonProtocolError`), pre-response transport failures (`TransportError`), external aborts (`RequestAbortedError` with the original reason), and deadline expiry (`DeadlineExceededError`);
- supports external `AbortSignal`, absolute `Date`/epoch deadlines, relative `timeoutMs`, and a bounded default timeout.

`iteratePages()` yields each page exactly once and rejects a repeated non-null cursor with `ContinuationProtocolError` before a duplicate page can be yielded.

## Continuation behavior

### Query units

First call:

```text
/api/query-units?expression=...&limit=...&origin=...
```

Follow-up calls:

```text
/api/query-units?continuation=<opaque q1 token>
```

The client sends only the token. The daemon decodes its `QueryTransactionRequest`, requires operation `query_units`, projection `default`, stable order `canonical`, exact argument keys `expression` and `session_filters`, a non-empty expression, a mapping filter payload, and the result-ref derived from the request's logical query-ref. It then reuses the server-owned page size, offset, expression, and normalized filters. A continuation mixed with any other query parameter returns `400 invalid_continuation`.

Malformed base64 and UTF-8 continuation bodies are normalized to `ValueError` by `QueryContinuation.decode()`, so the route returns the typed error instead of leaking a decoder exception through the handler boundary.

Query-unit `total` is the page row count, not a logical result cardinality. Every query page therefore reports:

```ts
{ kind: "qualified", total: number | null, qualification: "page" }
```

The daemon's stable `query_ref` and `result_ref` are passed through.

### Ranked session search

The first call contains query, filters, optional limit, and optional initial offset. A follow-up retains the original server filters, removes `offset`, and adds the opaque ranked-search cursor. Client code never reconstructs rank/keyset state.

A numeric total is exact only when `exactness` is `exact` or absent under the current schema default. `capped`, `sampled`, and `estimate` remain qualified and are not displayed as exact totals by consumers of `Page<T>`.

### Plain session list

The same `/api/sessions` operation has an offset-based `SessionListResponse` branch when `query` is absent. The generator exposes it through `searchSessions()`, but it does not fabricate an async iterator because the snapshot does not declare a server-owned list continuation. This is a deliberate honesty boundary for webui-02.

## Drift-check wiring

The new command is registered in both the operator command catalog and generated-surface registry:

```text
python -m devtools render webui-client
python -m devtools render webui-client --check
```

`webui/package.json` adds:

```text
npm run generate:client
npm run check:client
npm run typecheck
npm run test:unit
npm test
```

`npm test` runs client drift check, strict TypeScript compilation, and runtime unit tests. `python -m devtools render all --check` now includes `webui-client` after `openapi`, so changing Pydantic models, operation declarations, or page metadata without committing regenerated TypeScript fails CI drift verification.

The generator fixture and golden output are independent of the large production schema. This pins generator formatting and continuation behavior, while the committed-production drift test pins `docs/openapi/search.yaml` to `webui/src/api/generated.ts`.

## Per-vertical adoption map

### webui-02 — session list and read

Replace handwritten ranked session fetches and local `SearchEnvelope` interfaces with `client.search()`. Replace single-session read fetches and local read-view interfaces with `client.readSessionView()`.

Plain list mode can call `client.searchSessions()` and narrow the `SessionListResponse` branch, but it remains offset-based. A server-side list continuation must be added before webui-02 can meet its continuation-only list requirement. The client layer must not emulate one.

### webui-03 — search

Replace all `/api/sessions?query=...` fetches, local search envelope types, cursor merging, offset reset logic, and exactness interpretation with `client.search()`.

Use `client.query()` for terminal DSL unit reads. Both utilities keep filtering/ranking server-side and expose common coverage/refs.

### webui-04 — transcript rendering

The snapshot declares the current `SessionReadViewEnvelope`, exposed as `client.readSessionView()`. It does not yet declare the semantic-card document/detail endpoint required by webui-04. Add the server-owned Pydantic card-document model and OpenAPI path, then regenerate. Do not add a local card-document schema or handwritten fetch wrapper.

### webui-05 — insights, freshness, and status

The daemon has operational routes, but the generated OpenAPI artifact does not yet declare the registry-driven insights browser, named-source freshness contract, or target component-status protocol. Those are server declaration gates. Once typed route models are added to the OpenAPI derivation, this generator emits their client methods without runtime changes.

### webui-06 — cost and usage

The daemon has provider/session cost routes, but the generated OpenAPI artifact does not declare the required aggregate, cache-economics, coverage, and session-usage response contracts. Add typed server-owned response models and route declarations over existing operations, then regenerate. Client-side summing remains prohibited.

## Retargeting to the declaration kernel

The snapshot already contains `polylogue/declarations/` and `polylogue/mcp/declarations/`, contrary to the older wording that treats them as future work. The remaining migration is to make daemon HTTP route/OpenAPI derivation consume those declarations or their shared kernel equivalent.

The client renderer does not import declaration Python classes. It depends on the generated contract artifact. Retargeting therefore consists of making declaration derivation emit:

- standard OpenAPI component schemas;
- standard operations with operation IDs, parameters, JSON responses, and errors;
- the narrow `x-polylogue-page` transport metadata, or an equivalent standard extension translated into that shape.

If the generated artifact path changes, `--schema` and the generated-surface input path change. `runtime.ts`, vertical imports, `Page<T>`, error classes, iterator behavior, and committed drift semantics remain unchanged.

## Changed files

### Generated-surface control plane

- `.gitignore`
- `devtools/command_catalog.py`
- `devtools/generated_surfaces.py`
- `devtools/render_openapi.py`
- `devtools/render_webui_client.py`
- `docs/devtools.md`
- `docs/openapi/search.yaml`

### Production query behavior

- `polylogue/archive/query/transaction.py`
- `polylogue/daemon/http.py`

### WebUI client/runtime

- `webui/package.json`
- `webui/tsconfig.json`
- `webui/src/api/README.md`
- `webui/src/api/generated.ts`
- `webui/src/api/runtime.ts`

### Tests and fixtures

- `tests/fixtures/openapi/webui-client.yaml`
- `tests/fixtures/openapi/webui-client.generated.ts`
- `tests/unit/archive/query/test_transaction.py`
- `tests/unit/daemon/test_web_reader.py`
- `tests/unit/devtools/test_generated_surfaces.py`
- `tests/unit/devtools/test_render_openapi.py`
- `tests/unit/devtools/test_render_webui_client.py`
- `webui/tests/unit/client-contracts.test.mjs`

No existing test/helper was deleted. No dominated deletion is proposed in this slice.

## Acceptance matrix

| Requirement | Result | Evidence |
| --- | --- | --- |
| OpenAPI/generated-schema to TypeScript types | Complete for every operation/schema in current generated OpenAPI | `render_webui_client.py`, committed `generated.ts`, golden test |
| Thin typed client methods | Complete | `PolylogueClient` emits all six declared operations |
| Generic continuation envelope modeled once | Complete | `Coverage` and `Page<T, TEnvelope>` in `runtime.ts` |
| Deterministic committed output | Complete | stable renderer order, fixture golden, deterministic test |
| `--check` drift detection | Complete | command registration, production drift test, `render all --check` |
| Same-origin runtime | Complete | path/origin checks and transport test |
| Typed daemon errors | Complete | generated error unions plus `DaemonHttpError<TPayload>` |
| Deadline and abort support | Complete | runtime implementation and separate tests for both reasons |
| `for await` continuation iterator | Complete for ranked search and query units | generated `search()`/`query()`, runtime tests, real route test |
| Qualified/truncated total behavior | Complete | capped search test and page-qualified query test |
| Golden contract tests | Complete | fixture YAML + committed golden TS |
| Daemon/client drift test | Complete | production OpenAPI-to-client exact-byte check |
| Adoption notes webui-02…06 | Complete, with declaration blockers explicit | `webui/src/api/README.md` and this handoff |
| No parallel schema source | Complete | only generated OpenAPI plus local operation transport metadata |
| Zero new runtime dependencies | Complete | no dependency or lockfile additions |

## Apply order

Apply from a clean checkout of commit `536a53efac0cbe4a2473ad379e4db49ef3fce74d`:

```bash
git apply --check PATCH.diff
git apply PATCH.diff
```

Then verify in the repository's normal frozen environment:

```bash
python -m devtools render openapi --check
python -m devtools render webui-client --check
python -m devtools render all --check

cd webui
npm ci
npm test
cd ..

pytest -q \
  tests/unit/archive/query/test_transaction.py \
  tests/unit/devtools/test_command_catalog.py \
  tests/unit/devtools/test_render_webui_client.py \
  tests/unit/devtools/test_render_openapi.py \
  tests/unit/devtools/test_generated_surfaces.py \
  tests/unit/daemon/test_web_reader.py::TestReaderQueryUnits
```

When changing a server contract, regenerate in dependency order:

```bash
python -m devtools render openapi
python -m devtools render webui-client
```

## Verification performed

All checks below ran against the final staged tree in the supplied container unless marked otherwise.

- Focused Python contract/route suite: **37 passed in 9.09s**.
- WebUI drift + strict TypeScript + runtime tests: **8 passed**, with typecheck clean.
- `python -m devtools render openapi --check`: **passed**.
- `python -m devtools render webui-client --check`: **passed**.
- `python -m devtools render all --check`: **passed**, including CLI schemas, OpenAPI, WebUI client, docs/reference, equivalence/index, topology, and site pages.
- `python -m compileall` on changed Python modules/tests: **passed**.
- `git diff --check` and staged-patch whitespace check: **passed**.
- Fresh-checkout `git apply --check` and post-apply file equivalence: recorded in `TESTS.md` after package assembly.

The TypeScript checks used Node `22.16.0`, npm `10.9.2`, and globally available TypeScript `5.8.3`. The repository pins TypeScript `5.9.3`; installing that exact package was not possible without using the unavailable package network, so exact lockfile-toolchain execution remains unverified. Passing under strict 5.8.3 is useful but does not replace the operator's `npm ci` proof.

`python -m devtools verify --quick` was attempted and stopped immediately before its gates because the container has no `ruff` executable. Focused tests and generated-surface checks were run directly instead. Ruff formatting/lint and mypy are therefore unverified in this container.

## Important limitations and risks

1. The generated daemon artifact currently declares only six operations. Numerous existing daemon routes are still outside the typed OpenAPI surface. This patch does not pretend those undeclared routes are safe for generated adoption.
2. Plain session list mode is still offset-based. webui-02 needs a server-owned continuation before it can adopt continuation-only list paging.
3. Semantic-card transcript, registry insights/freshness/status, and cost/usage aggregate contracts are not in current OpenAPI. webui-04…06 require server declarations first.
4. The generator deliberately rejects request bodies. The current declared operations do not use one. A future body-bearing operation will fail closed until body type/request emission is added with tests.
5. Runtime response handling is a thin typed client, not a full runtime schema validator. It validates JSON syntax, HTTP/error envelopes, and page branch/item presence; compile-time types still trust the daemon's generated contract for detailed fields.
6. Ranked search does not currently expose `query_ref` or `result_ref` in its response schema, so common `Page` refs are `null` for search. Query-unit pages preserve both.
7. The original snapshot dirty state was not available as a tracked delta. The patch is intentionally against the named commit and does not include ignored operator state.
8. No live daemon, browser, operator archive, credentials, Nix build, wheel/sdist, or deployed WebUI was available. Those checks remain unverified.

## Value of another iteration

A small repair pass would add exact TypeScript `5.9.3`, Ruff, and mypy verification and address any mechanical findings. It would not materially change the design.

A substantial second pass becomes valuable when the parallel WebUI vertical contracts or daemon declaration-kernel integration are available. It could declare and generate the missing semantic-card, insights/freshness/status, cost/usage, and continuation-based session-list operations; add end-to-end browser/SSR adoption; and move `x-polylogue-page` derivation into the shared declaration registry. That is real additional product scope rather than polish on this package.

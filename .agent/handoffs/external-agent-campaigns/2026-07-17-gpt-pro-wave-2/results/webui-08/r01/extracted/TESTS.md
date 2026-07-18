# Test design and execution results

## Test strategy

The test set closes four different failure classes instead of relying on one generated-file snapshot:

1. **Generator correctness** — a small fixture schema pins deterministic TypeScript output and fail-closed behavior.
2. **Daemon/client drift** — the production generated OpenAPI must reproduce the committed production TypeScript exactly.
3. **Real production paging** — the daemon route must consume its own `QueryTransaction` continuation and preserve logical refs across pages.
4. **Runtime behavior** — the generated methods and dependency-free fetch transport must send the right requests, preserve qualification, and classify failures correctly.

No existing test was deleted or weakened.

## Python contract and production-route tests

### `tests/unit/devtools/test_render_webui_client.py`

| Test | Production dependency exercised | Anti-vacuity mutation/removal that makes it fail |
| --- | --- | --- |
| Golden fixture matches | `TypeScriptRenderer` ordering, schema lowering, operation generation, cursor-only iterator, page coverage | Change component/property sort order; merge first-page filters into cursor-only continuation; alter generated formatting; drop `Page<T>` |
| Check mode detects drift | exact-byte comparison in `render(..., check=True)` | Return success without comparing output; append content to committed output |
| Committed client matches OpenAPI | `docs/openapi/search.yaml -> webui/src/api/generated.ts` | Change a Pydantic schema/operation/page extension without regenerating; hand-edit generated TypeScript |
| Unknown page schema fails closed | page response-schema validation | Accept a missing component and emit `unknown` |
| Determinism | stable rendering from identical input | Introduce set/dict iteration without sorting or a timestamp/banner |
| Request body fails closed | operation feature inventory | Silently ignore `requestBody`, producing a client that cannot send the declared operation |
| Unknown qualification fails closed | `Coverage` vocabulary alignment | Allow a new server exactness value while casting it to an unrelated TypeScript union |

### `tests/unit/devtools/test_render_openapi.py`

The added test pins:

- publication of existing `SessionListResponse`;
- `GET /api/sessions` response as `SearchEnvelope | SessionListResponse`;
- ranked-search iterator metadata and exactness mapping;
- first-page-only query expression;
- declared opaque continuation parameter;
- cursor-only query iterator metadata, refs, and page-qualified total.

Removing any declaration or reverting `/api/sessions` to `SearchEnvelope`-only makes the test fail before TypeScript generation.

### `tests/unit/devtools/test_generated_surfaces.py`

Pins `docs/openapi/search.yaml` and `devtools/render_webui_client.py` as inputs to the `webui-client` generated surface. Removing either input makes cache/drift ownership incomplete and fails the test.

### `tests/unit/devtools/test_command_catalog.py`

Exercises uniqueness/category/dispatch invariants after registering `render webui-client`. A duplicate command name, invalid category, or unresolved entry point fails the catalog tests.

### `tests/unit/archive/query/test_transaction.py`

The new malformed-token cases use payloads that can raise `UnicodeDecodeError` or base64 decoder errors before JSON parsing. The contract requires all malformed continuations to become `ValueError("invalid query continuation")`.

Removing the new `binascii.Error`/`UnicodeDecodeError` handling causes this test to fail and, through the HTTP route, can turn a client input error into an internal handler failure.

### `tests/unit/daemon/test_web_reader.py::TestReaderQueryUnits`

| Test | Production dependency exercised | Anti-vacuity mutation/removal that makes it fail |
| --- | --- | --- |
| Consumes opaque continuations | live loopback daemon, handler decode/validation, `query_unit_request`, archive execution, `query_unit_envelope` | Remove continuation branch; reconstruct offset in test/client; fail to preserve filters/page size; change query/result identity |
| Rejects continuation overrides | handler's `set(params) == {"continuation"}` boundary | Merge caller-supplied limit/filter into decoded request |
| Rejects malformed continuation | shared decoder normalization plus handler `400 invalid_continuation` mapping | Remove malformed payload catches; let decoder exception escape or return 500 |

The paging test uses `limit=1`, exhausts all pages through the returned opaque token, asserts offsets `[0, 1, 2]`, stable `query_ref`, stable `result_ref`, and unique message IDs. It therefore fails if a continuation merely retries the first page or if the client/server reconstructs the query incorrectly.

## TypeScript/runtime tests

`webui/tests/unit/client-contracts.test.mjs` is compiled from strict TypeScript output and run with Node's built-in test runner.

| Test | Contract protected | Anti-vacuity mutation/removal |
| --- | --- | --- |
| Query iterator sends only continuation after page one | complete opaque QueryTransaction state remains server-owned | Merge expression/filter/limit into follow-up request; use offset instead of token |
| Capped search total stays qualified | evidence honesty for truncation/qualification | Treat every numeric total as exact; drop `exactness` mapping |
| Search continuation removes offset and keeps filters | ranked cursor stability | Keep offset on follow-up; drop provider/query/limit filters |
| Repeated cursor is rejected before duplicate yield | iterator termination and no duplicate pages | Remove seen-cursor set or perform check after yield |
| Daemon errors are typed and same-origin | error envelope mapping and secure request options | Return raw `Response`; omit code/detail/field/headers; use `include` credentials; permit cross-origin path |
| Expired deadline is distinct | deadline classification | Convert all aborts to generic transport errors |
| External abort reason is preserved | caller cancellation semantics | Replace the reason with deadline/generic abort state |
| Absolute/protocol-relative paths are refused | same-origin invariant | Call `new URL()` without path validation and allow origin escape |

The query test also asserts `Page.coverage = qualified/page` and stable query refs. The capped-total test is the required truncation/qualified-total regression.

## Drift and generated-surface tests

The following are separate gates by design:

```text
python -m devtools render openapi --check
python -m devtools render webui-client --check
python -m devtools render all --check
```

The first detects stale OpenAPI relative to Python contract owners. The second detects stale TypeScript relative to OpenAPI. The third verifies ordering and integration with the repository's complete generated-surface graph.

`npm test` repeats the client drift check before typechecking and runtime tests, so a WebUI-only CI job cannot accidentally skip the Python-generated contract boundary.

## Commands and final results

### Focused Python suite

```bash
PYTHONPATH=/mnt/data/polylogue_deps python -m pytest -q \
  tests/unit/archive/query/test_transaction.py \
  tests/unit/devtools/test_command_catalog.py \
  tests/unit/devtools/test_render_webui_client.py \
  tests/unit/devtools/test_render_openapi.py \
  tests/unit/devtools/test_generated_surfaces.py \
  tests/unit/daemon/test_web_reader.py::TestReaderQueryUnits
```

Result:

```text
37 passed in 9.09s
```

The external `PYTHONPATH` contains a temporary frozen-dependency approximation used only because the container's project environment could not resolve a missing package from the network. It is not part of `PATCH.diff` or the ZIP.

### WebUI drift, strict typecheck, and runtime tests

```bash
cd webui
PYTHONPATH=/mnt/data/polylogue_deps npm test
```

Result:

```text
client drift check: passed
TypeScript strict no-emit check: passed
Node runtime tests: 8 passed, 0 failed
```

Tool versions used:

```text
Node v22.16.0
npm 10.9.2
TypeScript 5.8.3
```

The repository pins TypeScript `5.9.3`. Exact pinned-tool execution remains unverified because `npm ci` could not be backed by a package network in this container.

### Generated surfaces

```bash
PYTHONPATH=/mnt/data/polylogue_deps python -m devtools render openapi --check
PYTHONPATH=/mnt/data/polylogue_deps python -m devtools render webui-client --check
PYTHONPATH=/mnt/data/polylogue_deps python -m devtools render all --check
```

Result: all passed. The aggregate run checked CLI reference/output schemas, OpenAPI, WebUI client, devtools reference, demo corpus datasheet, quality reference, product workflows, docs surface, MCP equivalence/index, topology status, and site pages.

### Syntax and patch hygiene

```bash
python -m compileall -q \
  devtools/render_webui_client.py \
  polylogue/archive/query/transaction.py \
  polylogue/daemon/http.py \
  tests/unit/devtools/test_render_webui_client.py \
  tests/unit/archive/query/test_transaction.py \
  tests/unit/daemon/test_web_reader.py

git diff --check
git diff --cached --check
```

Result: passed.

### Repository quick verification

```bash
PYTHONPATH=/mnt/data/polylogue_deps python -m devtools verify --quick
```

Result: not executed past the first gate. The command failed before checking changed files because the container has no `ruff` executable:

```text
FileNotFoundError: [Errno 2] No such file or directory: 'ruff'
```

This is an environment limitation, not a passing verification result. Ruff formatting/lint and mypy remain unverified.

### Apply verification

Package assembly performs these checks against a fresh detached worktree at the named base commit:

```bash
git apply --check PATCH.diff
git apply PATCH.diff
```

It then compares every patched path against the staged implementation tree and reruns the two generated-contract checks. Final results are recorded below after the package is assembled:

```text
`git apply --check`: passed
`git apply`: passed
byte comparison for all 22 patched paths: passed
OpenAPI drift check in fresh checkout: passed
WebUI client drift check in fresh checkout: passed
strict TypeScript check in fresh checkout: passed
post-apply `git diff --check`: passed
```

## Checks not performed

The following require resources unavailable in the supplied environment and are explicitly unverified:

- operator's live daemon and archive;
- browser/Playwright behavior and SSR integration;
- exact npm lockfile installation with TypeScript 5.9.3;
- Nix build, wheel/sdist packaging, and installed `devtools` entry point;
- live credentials/origin admission;
- full non-integration or integration pytest suite;
- Ruff and mypy;
- mutation runner execution beyond the concrete anti-vacuity mutations named above.

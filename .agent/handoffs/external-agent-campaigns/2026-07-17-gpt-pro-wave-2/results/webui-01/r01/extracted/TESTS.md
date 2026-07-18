# Test design and execution record

## Test strategy

The verification deliberately crosses the production seams instead of testing only helpers:

- Vitest exercises the real Preact island and its continuation loader contract.
- Python HTTP tests run the real daemon handler against the seeded split archive, the real query substrate, the packaged Vite manifest, and real HTTP asset responses.
- A clean detached checkout receives `PATCH.diff` with `git apply`, then repeats frontend, Python, formatting, typing, deterministic-build, and package checks.
- Wheel and sdist members are opened and compared with the expected packaged build files.

No test helper creates an alternate query or rendering framework.

## Frontend component proof

Test: `webui/src/islands/archive-overview.test.tsx::ArchiveOverviewIsland`

Production dependencies exercised:

- `ArchiveOverviewIsland`
- the `MessageQueryPage` contract
- Preact state/rendering
- continuation argument forwarding
- session ref URL encoding
- exhaustion/disabled state
- live status output

Anti-vacuity mutations that make it fail:

- replace `loadPage(continuation ?? undefined)` with an initial query or reconstructed offset;
- drop the opaque token argument;
- stop appending `page.items`;
- ignore `page.continuation`;
- remove the rendered session link or change its encoded ref;
- leave the button enabled after a `null` continuation.

Clean applied-checkout result:

```text
Test Files  1 passed (1)
Tests       1 passed (1)
```

Commands:

```sh
npm ci --prefix webui
npm run typecheck --prefix webui
npm test --prefix webui -- --run
```

`npm ci` installed 249 packages and reported zero vulnerabilities. `npm audit --prefix webui --audit-level=low` also reported zero vulnerabilities.

## Daemon production-route proof

Test: `tests/unit/daemon/test_web_reader.py::TestWebUIV2::test_app_serves_semantic_ssr_and_manifest_hashed_assets`

Production dependencies exercised:

- `DaemonAPIHTTPServer` and `DaemonAPIHandler`
- host/bootstrap route dispatch for `/app`
- `_web_reader_archive_root()`
- `load_archive_overview_page()`
- `QueryTransaction` and `query_unit_envelope()`
- Python semantic SSR
- `WebUIAssetBundle` manifest discovery
- packaged generated JavaScript/CSS
- WebUI CSP and no-store response policy
- immutable asset cache headers
- SHA-256 ETag generation
- conditional `304 Not Modified`
- manifest allowlisting and manifest privacy

Assertions include:

- HTTP 200 from `/app` against the seeded archive;
- semantic `<h1>` and archive list exist;
- seeded text `Hello reader` is present in server HTML;
- the island mount exists;
- HTML contains no `http://` or `https://` runtime dependency;
- script and stylesheet URLs are content-hashed and under `/app/assets/`;
- JavaScript and CSS return expected content types and immutable cache policy;
- conditional ETag request returns 304 with matching ETag/cache policy;
- `/app/assets/manifest.json` returns 404 rather than exposing build metadata.

Anti-vacuity mutations that make it fail:

- remove the `/app` dispatch or route contract;
- replace the shared query call with an empty/static page;
- remove the manifest entry lookup;
- serve arbitrary files from the static directory;
- expose `manifest.json`;
- remove content hashes, immutable caching, ETags, or conditional 304 handling;
- add an external runtime asset URL;
- make the first list client-only.

## Opaque continuation route proof

Test: `tests/unit/daemon/test_web_reader.py::TestReaderQueryUnits::test_query_units_endpoint_replays_opaque_continuation`

Production dependencies exercised:

- `GET /api/query-units`
- `QueryContinuation.decode()`
- operation and argument validation
- replayed expression/session filters/page size/offset
- `query_unit_request()`
- `query_unit_envelope()`
- stable `query_ref` and `result_ref`

The test asks for a one-row first page, sends only the returned `q1.*` token on the second request, and proves that the second page preserves identity, limit, and progression while returning a different message.

Anti-vacuity mutations that make it fail:

- ignore the `continuation` parameter;
- derive offset from browser query parameters;
- reset page size to the endpoint default;
- lose canonical query arguments;
- emit a new query/result identity for page two;
- return the first row again.

## Route publication and admission proofs

Tests in `tests/unit/daemon/test_route_contracts.py` verify that `/app` and `/app/assets/:asset` are published, resolve to the declared browser-shell contract, remain unauthenticated only for loopback shell bootstrap, and require the configured token when bound non-loopback.

Anti-vacuity mutations that make them fail:

- omit either route from `implemented_daemon_route_patterns()` or `ROUTE_CONTRACTS`;
- dispatch `/app` behind ordinary API auth on local loopback;
- bypass token admission on non-loopback.

## Focused Python execution

Final clean applied-checkout command:

```sh
/mnt/data/polylogue_repo/.venv/bin/pytest -q \
  tests/unit/daemon/test_route_contracts.py \
  tests/unit/daemon/test_web_reader.py \
  -k 'TestReaderQueryUnits or TestWebUIV2 or route_contract'
```

Result:

```text
112 passed, 140 deselected in 8.21s
```

The same slice also passed in the implementation checkout before patch generation.

## Formatting and static checks

Commands:

```sh
ruff format --check \
  polylogue/daemon/webui.py \
  polylogue/daemon/http.py \
  polylogue/daemon/route_contracts.py \
  tests/unit/daemon/test_route_contracts.py \
  tests/unit/daemon/test_web_reader.py

ruff check \
  polylogue/daemon/webui.py \
  polylogue/daemon/http.py \
  polylogue/daemon/route_contracts.py \
  tests/unit/daemon/test_route_contracts.py \
  tests/unit/daemon/test_web_reader.py

mypy --follow-imports=skip polylogue/daemon/webui.py
git diff --check
```

Results:

```text
5 files already formatted
All checks passed!
Success: no issues found in 1 source file
```

A direct `mypy --follow-imports=skip polylogue/daemon/http.py` comparison against the clean snapshot reports the same 12 existing diagnostics in both trees, with line numbers shifted by the new code. They are existing `no-any-return` and `unused-ignore` findings; no new category or changed-code diagnostic appears. An import-following `mypy polylogue/daemon/http.py` also reaches five unrelated existing diagnostics in `archive/query/predicate.py`, `storage/embeddings/materialization.py`, `storage/raw_retention.py`, and `daemon/uds.py`.

## Deterministic build proof

Clean applied-checkout commands:

```sh
before=$(sha256sum polylogue/daemon/static/dist/* | sort -k2)
npm run build --prefix webui
after=$(sha256sum polylogue/daemon/static/dist/* | sort -k2)
test "$before" = "$after"
git diff --exit-code -- polylogue/daemon/static/dist
```

Resulting hashes:

```text
6ccc25451ac6466539ab63006bcf519c9acf2e702b0ba0fbd4f18f10f0dae627  polylogue/daemon/static/dist/archive-overview-Ce5VissO.css
56858839cdba4ddb600d855b9e1904fb2494030f6fcc7621681751097651c816  polylogue/daemon/static/dist/archive-overview-CxKzwy2z.js
cf2c9bafe581d6faf4bec59a1f481e0e9a4dff3fa9b284411ab2c3d9e36409e3  polylogue/daemon/static/dist/manifest.json
```

The before/after strings were equal and the Git drift check passed.

## Patch application proof

Commands:

```sh
git worktree add --detach /mnt/data/polylogue_applycheck \
  536a53efac0cbe4a2473ad379e4db49ef3fce74d
git -C /mnt/data/polylogue_applycheck apply --check PATCH.diff
git -C /mnt/data/polylogue_applycheck apply PATCH.diff
git -C /mnt/data/polylogue_applycheck diff --check
```

All commands passed. Frontend and Python checks above were then run from that applied checkout, not only from the authoring tree.

## Distribution proof

Command:

```sh
uv build --clear \
  --out-dir /mnt/data/polylogue-applycheck-build \
  --default-index "$UV_INDEX_URL" \
  --index-strategy unsafe-best-match
```

Result:

```text
Successfully built polylogue-0.2.0.tar.gz
Successfully built polylogue-0.2.0-py3-none-any.whl
```

Inspected members in both artifacts:

```text
archive-overview-Ce5VissO.css   3,372 bytes
archive-overview-CxKzwy2z.js   18,134 bytes
manifest.json                     256 bytes
```

Artifact sizes:

```text
polylogue-0.2.0-py3-none-any.whl  4,185,796 bytes
polylogue-0.2.0.tar.gz            33,970,578 bytes
```

The member set and byte sizes matched exactly in wheel and sdist.

## Topology generation

Commands:

```sh
python -m devtools render topology-projection
python -m devtools render topology-status
```

The projection reported 1,024 rows and added the new `polylogue/daemon/webui.py` module to the generated topology artifacts.

## Incomplete or unverified execution

A full invocation of `pytest -q tests/unit/daemon/test_web_reader.py` exceeded an external 600-second command budget after 59 progress dots and no displayed failure. It is recorded as incomplete, not passing.

The following were not available or not claimed:

- Nix build/evaluation (`nix` executable unavailable);
- Playwright/browser install and E2E execution;
- a live operator archive or long-running daemon;
- a reverse proxy or non-loopback authenticated browser journey;
- NixOS/deployment smoke;
- repository-wide full test and full mypy matrices.

The existing CI workflow remains responsible for Playwright, and the added Nix post-fixup assertion should be exercised by the repository’s normal Nix lane.

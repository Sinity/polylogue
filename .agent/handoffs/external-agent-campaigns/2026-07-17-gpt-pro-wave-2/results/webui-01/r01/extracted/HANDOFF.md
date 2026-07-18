# Polylogue WebUI v2 workspace scaffold — implementation handoff

Revision: `webui-01-workspace-scaffold-r01`

## Outcome

This package implements one coherent SSR-first WebUI vertical against the supplied Polylogue snapshot. It adds a strict TypeScript + Preact + Vite workspace, a daemon-owned `/app` strangler mount, manifest-governed content-hashed assets, an archive-overview island that advances only through the existing opaque continuation contract, wheel/sdist/Nix packaging seams, and production-route tests. The existing Python-embedded web shell remains in place.

The first route is intentionally a bounded recent-message overview rather than an invented session-pagination API. The current canonical `/api/query-units` contract pages terminal evidence units, and `QueryTransaction` already owns query identity, bounded pages, and opaque continuations. The daemon renders the first page server-side through that substrate, while Preact hydrates only the continuation control and appends subsequent typed pages.

## Snapshot identity and authority

The Chisel overview identifies the source as `/realm/project/polylogue`, generated at `2026-07-17T180950Z`, on branch `master` at commit `536a53efac0cbe4a2473ad379e4db49ef3fce74d` (`fix(repair): harden raw authority convergence (#3046)`). `origin/master` and the merge base are the same commit, and the exported branch delta contains no commits or changed files.

The overview says `dirty=true`, but a clean checkout of the named commit overlaid with the exported `polylogue-working-tree.tar.gz` has no tracked or untracked Git delta. Therefore there is no reproducible tracked dirty patch to preserve. The most plausible explanation is ignored/local runtime state that the snapshot metadata observed but the working-tree export intentionally omitted. This patch is prepared directly against the named commit.

Authority order used here was current source, repository instructions, complete relevant Beads records, and then history. No API or parallel query layer was invented to satisfy the scaffold.

## Mechanism

`polylogue/daemon/webui.py` is the narrow production seam. `WebUIAssetBundle` discovers `polylogue.daemon/static/dist` through `importlib.resources` (or a test override), parses Vite’s manifest, requires a named `archive-overview` entry, permits only flat content-hashed JavaScript/CSS names declared by that manifest, computes SHA-256 ETags, and refuses all other requested files. The manifest itself is never public.

`load_archive_overview_page()` executes the bounded expression `messages where words >= 0 | sort by time desc` with a page size of six through `QueryTransaction` and `query_unit_envelope`. `render_archive_overview_page()` emits semantic HTML containing the first result page, normal links to the existing `/s/:session_id` reader, escaped bootstrap JSON, local stylesheet/script references, and a continuation island. The page remains useful without JavaScript.

`polylogue/daemon/http.py` publishes `/app` and `/app/assets/:asset` under the existing shell bootstrap admission policy. `/app` returns no-store HTML with a restrictive CSP and no inline executable script. Assets receive one-year immutable caching, SHA-256 ETags, conditional `304 Not Modified`, `nosniff`, and same-origin resource policy. Missing/unlisted asset names return `404`; a missing or corrupt packaged build returns a service-unavailable state.

The existing `/api/query-units` route now accepts the opaque `continuation` already emitted by `query_unit_envelope`. On replay, the daemon decodes the canonical request, verifies the operation and argument shape, and uses the continuation’s expression, session filters, page size, and offset. Browser code never reconstructs filters or offsets. Initial query behavior remains unchanged.

The browser workspace validates the current message query envelope at runtime before rendering it. `requestJson()` centralizes first-party HttpOnly credential bootstrap and same-origin fetch behavior. `fetchArchiveMessagePage()` sends the initial expression/limit only for the first page and sends only `continuation` thereafter. The island owns no router or archive semantics; it owns loading state, status text, and rendering of appended rows.

## Exact tree changed

Added production and packaged assets:

```text
polylogue/daemon/webui.py
polylogue/daemon/static/dist/archive-overview-Ce5VissO.css
polylogue/daemon/static/dist/archive-overview-CxKzwy2z.js
polylogue/daemon/static/dist/manifest.json
```

Added workspace files:

```text
webui/README.md
webui/index.html
webui/tsconfig.json
webui/vite.config.ts
webui/src/contracts/runtime.ts
webui/src/contracts/query-units.ts
webui/src/entrypoints/archive-overview.tsx
webui/src/islands/archive-overview.tsx
webui/src/islands/archive-overview.test.tsx
webui/src/lib/api.ts
webui/src/styles.css
webui/src/test/setup.ts
```

Modified integration, packaging, generated topology, and tests:

```text
.github/workflows/ci.yml
.gitignore
docs/plans/topology-target.yaml
docs/topology-status.md
flake.nix
polylogue/daemon/http.py
polylogue/daemon/route_contracts.py
pyproject.toml
tests/unit/daemon/test_route_contracts.py
tests/unit/daemon/test_web_reader.py
webui/package.json
webui/package-lock.json
```

## Daemon seam diff summary

The daemon changes are deliberately bounded:

1. Add route declarations for `GET /app` and `GET /app/assets/:asset`.
2. Add an optional `webui_dist_root` server constructor argument used only to make the real manifest/asset dependency controllable in tests.
3. Dispatch the new routes before the legacy shell routes, using the same loopback/bootstrap access decision.
4. Add WebUI-specific HTML and immutable-asset send helpers.
5. Add SSR and asset handlers that delegate all build and rendering behavior to `polylogue/daemon/webui.py`.
6. Add opaque continuation replay to the existing `GET /api/query-units` route, matching the shared query transaction contract already used by the MCP surface.

No existing shell route or shell module is deleted, redirected, or replaced.

## Build, test, and development commands

Use Node 22 and the committed npm lockfile:

```sh
cd webui
npm ci
npm run typecheck
npm test
npm run build
npm audit --audit-level=low
```

The CI-shaped build can be reduced to the requested form:

```sh
cd webui
npm ci && npm test && npm run build
cd ..
git diff --exit-code -- polylogue/daemon/static/dist
```

For development, run the daemon normally and start Vite:

```sh
cd webui
POLYLOGUE_DAEMON_URL=http://127.0.0.1:8787 npm run dev
```

Vite listens on `127.0.0.1:5173`, serves the development island, and proxies `/api/*` to the daemon. The production route remains daemon-owned at `http://127.0.0.1:8787/app`.

The generated production bundle is deterministic in the verified environment:

```text
6ccc25451ac6466539ab63006bcf519c9acf2e702b0ba0fbd4f18f10f0dae627  archive-overview-Ce5VissO.css
56858839cdba4ddb600d855b9e1904fb2494030f6fcc7621681751097651c816  archive-overview-CxKzwy2z.js
cf2c9bafe581d6faf4bec59a1f481e0e9a4dff3fa9b284411ab2c3d9e36409e3  manifest.json
```

A clean applied checkout produced these same bytes before and after `npm run build`.

## Packaging

The built files live inside the Python package at `polylogue/daemon/static/dist`. Hatch’s normal wheel package inclusion carries them into the wheel. An explicit sdist `force-include` ensures the committed build enters the source distribution so rebuilding a wheel from the sdist preserves the assets.

A clean `uv build` produced both artifacts and verified all three files inside each:

```text
polylogue-0.2.0-py3-none-any.whl  4,185,796 bytes
polylogue-0.2.0.tar.gz            33,970,578 bytes
```

The Nix derivation’s `postFixup` now asserts that `${python.sitePackages}/polylogue/daemon/static/dist/manifest.json` exists. This preserves the existing no-Node deployment chain: Node is required in development/CI to reproduce committed assets, not during wheel/Nix deployment.

The `nix` executable is unavailable in the execution environment, so the Nix derivation itself was not built. The package path assertion is source-reviewed but remains an external verification item.

## What the next seven vertical jobs should reuse

### 1. Session list

Import `requestJson` from `webui/src/lib/api.ts` and the primitive validators from `webui/src/contracts/runtime.ts`. Add a session-list-specific runtime validator under `webui/src/contracts/`; do not reinterpret `query_units` as session paging. Render the first bounded list page in Python through the canonical session-list owner, then hydrate only list controls in a new island. Reuse the `/app` asset manifest and `WebUIAssetBundle`; add another Vite entry only when the route needs independent code splitting.

### 2. Search

Reuse `requestJson`, endpoint-local runtime validation, the same first-party credential bootstrap, and the continuation-only replay pattern demonstrated by `fetchArchiveMessagePage()`. Keep the query expression and filter semantics in the existing archive/query substrate. Search UI state belongs in its feature/island; it must not introduce a browser-side query planner.

### 3. Transcript

Reuse the manifest/entrypoint pattern, semantic Python SSR, `requestJson`, and normal existing `/s/:session_id` refs. The transcript vertical should render useful message structure server-side and hydrate interactive tool blocks, outline controls, selection, or annotation affordances as islands. It should not port the legacy shell wholesale into one client-only root.

### 4. Insights/status

Reuse the same daemon route publication, no-store SSR headers, endpoint validators, and shared request helper. Keep readiness/error states typed by their current payload owners. The first page should expose useful semantic status without requiring JavaScript; live refresh can later compose through one shared SSE module rather than per-view subscriptions.

### 5. Cost

Reuse endpoint-local validators and `requestJson`. Preserve evidence-resolution: cost summaries must link or expand to their canonical usage evidence rather than calculating a parallel browser total. SSR the bounded initial summary; hydrate drill-down controls.

### 6. Design system

Reuse `webui/src/styles.css` only as the temporary system-color/base-layout layer. Do not hand-author `lib/tokens.css`: the ratified Bead rider requires that file to be generated from `theme.py` under `polylogue-9xuk`. The design-system vertical should replace or layer over the current base rules with generated tokens and shared components, retaining the no-CDN/system-font posture.

### 7. Client contracts

Import the reusable validators from `webui/src/contracts/runtime.ts` and `requestJson` from `webui/src/lib/api.ts`. Replace hand-maintained endpoint interfaces with the repository’s eventual generated OpenAPI client rather than creating a second schema source. Preserve runtime boundary checks even after type generation. `query-units.ts` is the concrete compatibility target and can be deleted once generated types and validators cover the same payload and continuation behavior.

## Eventual legacy-shell deletion boundary

No file below is deleted by this patch. Once `/`, `/s/:session_id`, `/w/:mode`, `/p`, and `/a` have verified semantic/interaction parity on WebUI v2, the presentation boundary to retire is:

```text
polylogue/daemon/web_shell.py
polylogue/daemon/web_shell_attachments.py
polylogue/daemon/web_shell_coordination.py
polylogue/daemon/web_shell_lineage.py
polylogue/daemon/web_shell_paste.py
polylogue/daemon/web_shell_provenance.py
polylogue/daemon/web_shell_reader.py
polylogue/daemon/web_shell_realtime.py
polylogue/daemon/web_shell_selection.py
polylogue/daemon/web_shell_semantic_cards.py
polylogue/daemon/web_shell_similar.py
polylogue/daemon/web_shell_workspace.py
```

That deletion cannot be mechanical. `http.py` currently imports reusable attachment/paste helpers from two satellites, and tests import several constants/builders. Before removal, non-presentation contracts must move to durable owners and their tests must follow. The deletion PR should also remove legacy-shell visual/unit tests only after equivalent v2 route and interaction proofs exist.

## Decisions an integrator may overturn

The following are implementation choices, not new product law:

- `/app` is the v2 mount and `/app/assets/:asset` is the local bundle route, following the ratified strangler direction.
- The first overview page is six recent message evidence rows using `messages where words >= 0 | sort by time desc` because that is a real canonical pageable contract today.
- SSR uses a small dedicated Python renderer rather than introducing a template engine. A later shared render abstraction may replace it once more routes establish repeated structure.
- The Preact island hydrates only continuation controls; it does not own the initial list, routing, or query semantics.
- Vite emits flat content-hashed files and the daemon rejects nested/non-hashed output. Future image/font assets may require expanding the allowlist while preserving manifest governance and no external runtime fetches.
- Committed `dist` plus CI drift detection is used instead of running Node in Python/Nix deployment builds.
- Browser payload validation is currently hand-maintained and intentionally small; the client-contract vertical should replace it with generated OpenAPI-derived contracts.
- The static asset route shares shell bootstrap admission. Remote authenticated browser deployment behavior should be checked in the operator’s actual reverse-proxy/auth setup.
- Node 22, npm 10.9.2, and exact dependency pins are selected for reproducibility. Renovation can move them as one tested lockfile change.
- The base stylesheet uses system colors/fonts and no hand-authored design-token contract while generated tokens are pending.

## Acceptance matrix

| Requirement | Implementation | Status |
|---|---|---|
| TypeScript + Preact + Vite workspace | Strict TS config, exact npm lock, Preact/Vite/Vitest scripts | Verified |
| No runtime CDN/network assets | Only same-origin `/app/assets` and `/api` calls; no external font/script/style references | Verified by source/HTML inspection |
| Semantic daemon SSR | `/app` renders first bounded message page and normal links | Verified by live route test |
| Progressive islands, not SPA-only | First list is server HTML; island owns continuation only | Verified by route and component tests |
| Typed existing JSON contract | Runtime-validated `QueryUnitEnvelope`; no new endpoint | Verified |
| Opaque continuation paging | Browser sends only token; HTTP replays canonical request | Verified by unit and route tests |
| Hashed immutable assets | Vite manifest allowlist, SHA ETag, immutable cache, conditional 304 | Verified |
| Wheel/sdist inclusion | Both built and inspected for exact asset members | Verified |
| Nix packaging note/check | `postFixup` manifest assertion added | Source-verified; Nix build unverified |
| CI build command | `npm ci`, Vitest, build, committed-dist drift check | Implemented |
| Vite daemon proxy | `/api` proxy configurable through `POLYLOGUE_DAEMON_URL` | Source-verified |
| Vitest smoke | Opaque token, typed row, link, exhaustion behavior | Passed |
| Python route proof | Seeded archive SSR, CSP, asset transport, ETag/304, manifest privacy | Passed |
| Existing shell retained | No legacy shell deletions or redirects | Verified by patch inspection |
| Topology regeneration | Projection and status regenerated after adding `webui.py` | Verified |

## Apply order

1. Check out commit `536a53efac0cbe4a2473ad379e4db49ef3fce74d` with a clean tracked worktree.
2. Run `git apply --check PATCH.diff`, then `git apply PATCH.diff`.
3. In `webui/`, run `npm ci`, `npm run typecheck`, `npm test`, and `npm run build`.
4. From the repository root, run `git diff --exit-code -- polylogue/daemon/static/dist`.
5. Run the focused Python commands in `TESTS.md`.
6. Build wheel/sdist and inspect the three packaged assets.
7. Run the repository’s normal Nix and browser/Playwright lanes in an environment that has those dependencies.
8. Land subsequent verticals behind `/app` while keeping the old routes until parity evidence supports one independent deletion change.

## Risks and limitations

The full `tests/unit/daemon/test_web_reader.py` file was attempted but exceeded a 600-second external command budget after printing 59 passing progress dots and no failure. The focused production slice is broad—112 tests—and passes, but it is not a substitute for the repository’s complete CI matrix.

No real operator archive, long-running daemon, browser session, reverse proxy, NixOS deployment, or secrets were available. The route test uses the repository’s seeded archive and real HTTP server. Playwright was not run because the mission requires the Vitest smoke and Python route proof; the existing CI job still owns browser installation and first-party-auth E2E.

`mypy --follow-imports=skip polylogue/daemon/webui.py` passes. `http.py` retains the same 12 pre-existing direct-file diagnostics as the clean snapshot under that mode, shifted only by inserted lines. A normal import-following check also reaches unrelated pre-existing errors in archive/storage/UDS modules. This patch introduces no diagnostic in `webui.py`, but it does not repair repository-wide typing debt.

The current route is foundation evidence, not list/reader parity and not the complete `polylogue-bby.11` epic. It intentionally omits SSE cache, palette, generated tokens, command routing, virtualized session list, transcript parity, and generated client contracts; those are separate verticals named by the mission.

## Value of another iteration

For this scaffold itself, another pass is most likely a small repair pass driven by CI, Nix, or authenticated remote-browser evidence. The implementation is apply-checked, route-tested, package-tested, and deterministic locally.

A substantial second pass would add a real vertical rather than more scaffold: the highest-value choices are generated OpenAPI client contracts plus the session-list route, or a Playwright semantic/no-JavaScript/browser-auth proof against a packaged daemon. Either would materially expand acceptance; neither is required to make the current patch coherent.

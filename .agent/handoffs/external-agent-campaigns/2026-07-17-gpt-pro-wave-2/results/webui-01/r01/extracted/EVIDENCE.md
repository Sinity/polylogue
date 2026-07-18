# Source, tracker, and history evidence

## Snapshot evidence

Observed:

- `polylogue-overview.md` in the supplied archive reports generation at `2026-07-17T180950Z` from `/realm/project/polylogue`.
- It names `master` at `536a53efac0cbe4a2473ad379e4db49ef3fce74d` and labels the source dirty.
- `polylogue-branch-delta.md` names `origin/master`, the same merge base, an empty diff stat, and no branch commits.
- The named commit subject is `fix(repair): harden raw authority convergence (#3046)`.
- Overlaying the exported working-tree tar onto a clean checkout of that commit yields an empty `git status` and empty tracked diff.

Inference:

- The `dirty=true` signal was likely caused by ignored/local runtime state not represented in the exported working-tree payload. There is no reproducible tracked dirty patch to preserve.

Unresolved:

- The exact ignored file(s) that caused the snapshot generator’s dirty flag are not present in the supplied working-tree export and therefore cannot be identified.

## Repository instructions and architecture

`CLAUDE.md` was inspected for repository laws, generated-surface rules, query vocabulary, and topology obligations. Its relevant hard requirement is that adding any module under `polylogue/` requires regenerating and committing `docs/plans/topology-target.yaml` and `docs/topology-status.md` via the topology render commands. This patch does so for `polylogue/daemon/webui.py`.

`docs/architecture.md` defines the daemon as a ring-3 surface and states that surfaces are leaf adapters over the same archive/query substrate rather than owners of stores or parallel semantics. It identifies the current daemon web reader as `polylogue/daemon/web_shell.py` and says new meaning belongs in substrate/insight owners first, then surfaces adapt it.

Implementation consequence:

- The new page calls `QueryTransaction` and `query_unit_envelope`; it does not add a WebUI query engine, session store, or browser-owned filtering model.
- Browser code validates and renders the canonical payload but does not decide query semantics.

## Daemon HTTP and authentication evidence

Inspected production files and symbols:

- `polylogue/daemon/http.py`
  - `implemented_daemon_route_patterns()`
  - `DaemonAPIHandler._do_get_impl()`
  - `_check_host_admission_logic()`
  - `_check_shell_bootstrap_access()`
  - `_check_auth()`
  - `_send_json()`
  - `_serve_web_shell()`
  - `_handle_query_units()`
  - `DaemonAPIHTTPServer`
- `polylogue/daemon/route_contracts.py`
  - `ROUTE_CONTRACTS`
  - route matching/publication helpers
- `polylogue/daemon/web_auth.py`
  - first-party cookie issuance/validation and exact-origin behavior
- Existing browser tests under `webui/tests/first-party-auth.spec.ts` and daemon route/auth tests.

Observed:

- The legacy shell bootstrap is unauthenticated only for loopback deployment/client pairs; non-loopback shell access delegates to machine-token auth.
- API JavaScript obtains a protected first-party credential through `/api/web-auth/session` and sends `X-Polylogue-Web-Client: 1`.
- GET dispatch already separates shell bootstrap from scoped JSON routes.
- `GET /api/query-units` returned a typed terminal-unit payload but did not accept its own emitted continuation token over HTTP.

Implementation consequence:

- `/app` and its subresources use the existing shell bootstrap posture rather than a new auth system.
- The new client uses the existing first-party web credential bootstrap before JSON requests.
- HTTP continuation replay was added at the canonical endpoint rather than hidden in the island.

## Query transaction and payload evidence

Inspected:

- `polylogue/archive/query/transaction.py`
  - `QueryTransactionRequest`
  - `QueryContinuation`
  - `QueryResultPage`
  - `QueryTransaction`
- `polylogue/archive/query/unit_results.py`
  - `query_unit_request()`
  - `query_unit_rows()`
  - `query_unit_envelope()`
- `polylogue/archive/query/execution_control.py`
  - workload classification and bounded execution
- `polylogue/surfaces/payloads.py`
  - `MessageQueryRowPayload`
  - `QueryUnitEnvelope`
- MCP continuation handling in `polylogue/mcp/server_tools.py` and response-budget rebasing in `polylogue/mcp/server_support.py`.

Observed:

- `QueryContinuation` carries the canonical operation, arguments, page size, offset, projection, stable order, and result reference in an opaque `q1.*` token.
- `query_unit_envelope()` already emits `query_ref`, `result_ref`, and continuation for a bounded page.
- `query_units` is a terminal evidence-unit query, not a session-list endpoint. Message rows contain stable message/session IDs, origin, role/type, timestamp, position, word count, and text.
- MCP already decoded continuations to recover query identity, page size, and offset. The HTTP surface was the parity gap needed by a browser continuation island.

Contradiction adjudicated:

- The mission permits a minimal archive-overview page. Calling the page a “session list” and paging it through `query_units` would be false to current source. The implementation instead renders recent message evidence and leaves canonical session listing to the dedicated next vertical.

## Beads evidence

### `polylogue-bby.11` — WebUI architecture v2

Observed current record:

- Chooses TypeScript + Preact + Vite for familiarity, typing, componentization, testing, and small runtime.
- Requires Vite development proxying to the daemon.
- Requires committed assets under `polylogue/daemon/static/dist`, CI reproducibility checks, and wheel/Nix shipment without Node in deployment builds.
- Requires a strangler mount at `/app`, retaining the old shell until parity.
- Later corpus refinement explicitly tightens the architecture to daemon-served semantic HTML plus typed JSON on every route, with Preact as progressive-enhancement islands rather than SPA-or-nothing.
- The ratified 2026-07-08 rider confirms the stack, Vitest per-PR, browser/visual lanes later, and requires `lib/tokens.css` to be generated from `theme.py`, never hand-authored.
- The recovered cockpit ruling forbids importing a static prototype or proposed envelope as a parallel contract.

Implementation consequence:

- No `tokens.css` was invented. The temporary stylesheet uses system colors/fonts and is explicitly handed to the design-system vertical for replacement by generated tokens.
- No recovered/prototype envelope was introduced.
- The existing `webui/` Playwright workspace was extended rather than creating a second frontend root.

Important scope distinction:

- The full Bead acceptance criterion includes generated client, SSE/cache, palette/routing, list/reader parity, old-shell retirement, and an agent-buildability proof. The mission here requests the scaffold and first route, not closure of the entire Bead. This package does not claim `polylogue-bby.11` is complete.

### `polylogue-t46.8` and children — protocol-native read algebra

Observed current records:

- `polylogue-t46.8` requires a small expressive read algebra backed by shared transactions, explicit result semantics, useful bounded pages, logical completeness, and no surface-local parser/query/policy.
- `polylogue-t46.8.1` requires declarations to state exhaustive/ranked/sample/aggregate/context/graph semantics, canonical plans, continuation/ref behavior, and replacement routes.
- `polylogue-t46.8.2` requires reads to preserve bounded physical pages with unbounded logical enumeration, stable refs/cursors, useful first-page evidence, cancellation, and no full-result adapter buffer or metadata-only refusal.
- Existing list/search aliases may be removed only after equivalence and cold-model proof.

Implementation consequence:

- The page starts with useful bounded evidence, not a metadata-only shell.
- Continuation is the existing transaction token and is replayed by the daemon.
- The browser never reconstructs the query or accumulates an unbounded logical result in one response.
- No existing API alias or tool is removed.

## Packaging and build evidence

Inspected:

- `pyproject.toml` Hatch wheel/sdist configuration
- `packaging/hatch_build.py`
- `flake.nix`
- `.github/workflows/ci.yml`
- `.gitignore`
- existing `webui/package.json`, lockfile, Playwright config, and auth E2E.

Observed:

- Hatch’s wheel target already packages the `polylogue` package tree, so adding an additional wheel force-include duplicates files and is rejected by the build. The final patch adds only the sdist force-include; clean wheel and sdist builds prove the result.
- The existing web CI job already installs Node 22, runs `npm ci`, installs Chromium, and executes Playwright. The scaffold inserts Vitest/build/drift checks before the existing E2E steps.
- The Nix derivation installs the Python package; a post-fixup file assertion is a narrow way to fail if the committed build is lost.

Decision:

- Commit the generated bundle and use CI drift detection. Do not add Node to the Nix/Python deploy build.

## History evidence

Relevant commits inspected:

- `9163d0134 feat(query): bound agent-facing archive reads (#3018)` introduced `polylogue/archive/query/transaction.py`, continuation/result identity, bounded transport work, and cross-surface tests. This is the current substrate the page consumes.
- `ed44be18f feat(mcp): declare the current tool algebra (#3004)` introduced declaration/equivalence infrastructure and reinforces the no-parallel-surface-contract direction.
- `0e0cddaee fix(web): bootstrap first-party daemon credentials (#2715)` is the browser-auth seam reused by `requestJson()`.
- `5c43673fc chore(beads): activation layer, webui v2, pref bundles, pace visibility` is earlier tracker history, but current Beads notes and current source supersede stale path/API assumptions.

## Existing legacy-shell boundary

Observed files:

- `polylogue/daemon/web_shell.py`
- `web_shell_attachments.py`
- `web_shell_coordination.py`
- `web_shell_lineage.py`
- `web_shell_paste.py`
- `web_shell_provenance.py`
- `web_shell_reader.py`
- `web_shell_realtime.py`
- `web_shell_selection.py`
- `web_shell_semantic_cards.py`
- `web_shell_similar.py`
- `web_shell_workspace.py`

Observed dependencies:

- `http.py` imports attachment and paste helpers directly.
- Unit and visual tests import the shell and several satellite constants/helpers.

Conclusion:

- The eventual deletion boundary is identifiable, but some reusable non-presentation code must be extracted first. Deleting those files in this scaffold would violate both the mission and current call sites.

## Runtime/network evidence

Source inspection found no external URL, font, telemetry, or CDN call in the new WebUI source or server HTML. The built Preact runtime contains standard XML namespace literals such as `http://www.w3.org/2000/svg`; these are DOM namespace identifiers, not network requests. Actual fetch calls are only same-origin `/api/web-auth/session` and `/api/query-units`.

## Remaining uncertainty

- Nix file placement has not been executed because `nix` is unavailable.
- The operator’s non-loopback/reverse-proxy browser auth path was not available. The new static route deliberately follows existing shell policy, but that policy’s deployment-specific behavior should be exercised in situ.
- The hand-maintained TypeScript contract may drift until the client-contract vertical generates it from the authoritative schema. Runtime validation and CI tests reduce but do not eliminate that risk.
- A full browser/no-JavaScript visual inspection was not executed. Semantic HTML is asserted at the HTTP level.

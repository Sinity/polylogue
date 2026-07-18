# Polylogue WebUI workspace

This workspace builds progressive Preact islands for daemon-served semantic HTML. It is not an SPA router and it makes no runtime requests to CDNs, font hosts, telemetry services, or other third parties.

## Reproducible build and tests

Use Node 22 and the committed npm lockfile:

```sh
cd webui
npm ci
npm test
npm run build
```

`npm run build` runs strict TypeScript checking and writes a Vite manifest plus content-hashed JavaScript/CSS into `polylogue/daemon/static/dist/`. Those generated files are committed because Python wheel, sdist, and Nix builds ship them without adding Node to the deployment build chain.

## Development mode

Run the daemon on its normal local address, then start Vite:

```sh
cd webui
POLYLOGUE_DAEMON_URL=http://127.0.0.1:8787 npm run dev
```

Open the Vite URL printed by the command. Vite serves the island source with hot reload and proxies `/api/*` to `POLYLOGUE_DAEMON_URL`. The production route remains `http://127.0.0.1:8787/app`; it reads the packaged manifest and serves the same island under `/app/assets/<content-hash>`.

## Contract boundary

`src/contracts/runtime.ts` owns the small reusable runtime-validation primitives; endpoint-specific validators live beside it. `src/lib/api.ts` exports `requestJson()` so every vertical shares same-origin web-credential bootstrap while validating its own typed payload. `src/contracts/query-units.ts` and `fetchArchiveMessagePage()` demonstrate opaque-continuation replay. Keep view-state types inside each island or feature directory.

## Test stack

Three lanes, per `polylogue-1ilk`:

- **Component/unit (vitest)** — `npm test` runs `src/**/*.test.tsx` against jsdom with `@testing-library/preact`: one file per island/design-system component, rendered against typed fixture payloads. Runs per-PR in CI (`web-first-party-auth` job) and is fast enough to run on every save with `npm run test:watch`.
- **End-to-end (Playwright)** — `npm run test:e2e` runs everything under `tests/*.spec.ts`:
  - `tests/design-system.spec.ts` drives the v2 verticals against an in-process synthetic fixture (`scripts/fixture-server.mjs`) — fast, deterministic, used for the vertical-contract/a11y/keyboard/SSR-without-JS checks.
  - `tests/first-party-auth.spec.ts` boots the real daemon (`tests/browser/web_auth_server.py`) over a demo archive to exercise the legacy shell's web-credential lifecycle (bootstrap, expiry, revocation, cross-origin rejection).
  - `tests/webui-v2-demo.spec.ts` boots the real daemon (`tests/browser/webui_v2_demo_server.py`) over a demo archive — **never the live archive** — extended with one synthetic 90-message session (the demo corpus alone has nothing large enough to page through), and drives the full v2 journey: list → open session → deep link → search → paginate. This is what proves `read_archive_session_page` (polylogue-07g6) is wired correctly end to end, not just at the unit level.
  - The same file's `visual: *` tests screenshot the three shipped v2 views (session list, session read, search) at 375/768/1440px against committed baselines under `tests/snapshots/webui-v2-demo.spec.ts/`.

  Run a single file locally with `npx playwright test tests/webui-v2-demo.spec.ts`, or `-g '<test name>'` to filter. `npm run test:e2e` alone (with no prior `npm run test:client-contracts` run) currently fails on an unrelated pre-existing issue — Playwright's default glob also picks up `tests/unit/client-contracts.test.mjs`, which needs `.cache/client-test/` populated by `test:client-contracts` first; CI's job ordering runs that step earlier so it never surfaces there.

**Re-baselining visual snapshots:** after an intentional visual change, rebuild the daemon-served assets and update baselines together — a stale `polylogue/daemon/static/dist/` makes the "regression" a rebuild artifact, not a real diff:

```sh
npm run build
npm run test:e2e:update           # design-system.spec.ts baselines
npx playwright test tests/webui-v2-demo.spec.ts -g "visual:" --update-snapshots
```

Review the diff like a syrupy snapshot re-baseline PR: a dedicated `fix(test):`/`chore(test):` commit, never folded into an unrelated feature change.

**`tests/visual` retirement (pytest DOM-smoke, not this workspace):** `tests/visual/test_reader_*.py` covers the legacy shell (`/`, `/w/stack`, `/w/compare`, `/s/:id`, `/p`, `/a`), not the `/app` verticals here. It retires surface-by-surface as each legacy view is replaced by a `/app` vertical with equivalent Playwright coverage: session list/read/search now have that coverage via `webui-v2-demo.spec.ts` above, but the legacy shell itself is still load-bearing (it remains the credential-bootstrap surface `first-party-auth.spec.ts` exercises, and `/w/stack`, `/w/compare`, `/p`, `/a` have no `/app` equivalent yet) — so no `tests/visual` file retires until the view it covers is actually replaced, not merely fronted by a v2 page that reads the same data.

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

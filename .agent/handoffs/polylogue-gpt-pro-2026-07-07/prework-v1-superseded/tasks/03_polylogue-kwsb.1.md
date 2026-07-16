# 03. polylogue-kwsb.1 — Daemon/capture Host, Origin, receiver-token, and spool hardening

Priority: **P1**  
Lane: **security**  
Readiness: **ready-now / code-local with extension smoke**

Depends on packet(s): polylogue-s7ae.6

## Why this is urgent / critical-path

The daemon and browser-capture receiver expose private archive data and live capture routes on localhost. Localhost is not a browser security boundary; DNS rebinding and same-host processes can reach it.

## Static diagnosis / likely mechanism

Root causes:
- Daemon GET dispatch has no central Host gate. `DaemonAPIHandler.do_GET` parses and dispatches directly (`polylogue/daemon/http.py:1294-1298`).
- Daemon token fallback accepts `?access_token=` for all paths when a token is configured (`http.py:1037-1055`), even though the comment only justifies EventSource.
- POST Origin checks are not centralized for GET, and missing Origin is accepted (`http.py:1301-1313`).
- Browser capture accepts no token when `auth_token is None`; the default receiver config sets `auth_token=None` (`polylogue/browser_capture/receiver.py:52-65`, `polylogue/browser_capture/server.py:68+`).
- Spool writing is bounded by filesystem success, not by receiver quota.

## Implementation plan

Patch in layers:
1. Add a central daemon request-admission helper called by GET/POST/DELETE before route dispatch. It should strip ports/brackets from `Host`, reject absent/malformed/foreign hosts, and allow only loopback/configured hostnames by default.
2. Keep shell bootstrap unauthenticated only for loopback deployments, but still Host-gated.
3. Restrict `access_token` query fallback to the exact SSE/EventSource route. If there is no current SSE route, remove the fallback and update tests/docs. Use `hmac.compare_digest` in token comparison.
4. Extend GET protection: API JSON routes should require configured bearer token and always pass Host gate. Browser-origin GETs from non-loopback origins should be rejected.
5. Browser capture: mint/load a receiver token at startup if none configured, persist it mode 0600, and require it for `/v1/browser-captures`. The extension should obtain it through the existing dev/install path, not from a public status endpoint.
6. Add receiver spool governor fields: `max_spool_bytes`, `max_spool_files`, and possibly `max_payload_bytes`. Enforce before committing a capture envelope; return a clear 413/507-style error and do not write partial files.

## Test plan

Tests:
- daemon GET `/api/sessions` with `Host: evil.example` is denied.
- daemon GET with allowed Host and valid bearer works.
- `?access_token=` is rejected for ordinary API GETs.
- POST cross-origin remains denied; legitimate extension/web shell flows still pass.
- browser-capture POST without token is denied by default.
- forged token denied; valid token accepted.
- spool quota rejects oversized/too-many captures without writing.
- extension/dev smoke proves the real extension can still capture with token.

## Verification command / proof

`devtools test tests/unit/daemon/test_daemon_http_security.py tests/unit/browser_capture -k 'host or origin or token or spool'` plus the existing browser extension smoke lane if present.

## Pitfalls

Do not rely on Origin alone; simple browser GETs can omit Origin. Do not leave a public endpoint that reveals the receiver token. Avoid breaking CLI/curl workflows; missing Origin is fine for non-browser clients only after Host/auth pass.

## Files/functions to inspect or touch

- `polylogue/daemon/http.py:1037-1055`
- `polylogue/daemon/http.py:1294-1359`
- `polylogue/browser_capture/receiver.py:52-65`
- `polylogue/browser_capture/server.py:54-90`
- `polylogue/browser_capture/server.py:259+`
- `browser-extension/src/background.js`

# 003. polylogue-kwsb.1 — Daemon/capture security hardening: Host/Origin gate, receiver token, spool governor

Priority/type/status: **P1 / bug / open**. Lane: **00-trust-floor**. Release: **A-trust-floor**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

Three confirmed holes (red-team, multiple independent confirmations): (1) DNS REBINDING reads the whole archive — GET routes have no Host check and Origin is checked only on POST and skipped when absent, so a malicious page resolving to 127.0.0.1 can read loopback HTTP; fix = ONE central Host/Origin allowlist middleware before dispatch (must admit the web shell own-origin — breaking same-origin shell is the named risk). (2) Browser-capture receiver has NO auth on loopback — any local process can POST forged captures into the spool; fix = auto-minted 0600 receiver token, hmac.compare_digest, restrict ?access_token= to the SSE route. (3) No spool quota — a runaway/hostile poster can fill disk; add a spool governor. Runtime+config only, no migration. Tier-0 credibility class. Verbatim spec: bundles/rnd-bundle-6-of-6.md L1802.

## Existing design note

All three holes live in polylogue/daemon/http.py: the Origin check exists only on the POST path (~L1305 headers.get Origin, skipped when absent) while GET routes (_static_get_routes ~L228, _parameterized_get_routes ~L257) have no Host/Origin gate at all — that is the DNS-rebinding read hole. Fix shape: one request-admission gate applied to EVERY route before dispatch — Host allowlist (127.0.0.1/localhost + configured), Origin required-and-matched on state-changing routes, capability token for the browser-capture receiver POSTs (_authenticated_post_routes ~L321 is the seam), and a spool-size governor in the receiver path (polylogue/daemon/browser_capture.py + spool writer) so a hostile page cannot disk-fill. Pitfall: the dev-loop and MCP localhost clients must keep working — gate by route class, not blanket; add regression tests per hole (rebinding GET, absent-Origin POST, spool flood).

## Acceptance criteria

Cross-origin GET with foreign Host is refused; unauthenticated capture POST refused; forged-token POST refused; web shell + extension keep working (fixture proof); spool bounded. Verify: daemon http tests + extension fixture.

## Static mechanism / likely defect

Daemon GET routes dispatch before a Host gate; Origin checks are POST-only and absent-Origin tolerant; browser-capture auth is optional on loopback; query-string tokens are broadly accepted; spool has only request-size bound.

## Source anchors to inspect first

- `polylogue/daemon/http.py:983` — _check_auth_logic uses direct equality and allows all when token is unset.
- `polylogue/daemon/http.py:1037` — _check_auth currently accepts query-string access_token broadly.
- `polylogue/daemon/http.py:1294` — do_GET dispatches without central Host/Origin admission.
- `polylogue/daemon/http.py:1301` — _check_cross_origin applies only to POST and allows absent Origin.
- `polylogue/browser_capture/receiver.py:45` — BrowserCaptureReceiverConfig defaults auth_token to None.
- `polylogue/browser_capture/server.py:54` — _origin_allowed accepts absent Origin.
- `polylogue/browser_capture/server.py:68` — _check_token accepts every request when auth_token is None and uses direct equality otherwise.
- `polylogue/browser_capture/server.py:47` — Only per-request max body exists; add spool file/count/bytes governor.

## Implementation plan

1. Add a single request-admission function called before GET/POST/DELETE dispatch. Strip port/brackets, allow loopback/configured hosts, reject foreign/absent malformed Host.
2. Make Origin/Referer policy explicit for browser-facing state-changing routes; absent Origin is not automatic trust when a browser route can be hit.
3. Generate or require a 0600 receiver token for browser capture; compare via `hmac.compare_digest`.
4. Restrict `?access_token=` to routes that genuinely need EventSource/SSE compatibility; prefer Authorization header everywhere else.
5. Add a spool governor: max queued files, max bytes, max age, and loud degraded status when full.

## Tests to add

- Negative GET with foreign Host cannot read archive.
- Negative POST with forged Origin/token fails.
- Legitimate shell/bootstrap and extension routes still work.
- Spool quota test proves oversized backlog returns 429/degraded without writing unbounded files.

## Verification commands

- ``devtools test tests/unit/daemon/test_daemon_http_security.py tests/unit/browser_capture -k 'host or origin or token or spool'` plus the existing browser extension smoke lane if present.`

## Pitfalls

- Do not broaden scope beyond the bead acceptance criteria; make a failing test first, then patch the smallest shared seam.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.

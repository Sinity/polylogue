# Reader Visual Smoke Lane

Polylogue's daemon-served local web reader is exercised by two automated
smoke lanes:

- [`tests/unit/daemon/test_web_reader.py`](../tests/unit/daemon/test_web_reader.py)
  owns fast endpoint and envelope contracts.
- [`tests/visual/test_reader_dom_smoke.py`](../tests/visual/test_reader_dom_smoke.py)
  owns browserless DOM/evidence assertions for the daemon-served shell and
  reader states.

Both lanes boot the production `DaemonAPIHTTPServer` against a synthetic
on-disk archive and drive the live HTTP surface — neither reads the operator's
real archive, serves committed private sample content, or makes pixel-diff
assertions that would freeze aesthetic iteration.

This page documents the harness so #848, #859, #865, and the design packs
have a stable reference for what the lane covers and how to run it. MK3 is the
current target and lives at [`docs/design/mk3/`](design/mk3/); the earlier MK2
visual-evidence handoff remains at
[`docs/design/mk2/coding-agent-pack/07-verification-and-visual-evidence.md`](design/mk2/coding-agent-pack/07-verification-and-visual-evidence.md).

## Running the lane

From the devshell:

```bash
nix develop -c pytest tests/unit/daemon/test_web_reader.py
nix develop -c pytest tests/visual
nix develop -c devtools lab-scenario run reader-visual-smoke
```

Both suites are part of the standard non-integration test run. There is no
browser binary or Playwright dependency in these fast lanes: they use Python's
standard `http.server`, `urllib.request`, and `html.parser` against the real
`DaemonAPIHTTPServer`. The `devtools lab-scenario` command is the
operator-facing wrapper for the visual/DOM lane.

## Coverage

| Reader state | Artefact id | Routes exercised |
|---|---|---|
| List / search | `polylogue.local_reader.search` | `/`, `/api/sessions`, `/api/sessions?query=...`, `/api/facets`, `/api/facets?origin=...`, `/api/facets?query=...` |
| Detail / session | `polylogue.local_reader.session` | `/c/{id}`, `/api/sessions/{id}`, `/api/sessions/{id}/messages`, `/api/sessions/{id}/raw` |
| Stack workspace | `polylogue.local_reader.workspace.stack` | `/w/stack?ids=...`, `/api/stack?ids=...` |
| Compare workspace | `polylogue.local_reader.workspace.compare` | `/w/compare?left=...&right=...&align=prompt`, `/api/compare?left=...&right=...&align=prompt` |
| Empty archive | — | `/api/sessions`, `/api/facets` |
| Degraded FTS | `polylogue.local_reader.degraded` | `/api/sessions?query=...` with message FTS absent |
| Privacy boundary | — | `/`, `/c/{id}`, `/api/facets`, `/api/sessions`, `/api/sessions/{id}`, `/api/sessions/{id}/messages` (auditing for absolute local paths) |
| Auth boundary | — | `/api/sessions` with/without `Authorization: Bearer ...` |

The artefact ids match the names referenced in the design packs and in #848 so
the visual-evidence companion can cross-reference them. MK3 expands the target
matrix to include reader, stack, topology, attachments, degraded states, and
palette screenshots under `docs/design/mk3/screens/`.

## What the lane checks

- **Page structure.** The HTML payloads at `/`, `/c/{id}`, and `/w/{mode}` contain the region
  hooks (`renderSidebarState`, `renderSessions`, `renderFacets`,
  `renderMain`, `renderWorkspaceToolbar`, `renderStackWorkspace`,
  `renderCompareWorkspace`, `renderInspector`) the JS bundle hydrates. A regression
  that drops a region fails here loudly without depending on pixel
  differences.
- **Evidence envelopes.** The visual DOM lane writes and reads JSON evidence
  manifests under pytest temp directories. Each manifest records artifact id,
  route, fixture id, command, evidence kind, checked structure, private-path
  status, and the explicit follow-up for the heavier browser screenshot gate.
- **Envelope shapes.** Every reader-facing JSON envelope is asserted by
  shape: `items`/`messages`/`raw_artifacts` plus `total` for the
  paginated list/detail surfaces, and the search route's `hits`/`total`
  envelope when `?query=` is supplied. Session rows, detail headers,
  and messages carry stable `target_ref` objects, deterministic reader
  anchors, and per-target action availability with explicit disabled
  reasons for actions the current reader cannot perform yet. Query search
  hits also carry both session-level targets and match-level message
  targets. Facets carry the `scoped_to_query`/`origins` shape and honour
  the `?origin=` and `?query=` filter contracts.
- **Empty / no-results state.** Distinguishes "archive is empty" from
  "query matched no rows", a discrimination the reader UI is required to
  expose.
- **Privacy.** The web shell HTML and ordinary reader JSON payloads are checked
  for absolute local-path prefixes (`/home/`, `/Users/`, `/realm/`,
  `/var/`, `/etc/`) so the loopback-only reader cannot quietly start
  leaking operator filesystem layout through enriched targets, anchors, or
  disabled action reasons. `/api/sources` and
  `/api/raw_artifacts/:id` deliberately surface absolute paths under
  the operator-level token (see [`security.md`](security.md)) and are
  out of scope here.
- **Auth boundary.** With a configured token, every authenticated
  endpoint refuses unauthenticated requests, and the unauthenticated
  web shell at `/` keeps responding so the reader can still bootstrap.
- **Hang protection.** Each route is asserted to return within a 10 s
  budget. The previous version of this harness omitted the
  `serve_forever()` thread so test workers hung forever on the first
  request — this iteration fixes the threading and pins the
  no-regression contract.

## What the lane explicitly does *not* do

- No pixel snapshots or image diffs that freeze implementation details.
- No real archive content, no operator data, no committed private
  fixtures — the synthetic seeder produces three single-message
  sessions with stable ids so the envelope assertions stay
  reproducible.
- No browser-binary requirement in the fast lanes. A later #865 slice should
  add a separate browser-backed screenshot lane for the richer reader, stack,
  topology, attachment, and degraded-state matrix rather than replacing this
  fast DOM/contract smoke.

## Follow-ups

The fast lane is still DOM/contract-only. The MK3 follow-up in #865 should add
a separate browser-backed screenshot lane for the richer reader, stack,
topology, attachment, and degraded-state matrix rather than replacing this
unit-speed smoke.

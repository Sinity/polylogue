# Reader Visual Smoke Lane

Polylogue's daemon-served local web reader is exercised by an automated
DOM/contract smoke lane committed in
[`tests/unit/daemon/test_web_reader.py`](../tests/unit/daemon/test_web_reader.py).
The lane boots the production `DaemonAPIHTTPServer` against a synthetic
on-disk archive and drives the live HTTP surface — it never reads the
operator's real archive, never serves committed sample content, and
never makes pixel-diff assertions that would freeze aesthetic
iteration.

This page documents the harness so #848, #859, #865, and the design packs
have a stable reference for what the lane covers and how to run it. MK3 is the
current target and lives at [`docs/design/mk3/`](design/mk3/); the earlier MK2
visual-evidence handoff remains at
[`docs/design/mk2/coding-agent-pack/07-verification-and-visual-evidence.md`](design/mk2/coding-agent-pack/07-verification-and-visual-evidence.md).

## Running the lane

From the devshell:

```bash
nix develop -c pytest tests/unit/daemon/test_web_reader.py
```

The suite is part of the standard unit run — `devtools verify` exercises
it end to end. There is no separate browser binary or Playwright
dependency: the lane uses Python's standard `http.server` and
`urllib.request` against the real `DaemonAPIHTTPServer`.

## Coverage

| Reader state | Artefact id | Routes exercised |
|---|---|---|
| List / search | `polylogue.local_reader.search` | `/`, `/api/conversations`, `/api/facets`, `/api/facets?provider=...` |
| Detail / conversation | `polylogue.local_reader.conversation` | `/api/conversations/{id}`, `/api/conversations/{id}/messages` |
| Empty archive | — | `/api/conversations`, `/api/facets` |
| Privacy boundary | — | `/`, `/api/facets`, `/api/conversations`, `/api/conversations/{id}`, `/api/conversations/{id}/messages` (auditing for absolute local paths) |
| Auth boundary | — | `/api/conversations` with/without `Authorization: Bearer ...` |

The artefact ids match the names referenced in the design packs and in #848 so
the visual-evidence companion can cross-reference them. MK3 expands the target
matrix to include reader, stack, topology, attachments, degraded states, and
palette screenshots under `docs/design/mk3/screens/`.

## What the lane checks

- **Page structure.** The HTML payload at `/` contains the five region
  hooks (`renderSidebarState`, `renderConversations`, `renderFacets`,
  `renderMain`, `renderInspector`) the JS bundle hydrates. A regression
  that drops a region fails here loudly without depending on pixel
  differences.
- **Envelope shapes.** Every reader-facing JSON envelope is asserted by
  shape: `items`/`messages`/`raw_artifacts` plus `total` for the
  paginated list/detail surfaces, and the search route's `hits`/`total`
  envelope when `?query=` is supplied. Conversation rows, detail headers,
  and messages carry stable `target_ref` objects, deterministic reader
  anchors, and per-target action availability with explicit disabled
  reasons for actions the current reader cannot perform yet. Facets carry
  the `scoped_to_query`/`providers` shape and honour the `?provider=`
  filter contract.
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
  conversations with stable ids so the envelope assertions stay
  reproducible.
- No browser-binary requirement. The MK3 follow-up in #865 should add a
  separate browser-backed screenshot lane for the richer reader, stack,
  topology, attachment, and degraded-state matrix rather than replacing this
  fast DOM/contract smoke.

## Follow-ups

Two FTS-dependent assertions in the harness are currently `@skip`ped
with explicit references to a #865 follow-up: query-based facets and
"no results for query" both require the FTS index to be primed on the
synthetic archive. Adding the priming is straightforward but out of
scope for the harness scaffolding.

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

This page documents the harness so workbench and reader changes have a stable
reference for what the lane covers and how to run it. Current UI work should
cite the owning issue/PR verification section and the tracked visual tests
under `tests/visual/`; old coding-agent handoff packs are not accepted as
verification authority.

Representative media has to be evidence, not decoration. Public evidence must
point to commands that run against synthetic fixtures or an explicitly local
operator profile and produce a named artifact: the DOM smoke lane below,
`devtools lab smoke run reader-visual-smoke`,
`devtools workspace dev-loop --tui-plan`,
`devtools workspace dev-loop --browser-provider-smoke`, or
`devtools workspace deployment-smoke --browser`. Screenshots, screencasts, and
store-listing images are useful only when they point back to one of those run
artifacts; standalone mockups belong in design notes, not release evidence.

## Public Screencast Tapes

`devtools render visual-tapes` writes the public VHS tape inventory. With
`--capture`, it also asks `vhs` to render GIFs when the binary is installed.
The default specs are deliberately self-contained and private-data-free:

- `demo-tour` runs `polylogue demo tour --out-dir demo-tour --force`, then
  shows the generated report.
- `query-tour` seeds `query-tour/archive`, runs the query/read drilldown, and
  summarizes facets for the same query.
- `reader-evidence-tour` runs the browserless reader smoke lane and renders the
  JSON report header.
- `browser-capture-tour` runs `devtools workspace dev-loop
  --browser-provider-live-follow --json`, which opens deterministic ChatGPT and
  Claude fixture pages in headless Chrome, loads the unpacked extension,
  captures both pages through the receiver, waits for archive/API convergence,
  opens the daemon web reader, and prints the redacted provider/popup/reader
  proof summary.

These are product evidence specs, not ad hoc recordings. If the command flow
changes, update the spec and regenerate the media rather than editing a GIF by
hand.

Current example renders:

- [`demo-tour.gif`](examples/demo-tour/demo-tour.gif)
- [`query-tour.gif`](examples/visual-tapes/query-tour.gif)
- [`reader-evidence-tour.gif`](examples/visual-tapes/reader-evidence-tour.gif)
- [`browser-capture-tour.gif`](examples/visual-tapes/browser-capture-tour.gif)

## Running the lane

From the devshell:

```bash
uv run devtools test tests/unit/daemon/test_web_reader.py
uv run devtools test tests/visual
uv run devtools lab smoke run reader-visual-smoke --json --report-dir .local/visual/reader-smoke
```

The first command owns the fast endpoint and envelope contracts. The second
command runs every browserless DOM/evidence test under `tests/visual`. The lab
wrapper runs the same `python -m pytest -q tests/visual` command and writes the
machine-readable report to `.local/visual/reader-smoke/reader-visual-smoke.json`.

Each visual test writes a JSON evidence manifest in its pytest temp directory.
Those per-test manifests use `schema_version: 1`, `evidence_kind: browserless-dom`,
`command: uv run devtools test tests/visual`, the artifact id, fixture id,
route, and the structural checks asserted by that test.

Both suites are part of the standard non-integration test run. There is no
browser binary or Playwright dependency in these fast lanes: they use Python's
standard `http.server`, `urllib.request`, and `html.parser` against the real
`DaemonAPIHTTPServer`. The `devtools lab smoke` command is the
operator-facing wrapper for the visual/DOM lane.

## Artifact inventory

The committed inventory below is exported by `devtools.visual_artifacts` and is
checked against the literal `write_evidence_manifest(...)` calls in
`tests/visual`, so new visual artifacts have to update the runnable inventory
instead of drifting into a decorative table.

| Artifact id | Owner | Fixture | Routes |
|---|---|---|---|
| `polylogue.local_reader.search` | `tests/visual/test_reader_dom_smoke.py` | `reader-visual-synthetic-v1` | `/`<br>`/api/sessions`<br>`/api/facets`<br>`/api/facets?origin=...`<br>`/api/facets?query=...` |
| `polylogue.local_reader.workspace.stack` | `tests/visual/test_reader_dom_smoke.py` | `reader-visual-synthetic-workspace-v1` | `/w/stack?ids=...&focus=...`<br>`/api/stack?ids=...` |
| `polylogue.local_reader.workspace.compare` | `tests/visual/test_reader_dom_smoke.py` | `reader-visual-synthetic-workspace-v1` | `/w/compare?left=...&right=...&align=prompt`<br>`/api/compare?left=...&right=...&align=prompt` |
| `polylogue.local_reader.session` | `tests/visual/test_reader_dom_smoke.py` | `reader-visual-synthetic-v1` | `/s/{id}`<br>`/api/sessions/{id}`<br>`/api/sessions/{id}/messages`<br>`/api/sessions/{id}/raw` |
| `polylogue.local_reader.search.query` | `tests/visual/test_reader_dom_smoke.py` | `reader-visual-synthetic-v1` | `/api/sessions?query=...` |
| `polylogue.local_reader.cost_panel` | `tests/visual/test_reader_dom_smoke.py` | `reader-visual-synthetic-v1` | `/api/sessions/{id}/cost` |
| `polylogue.local_reader.evidence_panel` | `tests/visual/test_reader_dom_smoke.py` | `reader-visual-synthetic-v1` | `/s/{id}`<br>`/api/sessions/{id}/artifacts`<br>`/api/sessions/{id}/neighbors` |
| `polylogue.local_reader.overlay_mutations` | `tests/visual/test_reader_dom_smoke.py` | `reader-visual-synthetic-v1` | `/s/{id}`<br>`/api/overlays/*` |
| `polylogue.local_reader.operator_flow` | `tests/visual/test_reader_dom_smoke.py` | `reader-visual-synthetic-v1` | `/s/{id}`<br>`/api/sessions/{id}/context`<br>`/api/overlays/*` |
| `polylogue.local_reader.insights_browser` | `tests/visual/test_reader_dom_smoke.py` | `reader-visual-synthetic-v1` | `/api/insights/sessions/{id}` |
| `polylogue.local_reader.degraded` | `tests/visual/test_reader_dom_smoke.py` | `reader-visual-synthetic-empty-and-degraded-v1` | `/api/sessions?query=...` |
| `polylogue.local_reader.paste_spans` | `tests/visual/test_reader_paste_spans.py` | `reader-visual-synthetic-v1+diff` | `/p`<br>`/api/paste-browser` |
| `polylogue.local_reader.paste_browser_empty` | `tests/visual/test_reader_paste_spans.py` | `reader-visual-empty-archive` | `/api/paste-browser` |
| `polylogue.local_reader.attachment_surface` | `tests/visual/test_reader_attachments.py` | `reader-visual-attachments-v1` | `/a`<br>`/api/attachments`<br>`/api/sessions/{id}/attachments` |
| `polylogue.local_reader.attachment_library_empty` | `tests/visual/test_reader_attachments.py` | `reader-visual-attachments-empty` | `/api/attachments` |
| `polylogue.local_reader.message_card` | `tests/visual/test_reader_action_rail.py` | `reader-visual-synthetic-v1` | `/`<br>`/api/sessions`<br>`/api/messages/{id}/actions` |

## What the lane checks

- **Page structure.** The HTML payloads at `/`, `/s/{id}`, and `/w/{mode}` contain the region
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
- No browser-binary requirement in the fast lanes. A later visual slice can add
  a separate browser-backed screenshot lane for the richer reader, stack,
  topology, attachment, and degraded-state matrix rather than replacing this
  fast DOM/contract smoke.

## Follow-ups

The fast lane is still DOM/contract-only. A later visual slice can add a
separate browser-backed screenshot lane for the richer reader, stack,
topology, attachment, and degraded-state matrix rather than replacing this
unit-speed smoke.

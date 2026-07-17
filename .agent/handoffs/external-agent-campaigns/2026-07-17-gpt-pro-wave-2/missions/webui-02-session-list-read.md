Title: "WebUI v2 vertical: session list and session read pages (SSR + islands) over the daemon's typed JSON"

Result ZIP: `webui-02-session-list-read-r01.zip`

## Mission

Build the first real WebUI v2 vertical on the webui-01 scaffold shape (if the
webui-01 result is not in your inputs, define the minimal interface you need
from it and state the assumption): a session LIST page (filter by origin,
time window, repo; paged) and a session READ page (message flow with roles,
material-origin distinction, tool use/result blocks with outcome flags,
attachments, lineage banner for fork/resume families).

Requirements:

1. Server-side render the first page of both routes as semantic HTML (the
   pages must be readable with JS disabled); hydrate islands for paging,
   filtering, and expand/collapse.
2. Consume ONLY continuation-based paged JSON (the daemon read surface is
   moving onto a shared bounded QueryTransaction: opaque cursor + stable
   result refs; design the client to that contract — inspect
   `polylogue/archive/query/transaction.py` in the snapshot for the page/
   continuation shape and mirror its vocabulary).
3. Honesty rules: distinguish exact vs qualified totals; render provider-
   marked failure structure (tool_result_is_error / exit codes) visibly;
   never render an empty state as zero when the underlying state is unknown/
   unconverged — surface the evidence state the JSON provides.
4. Deep-linkable refs: session and message anchors use the archive's stable
   ref scheme (session_id, message_id) so agents/humans can cite URLs.
5. Vitest component tests for list paging and read rendering of a fixture
   session (include a fixture with an error tool result and a fork family);
   a Python route test that SSR emits the semantic skeleton.

## De-overlap with the transcript-renderer job (webui-04)

A parallel job builds the full semantic-card transcript renderer over
`polylogue/rendering/semantic_card_registry.py` (the sole classification
owner — beads `ap7`/`4p1`). Your READ page owns the session SHELL: header
metadata (origin badge, title, times, repo, model set), lineage banner,
message-flow skeleton, paging, and anchors. For message BODIES, render a
deliberately simple placeholder projection (role + material_origin +
bounded text + a structural outcome flag for tool results) behind ONE
component interface (`<MessageBody card={...}>`-shaped) that webui-04's
card components will replace. Define that interface explicitly in
HANDOFF.md so the two jobs compose without rework; do not build rich
per-tool-family rendering here.

## Constraints

- Match the read semantics of the CLI/MCP surfaces — no client-side
  reinterpretation of roles/material_origin (read `core/enums.py`).
- Sanitized fixtures only; no real conversation content.
- Keep every network call same-origin; zero CDN.

## Deliverable emphasis

HANDOFF.md: route map, JSON contracts consumed (exact field lists), the
continuation-handling client utility (this becomes shared), fixture
inventory, and the exact list of old web-shell routes/files this vertical
supersedes (deletion candidates for the integrator, not deleted by you).

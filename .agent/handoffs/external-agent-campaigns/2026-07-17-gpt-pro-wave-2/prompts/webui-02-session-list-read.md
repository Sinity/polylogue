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


---

## Context and authority

You are a long-running ChatGPT Pro engineering worker. A recent Polylogue
project-state archive will be attached. Retrieve and inspect it broadly; do not
assume attachment bytes consume your active prompt context. The attached
snapshot is the code authority. This prompt defines your mission. Repository
instructions and complete relevant Beads records define constraints and intent;
later Beads notes may supersede older descriptions. Current source wins when a
stale plan names paths or APIs that no longer exist.

Start by reporting the snapshot commit/branch/dirty-patch identity you found and
the source, tests, Beads, and history you inspected. Follow dependencies beyond
the obvious files when they affect the production route. Do not invent an API,
test helper, product contract, or parallel framework to make the task easy.

## Working contract

- Produce the largest internally coherent implementation draft that fits the
  mission. Prefer one real end-to-end behavior over disconnected scaffolding.
- Preserve Polylogue's substrate-first architecture and existing typed
  interfaces. Small production seams are allowed only when real production
  behavior needs observation or control.
- Write concrete production changes and real-route tests. A test must name the
  production dependency it exercises and the representative implementation
  mutation/removal that should make it fail.
- Do not delete existing tests or helpers. Identify proposed dominated
  deletions separately for independent local certification.
- Use your container and run meaningful self-contained checks when possible.
  Never claim access to the operator's live daemon, browser, archive, secrets,
  NixOS deployment, or current worktree. Mark those checks `unverified`.
- If the full scope is unsafe, complete the strongest coherent subset and make
  the remaining decisions and exact continuation steps explicit. Do not return
  placeholders, ellipses, pseudocode presented as code, or a generic plan in
  place of implementation.

## Deliverable

Create the exact `Result ZIP` named near the top of this prompt under
`/mnt/data/`. Do not include the supplied repository/project-state archive or
other copied inputs in the result. The finished ZIP must be attached to the
conversation through a working, user-clickable download link. Work left only
in an internal shell directory, temporary notebook, scattered sandbox files,
or prose is not delivered.

The ZIP must contain:

- `HANDOFF.md`: mission, snapshot identity, inspected evidence, mechanism,
  decisions, changed files, acceptance matrix, apply order, risks, and exact
  verification performed/remaining;
- `PATCH.diff`: one apply-ready unified diff against the named snapshot;
- `TESTS.md`: test design, production dependencies, anti-vacuity mutation,
  commands, and honest execution results;
- `EVIDENCE.md`: relevant source/Bead/history findings and any contradictions;
- `FILES/`: complete replacements only where they materially disambiguate the
  patch; omit it when unnecessary.

Before answering, reopen the ZIP, list and validate its members, compute its
SHA-256 and byte size, and confirm that `PATCH.diff` has no placeholders or
copied source snapshot. Your final chat response must begin with a substantive
operator-readable report of what you did and why. It must also state important
limitations, missing or unverified work, and how much additional value another
iteration could plausibly add—distinguishing a small repair from a substantial
second pass. Then report verification and risks and give a prominent working
link to the exact `/mnt/data/` ZIP. A bare download receipt is not acceptable.

## Continuation protocol

Do not perform a separate adversarial review unless the user explicitly asks
for one. If the user asks to **iterate** or **continue**, preserve valid prior
work, perform the highest-value remaining implementation/research pass, and
publish a new cohesive package revision with the same complete structure—not a
loose supplemental patch. Explain exactly what changed, what improved, what
still remains, and whether another iteration is likely to pay off.

If the user explicitly asks for an **adversarial review**, attack your prior
result against the original mission and current attached authority: search for
unsupported claims, invented or stale APIs, missing call sites, composition
failures, unsafe assumptions, vacuous tests, patch/apply defects, incomplete
acceptance criteria, and evidence that would falsify the design. Preserve work
that survives. Then repair every legitimate finding you can, regenerate the
entire cohesive package as the next revision, and report findings, repairs,
remaining disputes, and the value of another adversarial/implementation pass.

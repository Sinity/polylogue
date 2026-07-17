Title: "WebUI v2 vertical: transcript rendering unified on the semantic-card registry (kill the render-path fragmentation)"

Result ZIP: `webui-04-transcript-render-r01.zip`

## Mission

Session→HTML rendering is currently fragmented across FIVE implementations
(a 2026-07-16 re-count under closed bead `polylogue-7le`, now owned by
`polylogue-4p1`/`polylogue-ap7`): the canonical rendering package
(`polylogue/rendering/` — `core.py`, `core_messages.py`, `blocks.py`,
`semantic_card_registry.py`, `semantic_card_models.py`), plus divergent
web-shell paths (`daemon/web_shell_reader.py`, `web_shell_semantic_cards.py`
and siblings), plus a CLI markdown-dialect axis introduced by ap7. The
ratified direction (read beads `ap7` + `4p1` in the snapshot's
`.beads/issues.jsonl`): `rendering/semantic_cards.py`-family stays the SOLE
classification/structure owner; every origin maps provider tools/envelopes
into normalized tool families; web and CLI PROJECT the same card document.

Build the WebUI v2 transcript renderer as the first true consumer of that
contract:

1. A typed card-document JSON endpoint (daemon-side): serialize the semantic
   card document (cards with structural outcomes, paths/targets, durations,
   exact refs, bounded disclosed previews, typed missing/unknown states) for
   a session/page — reusing the registry, adding NO web-local
   classification. If the current registry lacks a serialization seam, add
   the smallest one and document it as the shared contract.
2. Preact rendering of the card families: shell (command + exit code
   prominently, error-flagged), edit/write (path + bounded diff preview),
   read/search, task/delegation (link to child session), web, MCP,
   attachments, lineage banners, unknown-tool (renders evidence, never
   drops). Role + material_origin distinction visible (protocol rows vs
   human-authored — `core/enums.py` semantics; never render
   runtime-protocol rows as if the human wrote them).
3. Progressive disclosure: previews bounded server-side; expand fetches
   detail by exact ref (no unbounded payloads); virtualized long transcripts.
4. Parity harness: for 3+ fixture sessions (include: error tool results,
   a fork/compaction family, an unknown provider tool), assert the web card
   document and the CLI/terminal card projection agree on card count,
   family, and structural outcomes — this is the anti-fragmentation
   regression test the repo lacks.
5. SSR + islands per scaffold conventions (webui-01 interface assumptions
   stated if its result is absent from inputs).

## Constraints

- Do NOT add a sixth renderer: web code may only project the card document.
  Any needed semantics change goes into `rendering/` with tests there.
- `blocks.tool_result_is_error`/`exit_code` are provider-reported structure;
  NULL means unknown — render unknown as unknown (evidence-honesty rule).
- Sanitized fixtures; zero CDN.

## Deliverable emphasis

HANDOFF.md: the card-document JSON schema (this becomes a load-bearing
contract — spell it out fully), registry seams touched, parity-harness
design, per-family rendering decisions, the five old render paths with
exactly which this vertical supersedes, and open questions for ap7/4p1
integration.


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

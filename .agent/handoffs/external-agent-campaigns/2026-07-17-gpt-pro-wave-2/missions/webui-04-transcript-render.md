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

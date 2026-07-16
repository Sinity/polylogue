# 075. polylogue-0v9p — Language detection and preference facts for variant selection

Priority/type/status: **P1 / feature / open**. Lane: **07-content-variants**. Release: **E-content-variants**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

Why: agents should translate when useful, but the archive first needs honest language facts. Language detection is distinct from translation: it annotates source blocks/messages/sessions and informs projection defaults, filters, and agent prompts without creating transformed content.

## Existing design note

Add a language fact layer at block grain where practical, with message/session rollups derived from children. Automatic detections are rebuildable derived facts with detector/version/confidence; user corrections/preferences live in user.db/user_settings or assertion-backed corrections where appropriate. Support mixed-language messages by preserving block/span facts instead of forcing one session language. Expose query predicates and projection defaults such as preferred target language, translate-if-source-not-preferred, and confidence thresholds. Keep dependency choice pluggable; do not make a specific detector library part of the public contract.

## Acceptance criteria

Block/message/session language facts exist with confidence and provenance. Mixed-language messages are represented without collapsing to one false language. User preference/correction state overrides derived detection without altering source content. Query surfaces can filter by source language, and variant projection can choose candidate translation targets from language facts. Tests cover mixed-language blocks, low-confidence/unknown detection, user override, and no translation created merely by detection.

## Static mechanism / likely defect

Issue description localizes the mechanism: Why: agents should translate when useful, but the archive first needs honest language facts. Language detection is distinct from translation: it annotates source blocks/messages/sessions and informs projection defaults, filters, and agent prompts without creating transformed content. Design direction: Add a language fact layer at block grain where practical, with message/session rollups derived from children. Automatic detections are rebuildable derived facts with detector/version/confidence; user corrections/preferences live in user.db/user_settings or assertion-backed corrections where appropriate. Support mixed-language messages by preserving block/span facts instead of forcing one session language. Expose que…

## Source anchors to inspect first

- `polylogue/core/identity_law.py:42` — Current identity includes variant_index only for provider sibling messages, not transformed content variants.
- `polylogue/storage/sqlite/queries/message_query_reads.py:34` — Read model projects message variant_index as branch_index.
- `polylogue/surfaces/payloads.py:747` — Reader payload maps message variant_index to branch_index.
- `polylogue/daemon/compare.py:220` — Compare/alignment semantics exist for message diffing, not content-transform alignment.
- `polylogue/core/refs.py` — Existing ref model should be extended, not bypassed.
- `polylogue/storage/sqlite/archive_tiers/user_write.py:901` — Findings should reuse assertion/candidate lifecycle.
- `polylogue/surfaces/payloads.py:747` — Surface payloads currently expose branch/message refs; expand consistently.
- `polylogue/mcp/payloads.py:377` — MCP message payloads carry variant/branch metadata today.
- `polylogue/archive/query/spec.py:440` — SessionQuerySpec is the core read selection object.
- `polylogue/archive/query/spec.py:498` — from_params centralizes parameter lowering.
- `polylogue/cli/query_contracts.py:70` — QueryOutputSpec is current output/render contract seam.
- `polylogue/cli/archive_query.py:164` — CLI archive query execution remains a dense orchestration point.
- `polylogue/daemon/http.py:1734` — Daemon split-archive route already has from_params path for some cases.
- `polylogue/daemon/http.py:3296` — Another daemon path builds SessionQuerySpec from raw query params.
- `polylogue/mcp/query_contracts.py:54` — MCP query normalization has a distinct lowering path.

## Implementation plan

1. Add a language fact layer at block grain where practical, with message/session rollups derived from children.
2. Automatic detections are rebuildable derived facts with detector/version/confidence
3. user corrections/preferences live in user.db/user_settings or assertion-backed corrections where appropriate.
4. Support mixed-language messages by preserving block/span facts instead of forcing one session language.
5. Expose query predicates and projection defaults such as preferred target language, translate-if-source-not-preferred, and confidence thresholds.
6. Keep dependency choice pluggable
7. do not make a specific detector library part of the public contract.

## Tests to add

- Acceptance proof: Block/message/session language facts exist with confidence and provenance.
- Acceptance proof: Mixed-language messages are represented without collapsing to one false language.
- Acceptance proof: User preference/correction state overrides derived detection without altering source content.
- Acceptance proof: Query surfaces can filter by source language, and variant projection can choose candidate translation targets from language facts.
- Acceptance proof: Tests cover mixed-language blocks, low-confidence/unknown detection, user override, and no translation created merely by detection.

## Verification commands

- `devtools test <focused tests added for this bead>`

## Pitfalls

- Do not overwrite original message/block text; variants are separate evidence-linked objects.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.

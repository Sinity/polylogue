# 078. polylogue-d4zk — User and agent UX for creating, reviewing, and messaging about variants

Priority/type/status: **P1 / feature / open**. Lane: **07-content-variants**. Release: **E-content-variants**. Readiness: **blocked-hard**.

Hard blockers: polylogue-arso, polylogue-rlsb

## What the bead says

Why: the operator wants agents to translate at will and wants to view/interact with those translations. The human user should also participate in the same object-ref messaging substrate as agents: point at a block/message/assertion/session, ask an agent to create a variant, review the result, and send decisions back with refs.

## Existing design note

Build UX over existing addressing and coordination messages. Web/reader/in-page surfaces let the user select a session/message/block/span/assertion/variant-node and request create_variant/translate/simplify/summarize from an existing or new agent participant. Agents can send messages to user:local with attached refs such as variant-node low-confidence alignment, missing assertion translation, or review-needed candidate. MCP prompts/tools expose create_content_variant and translate_target as convenience over the generic variant write path. Review UX shows coverage, alignment, status, source language, target language, author/provenance, and missing translated assertions. Coordinate with s7ae.3 coordination messages, pj8 MCP prompts, bby.11 webui v2, and 90y in-page overlay.

## Acceptance criteria

A user can address an object ref and request a variant-producing action through CLI/MCP and at least one web/in-page UX path. Agents can create candidate variants with alignment metadata and send a user-addressed coordination message containing clickable refs. The user can accept/reject/supersede or otherwise mark variant status without changing original source content. UI distinguishes original assertions from translated assertion variants and handles missing assertion translations honestly. Tests or demo fixtures cover translate heavily annotated session, review low-confidence alignment, and agent-to-user message with attached variant refs.

## Static mechanism / likely defect

Issue description localizes the mechanism: Why: the operator wants agents to translate at will and wants to view/interact with those translations. The human user should also participate in the same object-ref messaging substrate as agents: point at a block/message/assertion/session, ask an agent to create a variant, review the result, and send decisions back with refs. Design direction: Build UX over existing addressing and coordination messages. Web/reader/in-page surfaces let the user select a session/message/block/span/assertion/variant-node and request create_variant/translate/simplify/summarize from an existing or new agent participant. Agents can send messages to user:local with attached refs such as variant-node low-confidence alignment, missing assertion translation, or review-needed candid…

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

1. Build UX over existing addressing and coordination messages.
2. Web/reader/in-page surfaces let the user select a session/message/block/span/assertion/variant-node and request create_variant/translate/simplify/summarize from an existing or new agent participant.
3. Agents can send messages to user:local with attached refs such as variant-node low-confidence alignment, missing assertion translation, or review-needed candidate.
4. MCP prompts/tools expose create_content_variant and translate_target as convenience over the generic variant write path.
5. Review UX shows coverage, alignment, status, source language, target language, author/provenance, and missing translated assertions.
6. Coordinate with s7ae.3 coordination messages, pj8 MCP prompts, bby.11 webui v2, and 90y in-page overlay.

## Tests to add

- Acceptance proof: A user can address an object ref and request a variant-producing action through CLI/MCP and at least one web/in-page UX path.
- Acceptance proof: Agents can create candidate variants with alignment metadata and send a user-addressed coordination message containing clickable refs.
- Acceptance proof: The user can accept/reject/supersede or otherwise mark variant status without changing original source content.
- Acceptance proof: UI distinguishes original assertions from translated assertion variants and handles missing assertion translations honestly.
- Acceptance proof: Tests or demo fixtures cover translate heavily annotated session, review low-confidence alignment, and agent-to-user message with attached variant refs.

## Verification commands

- `devtools test <focused tests added for this bead>`

## Pitfalls

- Do not overwrite original message/block text; variants are separate evidence-linked objects.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.

# 076. polylogue-arso — Content variant substrate: refs, nodes, alignment, storage

Priority/type/status: **P1 / feature / open**. Lane: **07-content-variants**. Release: **E-content-variants**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

Why: translations and other transformed content need a first-class substrate over existing public refs. A target_ref=session must mean the whole declared session composition; target_ref=message must mean the whole message; target_ref=block means exactly that block. The system must not encode these as loose notes or assertion blobs, because variants are transformed content artifacts with provenance and alignment, not epistemic claims.

## Existing design note

Implement typed models and storage for ContentVariant, VariantNode, and VariantAlignment. Extend public refs to include variant:<id> and variant-node:<id>; preserve existing assertion:<id> refs and allow variants to target assertion refs. Use closed vocabularies: kind translation/transliteration/simplification/summary; status candidate/active/rejected/superseded/stale; coverage complete/partial/sparse; relation translates/transliterates/simplifies/summarizes/omits/expands/reorders. Store source_hash/source_fingerprint or equivalent staleness evidence so variants can be marked stale if the target changes. Place rows in the correct tier: transformed user/agent artifacts are durable enough to protect, while cheap automatic language detections are rebuildable unless user-corrected. Avoid duplicating assertion lifecycle; reuse concepts such as author_ref, evidence_refs, supersedes, and staleness where appropriate.

## Acceptance criteria

Canonical types, storage DDL, repository/API read/write methods, and public ref resolution exist for variant and variant-node refs. Variants can target session/message/block/assertion refs. Alignment supports one-to-one, one-to-many, many-to-one, omitted, and partial mappings. Tests prove a session-level variant with complete coverage covers all declared child messages/blocks, a partial variant is labeled partial, a summary maps many source nodes to one variant node without positional hacks, and a translated assertion remains a variant of assertion:<id> rather than a projected original assertion. Generated schemas/docs are refreshed where required.

## Static mechanism / likely defect

Issue description localizes the mechanism: Why: translations and other transformed content need a first-class substrate over existing public refs. A target_ref=session must mean the whole declared session composition; target_ref=message must mean the whole message; target_ref=block means exactly that block. The system must not encode these as loose notes or assertion blobs, because variants are transformed content artifacts with provenance and alignment, not epistemic claims. Design direction: Implement typed models and storage for ContentVariant, VariantNode, and VariantAlignment. Extend public refs to include variant:<id> and variant-node:<id>; preserve existing assertion:<id> refs and allow variants to target assertion refs. Use closed vocabularies: kind translation/transliteration/simplification/summary; status candidate/active/rejected/superseded/stale; coverage complete/partial/sparse; relation tran…

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

1. Implement typed models and storage for ContentVariant, VariantNode, and VariantAlignment.
2. Extend public refs to include variant:<id> and variant-node:<id>
3. preserve existing assertion:<id> refs and allow variants to target assertion refs.
4. Use closed vocabularies: kind translation/transliteration/simplification/summary
5. status candidate/active/rejected/superseded/stale
6. coverage complete/partial/sparse
7. relation translates/transliterates/simplifies/summarizes/omits/expands/reorders.

## Tests to add

- Acceptance proof: Canonical types, storage DDL, repository/API read/write methods, and public ref resolution exist for variant and variant-node refs.
- Acceptance proof: Variants can target session/message/block/assertion refs.
- Acceptance proof: Alignment supports one-to-one, one-to-many, many-to-one, omitted, and partial mappings.
- Acceptance proof: Tests prove a session-level variant with complete coverage covers all declared child messages/blocks, a partial variant is labeled partial, a summary maps many source nodes to one variant node without positional hacks, and a translated assertion remains a variant of assertion:<id> rather than a projected original assertion.
- Acceptance proof: Generated schemas/docs are refreshed where required.

## Verification commands

- `devtools test <focused tests added for this bead>`

## Pitfalls

- Do not overwrite original message/block text; variants are separate evidence-linked objects.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.

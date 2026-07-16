# 079. polylogue-4smp — Content variants: language-aware transformed archive objects with alignment

Priority/type/status: **P1 / epic / open**. Lane: **07-content-variants**. Release: **E-content-variants**. Readiness: **epic-needs-child-closure**.

## What the bead says

Why: agents should be able to translate source content, annotations, and other addressable Polylogue objects for the operator, and the reader/export/query surfaces should let the operator view and interact with those translations without confusing transformed text with original evidence. The operator's "alternates" sketch is not the requirement; the requirement is a general algebraic substrate for transformed content. Translation is the motivating case, but the same primitive should support transliteration, simplification, and summary while preserving source provenance, coverage, and alignment.

Scope: add a content-variant primitive over existing public object refs, not a separate translation/export subsystem. Variants target addressable refs such as session/message/block/assertion/variant-node, carry kind/language/status/coverage/composition metadata, and are rendered through the existing Query x Projection x Render algebra. Alignment edges map source refs to variant nodes so message/block/session hierarchy is semantically meaningful and lossy transforms such as summaries remain honest. Assertions remain assertions; variants can target assertions, and assertions can target variants.

## Existing design note

Core model: ContentVariant(target_ref, kind, source_language, target_language, status, coverage, composition_policy, author_ref, evidence_refs, staleness/supersession, metadata). VariantNode represents structured variant content at session/message/block/span/assertion-body grain. VariantAlignment maps source_ref -> variant_node_ref with relation vocabulary such as translates, transliterates, simplifies, summarizes, omits, expands, reorders. Do not rely on positional convention such as "summary in first block"; agents may provide partial alignment when exact mapping is unavailable.

Placement: reuse public ObjectRef/target_ref semantics and user-state/provenance concepts; add new storage only where assertions are the wrong ontology. Variants are transformed content artifacts, not epistemic assertions. Assertions/annotations stay in the assertion substrate and may themselves be variant targets. Rendering and export are ProjectionSpec/RenderSpec policy, not bespoke commands. Query surfaces must distinguish original evidence text from variant text.

Extant bead anchors: polylogue-37t.1 for assertion lifecycle boundaries; polylogue-4p1 and polylogue-jnj.1 for read algebra; polylogue-fnm.2/fnm.6/fnm.10 for query projection stages; polylogue-bby.11 and polylogue-90y for web/in-page UX; polylogue-s7ae.3 for user/agent coordination messages; polylogue-pj8 for MCP prompt discoverability.

## Acceptance criteria

A typed content-variant model exists over public refs without treating variants as assertions. Variants support at least translation, transliteration, simplification, and summary with closed relation/status/coverage vocabularies. Variant nodes and alignment edges allow session/message/block/assertion variants to map source child elements honestly, including many-to-one summary relations and partial alignment. Query/read/export/web/MCP surfaces label source vs variant text and never present translations as original evidence. A demo or fixture shows a heavily annotated session translated with transcript variants plus variants of selected assertion annotations, with clickable alignment back to original source and original assertions. Extant read algebra, assertion, web, and coordination beads are linked so implementation lands as composed substrate, not a silo.

## Static mechanism / likely defect

Issue description localizes the mechanism: Why: agents should be able to translate source content, annotations, and other addressable Polylogue objects for the operator, and the reader/export/query surfaces should let the operator view and interact with those translations without confusing transformed text with original evidence. The operator's "alternates" sketch is not the requirement; the requirement is a general algebraic substrate for transformed content. Translation is the motivating case, but the same primitive should support transliteration, simpli… Design direction: Core model: ContentVariant(target_ref, kind, source_language, target_language, status, coverage, composition_policy, author_ref, evidence_refs, staleness/supersession, metadata). VariantNode represents structured variant content at session/message/block/span/assertion-body grain. VariantAlignment maps source_ref -> variant_node_ref with relation vocabulary such as translates, transliterates, simplifies, summarizes, …

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

1. Core model: ContentVariant(target_ref, kind, source_language, target_language, status, coverage, composition_policy, author_ref, evidence_refs, staleness/supersession, metadata).
2. VariantNode represents structured variant content at session/message/block/span/assertion-body grain.
3. VariantAlignment maps source_ref -> variant_node_ref with relation vocabulary such as translates, transliterates, simplifies, summarizes, omits, expands, reorders.
4. Do not rely on positional convention such as "summary in first block"
5. agents may provide partial alignment when exact mapping is unavailable.
6. Placement: reuse public ObjectRef/target_ref semantics and user-state/provenance concepts
7. add new storage only where assertions are the wrong ontology.

## Tests to add

- Acceptance proof: A typed content-variant model exists over public refs without treating variants as assertions.
- Acceptance proof: Variants support at least translation, transliteration, simplification, and summary with closed relation/status/coverage vocabularies.
- Acceptance proof: Variant nodes and alignment edges allow session/message/block/assertion variants to map source child elements honestly, including many-to-one summary relations and partial alignment.
- Acceptance proof: Query/read/export/web/MCP surfaces label source vs variant text and never present translations as original evidence.
- Acceptance proof: A demo or fixture shows a heavily annotated session translated with transcript variants plus variants of selected assertion annotations, with clickable alignment back to original source and original assertions.
- Acceptance proof: Extant read algebra, assertion, web, and coordination beads are linked so implementation lands as composed substrate, not a silo.

## Verification commands

- `devtools test <focused tests added for this bead>`

## Pitfalls

- Do not implement the epic as a single broad PR; use it to close/split child work and verify the terminal invariant.
- Do not overwrite original message/block text; variants are separate evidence-linked objects.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.

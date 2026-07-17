# Evidence and decision record

## Authority order

The implementation used this authority order:

1. the mission and exact Result ZIP contract;
2. the attached snapshot’s Git bundle at `f654480cadb7cc4c194704e24dfd483199547b35`;
3. repository instructions and current source at that commit;
4. complete current Beads records for `polylogue-ap7.1` and `polylogue-395j`;
5. relevant landed history and tests;
6. older plans or naming only where current source still confirmed them.

No API, test helper, deployment state, or live operator evidence was invented.

## Snapshot evidence

`polylogue-manifest.json` and `polylogue-overview.json` both report:

```text
branch: master
commit: f654480cadb7cc4c194704e24dfd483199547b35
dirty: true
generated_at: 2026-07-17T043202Z
```

The same archive contains:

```text
polylogue-branch-delta.patch       0 bytes
polylogue-branch-delta-files.txt   0 bytes
polylogue-branch-delta-log.txt     0 bytes
```

This is a material contradiction: the manifest calls the source dirty, but every captured tracked-delta surface is empty. The working-tree tar is also selective; treating it as a deletion-authoritative mirror would falsely remove tracked repository material. The safe reconstruction is therefore the named clean Git commit plus a non-destructive overlay. The delivered patch does not claim to preserve an unavailable pre-existing tracked dirty patch.

## Beads evidence

### `polylogue-ap7.1`

The record says the landed shell/edit/task/attachment core must be completed with provider-neutral family coverage and bounded lineage parity. Its design requires a coverage matrix over normalized tool families and executable Origins, explicit fallback, source-structural outcomes, target/path extraction, provenance, bounded previews, and one `SemanticCard.to_document` contract consumed by CLI Markdown and web. It specifically rejects full-family topology hydration and duplicate work-graph semantics.

Implementation consequences:

- enumerate policy over `SemanticBlockType` and `Origin`, not an aspirational list of provider tools;
- keep provider namespaces open and unknown tools inspectable as fallback;
- classify from persisted semantic type, structural MCP identity, or exact aliases only;
- keep delegation as its existing bounded query/read model;
- represent DB lineage as partial when the session row lacks root/completeness authority;
- represent archive lineage from the exact composed envelope.

### `polylogue-395j`

The record describes a real ChatGPT export with many typed but zero-content thinking blocks. Its later design note explicitly says the archive evidence is legitimate and must not be fabricated or erased; operator presentation should emit one compact absence marker per contiguous run with count and provenance. The rule must key on typed block kind and value state, never provider name or prose regex.

Implementation consequences:

- both `thinking` and storage-superset `reasoning` are typed absence candidates;
- whitespace-only content is absent;
- each notice contains every exact source coordinate;
- non-thinking typed rows terminate runs;
- non-empty thinking remains normal typed prose;
- raw/forensic evidence remains available through source coordinates and original storage, while the reader avoids full message-card weight.

## Current-source findings

### Existing card core was real and should be extended

PR #2700 already established a pure card model/registry, structural outcome trivalence, bounded preview logic, raw fallback, CLI Markdown, a public card schema, and hostile cross-provider fixtures. Replacing it with another renderer would violate the mission and architecture. The patch extends those types and builders into an ordered transcript rather than creating a parallel framework.

### Web wiring was card-only and not on the browser’s primary message route

PR #2736 attached card placement to full session-detail routes and gave the browser a card HTML hook. Current browser code loads a summary route and the paginated `/api/sessions/{id}/messages` route. Before this patch, the primary paginated route did not attach semantic cards/entries, and the presentation leaf still classified raw tool/thinking roles as a fallback. That made the landed web integration incomplete for the actual read path.

The patch enriches both paginated backends and makes `semantic_entries` authoritative. Legacy fallback remains only when the field is absent, preserving compatibility without allowing current payloads to drift.

### Card-only placement could not represent prose or typed absence

The prior placement model indexed cards and suppression by message. It could not carry exact authored prose, a compact run-level absence notice, or session-level ordered entries. The patch changes placement to distribute the shared transcript’s entry documents while retaining card-only compatibility views.

### One-pass result pairing is order-sensitive

The original renderer paired a result only when the use had already been observed. Provider exports can serialize a tool result before its use. That produced an orphan result when scanned, then a paired card later, duplicating the same evidence. The patch indexes both coordinate sets first and pairs FIFO by exact `tool_id`, independent of serialization order.

### Suppression must account for mixed rows

A message can contain a paired protocol result and independently meaningful runtime context/prose. Suppressing solely because its result is paired deletes real content. Placement now suppresses only `paired_result_ids - independent_message_ids`; the mixed-row fixture proves the context survives.

### Role and provider are insufficient for empty-thinking behavior

`role=thinking`, `has_thinking`, or ChatGPT-specific names are not enough to establish a compact absence. The typed block vocabulary in `BlockType` distinguishes `thinking` and `reasoning`, and the actual value establishes absence. The shared renderer therefore owns compaction, not the web leaf.

### Persisted semantic type and exact alias have different authority

`SemanticBlockType` is closed and established upstream; provider tool names are open. Treating the alias table as complete would be false. The registry therefore gives persisted semantic type precedence over aliases, except for structurally encoded MCP identity, and documents every unlisted origin namespace as fallback-capable.

### MCP identity parsing was duplicated

`polylogue/insights/tool_usage.py` already parsed `mcp__server__tool`; rendering needed the same protocol identity. Duplicating the string split would allow insight/render drift. The parser is now a core structural helper reused by both.

### Archive composition already has stronger authority than DB session rows

The archive envelope already carries lineage completeness and truncation reason from PR #2603. It did not retain which source session contributed each composed message or expose the exact inheritance/branch point on the envelope. Position-based inference would be unsafe. The patch adds those bounded facts at the composition seam.

Ordinary DB session rows carry parent/relation but not exact root/completeness/branch-point authority. The renderer labels those unavailable instead of hydrating a full topology family or inventing values.

### Storage decoupling initially lost metadata-backed display fields

Removing the renderer’s runtime import was architecturally correct, but the generic attribute projection initially failed to recover `language`, URL, name, and media type from persisted metadata JSON. The final audit found this concrete regression. `_coerce_mapping_block` now falls back to parsed metadata for those fields, and a focused test fails if the fallback is removed.

### ChatGPT web fixture name was stale

The old case name `chatgpt-web-fallback.json` encoded the prior gap. Current typed/alias policy can classify the structurally known ChatGPT web recipient as a web card. The case is renamed to `chatgpt-web.json` and asserts the specialized contract instead of preserving a stale fallback expectation.

### Generated text is marked binary by repository attributes

Normal Git stat output labels several generated JSON/Markdown/YAML files as binary. A default patch would therefore omit reviewable text hunks. `PATCH.diff` was exported from a temporary index with `--text --full-index --binary`; it contains ordinary unified text hunks and no binary patch marker. A real clean apply reproduced all changed bytes.

## Historical evidence

### PR #2700 — `0f5059068`

The commit message states that it added a pure semantic-card contract and registry for shell, edit, lineage, task, attachment, and unknown evidence; wired CLI Markdown; and deliberately left web parity and larger cross-provider coverage open. This confirms the correct extension point and the residual scope.

### PR #2736 — `0c251b600`

The commit message states that it wired the existing registry into the daemon reader, introduced per-message card placement, and covered only shell, file edit, and task end-to-end. It explicitly describes older-payload fallback. This confirms both the shared-registry intent and why complete ordered entries still needed work.

### PR #2629 — `7688d103d`

The commit corrected recipient-addressed ChatGPT tool calls to typed `TOOL_USE` instead of raw text. The patch preserves that authority: a typed tool-use envelope suppresses serialized invocation JSON in `message.text` rather than rendering it as prose.

### PR #2603 — `c06ca601c`

The commit added `lineage_complete` and `lineage_truncation_reason` to archive composition. The patch consumes those exact fields and does not recreate completeness logic in rendering.

### PR #2759 — `9ebc09ef0`

The commit established a bounded delegation evidence-card projection with its own query/read semantics. The semantic transcript uses task cards for tool dispatch evidence but does not copy that broader delegation model into the renderer.

## Test and route evidence

The frozen corpus now contains 23 cases. It covers every card kind plus ordered prose/notices, result-before-use, mixed result/context rows, message-envelope attachments, MCP structure, typed empty thinking, exact authoredness/variant/inherited metadata, hostile outcomes, non-UTF-8 replacement disclosure, missing results, unknown tools, and lineage uncertainty.

`tests/visual/test_reader_semantic_cards.py` was changed from the full-detail compatibility route to the actual paginated message route used by the browser. It reconstructs the ordered transcript envelope and validates it against `semantic-transcript.v1`, while nested cards are validated against `semantic-card.v1`. The test also pins the authoritative entry hooks in served HTML.

`tests/unit/storage/test_lineage_normalization.py` now asserts that a prefix-sharing child’s composed parent rows retain the parent source session, child-tail rows retain the child source session, and the envelope exposes exact inheritance and branch point.

## Unverified evidence

No claim is made about live operator data, current daemon state, private archives, secrets, browser-capture behavior, Nix deployment, or a real unsanitized provider export. The environment could not execute normal repository-wide imports/gates because locked dependencies are incomplete. The exact blocked checks are listed in `TESTS.md` and `HANDOFF.md`.

# EVIDENCE — beads-05 semantic transcript rendering

## Authority order

1. Mission document `[beads 05] Semantic transcript rendering`.
2. Attached snapshot source at commit `f654480cadb7cc4c194704e24dfd483199547b35`.
3. Current Bead records in `.beads/issues.jsonl`, especially `polylogue-ap7.1` and `polylogue-395j`.
4. Repository instructions and tests.
5. Older plans/history only where they remain consistent with current source.

## Snapshot findings

The supplied manifest records:

- branch `master`;
- commit `f654480cadb7cc4c194704e24dfd483199547b35`;
- generated time `2026-07-17T043202Z`;
- `dirty: true`.

The same snapshot contains zero-byte branch-delta patch, file-list, and log artifacts. Its human branch-delta report names `f654480cadb7cc4c194704e24dfd483199547b35` as the merge base and contains no diff or commits. This is a contradiction in metadata, not recoverable source evidence. The implementation therefore uses the named commit plus the supplied working-tree archive and does not claim to preserve an unknown tracked dirty patch.

## Primary Bead findings

### `polylogue-ap7.1`

The Bead requires:

- a generated coverage matrix over normalized tool families and executable Origins;
- file-read/search, web, MCP, write/edit variants, lineage/delegation, attachment, shell, task, and unknown cards;
- structural outcomes, target/path extraction, refs, bounded previews, and explicit missing/unknown behavior;
- one card document consumed by CLI and web;
- bounded database/archive lineage parity without complete-family hydration;
- preservation of the ChatGPT recipient-addressed `TOOL_USE` canary.

Implementation correspondence:

- `semantic_card_registry.py` now has exhaustive Origin policy, complete current persisted semantic-family policy, exact aliases, and exact MCP grammar.
- generated JSON/Markdown registry surfaces expose those policies;
- the 20-case fixture corpus covers all ten card kinds and ordered transcript entries;
- CLI and web receive the same card/transcript documents;
- database and archive paths preserve physical source ownership and use the same placement renderer;
- the ChatGPT positive and negative canaries pass.

The Bead’s strongest remaining gap is proof rather than mechanism: there is no single byte-for-byte database/archive parity test built from one delegation-heavy fixture, and the named live ChatGPT session was not available.

### `polylogue-395j`

The Bead explicitly says empty exported thinking is legitimate absence evidence, not a parsing error. It requires one compact marker per contiguous run, provenance/count retention, non-empty thinking folds, and no provider/prose heuristics.

Implementation correspondence:

- absence is a first-class typed transcript entry;
- contiguous empty `thinking` and `reasoning` blocks/messages compact once;
- references retain message, block, physical source session, role, message type, and authoredness;
- ordinary prose does not trigger compaction;
- non-empty thinking remains typed and foldable in Markdown and web;
- the mixed fixture freezes empty and non-empty behavior together.

Manual verification against a real reasoning-model ChatGPT export remains unavailable.

## Adjacent Bead findings

- `polylogue-e2yk` is already closed and establishes that recipient-addressed JSON must parse as `TOOL_USE`. This patch preserves and tests that boundary; it does not reimplement parser semantics in the renderer.
- `polylogue-4p1` calls for a future sole Query × Projection × Render read algebra. The current slice reuses existing read/composition seams and does not invent that larger framework.
- `polylogue-37km` requires canonical HTML aesthetics and browser-visible outcome-first tool blocks. This patch provides semantic structure but does not claim that visual program complete.
- `polylogue-1ilk` requires Playwright/component/visual-regression infrastructure. DOM/source tests here are not presented as browser proof.
- `polylogue-1lm` concerns a broader composable projection algebra and remains outside this patch.

## History findings

`0f5059068` introduced the original provider-neutral card model, registry, Markdown renderer, CLI integration, generated schema/registry, and 15-case fixture corpus. It established the correct extension point.

`0c251b600` added the web placement adapter, daemon card payloads, browser-shell card renderer, and visual DOM tests. It established web compatibility but still placed only cards, not one ordered content/card/absence stream.

Current source therefore won over any stale plan suggesting a new renderer. The patch extends these landed modules and removes no independent test/helper framework.

## Source findings and production-route consequences

### Existing split semantics

At the snapshot, CLI could render semantic cards through `build_semantic_transcript`, while the web reader received per-message card lists and still chose ordinary text/thinking/tool presentation with raw-message heuristics. That split could not express a compact run spanning multiple empty-thinking records or preserve ordered content/cards in one payload.

Resolution: `semantic-transcript.v1` orders content, absence, and card entries. Per-message placement is derived from that stream; it is not a second classifier.

### Tool family gaps

The original registry launched shell/edit and treated several normalized families as model-only or fallback. `polylogue-ap7.1` requires explicit current-family handling.

Resolution: add file-read, search, web, MCP, and task mappings from persisted semantics/exact aliases, while leaving unsupported names as raw-evidence fallback. Exact MCP parsing rejects malformed lookalikes.

### Physical ownership loss

Canonical composed rows already retain their physical `session_id` in `MessageRecord`, but `Message` hydration did not expose it. Archive-envelope rows likewise did not expose physical ownership. This made inherited content appear child-owned in renderer evidence.

Resolution: add optional `Message.source_session_id`, hydrate from canonical rows, carry through archive rows, and retain separate invocation/result owners.

### Inherited attachment loss

Repository full, render, paginated, and batch reads fetched attachments only for the requested logical session. Prefix-sharing inherited messages physically belong to ancestor sessions, so their attachments could disappear.

Resolution: derive the unique physical owner sessions from the already-composed rows and batch only those attachment lookups. A public-facade test proves full, paginated, and bulk reads all retain the ancestor attachment.

### Duration and code language loss

The canonical message SELECT omitted `duration_ms`, and hydration did not promote code language from block metadata. Archive conversion also omitted block language.

Resolution: select/map/hydrate duration and language and freeze them in typed semantic content/source tests.

### Lineage completeness asymmetry

Archive envelopes already carry `lineage_complete` and truncation reason; normal topology objects do not. Claiming normal-route completeness would invent evidence.

Resolution: archive transcripts emit concrete completeness/truncation state. Normal topology lineage cards explicitly caveat that composition completeness was not supplied.

### Pagination boundary

A paginated CLI/read response cannot pair a tool result or compact an absence run outside the current page without an additional read.

Resolution: keep the renderer pure and mark both pairing and absence-compaction scope as `page`; Markdown displays the warning.

## Contract reconciliation

`polylogue-ap7.1` says CLI and web consume `SemanticCard.to_document()`/schema, while `polylogue-395j` requires ordered non-card content and compact absence. Replacing the card schema would break the landed contract. The patch reconciles them by retaining `semantic-card.v1` unchanged as each card document and adding `semantic-transcript.v1` as an ordered envelope. The web still exposes `semantic_cards` for compatibility, but semantic presentation prefers `semantic_entries`.

The mission also says to remove superseded renderer duplication. The patch removes semantic reclassification from the new typed web path but retains old raw/card-only paths as backward compatibility for payloads that do not yet carry semantic entries. Deleting those fallbacks would be a separate compatibility decision and is not proposed here.

## Files and artifacts inspected but not copied

- supplied project-state TAR archives and their manifest/audit/branch-delta artifacts;
- all-ref Git bundle and working-tree archive;
- `.beads/issues.jsonl` and extracted primary/adjacent Bead records;
- prior commit diffs and stats;
- repository source/tests listed in `HANDOFF.md`.

None of the supplied archives, XML context slices, Git bundle, operator paths, or snapshot packaging files are included in the result ZIP or patch.

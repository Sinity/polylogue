# Semantic transcript rendering — implementation handoff

## Mission and delivered slice

This package implements the `beads-05` semantic transcript rendering mission for Beads `polylogue-ap7.1` and `polylogue-395j`. It extends the semantic-card core already landed on the snapshot into one ordered, provider-neutral transcript document shared by CLI Markdown and the daemon/web reader. The document represents typed prose, compact notices, tools/results, attachments, lineage, authoredness, variants, inherited prefixes, duration, and exact source coordinates without deriving semantics from prose.

The patch is one cohesive change against snapshot commit `f654480cadb7cc4c194704e24dfd483199547b35` on branch `master`. Apply `PATCH.diff` as a single unit. No complete replacement files are included because the unified diff is self-contained and was byte-compared after a clean apply.

## Snapshot identity

The attached project-state archive reported:

| Field | Value |
|---|---|
| Branch | `master` |
| Commit | `f654480cadb7cc4c194704e24dfd483199547b35` |
| Manifest generated | `2026-07-17T043202Z` |
| Manifest dirty flag | `true` |
| Captured branch-delta patch | empty, 0 bytes |
| Captured branch-delta file list | empty, 0 bytes |
| Captured branch-only log | empty, 0 bytes |

The working-tree payload was selective rather than deletion-authoritative. Reconstruction therefore used the attached Git bundle at the named commit and overlaid the payload non-destructively. No pre-existing tracked dirty change can be reconstructed from the archive because every captured branch-delta surface is empty. This patch contains only the implementation described here.

## Evidence inspected

Repository instructions and architecture were read from `AGENTS.md`/`CLAUDE.md`, including the substrate-first rule, typed `Origin` and `SemanticBlockType` vocabularies, lineage composition model, focused-test policy, and the requirement to regenerate topology inventories after adding a `polylogue/` module.

The complete Beads records inspected were:

- `polylogue-ap7.1`, “Complete semantic-card family coverage and bounded lineage parity.” Its acceptance criteria require exhaustive normalized-family/origin policy, schema-valid cards for all named families, shared CLI/web structure, bounded DB/archive lineage parity, and preservation of recipient-addressed ChatGPT tool structure.
- `polylogue-395j`, “Web/CLI transcript renderer gives zero-content ChatGPT thinking blocks full message-card weight.” Its later design note requires one compact absence marker per contiguous typed run, retained provenance, non-empty thinking rendered normally, and no provider/prose heuristic.

Relevant source and tests inspected include `polylogue/core/enums.py`, `polylogue/rendering/{block_models,semantic_card_models,semantic_card_registry,semantic_cards,semantic_card_placement,semantic_markdown}.py`, `polylogue/cli/messages.py`, the actual daemon session-detail and paginated-message routes in `polylogue/daemon/http.py`, the web shell reader fragments, archive composition in `polylogue/storage/sqlite/archive_tiers/write.py`, ChatGPT parser regressions, existing semantic fixtures, CLI tests, lineage normalization tests, and visual reader tests.

Relevant history inspected:

| Commit | Finding |
|---|---|
| `0f5059068` / PR #2700 | Landed the pure semantic-card contract, CLI Markdown backend, shell/edit/task/attachment core, fixtures, and card schema. Explicitly left broader coverage and web parity open. |
| `0c251b600` / PR #2736 | Wired card-only placement into the web reader for the full-detail route and retained presentation-leaf fallbacks. Explicitly covered only shell/edit/task. |
| `7688d103d` / PR #2629 | Corrected ChatGPT recipient-addressed calls to typed `TOOL_USE` rather than raw JSON prose. |
| `c06ca601c` / PR #2603 | Added exact lineage-completeness and truncation signals to composed archive reads. |
| `9ebc09ef0` / PR #2759 | Established bounded delegation evidence as a separate query/read projection; this patch does not duplicate that work-graph model. |

## Mechanism

### One ordered transcript contract

`SemanticTranscript` now contains ordered `SemanticTranscriptEntry` values, exactly one of:

- typed `TranscriptProse`, retaining role, message type, material origin, provider family/origin, block identity/type/language, timestamp, duration, parent identity, variant, active-path/leaf state, and inherited-prefix state;
- a `SemanticCard`, retaining machine coordinates, structural outcome, bounded previews, raw input/result evidence, and explicit caveats;
- a typed `TranscriptNotice`, currently `empty_thinking`, retaining every represented block coordinate.

The public wire contract is `semantic-transcript.v1`, defined by the new self-contained `docs/schemas/semantic-transcript-v1.schema.json`. `semantic-card.v1` remains the nested card contract and was expanded for all card kinds and exact source metadata.

### Structural classification and complete policy

The shared registry now covers every persisted `SemanticBlockType` and every executable `Origin`. Classification precedence is explicit:

1. structurally valid `mcp__<server>__<tool>` identity;
2. persisted `semantic_type` from normalization;
3. evidence-backed exact provider/tool alias;
4. generic fallback with raw evidence and a reason.

This prevents tool prose, output text, or open provider namespaces from inventing semantics. Card kinds now cover `shell`, `file_read`, `file_edit`, `search`, `web`, `task`, `mcp`, `lineage`, `attachment`, and `fallback`. The generated registry documents disclose open namespaces and fallback behavior.

MCP identity parsing was factored into `polylogue/core/tool_identity.py` and reused by rendering and tool-usage insights, removing duplicate structural parsing.

### Tool/result pairing and mixed rows

Tool uses and results are indexed by exact `tool_id`, then paired FIFO by structural coordinates. Pairing is independent of serialization order, so a result serialized before its use is absorbed once rather than emitted as both an orphan and a paired preview. A result message is suppressed only when it has no independently meaningful prose, context, attachment, or notice entry; mixed protocol/context rows retain their independent content.

Structural outcome remains trivalent. `tool_result_is_error` and `tool_result_exit_code` establish success/failure; absent evidence remains unknown. Conflicting fields remain failure with an explicit caveat. Output prose never controls outcome.

### Empty thinking

Empty or whitespace-only typed `thinking` and `reasoning` blocks form a contiguous absence run. The renderer emits one compact `empty_thinking` notice for the run and retains every message/block coordinate plus authoredness and lineage metadata. Any typed non-thinking row terminates the run, even when that row is absorbed into a paired card. Non-empty thinking remains a typed prose entry and is foldable in the browser.

### Attachments and exact metadata

Both typed attachment blocks and message-envelope attachments become attachment cards. Duplicate envelope/block representations are structurally de-duplicated. Attachment bytes are deliberately not embedded.

Generic block coercion no longer imports storage runtime types. It projects block-shaped objects structurally and parses persisted metadata JSON. During the final audit, a regression in this decoupling was found and repaired: `language`, URL, name, and media type are again retained from metadata-backed runtime records. A focused test fails if that fallback is removed.

### Bounded lineage

DB-backed reads project only facts already present on the session row and explicitly label root/completeness as unavailable where the row has no authority. They no longer hydrate a topology family solely to render a lineage card.

Archive-backed composed reads now retain `source_session_id` on every composed message and expose exact inheritance mode and branch-point identity on `ArchiveSessionEnvelope`. This lets the renderer identify inherited-prefix evidence without guessing by position. Archive lineage includes exact completeness and truncation reason from the existing composition authority; `resolved` is asserted only where parent, prefix inheritance, and complete composition are all established.

### Surface wiring

CLI Markdown renders the complete ordered transcript through `render_semantic_transcript_markdown`.

Both daemon full-detail routes and the actual paginated `/api/sessions/{id}/messages` routes attach authoritative `semantic_entries`. Compatibility `semantic_cards` and suppression fields remain for older consumers. The browser treats the presence of `semantic_entries`, including an empty array, as authoritative; raw role/text classification is only a legacy payload fallback. Session-level lineage entries survive initial load and pagination. Attachments are not rendered a second time when semantic entries are present.

## Key decisions

| Decision | Reason |
|---|---|
| Preserve `semantic-card.v1` and add `semantic-transcript.v1` | Existing card consumers remain compatible while prose/notices gain one ordered contract. |
| Keep compatibility card fields on daemon payloads | Avoid a breaking web/API cutover; the browser prefers the new authoritative field. |
| Use typed absence notices rather than omitting empty thinking | Operator views become compact without losing evidence that the provider exported a typed but unavailable block. |
| Pair structurally before rendering | Supports result-before-use serialization and prevents presentation-order bugs. |
| Retain raw evidence on fallback cards | Unknown open-namespace tools remain inspectable without guessed semantics. |
| Use exact archive composition provenance | Inherited prefixes are facts established by composition, not inferred from message position. |
| Use session-row lineage for ordinary DB reads | Satisfies boundedness and avoids whole-family topology hydration. |
| Build paginated DB placement from the complete session | Preserves cross-page tool/result identity with the current API; this is the principal performance risk and the best candidate for a storage-level follow-up. |
| Keep delegation as an existing separate bounded projection | Avoids duplicating the work-graph semantics landed by PR #2759. |
| Force generated text files to text in `PATCH.diff` | Repository attributes present some generated text as binary; an apply-ready reviewable unified diff is required here. |

## Changed files

The patch has 49 diff records spanning 50 paths, including one detected rename, with 4,119 insertions and 536 deletions.

Production and contracts:

- `polylogue/core/tool_identity.py`
- `polylogue/insights/tool_usage.py`
- `polylogue/rendering/block_models.py`
- `polylogue/rendering/semantic_card_models.py`
- `polylogue/rendering/semantic_card_registry.py`
- `polylogue/rendering/semantic_cards.py`
- `polylogue/rendering/semantic_card_placement.py`
- `polylogue/rendering/semantic_markdown.py`
- `polylogue/cli/messages.py`
- `polylogue/daemon/http.py`
- `polylogue/daemon/web_shell.py`
- `polylogue/daemon/web_shell_reader.py`
- `polylogue/daemon/web_shell_semantic_cards.py`
- `polylogue/storage/sqlite/archive_tiers/write.py`
- `docs/schemas/semantic-card-v1.schema.json`
- `docs/schemas/semantic-transcript-v1.schema.json`

Generated surfaces:

- `docs/generated/semantic-card-tool-map.json`
- `docs/generated/semantic-card-tool-map.md`
- `docs/plans/topology-target.yaml`
- `docs/topology-status.md`

Fixture and generator changes:

- `devtools/render_semantic_card_fixtures.py`
- `devtools/render_semantic_card_registry.py`
- all prior semantic-card cases updated for exact source metadata;
- `chatgpt-web-fallback.json` renamed to `chatgpt-web.json` because web is now structurally classified;
- new cases: `codex-search.json`, `empty-thinking-run.json`, `gemini-file-read.json`, `mcp-tool.json`, `message-attachment-envelope.json`, `mixed-tool-result-context.json`, and `typed-prose-metadata.json`.

Tests:

- `tests/unit/core/test_tool_identity.py`
- `tests/unit/rendering/test_semantic_cards.py`
- `tests/unit/cli/test_messages.py`
- `tests/unit/storage/test_lineage_normalization.py`
- `tests/visual/test_reader_semantic_cards.py`

## Acceptance matrix

| Requirement | Status | Evidence |
|---|---|---|
| Every persisted semantic family has an explicit card/fallback policy | Satisfied | Generated semantic-type policy; exhaustive enum assertion; registry check. |
| Every executable `Origin` has an explicit namespace/fallback policy | Satisfied | Generated origin policy; test compares policy set with `set(Origin)`. |
| Shell, read, edit/write, search, web, task, MCP, attachment, lineage, unknown | Satisfied | All ten card kinds occur in the 23-case corpus and validate against `semantic-card.v1`. |
| Structural outcome, target, refs, previews, unknown/missing states | Satisfied | Golden fixtures and 96 focused tests; outcome tests use only typed fields. |
| CLI and web consume one document | Satisfied in implementation; runtime route test not executed here | CLI and daemon import the same renderer; browser consumes `semantic_entries`; visual test targets the actual paginated route. |
| DB/archive bounded lineage parity without family hydration | Satisfied for lineage mechanism; storage route execution partially unverified | DB uses session-row authority; archive uses exact envelope authority; renderer tests cover complete/partial states; storage provenance test was added but could not run in this dependency-incomplete runtime. |
| Recipient-addressed ChatGPT tool calls remain typed and raw JSON does not leak | Preserved from current source; focused reparse unverified | PR #2629 parser path remains authoritative; tool-envelope message text is suppressed when a typed `TOOL_USE` block exists. |
| Empty thinking is compact in CLI and web, non-empty thinking remains normal | Satisfied in implementation and fixtures; live private-session check unverified | `empty-thinking-run.json`, ordered transcript assertions, Markdown and browser notice renderers. |
| Mixed protocol/context rows retain context | Satisfied | `mixed-tool-result-context.json`; suppression requires absence of independent entries. |
| Variants, active state, inherited prefixes, duration, authoredness | Satisfied | Exact metadata is present in prose/notices/card sources and covered by fixtures/tests. |
| Real attachments | Satisfied at normalized block/envelope level | Block and message-envelope attachment fixtures; bytes remain outside transcript by design. |
| Superseded renderer duplication removed | Satisfied for semantic classification | Browser leaf no longer classifies authoritative payloads; shared MCP parser removes duplicate protocol parsing. Legacy fallback remains intentionally for older payloads. |

## Apply order

1. Start from clean commit `f654480cadb7cc4c194704e24dfd483199547b35`.
2. Run `git apply --check PATCH.diff`.
3. Apply once with `git apply PATCH.diff`.
4. Enter the repository’s locked development environment.
5. Run the focused commands in `TESTS.md`, then the normal `devtools verify` baseline.
6. Manually exercise a real ChatGPT export containing unavailable reasoning blocks and the daemon/browser reader before merge.

The package generation process performed both `git apply --check` and a real apply into a fresh clone. Every changed path was then compared byte-for-byte with the implementation workspace.

## Verification performed

- Focused renderer suite: **96 passed** in 2.29 seconds. Two warnings report unavailable pytest timeout options in this reduced environment; no tests failed.
- Frozen semantic fixtures: **23 verified**.
- Generated semantic registry: **2 surfaces verified**.
- Topology projection/status: regenerated and checked; projection contains **1,008 rows**, with 9 existing TBD ownership rows.
- Public schemas: both Draft 2020-12 schemas are structurally valid.
- Fixture schema validation: **23 transcript documents** and **22 nested card entries** validated.
- Shared MCP identity tests: **2 executed** directly.
- Changed Python files: **21 compiled**.
- Browser JavaScript: reader fragment, semantic renderer fragment, and assembled shell script all passed `node --check`.
- `git diff --check`: clean.
- `PATCH.diff`: 275,889 bytes, 7,265 lines, no binary-patch marker, no copied input archive path, and no implementation placeholder marker.
- Clean-base `git apply --check`: passed.
- Clean-base actual apply and byte comparison: passed for all **50 changed paths**.

## Important limitations and risks

The complete repository environment was not available offline. `devtools render all --check` stops during import because `ijson` is missing. Normal CLI, storage, and visual-test collection also encounters missing runtime/test dependencies such as `dateparser`, `aiosqlite`, `tenacity`, and `hypothesis`. Ruff and mypy executables are absent. Therefore full generated-surface verification, strict typing, repository lint/format gates, the added storage test, the actual daemon route test, and the CLI test remain unverified in this runtime. The patch compiles and the independently runnable semantic suite is green, but the normal locked environment must run those gates.

No operator daemon, browser session, Nix deployment, secrets, private archive, or unsanitized provider export was available. The required manual check against a real ChatGPT reasoning export is unverified. Exact `reasoning` subtype behavior is covered by typed renderer logic and fixtures, not a live export in this environment.

The paginated DB route currently hydrates the complete session to preserve tool/result pairing across page boundaries. This is semantically correct but may be expensive for very large sessions. The substantial follow-up is a bounded storage/API pairing index or action-coordinate projection that lets a page retain cross-boundary identity without full-session hydration.

DB session rows do not carry exact root identity, composition completeness, branch point, or active leaf, so those fields are explicitly unknown rather than guessed. Archive-backed reads carry stronger authority. This asymmetry is deliberate and visible.

Legacy attachment byte envelopes remain separate from normalized attachment-card rendering. Attachment identity/metadata is represented; bytes are not copied into transcript documents.

## Value of another iteration

A small repair pass has high confidence and limited scope: run the locked `devtools verify`/Ruff/mypy/testmon gates, execute the added CLI/storage/visual tests, inspect any formatting/type findings, and perform the real ChatGPT/browser manual check. That pass should not require architectural changes unless the unavailable gates expose one.

A substantial second pass would add a bounded cross-page tool/result pairing projection in storage/API, converge legacy attachment envelopes with normalized attachment identity, and expose stronger DB lineage authority. That work could improve performance and parity materially, but it is larger than the remaining certification work and should be tracked independently rather than folded into this patch.

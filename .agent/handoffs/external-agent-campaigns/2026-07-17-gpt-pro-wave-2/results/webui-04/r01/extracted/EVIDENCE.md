# WebUI v2 transcript renderer — source, tracker, and history evidence

## Authority order used

Evidence was adjudicated in this order:

1. current production source and executable tests at snapshot commit `536a53efac0cbe4a2473ad379e4db49ef3fce74d`;
2. root repository instructions (`AGENTS.md` / `CLAUDE.md`);
3. current open/closed Bead descriptions, acceptance criteria, notes, and dated investigation comments;
4. Git history and tree contents at named commits;
5. generated snapshot reports and handoff receipts;
6. mission prose where the repository had no current mechanism.

When a generated receipt contradicted the current tree and named merge commit, source/history won. Assumptions introduced because a predecessor WebUI scaffold was absent are listed explicitly rather than presented as repository facts.

## Snapshot evidence

| Evidence | Observed fact |
|---|---|
| `git rev-parse HEAD` | `536a53efac0cbe4a2473ad379e4db49ef3fce74d` |
| branch recorded by snapshot | `master` |
| commit subject | `fix(repair): harden raw authority convergence (#3046)` |
| commit date | `2026-07-17T18:55:47+02:00` |
| snapshot generation | `2026-07-17T180950Z` |
| snapshot overview | source checkout marked dirty |
| supplied branch-delta patch | zero bytes |
| supplied branch-delta file list | zero bytes |
| supplied branch-delta log | zero bytes |
| outer archive SHA-256 | `ec3fa193b2f99a6daee0bc620af4f69b5728834e99f8196792a17c3fbae11155` |
| working-tree archive SHA-256 | `9b0664c982b58a980e52f47af7a7466f6f5f3b3b3cf4914c16dba232639bc8bf` |

The repository was reconstructed from the supplied all-refs bundle and working-tree archive. The supplied artifacts represented no pre-task file delta relative to the named commit. The patch is therefore valid against that commit; the original checkout's unexplained dirty flag remains an acknowledged provenance limit.

## Repository invariants applied

### Semantics belong below surfaces

Root instructions state that substrate/product layers own meaning and surfaces are leaf adapters. For this task, `polylogue/rendering/semantic_cards.py` and its models/registry are the product-layer semantic owner. The daemon and browser may serialize or present those semantics but may not introduce a provider classifier.

### `material_origin` is the authoredness axis

Root instructions identify `messages.material_origin` as the axis that distinguishes human/assistant authorship from provider protocol, runtime context, tool results, generated packs, and operator commands. Provider `role=user` alone is not evidence of human authorship. The document, SSR, and Preact surfaces therefore expose role and `material_origin` separately and label protocol/runtime rows explicitly.

### Outcomes must come from provider structure

`blocks.tool_result_is_error` and `blocks.tool_result_exit_code` are the load-bearing structural result fields. `NULL` means unknown. The implementation preserves optional booleans/integers and never scans result prose for success/failure.

### Exact references are evidence coordinates

`polylogue/core/refs.py::EvidenceRef` defines session/message/block coordinates. Detail disclosure must preserve that coordinate and may not silently widen it. This invariant produced the block-ref/`message_text` no-widening test.

### Offline/package-local browser posture

Current WebUI architecture evidence requires TypeScript + Preact + Vite, committed assets, SSR-first typed JSON, progressive islands, and no CDN. The patch uses only lockfile-resolved local dependencies and package-local built files.

## Canonical rendering source

### Models and contract

`polylogue/rendering/semantic_card_models.py` already defined:

- the closed semantic card family vocabulary;
- structural outcome states;
- exact source coordinates and result coordinates;
- bounded preview facts and strategies;
- lineage descriptors;
- ordered prose/card/notice transcript entries.

The patch adds transport types for:

- `SemanticCardDisclosure`;
- `SemanticCardDocumentPage`;
- `SemanticCardDocumentMessage`;
- `SemanticCardDocument`;
- `SemanticCardDetailChunk`;
- detail part/state and result-evidence-state enums;
- document/detail schema version constants.

These types serialize through `to_document()` into strict JSON-compatible structures. They do not own family classification.

### Classification and pairing

`polylogue/rendering/semantic_card_registry.py::classify_tool` is the provider-neutral classifier. It maps normalized tool evidence to the closed card family vocabulary.

`polylogue/rendering/semantic_cards.py::build_semantic_transcript` owns:

- message/block normalization;
- structural FIFO tool-use/result pairing by `tool_id`, independent of source order;
- source/result evidence coordinates;
- provider error/exit outcome derivation;
- family-specific card construction;
- bounded previews;
- fallback raw evidence;
- prose, attachment, lineage, and empty-thinking notices.

The new `build_semantic_card_document` calls that ordered transcript builder before adding paging, row placement, display bounds, omission records, and disclosure links. The browser source imports only document types; it never imports or recreates the classifier.

### Placement

`polylogue/rendering/semantic_card_placement.py` maps canonical transcript entries to source message rows and session-level placement. The patch introduces/reuses `semantic_card_placement_from_transcript(...)`, preventing the document serializer from rebuilding the transcript merely to determine row ownership. Pure paired-result rows are marked suppressed while their exact result evidence remains on the invocation card.

### Terminal parity

`polylogue/rendering/semantic_markdown.py` remains the terminal projection. The patch adds an explicit `card family` line alongside existing structural outcome facts so tests can compare terminal and web output without scraping presentation-specific headings.

### Outcome honesty

`SemanticCardOutcome.to_document()` omits `is_error` and `exit_code` when unknown. The document serializers and UI preserve `state=unknown` and absent optional facts. No code path turns absent error data into `false`, absent exit code into `0`, or missing result evidence into success.

### Exact detail

`build_semantic_card_detail(...)`:

- parses one exact ref;
- rejects a different session;
- requires a message-level ref for `message_text`;
- requires an exact block for block/input/raw/diff parts;
- returns typed `missing` for absent coordinates/data;
- returns typed `unknown` for unsupported parts;
- clamps one response to 64,000 Unicode code points;
- reports total, next offset, completion, and replacement count consistently.

This mechanism is directly tested with an emoji/non-BMP continuation and the scope-widening mutation control.

## Public schemas

The patch adds strict Draft 2020-12 schemas:

- `docs/schemas/semantic-card-document-v1.schema.json`;
- `docs/schemas/semantic-card-detail-v1.schema.json`.

Evidence from the schema and validators:

- root and nested typed objects reject unknown properties;
- page limit and returned rows are at most 200;
- card family is closed to ten values;
- title/summary/field/source/preview/caveat bounds are explicit;
- omission counts are structurally linked to truncation flags;
- result evidence ref is required exactly for `present` and prohibited otherwise;
- prose carries a bounded preview, not unbounded text;
- detail `available` and unavailable states have mutually exclusive shapes;
- detail text and returned count are bounded to 64,000.

Python tests validate every canonical fixture against the checked-in document schema. The TypeScript validator mirrors the schema and adds cross-field/page/ref invariants that JSON Schema alone does not conveniently express.

## Existing daemon and web evidence

### Two read authorities

`polylogue/daemon/http.py` has:

- a live/database path through `Polylogue.get_session(...)`;
- a split-archive path through `archive_read_context(...)` and archive session envelopes.

The old session-detail code flattens message text differently across those authorities, tracked by `polylogue-6o9b`. The new card routes avoid either backend's legacy flattened-message classifier and pass structured message/block data into the same renderer.

### Legacy browser paths

`polylogue/daemon/web_shell_reader.py::renderMessageBlocks` is a client-side per-message heuristic using flattened text and block hints. It calls legacy semantic-card HTML first and then applies fallback rendering behavior.

`polylogue/daemon/web_shell_semantic_cards.py::_polySemanticCardsHtml` is a separate legacy semantic-card renderer over per-message `semantic_entries`.

The new functions in `web_shell_semantic_cards.py` are different in kind: they accept a serialized `semantic-card-document.v1` and project HTML. They cannot classify provider blocks because provider blocks are not an input.

### Auth and credential lifecycle

The daemon already provides:

- loopback host admission;
- API bearer authentication;
- origin-bound browser credentials;
- `/api/web-auth/session` credential establishment.

The new shell reuses that lifecycle. It does not expose archive data merely because the HTML shell is loopback-bootstrappable. The real route test confirms an anonymous token-configured first response is data-free and the API remains unauthorized.

### Route/privacy headers

Current source shows:

- transcript JSON/HTML: no-store, no-referrer, nosniff;
- HTML frame denial;
- local asset allowlist with no-cache/no-referrer/nosniff;
- script-safe JSON escaping;
- no runtime CDN references.

The archive-backed route test verifies these headers and payload behavior over a real loopback server.

## Five live render implementations

The 2026-07-16 comment on `polylogue-4p1`, rechecked against current source, records five active message/block-to-output implementations:

1. generic HTML through `rendering/renderers/html.py`, `html_messages.py`, `core_messages.py`, and `blocks.render_blocks_html`;
2. generic Markdown through `rendering/core_markdown.py` and `blocks.render_blocks_markdown`;
3. legacy web semantic cards through `_polySemanticCardsHtml`;
4. legacy web flattened-message fallback through `renderMessageBlocks`;
5. semantic CLI Markdown through `cli/messages.py` and `rendering/semantic_markdown.py`.

The same investigation identifies a dead sixth:

- `build_projection_html_messages` fed by `get_render_projection`, tracked by `polylogue-a820`.

Source search confirms no production caller for the dead pair. This patch does not delete it because the mission is a coherent transcript-render vertical, not a broad dead-code cleanup.

## Relevant Beads adjudication

### `polylogue-ap7` — open

Title: `Semantic transcript renderer registry across normalized tool families`.

Current description requires shell, edit/write, file read/search, task/delegation, web, MCP, attachment, lineage, and unknown tools through one provider-neutral registry shared by terminal and web.

Acceptance emphasizes:

- complete generated/tested family coverage;
- same card document/schema for CLI and web;
- no backend reclassification;
- archive/database lineage parity;
- bounded previews and exact refs.

Decision applied: the canonical ordered transcript remains the semantic owner; new surfaces consume its document. Full non-hydrating lineage parity remains open because the future bounded read relation is absent.

### `polylogue-ap7.1` — open

Title: `Complete semantic-card family coverage and bounded lineage parity`.

Acceptance explicitly asks for every declared family/fallback, correct structural outcome/target/ref/bounded preview/missing state, CLI/web agreement from one card document, and archive/database parity without full-family hydration.

Decision applied: complete all family projections, strict schema, parity tests, fallback retention, and one real archive-backed route. Do not claim the non-hydrating topology/read requirement; current document construction still builds the full canonical transcript.

### `polylogue-4p1` — open, priority 1 epic

Title: `Make Query × Projection × Render the sole executable read algebra`.

Acceptance requires one read request/executor and stable keyset message windows with exact omission/continuation metadata and non-hydrating first-useful-content behavior.

Its 2026-07-16 investigation updates the render-path count to five live plus one dead sixth and records concrete thinking/tool-result/code-block divergence.

Decision applied: introduce the smallest serializing seam over current canonical semantics, with explicit paging metadata and no new read algebra. Document full-transcript hydration as the main integration debt rather than inventing a parallel cursor model.

### `polylogue-7le` — closed and stale

Title: `Consolidate the three session->HTML paths`.

Its three-path count predates the semantic CLI lane and later re-inventory. The issue itself requested re-verification. `4p1`'s five-path count is the current authority.

Decision applied: use `7le` only as historical consolidation intent, not as the current inventory.

### `polylogue-bby.11` — open

Title: `Webui architecture v2: the stack that can carry the ambition`.

Ratified/refined architecture evidence:

- TypeScript + Preact + Vite;
- committed reproducible local assets;
- strict no-CDN/offline posture;
- daemon semantic HTML plus typed JSON on every route;
- progressive islands;
- SSR-first, not SPA-only;
- static export should share the reader's render contracts.

Current snapshot reality: no predecessor v2 scaffold, generated API client, token pipeline, or `devtools render webui` owner exists.

Decision applied: use the ratified stack, local assets, strict handwritten TypeScript validator, direct Vite build, and a localized `/v2/s` strangler route. Record these scaffold assumptions.

### `polylogue-bby.8` — open

Title: `Web reader perceived performance: virtualized list, streamed search, optimistic navigation`.

Its notes mark the original 16k list premise stale and separate server read semantics (`4p1`) from client measured virtualization.

Decision applied: measured variable-height virtualization and request-count guards now; no claim of operator-machine 60-fps performance or full navigation/cache scope.

### `polylogue-6o9b` — open

Title: database-backed and archive-backed session-detail routes compute different flattened `message.text`.

Decision applied: the new route consumes structured messages and one semantic builder, bypassing the old flattened-string divergence. The underlying legacy route bug is not silently declared fixed.

### `polylogue-a820` — open

Title: remove dead `build_projection_html_messages` / `get_render_projection` path.

Decision applied: identify it as the dead sixth path and a future deletion candidate. Do not expand this vertical with unrelated removal risk.

## History inspected

| Commit | Date | Subject | Relevance |
|---|---|---|---|
| `0f5059068f115754070da40df9a55f59c1b8b5d7` | 2026-07-11 | `feat(rendering): add semantic transcript evidence cards (#2700)` | Introduced semantic cards, registry/outcomes/evidence foundations. |
| `0c251b6005ac8ab6421235b0e8dcbf205e4aa133` | 2026-07-12 | `feat(rendering): wire semantic transcript cards into the web reader (#2736)` | Added the legacy per-message web semantic-card projection. |
| `fc770dbd9a16227037a51a6882dc5cca9ef4eda1` | 2026-07-17 | `feat(reader): render ordered semantic transcripts (#3016)` | Established ordered transcript semantics and placement. |
| `9163d0134f3d334960e4c249c96c5671919a9a06` | 2026-07-17 | `feat(query): bound agent-facing archive reads (#3018)` | Relevant bounded-read precedent, but not the requested browser keyset transaction. |

History confirms the ordered transcript is the newest semantic authority and the prior web wiring consumed per-message entries rather than a strict paged document.

## Resolved contradiction: `result-before-use` fixture

Two existing tests referenced:

`tests/data/semantic_cards/cases/result-before-use.json`

Snapshot facts before the patch:

- the file was absent from current HEAD;
- `git log --all -- <path>` contained no commit for it;
- `git ls-tree` showed it absent from merge commit `fc770dbd9` (#3016);
- the full semantic-card test module failed exactly those two tests with `FileNotFoundError`;
- a generated receipt claimed the missing ordering fixture had been added before admission.

Adjudication: source/history disproved the receipt claim. The fixture is directly relevant to this mission because whole-document structural pairing and page slicing must remain correct when a result serializes before its invocation.

Resolution in this patch:

- add a sanitized fixture with a result preceding its use;
- retain the existing tests unchanged;
- verify the result pairs once and the pure result row is suppressed;
- full semantic-card baseline now passes 99 tests;
- fixture generator now verifies 24 cases.

This is not an invented unrelated baseline fix; it restores the exact adversarial case already demanded by the canonical renderer tests and the new paged document's correctness argument.

## WebUI scaffold assumptions

The requested predecessor WebUI result was absent from the supplied source/artifacts. The implementation localizes these assumptions:

- route: `/v2/s/:session_id` as a strangler route rather than replacing `/` or the old `/s` reader;
- assets: `polylogue/daemon/static/webui-v2/` because no current generated `static/dist` owner exists;
- build: direct `npm run build` using Vite;
- API validation: strict handwritten TypeScript mirroring checked-in JSON Schema, pending a future generator;
- styles: transcript-local CSS, pending a ratified theme-token pipeline;
- old reader remains until interaction and deletion certification.

None of these assumptions changes semantic classification or evidence truth.

## Package and reproducibility evidence

### npm lock

The first reconstruction inherited environment-specific internal npm registry URLs in `package-lock.json`. Those URLs were normalized to canonical public `https://registry.npmjs.org/` entries. A clean offline install from the available cache then passed, proving the normalization did not change package integrity.

Final npm evidence:

- 89 packages installed from a clean `node_modules` state;
- strict TypeScript passed;
- 24 Vitest tests passed;
- audit reported zero vulnerabilities;
- no private/internal registry references remain.

### Vite assets

Two consecutive clean builds were byte-identical:

- JS SHA-256 `6cc8fdfbf38abd9ccd66de5b7015ac6b9c0fafd61815d32c7536bc44d39006db`, 58,928 bytes;
- CSS SHA-256 `dc39c243ba661da3fa859bbbc366ed146542908341949fb34c8e430dbba4ebdb`, 5,740 bytes.

The bundle contains standard W3C DOM namespace strings from Preact/runtime code but no runtime third-party fetch/CDN URL.

### Wheel

The offline wheel build produced:

- SHA-256 `19483aa51be3d16372879be9e47a3a595b27529b74a7e26891dd931ffef0b54a`;
- size 4,200,885 bytes;
- valid ZIP CRC;
- both transcript assets at their exact final sizes;
- no `uv.lock` change.

### Topology

`devtools verify topology` reported:

- `realized=1023`;
- `declared=1023`;
- `blocking=False`;
- zero missing, orphan, conflict, or kernel-rule findings;
- nine pre-existing storage files marked `tbd`.

The patch adds no new Python module outside existing declared packages that requires topology projection regeneration.

## Browser execution evidence

A real Chromium journey was attempted against a loopback fixture server. Navigation failed before application code with `net::ERR_BLOCKED_BY_ADMINISTRATOR`.

Environment evidence:

- managed policy file: `/etc/chromium/policies/managed/000_policy_merge.json`;
- policy contains `"URLBlocklist": ["*"]`;
- no independent bundled Playwright Chromium cache was present.

Adjudication: this is an environment-policy block, not a passed or failed application journey. The package reports HTTP/SSR and jsdom/component proof, and explicitly leaves real-browser/performance proof open.

## Source-supported inferences

1. **A strict page document is the smallest coherent new seam.** The ordered transcript already owns semantics but lacked message-window transport, exact detail continuations, and a typed SSR/island boundary.
2. **Whole-transcript pairing before slicing is currently correctness-preserving.** Tool results may precede uses and pairs may cross requested page boundaries. Slicing first would break canonical FIFO pairing unless a future read transaction supplies neighboring structural evidence.
3. **Client accumulation must not mutate a server page contract.** Server `returned` is bounded by one page. Accumulated UI state is therefore a distinct type with stable total and exact continuity checks.
4. **Canonical alias adoption is necessary for exact refs.** Continuing to request pages/details with a native-id alias after the server returns canonical identity can produce coordinate/session drift.
5. **Omission metadata is part of evidence honesty.** A bound without an omitted count and reconstructable detail link can make clipped provider metadata appear complete.
6. **Legacy browser deletion requires interaction parity, not only rendering parity.** `/v2/s` proves the semantic seam but does not yet reproduce every old shell navigation, search, overlay, and action affordance.
7. **Another small code-polish iteration has low value.** The material next work is bounded keyset integration, unrestricted browser/performance evidence, operator-approved deployment/archive validation, and old-path retirement.

## Unresolved evidence

- no operator private archive or live service was available;
- no unrestricted real browser, screenshot, or performance trace was available;
- no NixOS/Nix deployment was exercised;
- no current executable `4p1` keyset read transaction exists for this route;
- no predecessor WebUI v2 scaffold/generated client was present;
- no complete non-hydrating topology/delegation page projection was present;
- no full repository test-suite result is claimed;
- legacy reader deletion remains uncertified.

# WebUI v2 transcript renderer — final implementation handoff

## Mission outcome

This package delivers the requested WebUI v2 transcript-rendering vertical against Polylogue snapshot commit `536a53efac0cbe4a2473ad379e4db49ef3fce74d`. The implementation does not add a browser-side or daemon-side tool classifier. `polylogue/rendering/semantic_cards.py` remains the sole owner of semantic family classification, structural tool-use/result pairing, provider-reported outcomes, and canonical card structure. The daemon serializes that canonical transcript into a strict, bounded page document; server-rendered HTML and the TypeScript/Preact island are projections of the serialized document.

The vertical includes:

- strict `semantic-card-document.v1` and `semantic-card-detail.v1` contracts with checked-in JSON Schemas;
- authenticated JSON and server-rendered HTML routes;
- `/v2/s/:session_id` with data-free first-visit credential bootstrap when read authority is absent and SSR when authority is present;
- local, committed, reproducible Preact/Vite assets with zero CDN dependencies;
- all ten canonical card families, explicit role versus `material_origin`, structural result states, child-session links, fallback evidence, bounded previews, exact-ref disclosure, and measured variable-height virtualization;
- anti-fragmentation parity tests over failed results, fork/compaction lineage, and unknown provider tools;
- a real archive-backed loopback HTTP test that crosses the production writer, archive reader, semantic renderer, route dispatcher, SSR projection, exact-detail endpoint, and local asset server;
- restoration of the sanitized `result-before-use` fixture required by two pre-existing canonical pairing tests.

This is a strangler vertical. It supersedes the two legacy browser render paths only for `/v2/s/:session_id`; it does not delete the old reader, generic HTML/Markdown renderers, or the semantic CLI renderer.

## Snapshot identity

| Item | Observed value |
|---|---|
| Project | `polylogue` |
| Snapshot branch | `master` |
| Snapshot commit | `536a53efac0cbe4a2473ad379e4db49ef3fce74d` |
| Commit subject | `fix(repair): harden raw authority convergence (#3046)` |
| Commit timestamp | `2026-07-17T18:55:47+02:00` |
| Snapshot generation timestamp | `2026-07-17T180950Z` |
| Supplied outer archive SHA-256 | `ec3fa193b2f99a6daee0bc620af4f69b5728834e99f8196792a17c3fbae11155` |
| Supplied working-tree archive SHA-256 | `9b0664c982b58a980e52f47af7a7466f6f5f3b3b3cf4914c16dba232639bc8bf` |
| Implementation branch used locally | `feature/webui-v2-transcript-r03` |
| Patch base | exact detached snapshot commit above |
| Patch SHA-256 | `73a9d9bcb94d1e52d8b0a2fe8d3315ba1a3d877cf76fbf66b57a9b0ead6e0598` |
| Patch size | 387,123 bytes |
| Patch scope | 30 files; 9,673 insertions; 38 deletions |

The snapshot overview marks the source checkout dirty. Its supplied branch-delta patch, branch-delta file list, and branch-delta log are zero bytes. Overlaying the supplied working-tree archive on the bundled commit produced no represented pre-task delta. This package therefore names and targets the exact commit while preserving the narrower fact that an unrepresented source-checkout dirtiness flag existed.

## Evidence inspected

Repository instructions and architecture:

- root `AGENTS.md` / `CLAUDE.md`;
- `material_origin` authoredness and provider-structure rules in `polylogue/core/enums.py`;
- exact references in `polylogue/core/refs.py`;
- archive identity, split-tier read authority, package topology, and generated-file conventions.

Canonical rendering and terminal projection:

- `polylogue/rendering/semantic_card_registry.py`;
- `polylogue/rendering/semantic_card_models.py`;
- `polylogue/rendering/semantic_cards.py`;
- `polylogue/rendering/semantic_card_placement.py`;
- `polylogue/rendering/semantic_markdown.py`;
- generic HTML/Markdown block renderers and CLI read-view call sites.

Daemon and browser paths:

- `polylogue/daemon/http.py` and `route_contracts.py`;
- `web_shell.py`, `web_shell_reader.py`, and `web_shell_semantic_cards.py`;
- database-backed and archive-backed session readers;
- existing web credential lifecycle and loopback host-admission rules;
- `webui/` TypeScript, Playwright, npm lock, Vite packaging, and wheel build hooks.

Tests and fixtures:

- semantic-card fixture generator and all canonical fixture cases;
- semantic renderer and placement tests;
- daemon route, metrics, host-admission, credential, and HTTP security tests;
- visual semantic-card reader snapshots;
- current browser test scaffold.

Tracker and history:

- Beads `polylogue-ap7`, `polylogue-ap7.1`, `polylogue-4p1`, `polylogue-7le`, `polylogue-bby.11`, `polylogue-bby.8`, `polylogue-6o9b`, and `polylogue-a820`;
- commits `0f5059068` (#2700), `0c251b600` (#2736), `fc770dbd9` (#3016), and `9163d0134` (#3018).

## Mechanism

### 1. Shared rendering seam

`build_semantic_card_document(...)` is a transport seam over the canonical ordered transcript. It:

1. normalizes message and block envelopes;
2. calls `build_semantic_transcript(...)` exactly once for family classification, structural pairing, outcome derivation, fallback retention, attachment/lineage handling, prose, and typed notices;
3. obtains row ownership and paired-result suppression from `semantic_card_placement_from_transcript(...)`;
4. serializes a bounded message page with exact evidence coordinates and omission metadata.

It does not call `classify_tool(...)`, inspect provider tool names to choose a family, regex output text to infer success/failure, or duplicate semantic registry tables.

`build_semantic_card_detail(...)` is the exact-ref disclosure seam. It parses `EvidenceRef`, verifies session identity, requires the coordinate appropriate to the requested part, and returns one server-clamped Unicode-code-point window. A block-level ref cannot request `message_text` and receive the broader parent message; that request returns typed `missing`. Unsupported parts return typed `unknown`. Valid absent coordinates return typed `missing`.

The only canonical family-field addition is structural task/delegation linkage: normalized `child_session_id`, `subagent_session_id`, or `session_id` input is exposed as the canonical `child session` field. SSR and Preact merely link that field to `/v2/s/<encoded-id>`.

### 2. Evidence honesty and bounds

Provider-reported `tool_result_is_error` and `tool_result_exit_code` remain structural facts. `None` remains unknown. The document and UI never coerce absent error data to success or an absent exit code to zero.

Server document bounds:

- default page size: 40 message rows;
- maximum page size: 200 message rows;
- card title: 512 Unicode code points;
- card summary and field value: 4,096 code points;
- source metadata values: 4,096 code points with typed per-field omission records;
- preview text: 16,000 code points;
- maximum card fields: 64;
- maximum card previews: 32;
- maximum card caveats: 64;
- detail request: default 16,000 and maximum 64,000 code points;
- client accumulated exact detail: maximum 128,000 code points.

Exact identities and source-complete entry arrays are not presentation-truncated. IDs remain usable as identity keys and evidence coordinates. Message page size, every display-bearing value, every preview, and each detail response are bounded. Unknown tools always retain bounded raw input/result evidence when present.

Character counts use Python Unicode code points on the server and `Array.from(...)` in TypeScript. Non-BMP characters therefore advance continuation offsets by one logical character on both sides.

### 3. Daemon routes and authority

New route contracts:

- `GET /v2/s/:session_id`
- `GET /assets/webui-v2/transcript.js`
- `GET /assets/webui-v2/transcript.css`
- `GET /api/sessions/:id/card-document?offset=<n>&limit=<n>&format=json`
- `GET /api/sessions/:id/card-document?offset=<n>&limit=<n>&format=html`
- `GET /api/sessions/:id/card-detail?ref=<EvidenceRef>&part=<part>&offset=<n>&limit=<n>`

The card document/detail API routes require normal read authority. Both database-backed and archive-backed handlers pass structured messages and lineage into the same renderer. Native-id aliases are resolved to the canonical archive session identity before SSR and before subsequent client pagination/detail requests.

Security and privacy behavior:

- JSON and HTML transcript responses use `Cache-Control: no-store`, `Referrer-Policy: no-referrer`, and `X-Content-Type-Options: nosniff`;
- HTML additionally uses `X-Frame-Options: DENY`;
- static assets are an explicit allowlist and use `Cache-Control: no-cache`, `Referrer-Policy: no-referrer`, and `nosniff`;
- all assets are package-local; no CDN or runtime third-party fetch is introduced;
- embedded JSON escapes `<`, `>`, `&`, U+2028, and U+2029 so untrusted text cannot terminate the script element.

When no daemon token is configured, or a bearer/origin-bound browser credential already grants read authority, `/v2/s/:session_id` embeds page zero and useful SSR HTML. With a configured token and no established credential, the first response contains no transcript data. The island invokes the existing same-origin `/api/web-auth/session` credential lifecycle and then fetches the canonical document.

### 4. SSR and Preact island

`render_semantic_card_document_html(...)` accepts only the serialized card document. It does not accept raw provider blocks or tool names for classification. The HTML remains useful without JavaScript and visibly includes:

- canonical family and structural outcome;
- shell command, error flag, and exit code;
- file target/path;
- task child-session links;
- role, message type, and `material_origin` as independent axes;
- explicit protocol/runtime authoredness labeling;
- omission counts and exact-ref disclosure controls;
- fallback raw evidence.

The Preact island validates embedded and fetched JSON recursively before rendering. Page zero is consumed from SSR without refetch. Client accumulation is a separate `LoadedTranscript` state rather than a mutated server `CardDocument`; this prevents accumulated `returned` counts from violating the page schema.

Pagination rejects:

- a nonzero initial page;
- a different canonical session id;
- a changed `total_messages` value;
- a noncontiguous offset;
- duplicate message ids;
- repeated session-level entries on later pages;
- incoherent `next_offset` or page counts.

A request-in-flight guard prevents duplicate page calls from rapid clicks.

Long loaded transcripts use `MeasuredVirtualList`: `ResizeObserver` records actual row heights, prefix offsets determine the visible window, and overscan keeps mounted DOM rows bounded. This replaces the first iteration's fixed-height approximation.

Exact-detail expansion verifies schema, session, ref, part, offset, count, total, completion, and continuation metadata. It rejects coordinate drift, changed totals, impossible completion flags, and overlapping or skipped chunks. Duplicate in-flight detail calls are suppressed.

## Load-bearing JSON contracts

The normative schemas are:

- `docs/schemas/semantic-card-document-v1.schema.json`
- `docs/schemas/semantic-card-detail-v1.schema.json`

Both use JSON Schema Draft 2020-12 and reject unknown properties at every typed object boundary.

### `semantic-card-document.v1` root

All root fields are required.

| Field | Type and bound | Meaning |
|---|---|---|
| `schema_version` | constant `semantic-card-document.v1` | Contract discriminator. |
| `session_id` | non-empty string | Canonical session identity, including after alias resolution. |
| `page` | page object | One bounded message-row window. |
| `session_entries` | array of entry objects | Session-level lineage/other entries; emitted only at offset zero. |
| `messages` | array, maximum 200 | Source message envelopes for this page. |

### Page object

Required: `offset`, `limit`, `returned`, `total_messages`. Optional: `next_offset`, `previous_offset`.

| Field | Contract |
|---|---|
| `offset` | Integer `>= 0`; server clamps values beyond the transcript to the end. |
| `limit` | Integer `1..200`. |
| `returned` | Integer `0..200`; equals `messages.length` and never exceeds `limit`. |
| `total_messages` | Integer `>= 0`; stable for one accumulated client transcript. |
| `next_offset` | Present only when another page exists; equals `offset + returned`. |
| `previous_offset` | Optional earlier page start; must be less than `offset`. |

Additional runtime invariants enforced by the server/browser validators: `offset <= total_messages`; `offset + returned <= total_messages`; an empty page is valid only at transcript end; session entries are absent after page zero.

### Entry union

Every entry has exactly one of these strict shapes:

- `{ "entry_type": "card", "card": <card> }`
- `{ "entry_type": "prose", "prose": <prose> }`
- `{ "entry_type": "notice", "notice": <notice> }`

### Card object

Required fields:

| Field | Type and bound |
|---|---|
| `schema_version` | constant `semantic-card.v1` |
| `kind` | one of `shell`, `file_read`, `file_edit`, `search`, `web`, `task`, `mcp`, `lineage`, `attachment`, `fallback` |
| `title` | non-empty string, maximum 512 code points |
| `source` | strict card-source object |
| `fields` | array, maximum 64 |
| `previews` | array, maximum 32 |
| `caveats` | array, maximum 64 strings; each string maximum 4,096 code points |

Optional fields:

| Field | Contract |
|---|---|
| `summary` | String, maximum 4,096 code points. |
| `summary_disclosure` | Omission object; only legal with `summary`. |
| `outcome` | Structural outcome object. |
| `raw_evidence` | At least one of bounded `input_preview` or `result_preview`. |
| `title_disclosure` | Omission object for bounded title. |

Outcome object: required `state` in `succeeded|failed|unknown`; optional provider-structured `is_error` boolean and `exit_code` integer. Their absence means unknown, not false or zero.

### Card source object

Required:

- `session_id`: non-empty canonical session id;
- `provider_family`: non-empty string;
- `evidence_ref`: non-empty exact `EvidenceRef` string;
- `result_evidence_state`: `present|missing|unknown`.

`result_evidence_ref` is required exactly when state is `present` and prohibited when state is `missing` or `unknown`.

Optional exact identity/coordinate fields are not display-truncated: `message_id`, `result_message_id`, `block_index`, `result_block_index`.

Optional bounded metadata fields include `origin`, `block_id`, `tool_name`, `tool_id`, `attachment_id`, `occurred_at`, `parent_message_id`, `result_block_id`, `result_role`, `result_message_type`, `role`, and `message_type`, each at most 4,096 code points where applicable. Optional structural fields include message/result durations, variant index, active-path/leaf flags, inherited-prefix flags, material origins, and result coordinates.

`truncations` is a non-empty array of at most eight strict records. Each record identifies one bounded source field and a positive `omitted_characters` count. This prevents metadata clipping from becoming silent.

### Field, omission, disclosure, and preview objects

Field object:

- required `label` (1..256 code points) and `value` (maximum 4,096);
- optional `truncated: true`, positive `omitted_characters`, and `detail`;
- `truncated` and `omitted_characters` are mutually dependent; `detail` requires both.

Omission object:

- required `truncated: true` and positive `omitted_characters`;
- optional exact `detail` disclosure.

Disclosure object:

- required non-empty `ref`;
- required `part` in `message_text|block_text|tool_input|tool_input_raw|derived_diff`.

Preview object requires:

- `kind` (1..256 code points);
- `text` (maximum 16,000 code points);
- nonnegative `line_count`, `omitted_lines`, `omitted_characters`, and `encoding_replacements`;
- `truncated` boolean;
- strategy `full|head_tail|character_bounded`;
- optional `detail`, prohibited when `truncated` is false.

### Prose object

Required:

- non-empty `message_id`, `role`, `message_type`, `provider_family`, `material_origin`, and `evidence_ref`;
- one bounded `preview` object.

Optional structured metadata: `origin`, `block_id`, `block_index`, `block_type`, `language`, timestamp, duration, parent id, variant index, active-path/leaf flags, and inherited-prefix flag. Unbounded prose text is not serialized directly.

### Notice object

The current typed notice is `empty_thinking`. It requires a positive `count` and a non-empty `sources` array. Every notice source retains exact message/block coordinates plus role, message type, provider family, `material_origin`, and evidence ref, with optional origin/timing/lineage metadata.

### Message object

Required:

- non-empty `message_id`, `role`, `message_type`, `material_origin`, and `provider_family`;
- `semantic_card_suppressed` boolean;
- complete ordered `entries` array.

Optional message metadata: `origin`, timestamp, duration, parent id, variant index, active-path/leaf flags, inherited-prefix flag, and `source_session_id`.

A pure structurally paired result row may have an empty entries array with `semantic_card_suppressed: true`; its exact result evidence remains attached to the invocation card.

### `semantic-card-detail.v1`

Always required:

| Field | Type and bound |
|---|---|
| `schema_version` | constant `semantic-card-detail.v1` |
| `session_id` | non-empty canonical session id |
| `detail_ref` | string, 1..65,536 code points |
| `part` | string, 1..128 code points |
| `state` | `available|missing|unknown` |
| `offset` | integer `>= 0` |
| `limit` | integer `1..64,000` |
| `returned_characters` | integer `0..64,000` |
| `encoding_replacements` | integer `>= 0` |

For `available`:

- `part` must be one of the five declared detail parts;
- `text`, `total_characters`, and `complete` are required;
- `reason` is prohibited;
- `returned_characters` equals the Unicode-code-point length of `text` and does not exceed `limit`;
- if incomplete, `next_offset` is required and equals `offset + returned_characters`;
- if complete, `next_offset` is absent and the returned end reaches `total_characters`.

For `missing` or `unknown`:

- non-empty `reason` up to 4,096 code points is required;
- returned and replacement counts are zero;
- `text`, `total_characters`, `next_offset`, and `complete` are prohibited.

## Per-family rendering decisions

| Family | Prominent projection | Evidence behavior |
|---|---|---|
| `shell` | Command, structural state, `is_error`, and exit code. Failed state receives explicit styling. | Input command and paired result are bounded and expandable by exact refs. |
| `file_read` | Canonical path/target plus bounded read preview. | Exact input/result disclosure when structurally addressable. |
| `file_edit` | Canonical path/target plus bounded derived diff or write preview. | `derived_diff`, structured input, and raw input are separate declared detail parts. |
| `search` | Query/pattern, path/scope, and bounded result preview. | Result absence stays `missing` or `unknown`. |
| `web` | URL/query/action fields and bounded returned evidence. | No URL/tool-name reclassification in browser code. |
| `task` | Delegation summary and child-session link when the canonical field exists. | Child identity originates in the shared card builder. |
| `mcp` | Server/tool identity, arguments summary, structural outcome. | Arguments and result evidence are bounded and exact-ref expandable. |
| `lineage` | Session-level banner with relation, parent, branch point, inheritance, and degraded/unavailable facts when supplied. | Emitted as a session entry only on page zero. |
| `attachment` | Attachment identity/name/media facts and bounded preview. | Missing payload remains explicit; exact attachment identity is retained. |
| `fallback` | Unknown tool identity plus bounded raw input/result evidence; never silently dropped. | Fallback retains exact ref even when structured input is malformed or unavailable. |

## Parity harness

`tests/unit/rendering/test_semantic_card_document.py` compares three independent projections from the same canonical transcript:

1. `SemanticTranscript.cards` as the semantic authority;
2. the web card document;
3. terminal semantic Markdown, which now emits explicit `card family` and `structural outcome` facts.

The required representative fixtures are:

- `codex-exec-failure-ten-thousand-lines.json`: provider-reported failed shell result and bounded large output;
- `fork-compaction-family.json`: fork/compaction lineage and session-level placement;
- `hermes-truncated-unknown-input.json`: unknown provider tool, malformed/truncated input, and fallback evidence.

For each, the harness asserts identical card count, order, family, and structural outcome. Removing the shared transcript call, reclassifying in web code, dropping fallback evidence, changing pairing order, or omitting terminal facts makes the test fail. A second parameterized test serializes every canonical fixture through the strict document schema.

The restored `result-before-use.json` case verifies whole-document FIFO tool-result pairing when a serialized result precedes its invocation and verifies suppression of the pure paired result row. This was referenced by existing tests but missing from the snapshot.

## Five live old render implementations

Current source and the 2026-07-16 `polylogue-4p1` investigation establish five active message/block-to-output implementations plus one dead sixth:

1. generic HTML: `rendering/renderers/html.py`, `html_messages.py`, `core_messages.py`, and `blocks.render_blocks_html`;
2. generic Markdown: `rendering/core_markdown.py` and `blocks.render_blocks_markdown`;
3. legacy web semantic cards: `_polySemanticCardsHtml` in `daemon/web_shell_semantic_cards.py`;
4. legacy web flattened-message heuristic: `renderMessageBlocks` in `daemon/web_shell_reader.py`;
5. semantic CLI Markdown: `cli/messages.py` plus `rendering/semantic_markdown.py`.

The dead sixth is `build_projection_html_messages` with `get_render_projection`, tracked by `polylogue-a820`; source search finds definitions/tests but no production caller.

This vertical does not add a sixth classifier. `/v2/s` projects the canonical semantic document. The old two browser paths remain for the old route and are independent deletion candidates only after interaction parity and local certification.

## Changed files

### Public contracts and canonical rendering

- `docs/schemas/semantic-card-detail-v1.schema.json`
- `docs/schemas/semantic-card-document-v1.schema.json`
- `polylogue/rendering/semantic_card_models.py`
- `polylogue/rendering/semantic_card_placement.py`
- `polylogue/rendering/semantic_cards.py`
- `polylogue/rendering/semantic_markdown.py`

### Daemon, SSR, and packaged assets

- `polylogue/daemon/http.py`
- `polylogue/daemon/route_contracts.py`
- `polylogue/daemon/web_shell_semantic_cards.py`
- `polylogue/daemon/static/webui-v2/transcript.js`
- `polylogue/daemon/static/webui-v2/transcript.css`

### Frontend source and toolchain

- `webui/package.json`
- `webui/package-lock.json`
- `webui/tsconfig.json`
- `webui/vite.config.ts`
- `webui/vitest.config.ts`
- `webui/src/transcript/api.ts`
- `webui/src/transcript/components.tsx`
- `webui/src/transcript/main.tsx`
- `webui/src/transcript/transcript.css`
- `webui/src/transcript/types.ts`
- `webui/src/transcript/virtual.tsx`
- `webui/tests/unit/setup.ts`
- `webui/tests/unit/transcript.test.tsx`

### Fixtures and tests

- `tests/data/semantic_cards/cases/fork-compaction-family.json`
- `tests/data/semantic_cards/cases/result-before-use.json`
- `tests/unit/rendering/test_semantic_card_document.py`
- `tests/unit/daemon/test_semantic_card_http.py`
- `tests/unit/daemon/test_webui_v2_ssr.py`
- `tests/unit/daemon/test_route_contracts.py`

## Acceptance matrix

| Mission requirement | Status | Evidence |
|---|---|---|
| Typed card-document endpoint | Complete | Strict schema, JSON/HTML route, database/archive shared builder, route tests. |
| Registry remains sole classifier | Complete | Document builder delegates to `build_semantic_transcript`; parity and placement tests. |
| All named card families plus unknown fallback | Complete | Closed family enum, SSR/Preact tests for all ten, fixture corpus schema check. |
| Provider structural error/exit honesty | Complete | Failed shell fixture, `unknown` preservation, archive-backed result assertions. |
| Role versus `material_origin` visible | Complete | SSR and component authoredness tests, protocol/runtime labels. |
| Unknown tools never dropped | Complete | Fallback raw-evidence tests and Hermes parity fixture. |
| Bounded server previews | Complete | Schema limits, omission metadata, renderer tests. |
| Exact-ref progressive disclosure | Complete | Detail contract, no-widening test, Unicode/continuation browser tests. |
| Virtualized long transcripts | Complete at implementation level | Measured variable-height virtualizer and DOM-budget unit test. Operator-machine frame-rate measurement remains unverified. |
| Three-plus fixture parity harness | Complete | Failed result, fork/compaction, unknown provider tool. |
| SSR plus islands | Complete | Data-free bootstrap, authorized SSR, embedded-state boot, credential/canonical-id tests. |
| Zero CDN and reproducible committed assets | Complete | Clean offline install, zero audit findings, byte-identical consecutive Vite builds, wheel inspection. |
| Real-route test | Complete | ArchiveStore → daemon HTTP → renderer → JSON/detail/SSR/assets without loader mocks. |
| Exact result ZIP | Complete | Four-member validated ZIP described in the final operator report. |

## Apply order

1. Check out commit `536a53efac0cbe4a2473ad379e4db49ef3fce74d` in a clean worktree.
2. Run `git apply --check PATCH.diff`.
3. Run `git apply PATCH.diff`.
4. Run `git diff --check` and the focused Python commands in `TESTS.md`.
5. From `webui/`, run `npm ci --offline --ignore-scripts`, `npm run typecheck`, `npm run test:unit`, and `npm run build`.
6. Confirm the generated asset hashes match the values in `TESTS.md` or review any deliberate toolchain update before committing.
7. Build the wheel and confirm both `polylogue/daemon/static/webui-v2/` assets are present.
8. Exercise `/v2/s/<session>` on a sanitized local archive before retiring any old reader path.

The patch was independently applied to a fresh detached worktree at the named commit. All 30 resulting files were SHA-256 compared with the implementation worktree and were byte-identical; both schemas validated and 44 focused tests passed in that applied copy.

## Risks and limitations

1. **Read work is not yet keyset-bounded.** Responses and browser DOM are bounded, but `build_semantic_card_document(...)` currently normalizes, hydrates, pairs, and builds the whole canonical transcript before slicing the message page. This preserves cross-page pairing correctness, including result-before-use, but does not satisfy `polylogue-4p1`'s future first-useful-content/keyset transaction.
2. **No operator archive or deployment proof.** The real route test uses a sanitized temporary split archive through production `ArchiveStore`; it is not the operator's private archive, live daemon, NixOS deployment, or secrets.
3. **Real Chromium navigation was blocked by environment policy.** An external Playwright/Chromium journey was attempted, but managed Chromium policy `/etc/chromium/policies/managed/000_policy_merge.json` contains `"URLBlocklist": ["*"]`; navigation failed with `ERR_BLOCKED_BY_ADMINISTRATOR`. Browser unit tests and SSR/HTTP tests pass, but no real browser journey or frame-rate trace is claimed.
4. **No measured 60-fps claim.** Variable-height virtualization is implemented and DOM-bounded in tests; operator-machine scroll performance remains unmeasured.
5. **Page entry volume follows source semantics.** Message rows, display fields, metadata, previews, and detail chunks are bounded. A pathological single message may still carry many exact semantic entries, and exact identity strings are intentionally retained rather than clipped.
6. **First token-authenticated visit needs JavaScript.** A token-configured daemon with no established web credential returns a data-free SSR shell; the first transcript fetch follows the existing JavaScript credential bootstrap. Established credentials, bearer requests, and no-token local operation get transcript SSR.
7. **CSP is not introduced.** The implementation uses first-party assets, script-safe JSON, no-referrer, no-store, nosniff, and frame denial. A future Content-Security-Policy requires a nonce/hash design for embedded JSON/module boot.
8. **Legacy render paths remain.** This patch proves the v2 seam; it does not certify deletion of old reader interactions or the dead sixth path.
9. **No full repository suite.** Strong affected lanes passed. A final duplicate invocation of the broad entire `tests/unit/rendering` directory exceeded the 300-second command ceiling; focused semantic/rendering coverage remained green and no product failure was observed.

## Open integration questions for `ap7` and `4p1`

- What bounded read transaction will provide enough neighboring tool-use/result and lineage evidence to preserve canonical FIFO pairing across keyset page boundaries without full-session hydration?
- Should `semantic-card-document.v1` become a registered `ReadPreset`/render profile under `4p1`, or remain an internal transport until the read algebra lands?
- Which source owns stable page cursors once offset paging is replaced: message identity, composed lineage position, or an archive generation plus key tuple?
- How should DB-backed and archive-backed lineage/delegation equivalence be certified without hydrating full families, as required by `ap7.1`?
- When should JSON Schema become the generator for TypeScript validators rather than the current strict handwritten mirror?
- Which old browser affordances must move to `/v2/s` before `_polySemanticCardsHtml` and `renderMessageBlocks` can be independently deleted?
- Should static HTML export reuse the SSR projection directly or consume a future render-profile abstraction under `bby.11`?

## Verification summary

Final checks completed against the frozen implementation:

- Ruff check and format check: pass on all 11 changed Python files.
- Python byte compilation and `git diff --check`: pass.
- strict Mypy: pass on 10 changed production/test modules with the normal import graph.
- focused rendering/daemon/route/metrics lane: 179 passed.
- canonical semantic-card baseline: 99 passed.
- daemon HTTP security matrix: 524 passed.
- semantic-card fixture generator: 24 cases verified.
- frontend strict TypeScript: pass.
- Vitest: 24 passed.
- clean offline npm install: pass; 89 packages installed; audit total 0.
- consecutive Vite builds: byte-identical.
- final JavaScript: SHA-256 `6cc8fdfbf38abd9ccd66de5b7015ac6b9c0fafd61815d32c7536bc44d39006db`, 58,928 bytes.
- final CSS: SHA-256 `dc39c243ba661da3fa859bbbc366ed146542908341949fb34c8e430dbba4ebdb`, 5,740 bytes.
- offline wheel: SHA-256 `19483aa51be3d16372879be9e47a3a595b27529b74a7e26891dd931ffef0b54a`, 4,200,885 bytes; CRC valid; both assets present.
- topology: `realized=1023`, `declared=1023`, `blocking=False`; zero missing/orphan/conflict/kernel-rule findings; nine pre-existing storage `tbd` warnings.
- patch scan: no secrets, private registry references, environment paths, copied input archive names, explicit placeholders, stub sentinels, or pseudocode markers.
- fresh-worktree patch application: pass; 30/30 files byte-identical; schemas valid; 44 tests passed.

Another small repair iteration is unlikely to add material value to this package. A substantial next pass would be valuable only if it integrates the document with `4p1` keyset reads, runs a real unrestricted browser/performance journey, exercises an operator-approved archive/deployment, and certifies retirement of the legacy reader paths.

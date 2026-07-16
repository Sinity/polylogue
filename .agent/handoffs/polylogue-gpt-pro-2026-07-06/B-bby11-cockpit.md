## [assistant 367]
According to a document from 2026-07-05, the target for `bby.11` is specifically **Webui-v2 evidence cockpit: result-list → linked evidence-graph → citable report**, centered on content-hash citation anchors rather than position-based refs. fileciteturn0file0 fileciteturn1file0 I’m treating the code snapshot as dated; live code may have moved and should be verified locally before implementation.

My recommendation: **keep the `bby.11` stack decision of TypeScript + Preact + Vite, but make Python/Jinja server-rendered HTML the canonical fallback for every route, with Preact as progressive enhancement rather than SPA-or-nothing.** [evidence] `polylogue-bby.11` already chooses Preact/Vite/TypeScript, generated API types, committed built assets, no-CDN/offline posture, and reader/export parity as the architecture direction. [proposal] Tighten that decision: the cockpit is not “a Preact app with some HTML fallback”; it is “the daemon serves semantic HTML and typed JSON; Preact hydrates graph/basket/report interactions when available.”

Runner-up rejected: **pure Jinja/HTMX-style server UI.** [proposal] It is tempting because A09 explicitly requires phone/curl/pandoc usability and progressive enhancement. But the force-directed graph, live evidence basket, report editor, drag/refocus, citation verification overlays, and typed API-client work are much better served by a small typed client. Pure server UI would likely recreate the current JS-in-Python-strings ceiling under a different name. The compromise is SSR-first routes plus typed Preact islands.

## Layer 1 — today’s substrate

[evidence] Polylogue already has the right strategic shape for this: it is described as a local-first system of record for AI work, with durable `source.db` / `user.db`, derived `index.db` / `embeddings.db`, and disposable `ops.db`. The orientation excerpt also says adjacent P2 vision already includes “web workbench v2 (result list → evidence cockpit)” and construct-validity metadata. fileciteturn1file1

[evidence] The current daemon already exposes the raw ingredients of an evidence cockpit. In the snapshot, `polylogue/daemon/http.py` registers `/api/sessions`, `/api/query-units`, `/api/refs/resolve`, `/api/read-view-profiles`, `/api/assertions`, `/api/sessions/:id`, `/api/sessions/:id/messages`, `/api/sessions/:id/read`, `/api/sessions/:id/raw`, `/api/sessions/:id/provenance`, `/api/sessions/:id/topology`, `/api/sessions/:id/topology/parent-chain`, `/api/sessions/:id/similar`, and `/api/sessions/:id/attachments` (`polylogue/daemon/http.py:257-305`). That is already result-list, reader, provenance, lineage, assertions, and attachments.

[evidence] The current web shell is a single Python-assembled HTML document, with JS modules embedded through string substitution. `polylogue/daemon/web_shell.py` calls itself the “MK2 web reader — single-page interactive archive cockpit” and assembles `WEB_SHELL_HTML` from imported CSS/JS modules (`web_shell.py:1-22`). That is exactly the foundation `bby.11` says is no longer enough.

[evidence] The current reader already has message anchors and permalink actions, but these anchors are DOM-safe IDs derived from message IDs, not content-hash citations. `reader_anchor("session"|"message", target_id)` sanitizes the target id into `session-...` or `message-...` (`polylogue/surfaces/payloads.py:551-555`), and the message action rail copies/jumps to those anchors (`polylogue/daemon/web_shell_reader.py:200-219`). Good UI affordance; wrong durability layer for citations.

[evidence] Current public refs are still position-shaped for evidence. `EvidenceRef` is `session_id[::message_id[::block_index]]`, explicitly using `::` because session ids contain colons (`polylogue/core/refs.py:116-171`). `ObjectRef` supports `block:<message_id>:<qualifier>` where the block qualifier is effectively the block index (`core/refs.py:76-114`). User-state target kinds include `block`, but the block identity template is `block:{session_id}:{target_id}` and target id is `message_id:block_index` (`polylogue/core/user_state_targets.py:77-82`; `polylogue/api/user_state_resolver.py:216-247`). This is the key gap: block references exist, but they are still position/index anchored.

[evidence] The `index.db` schema gives sessions and messages `content_hash`, but not blocks. Sessions have `content_hash` at `sessions.content_hash`; messages have `messages.content_hash`; blocks are keyed by `(message_id, position)` and expose `block_id = message_id || ':' || position`, `text`, `tool_name`, `tool_input`, `tool_result_is_error`, `tool_result_exit_code`, etc., but no `content_hash` (`polylogue/storage/sqlite/archive_tiers/index.py:39-75`, `92-126`, `182-209`). The writer computes message content hashes from block content (`write.py:1433-1476`) but writes blocks without a block hash (`write.py:1479-1523`).

[evidence] The current topology layer is useful but insufficient for A09. It models lineage as sessions/nodes/edges, with edge kinds `continuation`, `sidechain`, `fork`, `subagent`, `unknown`, and `unresolved_native`; it reports cycle detection and BFS-ordered nodes (`polylogue/insights/topology.py:1-21`, `34-55`, `126-153`). The HTTP envelope is bounded, returns nodes/edges/readiness/truncation/cycle/unresolved counts, and is consumed by the Lineage tab (`polylogue/daemon/topology_http.py:1-20`, `103-160`). But A09 wants a visual graph keyed by `prefix-sharing` vs `spawned-fresh`, quarantined cycle-break nodes, and branch-point highlighting. Those fields exist lower in `session_links`: `branch_point_message_id`, `inheritance IN ('prefix-sharing','spawned-fresh')`, `status IN ('repaired','quarantined')`, confidence, evidence JSON (`polylogue/storage/sqlite/archive_tiers/index.py:376-400`). The current topology envelope does not surface all of that.

[evidence] User overlays are already first-class enough to reuse. `user.db.assertions` stores `scope_ref`, `target_ref`, `kind`, `value_json`, `body_text`, `author_ref`, `author_kind`, `evidence_refs_json`, `status`, `visibility`, `confidence`, `staleness_json`, `context_policy_json`, and `supersedes_json` (`polylogue/storage/sqlite/archive_tiers/user.py:7-31`). `AssertionClaimPayload` mirrors this into a surface payload with evidence refs and lifecycle fields (`polylogue/surfaces/payloads.py:1465-1505`). The web shell already fetches `/api/user/marks` and `/api/user/annotations` into client state (`web_shell.py:717-735`), and `/api/assertions` lists assertion-backed overlay claims (`http.py:2950-2987`).

[evidence] Recall packs and workspaces are very close to “evidence basket” and “open investigation state.” `upsert_recall_pack` stores a recall-pack assertion with `target_ref=recall_pack:<id>` and `kind=recall_pack` (`user_write.py:591-613`). The API resolves recall-pack items into sessions, messages, annotations, marks, and registered target kinds, recording `resolved_count` and `degraded_count` (`api/archive.py:4834-5066`). The HTTP user-state routes already expose `GET/POST/DELETE /api/user/recall-packs` and `GET/POST/DELETE /api/user/workspaces` (`daemon/user_state_http.py:445-620`). So a crude evidence basket exists in spirit; it just lacks block-hash anchors, report coupling, and citation verification.

## Layer 2 — near-term substrate change

[proposal] The near-term substrate should be **not a rewrite**, but three precise additions: block-hash citation anchors, evidence baskets as user-tier objects, and a citation verifier. This gives `bby.11` enough power without waiting for the full daemon/protocol/composer refactor.

### 1. Anchor format

[proposal] Use this as the canonical textual anchor:

```text
<session_id>::<message_id>::block@sha256:<64-hex>
```

Example shape:

```text
claude-code-session:abc123::claude-code-session:abc123:42.0::block@sha256:7a0f...
```

[evidence] This deliberately reuses the existing `::` separator because Polylogue already notes that session ids are colon-bearing and therefore cannot safely use plain colon segmentation for evidence refs (`polylogue/core/refs.py:116-130`). The A09 mnemonic `session:message:block@<hash>` is the conceptual form; the concrete Polylogue-safe form should use `::`.

[proposal] Store anchors structurally wherever possible:

```json
{
  "anchor_version": 1,
  "kind": "block",
  "session_id_hint": "...",
  "message_id_hint": "...",
  "hash_algo": "sha256",
  "content_hash": "...",
  "position_hint": 3,
  "created_archive_epoch": "...",
  "created_at_ms": 1783210000000
}
```

The Markdown/report form can be compact; the `user.db` form should be structured.

### 2. Block content hash

[proposal] Add `blocks.content_hash BLOB NOT NULL CHECK(length(content_hash)=32)` to the derived `index.db` blocks table, plus indexes:

```sql
CREATE INDEX idx_blocks_content_hash
ON blocks(content_hash);

CREATE INDEX idx_blocks_session_message_hash
ON blocks(session_id, message_id, content_hash);
```

[evidence] `index.db` is derived/rebuildable, so this is an index-tier schema change rather than durable `user.db` migration. The swarm brief distinguishes durable source/user tiers from derived index/embedding tiers and says derived tiers rebuild/blue-green rather than migrate in place. fileciteturn1file10

[proposal] Define `block_content_hash_v1` over canonical block evidence, not identity:

```text
sha256(
  "block-v1\0",
  block_type,
  text or "",
  tool_name or "",
  canonical_json(tool_input) or "",
  semantic_type or "",
  media_type or "",
  language or "",
  tool_result_is_error or "",
  tool_result_exit_code or ""
)
```

Do **not** include `session_id`, `message_id`, `message.position`, `block.position`, or `tool_id`. The whole point is to survive fork-position shifts and re-ingest. Including `tool_id` would make the anchor less robust when providers regenerate tool identifiers; omit it from the hash but keep it in resolver metadata.

### 3. Anchor resolution algorithm

[proposal] Add a resolver, e.g. `resolve_evidence_anchor(anchor) -> CitationResolutionPayload`.

Resolution states:

```text
ok
drifted_position
drifted_message
relocated_lineage
ambiguous
missing
quarantined
hash_mismatch
stale_index
```

Algorithm:

1. Parse `<session_id_hint>::<message_id_hint>::block@sha256:<hash>`. Reject unsupported hash algorithms and malformed hex.

2. Exact scoped lookup:

```sql
SELECT b.*, m.position AS message_position, s.root_session_id
FROM blocks b
JOIN messages m ON m.message_id = b.message_id
JOIN sessions s ON s.session_id = b.session_id
WHERE b.session_id = ?
  AND b.message_id = ?
  AND b.content_hash = ?
```

If one row exists and the current `b.position` equals the stored `position_hint`, return `ok`.

3. If the row exists but the block position differs, return `drifted_position`, with `old_position_hint`, `current_position`, and the current live target ref. This is exactly the case A09 is protecting against: the content survived but position changed.

4. If scoped lookup fails, search the same session by hash. If exactly one row exists, return `drifted_message` if the message id changed, or `drifted_position` if only block position changed.

5. If same-session lookup fails, search the lineage neighborhood: topology root, ancestors, descendants, siblings. Prefer rows whose sessions are connected by `prefix-sharing` before `spawned-fresh`, because prefix-sharing means the evidence may be inherited content rather than merely related session content. The current lower schema has `session_links.inheritance` and `branch_point_message_id`; expose those in the resolver. If exactly one candidate remains, return `relocated_lineage`.

6. If global hash lookup finds multiple rows, return `ambiguous`, not a guessed success. Include candidates with session/message/block refs, titles, edge relation to original hint, and snippets. This is common for boilerplate prompts/tool outputs.

7. If no row exists, return `missing`. If the source session/raw artifact is known deleted/quarantined or the lineage path includes a quarantined edge, return `quarantined` rather than generic missing.

8. On any state except `ok`, preserve the old anchor and produce a new suggested anchor only if the live content hash still matches.

[evidence] This algorithm is aligned with the existing topology and user-state posture: current topology already surfaces `partial` readiness when truncation, unresolved edges, or cycles occur (`topology_http.py:80-100`), and recall packs already distinguish resolved/degraded items (`api/archive.py:5053-5060`). The missing piece is content-hash resolution.

### 4. Evidence basket → report flow

[proposal] Near-term storage can use existing `recall_pack` machinery, but with a new payload schema:

```json
{
  "schema_version": 2,
  "kind": "evidence_basket",
  "label": "Fable delegation rhetoric",
  "source_query_run_ref": "query_run:...",
  "items": [
    {
      "item_id": "basket_item:...",
      "anchor": "...::...::block@sha256:...",
      "selection_span": {"start": 120, "end": 355},
      "quote": "...",
      "note": "example of scope-policing",
      "tags": ["scope-control", "stern"],
      "added_from": {
        "surface": "webui-v2",
        "route": "/api/sessions?...",
        "query_ref": "query_run:..."
      },
      "last_resolution": {
        "state": "ok",
        "resolved_block_ref": "block:...",
        "resolved_at_ms": 1783210000000
      }
    }
  ]
}
```

[evidence] This fits current recall-pack capability because recall packs already normalize an item list, resolve items, and record resolved/degraded counts (`api/archive.py:5031-5066`). [proposal] But do not keep calling it “recall pack” in the UI. The product object is an **evidence basket**; recall pack is merely the initial storage adapter if adding a new `AssertionKind` is too expensive.

[proposal] The cockpit UI should have three panes:

Left: result list / query results.

Center: reader + evidence graph.

Right: evidence basket + live report draft.

Data flow:

```text
Query result row
  → open reader/session/topology
  → select block/message/span
  → Add to basket
  → resolver stores content-hash anchor + quote + provenance
  → report draft inserts Markdown footnote
  → verifier re-resolves anchors
  → export Markdown/HTML + citation manifest
```

[proposal] Report draft storage should be a second user-tier object, initially:

```text
target_ref = report:<report_id>
kind       = annotation or workspace_note initially; later report_draft
scope_ref  = evidence_basket:<basket_id>
value_json = {
  "markdown": "...",
  "basket_id": "...",
  "footnote_order": ["basket_item:1", "basket_item:2"],
  "export_policy": "fail_on_missing_or_quarantined"
}
```

Better full form later: add `AssertionKind.REPORT_DRAFT` and `AssertionKind.EVIDENCE_BASKET`, but that touches render/openapi/user-audit surfaces. The Claude memory excerpt even warns that new `AssertionKind` currently has several downstream schema/render implications, so do it deliberately rather than casually. fileciteturn1file4

### 5. Citation-integrity verifier

[proposal] Add `verify_citations(report_id|basket_id|anchors[]) -> CitationVerificationEnvelope`.

Envelope:

```json
{
  "ok": false,
  "verified_at_ms": 1783210000000,
  "archive_epoch": "...",
  "summary": {
    "ok": 9,
    "drifted": 2,
    "ambiguous": 1,
    "missing": 0,
    "quarantined": 1
  },
  "items": [
    {
      "anchor": "...",
      "state": "drifted_position",
      "severity": "warning",
      "old_ref": "...",
      "live_ref": "...",
      "message": "Content hash resolved in same message at block position 4, position hint was 3."
    }
  ]
}
```

Export policy:

`ok` and `drifted_position`: export allowed; footnote says “position drifted, content hash verified.”

`drifted_message` / `relocated_lineage`: export allowed only with visible warning unless operator explicitly promotes the new anchor.

`ambiguous`: export blocked by default; report can export only with an “unverified candidate” appendix.

`missing`: export blocked by default.

`quarantined`: export blocked by default unless report is explicitly about quarantined evidence.

`hash_mismatch`: hard fail; never silently rewrite.

[evidence] This matches A09’s instruction to use the same honesty doctrine that turned the recovery-digest fabrication into “unverified candidates.” fileciteturn1file0 It also aligns with the situation brief’s claim that the project’s strongest method is checking constructs against what they actually compute rather than accepting convenient regex/inference artifacts. fileciteturn0file11

## Layer 3 — full direction

[proposal] The full direction is not “a better web reader.” It is **publication-grade evidence operations over Polylogue’s read algebra**.

The underlying model should be:

```text
query_definition
  → query_run
  → result_relation
  → evidence_basket
  → annotation_batch/assertions
  → report_draft
  → verified_export
```

[evidence] This extends the C10 composer design, which already says committed runs should write recall entries with query text, resolved spec, result fingerprint, timestamp, and promotion to macros. fileciteturn1file6 [proposal] For the evidence cockpit, split that into explicit query-definition/query-run/result-relation objects, because report citations need to know which result set produced the basket.

[proposal] The “evidence graph” should not be just session lineage. It should combine four edge families:

1. **Lineage edges**: parent/fork/continuation/subagent, with inheritance `prefix-sharing` vs `spawned-fresh`, branch point, status, confidence, evidence.

2. **Evidence containment edges**: session → message → block → attachment/paste/tool call.

3. **User overlay edges**: assertion/annotation/mark/correction → target ref, and assertion → evidence refs.

4. **Analysis/report edges**: query_run → result_relation → evidence_basket → report_draft → export artifact.

That makes the graph useful for investigation rather than merely pretty. The user can click a report footnote, see the block, see the message/session, see the lineage edge that explains why the block moved, see assertions attached to it, and see which query originally surfaced it.

[proposal] The Webui-v2 cockpit should become the canonical UI for the “search/analyze/audit/remember” loop:

Search: query/composer/result list.

Analyze: graph + projections + aggregates.

Audit: citation verifier + assertion overlay + drift/quarantine state.

Remember: basket/report/annotation/cohort save.

[evidence] The current strategic language already frames Polylogue as “search / analyze / audit / remember,” not a chat viewer. fileciteturn1file3

## Visual design

[proposal] Replace the current BFS lineage list with a force-directed graph, but keep the BFS tree as the SSR fallback.

Graph behavior:

Nodes:

session nodes, with title/provider/model/date chips.

message/block nodes only when expanded.

assertion nodes as overlay badges or side nodes.

basket/report nodes in investigation mode.

quarantined nodes visibly red/striped.

cycle-break nodes diamond-shaped or warning-marked.

Edges:

`prefix-sharing`: solid, thick, cool color.

`spawned-fresh`: dashed.

`subagent`: arrowed child edge with “subagent” label.

`fork`: split/branch style.

`continuation`: simple forward edge.

`unresolved_native`: dotted, ghost node.

`quarantined`: red/striped edge, never hidden.

Interactions:

Click node: refocus reader/result list to that session/message/block.

Click edge: show branch point, inheritance, confidence, evidence JSON.

Hover branch point: highlight the inherited prefix up to `branch_point_message_id`.

Toggle overlays: assertions, marks, annotations, baskets, query-run provenance.

Lasso/select graph nodes: add all visible evidence blocks/messages to basket.

Keyboard: `g` graph focus, `b` add to basket, `r` report pane, `v` verify citations.

[proposal] Use `d3-force` rendered with Canvas/SVG inside a Preact component, not a huge graph framework at first. D3-force gives enough layout control, small enough dependency surface, and agent familiarity. If graph editing/large-topology performance becomes hard, reconsider Cytoscape.js later.

## Progressive enhancement

[evidence] The current daemon already has `_send_html` and serves the web shell HTML from Python (`daemon/http.py:1168-1175`, `1465-1468`). [proposal] Webui-v2 should formalize this instead of returning only the SPA shell.

Every reader/cockpit route should support three representations:

```text
Accept: text/html       → server-rendered semantic HTML
Accept: application/json → DTO JSON
Accept: text/markdown  → report/evidence-pack Markdown where meaningful
```

Routes:

`/s/<session>` returns server-rendered session reader with message/block anchors.

`/e/<anchor>` returns server-rendered evidence card.

`/g/<session>` returns server-rendered topology as nested tree/table, plus JSON script island for Preact graph hydration.

`/basket/<id>` returns server-rendered basket with verification chips and forms.

`/report/<id>` returns server-rendered Markdown preview with footnotes.

`/report/<id>.md` returns raw Markdown.

`/report/<id>/manifest.json` returns citation manifest.

A phone or `curl | pandoc` gets clean HTML. Preact upgrades the same HTML into the graph/basket/editor cockpit. No CDN. Built assets committed as `bby.11` already says.

## Minimal daemon API

[proposal] Keep the daemon API small and contract-shaped. B8’s core contract recommends `query`, `read`, `preview`, `complete`, `act`, `status`, plus facets as a separate method; the evidence cockpit should ride on that rather than invent a parallel substrate. fileciteturn0file3

Near-term HTTP API additions:

```text
GET  /api/evidence/resolve?anchor=...
POST /api/evidence/resolve-batch
POST /api/evidence/verify
```

```text
GET  /api/evidence-baskets
POST /api/evidence-baskets
GET  /api/evidence-baskets/:id
PATCH /api/evidence-baskets/:id
POST /api/evidence-baskets/:id/items
PATCH /api/evidence-baskets/:id/items/:item_id
DELETE /api/evidence-baskets/:id/items/:item_id
POST /api/evidence-baskets/:id/verify
```

```text
GET  /api/reports
POST /api/reports
GET  /api/reports/:id
PATCH /api/reports/:id
POST /api/reports/:id/render
POST /api/reports/:id/verify
POST /api/reports/:id/export
GET  /api/reports/:id/manifest
```

Extend existing topology:

```text
GET /api/sessions/:id/topology?include=inheritance,status,branch_point,evidence
```

Add overlay batch endpoint:

```text
POST /api/overlays/for-refs
```

Request:

```json
{"refs": ["...anchor...", "session:...", "message:..."], "include": ["assertions", "marks", "annotations"]}
```

Response: overlays grouped by normalized target ref and evidence refs.

For the UDS/thin-client future, these become service verbs rather than URL sprawl:

```text
evidence.resolve
evidence.verify
basket.get/save/add/remove/reorder
report.get/save/render/verify/export
topology.read
overlay.batch
```

[evidence] This matches the A2/A4/A3 direction: a warm daemon with one execution core, UDS fast path, and in-process break-glass rather than duplicate client substrate logic. fileciteturn0file7 fileciteturn0file6 fileciteturn0file4

## Implementation sequence

[proposal] I would land it in this order:

First PR: **block hash substrate.** Add block content hash to index DDL/writer/payloads, extend public-ref parser with `block@sha256`, add resolver and tests over re-ingest/position drift.

Second PR: **citation verifier.** Batch resolver, verification envelope, CLI smoke command, and report-manifest schema. No UI dependency.

Third PR: **basket as recall-pack v2.** Use existing recall-pack/user-state routes, but add evidence-basket payload schema and block-hash item resolver. Add web forms in SSR.

Fourth PR: **SSR evidence routes.** Add Jinja templates for session, evidence card, topology fallback, basket, report. Use existing `jinja2` dependency and HTML sanitizer stack (`pyproject.toml:32`; `rendering/renderers/html_template.py:8-34`).

Fifth PR: **Preact cockpit scaffold.** Typed API client, tokens, router, result list, reader parity, basket pane.

Sixth PR: **force graph.** Extend topology envelope to include inheritance/status/branch point, then add D3-force graph with fallback.

Seventh PR: **report editor/export.** Live Markdown report pane, footnotes, verifier gate, Markdown/HTML/manifest export.

Eighth PR: **assertion overlay.** Batch overlay endpoint, assertion badges on reader/graph/basket/report footnotes.

This sequence gives value early: block-hash verifier and baskets are useful even before the new graph is pretty.

## What I would change in the current design

[proposal] `bby.11` currently frames the stack as the foundation. I would sharpen the bead around **evidence integrity as the first acceptance gate**. The scaffold is not done when Preact renders list+reader parity; it is done when a selected block can survive re-ingest and export through a verifier.

Revised `bby.11` acceptance core:

1. A block content-hash anchor can be created from the reader.

2. Re-ingest or lineage recomposition that changes block position still resolves the anchor as drifted-but-verified.

3. A basket persists selected evidence blocks/messages with structured anchors.

4. A report draft renders those items as Markdown footnotes.

5. Export re-resolves every citation and blocks or flags missing/quarantined/ambiguous anchors.

6. Topology graph surfaces `prefix-sharing` vs `spawned-fresh`, quarantined edges, cycle state, and branch point.

7. The same session/basket/report routes work as server-rendered HTML with JS disabled.

The stack should serve this acceptance, not be the acceptance.

## Open questions for the operator

Should the first implementation store baskets/reports as `recall_pack` / `workspace_note` payloads to avoid new `AssertionKind`, or is it worth paying the schema/render/openapi cost now for `evidence_basket` and `report_draft`?

Should a quarantined citation hard-block export by default, or should it export with a red “quarantined evidence” appendix when the report is explicitly forensic?

Should the block hash include message role? I lean no for the hash itself and yes for resolver metadata. Including role reduces duplicates but makes the hash less purely block-content keyed.

Should report exports be saved back into `user.db` as report objects, written as files under a publication/export directory, or both?

Should the graph render message/block nodes by default, or only on expansion? I lean session graph by default with evidence-node expansion on click.

## What’s missing

I did not run live code or tests. I inspected the uploaded snapshot and extracted package only; live master may have moved.

I did not inspect an actual live archive to verify how often duplicate block-content hashes occur. That matters for ambiguity policy.

I did not inspect current `bd show polylogue-bby.11 --json` from a live repo; I used the uploaded bead export and A09 brief.

I did not prototype the graph or measure bundle size for `d3-force` versus Cytoscape.js.

I did not verify whether current OpenAPI generation can cleanly emit the new DTOs without additional schema-render work.

The most important unverified technical assumption: adding `blocks.content_hash` to derived `index.db` is straightforward. It should be, but the sync/async writer twin path must be checked before claiming it is a small change.

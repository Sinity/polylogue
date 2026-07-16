# B8 — The single client↔substrate contract

**Deliverable:** the minimal, complete verb+DTO surface a thin client (CLI/TUI,
any language) needs so it never re-implements substrate logic. Reconciled with
the live `polylogue.api` facade, `polylogue.surfaces.*` payloads, and the daemon
HTTP route table. Companion to A2 (wire protocol) and assuming t46 (thin CLI)
done first.

READ-ONLY design doc. `file:line` refs are evidence; the "Recommendation"
sections are proposal.

---

## 1. What already exists (evidence)

Polylogue is *most of the way* to a shared contract already — the pieces exist
but are not composed into one surface, and three of them are half-built.

### 1a. A nascent Protocol family — `polylogue/api/contracts/`
- `read_surface.py` declares `SessionListSurface`, `SessionSearchSurface`,
  `SessionTagsSurface`, `SessionStatsSurface`, `FacetsSurface`,
  `DaemonStatusSurface`, `TagMutationSurface`, and a composite `ReadSurface`
  (`read_surface.py:40-165`). Each is a `runtime_checkable` `Protocol`, async,
  taking `SessionQuerySpec` and returning a `surfaces.payloads` envelope.
- `write_surface.py` declares `IngestSurface`, `MaintenanceSurface`,
  `IndexMaintenanceSurface`, `TagMutationSurface`, `SessionDeleteSurface`,
  composite `WriteSurface` (`write_surface.py:49-162`).
- Adapters exist for **every** surface with static conformance pins:
  `cli_write_surface.py`, `mcp_write_surface.py`, `api_write_surface.py`,
  `tui_surface.py` — each ends in `assert_implements(Adapter, Protocol)` so mypy
  fails on drift (e.g. `tui_surface.py:102-105`).

**This is the right skeleton.** The gaps below are all "the skeleton is too
narrow and two verbs are missing," not "start over."

### 1b. Input contract — `SessionQuerySpec`
Frozen dataclass, ~50 fields (`archive/query/spec.py:440-495`): query/contains/
exclude terms, retrieval_lane, origins, tags, repos, action/tool terms,
has-flags, min/max messages/words, since/until, sort/reverse/limit/offset,
cursor, `similar_text`, `boolean_predicate: QueryPredicate`, and the newer
`with_units` / `with_unit_fields` (fnm.2). It is the *lowered* internal form.

**But there are three independent front doors that each re-derive it:**
- MCP: `MCPSessionQueryRequest` dataclass → `build_spec()`
  (`mcp/query_contracts.py:66-158`), routing free-text through
  `compile_expression_into`.
- CLI: `RootModeRequest.query_spec` (referenced `mcp/query_contracts.py:46`)
  builds the same spec from Click flags.
- Daemon HTTP: query-string parsing inside `daemon/http.py` (163 KB) builds it
  again from `?` params.

Three parsers, one target. There is no single wire request DTO.

### 1c. Output contract — `polylogue/surfaces/payloads.py` (104 KB)
All responses are pydantic `SurfacePayloadModel` (`frozen=True, extra="forbid"`,
`payloads.py:107`) with a `.to_json()`. The canonical envelopes already exist:
- `SessionListResponse` (`payloads.py:1155`) — items/total/limit/offset/
  diagnostics/route_state.
- `SearchEnvelope` (`payloads.py:1186`) — the ranked superset: hits/total/
  limit/offset/next_cursor/query/retrieval_lane/ranking_policy/
  **action_affordances**/diagnostics/route_state.
- `FacetsResponse`, `TagMutationResult`, `MetadataMutationResult`,
  `DeleteSessionResult`, `BulkTagMutationResult`, `SessionReadViewEnvelope`,
  `QueryUnitResultEnvelope`, `PublicRefResolutionPayload`.

### 1d. Read algebra — `polylogue/surfaces/projection_spec.py`
`QueryProjectionSpec = SelectionSpec × ProjectionSpec × RenderSpec`
(`projection_spec.py:141-146`) — the `4p1` algebra. `ProjectionSpec` = families
+ body_policy + exclude_block_kinds + budgets; `RenderSpec` = format ×
destination × layout. **But it self-documents as "a contract builder, not an
executor" (`projection_spec.py:216-218`).** The live `read` verb still executes
by *named view* — `SessionReadViewEnvelope.view: str` + `payload: Any`
(`payloads.py:1774-1788`), `projection_from_view` maps view→families
(`projection_spec.py:149-172`). Two read contracts coexist: the view-name one
(live) and the projection algebra (aspirational).

### 1e. Daemon route table — `daemon/route_contracts.py`
A *descriptive* metadata table (`ROUTE_CONTRACTS`, `route_contracts.py:48-435`):
method, pattern, auth_policy, and a **`response_contract: str`** free-text field
naming the payload ("SearchEnvelope", "FacetsResponse", ...). It is not typed —
dispatch lives in `http.py` and the request side is untyped query-string
parsing. Tests compare it to the live dispatcher (`route_contracts.py:6-8`).

### 1f. The facade — `polylogue/api/archive.py` (222 KB, ~130 async methods)
`Polylogue` (`api/__init__.py:44`) exposes the *entire* surface as loose
methods: `list_sessions`, `search_envelope`, `get_session`, `get_messages_
paginated`, `facets`, `stats`, `health_check`, `query_completions`, plus every
mutation (`add_tag`, `remove_tag`, `delete_session`, `add_mark`,
`save_annotation`, `record_correction`, `set_metadata`, `judge_assertion_
candidate`, `save_view`, `save_workspace`, `post_blackboard_note`, ...). This is
the real substrate API; the CLI reaches *around* it 45× (brief §Measured facts).

---

## 2. The gaps (what's missing / mis-shaped)

1. **No single wire request DTO.** `SessionQuerySpec` is a storage-adjacent
   dataclass, not a client-facing model, and three surfaces parse into it
   independently (§1b). The client needs ONE typed `QueryRequest` and ONE
   lowering `QueryRequest → SessionQuerySpec`.

2. **`preview` does not exist.** Grep for `preview` in api/mcp/surfaces/query
   returns nothing (only maintenance-plan preview). The composer headline UX
   (brief §Architecture) *requires* `preview(spec) → cheap count + first-N +
   facet skeleton` in single-digit ms. `search_envelope` is close but is
   query-string-in, full-hit-out, and not spec-driven.

3. **`complete` is too narrow.** `query_completions(kind, incomplete, unit,
   field)` (`api/archive.py:3103`) returns token metadata but does not take a
   *partial pipeline string* and return "valid next tokens + a live preview."
   The composer needs the richer form.

4. **Read is not executed through the algebra.** `4p1` is defined but the live
   path is named views with `payload: Any` (§1d). The contract should make the
   projection algebra the executor and demote views to server-side macros.

5. **The Protocol family is a strict subset of the real surface.** Read
   Protocols cover list/search/tags/stats/facets/status — not `get_session`/
   read-view, `get_messages`, insights, `resolve_ref`, `query_units`,
   `query_completions`. Write Protocols cover ingest/maintenance/index/tag/
   delete — not mark, annotate, continue, correction, metadata, saved-view,
   workspace, recall-pack, blackboard, assertion-judgment. Everything exists on
   the facade (§1f) but is not declared as a cross-surface contract, so parity
   is unenforced for ~80% of the surface.

6. **`ingest_path` is degenerate proof the contract isn't the real path.**
   Three of four write adapters return `status="failed"` because the real
   scheduling boundary is the daemon HTTP `/api/ingest`, which the CLI hits
   *directly*, bypassing the contract (`cli_write_surface.py:68-99`,
   `mcp_write_surface.py:70-90`). The contract describes a path nothing takes.

---

## 3. Recommendation — ONE `PolylogueService` Protocol, six verbs

Define a single Protocol that is the *entire* client↔substrate contract. Both
the in-proc `Polylogue` facade (what the daemon runs) and the thin client (what
speaks UDS to the daemon, A2) implement **exactly this Protocol**. Static pins
(`assert_implements(Polylogue, PolylogueService)` and
`assert_implements(WireClient, PolylogueService)`) make parity a compile error,
extending the existing `api/contracts/` machinery.

Every verb is `async def verb(req: ReqDTO) -> RespDTO` — one typed request in,
one typed response out. Because both DTOs are pydantic `SurfacePayloadModel`
(frozen, JSON-serializable), the UDS wire form is *generated*, not hand-written,
and `route_contracts.py`'s free-text `response_contract` is replaced by the
`(verb, ReqDTO, RespDTO)` triple.

### The six verbs

| Verb | Request DTO | Response DTO | Replaces |
|------|-------------|--------------|----------|
| `query` | `QueryRequest` | `SearchEnvelope` | `list_sessions` + `search_sessions` + `count` + `search_envelope` + `facets`* |
| `read` | `ReadRequest` | `ReadEnvelope` | `get_session` + `get_messages_paginated` + `SessionReadViewEnvelope` + all `--view` reads |
| `preview` | `QueryRequest` (partial-tolerant) | `PreviewEnvelope` | **new** (composer) |
| `complete` | `CompletionRequest` | `CompletionEnvelope` | generalizes `query_completions` |
| `act` | `ActionRequest` (discriminated union) | `ActionResult` (discriminated union) | `add_tag`/`remove_tag`/`delete_session`/`add_mark`/`save_annotation`/`record_correction`/`set_metadata`/`continue`/`judge_assertion_candidate`/`ingest`/... |
| `status` | `StatusRequest` | `DaemonStatus` | `health_check` + `ops status` + `readiness_check` |

\* `facets` can stay a seventh verb (`facets(QueryRequest) → FacetsResponse`) or
fold into `preview` (the composer already wants a facet skeleton). **Recommend
keep `facets` as its own verb** — it has a distinct expensive/deferred-family
policy (`api/archive.py:_FACET_DEFERRED_FAMILIES`) that `preview` should not pay.

So: **6 core verbs + `facets` = 7-method Protocol.** That is the complete
surface. Everything else on today's 130-method facade is either (a) an internal
lowering the daemon calls, or (b) a specialization expressible as a `read`
projection or an `act` kind.

### 3a. `query` — collapse list/search/count into one

`SearchEnvelope` is already the superset of `SessionListResponse` (it carries
`total` via the shared `spec.count()`, plus hits/cursor/lane/affordances). Make
**`SearchEnvelope` the sole query response**; `SessionListResponse` becomes a
render-time projection of it (drop the second envelope from the wire). The
`retrieval_lane` field + presence of `contains_terms` disambiguates
list-vs-search-vs-similar inside one verb — there is no client-visible reason
for three methods. `next_cursor` (keyset) is the pagination contract; offset is
"best-effort" (`payloads.py:1230`).

`QueryRequest` = the ONE wire model that replaces `MCPSessionQueryRequest` and
the CLI/HTTP parsers. It carries the *surface* vocabulary (a single `query`
DSL string + explicit scalar filters + `retrieval_lane` + pagination) and lowers
to `SessionQuerySpec` via one shared function (promote `build_query_spec`,
`mcp/query_contracts.py:40`, to `surfaces/requests.py`). SessionQuerySpec stays
as the internal lowered form; it is never on the wire.

### 3b. `read` — make the projection algebra the executor

`ReadRequest = { selection: refs|QueryRequest, projection: ProjectionSpec,
render: RenderSpec }` — i.e. wire up `QueryProjectionSpec`
(`projection_spec.py:141`) as the actual `read` input. Named views
(`summary`/`transcript`/`dialogue`/…) become **server-side macros** that expand
to a `ProjectionSpec` via the existing `projection_from_view`
(`projection_spec.py:176`); the client may send either a view name or a raw
`ProjectionSpec`. This directly executes the brief's "fewer named views →
composable projection algebra + user-defined views as `user.db` macros"
(generalize fnm.12). `ReadEnvelope` supersedes `SessionReadViewEnvelope` with a
typed per-family `payload` instead of `Any` (`payloads.py:1788`).

**Render boundary (open question, lean):** the daemon returns the structured
`ReadEnvelope` *and*, when `render.destination` implies bytes
(markdown/html/obsidian/file), the rendered artifact bytes. The thin client
does only *destination delivery* (stdout / clipboard / file / open-browser) —
never layout logic. This keeps a Go/Rust client dumb.

### 3c. `preview` + `complete` — the composer's reason to require the daemon

These two are why the warm daemon is non-optional (brief §Architecture):
- `preview(QueryRequest) → PreviewEnvelope` = `{ total, first_n:
  tuple[SessionSummaryPayload], facet_skeleton, cost_hint, diagnostics }`.
  Partial-tolerant: an unparseable trailing token yields the last valid prefix's
  preview, never an error. Target: single-digit ms on the warm index.
- `complete(CompletionRequest) → CompletionEnvelope`: takes the partial pipeline
  string + cursor position, returns valid next tokens (fields, values from the
  live index, operators, lanes, pipeline stages, set-ops) **plus** an inline
  `preview` for the current prefix. Generalizes `query_completions`
  (`api/archive.py:3103`) from `(kind, incomplete, unit, field)` to
  `(partial_expression, cursor_offset)`.

### 3d. `act` — one dispatched mutation verb, typed both ends

The pipeline terminal-action vocabulary already exists:
`SessionTerminalAction = read|analyze|select|mark|delete|continue`
(`expression.py:286-288`). Generalize it into a single `act` verb over a
**discriminated-union `ActionRequest`** (`kind` field selects the member:
`tag`/`untag`/`mark`/`unmark`/`delete`/`continue`/`annotate`/`correct`/
`set-metadata`/`judge-assertion`/`save-view`/`ingest`/…) returning a
discriminated-union `ActionResult` (reusing today's typed results:
`TagMutationResult`, `DeleteSessionResult`, `MetadataMutationResult`,
`BulkTagMutationResult`, and new `ContinueResult`/`MarkResult`/…).

Why one verb, not N methods: the client, the wire, and the pipeline terminal
(`… | mark`) all dispatch on one enum; a discriminated union preserves full
static typing per kind while the wire stays one message type. This is where
`continue` finally becomes a first-class contract member (today it is only a
pipeline terminal + daemon `thread_continue` templates,
`daemon/thread_continue.py:53`) and where `ingest` stops being the degenerate
`status="failed"` adapter (§2.6) — on the target arch the daemon *is* the
scheduler, so `act(kind=ingest)` is a normal in-proc call server-side and a
normal wire call client-side.

### 3e. `status` — already has its DTO

`DaemonStatus` (`daemon/status.py`, imported by `read_surface.py:31`) is the
shared status model with a `daemon_liveness=False` fallback
(`read_surface.py:104-113`). Keep as-is; `status(StatusRequest) → DaemonStatus`.

---

## 4. Where the DTOs live and how the two sides share them

```
polylogue/surfaces/
  requests.py       ← NEW: QueryRequest, ReadRequest, CompletionRequest,
                       ActionRequest (union), StatusRequest + the ONE
                       QueryRequest→SessionQuerySpec lowering (moved from
                       mcp/query_contracts.build_query_spec)
  projection_spec.py ← ProjectionSpec/RenderSpec/SelectionSpec (exists;
                       becomes the read executor input)
  payloads.py       ← SearchEnvelope, ReadEnvelope(new, was
                       SessionReadViewEnvelope), PreviewEnvelope(new),
                       CompletionEnvelope(new), FacetsResponse, DaemonStatus,
                       ActionResult union (exists, unioned)
polylogue/api/contracts/
  service.py        ← NEW: PolylogueService Protocol (the 7 methods) +
                       assert_implements pins for Polylogue facade and WireClient
```

- **In-proc side (daemon executor):** `Polylogue` facade implements
  `PolylogueService`; its 130 existing methods become the private lowerings the
  7 verbs call. `assert_implements(Polylogue, PolylogueService)`.
- **Wire side (A2):** each verb ↔ one UDS message type; request body = the DTO's
  JSON; reply = the response DTO's JSON. Schema is generated from the pydantic
  models (feeds `render openapi` and the OpenAPI at
  `docs/openapi/search.yaml`). The daemon route table (`route_contracts.py`)
  becomes *generated* from `(verb, ReqDTO, RespDTO)`, ending the free-text
  `response_contract` drift.
- **Thin client (t46):** a `WireClient` that also implements
  `PolylogueService`, serializing DTOs over UDS. `assert_implements(WireClient,
  PolylogueService)`. Because it only marshals DTOs, it can be reimplemented in
  Go/Rust against the generated schema for a sub-10 ms floor. CLI/TUI call the
  same 7 methods regardless of transport (`--no-daemon` break-glass = in-proc
  `Polylogue`, same Protocol).

---

## 5. Migration order (dovetails with t46 / A2)

1. Land `surfaces/requests.py` with `QueryRequest` + the single lowering;
   repoint MCP (`MCPSessionQueryRequest`) and the CLI/HTTP parsers at it. (Kills
   §2.1, the three-front-doors problem.)
2. Define `PolylogueService` Protocol + pin `Polylogue`. Collapse
   list/search/count → `query`→`SearchEnvelope` (§3a).
3. Make `read` execute `QueryProjectionSpec`; demote views to macros; type
   `ReadEnvelope.payload` (§3b).
4. Add `preview` + enrich `complete` (§3c) — unblocks the composer.
5. Introduce `act` union; migrate mutations behind it; make `ingest` a real
   `act` kind (§3d, kills §2.6).
6. Generate the daemon route table + OpenAPI from the Protocol; delete the
   free-text `response_contract` column.

---

## 6. Open questions for the operator

1. **`act`: one dispatched union verb vs. a handful of typed verbs?** Union is
   cleaner for the wire and the pipeline terminal; typed verbs are marginally
   easier to grep. Recommend union with a discriminated `ActionResult`.
2. **`analyze` / `select` pipeline terminals** (`expression.py:286`) — verbs, or
   projections of `read`/`query`? Recommend `analyze` = a `read` projection
   family, `select` = a `query` field-projection; not new verbs.
3. **Render locus:** daemon returns rendered bytes (dumb polyglot client) vs.
   structured-only (client renders). Recommend daemon-renders-bytes,
   client-delivers-destination (§3b).
4. **Streaming:** should `query`/`preview` stream first-rows-then-refine over
   UDS for the composer, or is a single fast reply enough? This is A2's call but
   the DTO must be stream-friendly (SearchEnvelope already paginates by cursor).
5. **`facets` as 7th verb vs. folded into `preview`** — recommend separate, to
   isolate the expensive deferred-family cost policy.

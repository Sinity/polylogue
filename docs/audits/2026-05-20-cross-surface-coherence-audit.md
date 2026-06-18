# Cross-Surface Coherence Audit (2026-05-20)

Audit of polylogue read/write surfaces against shared contract substrate.
Ref #1282.

## Scope

Surfaces in scope:

- **CLI** — `polylogue/cli/` (Click app, commands, query mode).
- **MCP server** — `polylogue/mcp/` (FastMCP tools).
- **Daemon HTTP** — `polylogue/daemon/http.py` (`/api/...` routes) and the
  satellite `polylogue/browser_capture/server.py` (`/v1/...` routes).
- **Python API** — `polylogue/api/__init__.py` (`Polylogue` async facade
  composed via `PolylogueArchiveMixin`, `PolylogueInsightsMixin`,
  `PolylogueIngestMixin`).

Out of scope here but enumerated where relevant: the daemon web reader,
the TUI, the browser extension, and the `polylogue-hook` bash shim under
`contrib/`.

This is an audit, not a refactor. Every substantive finding is filed as a
durable follow-up issue and cross-referenced below.

## Substrate

The contract substrate already exists:

- `polylogue/surfaces/payloads.py` (1127 lines): canonical typed
  payload envelopes — `ConversationListResponse`,
  `ConversationListRowPayload`, `FacetsResponse`, `DaemonStatus`,
  `MachineErrorPayload`, mutation result types.
- `polylogue/api/contracts/` — `CLIReadSurface`, `MCPReadSurface`,
  `APIReadSurface`, `TUIReadSurface`. These are `assert_implements`
  shadow adapters used by mypy contract checks. They are **not** in the
  request path of production CLI / MCP / HTTP code.
- `polylogue/mcp/payloads.py` — MCP-local envelope variants
  (`MCPPaginatedQueryResultPayload`, `MCPErrorPayload`) plus direct use of
  the shared `SearchEnvelope`.
- `polylogue/mcp/query_contracts.py` — `MCPConversationQueryRequest`,
  the typed input spec that already encodes every filter parameter
  declared one at a time in `mcp/server_tools.py`.

The gap is wiring, not modeling.

## Capability × Surface Matrix

`yes` = native, first-class implementation on that surface.
`indirect` = reachable through a different primitive but not a
named peer.
`partial` = present but with diverged envelope or partial coverage.
`—` = absent.

| Capability | CLI | MCP | Daemon HTTP | Python API |
| --- | --- | --- | --- | --- |
| List conversations | yes (`polylogue list`, query mode) | yes (`list_conversations`) | yes (`GET /api/conversations`) | yes (`Polylogue.list_conversations`) |
| Search conversations | yes (query mode, `--lexical`/`--semantic`) | yes (`search`) | yes (`GET /api/conversations?query=`) | yes (`Polylogue.search`) |
| Get conversation summary | yes (`polylogue show <id>`) | yes (`get_conversation`, summary only) | yes (`GET /api/conversations/{id}`, full) | yes (`Polylogue.get_conversation`) |
| Get conversation messages | yes (`show`) | yes (`get_messages`, separate call) | yes (folded into `get_conversation`) | yes |
| Get raw acquired artifacts | yes (`polylogue raw`) | yes (`raw_artifacts`) | yes (`/api/conversations/{id}/raw`) | yes |
| Per-conv cost | — | partial (forecast via `cost_outlook`) | yes (`/api/conversations/{id}/cost`) | indirect |
| Provenance | — | — | yes (`/api/conversations/{id}/provenance`) | indirect |
| Topology / lineage | — | partial (`get_session_tree`, different envelope) | yes (`/api/topology/...`) | indirect |
| Similar / neighbors | — | partial (`neighbor_candidates`, different envelope) | yes (`/api/conversations/{id}/similar`) | yes |
| Facets | — | — | partial (`providers`, `tags` only) | — |
| Stats / coverage | yes (`polylogue stats`) | yes (`stats`, `archive_coverage`) | — | yes |
| Daemon status | yes (`polylogue ops status`) | yes (`readiness_check`) | yes (`/api/status`, `/api/healthz`) | yes |
| Session insights (profile, classification, phases, work events, threads) | yes (insights commands) | yes (per-insight tools) | yes (web shell + read endpoints) | yes |
| Tag / metadata mutations | yes (`tags`) | yes (mutation tools) | yes (web shell routes) | yes |
| Maintenance / convergence ops | yes (`check`, `reset`) | yes (maintenance tools) | yes (HTTP endpoints) | yes |
| Browser captures intake | — | — | separate server (`/v1/...` on 8765) | — |
| Hook events ingestion | shell shim only | — | — | — |

## Findings

### F1. Five list-row shapes for the same data (extends #859, #873, #1266)

The "list of conversations" row is constructed independently in five
places:

- CLI rich rendering (`cli/query_output.py`).
- CLI JSON output (`cli/query_output.py:_conv_to_dict`).
- MCP (`MCPPaginatedQueryResultPayload` in `mcp/payloads.py`).
- Daemon HTTP `_do_list` / `_do_search_list` in
  `polylogue/daemon/http.py` (inline dicts with reader-only enrichment:
  `target_ref`, `anchor`, `actions`, `flags`, `repo`, `cwd_display`).
- TUI (`ConversationListResponse` in `surfaces/payloads.py`).

The shared `ConversationListResponse` model is only used by the TUI
and by the `api/contracts/` shadow adapters. Production CLI, MCP, and
HTTP paths never call into it. #859 closed without wiring this through.

**Recommendation R1.** New follow-up issue files the wiring work.

### F2. Three search-hit shapes

- Daemon HTTP returns `hits` with nested `match` (reader anchors,
  actions).
- MCP returns the shared `SearchEnvelope`.
- CLI rolls its own payload via `_search_hit_to_payload`.

#1266 standardized the ranked envelope but the production CLI/MCP/HTTP
search paths still construct local shapes.

### F3. Three error wire formats

- `MachineErrorPayload` — `{status, code, message, command, details}` —
  used by CLI JSON output and the cli-output schema artifacts under
  `docs/schemas/cli-output/`.
- `MCPErrorPayload` — `{error, code, detail?}` — `mcp/payloads.py`.
- Daemon HTTP — raw `{error: <code>}` strings inline.
- Browser-capture HTTP server (`browser_capture/server.py`) defines its
  own variants of all of the above.

Same logical concept ("operation failed, here is structured why"); four
encodings on the wire.

### F4. Two HTTP servers running in parallel

- Daemon `/api/...` on the configurable daemon port.
- Browser-capture `/v1/browser-captures`, `/v1/archive-state`,
  `/v1/status` on default 8765 (`polylogue/browser_capture/server.py`).

Each ships its own auth, CORS, error envelope, and status envelope code.
Browser extension config points at 8765 by default.

### F5. `FacetsResponse` half-implemented

`FacetsResponse` (`surfaces/payloads.py:965`) declares nine facet
dimensions. The daemon `_do_facets`
(`polylogue/daemon/http.py:1821`) serves only `providers` and `tags`
and hard-codes the other seven. CLI and MCP have no facet peer at all.
Either the model is too wide for the current behavior, or seven
fields are silently wrong.

### F6. MCP `get_conversation` returns summary only

Daemon HTTP and the Python API return the full conversation (header +
messages) in one call. MCP `get_conversation` returns only the summary
and requires a second `get_messages` round trip. The asymmetry is
unflagged.

### F7. MCP filter parameters duplicated 4 ways

`polylogue/mcp/server_tools.py` declares the ~30 filter parameters for
`list_conversations` and `search` by hand (lines 43–228, ~200 lines).
`MCPConversationQueryRequest` in `mcp/query_contracts.py` already
encodes every one of them. Adding a new filter today is a four-file
edit: filter chain, `query_contracts.py`, `server_tools.py`, and the
matching CLI flag.

### F8. Read surfaces missing on CLI and MCP

The daemon HTTP server exposes `/cost`, `/provenance`, `/topology`, and
`/similar` per conversation, used by the web reader. CLI has no
`polylogue conversation <id> cost|provenance|topology|similar` peer.
MCP has only partial peers (`cost_outlook` is forecast not retrieval;
`get_session_tree` and `neighbor_candidates` use diverged envelopes).

### F9. Browser-capture HTTP server is unprotected by daemon plumbing

The browser-capture receiver has its own token-and-origin discipline
(`BrowserCaptureReceiverConfig`). Any consolidation under F4 must
preserve that surface as a route-level guard.

### F10. `polylogue-hook` is an unpackaged bash shim

`polylogue-hook` lives in `contrib/` as a shell script rather than a
console_script entry point in `pyproject.toml`. Hook integration is the
only first-party surface that is not pip-installable. #1213 covers the
per-event library; the packaging gap is independent.

### F11. Cluster of contract adapters is shadow-only

`polylogue/api/contracts/{cli,mcp,api,tui}_*.py` exist solely to satisfy
`assert_implements` mypy checks. The CLI/MCP/HTTP surfaces do not
instantiate them. Resolving F1 / F2 / F3 in production code makes these
adapters either the real implementation or removable; today they are
neither.

### F12. No OpenAPI emission from typed payloads

`devtools render-openapi` emits `docs/openapi/search.yaml` from the
typed `SearchEnvelope` Pydantic models (#1266). The daemon's
`/api/conversations`, `/api/conversations/{id}`, `/api/topology/...`,
`/api/conversations/{id}/{cost,provenance,similar}`,
`/api/facets`, `/api/status`, `/api/healthz`, and the browser-capture
`/v1/...` routes have no machine-readable schema. Web reader, browser
extension, and external clients hand-type each fetch.

## Naming / vocabulary

The `provider` vs `source` dual-vocabulary period (see
`docs/architecture.md` § "Dual Vocabulary Period") affects every
surface uniformly: storage column `provider_name`, CLI `--provider`,
MCP `provider`, HTTP `?provider=`. No surface has moved yet;
#1022 and #1214 track the staged transition. The audit confirms no
half-moved surface exists today.

`conversation_id` vs `conv_id` is split inside the daemon:
`get_conversation`, `_do_get_conversation_attachments`,
`_do_get_conversation_raw`, `_do_get_conversation_cost`, etc. accept
`conv_id` parameters internally while the public path segment is the
conversation ID; the public surface vocabulary is consistent.
No external rename is required.

## Pagination

CLI, MCP, and HTTP all accept `limit` + `offset`. MCP and HTTP also
accept opaque `cursor`. CLI does not surface cursor pagination, which
is an asymmetric ergonomics gap but not a wire-format inconsistency
(it is the same envelope; CLI just ignores the cursor field).
Keyset pagination (#1268) is tracked separately.

## Refs

- #859 — shared read-surface query/status contracts (closed; F11 reopens
  the wiring gap as a new issue).
- #873 — make ranked results explainable and pagination-stable (open).
- #1266 — typed ranked-result envelope across surfaces (closed; F2
  remains for production wiring).
- #1247 — typed import operation contract (open).
- #1197 — persistent operation registry + status surface (open).
- #1224 — health-endpoint contract tests (open).
- #1269 — scoped vs global facets (open).
- #1250 — route MCP through facade not direct services (open).
- #1213 — per-event hook script library (open).
- #1022 / #1214 — source-vocabulary refactor (open).
- #1218 — CLI `status --convergence` parity (open).

## Follow-up issues filed

| Finding | Issue | Title |
| --- | --- | --- |
| F1, F11 | #1414 | refactor(surfaces): wire ConversationListResponse through CLI/MCP/daemon list paths |
| F2, F3 | #1415 | refactor(surfaces): adopt MachineErrorPayload across MCP, daemon HTTP, browser-capture |
| F7 | #1416 | refactor(mcp): auto-derive list/search tool parameters from MCPConversationQueryRequest |
| F4, F9 | #1417 | refactor(daemon): fold browser-capture HTTP server into daemon /api/v1/captures |
| F12 | #1418 | feat(devtools): emit daemon HTTP OpenAPI from typed payload models |
| F6, F8 | #1419 | feat(surfaces): CLI and MCP peers for daemon /cost, /provenance, /topology, /similar |
| F5 | #1420 | feat(facets): implement or shrink FacetsResponse dimensions (daemon serves 2 of 9) |
| F10 | #1421 | feat(hooks): promote polylogue-hook to console_script entry point |

R9 ("`polylogue serve {daemon,mcp,capture,all}` umbrella") from the
parent issue is intentionally not filed as a separate ticket; it is a
follow-on to R4 and would be reconsidered after F4 lands.

## Verification

This is an audit document. No production code is touched. The follow-up
issues each carry their own acceptance criteria for the implementation
PRs they will spawn.

# Web workbench v0 map (#1846)

Status: executable first slice plus residual map.

This file is intentionally narrow. #1846 is the local daemon web workbench over existing archive/query/read/assertion/ref contracts. It must not create a second query grammar, read-view registry, assertion model, evidence-ref model, or browser-capture authority path.

## Implemented in this slice

The workbench can now read compact evidence state for a selected session without loading raw transcripts by default.

Backend:

- `GET /api/sessions/:id/recovery?report=digest&format=json`
- `GET /api/sessions/:id/recovery?report=work-packet&format=json|markdown`
- `GET /api/sessions/:id/read?view=messages|recovery|context-pack|raw&format=json`
- `GET /api/assertions?target_ref=&scope_ref=&kind=&status=&context_inject=&limit=`

Route contracts:

- `/api/sessions/:id/recovery` is `read_detail`, `stable`, `bearer_if_configured`.
- `/api/sessions/:id/read` is `read_detail`, `stable`, `bearer_if_configured`.
- `/api/assertions` is `user_overlay`, `stable`, `bearer_if_configured`.
- All three routes return shared DTO envelopes and are published in the generated OpenAPI surface.

Shared DTOs:

- `RecoveryReadPayload` wraps `RecoveryDigest`, `RecoveryWorkPacket`, or rendered work-packet markdown.
- `AssertionClaimPayload` / `AssertionClaimListPayload` wrap user-tier assertion envelopes for daemon/MCP/web reads.
- The daemon route goes through `Polylogue.recovery_read_payload(...)` and `Polylogue.list_assertion_claim_payloads(...)`; it does not import user-tier storage serialization helpers.

Web shell:

- Adds an `Evidence` inspector tab.
- Fetches `RecoveryWorkPacket` and assertion claims for the selected session.
- Loads `GET /api/read-view-profiles` and renders a small profile selector from shared read-view metadata.
- Executes supported single-session profiles through `GET /api/sessions/:id/read`.
- Keeps raw data behind an explicit opt-in drawer: metadata loads first from provenance; bounded preview requires a separate click.
- Uses the existing user-state routes for overlay mutation flows: mark toggles, annotations, saved views, recall packs, and workspaces all return the shared mutation result envelope.

Explicit non-exposure:

- `continue` and `blame` remain CLI/MCP recovery-report presets. They are not exposed through `/api/sessions/:id/recovery`; the stable route is intentionally limited to the storage-free digest/work-packet DTOs.

## V0 page/component map

| Shell route | Role | Existing backing routes | New or strengthened components |
|---|---|---|---|
| `/` | Search/list/read cockpit | `/api/sessions`, `/api/facets`, `/api/query-*`, `/api/read-view-profiles` | `ReadProfileSelector`, `Evidence` inspector tab |
| `/s/:session_id` | Session deep link | `/api/sessions/:id`, `/messages`, `/provenance`, `/raw`, `/insights` | Evidence panel over recovery/assertions |
| `/w/:mode` | Stack/compare drilldown | `/api/stack`, `/api/compare` | unchanged in this slice |
| `/p` | Paste evidence browser | `/api/paste-browser` | unchanged in this slice |
| `/a` | Attachment evidence browser | `/api/attachments`, `/api/sessions/:id/attachments` | unchanged in this slice |

## API DTOs consumed by v0

Search and query:

- `SearchEnvelope` from `GET /api/sessions`.
- `FacetsResponse` / `FacetBucketsPayload` from `GET /api/facets`.
- query completion metadata from `GET /api/query-completions`.
- `QueryUnitEnvelope` / `QueryUnitAggregateEnvelope` from `GET /api/query-units`.

Reader:

- read-view profile payloads from `GET /api/read-view-profiles`.
- read-view execution envelopes from `GET /api/sessions/:id/read`, including the shared context-pack DTO.
- session detail payload from `GET /api/sessions/:id`.
- session message payloads from `GET /api/sessions/:id/messages`.
- provenance/raw payloads from `/provenance` and `/raw`, still explicit opt-in.
- insights, topology, similar, and attachment payloads from existing session routes.

Evidence:

- `RecoveryReadPayload` from the new recovery route.
- `AssertionClaimListPayload` from the new assertions route.
- `ObjectRef` / `EvidenceRef` strings inside recovery/work-packet entries stay the durable evidence identity.

Browser capture/readiness:

- Browser capture remains a separate #1824/#1847 capability boundary. The workbench displays read-only readiness from `/api/status.browser_capture` and `component_readiness.browser_capture`; it does not call receiver write routes as generic archive mutations.

## Remaining backend endpoints

1. Shared read-view execution route now covers the supported single-session profiles (`messages`, `recovery`, `context-pack`, `raw`). Broader profiles (`context`, `neighbors`, `correlation`) still need dedicated execution semantics before the selector enables them.
2. Overlay mutation envelopes for marks/annotations/saved views/recall/workspaces are implemented through `polylogue/daemon/user_state_http.py` and route through the shared mutation result envelope. Remaining work here is browser-flow polish, not another DTO family.
3. Browser-capture readiness is satisfied by `GET /api/status`; add a dedicated adapter only if a future panel needs more than safe receiver state (`spool_ready`, `allowed_origins`, `auth_required`, component readiness).
4. Bounded raw-preview contract promotion: this slice already prefers provenance `include_raw=1&bytes=` in the shell; #1847 can decide whether `/api/sessions/:id/raw` remains shell-supported only or becomes a narrower stable metadata route.
5. Remaining shell-supported workbench routes (`/api/sources`, cost, raw metadata, attachments, paste, stack/compare helpers) should either stay shell-only with explicit metadata or be promoted only after they have typed DTOs and OpenAPI coverage.

## Parallel PR slices

PR-1846-B, landed by this slice: recovery/work-packet HTTP, assertion-read HTTP, route contracts, web evidence panel, focused reader tests.

PR-1846-C, landed before this map refresh: overlay mutation paths use shared mutation/error envelopes and same-origin/bearer tests remain strict under #1847.

PR-1846-D, partially landed after the first evidence slice: execute supported single-session read profiles over HTTP instead of only displaying profile metadata.

PR-1846-E, partially landed by this slice: raw drawer hardening via provenance metadata, no automatic raw fetch, bounded preview button, and explicit privacy posture in the UI.

PR-1846-F, landed after the first evidence slice: browser-capture/readiness panel consumes #1824/#1847 status DTOs without merging receiver auth with archive auth.

PR-1846-G, partly landed by this slice: fixture-backed DOM smoke covers shell hooks plus recovery/assertion endpoint payloads. A full browser click-through remains useful once the visual lane is stable in every checkout.

## Test surface

Landed tests:

- route-contract lookup for `/api/assertions` and `/api/sessions/:id/recovery`.
- route-contract lookup for `/api/sessions/:id/read`.
- stable route-contract checks for `/api/assertions`, `/api/sessions/:id/recovery`, and `/api/sessions/:id/read`.
- generated OpenAPI checks for `/api/assertions` and `/api/sessions/:id/recovery`.
- recovery endpoint tests for digest JSON, work-packet JSON, work-packet markdown, invalid report/format pairs, and missing sessions.
- read-view execution route tests for `messages`, `recovery`, `raw`, unsupported views, and unsupported formats.
- assertion endpoint tests for default `active,candidate`, kind/context filters, `status=all`, and token-gated reads.
- shell smoke checks for `Evidence`, `read-view`, recovery/assertion endpoint hooks, and explicit raw opt-in copy.
- visual/DOM smoke checks for the evidence tab hooks plus recovery/assertion endpoint payloads over the reader fixture.
- raw drawer smoke checks that the shell uses provenance and only fetches bounded preview after an explicit button.
- no-local-path leak checks for the new reader JSON routes.
- browser-capture readiness chip wiring over `/api/status.browser_capture`, including safe-field checks that the web workbench never receives a receiver spool path.

Still needed:

- a full browser click-through smoke once the project has a stable Playwright/Vitest lane in this checkout.
- deeper browser click-through coverage for the stabilized routes once the visual lane can run deterministically in every checkout.

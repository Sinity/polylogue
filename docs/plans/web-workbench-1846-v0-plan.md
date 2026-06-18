# Web workbench v0 map (#1846)

Status: executable first slice plus residual map.

This file is intentionally narrow. #1846 is the local daemon web workbench over existing archive/query/read/assertion/ref contracts. It must not create a second query grammar, read-view registry, assertion model, evidence-ref model, or browser-capture authority path.

## Implemented in this slice

The workbench can now read compact evidence state for a selected session without loading raw transcripts by default.

Backend:

- `GET /api/sessions/:id/recovery?report=digest&format=json`
- `GET /api/sessions/:id/recovery?report=work-packet&format=json|markdown`
- `GET /api/assertions?target_ref=&scope_ref=&kind=&status=&context_inject=&limit=`

Route contracts:

- `/api/sessions/:id/recovery` is `read_detail`, `shell_supported`, `bearer_if_configured`.
- `/api/assertions` is `user_overlay`, `shell_supported`, `bearer_if_configured`.
- Both are real workbench routes, but not advertised as stable public API until #1847 promotes the daemon DTO boundary.

Shared DTOs:

- `RecoveryReadPayload` wraps `RecoveryDigest`, `RecoveryWorkPacket`, or rendered work-packet markdown.
- `AssertionClaimPayload` / `AssertionClaimListPayload` wrap user-tier assertion envelopes for daemon/MCP/web reads.
- The daemon route goes through `Polylogue.recovery_read_payload(...)` and `Polylogue.list_assertion_claim_payloads(...)`; it does not import user-tier storage serialization helpers.

Web shell:

- Adds an `Evidence` inspector tab.
- Fetches `RecoveryWorkPacket` and assertion claims for the selected session.
- Loads `GET /api/read-view-profiles` and renders a small profile selector from shared read-view metadata.
- Keeps raw data behind an explicit opt-in drawer: metadata loads first from provenance; bounded preview requires a separate click.

Explicit non-exposure:

- `continue` and `blame` remain CLI/MCP recovery-report presets. They are not exposed through `/api/sessions/:id/recovery` in this slice because #1847 has not promoted a general report-preset HTTP API.

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
- `QueryUnitEnvelope` from `GET /api/query-units`.

Reader:

- read-view profile payloads from `GET /api/read-view-profiles`.
- session detail payload from `GET /api/sessions/:id`.
- session message payloads from `GET /api/sessions/:id/messages`.
- provenance/raw payloads from `/provenance` and `/raw`, still explicit opt-in.
- insights, topology, similar, and attachment payloads from existing session routes.

Evidence:

- `RecoveryReadPayload` from the new recovery route.
- `AssertionClaimListPayload` from the new assertions route.
- `ObjectRef` / `EvidenceRef` strings inside recovery/work-packet entries stay the durable evidence identity.

Browser capture/readiness:

- Browser capture remains a separate #1824/#1847 capability boundary. The workbench may display capture readiness later, but must not call receiver write routes as generic archive mutations.

## Remaining backend endpoints

1. Shared read-view execution route: either `GET /api/sessions/:id/read?view=&format=` or an explicit route map for every profile. Current slice only displays profiles and maps supported ones to existing routes.
2. Typed overlay mutation envelopes for marks/annotations/saved views/recall/workspaces. Existing routes work, but #1847 should decide stability and shared mutation DTO guarantees.
3. Browser-capture read-only readiness adapter if `GET /api/status` is not enough for the workbench panel.
4. Bounded raw-preview contract promotion: this slice already prefers provenance `include_raw=1&bytes=` in the shell; #1847 can decide whether `/api/sessions/:id/raw` remains shell-supported only or becomes a narrower stable metadata route.
5. Generated route/OpenAPI surface that includes stable routes and intentionally documented shell-supported workbench routes without implying public API stability.

## Parallel PR slices

PR-1846-B, landed by this slice: recovery/work-packet HTTP, assertion-read HTTP, route contracts, web evidence panel, focused reader tests.

PR-1846-C: promote one overlay mutation path to shared mutation/error envelopes and keep same-origin/bearer tests strict.

PR-1846-D: execute read profiles over HTTP instead of only displaying profile metadata.

PR-1846-E, partially landed by this slice: raw drawer hardening via provenance metadata, no automatic raw fetch, bounded preview button, and explicit privacy posture in the UI.

PR-1846-F: browser-capture/readiness panel consuming #1824/#1847 DTOs without merging receiver auth with archive auth.

PR-1846-G, partly landed by this slice: fixture-backed DOM smoke covers shell hooks plus recovery/assertion endpoint payloads. A full browser click-through remains useful once the visual lane is stable in every checkout.

## Test surface

Landed tests:

- route-contract lookup for `/api/assertions` and `/api/sessions/:id/recovery`.
- check that these routes stay `shell_supported`, not stable public API.
- recovery endpoint tests for digest JSON, work-packet JSON, work-packet markdown, invalid report/format pairs, and missing sessions.
- assertion endpoint tests for default `active,candidate`, kind/context filters, `status=all`, and token-gated reads.
- shell smoke checks for `Evidence`, `read-view`, recovery/assertion endpoint hooks, and explicit raw opt-in copy.
- visual/DOM smoke checks for the evidence tab hooks plus recovery/assertion endpoint payloads over the reader fixture.
- raw drawer smoke checks that the shell uses provenance and only fetches bounded preview after an explicit button.
- no-local-path leak checks for the new reader JSON routes.

Still needed:

- a full browser click-through smoke once the project has a stable Playwright/Vitest lane in this checkout.
- generated OpenAPI/docs updates if #1847 decides shell-supported routes should be listed in generated API docs.

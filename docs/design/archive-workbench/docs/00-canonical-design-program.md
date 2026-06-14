# Polylogue Archive Workbench — canonical design program

## North star

Polylogue is not just a chat search UI. It should be a local, inspectable workbench for the archived reality of AI-agent work: conversations, messages, raw payloads, tool traces, pasted materials, attachments, costs, provenance, topology, user marks, saved views, context bundles, operations, and cross-surface command equivalents.

The product promise is: **show me what happened, why it is trustworthy, how it relates to other work, and what safe next action is available.**

The design should make long, messy, multi-agent technical sessions readable without hiding uncertainty. A user should be able to open a 300-message Claude Code run, identify copied repo context, inspect attachments, follow a sidechain, compare a Codex follow-up, select a range into a recall/context bundle, copy exact material, and see which actions are unavailable because the archive lacks evidence.

## Core loops

1. **Observe** — daemon status, source health, ingest/catch-up state, capture state, schema compatibility, maintenance and metrics.
2. **Find** — ranked search, scoped/global facets, query history, saved views, result explanations, exact raw/source access.
3. **Read** — long-session reader, message cards, folds, anchors, copy actions, raw/source toggles, paste/attachment badges.
4. **Relate** — native lineage: continuations, sidechains, subagents, forks, unresolved parents, repaired/quarantined edges, parent-chain stacks.
5. **Compose** — select messages/materials/topology facts into recall packs or context bundles with explicit included/omitted/unavailable objects.
6. **Continue** — launch/copy/open continuation prompts without pretending that launch creates canonical lineage.
7. **Verify** — visual smoke, OpenAPI/contracts, CLI/MCP parity, machine-readable error payloads, degraded-state fixtures.

## Current vs target discipline

Do not let the UI manufacture truth. If a route, parser field, derivative, or action is not backed by the codebase, the UI may show it only as `target`, `unavailable`, `needs evidence`, or `contract gap`.

Important semantics:

- User actions can launch work elsewhere, copy prompt material, or create non-canonical launch intents.
- Canonical topology edges come from observed/imported/repaired archive evidence.
- Forks must be native lineage objects when evidence exists: branch point, sibling attempts, divergence, raw evidence, status, and repair/quarantine state.
- A pasted-material boolean is a filter signal, not enough for precise typed-only or paste-only copy. Precise copy needs exact or projected paste spans with boundary metadata.
- Attachments are archive objects with identity, metadata, storage state, preview/extraction/derivative state, safety/privacy state, provenance, and actions.

## Information architecture

The workbench is organized into rooms, each with a clear purpose.

### Command Center

The starting view. Shows daemon readiness, active ingest/catch-up, latest conversations, source/capture health, maintenance warnings, schema/FTS/embedding states, and quick paths to search, reader, materials, lineage, and operations. This is not a marketing dashboard; it is a launchpad and health surface.

### Search Room

A query cockpit over the archive. It should expose rank, why-this-matched, lane/source, FTS/vector/metadata contribution where available, scoped/global facets, query limits, pagination, saved views, and command equivalents. Missing facets must render as contract gaps, not empty filters.

### Session Room

The main long-session reader. It uses stable message anchors, message cards, folds, dense/comfortable modes, selected ranges, action rails, paste/attachment/provenance badges, source/raw toggles, parent/branch indicators, cost/model usage, and huge-session virtualization/window state.

### Message Object Inspector

A side or modal object inspector for one message/content block/material/attachment/topology edge. Tabs include rendered, raw JSON, source artifact, provenance, usage/cost, topology, actions, and diagnostics. Every object should have deterministic copy and permalink behavior.

### Materials Lab

Pastes and attachments become first-class archive materials. The lab includes paste browser, attachment library, filters by state, object drawer, missing blob handling, private/redacted states, derivative slots, preview/extraction readiness, and copy/export actions.

### Lineage Room

Observed topology browser. Shows parent/current/child stacks, sidechains, subagents, continuations, forks, unresolved edges, repaired edges, quarantined cycles, branch points, confidence/evidence details, and missing-target placeholders. It must visually separate observed lineage from user launch actions.

### Workspaces

Multi-log modes are distinct:

- **Tabs** for quick switching.
- **Stack** for parent/current/child/sidechain reading.
- **Compare** for two attempts or parent-child diffing.
- **Timeline** for chronological reconstruction.
- **Cluster** only as a target overview/index, not the primary reader.

### Context Composer

A selection-to-bundle surface. Inputs can include messages, ranges, conversations, paste spans, attachment metadata, topology facts, search result groups, and user notes. Output is a copyable/savable bundle with included, omitted, unavailable, stale, and redacted items visible.

### Grounded Intelligence Layer

Not a chatbot bolted onto the archive. It is a set of grounded transformations: explain this result, summarize selected range with citations, find related sessions, construct a continuation prompt, compare attempts, detect unresolved lineage, produce a context bundle. Every result must cite archive object refs and caveats.

### Operations Room

A bounded operator dashboard over health/status/events/metrics/maintenance/source state. It must handle stale, disconnected, schema mismatch, catch-up backlog, WAL/memory pressure, source failure, and degraded ingest without blocking the reader.

## Object model

The current product can keep existing public target refs stable while introducing a broader UI object vocabulary internally.

### ArchiveObjectRef

A stable UI ref for `conversation`, `message`, `content_block`, `paste_span`, `attachment`, `topology_edge`, `workspace`, `context_bundle`, `saved_view`, `mark`, `annotation`, `raw_artifact`, `source`, and `operation` objects. It should support display labels, archive-local identity, source identity, object version, and stale/missing indicators.

### MessageRenderEnvelope

Shared between conversation detail and paginated messages. It should carry message id, conversation id, role, source/provider vocabulary, timestamps, content blocks, text fallback, parent message id, branch index, token/cost/model usage where safe, paste spans, attachment refs, topology hints, provenance refs, flags, anchors, and action availability.

### ActionAvailability

Every action is explicit: enabled, disabled, partial, loading, target, dangerous, or unavailable. Disabled actions need short human reasons and optional repair paths. This should cover copy text/markdown/raw/permalink, copy selected range, typed-only copy, paste-only copy, open raw/source, inspect provenance, add mark, annotate, add to context, open stack, compare, continue elsewhere, and export material.

### MaterialObject

A unified UI object for paste spans and attachments. It includes kind, boundary certainty, MIME/type, size, hash, storage state, preview state, extraction state, derivative state, private/redacted state, source message refs, raw artifact refs, and copy/export actions.

### TopologyEdgeDetail

A detailed edge view with branch type, status, source and target conversation/message refs, confidence, raw evidence, observed/resolved/repaired timestamps, repair/quarantine details, parser/source, and caveats. This should be include-gated if payload size or privacy require it.

### WorkspaceManifest

A saved workspace of tabs/stack/compare/timeline modes with object refs, layout, active selections, query state, and broken-ref repair metadata.

### ContextBundle

An inspectable context package: selected archive objects, rendered excerpts, omitted items, redacted items, unavailable items, topology facts, source caveats, size budget, intended model/task, copy recipe, and provenance manifest.

## State grammar

Every room and object should render these states intentionally:

- Loading: first load, pagination, virtualized placeholder, action pending.
- Empty: no results, no attachments, no paste spans, no topology, no selections.
- Partial: some fields unavailable, missing facets, summary without raw evidence, projected paste span.
- Stale: daemon data older than threshold, SSE disconnected, cached health, replay/catch-up in progress.
- Disconnected: daemon unreachable, reader still shows last known data if cached.
- Degraded: source/capture/FTS/vector/embedding/cost/provenance component impaired.
- Critical: schema mismatch, reader available but ingest disabled, privacy guard active, corrupted object.
- Missing target: topology edge points to not-yet-ingested or deleted conversation/message.
- Missing blob: attachment metadata exists but payload is unavailable.
- Redacted/private: object exists but display/copy is intentionally restricted.
- Unsupported: preview/extraction/action not supported for this object type.
- Failed: query/action/maintenance failure with machine-readable error and retry/inspect path.

## Microinteraction principles

- Hover reveals actions; keyboard focus reveals the same actions.
- Copy actions produce visible confirmation and specify what was copied.
- Disabled actions are not hidden when their absence would be confusing; they explain why they are unavailable.
- Every card has a stable anchor and copyable permalink.
- Raw/source access is one step away for evidence-heavy users.
- Long text, code, tool output, thinking, and pasted blocks have different fold policies.
- Dense mode preserves actionability; it does not become an unreadable table.
- The reader never claims a continuation/fork was created until the archive observes or repairs that relation.

## Implementation strategy

Land the workbench in slices. The first slice is not a complete redesign. It is a contract spine: `MessageRenderEnvelope` parity and `ActionAvailability`. That unlocks reader quality, visual smoke, and later rooms.

Recommended order:

1. MessageRenderEnvelope parity.
2. ActionAvailability registry.
3. ArchiveObjectRef and selection set.
4. Session Room message cards, folds, and copy actions.
5. Materials Lab paste/attachment object drawers.
6. Topology edge detail and fork lens.
7. Context Composer MVP.
8. Stack/compare workspaces.
9. Operations Room.
10. Cross-surface parity and OpenAPI.

## Non-goals

- No UI-only filters that contradict query semantics.
- No fake canonical topology from user launch actions.
- No attachment previews/derivatives unless storage/privacy/routes support them.
- No broad target ref expansion without migration tests for user-state identity.
- No advanced intelligence output without archive object refs and caveats.

# Implementation roadmap — Polylogue Archive Workbench

## Slice 1 — MessageRenderEnvelope parity

Make conversation detail and paginated messages emit the same rich message object. This removes the main blocker for huge-session readers, visual smoke, and consistent copy/inspect behavior.

Acceptance evidence: daemon contract tests, visual DOM test for paginated reader, fixture containing parent message, branch index, paste flag/span, attachment metadata, raw/source refs, and disabled action reasons.

## Slice 2 — ActionAvailability registry

Replace scattered frontend action assumptions with a typed registry. Each action has stable id, target object kind, state, reason, side effects, confirmation, and result shape.

Acceptance evidence: action matrix tests, UI disabled reasons, copy result fixtures, no hidden unavailable actions for confusing cases.

## Slice 3 — Session Room upgrade

Implement message cards, action rail, fold policies, dense mode, range selection, copied state, raw/source tabs, long-session outline, and virtualization-aware anchors.

Acceptance evidence: screenshot/DOM fixtures for ordinary, huge, paste-heavy, attachment-heavy, missing raw, private/redacted, and degraded sessions.

## Slice 4 — Materials Lab

Turn paste spans and attachments into object drawers. Exact/projected/fallback paste states must drive copy availability. Attachment drawers must separate metadata, storage, preview, extraction, privacy, and provenance states.

Acceptance evidence: exact span, projected span, whole-message fallback, missing blob, redacted attachment, unsupported preview, failed extraction fixtures.

## Slice 5 — Native lineage detail

Expose enough topology detail to render observed lineage confidently: branch type, status, raw evidence, confidence, observed/resolved/repaired timestamps, quarantine detail, source/target refs, and missing-target placeholders.

Acceptance evidence: unresolved, resolved, repaired, quarantined, fork, sidechain, subagent, continuation fixtures.

## Slice 6 — Context Composer MVP

Allow selection of messages/ranges/materials/topology facts into a bundle. The bundle must show included, omitted, unavailable, stale, redacted, and over-budget items.

Acceptance evidence: copyable bundle with provenance manifest and deterministic archive object refs.

## Slice 7 — Workspaces

Polish stack and compare first. Add saved workspace state and broken-ref repair UI before timeline or cluster modes.

Acceptance evidence: stack parent/current/sidechain fixture, compare two attempts fixture, missing-target restore fixture.

## Slice 8 — Operations Room

Render daemon health/status/events/maintenance/source state as operator objects with bounded latency, stale/disconnected state, schema mismatch, catch-up/WAL/memory warnings, and actionable maintenance commands.

Acceptance evidence: synthetic health fixtures and endpoint budget tests.

## Slice 9 — Surface parity

Generate OpenAPI from typed payloads and align CLI/MCP/TUI surfaces for cost, provenance, topology, similar, errors, and list/search rows.

Acceptance evidence: parity tests and generated OpenAPI checked into docs.

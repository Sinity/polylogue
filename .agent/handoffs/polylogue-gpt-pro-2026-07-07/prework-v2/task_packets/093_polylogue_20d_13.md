# 093. polylogue-20d.13 — Daemon push channel: SSE events for live UIs instead of polling

Priority/type/status: **P2 / feature / open**. Lane: **08-scale-performance-live**. Release: **B/G-storage-live-performance**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

Everything live currently polls: the webui polls for facets/status, live tailing (bby.4) would poll, CLI watch modes would poll. The daemon already HAS the event stream internally (ingest events, convergence stage events, daemon_events.db) — it just has no push transport. One SSE endpoint turns the daemon from a request-answering server into a live substrate: session-ingested, session-updated, convergence-state-changed, cache-invalidated events pushed to subscribers.

## Existing design note

(1) TRANSPORT: Server-Sent Events over the existing HTTP server — chunked responses work on BaseHTTPRequestHandler (one long-lived thread per subscriber; cap subscribers, loopback-only), so this is NOT blocked on the ASGI decision (dx1) — though if dx1 migrates, SSE gets cheaper; note the thread cost in the dx1 evidence. (2) EVENT VOCABULARY (small, versioned): archive.cursor_moved {cursor, sessions_delta}, session.ingested {ref, origin}, session.updated {ref}, convergence.state {snapshot payload from 4bu}, cache.invalidated {scope}. Source from the existing daemon event plumbing — no new bus, expose the one that exists. (3) CONSUMERS, in value order: webui header/chips/list subscribe (kills facets polling; converging banner updates live; new sessions appear in the list as they land — the 'my session appeared while I watched' demo moment); live session tailing (bby.4) becomes SSE-driven message append; CLI --watch flags (find --watch, status --watch) consume the same endpoint via the fast-path client. (4) CONTRACT: events carry refs + cursors, never full payloads (subscribers fetch through the cache; keeps the channel cheap and the cache authoritative). Reconnect with Last-Event-ID resume from a bounded ring buffer.

## Acceptance criteria

A subscribed browser receives session.ingested within 2s of ingest commit on the seeded corpus. Reconnect with Last-Event-ID replays missed events from the ring buffer. The workbench converging banner updates without page reload. Subscriber cap enforced; endpoint loopback-only.

## Static mechanism / likely defect

Issue description localizes the mechanism: Everything live currently polls: the webui polls for facets/status, live tailing (bby.4) would poll, CLI watch modes would poll. The daemon already HAS the event stream internally (ingest events, convergence stage events, daemon_events.db) — it just has no push transport. One SSE endpoint turns the daemon from a request-answering server into a live substrate: session-ingested, session-updated, convergence-state-changed, cache-invalidated events pushed to subscribers. Design direction: (1) TRANSPORT: Server-Sent Events over the existing HTTP server — chunked responses work on BaseHTTPRequestHandler (one long-lived thread per subscriber; cap subscribers, loopback-only), so this is NOT blocked on the ASGI decision (dx1) — though if dx1 migrates, SSE gets cheaper; note the thread cost in the dx1 evidence. (2) EVENT VOCABULARY (small, versioned): archive.cursor_moved {cursor, sessions_delta}, session.…

## Source anchors to inspect first

- `CONTRIBUTING.md:102` — Derived-tier schema changes require rebuild/blue-green planning.
- `AGENTS.md:168` — Agent guidance says schema mismatch should rebuild or blue-green-replace derived tiers.
- `polylogue/cli/commands/reset.py` — Current reset/rebuild commands are the operator path to replace derived tiers.
- `polylogue/daemon/convergence_stages.py` — Daemon convergence/readiness state should represent generation progress honestly.
- `polylogue/archive/query/spec.py:440` — SessionQuerySpec is the core read selection object.
- `polylogue/archive/query/spec.py:498` — from_params centralizes parameter lowering.
- `polylogue/cli/query_contracts.py:70` — QueryOutputSpec is current output/render contract seam.
- `polylogue/cli/archive_query.py:164` — CLI archive query execution remains a dense orchestration point.
- `polylogue/daemon/http.py:1734` — Daemon split-archive route already has from_params path for some cases.
- `polylogue/daemon/http.py:3296` — Another daemon path builds SessionQuerySpec from raw query params.
- `polylogue/mcp/query_contracts.py:54` — MCP query normalization has a distinct lowering path.
- `polylogue/storage/sqlite/archive_tiers/index.py:296` — messages_fts table DDL lives in archive_tiers/index.py.
- `polylogue/storage/fts/sql.py:11` — messages_fts DDL copy also exists in storage/fts/sql.py.
- `polylogue/storage/fts/fts_lifecycle.py:292` — ensure_fts_triggers_sync owns runtime repair/recreation.
- `polylogue/storage/fts/fts_lifecycle.py:512` — Thread FTS rebuild logic is duplicated in lifecycle repair path.
- `polylogue/daemon/convergence_stages.py:988` — daemon convergence has another repair/readiness path for archive messages_fts.

## Implementation plan

1. (1) TRANSPORT: Server-Sent Events over the existing HTTP server — chunked responses work on BaseHTTPRequestHandler (one long-lived thread per subscriber
2. cap subscribers, loopback-only), so this is NOT blocked on the ASGI decision (dx1) — though if dx1 migrates, SSE gets cheaper
3. note the thread cost in the dx1 evidence.
4. (2) EVENT VOCABULARY (small, versioned): archive.cursor_moved {cursor, sessions_delta}, session.ingested {ref, origin}, session.updated {ref}, convergence.state {snapshot payload from 4bu}, cache.invalidated {scope}.
5. Source from the existing daemon event plumbing — no new bus, expose the one that exists.
6. (3) CONSUMERS, in value order: webui header/chips/list subscribe (kills facets polling
7. converging banner updates live

## Tests to add

- Acceptance proof: A subscribed browser receives session.ingested within 2s of ingest commit on the seeded corpus.
- Acceptance proof: Reconnect with Last-Event-ID replays missed events from the ring buffer.
- Acceptance proof: The workbench converging banner updates without page reload.
- Acceptance proof: Subscriber cap enforced
- Acceptance proof: endpoint loopback-only.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools verify --quick`

## Pitfalls

- Do not optimize by hiding degraded/partial state; status must get more truthful, not prettier.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.

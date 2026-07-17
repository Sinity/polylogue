# 092. polylogue-20d.12 — Daemon result cache + post-ingest warming: precomputed answers, cursor-keyed invalidation

Priority/type/status: **P2 / feature / open**. Lane: **08-scale-performance-live**. Release: **B/G-storage-live-performance**. Readiness: **blocked-hard**.

Hard blockers: polylogue-20d.1

## What the bead says

The fast path (20d.1) makes the daemon reachable in milliseconds; this bead makes the daemon WORTH reaching: today every facets/status/aggregate request recomputes from SQLite (live evidence: /api/facets defers repos+action_types by default, was stuck 'loading... stale' for minutes during convergence; bare status re-probes the DB per invocation). A hot daemon should answer the common 80% from memory: facets, status snapshot, recent-session lists, saved-view results, common aggregates — computed once per archive change, not once per request.

## Existing design note

(1) CACHE KEY: the archive ingest cursor (ops.db already tracks it) + query fingerprint. A cached entry is valid until the cursor moves — no TTL guessing, no staleness lies; the convergence snapshot (4bu) rides the same key. (2) WRITE-TRIGGERED RECOMPUTE: after each ingest batch commits, the daemon refreshes the hot set in its idle loop (facets complete families INCLUDING the deferred ones, status snapshot, newest-sessions page, saved views marked hot) — the webui then never waits on facets; it reads the precomputed payload. (3) COLD-START WARMING: after startup/rebuild/reset, a warming pass touches hot indexes and precomputes the hot set before first request (measured: first-query-after-rebuild pays cold page cache today); mmap profile (20d.11) compounds. (4) MEMORY BUDGET: hard cap (config, default ~64MB) with LRU eviction; /metrics exposes cache hit/miss/size so effectiveness is measurable, and the SLO lane asserts hit-rate on the seeded corpus. (5) SCOPE HONESTY: this is an in-daemon memo layer over the same SQL, NOT a second materialization tier — rows still come from index.db; eviction or restart costs latency, never correctness. Serve stale-while-revalidating only with the stale flag the payload already carries. Sequence: lands with/after 20d.1 so CLI + webui + MCP all hit the same cache.

## Acceptance criteria

bench slo (interactive tier): cached facets/status p50 <30ms on the seeded corpus with warm daemon. Cache entries invalidate within one ingest batch of a cursor move (test: ingest a session, facets reflect it next request). /metrics exposes cache hit/miss/size; memory stays under the configured cap under a 10k-query soak.

## Static mechanism / likely defect

Issue description localizes the mechanism: The fast path (20d.1) makes the daemon reachable in milliseconds; this bead makes the daemon WORTH reaching: today every facets/status/aggregate request recomputes from SQLite (live evidence: /api/facets defers repos+action_types by default, was stuck 'loading... stale' for minutes during convergence; bare status re-probes the DB per invocation). A hot daemon should answer the common 80% from memory: facets, status snapshot, recent-session lists, saved-view results, common aggregates — computed once per archive ch… Design direction: (1) CACHE KEY: the archive ingest cursor (ops.db already tracks it) + query fingerprint. A cached entry is valid until the cursor moves — no TTL guessing, no staleness lies; the convergence snapshot (4bu) rides the same key. (2) WRITE-TRIGGERED RECOMPUTE: after each ingest batch commits, the daemon refreshes the hot set in its idle loop (facets complete families INCLUDING the deferred ones, status snapshot, newest-s…

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

1. (1) CACHE KEY: the archive ingest cursor (ops.db already tracks it) + query fingerprint.
2. A cached entry is valid until the cursor moves — no TTL guessing, no staleness lies
3. the convergence snapshot (4bu) rides the same key.
4. (2) WRITE-TRIGGERED RECOMPUTE: after each ingest batch commits, the daemon refreshes the hot set in its idle loop (facets complete families INCLUDING the deferred ones, status snapshot, newest-sessions page, saved views marked hot) — the webui then never waits on facets
5. it reads the precomputed payload.
6. (3) COLD-START WARMING: after startup/rebuild/reset, a warming pass touches hot indexes and precomputes the hot set before first request (measured: first-query-after-rebuild pays cold page cache today)
7. mmap profile (20d.11) compounds.

## Tests to add

- Acceptance proof: bench slo (interactive tier): cached facets/status p50 <30ms on the seeded corpus with warm daemon.
- Acceptance proof: Cache entries invalidate within one ingest batch of a cursor move (test: ingest a session, facets reflect it next request).
- Acceptance proof: /metrics exposes cache hit/miss/size
- Acceptance proof: memory stays under the configured cap under a 10k-query soak.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools verify --quick`

## Pitfalls

- Do not optimize by hiding degraded/partial state; status must get more truthful, not prettier.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.

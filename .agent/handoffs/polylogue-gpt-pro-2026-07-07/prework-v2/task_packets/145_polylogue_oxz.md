# 145. polylogue-oxz — Performance instrumentation doctrine: slow-query log, phase timings, logging discipline

Priority/type/status: **P2 / feature / open**. Lane: **08-scale-performance-live**. Release: **B/G-storage-live-performance**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

Beyond spans, three instrumentation gaps and one doctrine gap: (a) no SLOW-QUERY LOG — SQLite statements over a threshold should be recorded with their text and, on demand, their EXPLAIN QUERY PLAN, or every perf regression starts from scratch (the EQP sweep 20d.7 is a snapshot; this is the continuous version); (b) CLI has no phase breakdown — 'polylogue --debug-timing find X' should print import/config/db-open/compile/execute/render wall per phase (the 1.6s floor was diagnosed by hand; it should be one flag); (c) webui has no client-side perf beacons (first-paint, fetch timings) — cheap to add, feeds bby.8 acceptance. Doctrine gap: logging is structlog-based but undisciplined — no stated level policy, no request-id correlation between HTTP and converger lines, unbounded daemon log files under nohup-style runs, print()-vs-logger inconsistency in CLI.

## Existing design note

(1) SLOW-QUERY LOG: sqlite3 set_trace_callback (or the profile hook) on both connection profiles; statements >threshold (default 50ms, y4c-tunable) log normalized SQL + duration + connection profile into ops.db (bounded ring, not unbounded rows); 'ops slow-queries' renders top-N with optional EQP capture on a copy. Overhead check: trace callbacks cost ~nothing when the threshold filter is in C-side profile hook — VERIFY per sqlite3 module semantics; if Python-side per-statement cost is measurable, gate behind a daemon flag default-on only for write profile. (2) CLI PHASE TIMING: monotonic checkpoints already implicit in the startup path — surface as --debug-timing (or POLYLOGUE_DEBUG_TIMING=1) printing the phase table to stderr; the spans from self-tracing reuse the same checkpoints when the daemon serves the query. (3) LOG DOCTRINE (one page in internals): level semantics (info = state transitions an operator cares about; debug = per-item; warning = degraded-but-serving; error = failed request/effect), every daemon log line carries request-id/session-ref when in scope, journald is the sink under systemd (no bespoke rotation), CLI human output goes to stdout via renderers while diagnostics go to stderr as structlog (never print() for diagnostics — lint it like the clock-hygiene pattern). (4) WEBUI BEACONS: navigator.sendBeacon of paint/fetch timings to a daemon endpoint -> ops.db, sampled; bby.8's acceptance reads them.

## Acceptance criteria

Slow-query log captures a seeded slow statement with duration + normalized SQL and bounded storage; --debug-timing prints the phase table and matches span data for daemon-served queries; log-doctrine page committed + print()-diagnostic lint wired into verify quick; webui beacons land in ops.db on the seeded workbench.

## Static mechanism / likely defect

Issue description localizes the mechanism: Beyond spans, three instrumentation gaps and one doctrine gap: (a) no SLOW-QUERY LOG — SQLite statements over a threshold should be recorded with their text and, on demand, their EXPLAIN QUERY PLAN, or every perf regression starts from scratch (the EQP sweep 20d.7 is a snapshot; this is the continuous version); (b) CLI has no phase breakdown — 'polylogue --debug-timing find X' should print import/config/db-open/compile/execute/render wall per phase (the 1.6s floor was diagnosed by hand; it should be one flag); (c)… Design direction: (1) SLOW-QUERY LOG: sqlite3 set_trace_callback (or the profile hook) on both connection profiles; statements >threshold (default 50ms, y4c-tunable) log normalized SQL + duration + connection profile into ops.db (bounded ring, not unbounded rows); 'ops slow-queries' renders top-N with optional EQP capture on a copy. Overhead check: trace callbacks cost ~nothing when the threshold filter is in C-side profile hook — VE…

## Source anchors to inspect first

- `CONTRIBUTING.md:102` — Derived-tier schema changes require rebuild/blue-green planning.
- `AGENTS.md:168` — Agent guidance says schema mismatch should rebuild or blue-green-replace derived tiers.
- `polylogue/cli/commands/reset.py` — Current reset/rebuild commands are the operator path to replace derived tiers.
- `polylogue/daemon/convergence_stages.py` — Daemon convergence/readiness state should represent generation progress honestly.
- `polylogue/storage/sqlite/archive_tiers/index.py:296` — messages_fts table DDL lives in archive_tiers/index.py.
- `polylogue/storage/fts/sql.py:11` — messages_fts DDL copy also exists in storage/fts/sql.py.
- `polylogue/storage/fts/fts_lifecycle.py:292` — ensure_fts_triggers_sync owns runtime repair/recreation.
- `polylogue/storage/fts/fts_lifecycle.py:512` — Thread FTS rebuild logic is duplicated in lifecycle repair path.
- `polylogue/daemon/convergence_stages.py:988` — daemon convergence has another repair/readiness path for archive messages_fts.
- `polylogue/archive/query/spec.py:440` — SessionQuerySpec is the core read selection object.
- `polylogue/archive/query/spec.py:498` — from_params centralizes parameter lowering.
- `polylogue/cli/query_contracts.py:70` — QueryOutputSpec is current output/render contract seam.
- `polylogue/cli/archive_query.py:164` — CLI archive query execution remains a dense orchestration point.
- `polylogue/daemon/http.py:1734` — Daemon split-archive route already has from_params path for some cases.
- `polylogue/daemon/http.py:3296` — Another daemon path builds SessionQuerySpec from raw query params.
- `polylogue/mcp/query_contracts.py:54` — MCP query normalization has a distinct lowering path.

## Implementation plan

1. (1) SLOW-QUERY LOG: sqlite3 set_trace_callback (or the profile hook) on both connection profiles
2. statements >threshold (default 50ms, y4c-tunable) log normalized SQL + duration + connection profile into ops.db (bounded ring, not unbounded rows)
3. 'ops slow-queries' renders top-N with optional EQP capture on a copy.
4. Overhead check: trace callbacks cost ~nothing when the threshold filter is in C-side profile hook — VERIFY per sqlite3 module semantics
5. if Python-side per-statement cost is measurable, gate behind a daemon flag default-on only for write profile.
6. (2) CLI PHASE TIMING: monotonic checkpoints already implicit in the startup path — surface as --debug-timing (or POLYLOGUE_DEBUG_TIMING=1) printing the phase table to stderr
7. the spans from self-tracing reuse the same checkpoints when the daemon serves the query.

## Tests to add

- Acceptance proof: Slow-query log captures a seeded slow statement with duration + normalized SQL and bounded storage
- Acceptance proof: debug-timing prints the phase table and matches span data for daemon-served queries
- Acceptance proof: log-doctrine page committed + print()-diagnostic lint wired into verify quick
- Acceptance proof: webui beacons land in ops.db on the seeded workbench.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools verify --quick`

## Pitfalls

- Do not optimize by hiding degraded/partial state; status must get more truthful, not prettier.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.

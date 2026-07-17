# 144. polylogue-opc — Self-tracing: the daemon's own spans land in its own archive

Priority/type/status: **P2 / feature / open**. Lane: **08-scale-performance-live**. Release: **B/G-storage-live-performance**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

Polylogue has an OTLP receiver and stores spans — and instruments itself with none. Self-tracing closes the loop: daemon HTTP requests, converger stage executions, query compile+execute phases, ingest attempts, cache hits/misses, and embedding drain windows emit spans through the daemon's own OTLP intake into ops.db — making 'why was that slow' a query against the archive instead of a log-reading session. Dogfood value doubles as demo value: the tool debugging itself with its own forensics is the most polylogue-shaped exhibit possible.

## Existing design note

(1) A tiny internal tracer (no opentelemetry-sdk dependency — spans are dicts posted to the in-process intake; the OTLP wire format only matters for external emitters): span(name, attrs) context manager instrumenting: HTTP route handlers (route, status, duration), converger stages (per session/batch), archive_query compile vs execute vs render phases, write_effects (per effect once 0aj lands), embed windows, cache lookups (20d.12). Request-id correlation: HTTP handler opens the root span; downstream spans parent to it. (2) Sampling doctrine: routes/stages always-on (cheap, bounded); per-query phase spans sampled or threshold-gated (only when total >50ms) to avoid self-flooding; hard cap on spans/minute with drop counter. (3) Storage: existing ops.db span tables (disposable tier); retention pruning by age/count in the periodic loop. (4) Read surfaces: 'polylogue ops traces --slow' (top spans by duration, tree render); the latency projection (20d.14) reads span aggregates; webui gets a slow-requests panel later. (5) Explicit relation: 20d.14 histograms answer 'how slow is route X overall'; spans answer 'why was THIS request slow' — metrics for trends, traces for forensics, same substrate.

## Acceptance criteria

Spans emitted for routes/stages/query-phases on the seeded corpus daemon; request-id ties a route span to its query-phase children; sampling caps enforced with drop counters visible in /metrics; ops traces --slow renders a span tree for a real slow request; retention pruning works.

## Static mechanism / likely defect

Issue description localizes the mechanism: Polylogue has an OTLP receiver and stores spans — and instruments itself with none. Self-tracing closes the loop: daemon HTTP requests, converger stage executions, query compile+execute phases, ingest attempts, cache hits/misses, and embedding drain windows emit spans through the daemon's own OTLP intake into ops.db — making 'why was that slow' a query against the archive instead of a log-reading session. Dogfood value doubles as demo value: the tool debugging itself with its own forensics is the most polylogue-sh… Design direction: (1) A tiny internal tracer (no opentelemetry-sdk dependency — spans are dicts posted to the in-process intake; the OTLP wire format only matters for external emitters): span(name, attrs) context manager instrumenting: HTTP route handlers (route, status, duration), converger stages (per session/batch), archive_query compile vs execute vs render phases, write_effects (per effect once 0aj lands), embed windows, cache l…

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

1. (1) A tiny internal tracer (no opentelemetry-sdk dependency — spans are dicts posted to the in-process intake
2. the OTLP wire format only matters for external emitters): span(name, attrs) context manager instrumenting: HTTP route handlers (route, status, duration), converger stages (per session/batch), archive_query compile vs execute vs render phases, write_effects (per effect once 0aj lands), embed windows, cache lookups (20d.12).
3. Request-id correlation: HTTP handler opens the root span
4. downstream spans parent to it.
5. (2) Sampling doctrine: routes/stages always-on (cheap, bounded)
6. per-query phase spans sampled or threshold-gated (only when total >50ms) to avoid self-flooding
7. hard cap on spans/minute with drop counter.

## Tests to add

- Acceptance proof: Spans emitted for routes/stages/query-phases on the seeded corpus daemon
- Acceptance proof: request-id ties a route span to its query-phase children
- Acceptance proof: sampling caps enforced with drop counters visible in /metrics
- Acceptance proof: ops traces --slow renders a span tree for a real slow request
- Acceptance proof: retention pruning works.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools verify --quick`

## Pitfalls

- Do not optimize by hiding degraded/partial state; status must get more truthful, not prettier.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.

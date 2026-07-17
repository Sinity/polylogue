# 091. polylogue-20d.14 — Interactive SLO tier: named latency budgets, continuously measured, regression-gated

Priority/type/status: **P2 / task / open**. Lane: **08-scale-performance-live**. Release: **B/G-storage-live-performance**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

'Snappy' needs numbers or it regresses silently. docs/plans/slo-catalog.yaml + devtools bench slo exist but there is no interactive tier: no stated budget for daemon-served query round trip, completion latency, webui first-paint, cold CLI floor, or ingest-to-searchable lag — and no continuous measurement, so today's evidence (1.6-1.9s CLI floor for EVERYTHING including error paths, 5-9s helps, minutes-stuck facets) had to be discovered by ad-hoc probing rather than read off a dashboard.

## Existing design note

(1) BUDGETS (starting points, tune with evidence): daemon-served query p50<100ms/p95<400ms; completion round trip <50ms (it is keystroke-path); status/facets from cache <30ms; webui first meaningful paint <300ms warm daemon; cold CLI (no daemon) <700ms after 20d.2 import deferral; ingest-to-searchable <10s from JSONL write (measures the whole hook->ingest->FTS->cache-invalidate chain). Add as an 'interactive' tier in slo-catalog.yaml. (2) MEASUREMENT, two legs: (a) bench slo runs the tier against the seeded corpus with a live daemon — CI-runnable regression gate; (b) LIVE TELEMETRY: daemon records per-route latency histograms into /metrics (buckets, not averages) and the fast-path CLI records per-invocation spans to ops.db (verb, wall, daemon-vs-local route) — disposable telemetry tier, so schema-carve-out applies. (3) CONSUMPTION: Sessions: 0
Messages: 0
Attachments: 0
Origins: 0
Average messages: 0.0 gains a latency projection (the archive analyzing its own snappiness — dogfood + demo material); devtools verify does NOT gate on live numbers (host-dependent), only bench slo on seeded corpus gates. (4) The budgets become the acceptance criteria for the sibling beads: fast path, cache, push, import deferral all cite this tier instead of inventing their own targets.

## Acceptance criteria

slo-catalog.yaml contains the interactive tier with the stated budgets. devtools bench slo runs the tier green against the seeded corpus with a live daemon in CI. /metrics exposes per-route latency histograms. CLI invocation spans are queryable in ops.db and surfaced by a polylogue analyze latency projection.

## Static mechanism / likely defect

Issue description localizes the mechanism: 'Snappy' needs numbers or it regresses silently. docs/plans/slo-catalog.yaml + devtools bench slo exist but there is no interactive tier: no stated budget for daemon-served query round trip, completion latency, webui first-paint, cold CLI floor, or ingest-to-searchable lag — and no continuous measurement, so today's evidence (1.6-1.9s CLI floor for EVERYTHING including error paths, 5-9s helps, minutes-stuck facets) had to be discovered by ad-hoc probing rather than read off a dashboard. Design direction: (1) BUDGETS (starting points, tune with evidence): daemon-served query p50<100ms/p95<400ms; completion round trip <50ms (it is keystroke-path); status/facets from cache <30ms; webui first meaningful paint <300ms warm daemon; cold CLI (no daemon) <700ms after 20d.2 import deferral; ingest-to-searchable <10s from JSONL write (measures the whole hook->ingest->FTS->cache-invalidate chain). Add as an 'interactive' tier i…

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

1. (1) BUDGETS (starting points, tune with evidence): daemon-served query p50<100ms/p95<400ms
2. completion round trip <50ms (it is keystroke-path)
3. status/facets from cache <30ms
4. webui first meaningful paint <300ms warm daemon
5. cold CLI (no daemon) <700ms after 20d.2 import deferral
6. ingest-to-searchable <10s from JSONL write (measures the whole hook->ingest->FTS->cache-invalidate chain).
7. Add as an 'interactive' tier in slo-catalog.yaml.

## Tests to add

- Acceptance proof: slo-catalog.yaml contains the interactive tier with the stated budgets.
- Acceptance proof: devtools bench slo runs the tier green against the seeded corpus with a live daemon in CI.
- Acceptance proof: /metrics exposes per-route latency histograms.
- Acceptance proof: CLI invocation spans are queryable in ops.db and surfaced by a polylogue analyze latency projection.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools verify --quick`

## Pitfalls

- Do not optimize by hiding degraded/partial state; status must get more truthful, not prettier.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.

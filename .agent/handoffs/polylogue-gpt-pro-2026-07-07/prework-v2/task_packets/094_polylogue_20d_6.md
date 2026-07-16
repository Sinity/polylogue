# 094. polylogue-20d.6 — Live full-ingest catch-up latency + WAL shape

Priority/type/status: **P2 / task / open**. Lane: **08-scale-performance-live**. Release: **B/G-storage-live-performance**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

0.2 files/s full-ingest chunks; parse_s ~274s for 50 small files. Recent daemon backoff commits (no-op retry/catch-up chunks, filtered retry paths) address parts — re-measure before working. GH issue thread (body + comments) is input, not authority; this bead's scope statement wins where they conflict.

## Existing design note

Live evidence (gh#2391): full-ingest chunks ~0.2 files/s; 50 small files -> parse_s ~274s while convergence <2s; WAL ballooned during a 50-file chunk. Recent daemon backoff commits changed the shape — RE-MEASURE first (bounded catch-up + stage timings + ops diagnostics workload before/after). Related invariant to keep verified: full-replace re-ingest rewrites all messages in one transaction (correct for idempotency) — for live-tailed long sessions the append path (sources/live/append_ingest.py) must stay the hot route; run devtools bench ingest-amplification on real tails as a scheduled check, since append-vs-full-replace regressions multiply WAL churn. Suspects if still slow after re-measure: per-file parse overhead, per-file commit cadence, prepare-cache misses.

## Acceptance criteria

- RE-MEASURE first (recent daemon backoff commits changed the shape): bounded catch-up run + stage timings + `polylogue ops diagnostics workload` before/after are captured and the baseline recorded.
- The idempotency invariant is kept verified — full-replace re-ingest rewrites all messages in one transaction — while for live-tailed long sessions the append path (sources/live/append_ingest.py) stays the hot route; `devtools bench ingest-amplification` on real tails is wired as a scheduled check to catch append-vs-full-replace regressions.
- End-to-end ingest-to-searchable latency is measured with a synthetic session write on the seeded corpus (chain: hook/watcher debounce -> parse -> store -> FTS -> cache invalidation (20d.12) -> SSE announce (20d.13)); a session appears in find/webui within the ~10s interactive SLO budget (20d.14).
- If still slow after re-measure, the named suspects (per-file parse overhead, per-file commit cadence, prepare-cache misses) are investigated with evidence; the fix is verified by re-running the timing matrix and `devtools bench ingest-throughput`.

## Static mechanism / likely defect

Issue description localizes the mechanism: 0.2 files/s full-ingest chunks; parse_s ~274s for 50 small files. Recent daemon backoff commits (no-op retry/catch-up chunks, filtered retry paths) address parts — re-measure before working. GH issue thread (body + comments) is input, not authority; this bead's scope statement wins where they conflict. Design direction: Live evidence (gh#2391): full-ingest chunks ~0.2 files/s; 50 small files -> parse_s ~274s while convergence <2s; WAL ballooned during a 50-file chunk. Recent daemon backoff commits changed the shape — RE-MEASURE first (bounded catch-up + stage timings + ops diagnostics workload before/after). Related invariant to keep verified: full-replace re-ingest rewrites all messages in one transaction (correct for idempotency)…

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

1. Live evidence (gh#2391): full-ingest chunks ~0.2 files/s
2. 50 small files -> parse_s ~274s while convergence <2s
3. WAL ballooned during a 50-file chunk.
4. Recent daemon backoff commits changed the shape — RE-MEASURE first (bounded catch-up + stage timings + ops diagnostics workload before/after).
5. Related invariant to keep verified: full-replace re-ingest rewrites all messages in one transaction (correct for idempotency) — for live-tailed long sessions the append path (sources/live/append_ingest.py) must stay the hot route
6. run devtools bench ingest-amplification on real tails as a scheduled check, since append-vs-full-replace regressions multiply WAL churn.
7. Suspects if still slow after re-measure: per-file parse overhead, per-file commit cadence, prepare-cache misses.

## Tests to add

- Acceptance proof: RE-MEASURE first (recent daemon backoff commits changed the shape): bounded catch-up run + stage timings + `polylogue ops diagnostics workload` before/after are captured and the baseline recorded.
- Acceptance proof: The idempotency invariant is kept verified — full-replace re-ingest rewrites all messages in one transaction — while for live-tailed long sessions the append path (sources/live/append_ingest.py) stays the hot route
- Acceptance proof: `devtools bench ingest-amplification` on real tails is wired as a scheduled check to catch append-vs-full-replace regressions.
- Acceptance proof: End-to-end ingest-to-searchable latency is measured with a synthetic session write on the seeded corpus (chain: hook/watcher debounce -> parse -> store -> FTS -> cache invalidation (20d.12) -> SSE announce (20d.13))
- Acceptance proof: a session appears in find/webui within the ~10s interactive SLO budget (20d.14).
- Acceptance proof: If still slow after re-measure, the named suspects (per-file parse overhead, per-file commit cadence, prepare-cache misses) are investigated with evidence
- Acceptance proof: the fix is verified by re-running the timing matrix and `devtools bench ingest-throughput`.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools verify --quick`

## Pitfalls

- Do not optimize by hiding degraded/partial state; status must get more truthful, not prettier.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.

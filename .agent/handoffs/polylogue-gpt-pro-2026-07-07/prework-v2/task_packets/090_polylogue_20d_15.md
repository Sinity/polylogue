# 090. polylogue-20d.15 — Bulk ingest throughput + resource envelope: parallel parse, batched writes, bounded RSS/IO

Priority/type/status: **P2 / task / open**. Lane: **08-scale-performance-live**. Release: **B/G-storage-live-performance**. Readiness: **blocked-hard**.

Hard blockers: polylogue-20d.14

## What the bead says

Live evidence 2026-07-03: the full index rebuild replayed 16,725 raw rows at 12-15 rows/s whole-run (5/s when it hit big sessions) — 20-40 minutes of archive downtime for an operation the fresh-first doctrine treats as routine. Nobody has stated the machine impact budget either: daemon RSS during bulk ingest, write amplification per tier, page-cache pressure, and IO contention with the live desktop are unmeasured-in-anger even though the instruments exist (live_ingest_attempt RSS fields, bench ingest-amplification, bench ingest-throughput). 20d.6 owns the LIVE catch-up lane (single-session ingest-to-searchable); this bead owns the BULK lane: replays, resets, backfills.

## Existing design note

(1) MEASURE first on a live-archive copy: where do the 12-15 rows/s go (parse vs store vs FTS vs insights — the attempt rows record stage timings); bench ingest-throughput gives the synthetic baseline. (2) PARALLEL PARSE: parsing is CPU-bound JSON; pipeline/services/process_pool.py already provides the safe pool (spawn-context) — fan out parse across N workers, keep the store single-writer (SQLite reality); the parallel-parse dogfood branch from 2026-06-29 is prior art to consult. Expect the write leg to become the bottleneck: batch multi-session transactions (amortize fsync; measure against WAL autocheckpoint interplay per 20d.6), suspend per-row FTS in favor of the existing bulk trigger-drop path, defer insight materialization to a second pass (the daemon already stages fts/embed/insights separately — make bulk replay exploit it). Target: >=100 rows/s whole-run on the operator machine, rebuild <5 min — stated in the SLO catalog as a maintenance-tier budget (20d.14). (3) RESOURCE ENVELOPE: cap ingest RSS (bounded batch size + streaming lowering already exists for multi-GiB files — verify it holds in bulk mode); write amplification per tier via bench ingest-amplification before/after; IO: run bulk lanes with ionice-idle/self-throttle so a rebuild never makes the desktop stutter (the daemon can set its own IO class; do not rely on the operator remembering systemd slices). (4) REPORT: rebuild prints rows/s + ETA continuously (the devloop agent hand-computed ETA from logs today — the daemon should just say it; feeds the 4bu convergence snapshot).

## Acceptance criteria

Full replay of a live-archive copy sustains >=100 raw rows/s whole-run on the operator machine and finishes <5 min; rebuild prints live rows/s and ETA. Ingest RSS stays under the stated cap; bench ingest-amplification shows no per-tier regression; desktop remains responsive during a rebuild (idle IO class verified).

## Static mechanism / likely defect

Issue description localizes the mechanism: Live evidence 2026-07-03: the full index rebuild replayed 16,725 raw rows at 12-15 rows/s whole-run (5/s when it hit big sessions) — 20-40 minutes of archive downtime for an operation the fresh-first doctrine treats as routine. Nobody has stated the machine impact budget either: daemon RSS during bulk ingest, write amplification per tier, page-cache pressure, and IO contention with the live desktop are unmeasured-in-anger even though the instruments exist (live_ingest_attempt RSS fields, bench ingest-amplification… Design direction: (1) MEASURE first on a live-archive copy: where do the 12-15 rows/s go (parse vs store vs FTS vs insights — the attempt rows record stage timings); bench ingest-throughput gives the synthetic baseline. (2) PARALLEL PARSE: parsing is CPU-bound JSON; pipeline/services/process_pool.py already provides the safe pool (spawn-context) — fan out parse across N workers, keep the store single-writer (SQLite reality); the para…

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

1. (1) MEASURE first on a live-archive copy: where do the 12-15 rows/s go (parse vs store vs FTS vs insights — the attempt rows record stage timings)
2. bench ingest-throughput gives the synthetic baseline.
3. (2) PARALLEL PARSE: parsing is CPU-bound JSON
4. pipeline/services/process_pool.py already provides the safe pool (spawn-context) — fan out parse across N workers, keep the store single-writer (SQLite reality)
5. the parallel-parse dogfood branch from 2026-06-29 is prior art to consult.
6. Expect the write leg to become the bottleneck: batch multi-session transactions (amortize fsync
7. measure against WAL autocheckpoint interplay per 20d.6), suspend per-row FTS in favor of the existing bulk trigger-drop path, defer insight materialization to a second pass (the daemon already stages fts/embed/insights separately — make bulk replay exploit it).

## Tests to add

- Acceptance proof: Full replay of a live-archive copy sustains >=100 raw rows/s whole-run on the operator machine and finishes <5 min
- Acceptance proof: rebuild prints live rows/s and ETA.
- Acceptance proof: Ingest RSS stays under the stated cap
- Acceptance proof: bench ingest-amplification shows no per-tier regression
- Acceptance proof: desktop remains responsive during a rebuild (idle IO class verified).

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools verify --quick`

## Pitfalls

- Do not optimize by hiding degraded/partial state; status must get more truthful, not prettier.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.

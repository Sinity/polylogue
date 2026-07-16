---
created: "2026-06-28"
purpose: "Re-ingestion snappiness dogfood measurement plan for the 37GB/16K-session archive"
status: "active"
project: "polylogue"
related: "#2391 (Optimize live full-ingest catch-up latency and WAL shape)"
---

# Ingestion-Snappiness Dogfood Measurement Plan

## Current Baseline Correction

2026-06-30: this plan was written against the old pre-dedup 37GB/16K-session
archive shape. That is not the current live baseline. Current active archive:
2,397 raw rows, 2,390 indexed sessions, and 159,956 indexed messages. Keep the
old figures only as historical stress-test context; use the current
deduplicated baseline for new ingest/catch-up estimates.

## Context

A pricing/schema change forces `polylogue ops reset --database && polylogued run`,
which re-ingests the full archive from source. The operator's PERF focus is
making that re-ingest **snappy** —
good throughput, bounded RAM, bounded IO. This plan dogfoods (measures) the
ingest path. It is read-only design + light probing; **no production changes**.

Ties directly to issue **#2391** ("Optimize live full-ingest catch-up latency and
WAL shape"). #2391 is ~45% in code: AC2 (per-chunk stage timings) satisfied,
AC3 partially (append path ~10x faster; large-Codex `full.index.full_replace`
residual), AC1 (multi-provider small-file throughput benchmark) + AC4 (WAL
ceiling assertion) + AC5 (shutdown RSS release verification) still open.

## 1. The ingest path: stages and per-stage work

Stage vocabulary (`polylogue/pipeline/ingest_support.py`):
`INGEST_STAGE_SEQUENCES`, full path = `acquire → parse → materialize → index`
(+ `embed` opt-in). Leaf stages: acquire, parse, materialize, index, embed,
schema. `reprocess = parse+materialize+index`.

Core orchestration: `polylogue/pipeline/services/ingest_batch/_core.py` (1242
lines, the largest pipeline file). Timed sub-phases instrumented with
`time.perf_counter()` and accumulated onto a summary object
(`_models.py:BatchSummary`, field `stage_timings_s: dict[str, float]`):
- per-batch: `setup_elapsed_s`, `drain_elapsed_s`, `result_wait_s`,
  `teardown_elapsed_s`, `flush_elapsed_s`, `commit_elapsed_s`, `elapsed_s`.
- per-session insight sub-timings (`_core.py` ~1158-1196): updates, threads,
  aggregates.

Index write phases (`polylogue/storage/sqlite/archive_tiers/write.py`, helper
`add_timing(name, t0)`):
- coarse: `index.prepare`, `index.session_upsert`, `index.merge_prepare`,
  `index.full_replace`, `index.messages`, `index.blocks`, `index.web_constructs`,
  `index.attachments`, `index.paste_spans`, `index.parent_links`,
  `index.session_link`, `index.session_events`, `index.working_dirs`,
  `index.repo_edges`, `index.reported_costs`, `index.provider_usage_rollup`,
  `index.session_counts`, `index.graph_resolve`, `index.ingest_flags`.
- **`index.full_replace.*` sub-phases (the #2391 residual hotspot, write.py
  ~1514-1576):** `fts_probe`, `fts_delete`, `fts_suspend`,
  `clear_projection_rows`, `delete_messages`, `messages`, `blocks`,
  `web_constructs`, `fts_insert`, `fts_restore`. **This FTS suspend/insert/
  restore + message/block bulk rewrite on a full-replace is the dominant cost
  for large Codex files.**

Convergence stages (post-ingest, `polylogue/daemon/convergence_stages.py`,
`make_default_convergence_stages`): FTS repair → embed → insights. Each has
`check`/`execute` + `_many`/`_sessions` batch variants. Insights stage guards
against "hot" multi-GB source paths to avoid re-materialization storms
(`_source_path_is_hot_for_insights`, `_hot_insight_session_ids`).

## 2. Existing measurement surfaces (what's already capturable)

### A. `live_ingest_attempt` / `ingest_attempts` rows (the richest signal)
Model: `polylogue/daemon/live_ingest_attempt_models.py:LiveIngestAttemptState`.
Durable per-attempt fields already captured:
- **Throughput:** `files_per_second`, `source_mb_per_second`,
  `queued/needed/succeeded/failed_file_count`, `parse_time_s`,
  `convergence_time_s`, `total_time_s`, `stage_timings_s: dict[str,float]`.
- **IO/amplification:** `input_bytes`, `source_payload_read_bytes`,
  `cursor_fingerprint_read_bytes`, `total_read_bytes`, `read_amplification`,
  `archive_write_bytes_delta`, `written_raw_count`.
- **RAM:** `rss_current_mb`, `rss_peak_self_mb`, `rss_peak_children_mb`,
  full cgroup v2 set (`cgroup_memory_current/peak/swap/anon/file/inactive_file_mb`,
  `cgroup_path`).
- **Workers:** `worker_in_flight/completed/total_count`.

Durable storage: `ops.db` tables `ingest_attempts` (cols: attempt_id,
source_path, origin, status, phase, started/heartbeat/finished_at_ms,
parsed_raw_count, materialized_count, error_message, source_paths_json) and
`daemon_stage_events` (event_id, attempt_id, stage, status, observed_at_ms,
payload_json).

### B. `polylogue ops diagnostics workload` (read-only snapshot)
Impl: `devtools/daemon_workload_probe.py` (REPORT_VERSION=13). Reads ops.db /
index.db / source.db directly (no daemon IPC). Sections relevant here:
- `attempt_counts`, `recent_attempts` (per-attempt amplification + parse/
  convergence timings + source bundles).
- `convergence_stage_timings` — min/max/sum/mean over completed attempts for
  `parse_time_s`, `convergence_time_s`, `read_amplification`, **plus `per_stage_s`
  aggregated from `stage_timings_s` / `daemon_stage_events.payload_json`** (this
  is the per-phase rollup to diff before/after).
- `boundary_table_counts` (planner estimates; `--exact-table-counts` for exact),
  `archive_tiers` (per-tier file size + user_version + counts),
  `blob_lease_state`, `gc_state`, `fts_trigger_state`, `daemon_resource_signal`
  (RSS/cgroup from latest attempt), `source_path_churn`, `convergence_debt`,
  `query_plans`.
- **Compare mode:** `--compare before.json after.json` produces
  `{before,after,delta}` triples; refuses incompatible report_version.

### C. `devtools bench ingest-amplification` (deterministic byte attribution)
Impl: `devtools/ingest_amplification_probe.py` (REPORT_VERSION=1; registered in
`devtools/command_catalog.py`). Drives `parse_sources_archive` over a synthetic
deterministic corpus and attributes **bytes per tier** (source/index/embeddings/
user/ops .db via `PRAGMA page_count*page_size` + on-disk `-wal` size) **plus the
blob store**. Host-independent (byte counts + ratios, no wallclock). Flags:
`--batches`, `--seed`, `--provider`, `--json`. **Gap: single-provider, append-
shaped; no multi-provider small-file or large-Codex-full case (#2391 AC1).**

### D. Bench tests
`tests/benchmarks/`: `test_pipeline.py`, `test_daemon_convergence*.py`,
`test_fts_trigger_amplification.py`, `test_concurrent_throughput.py`,
`test_schema_linear_scaling.py`, `baselines/`. **Gap: no asserted WAL-growth
ceiling test (#2391 AC4); no small-file multi-provider throughput baseline.**

### E. Shutdown RSS (AC5)
`polylogue/core/memory.py:release_process_memory()` (gc.collect + libc
malloc_trim). `daemon/cli.py`: `_periodic_wal_checkpoint` (5 min TRUNCATE),
`_mark_interrupted_live_ingest_attempts_on_shutdown`. Capturable but **no
automated post-interrupt RSS assertion**.

## 3. Live evidence (cheap probe, 2026-06-28) — confirms #2391

Production archive: `~/.local/share/polylogue/index.db` = **37 GB** (env
currently points at the empty `/tmp/polylogue-archive` cloud sandbox; the real
tiers live under `~/.local/share/polylogue/` and a `/realm/db/polylogue/` copy
with index.db 20 GB, ops.db 369 MB).

Read-only from `/realm/db/polylogue/ops.db` (`ingest_attempts`):
- **Small-file throughput: 0.11–1.35 files/s** across recent completed
  attempts — matches #2391's "~0.2 files/s for KB-MB files" symptom exactly.
- **Large-Codex catastrophe (top durations):** single attempts of
  **2233 s (37 min, 3 raw), 1577 s, 1493 s, 1424 s, 1413 s** — these are the
  giant Codex session files (189B-style append-heavy). This is the
  `full.index.full_replace` FTS/messages/blocks residual called out in #2391
  AC3 remaining work, and it is the dominant re-ingest tax, not the small-file
  case.
- `daemon_stage_events.payload_json` sample carried the RSS/cgroup/IO/parse
  fields but **`stage_timings_s` was empty** in these older rows — per-stage
  durable persistence may post-date these attempts or land in a separate
  `stage_timings_json` column; **verify per-stage timings are actually persisted
  on a fresh run before relying on `per_stage_s` for the before/after diff.**

## 4. Re-ingestion dogfood measurement plan

### Goal metrics
1. **Throughput:** sessions/min and messages/min, overall + per origin
   (chatgpt / claude-code / claude-ai / codex / gemini). Codex segregated —
   it dominates.
2. **Per-stage attribution:** seconds in acquire/parse/materialize/index, and
   within index the `full_replace.*` FTS sub-phases.
3. **Peak RAM:** daemon `rss_peak_self_mb` + `rss_peak_children_mb` + cgroup
   `memory_peak_mb` over the whole run.
4. **WAL growth:** max `index.db-wal` size during a chunk + post-checkpoint
   floor (verify the 160 MB `WAL_JOURNAL_SIZE_LIMIT_BYTES` ceiling actually
   bounds it; #2391 observed 5.9 GiB transient).
5. **IO / write amplification:** `archive_write_bytes_delta`,
   `read_amplification`, bytes-per-payload-byte per tier.

### Procedure (a future session runs; do NOT run heavy now)
This is a multi-hour run against the 37 GB corpus — schedule deliberately,
under `sinnix-scope background`, with the env pointed at a scratch archive on
NVMe (not `/tmp`, which is wear-limited root).

```bash
# 0. Point at a fresh scratch archive on NVMe (NOT /tmp).
export POLYLOGUE_ARCHIVE_ROOT=/realm/tmp/polylogue-reingest-bench
mkdir -p "$POLYLOGUE_ARCHIVE_ROOT"

# 1. BEFORE snapshot (empty archive baseline).
polylogue ops maintenance archive-init --yes
polylogue ops diagnostics workload --json --exact-table-counts > before.json

# 2. Background WAL sampler (cheap; 5s cadence) to catch transient WAL peaks
#    that checkpoint hides from the after-snapshot.
( while sleep 5; do
    stat -c '%n %s %Y' "$POLYLOGUE_ARCHIVE_ROOT"/index.db-wal 2>/dev/null;
  done ) >> wal-samples.log &

# 3. Full re-ingest under the background slice.
sinnix-scope background -- polylogued run --no-api --no-browser-capture

# 4. AFTER snapshot + arithmetic diff.
polylogue ops diagnostics workload --json --exact-table-counts > after.json
polylogue ops diagnostics workload --compare before.json after.json --json > diff.json
```

### Deriving the numbers
- **Throughput:** from `diff.json` boundary_table_counts delta on `sessions` /
  `messages` ÷ wall-clock span (max-min `finished_at_ms` in `ingest_attempts`).
  Per-origin throughput from a direct ops.db query grouping `ingest_attempts`
  by `origin` with `parsed_raw_count / ((finished-started)/1000)`.
- **Per-stage:** `convergence_stage_timings.per_stage_s` in after.json (verify
  `stage_timings_s` is populated first — §3 gap). Cross-check with
  `daemon_stage_events` payloads.
- **Peak RAM:** `max(rss_peak_self_mb + rss_peak_children_mb)` and
  `max(cgroup_memory_peak_mb)` over `ingest_attempts` payloads; plus
  `daemon_resource_signal` in after.json.
- **WAL:** `max` column 2 of `wal-samples.log`; compare to 160 MB ceiling.
- **IO/amplification:** `sum(archive_write_bytes_delta)`, percentile
  `read_amplification` from `convergence_stage_timings.read_amplification`.

### Cheap deterministic companion (host-independent, no full corpus)
```bash
devtools bench ingest-amplification --json --batches 8 --seed 1851 > amp-baseline.json
```
Run before/after any write-path change to detect per-tier byte regressions
without the multi-hour corpus run.

## 5. Instrumentation gaps to close (drive #2391 to done)

1. **AC1 — multi-provider small-file + large-Codex throughput benchmark.**
   `ingest_amplification_probe.py` is single-provider/append-only. Add a fixture
   set: N small files across 4-5 origins **and** one large synthetic
   Codex-shaped full file, emitting files/s + per-stage seconds, with a
   committed baseline under `tests/benchmarks/baselines/`.
2. **AC4 — asserted WAL ceiling test.** No test asserts `index.db-wal` stays
   under `WAL_JOURNAL_SIZE_LIMIT_BYTES` (160 MB) through a bounded full-ingest
   chunk. Add to `tests/benchmarks` or `tests/unit/storage`.
3. **Per-stage durable persistence verification.** §3 found empty
   `stage_timings_s` in older `daemon_stage_events` rows. Confirm a fresh run
   persists `full.index.full_replace.*` sub-phase timings so the before/after
   per-stage diff is real and not silently empty.
4. **AC3 residual — `full.index.full_replace` profile.** The 1400-2200 s Codex
   attempts are the real re-ingest tax. Profile FTS suspend/insert/restore +
   message/block bulk rewrite (`write.py` ~1514-1576) for large single-session
   files; this is where snappiness is won or lost, far more than the small-file
   0.2 files/s case.
5. **AC5 — post-interrupt RSS assertion.** No automated check that daemon RSS
   drops after Ctrl-C + checkpoint (observed 3.6 GiB residual until SIGTERM).
   Add a supervised dogfood check around `release_process_memory()` +
   shutdown path.

## Outcome
Existing surfaces (`live_ingest_attempt` fields, `ops diagnostics workload
--compare`, `bench ingest-amplification`) already capture throughput, per-stage
timing, RAM, WAL, and IO — enough to run the before/after dogfood above without
new code. The real re-ingest cost is the large-Codex `full.index.full_replace`
FTS path (live: up to 37 min/file), not the small-file symptom. Closing #2391
needs: a multi-provider+large-Codex throughput benchmark (AC1), a WAL ceiling
assertion (AC4), per-stage-persistence verification, the full_replace profile
(AC3), and a post-interrupt RSS check (AC5).

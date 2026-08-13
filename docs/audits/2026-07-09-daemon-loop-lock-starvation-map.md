# Daemon loop lock/starvation map

**Date**: 2026-07-09
**Bead**: polylogue-9e5.7
**Method**: static trace of every long-lived loop `polylogue/daemon/cli.py`
starts (plus the loops it wires in from `sources/live/watcher.py`,
`daemon/embedding_backlog.py`, `daemon/fts_automerge.py`), cross-checked
against one live observation session against the real `polylogued` daemon
(PID 22367, running ~11h at observation time, systemd unit `polylogued.service`,
started 2026-07-09 01:09:41 CEST). Read-only throughout: `systemctl --user
status`, `/proc/<pid>` inspection, `polylogue ops status --json`, direct
read-only `sqlite3` queries against `ops.db`, and `journalctl --user -u
polylogued`. No mutating command was sent to the daemon and no product code
was changed to produce this audit.

## Method

For each loop: identify its trigger (fixed-interval `asyncio.sleep`,
event-gated, or config-driven), which DB file(s)/table(s) it touches, which
connection factory it uses (`open_connection` = full write profile / 128 MiB
cache; `open_daemon_connection` = daemon write profile / 16 MiB cache;
`open_readonly_connection` = `mode=ro` URI, `query_only` pragma), and its
transaction/lock discipline. All write connections run in `journal_mode=WAL`
with `busy_timeout` set per profile (30 s for `open_connection`/
`open_daemon_connection`, 5 s for `open_readonly_connection` — see
`storage/sqlite/connection_profile.py`). WAL mode means readers never block
writers and writers never block readers; the only real contention axis is
**writer vs. writer** (SQLite allows exactly one write transaction at a time
per file) and **checkpoint vs. long-lived reader/writer**.

Starvation-risk classification (same three buckets as the sibling
`polylogue-9e5.4` race audit):

- **ruled-out-safe** — read-only, or self-bounded by design (explicit small
  timeout/limit), or the write is a single fast autocommit statement.
- **theoretically-possible-but-not-observed** — a plausible interleaving
  exists (two loops both need the writer lock) but nothing in 9 days of live
  logs shows it actually biting.
- **evidenced** — real timing/log data from the live daemon shows the window
  actually opening.

## Loop inventory

`run_daemon_services` (`daemon/cli.py:796`) starts these tasks. All periodic
maintenance loops except WAL checkpoint/FTS merge/heartbeat/health/optimize/
status-snapshot/Drive-catchup wait on a `catch_up_complete` `asyncio.Event`
bridged from the watcher's own startup catch-up scan (`_bridge_catch_up_complete`),
so they cannot contend with the *initial* catch-up burst — only with
steady-state ingest.

| # | Loop | Trigger | DB/table touched | Connection profile | Txn/lock shape | Starvation risk |
|---|------|---------|-------------------|---------------------|-----------------|------------------|
| 1 | `LiveWatcher.run()` (`sources/live/watcher.py:178`) — debounced append ingest + `_periodic_catch_up` (15 s) + failed-retry scan | filesystem events (debounced 2 s) + 15 s poll | `source.db` (`raw_sessions`), `index.db` (full parsed tree), FTS triggers, `ops.db` (`ingest_attempts`, cursors) | `write_source_raw_session`/`write_raw_and_parsed` path via the async backend's cached write connection (`open_connection`-class profile) | Single-transaction commit per file/batch (`commit_archive_write_effects`, already audited safe in 9e5.4 #6); `_ingest_append_plans_archive` catches `Exception` broadly but does **not** classify/retry `sqlite3.OperationalError` the way `_catch_up`/`batch.py` do | **evidenced** — see narrative; the debounced single-file append path is the one that actually failed live |
| 2 | `_periodic_raw_materialization_convergence_after` (`cli.py:447`) | gated on catch-up, then 30 s | `source.db` (blob-reference restore), `index.db` (raw→index repair) | mix of `open_connection`/repository calls inside `repair_raw_materialization` | Bounded batch (`limit=25`), catches `sqlite3.OperationalError`/`is_transient_sqlite_lock` and defers to next tick | theoretically-possible-but-not-observed |
| 3 | `_periodic_session_insight_convergence_after` (`cli.py:468`) | gated on catch-up, then 60 s, bursts up to 10× with 1 s pause | `index.db` session profiles/insights | `open_daemon_connection` (`_drain_session_insights_once`) | One connection per burst iteration, closed each time; catches transient-lock | theoretically-possible-but-not-observed |
| 4 | `_periodic_convergence_check` → `_drain_convergence_debt_once` (`cli.py:411`,`557`) | gated on catch-up, then 60 s | `ops.db` (`convergence_debt` via `CursorStore`), `index.db` via a **fresh `DaemonConverger`+`ProcessPoolExecutor(max_workers=2)` constructed every tick** | `CursorStore` uses `_connect_ops()` one-shot connections (see row 9 below); converger batch work happens off-process | Catches `is_transient_sqlite_lock`, defers via `convergence_debt` retry semantics (`false_means_pending`) | ruled-out-safe for locking; separately worth noting the fresh process pool per 60 s tick is a resource-churn smell, not a lock-starvation one — not filed here (out of scope) |
| 5 | `_periodic_wal_checkpoint` (`cli.py:249`) | unconditional, 300 s | all 5 tiers via `maybe_checkpoint_archive_wals` | fresh `open_connection(db, timeout=1.0)` per tier | PASSIVE checkpoint first (never blocks); TRUNCATE only attempted if PASSIVE reports `busy==0`, and even then bounded by the 1 s `timeout_s` — worst case a 1 s wait then a caught `sqlite3.Error` | **ruled-out-safe** — explicit engineering (see `wal_checkpoint.py` docstring) already targets exactly this starvation shape; confirmed live (`wal_busy_pages=0` on every observed batch) |
| 6 | `_periodic_fts_merge` (`cli.py:227`) | unconditional, 300 s | `index.db` FTS surfaces (`messages_fts`, `session_work_events_fts`, `threads_fts`) | `open_connection(db, timeout=5.0)` | One `INSERT ... ('merge', 500)` per surface, work-unit-bounded (~2-4 MiB WAL each), autocommit-scale | ruled-out-safe |
| 7 | `_periodic_heartbeat` (`cli.py:361`) | unconditional, 900 s | `index.db` (`sessions`/`messages` counts) | `open_readonly_connection` | Read-only | ruled-out-safe |
| 8 | `periodic_embedding_backlog_check` (`daemon/embedding_backlog.py:21`) | gated on catch-up, then 60 s | `index.db`/`embeddings.db` | `open_readonly_connection` for the existence probe, write path for the actual drain | Catches transient-lock, skips a tick | theoretically-possible-but-not-observed |
| 9 | `_periodic_health_check` (`cli.py:664`) | unconditional, config-driven (`health_check_interval_s`, default 300 s) | `ops.db` read via `open_readonly_connection` (`_check_schema_version_fast`) plus config reload | Read-only + notification dispatch (no DB write) | ruled-out-safe (read-only) |
| 10 | `_periodic_db_optimize` (`cli.py:380`) | unconditional, 86 400 s (24 h) | all 5 tiers, `PRAGMA optimize` | `maybe_optimize_archive_tiers` (own connections per tier) | Once/day; SQLite's own optimize pragma is itself a bounded scan | theoretically-possible-but-not-observed (never observed in this daemon's 9-day log window — process is younger than 24h at last restart) |
| 11 | `_periodic_status_snapshot_refresh` (`cli.py:282`) | unconditional, 10 s | `index.db`/`ops.db` reads only, via `daemon_status_payload()` | Read-only surfaces feeding an in-process cache (`status_snapshot.py`); no DB write, guarded by a non-blocking `_REFRESH_LOCK` that skips overlapping refreshes | ruled-out-safe |
| 12 | `_periodic_drive_source_catchup` (`cli.py:352`, only if `enable_source_catchup`) | unconditional, 3600 s | `source.db` (`raw_sessions`, `source_file_cursor`), `index.db` | `ParsingService.ingest_sources` → `AcquisitionService.acquire_sources`, own `build_runtime_services` backend (**separate connection from the watcher's own**), including `_persist_source_cursors` walking the entire configured source directory and upserting a `source_file_cursor` row per file | No per-file transient-lock classification visible from the acquire-stage timing; wraps the whole hourly source scan | **evidenced** — this is the loop implicated in the live starvation window below |
| 13 | Browser-capture HTTP server (`server.serve_forever`, thread via `asyncio.to_thread`) | HTTP requests | writes land through the normal watcher/append path once captured to spool | N/A (thread, not asyncio loop) | Not itself a DB writer | ruled-out-safe |
| 14 | Daemon API HTTP server (`api_server.serve_forever`) | HTTP requests | `ops.db`/`index.db` reads for status/health endpoints | `open_readonly_connection` | Read-only | ruled-out-safe |
| 15 | `notify-rs` inotify thread (native `watchfiles`) | filesystem events | none (feeds row 1's queue) | N/A | N/A | ruled-out-safe |

That is 10 periodic `asyncio` maintenance loops (rows 2–12, one conditional)
plus the watcher's own two internal loops (debounced batch + 15 s periodic
catch-up, row 1) plus 2 HTTP server threads and 1 inotify thread — matching
the bead's "~9 concurrent loops" framing once the watcher-internal and
HTTP-server tasks are folded in.

## Live daemon observation

**Session**: read-only, against the production `polylogued` (PID 22367,
`systemctl --user status polylogued`: active since 2026-07-09 01:09:41 CEST,
Tasks: 12, VmRSS 728 MB, IO 610 GB read / 13.7 GB written since start).
`/proc/22367/task` enumerates 12 threads: the asyncio main thread, a
`notify-rs inoti[fy]` thread (row 15), and the rest are the default
`asyncio.to_thread`/executor pool workers used by the two HTTP `serve_forever`
calls and the `asyncio.to_thread(...)` wrapped sync DB work — consistent with
the static trace; there is no separate OS thread per periodic loop since they
are all `asyncio.Task`s multiplexed on the single event loop.

`polylogue ops status --json` reported `daemon_liveness: true`,
`daemon_watcher`/`daemon_api`/`browser_capture` all `state: ready`,
`daemon_ingest.running_count: 0` (idle at observation time), `embeddings`
`state: blocked` (Voyage key not configured — unrelated to locking),
`raw_materialization` 0 debt/actionable. `ops.db`'s `convergence_debt` table
was **empty** (0 rows) at observation time — no outstanding derived-debt
backlog, consistent with rows 2–4 above not currently fighting anything.

Querying `ops.db` directly (`sqlite3 "file:ops.db?mode=ro"`) hit
`Error: database is locked` on the very first bare attempt — a live,
first-hand demonstration that even read-only access can transiently fail:
the default `sqlite3` CLI opens with `busy_timeout=0`, unlike this daemon's
own `open_readonly_connection` (5 s timeout). Adding `.timeout 3000` made
every subsequent query succeed; this is a CLI-tooling artifact, not a daemon
defect (the daemon's own read connections already carry the 5 s timeout).

**`daemon_events` gap analysis**: over the last 500 `ingestion_batch` events,
gaps between batches ranged 7 s to 4855 s (avg 34 s); the 10 largest all-time
gaps are all in the tens-of-thousands-of-seconds range (up to 99 158 s ≈ 27 h)
and land overnight — these are genuine idle windows (no new files to ingest),
**not** starvation; the watcher's periodic catch-up scan finds nothing to do
and correctly emits no batch event.

**The one real starvation window found**: cross-referencing
`journalctl --user -u polylogued` against the Drive-source-catchup loop's own
`acquire elapsed_s=` log line (`pipeline/services/parsing_workflow.py:201`)
shows its hourly runs are usually 5-17 s (nothing to acquire, 246 files
skipped) but three times in the last 9 days ballooned:

```
2026-07-09 03:13:18  acquire elapsed_s=204.42   raw_ids=0 skipped=246
2026-07-09 04:16:14  acquire elapsed_s=175.98   raw_ids=0 skipped=246
2026-07-09 05:32:28  acquire elapsed_s=968.25   raw_ids=0 skipped=246   (~16 min)
```

During each of those windows the live watcher logged repeated
`live.watcher: archive busy during periodic catch-up; will retry` /
`archive busy; requeueing N changed file(s)` warnings (03:10:58-03:13:14,
04:13:52-04:16:10, 05:29:05-05:32:28) — its own bounded-retry path
(`_is_database_locked` classification in `watcher.py:229`) correctly caught
and requeued those. But at **05:16:25** and **05:16:30** — right at the start
of the 05:32:28-ending run — two single-file debounced appends hit an
**uncaught** `sqlite3.OperationalError: database is locked` inside
`write_source_raw_session` (`archive_tiers/source_write.py:239`, reached via
`append_ingest.py:91` → `archive.py:941/982`), logged as
`live.watcher: archive append ingest failed for <path>` with a full traceback.
Repo-wide, `append_ingest.py`'s catch-all (`except Exception:` at
`append_ingest.py:101-102`) is the **only** ingest write path that does not
route through `is_transient_sqlite_lock`/`_is_database_locked` classification
(`grep` confirms that helper is used in `watcher.py`, `batch.py`,
`convergence_stages.py`, `embedding_backlog.py`, and `cli.py`, but not
`append_ingest.py`). Frequency: exactly 2 occurrences of "archive append
ingest failed" in `journalctl` since 2026-07-01 (9 days), both inside this
one incident. Both failed files were still on disk and unmodified after the
failure, so they are picked up by the next 15 s periodic catch-up scan
(row 1) — this is a bounded (\<30 s) availability hiccup, not data loss, but
it is a real, reproduced starvation event: the hourly Drive-catchup writer
holds/contends for the single SQLite writer role long enough to defeat a
concurrent real-time append that lacks the same retry classification every
sibling write path already has.

Root cause is not fully isolated to one SQL statement from static reading
alone (the 968 s run is dominated by `_persist_source_cursors` walking the
entire configured source tree and upserting one `source_file_cursor` row per
file even when every file is already cached/skipped, run on the Drive
catchup's own separate `build_runtime_services` connection rather than the
watcher's) — but the correlation between the three multi-hundred-second
Drive-catchup runs and the "archive busy" bursts, plus the two exact-timestamp
uncaught failures, is strong enough to file as an evidenced (not merely
theoretical) starvation pairing.

## Starvation-risk summary

| Pair | Verdict |
|---|---|
| Drive-source-catchup (row 12) vs. debounced single-file append (row 1) | **evidenced** — 3 slow runs / 9 days, 2 uncaught append failures exactly overlapping one of them |
| WAL checkpoint (row 5) vs. any writer | **ruled-out-safe** — PASSIVE-first, 1 s bounded TRUNCATE attempt, `busy_pages=0` on every observed sample |
| FTS periodic merge (row 6) vs. ingest writes | ruled-out-safe — 500-work-unit bound, ~2-4 MiB WAL ceiling per call |
| Convergence-debt retry / raw-materialization / session-insight loops (rows 2-4) vs. ingest | theoretically-possible-but-not-observed — all three already classify `is_transient_sqlite_lock` and defer via `convergence_debt`/next-tick, and `convergence_debt` was empty at observation time |
| Embedding backlog drain (row 8) vs. ingest | theoretically-possible-but-not-observed — same transient-lock classification present |
| Heartbeat / health-check / status-snapshot / DB-optimize (rows 7, 9-11) vs. anything | ruled-out-safe — read-only or non-blocking by construction |
| HTTP servers / inotify thread (rows 13-15) vs. anything | ruled-out-safe — not DB writers themselves |

## Follow-up filed

- **polylogue-iwmt** (`discovered-from:polylogue-9e5.7`) — give
  `append_ingest.py`'s single-file debounced append path the same
  `is_transient_sqlite_lock` classification/requeue behavior that
  `watcher.py`'s periodic catch-up, `batch.py`, `convergence_stages.py`, and
  `embedding_backlog.py` already have, so a concurrent long-held writer (the
  hourly Drive-source catch-up in particular) degrades to a bounded requeue
  instead of a logged hard failure. No fix implemented here — this audit is
  read-only per the epic's contract (`polylogue-9e5`).

## What was not verified

- No way was found to get a true per-`asyncio.Task` breakdown from the live
  process (Python doesn't expose task identity via `/proc`); the "12 threads"
  view only distinguishes OS-thread-level concurrency (HTTP server threads,
  inotify thread, executor pool) from the single-event-loop `asyncio.Task`s,
  which all share one thread and were confirmed only via the static trace,
  not runtime introspection (no debug/introspection endpoint exists for
  listing live `asyncio.Task`s on this daemon).
- The exact SQL statement(s) responsible for the Drive-catchup loop's 175 s-
  968 s runtimes were not isolated line-by-line (would require adding
  instrumentation or reproducing under a debugger against a large synced
  source tree) — the finding rests on log-timestamp correlation plus static
  reading of `_persist_source_cursors`'s per-file upsert loop, not a proven
  single root cause.
- `_periodic_db_optimize` (24 h cadence) was not observed firing during this
  session; the daemon's current continuous-uptime process (PID 22367) has
  been running only ~11 h, and the log window checked (`journalctl --since
  2026-07-01`) spans prior process incarnations restarted more often than
  daily, so no real optimize-vs-ingest timing sample exists yet.
- `cursor_lag_samples` (ops.db) was empty at observation time — the cursor-lag
  sampler either hasn't fired recently or samples are pruned; no live timing
  evidence was available from that table.

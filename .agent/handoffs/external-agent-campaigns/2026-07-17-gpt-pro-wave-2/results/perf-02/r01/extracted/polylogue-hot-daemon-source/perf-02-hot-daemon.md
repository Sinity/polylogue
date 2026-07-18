Title: "Hot/thick daemon: keep-warm read services, bounded catch-up latency, and a measured memory envelope"

Result ZIP: `perf-02-hot-daemon-r01.zip`

## Mission

The operator wants Polylogue to feel instant: the daemon (`polylogued`) is
always running, so reads should never pay cold-start, cache-cold, or
catch-up-storm costs. Today: status assembly has been observed at 15+ seconds
(`polylogue-20d.17`), coordination reads at 2.6–16.6 s, full-ingest catch-up
latency and WAL shape are untracked (`20d.6`), and the daemon catch-up memory
envelope is unmeasured/unbounded (`ng9m`). Read those Beads plus `20d` epic
context, `daemon/cli.py`, `daemon/convergence*.py`, `daemon/http.py`,
`daemon/status.py`, and the interruptible-read/QueryTransaction seams in
`archive/query/`.

Produce an implementation draft for the thick-daemon architecture:

1. **Keep-warm read plane**: persistent read connections with page-cache
   affinity (mmap/cache_size pragmas chosen deliberately), pre-warmed
   statement caches for the hot read families (list/read/search/status),
   and an explicit warm-up pass at daemon start + after index generation
   switches. Justify every pragma against SQLite documentation semantics.
2. **Snappy status**: integrate with (do not duplicate) the per-component
   budgeted snapshot direction of `20d.17` — a component scheduler with
   fingerprints, last-good values, deadlines, and staleness marks, so
   `polylogue status` and the HTTP status route serve sub-second from
   snapshots while gathering happens in the background.
3. **Catch-up discipline**: bound ingest catch-up bursts (batch sizes, WAL
   checkpoint cadence, backpressure between watcher and converger) so a
   returning daemon neither starves interactive reads nor balloons RSS;
   define the measured memory envelope and the enforcement mechanism
   (`ng9m`) with degradation instead of OOM.
4. **CLI fast path**: the CLI should prefer daemon-served reads when the
   daemon is healthy (existing HTTP surface) and fall back to direct SQLite
   cleanly; specify detection, timeout, and staleness semantics.
5. Tests: latency assertions on the snapshot path (fixture-scale), catch-up
   backpressure unit tests with the frozen clock, and a memory-envelope test
   using deterministic work counters rather than wall-clock sleeps.

## Constraints

- The daemon remains the sole writer; nothing here may add a second writer
  or a cache that can serve archive-contradicting data without a staleness
  mark (evidence-honesty rule).
- Respect the convergence architecture (check/execute stages, debt); do not
  replace it — bound and schedule it.
- Mark all live-host measurements `unverified`; provide the measurement
  commands for the integrator to run locally.

## Deliverable emphasis

HANDOFF.md: architecture summary, exact files changed/added, pragma/limit
table with rationale, the status-snapshot integration contract with 20d.17,
measurement plan (commands + expected envelopes), and staged integration
order (what can merge independently).

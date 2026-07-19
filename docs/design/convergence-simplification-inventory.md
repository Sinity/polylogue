# Convergence simplification inventory (polylogue-m6tp)

Deletion/collapse inventory for the daemon convergence redesign (polylogue-m6tp,
related P0 polylogue-5jak). This document is deliberately scoped: it lists what
later phases of the redesign will delete or collapse, verified against the
current tree, and states why. **It deletes nothing itself** — phase (a)
(this PR) only adds the parse-stage extraction behind a config flag. Phases
(b)-(d) are tracked follow-up work on polylogue-m6tp.

Read `docs/architecture.md`/`docs/internals.md` for the daemon's general
shape before reading this table; each row assumes the reader already knows
the census -> replay -> materialize pipeline.

## Sequencing recap (from polylogue-m6tp's design sketch)

1. **(a) parse-stage extraction behind a flag on the standard build** — this
   PR. Proves the parse/apply seam works and is equivalence-safe; ships at
   reduced benefit on a GIL build.
2. **(b) 3.14t (free-threaded) daemon deploy** — the same thread-pool code
   path becomes a real 3.9x-9.6x parse speedup once the GIL is provably off
   (`parallel_threads_effective()` gates this; see
   `polylogue/pipeline/services/process_pool.py:62`).
3. **(c) bulk-scale routing** — candidate count/bytes above the
   polylogue-m6tp threshold route to an in-process blue-green generation
   build instead of the trickle conveyor.
4. **(d) deletions** — this table. Each mechanism below exists to work
   around a constraint (process-pool spawn cost, GIL-era writer-starvation
   risk, per-pass bounded-batch orchestration) that (b)/(c) remove.

## Inventory

### 1. Process-pool machinery + spawn workarounds

**What it is:** `polylogue/pipeline/services/process_pool.py` — the shared
`ProcessPoolExecutor` helpers used by every CPU-bound parse dispatch on a
standard (GIL) build:

- `process_pool_context()` (`polylogue/pipeline/services/process_pool.py:23`)
  — forces the `spawn` start method specifically to avoid the forkserver
  deadlock found in production (polylogue-p0pw: 17 minutes with zero parse
  workers ever spawned, parent parked in `as_completed`).
- `process_pool_executor()` (`polylogue/pipeline/services/process_pool.py:93`)
  — constructs a pool with `_initialize_worker_logging` as the per-worker
  initializer, needed only because a spawned worker starts with a fresh,
  unconfigured logging stack.
- `terminate_process_pool()` (`polylogue/pipeline/services/process_pool.py:102`)
  — bounded-timeout cancel/terminate/kill sequence, needed only because a
  process (unlike a thread) cannot be cooperatively interrupted from the
  parent.
- `resolve_parse_worker_count()` (`polylogue/pipeline/services/process_pool.py:43`)
  — resolves `POLYLOGUE_INGEST_PARSE_WORKERS` / cpu-1 default; the worker
  *count* concept survives past this deletion (a thread pool still wants a
  bound), only the process-specific plumbing goes.

**Why it exists today:** on a standard CPython build, `ThreadPoolExecutor`
gives no CPU-bound parse speedup (the GIL serializes it) and, worse, running
parse threads concurrently with an actively write-holding thread measured
~5000x commit-latency inflation (the polylogue-7mtf control-run finding cited
throughout `revision_backfill.py`). `ProcessPoolExecutor` is the only way to
get real parallelism on this build, at the cost of spawn tax, pickling, and
no shared memory.

**What makes it deletable:** phase (b)'s free-threaded 3.14t deploy makes a
plain `ThreadPoolExecutor` both safe (no writer-thread contention, since
phase (a) already sequences parse-then-apply so no writer thread is ever
active *during* parse) and fast (proven 3.9x-9.6x, zero writer interference
in the 7mtf control run). Once the daemon's runtime is provably free-threaded,
every process-pool call site collapses to the thread-pool call site that
phase (a) already introduces for the daemon's own conveyor
(`polylogue/daemon/parse_prefetch.py`) and that `revision_backfill.py`
already has for `parallel_threads_effective()`-gated callers
(`_parse_unique_retained_raws_via_threads`,
`polylogue/sources/revision_backfill.py:989`).

**Which phase deletes it:** (b) removes the `ProcessPoolExecutor` branch from
every call site that currently gates on `parallel_threads_effective()`
(`polylogue/sources/revision_backfill.py:1066` `_parse_unique_retained_raws`);
`process_pool.py`'s process-specific helpers (`process_pool_context`,
`process_pool_executor`, `terminate_process_pool`) are deleted once no caller
remains. `resolve_parse_worker_count()`'s bound survives, retargeted at
thread-pool sizing.

### 2. Pool-amortization heuristics (dispatch-size + aggregate-bytes floors)

**What it is:** two independent guards in `polylogue/sources/revision_backfill.py`
that decide whether a batch is even worth spawning a process pool for:

- `_partition_raws_by_dispatch_size()` (`polylogue/sources/revision_backfill.py:879`)
  + `_parse_dispatch_max_bytes()` (`polylogue/sources/revision_backfill.py:857`,
  default `_DEFAULT_PARSE_DISPATCH_MAX_BYTES = 262_144` / 256 KiB, override
  `POLYLOGUE_REVISION_PARSE_DISPATCH_MAX_BYTES`) — raws at or above 256 KiB
  parse sequentially in-process; the process-pool round trip pickles the
  returned `ParsedSession` list back across the process boundary, which
  measured a net LOSS (0.63x) above this size (polylogue-amg1/#3136).
- `_pool_dispatch_amortizes()` (`polylogue/sources/revision_backfill.py:922`)
  + `_parse_pool_min_aggregate_bytes()` (`polylogue/sources/revision_backfill.py:899`,
  default `_DEFAULT_PARSE_POOL_MIN_AGGREGATE_BYTES = 48 * 1024 * 1024` / 48 MiB,
  override `POLYLOGUE_REVISION_PARSE_POOL_MIN_BYTES`) — an aggregate
  pool-eligible batch under ~45 MB doesn't amortize the ~1.5-2s
  per-worker spawn+import cost (measured live 2026-07-19: 20 short-lived
  workers spending ~95% of their lifetime inside `importlib`).

**Why it exists today:** both guards protect against process-pool-specific
costs (pickle-back of large payloads; per-worker spawn+import tax) that only
exist because `ProcessPoolExecutor` workers are separate interpreters.

**What makes it deletable:** `_parse_unique_retained_raws_via_threads`'s own
docstring (`polylogue/sources/revision_backfill.py:989`) already states the
reason precisely: "Both `_partition_raws_by_dispatch_size` and
`_pool_dispatch_amortizes` exist solely to protect against those two
process-pool-specific costs (#3136/#3149), so this path applies NEITHER" —
a free-threaded `ThreadPoolExecutor` shares `ParsedSession` object graphs by
reference (no pickle) and reuses the one already-imported interpreter (no
per-worker spawn). Once the process-pool branch is gone (item 1), these two
size/aggregate floors have no remaining caller.

**Which phase deletes it:** (b), in the same sweep as item 1 (they gate the
same dead branch).

### 3. The 64 MiB daemon parse envelope narrowing

**What it is:** `_RAW_MATERIALIZATION_DAEMON_BLOB_LIMIT_BYTES = 64 * 1024 * 1024`
(`polylogue/daemon/cli.py:89`), threaded as `max_payload_bytes` into every
daemon-driven `repair_materialization` call
(`polylogue/daemon/cli.py:154` and `:850`). It caps how large a raw's blob
the daemon's conveyor will parse per pass; a raw above this envelope is
deferred (`record_resource_blocked_revision_census`,
`polylogue/sources/revision_backfill.py`) rather than parsed in-line, so one
whale raw cannot balloon the writer hold's memory footprint or duration.

**Why it exists today:** parse currently happens *inside* the writer hold
(`daemon_write_coordinator().run_sync`, `polylogue/daemon/cli.py:685-692`),
so an unbounded parse of a multi-GB raw would hold the process-wide writer
lock — starving live ingest, status queries, and every other write actor —
for as long as that one parse takes. The 64 MiB ceiling is a blunt,
per-component admission gate that trades completeness (whales are refused,
not parsed) for a bounded worst-case hold duration.

**What makes it deletable:** phase (a) (this PR) already moves the parse
itself off the writer hold via `DaemonParseStage`
(`polylogue/daemon/parse_prefetch.py`) — the memory/duration risk from a
large parse no longer threatens the writer hold's own duration once parse
runs before the hold is even requested. What replaces the blob-size ceiling
is `DaemonParseStage`'s explicit in-flight parsed-bytes budget
(`daemon_parse_stage_max_inflight_bytes()`,
`polylogue/daemon/parse_prefetch.py:72`, default 64 MiB, override
`POLYLOGUE_DAEMON_PARSE_STAGE_MAX_INFLIGHT_BYTES`) — a budget on cached
*parsed* memory, not a hard admission refusal on raw *blob* size. Once every
daemon parse path routes through the parse stage (phase (b)/(c) make this
the only path, not an opt-in flag), the per-component refusal ceiling
becomes redundant with the budget and can go.

**Which phase deletes it:** (b)/(c) — specifically, once
`daemon_parse_stage_split` is no longer a flag (the parse-stage path is the
only path), `_RAW_MATERIALIZATION_DAEMON_BLOB_LIMIT_BYTES` and its two call
sites in `polylogue/daemon/cli.py` are replaced by
`DaemonParseStage`'s budget alone.

### 4. Census burst-escalation constants

**What it is:** the daemon conveyor's bounded-pass sizing and back-to-back
burst logic in `polylogue/daemon/cli.py`:

- `_RAW_MATERIALIZATION_CONVERGENCE_BATCH_LIMIT = 16` (`:80`) — replay-sized
  per-pass limit (bounds writer-transaction length).
- `_RAW_MATERIALIZATION_CENSUS_BATCH_LIMIT = 64` (`:88`) — a larger
  census-only-mode limit, because a census-paused pass runs no replay
  transaction and the smaller replay-sized limit "only throttles
  parse-bound census throughput and stretches a large census into days."
- `_RAW_MATERIALIZATION_BACKLOG_BURST_PAUSE_SECONDS = 1` (`:83`) — the yield
  between back-to-back burst passes.
- The `census_mode` escalation switch itself
  (`polylogue/daemon/cli.py:663`, `:678-681`, `:692-696`) — a pass that
  censused components but repaired/executed nothing is treated as progress
  and escalates the *next* pass's limit from 16 to 64.

**Why it exists today:** this whole mechanism compensates for parse and
apply sharing one writer-held pass. Splitting "how many raws to census this
tick" from "how long the writer transaction can safely stay open" is the
root problem #3145/polylogue-m6tp/polylogue-5jak all describe — the batch
limit is doing double duty as both a parse-throughput knob and a
writer-hold-duration knob, and a single number cannot serve both jobs well
(too small starves census throughput on a census-paused backlog; too large
extends the writer hold on a replaying pass).

**What makes it deletable:** once parse is a persistent, continuously-running
background stage (phase (b)/(c) make `DaemonParseStage` — or its bulk-mode
successor — an always-on backlog iterator rather than a per-pass batch), the
writer-held "apply" pass only needs to bound *its own* transaction length
(a much simpler, single-purpose number), and there is no separate
census-vs-replay batch-size distinction left to escalate between: census
throughput is bounded by the parse stage's own worker count and in-flight
budget, not by a per-tick candidate limit.

**Which phase deletes it:** (b) collapses the census/replay batch-size
distinction once parse is continuously running rather than per-tick;
(c)'s persistent backlog iterator (item 5 below) removes the remaining
burst-pass bookkeeping (`census_mode`, the burst `while` loop's pause/yield
logic) entirely.

### 5. Per-pass candidate requery / resume recompute

**What it is:** `repair_raw_materialization`
(`polylogue/storage/repair.py:5695`) recomputes its FULL candidate set from
scratch via `_raw_materialization_candidate_ids()` up to twice per call:
once at entry (`polylogue/storage/repair.py:5783`) and again after the
census loop, to re-check what's still uncensused
(`polylogue/storage/repair.py:5844`). Each call re-scans `raw_sessions`
joined against `index_tier.sessions`/`raw_revision_applications`/
`raw_membership_census` (`polylogue/storage/repair.py:3618` onward) — an
O(backlog size) query repeated every daemon tick regardless of how much of
the backlog actually changed since the previous tick.

**Why it exists today:** the conveyor has no persistent memory of "where it
left off" beyond what's durably recorded in `source.db`/`index.db`
themselves (deliberately — restart-safety requires deriving the candidate
set from durable state, not an in-memory cursor that a crash would lose).
Recomputing from scratch is the simplest way to stay crash-consistent under
today's per-tick, stateless-between-ticks design.

**What makes it deletable:** polylogue-m6tp's design sketch calls for "a
persistent in-daemon backlog iterator" that replaces per-pass candidate
requeries. Once the parse stage (and its eventual bulk-routing successor)
own a long-lived, incrementally-updated view of the pending backlog — fed by
the same durable receipts, but maintained incrementally rather than
recomputed by a full query every tick — a restart still recovers correctly
by rebuilding that view once at startup (not per tick), and steady-state
ticks no longer pay the full-backlog scan cost.

**Which phase deletes it:** (c) — the persistent backlog iterator is
explicitly the mechanism the bulk-routing design introduces (needed there
regardless, to avoid re-scanning tens of thousands of raws once bulk-scale
generation building is in play); once it exists, the per-pass
`_raw_materialization_candidate_ids()` requery in
`repair_raw_materialization`'s trickle path becomes redundant with it.

### 6. The CLI bulk importer's operator-surface status

**What it is:** `polylogue ops maintenance rebuild-index`
(`polylogue/cli/commands/maintenance/_rebuild_index.py:281`
`@click.command("rebuild-index")`, handler `rebuild_index_command` at `:349`)
— today a live operator tool: #3145's daemon-side loud recommendation
(`polylogue/daemon/cli.py` `_maybe_recommend_bulk_rebuild`) tells an operator
to run it by hand when the trickle conveyor's backlog is bulk-scale, and the
2026-07-19 restore incident (polylogue-5jak notes) used it directly as the
only viable path once the daemon's own conveyor made a live backlog
net-negative.

**Why it exists today:** it is the one code path that already does the
right thing for a bulk backlog — one resumable transaction, blue-green
generation, full parse envelope, one census+replay sweep — because it does
not share the daemon's live-ingest constraints (no concurrent watcher, no
per-tick writer-sharing budget, can run with the daemon stopped).

**What makes it deletable as an *operator* tool (not as code):** polylogue-m6tp's
2026-07-19 operator direction states the target plainly: "with free-threaded
3.14t ... normal daemon convergence could BE the fast path, making the CLI
bulk importer unnecessary for ordinary backlogs." Phase (c)'s in-process
blue-green generation building (an inactive generation on a second writer
connection, live ingest continuing on the active index, promoted via the
existing generation-store pointer swap) gives the daemon itself everything
`rebuild-index` does today, without stopping the daemon or freezing the
source. Once that lands, an operator should never need to invoke
`rebuild-index` for routine backlog drains.

**What survives:** the command itself is NOT deleted. Per the
automagic-invariants doctrine (an operator surface for routine work must be
subsumed by automatic convergence, not merely duplicated by it), it becomes
**disaster-recovery break-glass only** — the path used when the daemon
itself cannot run (corrupted state, daemon-dead scenarios; polylogue-k8kj's
robustness work already targets this exact scenario). Its resumable,
transactional, blue-green-generation design is exactly what a break-glass
path needs and should not be simplified away.

**Which phase deletes/collapses it:** (c) is what makes the daemon's own
convergence loop sufficient for ordinary bulk backlogs; the CLI command's
*operator-tool* status (documented recommendation, routine-use expectation)
is retired at that point, while the command and its underlying
`polylogue/maintenance/rebuild_index.py` machinery remain as the
break-glass path. No code deletion is scoped here — only a documentation/
recommendation change (stop telling operators to run it routinely) plus,
optionally, gating it behind an explicit `--i-know-the-daemon-is-down` style
confirmation in a later bead if the break-glass framing needs to be load-bearing
in the CLI itself.

## What phase (a) (this PR) does NOT touch

For clarity, since this document sits next to the parse-stage extraction
PR: none of the six items above are deleted, narrowed, or behaviorally
changed by phase (a). Every mechanism above continues to run exactly as
before when `daemon_parse_stage_split` is off (the default), and continues
to run unchanged even when the flag is on for every code path except the
one new prefetch-cache-hit shortcut in `_parse_retained_raws`
(`polylogue/sources/revision_backfill.py`), which is additive and
equivalence-tested (see
`tests/unit/daemon/test_raw_materialization_parse_stage_equivalence.py`).

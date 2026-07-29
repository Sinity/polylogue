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

### 2. Pool-amortization heuristics (dispatch-size + aggregate-bytes floors) — DELETED

**Status: done.** Both guards, their two constants, their two config
properties (`revision_parse_dispatch_max_bytes`,
`revision_parse_pool_min_bytes`) and their two environment overrides
(`POLYLOGUE_REVISION_PARSE_DISPATCH_MAX_BYTES`,
`POLYLOGUE_REVISION_PARSE_POOL_MIN_BYTES`) are removed, together with the
process-pool branch in `_parse_unique_retained_raws` that they gated.

They existed only to keep a `ProcessPoolExecutor` a net win under the GIL: a
256 KiB payload ceiling (pickling `ParsedSession` graphs back across the
process boundary measured 0.63x, a net loss, polylogue-amg1/#3136) and a
48 MiB aggregate floor (per-worker spawn+import measured ~1.5-2.0s, #3149).
A free-threaded `ThreadPoolExecutor` shares object graphs by reference and
reuses the already-imported interpreter, so neither cost exists and neither
guard has a caller.

Measured cost of keeping them: on the reference archive the 256 KiB ceiling
sent 16,417 of 41,363 raws — 90.84 GiB of 92.22 GiB, **98.5% of all bytes** —
to a single-core sequential parse, which is what made a full index rebuild a
~9-hour job on a 24-thread machine.

`resolve_parse_worker_count()` survives as the single knob bounding parse
width, retargeted at thread-pool sizing.

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

**Status (2026-07-29): landed.** `_RAW_MATERIALIZATION_CENSUS_BATCH_LIMIT` and
the `census_mode` escalation switch are deleted from
`polylogue/daemon/cli.py`. The writer-held pass's limit
(`_RAW_MATERIALIZATION_CONVERGENCE_BATCH_LIMIT`, unchanged at 16) is now a
pure writer-hold-duration bound with no second job. Census/parse throughput
moved entirely to a new, independent knob,
`_RAW_MATERIALIZATION_PARSE_STAGE_WARM_LIMIT = 64`, that only bounds
`_maybe_warm_raw_materialization_parse_stage`'s off-writer-hold prefetch —
this runs before the writer hold is ever requested (`DaemonParseStage`,
`polylogue/daemon/parse_prefetch.py`), so it never trades against hold
duration. The writer-held pass's own limit now widens (bounded floor
`_RAW_MATERIALIZATION_CONVERGENCE_BATCH_LIMIT`, bounded ceiling
`_RAW_MATERIALIZATION_PARSE_STAGE_WARM_LIMIT`) to match however many raws the
prefetch warmer *actually* admitted this tick (`warmed_count`), rather than
guessing via a `census_mode` boolean derived from the PRIOR pass's outcome —
consuming an already-parsed cache hit costs one receipt write, not a
reparse, so widening to match real warmed state does not meaningfully extend
the hold. With `daemon_parse_stage_split` off (today's default,
`polylogue/config.py`), the warmer is a no-op and the pass limit is simply
the floor, unchanged from before this change in that configuration. See
`test_periodic_raw_materialization_burst_continues_through_census_passes`
(flag-off floor behavior) and
`test_periodic_raw_materialization_flag_on_widens_limit_to_match_warmed_count`
(flag-on widening) in `tests/unit/daemon/test_daemon_cli.py`.

**What it was:** the daemon conveyor's bounded-pass sizing and back-to-back
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

**Status (2026-07-29): investigated, NOT deletable this session — reverted a
naive fix, documenting the dead end to save the next attempt the same
detour.** Re-verifying against the current tree found the requery is far
more entrenched than this row originally described: `repair_raw_materialization`
(now `polylogue/storage/repair.py:5959`, not `:5695` — the file has grown
substantially since this doc's snapshot) calls
`_raw_materialization_candidate_ids()` **five** times in one function body
(`:6047`, `:6109`, plus three more further down at roughly `:6481`, `:6626`,
`:6674` in the current tree), not two, and the same helper has a dozen call
sites across `polylogue/storage/repair.py` total.

The first fix attempted was a per-process memoization cache keyed on
`(archive_root, filter scope, source.db data_version, index.db data_version)`,
using `PRAGMA data_version` to detect "has either durable tier changed since I
last computed this." It was reverted after breaking ~29 tests in
`tests/unit/storage/test_repair.py` and
`tests/unit/devtools/test_raw_authority_scale_proof.py` with genuine
staleness (wrong candidate counts, spurious "postflight changed a
retryable/carried-forward plan" errors) — root-caused with a standalone
repro (`sqlite3` CLI and Python bindings both), not a guess:

- `PRAGMA data_version`, read from a freshly-opened `mode=ro` connection
  immediately before/after commits on a separate long-lived writer
  connection to the same file, did not change at all across three
  consecutive commits, with or without WAL mode, with or without an
  intervening `wal_checkpoint`. This contradicts the naive reading of the
  pragma's purpose and made it useless as implemented here.
- The SQLite file-header change counter (bytes 24-27, read directly, no
  pragma) DOES increment reliably per commit under rollback-journal mode,
  but under WAL mode (source.db/index.db's actual journal mode) it only
  updates at checkpoint — writes land in the `-wal` file, which the header
  counter does not see until the WAL is checkpointed back into the main
  file.
- Combining the header counter with the WAL file's own size as a
  WAL-mode-aware fallback signal fails too: `wal_checkpoint(TRUNCATE)`
  resets the WAL file to empty, so the combined signature after
  checkpoint-N + one write can numerically collide with the signature after
  checkpoint-(N-1) + one write, producing a false cache hit across a
  checkpoint boundary.

No cheap, purely-read-only, single-query-per-tick invalidation signal was
found that is provably sound against every predicate the five-call-site
query depends on (parse_error, membership-census completeness,
byte-authority state, application-terminal state — see
`_raw_materialization_candidate_ids`, `polylogue/storage/repair.py:3758`).
Building one correctly requires either explicit invalidation hooks wired
into every write path that mutates `raw_sessions`, `raw_revision_applications`,
`raw_membership_census`, or `raw_session_memberships` (those write paths live
in `polylogue/storage/raw_authority.py` and
`polylogue/storage/sqlite/archive_tiers/revision_application.py`, both
outside this session's write scope) — or the real persistent iterator this
row already calls for below, sized as its own tracked follow-up rather than
a same-session bolt-on. The **candidate cache code was reverted in full**;
`polylogue/storage/repair.py` is unchanged from before this investigation.

**What it is (unchanged from the original finding):** `repair_raw_materialization`
recomputes its FULL candidate set from scratch via
`_raw_materialization_candidate_ids()` repeatedly per call. Each call
re-scans `raw_sessions` joined against `index_tier.sessions`/
`raw_revision_applications`/`raw_membership_census`
(`polylogue/storage/repair.py:3758` onward) — an O(backlog size) query
repeated every daemon tick regardless of how much of the backlog actually
changed since the previous tick.
||||||| b64a074e5
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

**Second attempt (polylogue-iy3n, 2026-07-29) — also reverted, and it identifies the failure as STRUCTURAL rather than a tuning problem.**

**What it is:** `repair_raw_materialization`
(`polylogue/storage/repair.py:5959`, verified against the current tree —
this line has drifted before and will again) recomputes its FULL candidate
set from scratch via `_raw_materialization_candidate_ids()` up to five times
per call (entry, after the census loop, on a stale-plan rejection path, once
per executed replay component inside the apply loop, and once more at the
end) plus several more call sites elsewhere in the file. Each call re-scans
`raw_sessions` joined against `index_tier.sessions`/
`raw_revision_applications`/`raw_membership_census` (`_raw_materialization_candidate_ids`,
`polylogue/storage/repair.py:3758` in the pre-fix tree) — an O(backlog size)
query repeated every daemon tick regardless of how much of the backlog
actually changed since the previous tick.

**Attempted fix (polylogue-iy3n, polylogue-m6tp fast-follow, 2026-07-29) —
reverted, second failed attempt in this same class:** a generation-counted, explicitly
invalidated in-process cache was built and then reverted after finding the
same failure shape as the earlier `PRAGMA data_version` attempt this section
already describes, just via a different mechanism. Design tried: memoize
`_raw_materialization_candidate_ids` per `(archive_root, raw_artifact_id,
provider, source_family, source_root)`, invalidated by an explicit
`bump_raw_materialization_candidate_generation()` counter called from every
writer of the five relevant tables that this bead's write scope
(`repair.py`, `raw_authority.py`,
`storage/sqlite/archive_tiers/revision_application.py`, `daemon/**`) could
reach directly, plus a default-invalidate backstop in
`daemon/write_coordinator.py` (the one choke point every daemon writer
passes through) for actors outside that scope, with a narrow, individually
source-audited exemption list for actors proven never to touch those tables
(FTS merge, embedding backlog, judgment automation, blob GC, heartbeat,
`PRAGMA optimize`, convergence-debt retry).

This reduced generation-bump noise correctly for every writer *this bead is
allowed to instrument*. It failed against writers it is not allowed to
instrument: `tests/unit/storage/test_repair.py::test_raw_materialization_replays_governed_bundle_after_index_reset`
deletes and reinitializes `index.db` directly between two
`repair_raw_materialization` calls (modeling the documented, ordinary
`polylogue ops reset --index && polylogued run` operational flow — see this
repo's schema-regimes doctrine) and
`test_raw_materialization_reports_uncensused_append_fragments_as_pending_debt`
writes `raw_membership_census` via direct SQL between two
`_raw_materialization_candidate_ids` calls. Neither path routes through any
function reachable from this bead's write scope: the real writers of
`raw_sessions`/index-tier `sessions`/`raw_membership_census`/
`raw_session_memberships` live in `polylogue/sources/live/*` (live ingest,
explicitly out of scope), `polylogue/storage/repository/**` (explicitly out
of scope), and `polylogue/storage/sqlite/archive_tiers/archive.py` /
`storage/sqlite/queries/{raw_writes,raw_state}.py` (not owned by this bead).
Both failing tests are legitimate simulations of real external-mutation
scenarios, not test artifacts to route around — an index reset genuinely can
run against a live archive, and out-of-band census/evidence writes are the
documented shape of `raw_membership_census`. The cache returned stale
candidate sets in both cases, the same observable failure the
`data_version` attempt produced, just reached through unaudited writers
instead of a WAL-invisible change counter. Reverted cleanly (`git checkout
--`); `tests/unit/storage/test_repair.py` +
`tests/unit/devtools/test_raw_authority_scale_proof.py` reconfirmed green
(86 passed) on the reverted tree.

**Conclusion:** a correct fix needs either (a) invalidation hooks in the
actual writer modules above, which sit outside every write-scope grant this
bead has received twice now, or (b) the persistent backlog iterator this
section already names as the true fix, which restructures the source of
truth itself instead of trying to cache a query over it. (a) is a
cross-lane change (touches `sources/live/*`, `storage/repository/**`,
`storage/sqlite/archive_tiers/archive.py` — each plausibly another lane's
scope); (b) is squarely phase (c) below. Re-attempting this item with a
narrower cache (e.g. scoped to only fire when `raw_artifact_id` is set, or
gated on a coarser "has the daemon done anything since boot" flag) would
just be a smaller version of the same unsound shape — the failure is
structural (writers outside instrumentable scope), not a tuning problem.

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

**What survives:** the *machinery*, not the operator surface. The resumable,
transactional, blue-green-generation implementation in
`polylogue/maintenance/rebuild_index.py` becomes daemon-internal — phase (c)
invokes it from convergence routing. The `ops maintenance rebuild-index` CLI
command is **deleted** once the daemon path is proven equivalent
(polylogue-gd6v acceptance). There is no break-glass tier (operator doctrine,
2026-07-19): a redundant manual surface kept "just in case" is exactly the
random-machinery packing this codebase aggressively purges. The scenarios
that seemed to justify one dissolve on inspection — a daemon that cannot run
is a bug to fix in the daemon, and a corrupted/mismatched derived tier is
already the daemon's own rebuild-on-mismatch invariant (the derived-tier
schema regime). Read-only *diagnostic* inspection surfaces may survive;
nothing that mutates does.

**Which phase deletes/collapses it:** (c) lands the daemon routing and, in
the same change-train once equivalence is proven, deletes the CLI command,
its Click plumbing, and its operator documentation. No confirmation flags,
no deprecation period, no alias.

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

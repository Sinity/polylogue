# Daemon core

The resident authority: what it owns, how writes are serialized, what the CLI
and MCP call, how ingest is shaped, how derived models converge, how services
live and die, and what status costs.

Scope: `polylogue/daemon/`, the live intake path in `polylogue/sources/live/`,
and the write-connection seam in `polylogue/storage/sqlite/`. Owning beads:
polylogue-bp12n (resident authority), polylogue-bp12n.1 (convergence),
polylogue-avmq (service lifecycle), polylogue-8qm4k (concurrent source.db
writers), polylogue-623q (import performance envelope), polylogue-gmw2 (fair
intake), polylogue-20d.17 (budgeted status).

## 1. Measured starting point

Rehearsal on a scratch root, production route (`polylogued run`), free-threaded
CPython 3.14, i7-13700K (24 threads). Receipts: `daemon_stage_events` in the
rehearsal `ops.db`, `status='completed'`; driver
`/realm/tmp/work/rehearsal-4/run.sh`.

> **Rehearsal-11 is not a valid throughput benchmark.** From catch-up chunk
> 571 of 12,150 to the end of the run, claude-code — 23 GB of the 64.5 GB
> backlog — was in a terminal `refusing further ingest until restart` state
> (`daemon.log:28550`, one line, logged once), and 926 subsequent chunks
> planned four files, ingested none, and each still took the writer lease for
> 1.0–3.1 s (polylogue-kqrbw). Separately, `input_bytes` counts *offered*
> bytes, not ingested ones (polylogue-bsv8j): 6.83 GB offered against 2.09 GB
> ingested, the difference dominated by two refused files of 3,557 MB and
> 1,042 MB. The **honest baseline is 0.35 MB/s ingested**. The per-chunk stage
> attribution below is still sound — it measures real work on chunks that did
> real work — but the throughput headline and the process-parallelism sample
> must be retaken on a healthy run (rehearsal-12, from a pinned worktree).

| Quantity | Measured |
| --- | --- |
| Input ingested | 2.09 GB in 5,999 s wall = **0.35 MB/s** |
| Input offered (the inflated figure) | 6.83 GB = 1.14 MB/s |
| Backlog | 64.5 GB / 46k session files, plus 339k hook sidecars (1.7 GB) |
| Projection at this rate | ~15.7 h against a sub-1 h target |
| Process CPU / wall, 12×30 s samples | **0.55–1.36, mean 1.13** on 24 cores |
| Chunk `total_time_s`, 587 chunks | 4,811 s |
| — named stage timings | 1,536 s (32 %) |
| — convergence | 567 s (12 %) |
| — residual (unoverlapped parse and reads) | 2,708 s (56 %) |
| `parse_time_s` charged to the same chunks | 3,771 s |
| Bytes read for fingerprint + payload | 4.06 GB + 1.72 GB = 5.78 GB against 6.83 GB input |
| `archive_write_bytes_delta` | 2.19 GB over 1,536 writer-seconds = **1.4 MB/s of DB growth** |
| Chunk shape | median 4 files / 1.48 MB / 3.5 s; ceiling 4 files or 16 MiB |

Writer-side stage sums over the same 587 chunks:

```
816.9 s  full.index_parsed_write
  518.1 s  full.index.full_replace
    396.7 s  full.index.full_replace.blocks
   64.8 s  full.index.prepare
247.8 s  fts
215.1 s  derived
  142.6 s  derived.tag_rollup_count
128.1 s  full.source_raw_write
 40.3 s  full.provider_parse
```

Five facts follow, and they are what the design has to answer.

1. **The program is single-threaded on a free-threaded interpreter.** One
   thread (`polylogued`, or `polylogue-write` during a write) carries ~90 % of
   one core; two asyncio executor threads carry the remainder; 19 threads
   exist. Parse, fingerprint, write, FTS, derived and convergence are one
   serial thread of control per chunk. 23 of 24 cores are idle.
2. **The writer is CPU-bound, not I/O-bound.** 1.4 MB/s of database growth on
   NVMe. Nothing about the single-writer rule requires this cost; the cost is
   Python and SQLite btree work on one core.
3. **Every file is read three to five times.** The prefetch reads it
   (`sources/live/batch.py:469`, uncounted), the ingest reads it again
   (`batch.py:2416`, the 6.83 GB figure), and then `_record_full_cursor`
   (`batch.py:1455`) re-hashes the full prefix from disk two to four more
   times: `_full_capture_still_matches` (`batch.py:1669`), the cursor-hash
   authority read (`batch.py:1538`), the post-frontier proof
   (`batch.py:1562`), and the Claude semantic frontier (`batch.py:1546`). The
   observed 1.85× is a floor, held down only by the short-circuit at
   `batch.py:1641`. **The re-reads exist to prove the file did not change
   between capture and cursor write** — a concurrency proof, not a hashing
   requirement. All of these hashes are computable in one pass over bytes
   already in memory.
4. **Archive-wide deferral exists and leaks.** `ConvergenceStage.whole_archive`
   (`daemon/convergence.py:90`) plus `whole_archive=False` on every catch-up
   chunk but the last (`sources/live/watcher.py:695`) already skips four
   stages. Two escape it: `raw_parse_recovery` (`convergence_stages.py:1005`)
   is keyed on the *source root path*, so its scan is corpus-scoped every
   chunk, and `standing-queries` (`convergence_standing_queries.py:78`)
   re-evaluates every watched query per chunk. Separately,
   `derived.tag_rollup_count` — 142.6 s over 567 invocations — sits inside the
   per-session `derived` stage. The deferral is also all-or-nothing per chunk
   and runs *inside* the chunk's writer hold (`batch.py:1163`), so the final
   chunk's archive-wide audit extends one hold by its whole duration.
5. **The cold backlog is driven by live-trickle constants.** Four files or
   16 MiB per chunk (`watcher.py:95-96`), applied strictly serially
   (`watcher.py:668`), is right for a session file appended a line at a time
   and wrong for 46k files sitting on disk. `BULK_BUILD_WRITE_CONNECTION_PROFILE`
   (−25 % apply time on bead 623q) has exactly one selection site,
   `archive_tiers/archive.py:826`, gated solely on `owned_inactive_generation`
   (`archive.py:721`) — reachable only from `revision_backfill.py:2486` and
   `storage/repair.py:6211`, i.e. **the manual rebuild engine being deleted in
   PR #4698**. Unless it is re-homed onto the batch policy it dies with that
   engine.

The per-session full-replace DELETE cascade is *not* on the list:
`full.index.full_replace.clear_projection_rows` fired on 5 of 565 chunks, so
the polylogue-cs86 skip is working and `full_replace` cost is genuine insert
cost.

## 2. Shape

One resident process owns four things and nothing else:

| Authority | Owns |
| --- | --- |
| **Write lease** | every writable connection to every tier |
| **Operation table** | the declared request/result contract every surface calls |
| **Intake** | what to ingest next, fairly and resumably |
| **Derivation kernel** | what derived state is missing, and publishing it |

Everything else in the process is a declared service under one supervisor.
Browser HTTP, the web shell and the UI are a separate process
(polylogue-bp12n.2) and are out of this document's scope.

## 3. The write lease

polylogue-8qm4k's stated diagnosis does not survive reading the code: Drive
catch-up (`daemon/cli.py:787`) and the catch-up chunk (`watcher.py:717`) are
both on the *same* coordinator instance — one memoized per event loop
(`write_coordinator.py:685`) — and cannot overlap in-process. Three real
defects fit the evidence instead, and they are what the lease must close:

1. **Holds are unbounded and the gate is advisory.**
   `DaemonWriteCoordinator` is an in-process priority gate around callables
   (`write_coordinator.py:64-121`, `:282-354`). It takes no SQLite lock.
   `cli.py:140-160` records a measured `maintenance.drive_catchup`
   `hold_max` of **18,623 s**, and says outright that
   `_DRIVE_CATCHUP_MAX_PASS_SECONDS = 20.0` does not bound the acquire stage.
   Any writer not on the gate exhausts `DB_TIMEOUT = 30`
   (`connection_profile.py:112`) long before the holder finishes.
2. **The gate fails open on a duck-typing miss.** `watcher.py:365-368`,
   `watcher.py:1787-1790`, `cli.py:1613-1618` and `cli.py:1659-1676` do
   `getattr(coordinator, "run", None)` and, on a miss, fall through to a bare
   `await operation()` — a full archive write with no gate and no log line.
   `LiveWatcher`'s `write_coordinator` parameter defaults to `None`
   (`watcher.py:295`). A test double's shape can disable the single-writer
   invariant in production.
3. **Backup takes a live write lock with no daemon guard.**
   `daemon/backup.py:467-491` opens `sqlite3.connect(live_path, timeout=30.0)`
   and runs `PRAGMA wal_checkpoint(TRUNCATE)` and `BEGIN IMMEDIATE` against the
   live tier, with no `offline_guard` and no lease check.

`DaemonConverger` holds no coordinator at all (`convergence.py:172`); both
production constructions merely happen to be invoked from inside a held gate.

**Rule: a writable connection is obtainable only from a lease.**

- `WriteLease` is the sole constructor of write-mode tier connections. It is
  held by exactly one thread at a time within the process, and it carries the
  connection profile (§5) for the batch policy in force.
- Every write entry point takes a lease token parameter. `ArchiveStore` and the
  tier modules refuse a write-mode open without one. There is no module-level
  writable connection cache reachable without a token, and **no `getattr`
  fallback**: an absent lease is an error, never a bypass.
- **A hold is bounded by declaration.** Every lease acquisition names a maximum
  hold; exceeding it is a typed failure, not a longer wait. Long work
  (Drive acquisition, archive-wide audits) is restructured to compute outside
  the lease and publish in bounded holds, which is the same rule §6 imposes on
  derivations.
- Backup acquires the lease like any other writer.
- Cross-process exclusion stays as it is (the archive-root writer lock).
  In-process exclusion becomes structural rather than conventional.
- A `database is locked` inside a leased write is a retryable unit failure,
  never process death.

Anti-vacuity: a test double that opens a writable tier connection without a
token must be refused, and deleting the refusal must make that test red. A
census test enumerates write-mode opens in `polylogue/` and fails on any that
does not route through the lease. A third: a lease held past its declared
bound must fail rather than block.

## 4. The operation contract

`DaemonOperationSpec` landed with polylogue-bp12n.3
(`polylogue/operations/daemon_protocol.py`) and currently declares five
read operations. It is the right shape; it is missing everything the CLI
program needs.

This design extends the same table rather than adding a second one:

- **Read** (`DIRECT_READ` fallback): unchanged.
- **Write**: one declaration per mutation family — assertions and settings,
  source admission, tagging, deletion apply. `NEVER` fallback, idempotency key
  in the request, typed result. The daemon executes them under the write lease.
- **Control**: pause/resume intake, seal a generation, cancel an operation.
  `NEVER` fallback.
- **Long-running**: candidate construction, reindex, backlog drain. Returns an
  immutable reference after durable acceptance; completion arrives by event,
  with a bounded status read for recovery. No polling DTO.

Consequences for the CLI program: polylogue-5vps8.1 (migrate commands onto the
contract) and polylogue-5vps8.2 (delete direct-writer bypasses) are unblocked
by the write and control declarations plus §3's refusal, because "the CLI must
not open a writable tier" stops being a review rule and becomes an exception.
polylogue-9hgth and polylogue-vbsc0 consume the same table.

`polylogue/daemon/uds.py` currently serves the machine socket through the
browser HTTP handler, which bp12n.3's design excluded. Splitting it is part of
this work, not a separate concern.

## 5. Ingest

One pipeline, two threads of control, one decision point.

```
discover ─→ [ stage A: N threads ]              ─→ [ stage B: 1 thread ]
            read once, fingerprint from             drain prepared batches,
            those bytes, parse, build rows          one transaction per batch
                       │                                     │
                  bounded by bytes in flight           the write lease
```

**Stage A** reads each file exactly once and computes every hash the cursor
needs — raw fingerprint, prefix authority, semantic frontier — from the bytes
it just read. The re-reads it replaces are a *proof that the file did not
change between capture and cursor write*, so the single pass carries that proof
instead: capture `(inode, size, mtime_ns)` with the bytes and re-check the
identity, not the content, at cursor-write time. A changed file fails the
proof and is re-queued, exactly as a hash mismatch does today. It parses
and builds the row tuples — everything that is pure computation over a stable
read snapshot — and hands the writer a sealed SQLite shard (with parsed
sessions retained for governance) when the full-replace path can use one. The
shard has no secondary indexes or authority; Stage B attaches it read-only and
copies the `messages` and `blocks` ranges with `INSERT ... SELECT`. A worker
that dies before sealing leaves no admitted shard, and a stale or malformed
seal is refused and rebuilt inline. It is bounded by bytes in flight, not by
file count, so one whale cannot inflate memory and a thousand small files are
one batch.

**Stage B** is the only thread that touches the lease. It never waits on
parsing, network, or a future: it drains what stage A has already prepared.

**The decision point** is a single `BatchPolicy` function of queue depth and
queue age. There is no cold path and no live path in the code:

| Signal | Cold backlog | Live trickle |
| --- | --- | --- |
| queue deep, oldest item old | | |
| batch size | bytes-bounded, large | small, latency-bounded |
| connection profile | bulk-build | live writer |
| FTS | `bulk_build`: guarded during replay, repopulated once at the boundary | session-scoped delete-and-reinsert bracket, as today |
| archive-wide derived | deferred to the quiescence boundary | same |

Both ends of the table run the same functions with different parameters. The
policy is the only place that reads the difference.

**Fairness (polylogue-gmw2).** Intake is a deficit round-robin over domain
adapters — hook sidecars, browser spool, configured sources, Drive cache,
already-admitted raws — each providing stable item identity, bounded discovery
pages, and idempotent acknowledgement. A permanently failing item backs off to
a typed acquisition failure without preempting its class siblings; a class with
339k queued files cannot starve the others. Scheduling order, deficits and
discovery cursors are reconstructible hints; deleting them and restarting
resumes through bounded discovery without an eager whole-spool scan.

## 6. Convergence

Follows polylogue-bp12n.1's selected SELECT-HYBRID contract: domain adapters
expose `required` / `inspect` / `compute` / `publish`; the pending set is
`required` minus `valid`, derived from the authoritative output relation, never
from a queue, cursor or debt row. Compute runs outside the lease; publish
re-derives its input binding under `BEGIN IMMEDIATE` and returns pending on
mismatch.

What this design adds is the **scope split**, which is where the measured cost
is:

- **Per batch**: index rows, FTS membership for the sessions in the batch,
  session-scoped derived records. Bounded by the batch.
- **Per archive**: tag rollup, provider-usage rollup, global counts, FTS
  parity. These run at a **quiescence boundary** — when the intake queue
  drains, and once at the end of a cold build — never per batch, and **outside
  the write lease** except for the bounded publish.

`whole_archive` already expresses this for four stages; the design closes the
three leaks. `raw_parse_recovery` re-keys from source root to the batch's
sessions. `standing-queries` moves to the boundary. `tag_rollup_count` (142.6 s
over 567 invocations) leaves the per-session `derived` stage for the boundary.
The flag stops being all-or-nothing per chunk: scope is a property of the
derivation, not of the caller's mood. Together these remove the quadratic term
— the rehearsal measured it at a tenth of corpus scale.

Debt is the pending set, recomputed; `convergence_debt` rows in `ops.db` are
retry hints and carry no correctness authority.

### The aggregate families

There is one materialized aggregate family, keyed by `session_id`: the session
partition — `session_profiles` plus the `session_latency_profiles`,
`session_work_events` and `session_phases` rows written in the same
replacement. Its binding is `session_profiles.input_content_hash`, a digest over
the session-row and message projections declared in
`storage/derived/session/input_binding.py`; its recipe version is
`SESSION_INPUT_RECIPE_VERSION`. Inspection compares that digest and the
partition's sibling row counts, so a half-replaced partition and a partition
whose input values moved are both non-valid.

Threads, tag rollups and provider/day rollups are SQL views over that relation.
They have no output of their own to inspect and no partition to publish: they
are current exactly when the partitions feeding them are, which is how status
counts them. A freshness column on a view — `session_tag_rollups.materialized_at`
is the literal `'query-time'` — can only ever compare equal to itself.

Freshness is decided in exactly one place,
`storage/derived/session/derivation.py`. The archive-wide route and the
per-batch route call the same inspection over different key sets, and the
async route shares its SQL and classification with the sync one. Identity —
a sort key, an updated-at, a row count — narrows nothing and certifies nothing:
what a scope query may still answer is whether a partition was ever built, and
`SESSION_PROFILE_UNBUILT_CANDIDATES_SQL` answers only that.

## 7. Service lifecycle

polylogue-avmq's acceptance criteria are the runner design. One
`DaemonServiceSpec` registry at the composition root declares identity,
prerequisites, dependencies, trigger mode, readiness, failure policy, shutdown
deadline, status component and execution profiles. One supervisor is the sole
task owner: deterministic dependency order, no duplicate starts, declared
failure isolation, bounded cancel-and-await on shutdown with orphan
diagnostics. Focused tests select a minimal named profile from the *production*
registry, so a daemon test cannot silently start raw materialization.

An inventory test that instruments the production composition route and fails
on any spawned task absent from the registry is the anti-vacuity condition.

### Halted work is visible and unscheduled

The claude-code wedge (polylogue-kqrbw) was not one bug. A component reached a
terminal state, said so once in a log line, and every layer above it went on
planning, chunking and taking the writer lease for twenty minutes — while the
run looked healthy from outside. That is a missing property, and it is stated
here as one because it will recur in any component the supervisor owns.

**Property.** For every declared unit of work — a service, a source, an intake
class, a derivation domain — there is exactly one halt state, and it has two
consequences that are not optional and not separately implemented:

1. **Unschedulable.** A halted unit is excluded at the point work is
   *selected*, not refused at the point work is *executed*. Nothing downstream
   of a halt may acquire the write lease, form a batch, or consume a budget.
   Refusing late is the defect; a chunk that cannot succeed must never have
   been planned.
2. **Reported.** A halt is a durable state on the unit with its typed reason
   and the frame that produced it, published as that unit's status observation
   (§8). It is never only a log line. Compact status names every halted unit;
   a daemon with a halted unit is not `ok`.

The two are one state, read from one place. A halt that is visible but still
scheduled, or unscheduled but invisible, is a failure of the property, not a
partial implementation of it.

**Anti-vacuity, three mutations.** Put one source into terminal refusal and
assert zero subsequent lease acquisitions attributable to it — removing the
planning-time exclusion must make it red. Assert compact status names it and
reports not-`ok` — removing the observation must make it red. Assert the halt
survives a restart with its reason intact — demoting it to process-local state
must make it red.

This property belongs to the lifecycle contract, so polylogue-avmq owns it;
polylogue-kqrbw is one instance and closes under it rather than beside it.

## 8. Status and telemetry budget

polylogue-20d.17: each service publishes a bounded immutable last observation
with its frame identity, observed time and completion state. Compact status
composes those observations and performs no probe. Exact denominators are
explicit bounded read operations returning typed references. Readiness derives
from domain inspection (§6), not from queue rows, debt or snapshot age;
operation health is a separate projection and the two are never collapsed.

Telemetry budget: the per-batch receipt keeps the fields the rehearsal
attribution needed — offered/ingested/refused bytes, bytes read, parse CPU,
per-stage writer timings, queue depth and age, RSS — and nothing that requires
a scan. A status request must not recompute an FTS audit.

### The daemon reports the identity it started with

`derived_schema_identity` is computed from an AST closure over 442 of 1,241
modules, so it moves on ordinary parser, storage and archive edits, and the
shifts compose only on the merge: master, a branch, and that branch merged with
its siblings give three different values. A daemon started before a
closure-touching merge is therefore running against an identity that no longer
matches the checkout, and today the only way that becomes visible is when a
rebuild fails.

The daemon captures the identity at startup and publishes it as an ordinary
status observation, alongside the identity computed from the code it is
running now. Equal is the normal case and says nothing; unequal means the tree
moved underneath a running daemon, and status says so before a rebuild
discovers it. This is what would have made rehearsal-11's wedge legible in the
first minute rather than the fifth hour — the halt property (§7) reports that a
source stopped, and this reports *why* it was going to.

## 9. Stage 1: what the writer actually costs

Measured on the pinned worktree `/realm/worktrees/daemon-core-bench` at
`0f1d1cb09`, quiet host, 516 real sessions / 200.8 MB parsed through the
production route (`_live_parse_stage_candidates` → `live_parse_worker`) and
written through the production choke point (`ArchiveStore.write_parsed` →
`write_parsed_session_to_archive`). Each arm writes a fresh archive from empty.
Harness: `/realm/tmp/work/daemon-core/stage1_arms.py`.

| Arm | CPU s | Wall s | MB/s | DB bytes |
| --- | --- | --- | --- | --- |
| A baseline | 69.4 | 149.5 | 1.34 | 404,455,424 |
| B + memoized fingerprint closure | 37.9 | 111.8 | 1.80 | 404,455,424 |
| C + bulk-build pragmas | 70.7 | 71.9 | 2.79 | 414,044,160 |
| D + query-serving index deferral | 59.4 | 110.0 | 1.83 | 389,332,992 |
| **E all three** | **26.6** | **29.2** | **6.87** | 391,512,064 |
| A2 baseline repeat | 55.2 | 101.2 | 1.99 | 404,455,424 |

A2 is 20 % faster than A on the same arm, so page-cache drift is real and
single-arm deltas are read against A2, not A.

**Three findings, one of them unexpected.**

1. **Forty-five percent of writer CPU is the parser-fingerprint source
   closure.** `sources/origin_specs.py:244:_semantic_source_paths` is called
   per session write from `_fingerprint_sources` (`origin_specs.py:332`) and is
   not memoized, though its neighbours `_local_import_paths` (`:217`) and
   `_fingerprint_sources_cached` (`:288`) are. Every session write re-walks the
   parser import closure, rebuilding and sorting `Path` objects and issuing a
   `stat()` per closure member. Under cProfile the `pathlib` methods it drives
   — `_str_normcase`, `__init__`, `__str__`, `__eq__`, `__lt__`, `drive`,
   `__hash__`, `_format_parsed_parts`, `with_segments`, `_parse_path` — are
   about 38 % of writer time. Memoizing it (arm B) cuts CPU 69.4 → 37.9 s with
   byte-identical output size.
2. **The bulk-build pragma profile is a wall-time lever, not a CPU lever.**
   Arm C leaves CPU unchanged (69.4 → 70.7) and collapses wall 149.5 → 71.9.
   It buys fsync and WAL wait, not computation. This is why the profile must
   survive PR #4698: only its *selection site* (`archive.py:826`) is deleted
   with the rebuild engine, not the constant, so stage 4 re-homes it onto the
   batch policy and no hold on #4698 is required.
3. **Index deferral is not worth a schema change.** Arm D is −14 % CPU against
   A and slightly *worse* than A2; wall is unchanged, and the database shrinks
   3.7 %. Dropping the ten query-serving indexes on `blocks` and `messages`
   does not buy throughput. polylogue-9soj should not be pursued on performance
   grounds, and the wipe-day schema owner should not reduce the written index
   set for this reason.

Combined, the three give **3.5× on wall against A2, 5.1× against A**, and arm E
runs at 91 % CPU (26.6 s CPU / 29.2 s wall) — so after these fixes the writer
is genuinely CPU-bound again, and parallel row construction is what attacks
what remains.

## 10. Predicted throughput

Per GB of input, scaled from the measured 6.83 GB:

| Term | Today | After |
| --- | --- | --- |
| Parse CPU | 552 s/GB serial | 552 s/GB over ~12 effective threads = **46 s/GB**, overlapped with the writer |
| Fingerprint re-read | 0.6 GB extra read per GB | 0 |
| Writer stages | 225 s/GB | archive-wide derived removed (−21), FTS to bulk membership (−28), bulk profile −25 % on the rest: **132 s/GB** |
| Convergence per chunk | 83 s/GB | at the quiescence boundary, amortized |

With parse fully overlapped, the writer is the constraint at 132 s/GB:
**64.5 GB → 8,500 s ≈ 2.4 h**, against 15.7 h today. That is 6.5× from
structure alone, and it is what this design predicts on measured numbers.

It is not under an hour. Under an hour needs 56 s/GB, and the remaining
distance is inside the writer thread, where two levers are unsized:

- **Row construction moved out of the writer.** The writer is CPU-bound at
  1.4 MB/s of database growth, so its time is split between Python row
  construction and SQLite btree work. Only the second must be serial. If the
  split is even, moving construction into stage A gives 66 s/GB — about 72
  minutes — and the rest comes from batching.
- **Selective secondary-index deferral during a cold build**
  (polylogue-9soj). Seven indexes on `blocks` and eleven on `messages` are
  maintained on every insert. Blanket deferral measured +69 % worse because of
  the then-unindexed delete cascade; that cascade is now skipped
  (polylogue-cs86), so a deferral that keeps the `session_id`-scoped indexes
  and defers the query-serving ones is safe to measure again.

**The measurement that decides this** is the Python-versus-SQLite split of
writer time, taken by deterministic profiling of the real write route on a
quiet host. It is stage 1 of implementation. If the split is unfavourable —
most of the writer's time inside SQLite — then under an hour requires reducing
what is written (index schema), and that is an operator decision, not a
daemon-core one.

## 11. Stages

Each stage is a feature branch, lands green, keeps the daemon runnable, and is
re-measured on the same slice with MB/s recorded on polylogue-623q. Rehearsals
run from a pinned worktree; a benchmark against a moving checkout is not a
benchmark.

1. **Measure.** Done — §9.
2. **Memoize the fingerprint closure.** §9 finding 1. One-line-shaped change,
   −45 % writer CPU, byte-identical output. Lands first because it is the
   cheapest large win in the file.
3. **Write lease.** §3. Closes polylogue-8qm4k structurally: bounded holds, no
   `getattr` bypass, backup on the lease.
4. **Empty chunks and honest bytes.** polylogue-kqrbw (a terminally refused
   source stops being planned; a chunk that will ingest nothing never takes the
   lease) and polylogue-bsv8j (offered / ingested / refused bytes reconcile).
   Without these the benchmark cannot be trusted.
5. **Read once, parse ahead.** §5 stages A and B, bytes-in-flight bound.
   Removes the read amplification and the unoverlapped parse residual.
6. **Batch policy.** §5 decision point. Re-homes
   `BULK_BUILD_WRITE_CONNECTION_PROFILE` onto the policy — §9 finding 2 makes
   this the largest single wall-time lever.
7. **Convergence scope split.** §6 quiescence boundary; closes the three
   `whole_archive` leaks. Removes the quadratic term.
8. **Fair intake.** §5 dispatcher. Closes polylogue-gmw2.
9. **Supervisor and status.** §7, §8. Closes polylogue-avmq, polylogue-20d.17.
10. **Operation table.** §4 write/control/long-running declarations; unblocks
    the CLI program.

Stages 2 and 5–7 carry the throughput. Stages 3, 4 and 8–10 carry the contract.
Index deferral is not a stage: §9 finding 3 measured it as noise.

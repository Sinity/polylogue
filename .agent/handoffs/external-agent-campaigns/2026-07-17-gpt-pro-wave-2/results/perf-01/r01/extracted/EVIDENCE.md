# EVIDENCE — Budgeted status snapshots

## Authority and snapshot reconciliation

The attached archive's manifest identifies project `polylogue`, branch `master`, commit `536a53efac0cbe4a2473ad379e4db49ef3fce74d`, generated `2026-07-17T180950Z`, and `dirty=true`. The archive simultaneously contains zero-byte `polylogue-branch-delta.patch`, `polylogue-branch-delta-log.txt`, and `polylogue-branch-delta-files.txt`. Replaying the tracked working-tree overlay over the named commit produced no tracked source difference. The named commit is therefore the patch authority; the dirty flag most plausibly represents ignored/local state rather than an additional tracked implementation delta.

The exact commit subject is:

```text
536a53efac0cbe4a2473ad379e4db49ef3fce74d
fix(repair): harden raw authority convergence (#3046)
```

## Beads findings

### `polylogue-20d.17` — primary mission

The P1 record reports two independent live failures: `polylogued status` did not return within 15 seconds while heartbeat/DB descriptors were healthy, and coordination compact/detail reads measured 2.6–16.6 seconds. It identifies request-time mixing of millisecond facts with raw census, debt, embedding, Beads, process, archive and handoff probes as the root cause. Its acceptance criteria require one shared component protocol, per-component isolation, explicit freshness/last-good/deadline/detail metadata, source-keyed invalidation, bounded/cancellable/resumable exact diagnostics, sub-second product behavior and production-route mutation tests.

The later note supersedes `polylogue-703` into this mechanism and records a separate MCP failure: a 27,673-byte readiness assembly crossed the 25 KiB boundary while its summary contradicted underlying degraded/raw-frontier evidence. This motivated compact lifecycle metadata plus explicit detail references rather than retrying a whole report.

### `polylogue-703` — one assembly

This record documents three divergent status implementations: daemon status, CLI direct status and workload diagnostics. The implementation converges shared request behavior through `StatusComponentSpec`/`StatusSnapshot` and makes surfaces consumers. It intentionally does not delete all rich compatibility/diagnostic code; those deletions remain a separately certifiable residual.

### `polylogue-s7ae.8` — coordination latency

The record measured cold CLI compact 13.710 seconds, CLI detail 5.228 seconds, MCP compact 4.490 seconds and MCP detail 2.619 seconds, with a later refresh observing 13.2–16.6 seconds. It explicitly states that compact output size does not imply responsive collection and leaves source-keyed invalidation/randomized sampling incomplete. This implementation replaces whole-envelope collection with seven source-keyed component snapshots and keeps compact bytes below 8 KiB.

### `polylogue-20d.14` — interactive SLO contract

This record treats interactive latency as correctness and cites daemon CLI query p50 below 100 ms/p95 below 400 ms for a live UDS route. The current no-daemon cold subprocess p95 is 364 ms for `polylogued status` and 558 ms for the broader `polylogue` CLI startup path; warm in-process projections are below 1 ms p95. Live UDS validation on the operator deployment remains unverified.

### `polylogue-cuxz` — evidence semantics

This P1 record requires unknown, unavailable, stale and degraded states not be flattened into zero/null. The snapshot protocol keeps state, freshness, authority metadata, last-good evidence and current compatibility values separate. Direct mode therefore marks unavailable components explicitly instead of emitting healthy-looking defaults.

### `polylogue-feqr` — disappearing components

This closed record repaired an `except Exception: pass` cluster where failed components vanished. The shared registry generalizes the invariant: every declared selected component projects a lifecycle entry even before first collection or after exception/timeout.

## Source findings

### Existing daemon status

`polylogue/daemon/status.py` is a large rich collector combining runtime, storage, FTS, insights, raw materialization/frontier, live-ingest, convergence, failures, embedding and health facts. Earlier `daemon/status_snapshot.py` retained a whole payload, so an expensive refresh could still dominate and a single TTL/failure applied to the whole.

### Existing CLI status

`polylogue/cli/commands/status.py` contains its own direct SQLite/readiness/embedding/debt probes. The default command now returns before those legacy rich helpers and uses one bounded daemon request or descriptor-only direct projection. The helpers remain for separate diagnostics and `polylogue-703` cleanup.

### Existing coordination status

The old coordination builder collected repository, process tree, Beads, archive and handoff evidence before compact projection. The prior whole-envelope cache reduced some repeated cost but was TTL-oriented and still coupled unrelated stages. The new inventory refreshes seven independent fragments and derives overlap only from retained dependencies.

### Daemon periodic-loop idiom

`daemon/cli.py` already composes long-running periodic coroutines. The new one-second status scheduler follows that architecture. It calls daemon and coordination scheduler functions through `asyncio.to_thread`, then each registry starts bounded component worker threads. This separates fingerprint/collector work from the asyncio loop and all request handlers.

### MCP surfaces

Readiness tool/resource and embedding status previously had paths into rich readiness/embedding collection. They now consume `operations/status_client.py`. MCP call-log delivery retains outbox pressure under its dispatcher lock; exact directory scanning remains off-request. Agent coordination and the coordination brief prompt consume the daemon coordination snapshot.

### Exact diagnostics

`storage/archive_readiness.py` now exposes bounded resumable raw-materialization and raw-replay pages. `operations/archive_debt.py` exposes a work-budgeted archive-debt page and adds assertion-candidate offset paging to `storage/sqlite/archive_tiers/user_write.py`. The manual components preserve partial progress in memory and resume only under an unchanged fingerprint.

## History findings

Status/performance history relevant to the design:

```text
536a53efa fix(repair): harden raw authority convergence (#3046)
ef17859b3 fix(status): omit archive debt from periodic snapshots (#3002)
fa6691b61 fix(status): bound periodic readiness classification (#3001)
885f2938d fix(daemon): bound periodic status snapshots (#2999)
3556de6ed fix(cli): report failed status components as explicit unknown entries (#2959)
d0bc0a927 perf(cli,daemon): interactive SLO tier and fast paths (#2874)
b034a4bd7 perf: reduce interactive coordination latency (#2809)
81bfedd87 perf: improve interactive CLI and coordination latency (#2784)
de7f2b909 fix(coordination): compact agent status projections (#2656)
```

The prior fixes bounded individual whole-snapshot operations or reduced payloads. They did not establish independent lifecycle, deadline and invalidation policy per fact family. The current patch supplies that missing substrate.

## Final route map

```text
polylogue status
  -> operations.status_client.fetch_daemon_snapshot('/api/status')
  -> HTTP get_status_snapshot_payload()
  -> StatusSnapshotRegistry.project() + component_values()

polylogued status
  -> daemon.entrypoint lightweight Click dispatcher
  -> same bounded status client/direct projection

polylogue agents / MCP agent_coordination / prompt brief
  -> operations.status_client.coordination_snapshot_payload()
  -> HTTP get_coordination_snapshot_payload()
  -> coordination StatusSnapshotRegistry.project() + component_values()

MCP readiness/resource/embedding
  -> operations.status_client readiness/embedding projections
  -> daemon component metadata + retained compatibility values
```

The HTTP `fresh=1` branches call `request_status_component_refresh` or `request_coordination_refresh`, which start scheduler work and return immediately.

## Measured evidence

Final isolated no-daemon cold sampling:

```text
polylogued JSON: p50 251.667 ms, p95 363.913 ms, max 363.913 ms
polylogue JSON:  p50 453.822 ms, p95 558.476 ms, max 558.476 ms
```

Final 600-sample warm projections, seed `20017`:

```text
daemon:      p50 0.258899 ms, p95 0.352307 ms, max 0.392996 ms
coordination:p50 0.362561 ms, p95 0.682722 ms, max 0.722449 ms, 6,076 bytes
no-daemon:   p50 0.216778 ms, p95 0.880271 ms, max 1.132020 ms
```

Exact-base earlier comparison:

```text
base polylogued p95 2146.949 ms
base polylogue  p95 2536.968 ms
```

The package does not copy these raw evidence files; summarized measurements and reproduction commands are in `HANDOFF.md` and `TESTS.md`.

## Verification evidence

Implementation-tree focused lanes passed 76, 217, 154, 142 and 33 tests. The patch then clean-applied at `536a53efac0cbe4a2473ad379e4db49ef3fce74d` and a 146-test clean-tree smoke passed. Changed-file Ruff, strict mypy, byte compilation and diff whitespace checks passed. Generated equivalence/topology checks, degradation policy and test-clock hygiene passed.

The patch changes 47 files with 7,928 insertions and 666 deletions. `PATCH.diff` is a binary-capable unified Git diff so generated files with working-tree encodings apply exactly. It contains all new files in full and no copied archive, project-state bundle, mission attachment or delivery document.

## Contradictions and resolutions

- Manifest says dirty; tracked branch delta and replayed tracked overlay are empty. Resolution: use the named commit as authority and document the contradiction.
- The old compact coordination payload was byte-bounded but collection remained slow. Resolution: bound collection scheduling independently; keep byte projection as a separate contract.
- Existing whole snapshots were cached but stale invalidation was TTL-centric. Resolution: source fingerprints and dependency completion trigger refresh independent of TTL.
- MCP readiness needed outbox delivery evidence, but scanning the outbox in a request would violate the mission. Resolution: independent background component plus lock-only retained local fallback.
- Exact archive-debt code initially limited rendered rows but was not wired into the component. Resolution: collection work paging and component resume tests.

## Unverified external evidence

The operator's live daemon, archive, browser, MCP client, secrets, NixOS service and current checkout were not accessible. No claim is made that deployment restart, active-archive collector distributions or live web rendering were observed. Those checks belong after applying/installing the package in the operator environment.

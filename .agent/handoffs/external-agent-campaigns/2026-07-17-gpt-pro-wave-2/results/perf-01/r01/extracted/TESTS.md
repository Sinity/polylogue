# TESTS — Budgeted status snapshots

## Test strategy

The tests target production seams rather than mock-only scaffolding. Scheduler tests use injected collectors, clocks, thread starters and fingerprints. Route tests replace the former rich production dependency with a function that raises so a regression cannot pass merely because fixture data is small. CLI tests execute Click commands and rendering. HTTP and MCP tests exercise their real adapters. Benchmarks call the production projection functions after retained evidence is seeded.

The repository frozen-clock policy is respected. Scheduler time tests use `tests/infra/frozen_clock.py` or injected monotonic/wall clocks; no new test reads `time.time` directly.

## Production dependencies and anti-vacuity mutations

| Test area | Production dependency exercised | Representative mutation/removal that must fail |
|---|---|---|
| Shared scheduler lifecycle | `StatusSnapshotRegistry.refresh_due`, `refresh_component`, deadline expiry and projection | Allow a hanging worker to be joined by reads; start replacement threads after timeout; accept a late timed-out result; mark degraded evidence last-good. |
| Staleness and last-good | effective snapshot and component projection | Stop comparing monotonic age to `stale_after_s`; discard fresh last-good on exception/timeout; include current fresh value as redundant last-good. |
| Fingerprint/dependency invalidation | scheduler due calculation and resume fingerprint guard | Honor only TTL; run fingerprints inside projection; reuse a partial diagnostic after its source fingerprint changes. |
| Direct daemon status | `direct_status_snapshot_payload` | Open SQLite readiness/debt/embedding collectors or omit expensive component declarations. |
| Direct coordination | `_unavailable_coordination_payload` | Instantiate Git/process/Beads/archive collectors when the daemon is absent. |
| Daemon periodic wiring | `_periodic_status_snapshot_refresh` | Execute fingerprint scheduling on the asyncio loop instead of `asyncio.to_thread`; remove either daemon or coordination refresh. |
| HTTP status | `/api/status` and `/api/agents/coordination` handlers | Call `daemon_status_payload`/whole coordination builder; make `fresh=1` wait; omit documented cold-start fields. |
| CLI status | `polylogue status`, lightweight `polylogued` entry point and component renderer | Import full daemon composition for `polylogued status`; fall through to old direct rich helpers; omit mixed state/deadline/age/last-good/detail rendering. |
| MCP readiness | `readiness_snapshot_payload` and readiness tool/resource | Rebuild rich readiness; synchronously scan the MCP outbox; omit `mcp_call_delivery`; exceed compact boundary. |
| MCP embedding | `embedding_snapshot_payload` | Call exact embedding status on default/detail request instead of retained component evidence. |
| MCP/CLI coordination | `coordination_snapshot_payload` and typed envelope | Call synchronous `build_coordination_envelope`; lose compact omission counts/status metadata. |
| Exact raw materialization | `raw_materialization_readiness_page` and exact component | Ignore collection limit/cancellation/cursor; label partial evidence fresh. |
| Exact raw replay | `raw_materialization_replay_backlog_page` | Invoke the weighted unbounded replay planner; lose exact/partial candidate semantics. |
| Archive debt | `archive_debt_page` and manual component | Truncate only rendered rows; ignore candidate offsets/raw cursor; exceed per-page work budget; fail to forward previous progress. |
| Entry-point/import graph | lazy package roots and `polylogue.daemon.entrypoint` | Eagerly import archive/daemon composition before status dispatch; point `polylogued` back to `daemon.cli:main`. |
| Projection SLO | real daemon/coordination/direct assemblers | Reintroduce rich collection from a projection read or exceed the 8 KiB compact coordination contract. |

## Commands and results in the implementation tree

All commands were run from `/mnt/data/work_perf01/repo` at base HEAD `536a53efac0cbe4a2473ad379e4db49ef3fce74d` with the staged patch present.

### Static and generated checks

```text
ruff format --check <42 changed Python files>          PASS
ruff check <42 changed Python files>                   PASS
mypy --strict <23 changed production Python files>    PASS
python -m py_compile <42 changed Python files>         PASS
git diff --check                                      PASS
python -m devtools render all --check                  PASS
python -m devtools verify degrade-loudly              PASS (70 allowlisted, 0 new, 0 stale)
python -m devtools verify test-clock-hygiene           PASS (0 violations)
```

The `render all --check` command completed successfully in the implementation tree. One combined shell invocation later hit its external execution cap after the render phase; the two verification gates were rerun independently and passed.

### Shared protocol, status client, coordination, exact paging and benchmarks

```bash
pytest -q   tests/unit/core/test_status_snapshots.py   tests/unit/operations/test_status_client.py   tests/unit/coordination/test_status_snapshot.py   tests/unit/storage/test_archive_readiness.py   tests/unit/operations/test_archive_debt.py   tests/benchmarks/test_status_snapshots.py
```

Result: `76 passed in 2.80s`.

### Daemon, lightweight entry point and HTTP routes

```bash
pytest -q   tests/unit/daemon/test_daemon_status.py   tests/unit/daemon/test_daemon_cli.py   tests/unit/daemon/test_daemon_entrypoint.py   tests/unit/daemon/test_daemon_http_contracts.py   tests/unit/daemon/test_daemon_events_endpoint.py
```

Result: `217 passed in 13.25s`.

### CLI status routes and rendering

```bash
pytest -q   tests/unit/cli/commands/test_status.py   tests/unit/cli/test_status.py
```

Result: `154 passed in 8.65s`.

### MCP readiness, embeddings, coordination, call-log and surface contracts

```bash
pytest -q   tests/unit/mcp/test_agent_coordination.py   tests/unit/mcp/test_embedding_status_tool.py   tests/unit/mcp/test_envelope_contracts.py   tests/unit/mcp/test_mcp_call_log.py   tests/unit/mcp/test_server_surfaces.py
```

Result: `142 passed in 31.26s`.

### Entry-point and public operation contracts

```bash
pytest -q   tests/unit/test_entrypoints_runtime.py   tests/unit/operations/test_operation_contract.py   tests/unit/operations/test_specs.py   tests/unit/daemon/test_daemon_entrypoint.py   tests/unit/smoke/test_installed_scripts.py
```

Result: `33 passed in 4.13s`.

The daemon-entrypoint file is repeated between this lane and the daemon lane; results are reported per command rather than summed as unique tests.

## Clean-base application verification

`PATCH.diff` was checked and applied to `/mnt/data/work_perf01/applycheck-r01-final`, a detached worktree at `536a53efac0cbe4a2473ad379e4db49ef3fce74d`:

```text
git apply --check PATCH.diff       PASS
git apply PATCH.diff               PASS
git diff --check                   PASS
ruff format --check                PASS (42 files)
ruff check                         PASS
mypy --strict                      PASS (23 changed production files)
python -m py_compile               PASS
render mcp-equivalence --check     PASS
render topology-status --check     PASS
verify degrade-loudly              PASS
verify test-clock-hygiene          PASS
```

The clean-applied behavioral smoke command covered the new scheduler, client, coordination, archive paging/readiness, daemon entry point, HTTP routes, MCP call log/coordination/embedding and benchmark projections. The first combined run completed as `146 passed in 12.87s`. After regenerating the final binary-capable patch representation, a repeated combined shell invocation reached its external cap late in the run; splitting the identical set completed as `75 passed in 1.72s` and `71 passed in 11.90s` (146 total), with no failure.

A full `render all --check` in the clean worktree reached the site-render phase but exceeded the external command cap; the same full render had already passed in the implementation tree, and the changed generated targets passed directly in the clean-applied tree.

## Performance measurements

### Final isolated cold subprocess sampling

Environment: no daemon, isolated empty archive, loopback port with no listener, eight samples per command.

```text
polylogued status --format json:
  p50 251.667 ms
  p95 363.913 ms
  max 363.913 ms

polylogue status --json:
  p50 453.822 ms
  p95 558.476 ms
  max 558.476 ms
```

Base-commit comparison captured earlier with the same class of isolated environment:

```text
base polylogued status:
  p50 1731.175 ms
  p95 2146.949 ms
  max 2169.608 ms

base polylogue status --json:
  p50 2092.335 ms
  p95 2536.968 ms
  max 2670.370 ms
```

### Final warm projection distribution

The 600-sample run used seed `20017`, seeded mixed retained component states and the production projection functions. Direct mode used an isolated descriptor-only archive.

```text
daemon randomized selection:
  p50 0.258899 ms
  p95 0.352307 ms
  max 0.392996 ms
  serialized bytes 7,956

coordination randomized view/limit:
  p50 0.362561 ms
  p95 0.682722 ms
  max 0.722449 ms
  serialized bytes 6,076 (budget 8,192)

direct descriptor projection:
  p50 0.216778 ms
  p95 0.880271 ms
  max 1.132020 ms
  serialized bytes 15,120
```

These measurements test projection cost, not collector cost. That distinction is intentional: collector cost is isolated by per-component deadlines and is absent from request latency.

## Defects found during continuation and regression tests added

The continuation found that MCP readiness had lost the existing `mcp_call_delivery` field. The repair introduced a separately refreshed daemon `mcp_delivery` component and a local lock-only retained call-log snapshot. Tests monkeypatch the directory scanner to raise, proving readiness does not regress to request-time scanning.

The continuation also found fingerprint scheduling still ran on the daemon event loop. `_periodic_status_snapshot_refresh` now offloads both scheduler calls with `asyncio.to_thread`; a test records thread identities and fails if callbacks run on the event-loop thread.

Finally, archive-debt status had a resumable page implementation that was not wired to its component. The component now forwards collection limit, cancellation and previous progress. Tests create multiple real candidate assertions, assert `work_units_this_page <= limit` on every page, converge through resume, and separately prove pre-probe cancellation and component cursor forwarding.

## Unexecuted or externally blocked checks

- Operator live daemon/archive/browser/MCP-client and deployment restart checks: unverified because those resources were unavailable.
- Full unrestricted repository pytest: not run; focused affected lanes were used.
- Whole-tree strict mypy: not green because five errors remain in unchanged modules. Changed production files are green.
- The repository `devtools test` wrapper was unsuitable in this container because `/dev/shm` is 64 MiB; raw pytest was run with explicit `/tmp` basetemps. This does not change test semantics.

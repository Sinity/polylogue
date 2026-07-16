---
created: 2026-07-16
purpose: Top-down audit packet for daemon, convergence, HTTP, and concurrency tests
status: prepared-awaiting-real-sqlite-witness
project: polylogue
---

# Daemon and convergence

Generated dossier:
[`../dossiers/convergence-restart-retry.md`](../dossiers/convergence-restart-retry.md).

## Exact production route

- `polylogue/daemon/convergence.py:DaemonConverger`, `ConvergenceStage`;
- `polylogue/daemon/convergence_stages.py` stage checks/executors;
- `polylogue/sources/live/convergence_debt.py` durable debt operations;
- `polylogue/sources/live/convergence_debt_retry.py` retry selection;
- `polylogue/daemon/convergence_debt_status.py` public status projection.

The first survivor should be
`tests/unit/daemon/test_convergence_restart_law.py`, with any deterministic
barrier/failpoint helper isolated in
`tests/infra/convergence_harness.py`. Existing focused sources are
`tests/unit/daemon/test_daemon_convergence.py`,
`test_convergence_stages.py`, `test_convergence_final_state.py`, and
`test_convergence_debt_alert.py`. Notification/auth/process seam tests remain
out of subtraction scope.

## Scope and scale

`tests/unit/daemon` contains 65 Python files, about 27.0k nonblank lines, and
roughly 1,257 test/class declarations. The coupling census found 181 daemon
tests inspecting mock interactions. That is a review population, not a deletion
count: network, notification, clock, and process boundaries legitimately need
test doubles. The likely low-value cases mock Polylogue's own repository and
convergence implementation, then assert internal call order.

Partition the audit by temporal responsibility:

1. sole-writer/process ownership and handoff;
2. acquire/parse/materialize/index stage transitions;
3. convergence debt, bounded batches, quiet deferral, and retry;
4. restart/recovery and exactly-once/idempotent effects;
5. watcher/cursor/append concurrency;
6. HTTP authentication, error envelopes, and public read semantics;
7. health/status/metrics as projections of the same underlying state;
8. notification/external integrations.

## Deterministic temporal harness

Use real SQLite, the frozen clock, explicit barriers/failpoints, and a controlled
executor. A test sequence should advance events deliberately:

```text
source append → watcher observes → writer commits durable row
→ stage performs bounded work → debt recorded → restart
→ debt retries → archive quiesces
```

At each boundary, assert durable/public state. Avoid `sleep`, polling races, and
assertions that merely say helper X was awaited once. Provide a compact event
trace on failure so stronger tests remain diagnosable.

High-value state laws:

- interrupted and uninterrupted executions converge to the same facts;
- a stage returning “pending” records remaining work without reporting failure;
- repeated retry/restart never duplicates material or loses debt;
- still-hot input defers bounded work, then converges after the frozen quiet
  window;
- concurrent source events are serialized by the one writer while reads see a
  valid old or new state, never a torn state;
- health/status/metrics agree on terminal and degraded state;
- adding irrelevant backlog does not cause unbounded work in a session-scoped
  retry.

## HTTP and surface consolidation

For stable daemon endpoints, prefer an in-process real HTTP application over a
semantic archive and assert public JSON/error/auth behavior. Keep mocks only at
remote notification/auth/process boundaries. One planted-fact route can replace
separate repository-response fabrication, endpoint forwarding, and response
shape tests when it retains explicit security/error cases.

The Schemathesis pilot belongs here for stable OpenAPI routes. It should not be
applied to the current web reader, which is a rewrite boundary.

## First implementation slice

Choose convergence debt across bounded work → restart → retry → quiescence.
Build one deterministic real-SQLite scenario, compare final facts with an
uninterrupted run, and mutate debt recording/session scoping/retry removal.
Then inspect which mock call-order tests it dominates. This is more likely to
find composition failures than broad endpoint-by-endpoint cleanup.

Required sensitivity temporarily removes or corrupts debt persistence,
session-scoped retry selection, `false_means_pending`, and retry execution. The
new law must fail on durable/public final state, not a mocked call count.

Permitted worker command:

```bash
devtools test tests/unit/daemon/test_convergence_restart_law.py
```

Deletion candidates are named only after coverage/history evidence proves that
the survivor retains pending, restart, retry, idempotency, scoping, and
diagnostic obligations.

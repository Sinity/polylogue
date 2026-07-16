Title: "[testdiet 02] Convergence restart and debt liveness"

Job ID: `testdiet-02`
Result ZIP: `testdiet-02-convergence-liveness-r01.zip`

## Mission

Implement a survivor law for daemon convergence across interruption, restart,
retryable debt, and eventual quiescence. Use the current active-growing or
partial-convergence workload variant and real `DaemonConverger`/ops debt/write
routes. Demonstrate that bounded work may defer honestly, durable debt survives
process restart, retries address the intended session/stage without replaying
unrelated acquisition, and repeated convergence reaches the same terminal
facts without duplication.

Prefer deterministic barriers/failpoints or existing injected clock/executor
seams over sleeps. If one minimal production trace seam is genuinely needed,
make it report the real stage/debt lifecycle rather than mirror the algorithm
inside tests. The independent oracle should cover debt identity, retry count,
terminal state, materialized facts, and cleanup.

Name a mutation such as dropping the debt write, consuming debt before stage
success, skipping the retry path, or re-resolving the wrong source, and show
why the survivor fails. Propose dominated restart/example tests but do not
delete them. Do not build a new convergence state machine or corpus generator.

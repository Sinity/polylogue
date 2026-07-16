---
created: 2026-07-16
purpose: Decide installed capture delivery, service lifecycle, component status, and runtime evidence for L29-L30
status: recommended-decision
project: polylogue
---

# Capture delivery and deployed status

## Decision

Describe installed runtime composition once as a production `DaemonServiceSpec`
graph. The daemon launcher and `ServiceHarness` both instantiate that graph with
explicit enabled components, prerequisites, owned/borrowed resources, startup
readiness, shutdown order, and time/resource budgets. The harness does not own a
parallel test registry.

Capture delivery is a durable, idempotent queue/receipt state machine. Status is
an off-request composition of independently refreshed component snapshots bound
to running build, deployment, daemon run, archive, and host evidence. Unknown or
unavailable host evidence remains explicit rather than inferred.

## Runtime identity

`RuntimeIdentity` contains:

- exact build identity from packaged/version metadata, not checkout `HEAD`;
- deployment/installation identity and service profile;
- stable daemon `run_id` and process start evidence;
- archive id and resolved config snapshot digest;
- host/boot identity where locally available and disclosure-allowed.

The existing `polylogue/version.py` build identity and daemon lifecycle UUID are
the starting mechanisms. PID alone is not identity and may be reused.

## Production service graph

Each `DaemonServiceSpec` declares:

- component id, factory, dependencies, enable predicate, and readiness probe;
- resource ownership (`owned` or `borrowed`) for executors, servers, watchers,
  DB handles, receiver, and temp/spool paths;
- startup and shutdown deadlines/order;
- failure propagation and degraded-mode policy;
- status component collector and evidence fingerprint;
- privacy/disclosure class.

Partial startup unwinds only resources it owns, in reverse dependency order.
Repeated start/stop is bounded and leaves no threads, sockets, processes,
readers, or temporary artifacts. Tests consume the same graph with injected
fixture resources and clocks.

## Capture delivery state

The durable state machine is:

```text
discovered -> eligible -> leased -> captured -> receiver_acked -> ingested
            \-> paused/backoff/auth_required/blocked/failed
```

An ACK proves durable receiver spool acceptance and includes request id, exact
content hash, receiver id, extension instance id, provider/native identity, and
contract version. It does not prove archive ingestion. Ingestion produces a
separate raw/acquisition receipt; queryability/derived freshness are later
checkpoints.

Leases use owner identity/generation and explicit recovery, not TTL-only
completion. Provider throttling, auth challenge, receiver down, native empty,
no turns, budget exhaustion, and contract drift are distinct states. Backfill
and live capture share the receipt/delivery vocabulary while retaining their
own scheduling policy.

## Component status snapshots

Complete the existing whole-payload `daemon/status_snapshot.py` cache into the
`StatusComponentSpec`/`StatusSnapshot` contract from `polylogue-20d.17`.
Each component declares collector, dependencies, cost/detail class, deadline,
event/fingerprint invalidation, staleness policy, privacy, and projection.

The off-request scheduler refreshes components independently and retains:

- `fresh`, `stale`, `refreshing`, `timed_out`, `unavailable`, or `degraded`;
- observed/start/finish times and monotonic age;
- last-good `EvidenceValue`, failure reason, and evidence/detail ref;
- source fingerprint/config/runtime identities;
- collection work and resource receipt.

Compact CLI/MCP/HTTP status reads only snapshots and cannot synchronously run
archive-scale probes. Exact replay, debt, embedding, Beads, archive, or handoff
diagnostics are explicit cancellable/resumable detail queries under the shared
query execution contract.

Overall status is a declared composition of required components. `ok=true`
cannot coexist with a required blocking/error component merely because the
payload was truncated.

## Termination receipts

On normal shutdown, record signal/reason, stage, work/debt snapshot, cleanup,
and terminal time under the daemon run id. Because SIGKILL/OOM/power loss cannot
run `finally` or `atexit`, the next start runs an external reconciler:

- correlate the prior run with systemd unit result, journal/cgroup/kernel OOM
  evidence, heartbeat cessation, boot identity, and workload/resource samples;
- record every observation with source and time confidence;
- distinguish `observed` cause, `inferred` hypothesis, and `unknown`;
- never promote temporal coincidence to exact cause;
- link residual capture/convergence/checkpoint debt.

Host evidence collection is local, bounded, redacted, and optional. If access
is unavailable, termination cause stays unknown; normal product operation is
not blocked.

## Competitive alternatives

| Alternative | Advantage | Why not chosen |
| --- | --- | --- |
| Test-only service harness | Fast to build | Drifts from production wiring and proves its own registry |
| PID plus health endpoint | Simple | Cannot bind a response to build/deployment/archive/run or explain reuse/restart |
| Assemble rich status inline | Always newest | One slow collector blocks every status request and can exhaust resources |
| Whole-payload TTL cache | Minimal change | One stale/expensive component dominates; source changes hide until TTL |
| ACK means fully archived/queryable | Simple UX | Collapses spool, ingest, index, and derivation durability |
| Atexit-only termination record | Easy | Cannot observe the failures most worth explaining |
| Infer OOM from missing heartbeat | Often plausible | Conflates kill, crash, host restart, suspension, and storage stalls |
| Require host telemetry | Rich diagnosis | Makes portable/local operation fail when optional evidence is absent |

## Migration sequence

1. Extract the production service graph from current daemon wiring and build the
   harness as a consumer.
2. Keep current capture queue contracts; add explicit checkpoint/receipt
   projection through queryability.
3. Split the existing status snapshot into independent component collectors,
   preserving current payload fields through a compatibility projection.
4. Bind status to runtime/build/archive/config identities.
5. Add next-start termination reconciliation and optional host evidence.
6. Finish packaged multi-profile receiver proof; retain real-host canaries
   outside deterministic fixture tests.

## Required proof

- production launcher and harness instantiate the same component graph;
- every partial-start and shutdown boundary releases exactly owned resources;
- service-worker/daemon/restart permutations preserve or idempotently drain
  capture delivery without duplicate logical effects;
- stalled collectors cannot delay compact status and source changes invalidate
  only affected components;
- status from a stale build/archive/run is detected even when PID/port are live;
- injected graceful, signal, crash, OOM-evidence, host-reboot, and no-evidence
  terminations produce correct observed/inferred/unknown receipts;
- mutation of ACK durability, identity binding, component invalidation, or
  resource cleanup fails real installed-route tests.

Primary evidence: `polylogue-enj7`, `polylogue-20d.17`, `polylogue-s8q`,
`polylogue-peo`, `polylogue-jlme.1`, `polylogue-jlme.5`, `polylogue-3v1`;
daemon lifecycle/status modules, browser-capture receiver and extension queues,
`polylogue/version.py`, and ops lifecycle storage.

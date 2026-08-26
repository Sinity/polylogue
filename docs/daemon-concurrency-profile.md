# Daemon concurrency profile

This is the checked architecture record for `polylogue-bp12n.4`. The daemon
uses CPython 3.14t with free-threading enabled, one bounded in-process compute
adapter, one writer coordinator, and one archive-scoped typed operation
exchange over AF_UNIX. Reads execute sequentially by default; no result memo
cache is part of the initial architecture.

The managed profile is:

```text
devtools bench daemon-operation
```

It runs the installed CLI and the direct `polylogue.daemon-operation/v1`
request route. The workload manifest covers cold and warm status, find/read,
static and archive-backed completion, concurrent reads, cancellation,
incremental ingest, derivation catch-up, and inactive-candidate construction.
The route tests record protocol identity/readiness/authority, result state,
queue admission, runtime, and rows/bytes where the production route exposes
them. Host CPU/RSS and process startup remain benchmark observations, not
correctness keys.

Predeclared budgets are p50/p95: daemon operation 100/400 ms, installed warm
CLI 400/800 ms, cold status 500/800 ms, static completion 80/150 ms,
archive-backed completion 150/300 ms, first useful cell 250/500 ms, and
cancellation 250/500 ms. These rows are informational until the import-light
launcher and complete background workloads are production routes.

The current managed profile on 2026-08-26 recorded typed status at 6.1285 ms
median, typed find/read at 7.6313 ms, static/live completion at 2.4288/2.4925
ms, four concurrent reads at 34.1286 ms, cancellation-route completion at
199.2206 ms, and installed warm status at 1,398.8718 ms. These are descriptive
host measurements; they do not claim the cold-status target is met.

## Deletion ledger

This lane relocates query admission and cancellation into
`polylogue/daemon/execution.py`, and protocol identity into
`polylogue/operations/daemon_protocol.py`. It removes the CLI fast-path
health/probe dependency and replaces the measured legacy UDS benchmark with
the typed operation profile.

| disposition | gross LOC | detail |
| --- | ---: | --- |
| deleted | 61 | Gross removed lines in the tracked diff; the browser HTTP surface remains needed by the web UI and the old health probe is no longer on the CLI measured path. |
| relocated | 0 | New kernel/protocol seams are additive in this lane; the old adapter remains for compatibility routes and tests. |
| added | 1259 | 493 tracked additions plus 766 lines in eight new files, including tests/docs/tooling. |
| net maintained | 1198 | Added minus gross deleted; no physical file relocation was claimed. |

The sharded rebuild implementation is not deleted in this lane because its
current callers and dedicated provenance laws remain live in this checkout;
the packet's historical K=4/K=8 result is recorded as a rejected optimization,
not as evidence that those callers are abandoned. A follow-up deletion must
remove its option, callers, and oracle together.

# Runtime and concurrency contract

Polylogue runs on CPython 3.14 free-threading. Every shipped entrypoint fails before archive or network work when the interpreter is older, non-CPython, or has the GIL enabled. The same assertion is exposed through `devtools verify runtime`, and the devshell checks both its interpreter and the venv it creates. Required native packages are imported as part of that check: `sqlite_vec`, `nh3`, and `watchfiles`. The optional `msgspec` accelerator is checked when installed and is never allowed to use an incompatible build.

Use `devtools verify runtime-census` for the production import census. It reports executor construction, SQLite connection sites, synchronization primitives, caches, registries, and process globals under the daemon, CLI, MCP, API, pipeline, source, storage, and operations packages. Every item has one of the following dispositions: immutable, thread-confined, lock-protected, transactionally-serialized, or intent-passing.

Use `devtools bench concurrency --json` for the managed profile. It exercises the production `BoundedComputeAdapter` and `DaemonWriteCoordinator` over tiny-file, ordinary, whale, mixed-ingest, derivation, and interactive-read workloads. The result records worker/admission settings, throughput, p50/p95/p99 latency, queue count and bytes, CPU utilization, writer hold, RSS, cancellation count, rejection count, and background progress. Worker counts are derived from available CPU and remain bounded. The profile rejects an unbounded queue and a worker count above the CPU-derived bound.

The writer contract is unchanged: worker threads may parse, hash, decode, prepare, or read, but they do not own SQLite mutation connections. Daemon mutations enter `DaemonWriteCoordinator`; publication must occur after the caller's current generation and input binding are revalidated. Results from a stale generation are discarded before publication.

## Deletion ledger

| disposition | result |
| --- | --- |
| retained | one runtime identity contract, one daemon bounded compute adapter, one daemon writer-intent coordinator, and the existing process-confined parser path remain reachable production routes |
| removed | no dual GIL/free-threaded package switch or compatibility runtime flag was added |
| rejected | an unbounded executor queue and CPU-unbounded worker profile are rejected by the managed admission contract |
| residual | process-confined ingestion and validation executors remain separate because their current production call sites and measured C-extension behavior have not been replaced by this proof slice |

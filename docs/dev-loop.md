# AgentCTL development-loop proof

The browser-capture development proof is a declared Polylogue AgentCTL operation. AgentCTL binds the job to the registered worktree and exact starting HEAD, allocates the API and receiver ports, starts and stops the systemd service cgroup, enforces the 15-minute deadline, handles cancellation, releases the lease, and retains the bounded JSON result.

Create or use a managed workspace, then start the fixed operation:

```bash
agentctl workspace create polylogue browser-proof --branch feature/browser-proof
agentctl job start polylogue dev_loop_proof --workspace <workspace-id>
agentctl job wait <job-id>
agentctl job result <job-id>
```

The operation accepts no parameters. In particular, callers cannot choose a port, service command, environment overlay, readiness probe, timeout, or process-control policy.

Polylogue's project command receives `POLYLOGUE_API_PORT` and `POLYLOGUE_BROWSER_CAPTURE_PORT` only from the lease. It creates a disposable archive in AgentCTL's NVMe scratch space, proves unauthenticated receiver rejection and authenticated capture acceptance, runs the deterministic ChatGPT and Claude fixture capture, waits for archive materialization, and reads the captured messages through the API. Its one result object reports only the two allocated ports, receiver-auth verdict, provider names, and archive/API convergence flags. Paths, tokens, process identifiers, and raw captures stay out of the result.

The operation is intentionally finite. There is no Polylogue PID file, port allocator, systemd control path, generic process-tree terminator, service lease, or launcher-status API. Do not start `polylogued` from a branch with an ad hoc port pair. Use the AgentCTL job receipt for lifecycle and status.

## Product-level deterministic smoke

`devtools workspace dev-loop-service` is the fixed project command executed by the declared operation. It is not an operator-facing host-control command. The receiver-auth smoke remains a deterministic in-process product check, covered by `tests/unit/devtools/test_dev_loop_service.py`; the operation composes it with the actual browser/provider and archive/API route.

For a manually focused check, use the managed test harness:

```bash
devtools test tests/unit/devtools/test_dev_loop_service.py
```

Live provider pages and copied browser profiles remain operator-local investigation workflows. They are not a substitute for the declared deterministic proof and do not create an alternative Polylogue service lifecycle.

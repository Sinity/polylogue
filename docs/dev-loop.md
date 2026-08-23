# AgentCTL development-loop proof

The browser-capture development proof is a declared Polylogue AgentCTL operation. AgentCTL binds the job to the registered worktree and exact starting HEAD, allocates the API, receiver, and Chrome CDP ports, starts and stops the systemd service cgroup, enforces the 15-minute deadline, handles cancellation, releases the lease, and retains the bounded JSON result.

Create or use a managed workspace, then start the fixed operation:

```bash
agentctl workspace create polylogue browser-proof --branch feature/browser-proof
agentctl job start polylogue dev_loop_proof --workspace <workspace-id>
agentctl job wait <job-id>
agentctl job result <job-id>
```

The operation accepts no parameters. In particular, callers cannot choose a port, service command, environment overlay, readiness probe, timeout, or process-control policy.

Polylogue's private project command receives `POLYLOGUE_API_PORT`, `POLYLOGUE_BROWSER_CAPTURE_PORT`, and `POLYLOGUE_BROWSER_CDP_PORT` only from the lease. It validates the fixed descriptor, AgentCTL project/operation/job identity, managed-workspace exact head, and all three leased port ranges before it creates a disposable archive in AgentCTL's NVMe scratch space. It then proves unauthenticated receiver rejection and authenticated capture acceptance, runs deterministic ChatGPT and Claude fixture capture, waits for archive materialization, and reads the captured messages through the API. Its result reports only the allocated ports, receiver-auth verdict, provider names, and archive/API convergence flags. Paths, tokens, process identifiers, and raw captures stay out of the result.

The operation is intentionally finite. There is no Polylogue PID file, port allocator, systemd control path, generic process-tree terminator, service lease, or launcher-status API. Do not start `polylogued` from a branch with an ad hoc port pair. Use the AgentCTL job receipt for lifecycle and status.

## Product-level deterministic smoke

The fixed Python module is intentionally absent from the public devtools command catalog. Sinnixd executes it only through the declared `dev_loop_proof` operation after exact-head revalidation. Chrome stays in that service cgroup, uses the leased CDP port, receives a graceful CDP close request when the proof completes, and relies on systemd for final cancellation and descendant cleanup. The receiver-auth smoke remains a deterministic in-process product check, covered by `tests/unit/devtools/test_dev_loop_service.py`; it does not prove AgentCTL or systemd behavior.

For a manually focused check, use the managed test harness:

```bash
devtools test tests/unit/devtools/test_dev_loop_service.py
```

Live provider pages and copied browser profiles remain operator-local investigation workflows. They are not a substitute for the declared deterministic proof and do not create an alternative Polylogue service lifecycle.

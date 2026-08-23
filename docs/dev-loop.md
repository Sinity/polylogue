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

Polylogue's private project command receives `POLYLOGUE_API_PORT`, `POLYLOGUE_BROWSER_CAPTURE_PORT`, and `POLYLOGUE_BROWSER_CDP_PORT` from the lease. Sinnixd, not this checkout, authorizes the job, revalidates the managed workspace and exact head, allocates the ports, and owns the transient systemd cgroup. Polylogue only rejects ordinary shell invocation when that expected context is absent, then creates a disposable archive in AgentCTL's NVMe scratch space. It proves unauthenticated receiver rejection and authenticated capture acceptance, runs deterministic ChatGPT and Claude fixture capture, waits for archive materialization, and reads captured messages through the API. Its result reports only the allocated ports, receiver-auth verdict, provider names, and archive/API convergence flags. Paths, tokens, process identifiers, and raw captures stay out of the result.

The operation is intentionally finite. There is no Polylogue PID file, port allocator, systemd control path, generic process-tree terminator, service lease, or launcher-status API. Do not start `polylogued` from a branch with an ad hoc port pair. Use the AgentCTL job receipt for lifecycle and status.

## Product-level deterministic smoke

The fixed Python module is intentionally absent from the public devtools command catalog. Sinnixd executes it through the declared `dev_loop_proof` operation after its own exact-head revalidation. Chrome stays in that service cgroup, uses the leased CDP port, receives a graceful CDP close request when the proof completes, and relies on systemd for final cancellation and descendant cleanup. The receiver-auth and mocked convergence tests cover product wiring only. They do not prove Sinnixd authorization, a systemd cgroup, lease allocation, or descendant cleanup. The coordinator-owned AgentCTL receipt is the live proof for those facts.

For a manually focused check, use the managed test harness:

```bash
devtools test tests/unit/devtools/test_dev_loop_service.py
```

## Copied-profile live-provider investigation

The product helper remains private at `browser-extension/scripts/live_provider_proof.mjs`. It can only be imported by a future fixed AgentCTL operation. It requires typed coordinator-supplied inputs for a copied, unlocked profile directory, a receiver URL and token, a fixed Chrome binary, a descriptor-leased CDP port, disposable output, and an explicit provider list. It refuses common live profile roots and emits a redacted summary without transcript text.

The current descriptor accepts no typed credentials or copied-profile input, so there is deliberately no public npm script, devtools command, or direct host-control compatibility route. The coordinator must add that typed operation and retain its AgentCTL receipt before a live invocation. This helper is not a substitute for the deterministic proof and does not create an alternative Polylogue service lifecycle.

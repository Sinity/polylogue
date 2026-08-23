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

Polylogue's private project command receives `POLYLOGUE_API_PORT`, `POLYLOGUE_BROWSER_CAPTURE_PORT`, and `POLYLOGUE_BROWSER_CDP_PORT` from the lease. Sinnixd, not this checkout, authorizes the job, revalidates the managed workspace and exact head, allocates the ports, and owns the transient systemd cgroup. Before it starts polylogued, Node, or Chrome, Polylogue confirms both that `/proc/self/cgroup` names the matching `sinnixd-job-<UUID>.service` in `agent.slice` and that the user manager reports the same job, project, and operation for that unit. This rejects a direct shell with forged environment variables. It does not replace Sinnixd's exact-head or lease authority. The command then creates a disposable archive in AgentCTL's NVMe scratch space. It proves unauthenticated receiver rejection and authenticated capture acceptance, runs deterministic ChatGPT and Claude fixture capture, waits for archive materialization, and reads captured messages through the API. Its result reports only the allocated ports, receiver-auth verdict, provider names, and archive/API convergence flags. Paths, tokens, process identifiers, and raw captures stay out of the result.

The operation is intentionally finite. There is no Polylogue PID file, port allocator, systemd control path, generic process-tree terminator, service lease, or launcher-status API. Do not start `polylogued` from a branch with an ad hoc port pair. Use the AgentCTL job receipt for lifecycle and status.

## Product-level deterministic smoke

The fixed Python module is intentionally absent from the public devtools command catalog. Sinnixd executes it through the declared `dev_loop_proof` operation after its own exact-head revalidation. Chrome stays in that service cgroup, uses the leased CDP port, receives a graceful CDP close request when the proof completes, and has a local process-group termination fallback for every launcher failure path. Systemd remains the outer cancellation and descendant-cleanup authority. The receiver-auth and mocked convergence tests cover product wiring, and the launcher test proves local process-group cleanup. They do not prove Sinnixd authorization, a systemd cgroup, lease allocation, or the coordinator's outer cleanup. The coordinator-owned AgentCTL receipt is the live proof for those facts.

For a manually focused check, use the managed test harness:

```bash
devtools test tests/unit/devtools/test_dev_loop_service.py
```

## Copied-profile live-provider proof

`live_provider_proof` is a separate declared AgentCTL operation. Its internal Node implementation has no exported launcher and accepts no caller-selected executable, profile, token, receiver, output path, port, provider list, or timeout. The service fixes Chrome resolution, the unpacked extension in its registered checkout, both providers, a copied profile at `/realm/state/polylogue/live-provider-proof-profile`, scratch output, and a fresh in-service receiver token. The descriptor leases the CDP and receiver ports. It refuses common live browser profile roots and emits a redacted summary without transcript text.

The operation is the only live invocation route. There is no npm script, devtools command, or direct host-control compatibility route. Its receiver is scoped to the operation and is not an alternative Polylogue daemon lifecycle. The focused tests prove local rejection and process-group cleanup. A completed Sinnixd receipt remains the only live proof of its exact-head authorization, lease allocation, unit creation, and daemon-owned lifecycle.

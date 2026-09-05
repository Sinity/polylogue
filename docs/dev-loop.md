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

Polylogue's private project command receives only `POLYLOGUE_API_PORT` and `POLYLOGUE_BROWSER_CAPTURE_PORT` from the lease. The Sinnix runtime (agentctl), not this checkout, authorizes the job, revalidates the managed workspace and exact head, allocates the ports, and owns the transient systemd cgroup. Before it starts polylogued or Node, Polylogue confirms both that `/proc/self/cgroup` names the matching `agentctl-job-<UUID>.service` (or `sinnixd-job-<UUID>.service` on older hosts) in `agent.slice` and that the user manager reports the same job, project, and operation for that unit. This rejects a direct shell with forged environment variables. It does not replace the runtime's exact-head or lease authority. The command creates a disposable archive in AgentCTL's NVMe scratch space, asks Sinnix's control boundary to load the unpacked extension, opens one `about:blank` window parked on the named `agentbrowser` workspace, and closes only that returned target. It proves unauthenticated receiver rejection and authenticated deterministic ChatGPT and Claude capture acceptance, waits for archive materialization, and reads captured messages through the API. Its result reports only the allocated ports, receiver-auth verdict, shared-Chrome verdict, provider names, and archive/API convergence flags. Paths, tokens, process identifiers, raw captures, browser profiles, and CDP ports stay out of the result.

The operation is intentionally finite. There is no Polylogue PID file, port allocator, systemd control path, generic process-tree terminator, service lease, or launcher-status API. Do not start `polylogued` from a branch with an ad hoc port pair. Use the AgentCTL job receipt for lifecycle and status.

## Product-level deterministic smoke

The fixed Python module is intentionally absent from the public devtools command catalog. The runtime executes it through the declared `dev_loop_proof` operation after its own exact-head revalidation. Its Node child can invoke only the installed `sinnix-chrome-control` boundary for the one existing Chrome at `127.0.0.1:9222`; it cannot launch Chrome or Chromium, create a profile, or allocate a CDP port. The Node cleanup is bounded and addresses only the target ID returned by `agent-window`. Systemd remains the outer cancellation and descendant-cleanup authority. The receiver-auth, shared-control, and mocked convergence tests cover product wiring. They do not prove runtime authorization, a systemd cgroup, lease allocation, or the coordinator's outer cleanup. The coordinator-owned AgentCTL receipt is the live proof for those facts.

For a manually focused check, use the managed test harness:

```bash
devtools test tests/unit/devtools/test_dev_loop_service.py
```

## Shared-Chrome live-provider proof

`live_provider_proof` is a separate declared AgentCTL operation. Its internal Node implementation has no exported launcher and accepts no caller-selected browser executable, profile, token, receiver, output path, port, provider list, or timeout. It uses the installed `sinnix-chrome-control` boundary against the already-running authenticated Chrome at `127.0.0.1:9222`. The proof runtime-loads the unpacked extension, configures a fresh in-service receiver token, and opens one agent window for each fixed provider. `agent-window` parks and verifies every window hidden on `agentbrowser`; proof capture is correlated with the resulting CDP target and browser window, so it addresses no operator tab. Receiver settings are restored and cleanup closes only those created target IDs. The descriptor leases only the receiver port and emits a redacted summary without transcript text.

The live proof's Node workflow has a 90-second total deadline, the Python child wait is 120 seconds, and the AgentCTL operation deadline is 180 seconds. Timeout failures terminate the Node process group and return a bounded typed JSON result.

The operation is the only live invocation route. There is no npm script, devtools command, or direct host-control compatibility route. Its receiver is scoped to the operation and is not an alternative Polylogue daemon lifecycle. The focused tests prove local rejection and process-group cleanup. A completed AgentCTL receipt remains the only live proof of its exact-head authorization, lease allocation, unit creation, and daemon-owned lifecycle.

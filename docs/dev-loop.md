# AgentCTL development-loop proof

The browser-capture development proof is a declared Polylogue AgentCTL operation. AgentCTL binds the job to the registered worktree and exact starting HEAD, starts and stops the systemd service cgroup, enforces the 15-minute deadline, handles cancellation, and retains the bounded JSON result. The daemon binds both loopback listeners to port zero and retains their sockets. It atomically publishes actual bound addresses to a unique private startup file; the proof verifies the child PID, then uses those ports for API and receiver convergence and reports them in the bounded result. Child exit during startup fails immediately.

Create or use a managed workspace, then start the fixed operation:

```bash
agentctl workspace create polylogue browser-proof --branch feature/browser-proof
agentctl job start polylogue dev_loop_proof --workspace <workspace-id>
agentctl job wait <job-id>
agentctl job result <job-id>
```

The operation accepts no parameters. In particular, callers cannot choose a port, service command, environment overlay, readiness probe, timeout, or process-control policy.

The Sinnix runtime (agentctl), not this checkout, authorizes the job, revalidates the managed workspace and exact head, and owns the transient systemd cgroup. Before it starts polylogued or Node, Polylogue confirms that the runtime exported a job id for this project and this operation and that `/proc/self/cgroup` is under the declared pool's slice, `agentctl-interactive.slice` (`sinnixd-pueue-interactive.slice` on older hosts). This rejects a direct shell with forged environment variables. The command creates disposable state under the job's temporary directory, isolates XDG configuration from operator settings, creates an independently keyed page-only proof extension and independently named native manifest, loads that extension through Sinnix's control boundary, and opens its owned proof page on `agentbrowser`. No background worker or provider content script is registered. The proof verifies exact upload/response bytes through the actual native host, closes its target, uninstalls its extension, and removes only its owned native manifest and launcher. It proves unauthenticated receiver rejection and authenticated deterministic ChatGPT and Claude capture acceptance, waits for archive materialization, and reads captured messages through the API. Its result reports the selected ports, receiver-auth verdict, shared-Chrome verdict, provider names, and archive/API convergence flags. Paths, tokens, process identifiers, raw captures, browser profiles, and CDP ports stay out of the result.

The operation is intentionally finite. There is no Polylogue PID file, port allocator, systemd control path, generic process-tree terminator, service lease, or launcher-status API. Do not start `polylogued` from a branch with an ad hoc port pair. Use the AgentCTL job receipt for lifecycle and status.

## Product-level deterministic smoke

The fixed Python module is intentionally absent from the public devtools command catalog. The runtime executes it through the declared `dev_loop_proof` operation after its own exact-head revalidation. Its Node child can invoke only the installed `sinnix-chrome-control` boundary for the one existing Chrome at `127.0.0.1:9222`; it cannot launch Chrome or Chromium, create a profile, or allocate a CDP port. The Node cleanup is bounded and addresses only the target ID returned by `agent-window`. Systemd remains the outer cancellation and descendant-cleanup authority. The receiver-auth, shared-control, and mocked convergence tests cover product wiring. They do not prove runtime authorization, a systemd cgroup, lease allocation, or the coordinator's outer cleanup. The coordinator-owned AgentCTL receipt is the live proof for those facts.

For a manually focused check, use the managed test harness:

```bash
devtools test tests/unit/devtools/test_dev_loop_service.py
```

## Shared-Chrome live-provider proof

`live_provider_proof` takes `--conversations-file /absolute/private/conversations.json`, a private JSON array containing exact `https://chatgpt.com/c/<id>` or `https://claude.ai/chat/<id>` URLs, with one selected conversation per provider. Homepages are rejected before browser mutation. Start it through `agentctl job start polylogue live_provider_proof --workspace <checkout> -- --conversations-file <private-file>`; do not put conversation URLs or tokens in job arguments.

This operation currently refuses with `provider_target_isolation_unavailable`
before starting a receiver, loading an extension or changing native manifests.
A full-runtime proof copy would register provider content scripts on operator
tabs before an automatic-capture pause. The pause does not prevent the ChatGPT
MAIN bridge from staging responses or background activation from inspecting
existing tabs. Restoring this route requires script registration and capture
authority restricted to owned provider targets. It must also use a unique
extension key/ID and independently named host with neutral credentials; the
operator's fixed native manifest and extension settings remain untouched.

The page-only `dev_loop_proof` establishes actual Chrome nativePort transport
and byte conservation; it does not establish provider acquisition or automatic
capture lifecycle. Literal parser/artifact controls remain independent of this
currently refused browser invocation.

Native canonical preparation also writes fixed best-effort markers through the
existing private bounded background debug log. The capture observes only new
log rows for its acquired raw reference and the exact original MAIN native
fetch request ID after registration and a new `normalize_admission BEGIN`.
Retained rows, foreign request IDs and missing or malformed correlation are
ignored; there is no wall-clock fallback. These entries carry
`source: background_debug_log` in the sanitized
`native_progress` report, distinct from the two-field native-fetch markers.
The original reference and request ID remain private and carry no receipt or
mutation authority. Repeated admission of the same request is ambiguous:
observed overlap discards that background evidence, and absent or lost
markers leave the boundary unknown. Abort and settlement remove the observer.
Diagnostics do not await storage writes or certify successful effects.

| Background marker | Awaited interval |
| --- | --- |
| `normalize_admission` | Original invocation, receiver trust and throttle admission |
| `native_prepare` | Receiver preparation entry through raw prefix/member preparation and retained summary |
| `native_assets` | Exact asset-plan iteration, acquisition and receipts |
| `native_finalize` | Owner refresh and final native artifact seal |

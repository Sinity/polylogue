# AgentCTL development-loop proof

The browser-capture development proof is a declared Polylogue AgentCTL operation. AgentCTL binds the job to the registered worktree and exact starting HEAD, starts and stops the systemd service cgroup, enforces the 15-minute deadline, handles cancellation, and retains the bounded JSON result. The daemon binds both loopback listeners to port zero and retains their sockets. It atomically publishes actual bound addresses to a unique private startup file; the proof verifies the child PID, then uses those ports for API and receiver convergence and reports them in the bounded result. Child exit during startup fails immediately.

Create or use a managed workspace, then start the fixed operation:

```bash
agentctl workspace create polylogue browser-proof --branch feature/browser-proof
agentctl job start --project /realm/project/polylogue/repo --workspace <workspace-path> dev_loop_proof -- --chrome-user-data-dir <actual-running-Chrome-user-data-directory>
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

`live_provider_proof` takes `--conversations-file /absolute/private/conversations.json`,
a private JSON array of exact `https://chatgpt.com/c/<id>` or
`https://claude.ai/chat/<id>` URLs, with one selected conversation per provider.
Pass `--chrome-user-data-dir` for the actual shared Chrome profile and
`--evidence-root` for a nonexistent private per-run directory. Homepages
are rejected before browser mutation. Conversation URLs and credentials stay
out of job arguments and public reports.

The proof starts its own neutral receiver and independently named native host,
then creates a fresh extension key and ID. Its manifest has no static content
scripts. The Node runner opens the declared windows first and seals their
actual window/tab identities before loading the guarded production runtime.
Tab lookup, events, scripting and capture effects are restricted to those
owned documents. `currentWindow` means the first declared owned window.
Automatic capture is enabled only within that authority; explicit captures
use the same owner. Reports verify admitted artifact bytes against the exact
selected native identity and the canonical parser, and record the isolation
and loaded-resource bindings. This route does not prove archive convergence.

On completion, error or cancellation, the runner closes its owned windows
and unloads its extension before the Python owner removes its scoped host,
neutral receiver and private scratch. The sealed selected spool, original
constructor/runtime bindings and per-file byte hashes remain in the evidence
directory on successful and failed settled exits. The operator's extension, settings,
fixed native manifest and provider tabs remain untouched. The page-only
`dev_loop_proof` remains the separate nativePort byte-conservation proof.

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

The scoped native manifest is created only under the explicitly bound running Chrome user-data directory, in `NativeMessagingHosts`. A custom `--user-data-dir` changes that lookup location; the proof never publishes a second manifest to the default profile.

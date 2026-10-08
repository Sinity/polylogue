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

The Sinnix runtime (agentctl), not this checkout, authorizes the job, revalidates the managed workspace and exact head, and owns the transient systemd cgroup. Before it starts polylogued or Node, Polylogue confirms that the runtime exported a job id for this project and this operation and that `/proc/self/cgroup` is under the declared pool's slice, `agentctl-interactive.slice` (`sinnixd-pueue-interactive.slice` on older hosts). This rejects a direct shell with forged environment variables. The command creates disposable state under the job's temporary directory, isolates XDG configuration from operator settings, asks Sinnix's control boundary to load the unpacked extension, opens one `about:blank` window parked on the named `agentbrowser` workspace, and closes only that returned target. It proves unauthenticated receiver rejection and authenticated deterministic ChatGPT and Claude capture acceptance, waits for archive materialization, and reads captured messages through the API. Its result reports the selected ports, receiver-auth verdict, shared-Chrome verdict, provider names, and archive/API convergence flags. Paths, tokens, process identifiers, raw captures, browser profiles, and CDP ports stay out of the result.

The operation is intentionally finite. There is no Polylogue PID file, port allocator, systemd control path, generic process-tree terminator, service lease, or launcher-status API. Do not start `polylogued` from a branch with an ad hoc port pair. Use the AgentCTL job receipt for lifecycle and status.

## Product-level deterministic smoke

The fixed Python module is intentionally absent from the public devtools command catalog. The runtime executes it through the declared `dev_loop_proof` operation after its own exact-head revalidation. Its Node child can invoke only the installed `sinnix-chrome-control` boundary for the one existing Chrome at `127.0.0.1:9222`; it cannot launch Chrome or Chromium, create a profile, or allocate a CDP port. The Node cleanup is bounded and addresses only the target ID returned by `agent-window`. Systemd remains the outer cancellation and descendant-cleanup authority. The receiver-auth, shared-control, and mocked convergence tests cover product wiring. They do not prove runtime authorization, a systemd cgroup, lease allocation, or the coordinator's outer cleanup. The coordinator-owned AgentCTL receipt is the live proof for those facts.

For a manually focused check, use the managed test harness:

```bash
devtools test tests/unit/devtools/test_dev_loop_service.py
```

## Shared-Chrome live-provider proof

`live_provider_proof` takes `--conversations-file /absolute/private/conversations.json`, a private JSON array containing exact `https://chatgpt.com/c/<id>` or `https://claude.ai/chat/<id>` URLs, with one selected conversation per provider. Homepages are rejected before browser mutation. Start it through `agentctl job start polylogue live_provider_proof --workspace <checkout> -- --conversations-file <private-file>`; do not put conversation URLs or tokens in job arguments.

The Python operation holds the existing `POLYLOGUE_ARCHIVE_ROOT` authority at its private receiver root before server construction through request-thread and server shutdown, so status and attestation identities and their locks stay private. The private server owns non-daemon request handlers and joins them on close before restoring that scope or removing its directory. Successful teardown restores the prior process scope; failed teardown retains the directory and scope until process exit. The selected proof declines native credential refresh as well as canonical receiver recovery.

The internal Node implementation has no exported launcher and accepts no caller-selected browser executable, profile, token, receiver, output path, port, or timeout. It uses installed `sinnix-chrome-control` against authenticated Chrome at `127.0.0.1:9222`, loads the candidate extension, and verifies its actual installed manifest and source bytes. An owned extension page pauses automatic capture and configures and pairs the fresh standalone receiver through the extension's ordinary handshake. The proof grants only its temporary loopback host permission when needed, then opens exact selected conversations in owned windows hidden on `agentbrowser`. Before the popup opens, the proof captures the original compositor window and active workspaces on its monitors. It binds only its disposable popup to the original CDP target, then restores the captured workspace on the prompt’s monitor and the original operator focus as soon as the optional-permission request settles, including refusal or cancellation. The compositor checks the same owned popup and original window within the focus transaction; a newer independent user focus is preserved. Each provider window creation first checks that no monitor shows `agentbrowser`, matching the installed window helper’s hidden-workspace requirement. Capture is correlated with each resulting CDP target and browser window; operator tabs are untouched.

Success requires the actual receiver artifact digest and canonical message, branch, and full block fields to agree with literal native evidence through the existing parser. The thin browser summary alone is insufficient. The result exposes counts and hashes without transcript text. The private receiver artifacts are removed when the operation ends; this route does not exercise archive convergence or qualify scalar-independent preparation. Cleanup is registered before permission or receiver changes. The proof requests only the declared optional exact receiver origin from its owned popup using Chrome’s user-gesture permission API, matching the popup Save action. Preexisting permission is retained; cleanup removes only permission newly acquired by this proof through `chrome.permissions.remove`. An owned permission request must pass the same `chrome.permissions.contains` check used by receiver settings before configuration starts; a denied postcondition or the known settings refusal reports `receiver_permission_refused` and still settles the owned cleanup. Signal and normal cleanup share the same mutation settlement: an in-flight grant finishes before its owned permission is removed, and a concurrent receiver-settings change refuses restore without skipping permission removal. The same lifecycle retains any in-flight owned-window creation until its response registers the target, then closes only those created targets. Shutdown refuses later opening and capture phases. Automatic capture remains paused. The unpacked extension remains installed from the candidate path, which must remain available while it is loaded. Automatic-scheduler and attachment behavior still need their own live evidence before automatic capture can resume.

The live proof's Node workflow has a 90-second total deadline, the Python child wait is 120 seconds, and the AgentCTL operation deadline is 180 seconds. Timeout failures terminate the Node process group and consume its final strict phase/error/cleanup report together with the original receiver request observations. Unparseable or absent child output reports an unknown phase and cleanup states; it never substitutes stderr or claims cleanup settled. The same strict failure shape retains observations for unsuccessful or malformed child results.

Failures expose a fixed `error.phase` and `error.category`, a bounded `native_progress` array of fixed stage and `BEGIN`/`END` markers, with receiver restore, permission removal, mutation settlement, and owned-target cleanup states (`not_required`, `settled`, `failed`, or `unknown`). An original window-helper visibility refusal reports `window_visibility_refused`; its other fixed placement refusals report `window_refused`, while unrecognized helper failures retain the generic category. Stderr is consumed only to recognize that installed fixed refusal and is never retained or published. The original CDP exception preserves only exact whitelisted proof errors; unrelated exception descriptions and stacks never enter the report. The private receiver also reports fixed method, path and status observations for status and attestation requests, including preflight, after its request handlers settle. The failure report also retains `capture_evidence` for each already returned supported provider response: a fixed terminal category, the unique `page_bridge_fetch` acceptance and numeric HTTP status (or unknown), the original caught Claude acquisition boundary (`admission`, `provider_fetch`, or `staging`, otherwise `unknown`), and booleans for every existing summary acceptance check. Missing, malformed, deferred and rejected responses remain distinguishable; arbitrary response errors are unknown. The caught boundary identifies the original await that began unwinding; it does not identify the exception cause or certify subsequent cleanup. These fields contain no conversation identity, body, headers or URL and do not prove artifact admission or canonical parity. Python accepts only this exact diagnostic shape and vocabulary from a nonzero child; malformed or private-bearing output produces a sanitized generic failure. A signal retains the main phase active before cancellation; cancellation completing a capture cannot advance that phase to summary. Signal cleanup is reported separately through the receiver, permission, mutation and target custody states. Failure exits stay nonzero, including when cleanup succeeds. A failed receiver mutation retains its original operation category when normal cleanup re-awaits it; the separate cleanup states record physical restoration and removal.

ChatGPT records progress synchronously in the existing six-entry native-attempt ring owned by the original capture. MAIN staging, provider authentication, provider response and body markers use the original native-fetch response channel and request ID. A progress frame cannot settle that response. Cancellation freezes the last pre-cancellation markers; marker `END` means the await returned, including a refusal, and never certifies capture success. Missing markers mean unavailable evidence. No URL, credential, body, exception or transcript field is accepted by the strict proof progress validators.

| Last unmatched `BEGIN` | Await being observed |
| --- | --- |
| `throttle` | Original provider throttle response |
| `pending_header` | Previously observed native header preparation |
| `restore` | Original retained native lookup |
| `staging` | Durable acquisition admission before provider traffic |
| `provider_auth` | Existing page authentication lookup |
| `provider_response` | Original provider response headers |
| `body` | Original response body and staging seal |
| `header` | Acquired native header preparation |
| `canonical` | Canonical receiver envelope preparation |
| `publication` | Original capture delivery response |

The operation is the only live invocation route. There is no npm script, devtools command, or direct host-control compatibility route. Its receiver is scoped to the operation and is not an alternative Polylogue daemon lifecycle. The focused tests prove local rejection and process-group cleanup. A completed AgentCTL receipt remains the only live proof of its exact-head authorization, unit creation, and daemon-owned lifecycle.

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

# Branch-Local Development Loop

Use this workflow when working on the daemon web shell, daemon HTTP routes,
browser-capture receiver, or browser extension from a feature branch. The goal
is to debug the branch you are editing without deploying it systemwide or
pointing it at the production archive by accident.

## Preflight

Run the workspace preflight first:

```bash
devtools workspace dev-loop
devtools workspace dev-loop --json
```

It reports:

- the current checkout path, branch, and commit;
- a deterministic branch-local run id for correlating daemon, browser, and
  receiver artifacts;
- whether the user `polylogued.service` is active;
- listeners on the daemon API port and browser-capture receiver port;
- the branch-local archive root, run log directory, daemon log, browser
  artifact directory, terminal/TUI artifact directories, and preflight JSON
  path to use;
- a concrete `polylogued run` command with branch-local environment variables.

The default branch-local paths are:

```text
.local/dev-archive/
.cache/dev-loop/
```

Create them explicitly when needed:

```bash
devtools workspace dev-loop --prepare
```

`--prepare` creates the branch-local archive root, the run-specific browser
artifact directory, and writes the preflight payload to:

```text
.cache/dev-loop/<run-id>/preflight.json
```

These paths are local development state. Do not commit archive data, logs,
browser profiles, cookies, screenshots, or receiver payloads from them.

## Service Separation

The simplest safe posture is to stop the deployed user service while running a
branch-local daemon:

```bash
systemctl --user stop polylogued.service
```

If you keep the deployed service running, use different ports and verify the
preflight output before starting the branch daemon:

```bash
devtools workspace dev-loop --isolated-ports --prepare
devtools workspace dev-loop --api-port 8876 --browser-capture-port 8875
```

`--isolated-ports` asks the dev-loop helper to allocate currently-free loopback
ports and writes those exact ports into `preflight.json`, generated browser/TUI
plans, and the copy-pastable daemon/CLI commands. Use explicit `--api-port` /
`--browser-capture-port` when you need stable numbers for a longer debugging
session.

Then launch the branch-local daemon with the managed helper:

```bash
devtools workspace dev-loop --isolated-ports --launch-daemon
devtools workspace dev-loop --api-port 8876 --browser-capture-port 8875 --launch-daemon
```

The helper refuses to start if the selected API or browser-capture port already
has a listener. It starts `polylogued run --no-watch` in a separate process
group, writes `polylogued.pid`, `polylogued.env.json`,
`polylogued.launch.json`, `dev-loop.events.jsonl`, and the daemon log into the
run-local directory, and waits briefly for both the API and receiver ports to
become reachable. The important variables are:

```bash
POLYLOGUE_ARCHIVE_ROOT=.local/dev-archive
POLYLOGUE_API_PORT=8876
POLYLOGUE_BROWSER_CAPTURE_PORT=8875
POLYLOGUE_DEV_LOOP_RUN_ID=<run-id>
POLYLOGUE_DEV_LOOP_LOG_DIR=.cache/dev-loop/<run-id>
```

The JSONL event stream records the launcher-side lifecycle in order:

```text
launch_requested
process_spawned
readiness_succeeded | readiness_failed
```

If the selected ports are already occupied, the helper writes a
`launch_rejected` event before exiting. Each event carries the run id, checkout
path, branch, commit, archive root, status, timestamp, and a small event
payload. For a port-blocked launch, this event stream is the primary artifact:
the daemon process is never spawned, so `polylogued.launch.json` and
`polylogued.log` are not expected to exist. When the process does spawn, use the
event stream together with `polylogued.launch.json` and `polylogued.log` to
diagnose whether a branch-local failure happened during API/receiver readiness
or inside the daemon itself.

Once the daemon passes its schema preflight, it also writes durable
`daemon.lifecycle` rows to the existing `ops.db` daemon event ledger. These
rows carry the same dev-loop run id when `POLYLOGUE_DEV_LOOP_RUN_ID` is set and
record startup, API/browser-capture binding, FTS readiness, converger/watcher
start, skipped watcher state, and shutdown. Use the launcher JSONL stream for
process-spawn/readiness questions and `/api/events?kind=daemon.lifecycle` or
the ops event reader for daemon-internal lifecycle questions. Schema-blocked
startup intentionally does not write these rows, because that path must remain
observable without opening archive databases.

## Web Shell Debugging

The branch-local web shell is served by the branch-local `polylogued run`
process. Open the URL printed by the preflight, usually:

```text
http://127.0.0.1:8766/
```

Managed branch-local daemons also expose:

```text
GET /api/dev-loop
```

The route uses the same bearer-if-configured API posture as the other daemon
JSON routes. It returns only allowlisted debug fields: dev-loop run id, run log
directory, archive root, selected API/browser-capture ports, daemon PID, and
current working directory. It does not dump process environment or secrets. The
web shell reads this endpoint and shows a compact `dev:` status chip only when
`POLYLOGUE_DEV_LOOP_RUN_ID` or `POLYLOGUE_DEV_LOOP_LOG_DIR` is present.

For web shell work, use the run-local daemon log so UI errors can be correlated
with route logs and request failures. The development loop should make these
facts obvious before an agent starts debugging:

- which checkout started the daemon;
- which run id ties together the daemon log, browser artifacts, and receiver
  observations;
- which archive root the daemon serves;
- which API port the browser should open;
- where daemon logs are written;
- whether the deployed service is still active.

The web shell `dev:` chip and `/api/dev-loop` endpoint expose the run id and log
directory from the daemon process. The launcher-side `dev-loop.events.jsonl`
then gives agents a stable local artifact to correlate that UI-visible run id
with the process spawn and readiness state that created it.

The shell also records the latest API request it made. Every request sent by
the shell carries an `X-Request-ID` header, measures client-side duration, and
updates the `api:` status chip with success latency or failure status. Hover the
chip to see the route, request id, status, duration, any echoed response request
id, and a bounded response-body summary. This is intentionally a local UI-debug
aid: use it with browser network logs and daemon logs to identify which route
handled the visible action, without storing raw archive payloads in a separate
ledger.

## CLI and TUI Capture

Human-facing CLI/TUI behavior should be debuggable as rendered, not only as
JSON payloads. The preflight reserves run-local directories for those artifacts:

```text
.cache/dev-loop/<run-id>/terminal/
.cache/dev-loop/<run-id>/tui/
```

Use the managed capture helper for normal command-line runs:

```bash
devtools workspace dev-loop --capture-cli -- polylogue ops status
devtools workspace dev-loop --capture-cli -- polylogue analyze insights profiles --limit 5
```

The helper runs the command with the branch-local `POLYLOGUE_*` environment and
writes stdout, stderr, a combined transcript, the relevant environment snapshot,
and a JSON summary under:

```text
.cache/dev-loop/<run-id>/terminal/
```

It also appends `cli_capture_requested` and `cli_capture_finished` rows to the
same `dev-loop.events.jsonl` stream used by daemon launches. Those rows carry
the command, timeout, exit code, duration, timeout flag, and artifact paths, so
a run directory has one timeline covering the branch-local daemon and any CLI
captures performed against it.

The preflight payload still includes `commands.capture_cli_status` as a
copy-pastable `script` example when an actual terminal typescript is useful.

For TUI or full-screen terminal work, record into the run-local `tui/`
directory using the local terminal-control surface, `script`, or VHS-style
recording when visual playback is needed. Polylogue only provides the
branch-local environment and artifact locations; visual inspection and terminal
control remain local agent/operator capabilities.

Generate the run-local visual inspection plan with:

```bash
devtools workspace dev-loop --tui-plan
devtools workspace dev-loop --tui-plan --json
```

The plan writes:

```text
.cache/dev-loop/<run-id>/tui/tui-plan.json
.cache/dev-loop/<run-id>/tui/tui-plan.md
.cache/dev-loop/<run-id>/tui/polylogue-status.tape
.cache/dev-loop/<run-id>/tui/polylogue-status.gif
.cache/dev-loop/<run-id>/tui/screenshots/
```

It records the branch-local `POLYLOGUE_*` environment, a `script` command for a
plain CLI transcript, and a starter VHS cassette for visual playback. Edit the
cassette or drive the terminal through the local control plane when the target
flow is richer than `polylogue ops status`; keep generated recordings under the
ignored run-local directory.

## Browser-Capture Extension Development

Load `browser-extension/` unpacked into a development Chrome profile and point
the extension at the receiver URL printed by the preflight. Polylogue does not
provide or verify the browser-control substrate itself; local agents/operators
use their existing browser tooling to open the branch-local web shell, inspect
DOM/network state, and load the unpacked extension. For authenticated ChatGPT
or Claude.ai adapter work, an operator-approved copy of the real browser
profile may be used on this workstation only.

Copied-profile rules:

- copy the profile into `.local/browser-profiles/` or another ignored local
  path;
- restrict permissions to the local user;
- never use copied profiles in CI or cloud agents;
- delete the copy after the debugging session if it is no longer needed;
- do not commit profile data, cookies, screenshots, or raw captured content.

The extension loop should verify both failure and success cases:

- receiver unavailable;
- missing or invalid auth token when auth is configured;
- valid capture POST accepted by the branch-local receiver;
- service-worker logs show the receiver URL, status, and last error;
- content adapters expose diagnostic state without dumping raw conversation
  text by default.

Before involving a real browser, run the deterministic receiver smoke:

```bash
devtools workspace dev-loop --receiver-smoke
devtools workspace dev-loop --receiver-smoke --json
```

The smoke starts an in-process loopback receiver on an ephemeral port, configures
auth, verifies that an unauthenticated capture is rejected, verifies that an
authenticated synthetic capture is accepted, and reports the written spool
artifact. It uses the branch-local run directory and does not talk to the
deployed `polylogued.service`.

For service-worker-to-receiver coverage, run the extension smoke:

```bash
devtools workspace dev-loop --extension-smoke
devtools workspace dev-loop --extension-smoke --json
devtools workspace dev-loop --browser-plan --extension-smoke --json
```

This starts a temporary local browser-capture receiver, imports the actual
extension background worker with a small Chrome API mock, sends the same
runtime messages that popup/content scripts use, proves unauthenticated capture
rejection, configures the branch-local receiver token, checks receiver status,
and posts a deterministic capture envelope. It appends
`extension_smoke_requested` / `extension_smoke_finished` rows to
`dev-loop.events.jsonl` and writes stdout, stderr, a summary JSON, redacted
environment snapshot, and the receiver spool artifact under:

```text
.cache/dev-loop/<run-id>/browser/
```

This is the CI-safe branch-local proof for the extension background HTTP path.
It does not need authenticated ChatGPT/Claude.ai cookies and does not claim to
prove GUI content-script injection. Use the same run id and receiver settings
when you then load `browser-extension/` unpacked into an agent/private Chrome
profile for real-page inspection.
Combine it with `--browser-plan` when you want one run id and browser artifact
directory containing both the synthetic extension proof and the real-browser
handoff plan.

For a local proof that real Chrome can load the unpacked extension and reach the
branch-local receiver from the extension-origin service-worker context, run:

```bash
devtools workspace dev-loop --browser-smoke
devtools workspace dev-loop --browser-smoke --json
devtools workspace dev-loop --browser-plan --extension-smoke --browser-smoke --json
```

This launches `google-chrome-stable` in headless mode with
`--load-extension=browser-extension`, discovers the Polylogue MV3 service worker
over Chrome DevTools Protocol, reads the extension manifest, starts a temporary
branch-local receiver, then sends unauthenticated and authenticated receiver
requests from the extension service-worker context. It writes
`browser_smoke_requested` / `browser_smoke_finished` rows to
`dev-loop.events.jsonl` plus:

```text
.cache/dev-loop/<run-id>/browser/browser-smoke.json
.cache/dev-loop/<run-id>/browser/browser-smoke-result.json
.cache/dev-loop/<run-id>/browser/browser-smoke-profile/
.cache/dev-loop/<run-id>/browser/browser-smoke-spool/
```

This still does not claim authenticated ChatGPT/Claude.ai DOM adapter coverage.
It proves the repo-owned unpacked-extension + real-browser + receiver plumbing.
Use `--browser-plan` for the visible/private browser handoff when content-script
diagnostics, screenshots, copied-profile cookies, or real provider pages are
needed.

For GUI/browser inspection, generate a branch-local browser plan:

```bash
devtools workspace dev-loop --browser-plan
devtools workspace dev-loop --browser-plan --json
```

The plan does not launch or certify an agent browser. It writes the local
control-plane handoff under `.cache/dev-loop/<run-id>/browser/`:

```text
browser-plan.json
browser-plan.md
chrome-profile/
screenshots/
downloads/
```

The JSON and Markdown plans include:

- the unpacked extension path (`browser-extension/`);
- the branch-local receiver URL and whether a local receiver token is set;
- the branch-local web-shell URL;
- a `google-chrome-stable --user-data-dir=... --load-extension=...` command;
- a `chromium` variant of the same command;
- supported real-page probe URLs for `chatgpt.com` and `claude.ai`;
- screenshot/download artifact directories for the browser-control layer.

Use this when the local agent/operator browser-control surface is going to load
the unpacked extension, inspect popup/options state, inspect service-worker
logs, capture screenshots, or test authenticated ChatGPT/Claude.ai pages with
an operator-approved copied profile. Copied profiles remain local ignored
artifacts; they are never CI/cloud inputs and are not Polylogue repo state.

## Run Inspection

After a branch-local run has produced artifacts, summarize the run directory:

```bash
devtools workspace dev-loop --inspect-run .cache/dev-loop/<run-id>
devtools workspace dev-loop --inspect-run .cache/dev-loop/<run-id> --json
```

The summary reads `preflight.json`, `dev-loop.events.jsonl`, daemon launch
summaries, browser/extension/TUI plans, and terminal capture summaries. It
reports event counts by status/surface, the last event, failed or blocked
events, failed summary files, failed terminal captures, slowest timed events,
missing core artifacts, and an artifact index grouped by surface. Use it as the
first stop when a daemon/web/extension loop looks stuck: it tells you which
surface failed, which command or smoke was slow, and which stdout/stderr/log
artifact to open next instead of making you scan the whole run directory.

## Current Boundary

`devtools workspace dev-loop` owns branch-local preparation and local debug
artifacts. It can launch a branch-local daemon, run synthetic receiver and
extension smokes, write browser/TUI handoff plans, capture CLI commands, and
summarize a run directory. It still does not start or stop the deployed
`polylogued.service`; stopping production service state remains an explicit
operator action.

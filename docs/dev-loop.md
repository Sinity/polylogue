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
devtools workspace dev-loop --api-port 8876 --browser-capture-port 8875
```

Then run the command printed by the preflight. The important variables are:

```bash
POLYLOGUE_ARCHIVE_ROOT=.local/dev-archive
POLYLOGUE_API_PORT=8876
POLYLOGUE_BROWSER_CAPTURE_PORT=8875
```

## Web Shell Debugging

The branch-local web shell is served by the branch-local `polylogued run`
process. Open the URL printed by the preflight, usually:

```text
http://127.0.0.1:8766/
```

For web shell work, keep the daemon output tee'd into `.cache/dev-loop/` so UI
errors can be correlated with route logs and request failures. The development
loop should make these facts obvious before an agent starts debugging:

- which checkout started the daemon;
- which run id ties together the daemon log, browser artifacts, and receiver
  observations;
- which archive root the daemon serves;
- which API port the browser should open;
- where daemon logs are written;
- whether the deployed service is still active.

## CLI and TUI Capture

Human-facing CLI/TUI behavior should be debuggable as rendered, not only as
JSON payloads. The preflight reserves run-local directories for those artifacts:

```text
.cache/dev-loop/<run-id>/terminal/
.cache/dev-loop/<run-id>/tui/
```

The `commands.capture_cli_status` entry shows a concrete terminal transcript
capture for `polylogue ops status` against the branch-local archive. Use the
same pattern for other CLI commands whose layout, colors, wrapping, or progress
behavior matters.

For TUI or full-screen terminal work, record into the run-local `tui/`
directory using the local terminal-control surface, `script`, or VHS-style
recording when visual playback is needed. Polylogue only provides the
branch-local environment and artifact locations; visual inspection and terminal
control remain local agent/operator capabilities.

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

## Current Boundary

`devtools workspace dev-loop` is a preflight and directory-preparation command.
It does not start or stop services for you. That keeps destructive/runtime acts
explicit while still giving agents a single place to discover the active ports,
service state, and branch-local paths before they start a daemon/browser loop.

# Polylogue Browser Capture

Local-first Manifest V3 extension for capturing ChatGPT and Claude.ai
sessions into Polylogue, including resumable inventory-driven background
backfills.

## Install

There are three supported installation paths. All of them require the
local receiver to be running first.

### 1. Start the receiver

```bash
polylogued browser-capture serve
```

Keep this terminal open. The receiver runs on `http://127.0.0.1:8765`.
For normal long-running use, `polylogued run` starts the receiver together
with live source watching.

### 2a. Chrome / Chromium — unpacked from a clone

For active development against the in-tree source:

1. Open `chrome://extensions`
2. Enable **Developer mode** (toggle, top right)
3. Click **Load unpacked** and select this directory (`browser-extension/`)
4. Pin the extension to the toolbar so the badge is always visible

### 2b. Chrome / Chromium — packed `.zip` from a release

For users who want a stable artifact rather than a working tree:

1. Download `polylogue-browser-capture-<version>-chrome.zip` from the
   [latest GitHub release](https://github.com/Sinity/polylogue/releases/latest)
2. Extract the archive somewhere stable
3. Open `chrome://extensions`, enable **Developer mode**
4. Click **Load unpacked** and select the extracted directory

The Chrome Web Store listing is tracked separately (see [#1238 follow-up](
https://github.com/Sinity/polylogue/issues/1238)); until that lands the
packed-zip path is the supported install for Chromium browsers.

### 2c. Firefox — `.xpi` temporary install

Download `polylogue-browser-capture-<version>-firefox.xpi` from the same
release. Firefox temporary install:

1. Visit `about:debugging#/runtime/this-firefox`
2. Click **Load Temporary Add-on…**
3. Select the downloaded `.xpi`

Temporary installs are cleared when Firefox restarts. AMO-signed permanent
installs require store submission, which is tracked as follow-up; until
then either repeat the temporary install or use Firefox Developer Edition /
Nightly with `xpinstall.signatures.required = false`.

### 3. Verify it works

```bash
polylogued browser-capture status
```

Then navigate to `chatgpt.com`, `claude.ai`, or a supported Grok/X route.
Open the popup and use **Capture page** for the current page or
**Sync open tabs** for all currently open supported tabs. The extension does
not continuously watch page mutations or capture while you type.
The popup refreshes receiver status automatically when opened and then on a
short cadence while it remains open; **Check status** is a manual refresh, not
the only way to update state.

For branch-local development, point **Local receiver URL** in the popup at the
URL printed by `devtools workspace dev-loop`, usually
`http://127.0.0.1:8875` when the production receiver is still running on the
default port. Each status/archive/capture request sends `X-Request-ID`; the
popup shows the receiver's echoed request id and records the request/response
stage in **Debug log**. Use **Export JSON** to save the redacted debug packet
and correlate popup action, service-worker request, receiver decision,
artifact ref, and archive state. The debug packet intentionally records stage
metadata rather than transcript text.

Before loading a GUI browser, run the branch-local background/receiver smoke:

```bash
devtools workspace dev-loop --extension-smoke
devtools workspace dev-loop --browser-provider-smoke
```

The smoke starts a temporary local receiver, imports the actual background
worker with a Chrome API mock, proves unauthenticated rejection, configures the
receiver token, checks receiver status, and posts a deterministic capture
envelope. Artifacts are written under `.cache/dev-loop/<run-id>/browser/`.
This proves the extension service-worker HTTP path without using real
ChatGPT/Claude.ai profile data. The provider page smoke then loads the unpacked
extension into real headless Chromium/Chrome, serves deterministic ChatGPT and
Claude fixture pages behind a local CONNECT proxy on their normal origins, opens
those pages through extension-owned tab APIs, triggers the content-script
capture path through extension-visible tabs, and verifies provider/adapter
identity, receiver request ids, and spool artifacts without copying browser
profiles or writing raw turn text into the summary. Automated unpacked-extension
proof prefers Chromium or Chrome for Testing, including the local Nix-store
Chromium when it is not on `PATH`, because branded Google Chrome 137+ can expose
a partial service-worker target while withholding content-script and extension
page behavior from `--load-extension` automation. Set
`POLYLOGUE_PROVIDER_SMOKE_CHROME=/path/to/browser` to override the binary. If
manifest content-script delivery is unavailable in the fixture browser, the
smoke uses the same `chrome.scripting.executeScript` retry path as the popup and
records the `injection_mode`. If Chrome creates the page target but the
extension cannot see the corresponding tab, the summary records both the CDP
page target and the extension-visible tab inventory.

For live authenticated ChatGPT/Claude.ai work, generate the operator-local plan
and run the copied-profile proof from the repo instead of inventing a private
checklist:

```bash
devtools workspace dev-loop --browser-plan
devtools workspace dev-loop --browser-live-proof \
  --browser-live-profile-dir .local/browser-profiles/<run-id>-chrome-user-data \
  --browser-live-chatgpt-url https://chatgpt.com/c/<conversation-id> \
  --browser-live-claude-url https://claude.ai/chat/<conversation-id>
```

`--browser-plan` writes `browser-live-proof-checklist.md` and
`browser-live-proof.env.example` under the run-local browser artifact directory.
`--browser-live-proof` opens a visible local Chrome/Chromium with the unpacked
extension and the operator-approved copied profile, configures a temporary
branch-local receiver, asks content scripts to capture live provider pages, and
writes a redacted summary plus request-id/artifact evidence. It refuses CI by
default and rejects common live profile roots or Chrome `Singleton*` lock files
unless the operator explicitly overrides that guardrail in the local shell.
Raw captured content remains only in ignored local receiver spool artifacts.

If headless Chromium cannot expose MV3 extension service workers, the browser
smokes fail cleanly with `Polylogue extension service worker not found` in the
stderr artifact. Set `POLYLOGUE_BROWSER_SMOKE_CHROME` or
`POLYLOGUE_PROVIDER_SMOKE_CHROME` to a Chrome/Chromium binary with extension
service-worker support. When Chrome exposes the service worker but does not
deliver provider content scripts, the provider smoke records page diagnostics,
tab inventory, and injection mode. Set `POLYLOGUE_PROVIDER_SMOKE_HEADLESS=0` to
repeat the same isolated-profile proof visibly, or use `--browser-live-proof`
with a visible copied profile for operator-local live-page evidence.

## Supported Sites

| Site | Provider | Notes |
|------|----------|-------|
| `chatgpt.com` | ChatGPT | Provider adapter; prefer structured app payloads and fall back to DOM text |
| `claude.ai` | Claude (web) | Provider adapter; DOM text fallback until a structured source is proven |

The extension only captures content from supported pages. On unsupported
pages the badge shows grey and no data is sent.

## Health & Privacy

- **Receiver reachable**: badge is green when the local receiver is up
- **Site supported**: badge shows a document icon when the current page is a known LLM site
- **Last capture**: popup shows the timestamp and provider of the most recent capture
- **Offline**: badge turns red when the receiver is down
- **DOM fallback (`dom_degraded`)**: the provider-native app payload was not
  available, so the extension captured visible DOM text. This can be useful
  live evidence, but may omit branches, provider ids, timestamps, or
  attachments.
- **Stale archive**: the receiver spool is newer than the indexed archive. Keep
  the daemon running; live convergence should advance this to **Archived**
  automatically.
- **Explicit background collection only**: ordinary page capture reads content
  only on an explicit capture action. A backfill runs in the service worker
  only after **Start** is pressed with a provider and cutoff; it uses the
  authenticated provider inventory/API, never activates foreground tabs, and
  can be paused or cancelled from the popup.
- **Local only**: content is posted to the configured `127.0.0.1` receiver and never leaves your machine
- **Privacy diagnostics**: the popup shows capture counts and timestamps, never message content

## Troubleshooting

| Symptom | Check |
|---------|-------|
| Badge is grey | Navigate to a supported page (chatgpt.com, claude.ai, or Grok/X) |
| Badge is red | Receiver is not running — start `polylogue browser-capture serve` |
| Captures not appearing in archive | Run `polylogue check` to verify the daemon is ingesting |
| Popup says `stale` | The receiver has a newer spool artifact than the indexed archive. Leave the daemon running and inspect the debug log request id if it does not converge. |
| Popup says `dom` / `dom_degraded` | Reload the provider page, wait for the conversation to load fully, then capture again so the native app payload can be observed. |
| A button click seems ineffective | The button status line should show Working/Done/Failed. Open **Debug log** and export JSON if the state does not change. |
| "Failed to load extension" in Chrome | Ensure you selected the `browser-extension/` directory (not `src/`) |
| Extension not updating | Go to `chrome://extensions`, click the refresh icon on the extension card |

## Architecture

```
Browser (ChatGPT/Claude page/app state)
    │
    │  content script captures provider-native state or DOM fallback
    ▼
Extension popup / background
    │
    │  HTTP POST to configured 127.0.0.1 receiver
    ▼
polylogue browser-capture serve (Python)
    │
    │  writes to archive inbox
    ▼
polylogued daemon → ingests → FTS index
```

## Resumable background backfill

The popup's **Background backfill** panel starts a provider-native inventory
delta from a user-selected cutoff. Jobs and queue entries live in IndexedDB,
so MV3 service-worker suspension or a browser restart does not lose the cursor,
captured artifact, retry deadline, or receiver receipt. `chrome.alarms` wakes
the next eligible item; expired leases are recovered and only one extension
instance may own an item at a time.

The scheduler uses concurrency one per provider, a conservative learned
request cadence, daily and per-wake budgets, `Retry-After`, exponential
full-jitter backoff, and a circuit breaker. A 403/auth/challenge pauses for
operator action; repeated 429s or transport failures pause the provider job.
Native-empty conversations, authorization failures, bounded retry exhaustion,
receiver outages, and successful durable ACKs remain distinct in the exported
diagnostic ledger. Receiver-down artifacts remain queued and are retried
without refetching provider data.

Execution itself is leased in IndexedDB, not only the queue item. Request
budget is reserved atomically before a provider call, and pause/cancel bumps a
generation checked by every asynchronous continuation, so two service workers
cannot spend the same token or resurrect cancelled work. Provider requests
time out before the execution lease can expire. Exact provider revision
timestamps are recorded in a durable native-id ledger; a later job skips only
an exact known revision, while missing/untrusted revisions are fetched again.
Receiver retries and stored native envelopes are bounded and fail-paused on
attempt, byte, or IndexedDB quota exhaustion.

Completion means the loopback receiver atomically wrote the artifact and
returned both a request id and the exact submitted JSON-byte SHA-256. The
deduplication identity is provider native id plus content hash. It does **not**
mean the provider inventory was historically complete: Polylogue honors the
provider's authentication, inventory visibility, rate limits, challenges, and
deletion semantics and cannot prove completeness beyond what that authenticated
inventory exposes. It does not bypass anti-bot controls or scrape records the
provider does not enumerate.

## Development

```bash
npm install
npm test              # vitest
npm run test:watch    # watch mode
npm run lint          # eslint
npm run validate      # in-tree manifest validation
npm run dev-loop-smoke # background worker -> local receiver smoke
npm run dev-loop-browser-smoke # real Chromium/Chrome service worker -> local receiver smoke
npm run dev-loop-provider-smoke # real Chromium/Chrome content scripts -> deterministic provider fixtures
npm run dev-loop-live-provider-proof # visible copied-profile live provider proof
npm run build         # build Chrome .zip + Firefox .xpi under dist/
npm run screenshots   # capture store-submission screenshots (Playwright)
```

Tests run against deterministic fixture HTML, not live ChatGPT/Claude pages.
The receiver contract test verifies envelope compatibility with
`polylogue/browser_capture/models.py`.

## Release Packaging

The release artifacts are built and attached to GitHub Releases by
[`.github/workflows/extension-release.yml`](../.github/workflows/extension-release.yml).
On `v*.*.*` tag push the workflow:

1. Runs the in-tree manifest validator (`npm run validate`)
2. Runs ESLint + Vitest (incl. build-script regression tests)
3. Builds `polylogue-browser-capture-<version>-chrome.zip` and
   `polylogue-browser-capture-<version>-firefox.xpi`
4. Runs `web-ext lint` against the unpacked Firefox bundle
5. Captures Playwright screenshots of the popup at Chrome Web Store and
   AMO submission aspect ratios, bundled as `store-screenshots-<tag>.tar.gz`
6. Uploads all artifacts to the matching GitHub Release (`gh release upload --clobber`)

The build script reads the canonical project version from the [project]
table of `pyproject.toml` and rewrites `manifest.json` + `package.json`
to match before packaging. The Firefox manifest is generated separately
with a `browser_specific_settings.gecko` block so the same source tree
ships under both stores.

To rebuild locally:

```bash
cd browser-extension
npm install
npm run build
ls dist/
```

`scripts/build.mjs` accepts `--version X.Y.Z` and `--out DIR` for ad-hoc
runs (release smoke testing, manual store submission, etc.).

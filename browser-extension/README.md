# Polylogue Browser Capture

Local-first Manifest V3 extension for capturing ChatGPT and Claude.ai
sessions into Polylogue.

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

Then navigate to `chatgpt.com` or `claude.ai`. The extension badge should
turn green. Start a conversation — each exchange is captured as you type.

For branch-local development, point **Local receiver URL** in the popup at the
URL printed by `devtools workspace dev-loop`, usually
`http://127.0.0.1:8875` when the production receiver is still running on the
default port. Each status/archive/capture request sends `X-Request-ID` and the
popup shows the receiver's echoed request id. Use that value to correlate the
popup/service-worker result with browser network traces and receiver logs.

Before loading a GUI browser, run the branch-local background/receiver smoke:

```bash
devtools workspace dev-loop --extension-smoke
```

The smoke starts a temporary local receiver, imports the actual background
worker with a Chrome API mock, proves unauthenticated rejection, configures the
receiver token, checks receiver status, and posts a deterministic capture
envelope. Artifacts are written under `.cache/dev-loop/<run-id>/browser/`.
This proves the extension service-worker HTTP path without using real
ChatGPT/Claude.ai profile data.

## Supported Sites

| Site | Provider | Notes |
|------|----------|-------|
| `chatgpt.com` | ChatGPT | DOM adapter for conversation thread |
| `claude.ai` | Claude (web) | DOM adapter for chat messages |

The extension only captures content from supported pages. On unsupported
pages the badge shows grey and no data is sent.

## Health & Privacy

- **Receiver reachable**: badge is green when the local receiver is up
- **Site supported**: badge shows a document icon when the current page is a known LLM site
- **Last capture**: popup shows the timestamp and provider of the most recent capture
- **Offline**: badge turns red when the receiver is down
- **No background collection**: the extension only reads the DOM when you are actively on a supported page
- **Local only**: content is posted to `127.0.0.1:8765` and never leaves your machine
- **Privacy diagnostics**: the popup shows capture counts and timestamps, never message content

## Troubleshooting

| Symptom | Check |
|---------|-------|
| Badge is grey | Navigate to a supported page (chatgpt.com or claude.ai) |
| Badge is red | Receiver is not running — start `polylogue browser-capture serve` |
| Captures not appearing in archive | Run `polylogue check` to verify the daemon is ingesting |
| "Failed to load extension" in Chrome | Ensure you selected the `browser-extension/` directory (not `src/`) |
| Extension not updating | Go to `chrome://extensions`, click the refresh icon on the extension card |

## Architecture

```
Browser (ChatGPT/Claude DOM)
    │
    │  content script reads the DOM
    ▼
Extension popup / background
    │
    │  HTTP POST to 127.0.0.1:8765
    ▼
polylogue browser-capture serve (Python)
    │
    │  writes to archive inbox
    ▼
polylogued daemon → ingests → FTS index
```

## Development

```bash
npm install
npm test              # vitest
npm run test:watch    # watch mode
npm run lint          # eslint
npm run validate      # in-tree manifest validation
npm run dev-loop-smoke # background worker -> local receiver smoke
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

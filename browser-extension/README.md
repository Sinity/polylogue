# Polylogue Browser Capture

Local-first Manifest V3 extension for capturing ChatGPT and Claude.ai
sessions into Polylogue.

## Install

### 1. Start the receiver

```bash
polylogue browser-capture serve
```

Keep this terminal open. The receiver runs on `http://127.0.0.1:8765`.

### 2. Load the extension in Chrome / Chromium

1. Open `chrome://extensions`
2. Enable **Developer mode** (toggle, top right)
3. Click **Load unpacked** and select this directory (`browser-extension/`)
4. Pin the extension to the toolbar so the badge is always visible

### 3. Verify it works

```bash
polylogue browser-capture status
```

Then navigate to `chatgpt.com` or `claude.ai`. The extension badge should
turn green. Start a conversation — each exchange is captured as you type.

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
```

Tests run against deterministic fixture HTML, not live ChatGPT/Claude pages.
The receiver contract test verifies envelope compatibility with
`polylogue/browser_capture/models.py`.

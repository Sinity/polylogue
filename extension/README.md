# Polylogue browser extension

A Manifest V3 extension that captures ChatGPT, claude.ai, Claude Design,
Gemini and Grok conversations from the provider's own data (API responses),
as they happen and as a background backfill, and hands them to Polylogue.

## How it reaches Polylogue

The extension talks only to its native messaging host,
`com.polylogue.browser_capture`, which is the `polylogue` binary itself.
Chrome starts the host only for the extension IDs its manifest names, so
there is no token or pairing step.

- A capture is written into the spool,
  `$XDG_STATE_HOME/polylogue/browser-capture/<provider>/`, one file per
  conversation, synced before the extension is told it is stored. The
  daemon ingests the spool like any other location.
- Reads (status, what the archive holds for a tab) go from the host to the
  daemon over its Unix socket. When the daemon is not running, what the
  archive holds is reported as unknown.
- Chat actions (draft or submit a turn in a provider's web UI) are queued
  by the operator through `polylogue` and run by the extension only while
  the archive setting `browser.actions` is `on`.

## Install

1. In `chrome://extensions`, enable Developer mode and choose **Load
   unpacked** on this directory, from a checkout that stays put.
2. Register the host with the extension ID Chrome shows:

   ```bash
   polylogue install-agent chrome --extension-id <id>
   ```

   `polylogue install-agent chrome --remove` takes it out again.
3. Run the daemon (`polylogue daemon`) so captures are ingested.

## Develop

```bash
npm install
npx vitest run       # unit tests
npm run validate     # manifest checks
npm run build        # packaged build
```

# Polylogue browser extension

A Manifest V3 extension that captures ChatGPT, claude.ai, Claude Design,
Gemini and Grok conversations from the provider's own data (API responses),
as they happen and through background backfill, and hands them to Polylogue.

## How it reaches Polylogue

The extension talks only to its native messaging host,
`com.polylogue.browser_capture`, which is the `polylogue` binary itself.
Chrome starts the host only for the extension IDs its manifest names, so
there is no token or pairing step.

- ChatGPT, claude.ai and Grok raw replies are stored locally. The native
  host uses the shared source parsers to inspect them; external attachments
  are downloaded by the original provider document and included by value.
  Native inline assets stay in the raw reply. The final capture uses the
  ordinary durable capture route.
- A capture is written into the spool,
  `$XDG_STATE_HOME/polylogue/browser-capture/<provider>/`, one file per
  conversation, synced before the extension is told it is stored. The
  daemon ingests the spool like any other location.
- Reads (status, what the archive holds for a tab) go from the host to the
  daemon over its Unix socket. When the daemon is not running, what the
  archive holds is reported as unknown.
- The native action routes support claiming, approval, attachments and
  reporting. The extension executes queued actions only while the archive
  setting `browser.actions` is `on`. A public CLI/reader enqueue command
  is still missing; the synthetic host tests queue actions internally.
- **Open Polylogue** uses the address the running reader reports. It reports
  unavailable when no reader is running; it does not guess a port or start
  another server.

## Install

1. In `chrome://extensions`, enable Developer mode and choose **Load
   unpacked** on this directory, from a checkout that stays put.
2. Register the host with the extension ID Chrome shows:

   ```bash
   polylogue install-agent chrome --extension-id <id>
   ```

   `polylogue install-agent chrome --remove` takes it out again.
3. Run the daemon (`polylogue daemon`) so captures are ingested.

## Save a selected passage as a note

Select up to 2,000 characters inside one identified provider message and open
Polylogue's in-page panel (Alt+P). Edit **Note body**, then choose **Save note**.
The quote is kept separately from your note, with the supplied provider,
conversation ID, message ID, page URL and selection time. The daemon verifies
that the exact message belongs to the supplied archived conversation. Missing
or ambiguous identity is shown as a save failure; capture the conversation
before retrying. The quoted passage is a browser observation, not a claim about
an archived source version or what a model received.

**Note saved** appears only after the daemon acknowledges the ordinary operator
note. An uncertain reply keeps the draft and its note ID: retrying it creates no
duplicate, including after a native-host or daemon restart. Editing the body
creates another note ID. Drafts live in the page; reloading the page discards an
unsaved draft. Read saved notes with `polylogue note list session:ORIGIN:ID` or
the reader's ordinary note API. Notes survive daemon restarts.

## Local backups and recovery

Backfill jobs and pending bytes live in IndexedDB and OPFS across worker
restarts. Export downloads the prepared `polylogue.backfill-export.v2`
file, including pending bodies, raw replies, normalized records and attachment
bytes. Choose **Confirm saved** only after saving it; this releases the
export copy without removing the original job or its pending evidence.

Import verifies file hashes, sizes, part order and references, then commits
the restored job in one database transaction. Restored jobs stay paused and
keep their original account scope. A different profile's scope is not
silently reassigned. Missing record data in an older backup requires
explicit recovery. The importer holds one acquired entry, including its
encoded bytes, while parsing; large individual files can use substantial memory.

There is no server or automatic cross-profile recovery. Local file import is
an explicit backup operation. Mission intelligence is unavailable on the
native host. Selected passages are saved as ordinary notes.
The tests use a synthetic transport and parsers, not real browser accounts.

## Develop

```bash
agentctl job start polylogue extension -- ci
agentctl job start polylogue extension -- test -- tests/native_fetch.test.js
agentctl job start polylogue extension -- run validate
agentctl job start polylogue extension -- run build
```

# Polylogue browser extension

A Manifest V3 extension that saves ChatGPT, claude.ai, Grok and Gemini
conversations into Polylogue as they change. ChatGPT, claude.ai and Grok
conversations are read from the provider's own API, the same JSON their
exports hold; Gemini, which has no such API, is read from the page.

## How it works

The extension talks only to its native messaging host,
`com.polylogue.browser_capture`, which is the `polylogue` binary. Chrome
starts the host only for the extension IDs its manifest names, so there is no
token or pairing step.

- When a conversation page settles after a change (and is not still
  answering), the extension fetches the conversation, asks the host which
  files it names, fetches their bytes through the page and hands the host one
  capture. The host appends it to the conversation's capture log in the spool,
  `$XDG_STATE_HOME/polylogue/browser-capture/<provider>/`, and answers only
  once the file is synced. The daemon reads the spool like any location.
- Until the host has answered, a capture waits in the extension's outbox, so
  a stopped service worker, a browser restart or an extension update loses
  nothing: on start, the extension sends what waits again. The host does not
  append bytes the log already holds. The outbox is the only thing the
  extension keeps; everything else lives in Polylogue.
- A page is announced only when the conversation's text on it changes, and a
  conversation whose content has not changed since its last capture is not
  captured again, across restarts too (the comparison ignores the order of
  plain lists such as ChatGPT's `safe_urls`, which the provider reshuffles on
  every read). Open chat tabs are checked every 15 minutes as well.
- Reads (what the archive holds of a conversation, whether the daemon runs)
  go from the host to the daemon. While the daemon is not running, captures
  are still stored, and what the archive holds is shown as unknown.

## Backfill

`polylogue backfill start chatgpt` (or `claude-ai`, `grok`; `--since DATE`
limits it) asks the extension to capture a provider's older conversations.
The job lives in Polylogue: the extension claims it, pages through the
provider's conversation list in an open tab of that provider (or a background
tab it opens and closes), captures each conversation the archive lacks or
holds an older copy of, and reports where it got to. A restarted extension
continues from there. `polylogue backfill list` shows how far each job got;
`polylogue backfill cancel ID` stops one.

## Chat actions

`polylogue chat send chatgpt "TEXT" --model "MODEL"` asks the extension to
draft a turn in a new ChatGPT conversation, or in an existing one with
`--conversation ID`; `--submit` sends it, `--attach FILE` attaches files. The
archive setting `browser.actions` must be `on` (`polylogue setting
browser.actions on`); setting it to anything else stops the extension from
taking more. A send into an existing conversation waits for approval in the
extension's popup or with `polylogue chat approve ID`. The extension works in
a background tab, leaves a draft open there, and captures the conversation
after a send. MCP offers `chat_send` and `chat_list` too.

## The in-page button

On a chat page, the Polylogue button (Alt+P) shows whether the conversation
is in the archive, captures it on request, and saves a selected passage as a
note: select text inside one message, write the note and choose **Save note**.
The quote, the note and the message's provider, conversation and message IDs
are kept; the daemon checks that the message belongs to that archived
conversation, so capture the conversation first. Retrying a save sends the
same note ID, so a note is never saved twice. Read notes with
`polylogue note list session:ORIGIN:ID`.

## Install

1. In `chrome://extensions`, enable Developer mode and choose **Load
   unpacked** on this directory, from a checkout that stays put.
2. Register the host with the extension ID Chrome shows:

   ```bash
   polylogue install-agent chrome --extension-id <id>
   ```

   `polylogue install-agent chrome --remove` takes it out again.
3. Run the daemon (`polylogue daemon`) so captures are read.

## Develop

```bash
agentctl job start polylogue extension -- ci
agentctl job start polylogue extension -- test
agentctl job start polylogue extension -- run lint
agentctl job start polylogue extension -- run validate
```

The tests run the background against a fake Chrome whose native messaging
host speaks the real frames, and the page scripts in jsdom pages; inputs are
synthetic.

# Browser extension and clients

## Browser extension
The extension (JavaScript, Manifest V3, a separate directory in the
repository) captures ChatGPT, claude.ai, Claude Design, Gemini and Grok
conversations from the provider's own data (API responses), not from the
page's HTML, as they happen and as a background backfill of older chats. It
also fetches attachment and generated-file bytes before their links expire.

Contract with Polylogue:

- The extension sends each capture to the native messaging host, `polylogue
  browser-host`, which writes it into the capture spool. Chrome starts the
  host only for extensions its manifest lists as allowed origins, so the
  extension needs no token or pairing step (the Python version's bearer
  token and `pairing` commands have no successor); `install-agent chrome`
  writes that manifest (`cli.md`; Home Manager may declare the same file
  instead), and `status` says when it is missing or names another binary. The spool's
  location, file names and capture file format are defined once, in
  `sources/README.md` (browser captures). The host names files by content hash,
  never by a name taken from the page, and does not follow symlinks. It
  answers the extension only after the spool file and its directory are
  synced, and its answer names the stored file's hash and whether it was new
  or a duplicate, so the extension knows exactly what was kept.
- The daemon ingests the spool like any location. Captures, exports and
  repeated captures of the same conversation merge into one session
  (`decisions/merging.md`).
- No job queue, leases or receiver state machine: the spool directory is the
  queue, and a file that is stored is done.
- **The extension obeys only its own code and the operator.** It records
  only responses it intercepted itself from the provider's API; it never
  accepts conversation data, commands, conversation IDs or wake-ups posted
  by page scripts (window messages), because any script on the page can
  forge them. Captures start only from its popup, badge and background
  worker.
- **Polite capture.** A conversation is fetched again only when the
  provider's update time or message count changed since the last capture. A
  rate limit or server error stops all fetching from that provider, asset
  downloads included, for the provider's `Retry-After` or an exponential
  backoff, and nothing can bypass it. The popup's "Capture again" skips only the unchanged check (update time and message count) for that one conversation; it waits out a backoff like any other capture. Pending captures survive a browser
  restart. Tests run the extension's real fetch and scheduling code on
  synthetic data and never contact a provider.
- **Capture robustness** (slice 4): backfill is paced; the extension never
  captures its own requests (no feedback loops); each capture carries a
  browser-instance identity so two browsers never overwrite each other's
  state; Grok on x.com has a capture path; temporary chats show their real
  state (not saved by the provider, captured or not by Polylogue); provider
  projects, custom GPTs and Claude Design projects are captured with the
  conversations they hold (`sources/chat-exports.md`, collections). Also
  (`polylogue-jlme.7`): content scripts are added to tabs already open when
  the extension is installed or updated; a turn holding only thinking is
  still a captured turn; a temporary chat's ID is settled before the
  freshness check; the periodic freshness sweep runs only while the native
  host answers (otherwise the state is *Needs attention*); a provider with no
  update-time signal is recaptured on its message count; and asset fetches
  end within the capture's own timeout, a timed-out asset being retried like
  a failed download.
- **Where things appear in the page** (design of 2026-09-26): per-message
  archive state and a save action sit in the provider's own message action
  row and blend in; recall across conversations, cost and deeper inspection
  live in a summonable chip and panel that floats; selected text can be
  captured as a note; nothing shifts the page's layout; all of these read
  one capture-state signal. Mounted in an isolated Shadow DOM; when the
  provider page or Polylogue changes shape, the extension fails open (the
  provider page keeps working, the extension says it is degraded).
  Per-message state and save: slice 4. Panel, selection capture, archive
  search in the page, posting to a chat and reorganizing chat titles: slice
  12.
  Posting to a chat (slice 12) keeps three facts apart and never collapses
  them into one success: the operator's request, the extension's send, and
  the provider's reply observed in a later capture. Each send carries a
  request ID (`decisions/process.md`, writes carry their own ID), so two
  intended sends of equal text are two sends and a lost acknowledgement
  is not taken as "not sent"; the post shows as sent only when a capture
  contains it (`polylogue-2n6`). An attachment of any size goes in chunks,
  as captures do (`sources/README.md`, browser captures); size never refuses
  an action (the Python route refused attachments over 16 MiB,
  `polylogue-zqp6g`).
- **Capture state in plain words**: each conversation shows one of six
  states, in the popup, the badge and the in-page indicator: *Safe*
  (captured and stored), *Catching up* (being captured), *Needs attention*
  (a capture problem the operator can fix), *Failed* (a capture failed and
  will be retried), *Not saved* (not captured, e.g. a temporary chat),
  *Partial fidelity* (stored from a weaker route, such as the page text, and
  missing structure).
- **Reads from the extension** (for the panel and in-page search) go through
  the same native messaging host, as request and response on its channel;
  the host passes each one to the daemon as a request on its Unix socket
  (`README.md`, one path through the daemon), not through a loopback HTTP
  receiver. The recorded extension design used a
  loopback receiver; native messaging removes its problems of passing an
  access token through the page and of finding the receiver's address. The
  rest of that design is adopted. Decided by the spec authors from those
  recorded problems (2026-10-11); the operator may overrule.

## Library and Python client
**Rust library.** `polylogue-engine` exposes the same operations the CLI
uses: `query(text) -> Rows`, `read(ref, view) -> Rendered`,
`analyze(query, group_by) -> Table`, `note(target, kind, text)`,
`status()`. The daemon is built on it and answers socket requests through
it; tests call it in-process to check that a socket request and the same
library call return the same rows (law 83). Other programs reach Polylogue
through the socket, not by linking the library against the archive files.

**Python client** (`polylogue-client` on PyPI and in Nix), a small package
that runs `polylogue find QUERY list --jsonl` (or the verb the method needs) as a subprocess,
passing arguments as a list and never through a shell, and parses the
output. Its writes are authored `script` (`user-data.md`, who the author is). It
needs no knowledge of the socket or the SQLite files and can never
disagree with the CLI; like the CLI, it reports when the daemon is not
running.
It covers what Lynchpin uses today, renamed to plain names:

| Python | Runs | Replaces Lynchpin's current call |
|---|---|---|
| `client.sessions(query="", limit=None)` | `polylogue find 'QUERY \| limit N' list --jsonl` (`limit` is a selection, added to the query) | `list_summaries` |
| `client.session(id)` | `polylogue ID list --json` (the session's row with its facts) | per-session profile (`get_session_profile_insight`) |
| `client.session_stats(query="")` | `polylogue find QUERY list --fields … --jsonl` | `list_session_profile_insights` |
| `client.messages(id, after=None, page_size=500)` | `polylogue 'messages where session.id:ID' list --jsonl --page-size 500`, then `--cursor` with each trailer's `next_cursor` until it is null | `get_messages_paginated` |
| `client.messages_bulk(ids)` | `polylogue 'messages where id:(ID1\|ID2\|…)' list --jsonl` | `bulk_get_messages` |
| `client.coverage(by="day")` | `polylogue find QUERY analyze --by origin,day --json` | `list_archive_coverage_insights` |
| `client.work_groups(query="")` | `polylogue find QUERY analyze --by cwd --json` plus lineage links | `list_thread_insights` |
| `client.status` | `polylogue status --json` | `insight_readiness_report` |
| `client.totals(query="")` | `polylogue find QUERY analyze --json` | `stats` |
| `client.query(text)` | `polylogue find TEXT --jsonl` | anything else, through the query language |

Large results stream as JSON lines, so memory stays flat. A process start
per call costs a few milliseconds, which is negligible for Lynchpin's batch
reads; if a hot loop ever needs more, the client can keep one `polylogue
mcp` process open instead, with no change to its interface.

"Phases" (`get_session_phase_insights`) is dropped: phases were heuristic
classification of the kind the operator ruled out. Lynchpin's direct reads of
the SQLite files stop; the files' layout is not a public contract.

Bead and audit IDs behind each section: `evidence/beads/spec-sources-g3.md`.

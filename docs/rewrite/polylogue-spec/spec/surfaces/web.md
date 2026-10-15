# Web reader

Served by the daemon on `http://127.0.0.1:8765`, loopback only.

Payloads served over HTTP to the browser never contain the archive
directory's path: the field naming it is left out, and the path inside any
message or note text is shown as `[archive]`. The CLI and MCP keep paths
(`status` names the directory, rule R53).

**Access.** Loopback alone does not stop a web page in the operator's
browser from reaching the port, so:

- a request is accepted only if its `Host` header is `127.0.0.1:8765`,
  `localhost:8765` or `[::1]:8765` (this stops DNS rebinding);
- `/api/` and every write need the access token, a random value the daemon
  writes at start to `$XDG_RUNTIME_DIR/polylogue/<key>/token` (mode 0600).
  `$XDG_RUNTIME_DIR/polylogue/<key>/` holds socket, token, port and rows,
  where `<key>` is the first 12 hex digits of BLAKE3 over the archive
  directory's resolved path; an archive's daemon listens on 8765 unless the
  archive was marked as a demo or test archive when created, in which case
  it takes a port the system assigns,
  written to `<key>/port`, so nothing reaches another archive's daemon or
  overwrites its token. `polylogue open` hands it to the browser as a
  `SameSite=Strict` cookie;
- the HTTP port serves the browser only; every other program uses the Unix
  socket (`README.md`, one path through the daemon);
- before it sends the token (`polylogue open`), Polylogue checks that the listening socket on the port
  belongs to the operator's uid (on Linux, the socket's inode in
  `/proc/net/tcp{,6}` names its owner); the daemon checks each accepted
  connection's peer uid the same way and closes connections from other
  users. If the owner cannot be determined, both refuse;
- writes are POST only; no CORS headers are sent; pages carry a strict
  Content-Security-Policy that allows only the reader's own scripts (htmx,
  served by the daemon; `rendering.md`);
- excision is not available over HTTP;
- an ID inside a URL is percent-encoded once as a whole and decoded once,
  never split (IDs are opaque, `model.md`);
- each JSON route is the same request as on the socket, mapped from the
  same verb and command tables as the CLI (one request shape, two
  transports, no second registry), so a command and its route cannot drift
  apart.

**First version** (reading, plus tags and notes):

| Page | Shows |
|---|---|
| `/` | the same overview as bare `polylogue`, plus a search box |
| `/search?q=QUERY` | results as in the CLI, with snippets; any query the CLI accepts |
| `/s/SESSION` | the session reader (`rendering.md`): tool calls collapsed to one line, outcome first, expandable; lineage links; notes on the message header line, as in the terminal; a link to the raw source records |
| `/cost` | cost by day, model and origin, with unknowns shown as unknown |
| `/live` | the Running page: the live sessions grouped by repository (slice 10): each with its live state and since when, harness, model, worktree, open tool call or todo item, the last tool call and its outcome, cost so far, task, subagents as a tree, and the processes under it; sessions waiting on the operator first; refreshed from the change feed (`model.md`, Live state) |
| `/s/SESSION/report` | the session report (slice 10; `workflows.md` §18): turns on a timeline, every tool call with outcome and duration, files touched, cost so far, compactions, corrections, lineage, and agents' notes where they apply |
| `/s/SESSION/share` | share an annotated excerpt (slice 10; `workflows.md` §19): choose a message range and its notes, review what is included and the flagged credential spans, save one self-contained HTML file |

Every page has "copy as CLI command". The JSON API serves the same data as
`--json`, under `/api/v1/`, one route per row of `cli.md`'s verb and
command tables and nothing else: `GET /api/v1/<verb>?q=QUERY&…` for the
reading verbs (`/api/v1/list?q=…`, `/api/v1/read?q=ID&view=…`,
`/api/v1/analyze?q=…&by=…`), `POST /api/v1/<verb>` for the writing verbs,
and `GET`/`POST /api/v1/cmd/<command>` for commands (`/api/v1/cmd/status`);
options are the CLI's, by the same names, and result pages continue with
`page_size` and `cursor` (`query.md`, four bounds). It is versioned because pages left open across a
daemon upgrade depend on it; other programs use the socket, not these routes.

**Every page works by keyboard and screen reader**: search focus, next and
previous result, expand a tool call, copy as CLI command all have keys;
pages use semantic HTML with labels; light and dark follow the system, and
both meet contrast for body text; animation is off when the system asks for
reduced motion (`polylogue-lu1`).
States (unknown, not recorded, zero, pending, stale) look the same as in the
terminal, from one table in `rendering.md`.

**Long sessions and search help** (slice 10): a minimap of a long session
from time buckets, authors and errors (never guessed phases), click to jump;
the search box offers the same completions and `explain` as the CLI.

**Live updates** (slice 10) are server-sent events over the change feed
(`storage/index-db.md`): each event's ID is the change feed's position `GEN:COUNTER:ARCHIVE`
(`cli.md`, `changes`), so a reconnect resumes past new notes and excisions
too; an event is sent
only after its transaction commits and carries references and small deltas,
never whole sessions; a reconnect resumes from the last event ID, and a new
generation or an unknown position makes the page reload. The same rules
serve live following of a running session.

**Later** (`plan/slices.md`): `/day/DATE` (the day page), `/p/PAGE` (pages,
`user-data.md`) and `/n/NOTE` (one note with its revisions and what refers
to it), and the timeline workbench below.

**Timeline workbench** (slice 10 or later). One screen with three bindings
shared by its timeline, table and measures:

- a time window (scope, event window, the evidence cutoff `evidence_until:`, grain);
- a focused piece of evidence, opened directly at its neighborhood (never by
  loading every earlier message);
- the state of a file at a chosen moment, graded by evidence: known from a
  successful edit, possible after a failed one, unknown (`rendering.md`,
  `changes` view).

A newer selection supersedes an older read in flight, rapid cursor moves are
coalesced, and long lanes are virtualized.

**How the reader is verified.** By a real browser loading the reader over the
Morrow archive, never by screenshots of a design canvas or a metadata-only
check: the session list, search results and a session's detail render
non-empty and readable, carry no absolute local paths or archive file names,
and are backed by the same socket requests as the CLI (operator decision,
2026-05-07, `polylogue-reader-visual-harness`).

Bead and audit IDs behind each section: `evidence/beads/spec-sources-g3.md`.

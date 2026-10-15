# Surfaces

How people and programs use Polylogue. Every surface is a thin layer over the
same query language (`../query.md`) and the same rendering code
(`../rendering.md`); none has features of its own that the others cannot
reach.

| File | Covers |
|---|---|
| [cli.md](cli.md) | the command line: grammar, IDs and row numbers, verbs, commands, output rules, examples, exit codes, completion and interactive selection |
| [mcp-and-hooks.md](mcp-and-hooks.md) | the MCP server for agents, what `install-agent` installs, and the hook command (recording and briefings) |
| [web.md](web.md) | the web reader and its JSON API |
| [extension-and-clients.md](extension-and-clients.md) | the browser extension, the Rust library and the Python client |

All examples run against the Morrow example archive
(`examples/archive/FACTS.md`) with the clock fixed at **2026-10-08 12:00
Europe/Warsaw** (tests set `POLYLOGUE_NOW`). Message texts quoted here are
part of the example archive; the raw files are written to contain them.

**What each surface looks like** is drawn on the design canvas "Polylogue UX"
(https://claude.ai/artifact/NCKHcBjr51Qg4EC1kXiSpb, private to the operator):
the terminal overview, search, read, summary, status, `continue` and a query
read back (`explain`, `read_as`); the session-start briefing (with the
machine section), the takeover briefing and an MCP search result; the web
home, session reader, session report, share-an-excerpt, cost and Running
pages; an investigation page, the page list, claims with check states and
judging a pass, and a project story; the extension popup. Every board uses Morrow data; the Running page uses
the archive's live set (`examples/archive/FACTS.md`, Live set, clock
2026-10-09 15:00). Where a board and this text disagree, the text wins and the
board is redrawn. `mockups/` holds the older static drafts.

## Outcome boundary

All surfaces preserve the daemon's typed outcome and error detail. A transport failure is `not_running`, `timeout` or `io`, never an empty read. CLI exit codes come from `cli.md` and output failure is `io`; errors go to diagnostics with their refusal reason, position, suggestion and hint. MCP returns the outcome and payload to its encoder rather than discarding them. Unsupported tools/options are explicit `not_yet_available` errors until their request mapping exists; unknown/malformed requests are `invalid_query`.

These contracts are partly executable in the composition tree. Value tests and proxy call-site wiring do not establish working socket/stdio transport, row rendering, masking or the complete output schema. Those remain bootstrap/product acceptance.

## Processes

There is one binary, `polylogue`.

| Process | Started by | Does |
|---|---|---|
| `polylogue daemon` | systemd user service (Sinnix) or by hand | runs the convergence loop (`../convergence.md`); watches provider locations and spools; answers every request on the local socket; serves the web reader and JSON API (`web.md`) |
| `polylogue …` (any other command) | the operator, an agent, a script | sends one request to the daemon over the local socket and renders the answer |
| `polylogue mcp` | an agent harness, over stdio | a thin proxy: each MCP tool call becomes the same request on the socket |
| `polylogue hook EVENT` | Claude Code and Codex hooks | appends one event line to a spool file; for the briefing events also asks the daemon for the briefing the harness injects (`mcp-and-hooks.md`) |

**One path through the daemon.** Polylogue runs as the daemon, and every
read and write goes through it. The CLI, the MCP proxies, the hook command
and the extension's native-messaging host send newline-delimited JSON
requests over one Unix socket (`$XDG_RUNTIME_DIR/polylogue/<key>/socket`,
owner-only) and the daemon answers; the Python client goes through the CLI.
The web reader is a browser page, so the daemon also serves it over loopback
HTTP, where each route maps to the same request shape: one request shape,
two transports, no second registry (`web.md`).
`$XDG_RUNTIME_DIR/polylogue/<key>/` holds socket, token, port and rows, where
`<key>` is the first 12 hex digits of BLAKE3 over the archive directory's
resolved path. The daemon keeps warm state that makes commands fast:
compiled query plans, recent results with their next pages (continuation
stays by key, with no SQLite transaction held between pages; rule R52), and
embedding vectors in memory for `near:`. Completion goes through it too
(under 50 ms).

**One request shape.** A request is a query in the query language
(`query.md`) with a unit, a verb from the verb table of `cli.md` and
output options; or one command from the commands table of `cli.md` that is
not purely local. The verb and command tables are the closed list: a new capability is a query field, a
verb or a row in those tables, generated into the request type, never a
separate protocol operation. Every request may carry `deadline_ms`. The
MCP tools send a subset of these requests, with their arguments mapped as
`mcp-and-hooks.md` lists, and the answer is the result shape of "Results and errors" below. Every write request carries a
`request_id` chosen by the client (a UUIDv7; the MCP proxy chooses it per
tool call), generated once per user action and
reused on retry; the rows it creates take their IDs from it, so a request
resent after a timeout finds its rows and writes nothing, and a resent `id`
with different content is refused (`conflict`), checked against the
`client_request` row each write leaves (`storage/archive-db.md`). A queued write whose client
deadline passed before it started is dropped, not run. Cached answers are
keyed by the full `view` (`query.md`, one view per query) and by the
resolved clock for queries that read it (`behavior.md`, warm state). There is no per-command
operation registry, receipt or contract layer: those made the Python
version's protocol 15–25k lines, and this one is about 1–2k.

**When the daemon is not running,** every command that needs it prints one
line saying so and how to start it (`systemctl --user start polylogue`, or
`polylogue daemon`), exits with code 6 (`cli.md`, exit codes), and opens no
database. The one exception is `hook`: whatever it prints is injected into
the agent's context, so with the daemon down it prints nothing and exits 0
(`mcp-and-hooks.md`). Purely local commands need nothing from the daemon and
work the same with it down: `--help`, `--version`, `completions SHELL`,
grammar-only completion, `install-agent`, `browser-host` (it reports the
daemon's absence to the extension), `auth`, `demo` (it starts its own
daemon) and `daemon` itself (`polylogue-1fp`, `polylogue-vtgpi`).
There is no fallback read path and no standalone mode. The hook recording
entry still appends to its spool, which the daemon collects when it starts.

Sources: operator ruling 2026-09-14 ("Daemon is the standard"); `decisions/process.md`.

## Results and errors

Query results (`list`, search, `analyze`, MCP `search`, the HTTP API) have
the result fields defined once, in `query.md` ("Results"), including
`up_to_date`. A session document (`read --format json`, `export`) is the
export format (`export-format.md`) instead. MCP adds `next`, the
ready-to-use follow-up call when a result was cut by its limit or budget.
JSON times are always UTC with `Z` and milliseconds. Text for a person or
a model (the terminal, MCP `read` and briefings) shows local time, and
model-bound text names the zone once at its top. An empty result is a normal result. Unknown
values are `null` in JSON and "—" in text, never `0`. An error names its
kind (`query.md`, Results) and a message that names the reference involved; a malformed reference or argument is quoted to at most its first
200 characters plus the BLAKE3 of the whole value; over MCP and HTTP it
never includes local paths or stack traces.

Bead and audit IDs behind each section: `evidence/beads/spec-sources-g3.md`.

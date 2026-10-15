# Decision: how commands talk to the daemon

Status: **decided** (operator correction, 2026-10-11). Every read and every
write goes through the daemon over its one local socket (the web reader's
HTTP routes are the same requests). The daemon holds warm
state, so commands answer fast, and it is the only process that writes any
of the three databases. When the daemon is not running, commands say so and
how to start it; there is no fallback read path and no standalone mode.

## Why

- **Snappy answers need warm state.** A fresh process that opens SQLite,
  compiles a query and, for `near:`, loads vectors pays that cost on every
  command. A resident daemon pays it once: embedding vectors in memory,
  compiled query plans, recent results and cursors for paging (an anchor, not
  a held snapshot) (`spec/behavior.md`, warm state). Live state is not cached:
  it is computed per read from stored facts and the clock.
- **The daemon is the standard.** The operator: "There is no requirement or
  need for CLI to ever work standalone. Daemon is the standard"
  (2026-09-14). The 2026-09-25 normative specification chose the same shape:
  a thin query-first CLI, the resident daemon, bounded caches
  (`applied-specification/specs/01-resident-read-runtime.md`).
- **One process per archive, not one per agent.** The Python MCP server
  opened the archive itself and cost about 250 MB per agent session; a thin
  MCP proxy over the socket costs almost nothing.
- **One writer.** With the daemon writing everything, notes and tags
  included, there is no writer coordination between processes at all.

## Writer lifetime

Acquire the archive lock before opening writable state or changing the socket. Keep it until every writer-capable task and admitted client handler has stopped. Shutdown stops admission, closes client connections to wake blocked reads and sends, drains in-flight mutations, then joins the owned tasks before releasing the lock. A replacement daemon must never overlap with writable state retained by an old client. Dropping an in-process daemon owner performs the same cleanup as explicit stop. A lost reply still uses the existing request-ID retry rule; shutdown does not invent a second delivery ledger.

## The request shape, and what keeps it small

The Python CLI-to-daemon protocol grew to 15–25k lines (audit,
`audit/bloat-patterns.md`) because it declared one operation per command in a
registry, with a preview and authorize ceremony, receipts and per-operation
contracts. The rewrite has none of that:

- **One request shape**: a query in the query language with a unit, a verb
  from the verb table of `spec/surfaces/cli.md` and output options; or one
  command from that file's commands table that is not purely local. Those
  two tables are the only list of what a request can do. The verb and command
  tables are the closed list: a new capability is a query field, a verb or a
  row in those tables, generated into the request type, never a separate
  protocol operation. Every request may carry `deadline_ms`. The arguments
  are the same as the MCP tools'. The answer is the result shape in
  `spec/query.md` (Results) or the export format.
- **Writes carry their own ID.** Every write request carries a `request_id`
  chosen by the client (a UUIDv7), generated once per user action and reused
  on retry; the MCP proxy generates one per tool call.
  The rows a request creates take their IDs from it (`note.id`, the
  excision's ID), so a resent request finds them and writes nothing; a resent
  ID with different content is refused with `conflict`. A client whose write
  is indeterminate (the reply was lost) resends the same request. A queued
  write whose client deadline passed before it started is dropped. This is
  the delivery-ID rule applied to every write, not a receipt: beyond the
  rows it writes, the only thing stored is one `client_request` row (the ID,
  the hash of the request's content and its time;
  `spec/storage/archive-db.md`), which is how a resent ID with different
  content is told from a true resend.
- **No operation registry, receipts or contracts**: the closed tables above
  are the whole protocol.
- **Who may excise.** The daemon cannot tell an agent from the operator on
  its socket; the excise rule is enforced by the CLI's terminal check and by
  MCP exposing no excise tool. A process running as the operator can bypass
  both, which `spec/behavior.md`, untrusted input, places out of scope.
- **One transport for clients**: newline-delimited JSON over the Unix
  socket `$XDG_RUNTIME_DIR/polylogue/<key>/socket` (`<key>` as in
  `spec/surfaces/web.md`), readable only by the operator. The CLI, the MCP
  servers (stdio proxies), the hook command, the browser host and the
  Python client (through the CLI) all use it; none opens a database file.
- **The web reader is part of the daemon**, not a client: the daemon serves
  its pages and their JSON routes over loopback HTTP for the browser only,
  and each route is the same request as on the socket. No other program
  uses the HTTP port.

Target size: about 1–2k lines for the socket server, request parsing and the
client.

## When the daemon is not running

Every command prints one line, "the Polylogue daemon is not running; start
it with `systemctl --user start polylogue` (or `polylogue daemon`)", and
exits with code 6 (`spec/surfaces/cli.md`). In slice 1 the message names only
`polylogue daemon`; the systemd unit and its wording arrive in slice 2. The hook command exits 0 without injecting a briefing, so a
harness is never slowed or broken. Static `--help`, `--version`, `config schema` and shell
completion of command and field names work without the daemon; `config schema`
exports the compiled configuration schema without opening an archive or connecting
to a socket, so package/module generation works before installation; completion
of live values (tags, repos, session IDs) asks the daemon and offers nothing
when it is down.

## Rejected

- **Commands read SQLite directly and write intent rows to `archive.db`**
  (this file's previous decision). It avoided a protocol, but every command
  paid cold-start costs, `near:` could not use vectors held in memory, each
  MCP server held its own copy of the archive in memory, and two kinds of
  process wrote `archive.db`. The bloat it guarded against came from the
  registry and ceremony, not from having a socket.
- **Reads direct, writes over a socket.** Two paths for one archive; still
  cold reads.

# Polylogue

Polylogue keeps every AI conversation and agent session your tools leave
behind (Claude Code, Codex, Gemini CLI, Grok CLI, Hermes, and exports from
ChatGPT, claude.ai, AI Studio, the Gemini app, Grok account exports and others)
in one local archive, exactly as written, and lets you and your agents search, read,
measure and annotate it from the command line, over MCP or in a web reader.

This repository is the Rust rewrite. The Python predecessor is kept for
reference; the rewrite uses its own archive.

## Build

```sh
nix build .#polylogue          # or: nix develop -c cargo build -p polylogue
./result/bin/polylogue --help
```

## Run

Start the daemon, then talk to it from another terminal:

```sh
polylogue daemon                     # collects from every tool's own location
polylogue import ~/Downloads/chatgpt-export.zip
polylogue repo:polylogue since:7d    # a query: sessions in a repository this week
polylogue read claude-code:SESSION   # read one session
polylogue measure usage since:30d    # tokens and cost
```

The daemon owns the archive: it holds the archive lock, runs the loop that
collects and parses every input, runs the only writer, and serves clients
over a Unix socket. Commands that read or write the archive go through a
running daemon; without one they exit with code 6.

The daemon reads each tool's files where the tool writes them
(`$CLAUDE_CONFIG_DIR` or `~/.claude`, `$CODEX_HOME` or `~/.codex`, and so on);
exports come in only by `import`. Every version of every input is kept, so
a file the tool deletes or rewrites stays in the archive.

Grok account JSON exports retain native conversation and response IDs when
supplied. Without a conversation ID, complete conversation content determines
a marked derived ID: filenames and response ordering do not change it, while
an append or edit can create another session because continuity is unproved.
Repeated identical turns remain distinct. Unknown authors or structured message
shapes remain in the retained input with import diagnostics, rather than being
coerced into text messages.

The archive is `$XDG_DATA_HOME/polylogue` (`~/.local/share/polylogue`). It
holds `archive.db` (what cannot be recomputed: inputs and their revisions,
notes, excisions, settings, delivery records), `index.db` (everything parsed
and derived, rebuilt from the stored inputs when it is deleted or its format
changes), `embeddings.db` and `blobs/`.

## Commands

`polylogue --help` lists CLI commands; each takes `--json`. One command table
(`crates/polylogue/src/commands.rs`) declares them; the CLI, its
completions, the MCP tools and the HTTP API under `/api/v1/` are built from
it.

The browser action commands are HTTP-only table entries: JSON enqueue at
`/api/v1/browser-actions` and binary upload at `/api/v1/browser-actions/attachments`.
They retain token/origin checks and a 2 MiB body limit. The daemon owns enqueue
permission and idempotence; the extension still cannot enqueue. Uploads remain
binary through the HTTP adapter, and neither operation is offered through CLI or MCP.

| Command | What it does |
|---|---|
| `QUERY` | Run a query: sessions, messages, documents, blocks, tool calls, files, events, notes, links, turns, runs, deliveries, processes, work and changes, with conditions, `\| sort by`, `\| group by`, `\| matches` and `\| measure`. `--why` explains an empty result; `--fields` picks columns; `@N` names row N of the last list. |
| `status` | What the archive holds and what the daemon is doing. |
| `daemon` | Run the daemon in the foreground. |
| `list`, `read SESSION` | List sessions; read one in a view (`full`, `dialogue`, `prompts`, `actions`). |
| `read document:ID` | Read retained instructions, supplied context or memory without conversation roles. `full` and `dialogue` views, all formats and budgeted pages are available. |
| `import PATH` | Bring in an export or a directory of them. |
| `export SESSION…` | Write sessions or `document:ID` references as JSON (read back by `import`), Markdown, HTML or terminal text. |
| `note add\|tag\|withdraw\|list` | Notes and tags on sessions, messages, documents, tool calls, repositories and directories, with revisions. |
| `keep QUERY` | Keep a query's rows, to read back later as `result:<id>`. |
| `brief`, `takeover SESSION`, `recall WORDS` | Context for a model: the session-start brief, what to take over another session with, and a search of past work with a record of how it searched. |
| `continue [SESSION]` | Resume a session with its tool's own command; without one, the sessions in this directory that left something open. |
| `measure usage\|cost\|sessions\|quota\|outcomes\|failing\|drift\|unanswered\|survival\|survival-turns\|daily\|process\|rhythm` | Tokens by kind and cost, subscription quota windows, tool outcomes, commands failing now, format drift and other analyses. Unknown figures stay unknown, never 0. |
| `live`, `notices`, `work` | Sessions running now, signal notices and the work graph. |
| `changes [--since POSITION]` | A finite JSON-lines batch of session/note/removal references and excision selections. Apply the complete batch before persisting its final position. |
| `sql STATEMENT` | Read-only SQL over the stable views. |
| `excise TARGET` | Remove a session, input or text. [Text excision and supported representations](docs/excision.md). |
| `embed preview\|build\|status` | Opt-in embeddings for semantic search, with a cost preview. |
| `setting KEY [VALUE]` | Read or set a setting. |
| `web` | Open the web reader. |
| `mcp` | Serve MCP over stdio. |
| `hooks install\|uninstall`, `hook EVENT` | Claude Code hooks: live state and the session-start brief. |
| `install-agent chrome` | Register the browser extension's native messaging host. |
| `completions SHELL` | Shell completions for bash, elvish, fish, PowerShell or zsh. |
| `sinex status\|cutover` | Not built yet, so hidden from help: the optional Sinex backend's state and the move to it. |

Exit codes: 0 success, 1 an error the daemon reported or an I/O error, 2 an
invalid request, 6 no daemon running.

Credentials in session text (API keys, tokens, private keys) are masked in
every output; the stored bytes are not changed.

Budgeted session reads (`read SESSION --budget N`) return a continuation in `next`.
Pass it as `--cursor`; the budget may change between pages. Changed selected
content returns `stale_cursor` and requires starting again. See
[session paging](docs/architecture.md) for cursor binding, masking and memory costs.

Claude.ai account `memories.json`, project `projects/*.json` and older
`projects.json` exports retain textual memories, instructions and supplied
project documents as `documents`. Search them with
`polylogue 'documents where origin:claude-ai "phrase"'`, then use
`polylogue read document:ID` (or `document:@N` after a terminal list).
Their authorship stays unknown unless the source establishes it; they add no
conversation turns or model usage. Missing document IDs are scoped to the
retained input and physical record position. A different input cannot prove
continuity for these records. Binary and unresolved references remain in raw
inputs with interpretation diagnostics.

JSON exports containing documents use `polylogue-export` version 2; session-only
exports retain version 1. Both can be imported. Document removals use
`excise document:ID --yes`, keep unaffected sibling values readable and retain a
standing exclusion through rebuild and Sinex restore. Archive format 7 stores
that policy; older local readers refuse it. The derived index rebuilds at
version 31 without changing existing message hashes or authored references.

Logical reads (`read SESSION --logical`) include only proven inherited prefixes.
A fork's anchor is resolved in its parent's composed history, including that
parent's inherited messages. Missing parents or anchors and session cycles keep
available content readable with `lineage.complete: false` and typed `issues`.
Every model page carries that status; prose formats show a notice. Continuation
identity includes lineage status, so a changed missing link invalidates a page.
Takeover context lists missing ancestry in its omissions. Logical totals refuse
incomplete ancestry instead of presenting the available subset as complete.

Session reads and query rows use the stored display title: the recorded title
when present, otherwise the derived first-prompt title. JSON reads carry this
as `display_title`, separate from the recorded `session` metadata. Importable
exports retain source fields.

The web reader links Document search results to `/d/ID`, with `full` and
`dialogue` views, copied CLI commands, and masked source metadata including
retained input ranges and versions. **Browse Documents** opens the document
query from the overview. Query column descriptors identify supported reader
links; message, note and untyped values remain text.

The web reader's `/cost` page runs the same physical usage report as
`measure usage`, including unknown usage, unpriced responses, independent
recorded observations and the price catalog. Enter a session query to narrow
its scope, or follow **Session cost and usage** from a session. **Browse selected
sessions** returns to the query reader. API-equivalent cost and tool-reported
cost are separate figures.

Session pages show logical ancestry completeness, unresolved ancestry issues,
and links to related sessions with the retained relationship evidence and branch
point when available. Missing ancestry remains explicit. These are session
relationships; the reader does not yet expose exact retained input versions and
raw source ranges.

## Repository layout

- `crates/model`: shared types (identities, sessions, messages, blocks,
  events, notes, prices) and the export format; no I/O.
- `crates/sources`: one declaration and one parser per origin, on a shared
  parsing framework.
- `crates/store`: every SQLite connection and write, and the blob store.
- `crates/engine`: the loop that collects, stores, parses and derives, and
  its parse pool.
- `crates/query`: the query language, compiled to SQL.
- `crates/render`: one stream of rendered blocks in terminal, Markdown, HTML
  and JSON, and pages sized for models.
- `crates/polylogue`: the binary: daemon, CLI, MCP, web reader, hooks and the
  browser extension's host.
- `crates/sinex`: the Sinex backend, behind the `sinex` feature.
- `crates/testkit`: test archives built through the real loop, including
  the synthetic example archive in `examples/morrow/`.
- `examples/sources/`: golden parser cases, one directory per origin.
- `extension/`: the browser extension.

The content-addressed blob store (`hashed-blobs`), credential masking and
the core of `xtask` (`devkit`) come from
[polylogue-sinex-common](https://github.com/Sinity/polylogue-sinex-common),
shared with Sinex.

[docs/development.md](docs/development.md) describes the development checks
and [docs/REPLACEMENT.md](docs/REPLACEMENT.md) says what replaced the Python
version.

## Sinex startup failures

With the `sinex` feature, `sinex.toml` selects the backing. A missing file uses
standalone. Malformed configuration, an unreadable local publication inventory,
or a failed required restore pauses startup work without choosing another
backing. `status.backing` shows `state: blocked`, the configuration path and the
error. Existing reads, queries, exports, notes, read-only settings and context
without a delivery sender remain available. Archive mutations return
`backing_unavailable`; collection, signals and publishing do not start.

Correct the configuration or restore failure and restart the daemon. Source
capture spools remain pending until collection can run. An existing nonempty
primary archive can start and serve reads while Sinex is offline.

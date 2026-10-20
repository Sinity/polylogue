# Polylogue

Polylogue keeps every AI conversation and agent session your tools leave
behind (Claude Code, Codex, Gemini CLI, Grok CLI, Hermes, and exports from
ChatGPT, claude.ai, AI Studio, the Gemini app and others) in one local
archive, exactly as written, and lets you and your agents search, read,
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

The archive is `$XDG_DATA_HOME/polylogue` (`~/.local/share/polylogue`). It
holds `archive.db` (what cannot be recomputed: inputs and their revisions,
notes, excisions, settings, delivery records), `index.db` (everything parsed
and derived, rebuilt from the stored inputs when it is deleted or its format
changes), `embeddings.db` and `blobs/`.

## Commands

`polylogue --help` lists them all; each takes `--json`. One command table
(`crates/polylogue/src/commands.rs`) declares them; the CLI, its
completions, the MCP tools and the HTTP API under `/api/v1/` are built from
it.

| Command | What it does |
|---|---|
| `QUERY` | Run a query: sessions, messages, blocks, tool calls, files, events, notes, links, turns, runs, deliveries, processes, work and changes, with conditions, `\| sort by`, `\| group by`, `\| matches` and `\| measure`. `--why` explains an empty result; `--fields` picks columns; `@N` names row N of the last list. |
| `status` | What the archive holds and what the daemon is doing. |
| `daemon` | Run the daemon in the foreground. |
| `list`, `read SESSION` | List sessions; read one in a view (`full`, `dialogue`, `prompts`, `actions`). |
| `import PATH` | Bring in an export or a directory of them. |
| `export SESSION…` | Write sessions out as JSON (read back by `import`), Markdown, HTML or terminal text. |
| `note add\|tag\|withdraw\|list` | Notes and tags on sessions, messages, tool calls, repositories and directories, with revisions. |
| `keep QUERY` | Keep a query's rows, to read back later as `result:<id>`. |
| `brief`, `takeover SESSION`, `recall WORDS` | Context for a model: the session-start brief, what to take over another session with, and a search of past work with a record of how it searched. |
| `continue [SESSION]` | Resume a session with its tool's own command; without one, the sessions in this directory that left something open. |
| `measure usage\|cost\|sessions\|quota\|outcomes\|failing\|drift\|unanswered\|survival\|survival-turns\|daily\|process\|rhythm` | Tokens by kind and cost, subscription quota windows, tool outcomes, commands failing now, format drift and other analyses. Unknown figures stay unknown, never 0. |
| `live`, `notices`, `changes`, `work` | Sessions running now, signal notices, what changed since a cursor, the work graph. |
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

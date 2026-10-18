# Polylogue

Polylogue keeps a local archive of your AI coding sessions (Claude Code and
Codex) and lets you list, search, read and annotate them from the command line
or through MCP.

This repository is the Rust rewrite. The Python Polylogue is still the deployed
version; the rewrite runs against fresh, separate archives.

## What works today

- **Collecting sessions.** `import` copies one Claude Code or Codex file into the
  archive. The daemon can also follow one configured source: a Claude Code
  session file, a Claude Code projects directory, or a Codex home.
- **Listing and search.** List known sessions and search the human and model
  text of their messages by words, phrases and prefixes.
- **Reading.** Read a whole session as JSON or terminal text, read one message,
  or read in pages sized to a token budget (for agents).
- **Export.** Export one session as JSON into a directory.
- **Notes and corrections.** Attach a note or correction to a message or block,
  revise it, or withdraw it. Notes keep their full history.
- **MCP.** `polylogue mcp` serves `status`, `search`, `read` and `note` over
  stdio.
- **Recovery.** `recover` copies a stopped archive whose last write-ahead log
  segment is damaged; `daemon --prepare-copy restore` reopens a stopped local
  copy of an archive.

Credentials in session text (API keys, tokens, private keys and similar) are
masked in every output. The stored files are not changed.

Not done yet: automatic format detection, a general query language, export of
more than one session, snippets in search results, and an HTTP interface.

## Build

```sh
nix build .#polylogue          # or: nix develop -c cargo build -p polylogue
./result/bin/polylogue --help
```

## Run

Start the daemon, then talk to it from another terminal:

```sh
polylogue daemon
polylogue import ~/.codex/sessions/2026/10/01/rollout-....jsonl --format codex-rollout --wait
polylogue                         # list sessions
polylogue find "flaky test"       # search message text
polylogue SESSION_ID read         # read a session in the terminal
```

The daemon owns the archive: it holds the archive lock, runs the only writer,
and serves clients over a socket under `$XDG_RUNTIME_DIR` (which must be set
and absolute). Ctrl-C stops it cleanly. Every other command except `--help`,
`--version` and `config schema` needs a running daemon.

The archive is `$XDG_DATA_HOME/polylogue`, or `~/.local/share/polylogue`;
`--archive DIR` selects another one. Use a fresh archive for the rewrite; it
refuses archives written by newer versions and older formats it cannot
migrate.

### Commands

| Command | What it does |
|---|---|
| `daemon` | Run the archive daemon in the foreground. |
| `daemon --config FILE [--until-idle] [--json]` | Run the daemon and collect from one configured source. `--until-idle` stops once the startup snapshot is stored (not available for a projects directory). `config schema` prints the file's JSON schema. |
| `import PATH --format FORMAT [--context FILE] [--wait]` | Store one file. Formats: `claude-code-jsonl`, `codex-rollout`, `codex-history-sqlite`. Claude files need `--context FILE`. `--wait` waits until the file's messages are parsed. |
| `progress INPUT` / `retry INPUT` | Show parsing progress for an imported file, or retry a failed parse. |
| `input cat INPUT` | Print a stored file, with credentials masked. |
| `[QUERY] [--json]` | List sessions. A query may use `origin:`, `kind:`, `id:`, words, `"phrases"`, `prefix*`, `\| sort by last_active desc` and `\| limit N`. |
| `find QUERY [--json]` | Search message text. |
| `SESSION read [--format json\|terminal]` | Read a whole session. |
| `message:ID read` | Read one message in the terminal. |
| `REF read --budget TOKENS` | Read a session, message, block or note as JSON pages that fit the budget. Follow `next` for the following page. `--view dialogue` keeps only human and model text; `--view transcript --tools full --thinking full --harness full` includes everything. |
| `SESSION export --format json --to DIR` | Export one session into DIR. |
| `REF note TEXT [--kind note\|correction] [--targets FILE]` | Add a note to a `message:` or `block:` reference. `--targets` names the target and the `seen_hash` you read (a JSON array of `{ref, seen_hash}`; a read returns it as `target_hash.seen_hash`), so later reads can tell when the text changed. |
| `note:ID revise TEXT` / `note:ID withdraw --reason TEXT` | Change or withdraw a note. |
| `mcp` | Serve MCP over stdio. |
| `recover --to NEW_DIR` | Copy a stopped, damaged archive into a new directory. |

`import` and note commands print a request ID before sending. After a lost
reply, repeat the command with the same `--request-id` (and, for `import`, the
printed `--seen-at`); the archive applies the action once.

### Exit codes

| Code | Meaning |
|---|---|
| 0 | Complete result. |
| 1 | Nothing matched. |
| 2 | Invalid or not-yet-supported request. |
| 3 | Refused. |
| 4 | Not found, conflict or I/O error. |
| 5 | Partial result: usable, but some data is missing; the reply names what. |
| 6 | The daemon is not running. |
| 7 | Timed out. |

JSON replies carry the same outcome in `outcome` (`ok`, `empty`, `degraded`,
`error`).

## Repository layout

- `crates/model`: identity, value and outcome types; no I/O.
- `crates/store`: the SQLite archive. Nothing else opens a database.
- `crates/hashed-blobs`: the content-addressed file store shared with Sinex
  ([release process](docs/shared-file-layer-release.md)).
- `crates/sources`, `crates/engine`: provider parsers and the work that turns
  stored files into sessions and messages.
- `crates/query`: the query parser.
- `crates/credential-output`: credential masking for output.
- `crates/polylogue`: configuration, daemon, CLI and MCP.

[docs/development.md](docs/development.md) describes the development checks,
[docs/rewrite/](docs/rewrite/README.md) has notes on how parts work, and
[docs/REPLACEMENT.md](docs/REPLACEMENT.md) lists what the Python version had
and where it went. The design is in the Polylogue spec repository.

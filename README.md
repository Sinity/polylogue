# Polylogue

Polylogue keeps every AI conversation and agent session your tools leave
behind (Claude Code, Codex, Gemini CLI, Grok CLI, Hermes, and exports from
ChatGPT, claude.ai, AI Studio, the Gemini app, Grok account exports and others)
in one local archive, exactly as written, and lets you and your agents search, read,
measure and annotate it from the command line, over MCP or in a web reader.

This repository is the Rust rewrite. The Python version is kept for
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
running daemon; without one they exit with code 6. Offline `schema` commands
read explicitly selected stored revisions directly.

The daemon reads each tool's files where the tool writes them
(`$CLAUDE_CONFIG_DIR` or `~/.claude`, `$CODEX_HOME` or `~/.codex`, and so on);
exports come in only by `import`. Every version of every input is kept, so
a file the tool deletes or rewrites stays in the archive.

The archive is `$XDG_DATA_HOME/polylogue` (`~/.local/share/polylogue`). It
holds `archive.db` (what cannot be recomputed: inputs and their revisions,
notes, excisions, settings, delivery records), `index.db` (everything parsed
and derived, rebuilt from the stored inputs when it is deleted or its format
changes), `embeddings.db` and `blobs/`.
Opening an `archive.db` written by an older build applies the migrations it
lacks, in one transaction. The daemon refuses an archive newer than itself,
or one whose tables match no version it knows, and exits with code 78; a
service manager should not restart on that code (systemd:
`RestartPreventExitStatus=78`).

## Commands

`polylogue --help` lists the commands; each takes `--json`. One command
table (`crates/polylogue/src/commands.rs`) declares them, and the CLI, its
completions, the MCP tools and the HTTP API under `/api/v1/` are built from
it.

| Command | What it does |
|---|---|
| `QUERY` | Run a query over sessions, messages, documents, blocks, tool calls, files, events, notes, links, turns, deliveries, work or changes, with conditions, `\| sort by`, `\| group by`, `\| matches` and `\| measure`. Queries combine: `\| union (…)`, `\| intersect (…)`, `\| except (…)`, `\| fuse (…)`, `count(UNIT where …) >= N` and `FIELD in (… \| group by FIELD)`. `--why` explains an empty result; `--fields` picks columns; `@N` names row N of the last list. |
| `demo tour DIRECTORY` | Walk through ordinary commands over a separate synthetic Morrow archive. |
| `explain SUBJECT` | Explain query grammar or page through current units, fields, operators and capabilities. |
| `status` | What the archive holds and what the daemon is doing; during a long pass the counts come from memory, with the read position they are from. |
| `daemon` | Run the daemon in the foreground. |
| `list`, `read SESSION` | List sessions; read one in a view (`full`, `dialogue`, `prompts`, `actions`). `read document:ID` and `read event:ID` read a document or an event. |
| `outline SESSION` | Page references to human prompts, recorded tool failures, compactions and parent branch points without reading message bodies. |
| `import PATH` | Store files or the files inside an export ZIP, and parse the ones in a recognized format. |
| `attachment MESSAGE BLOCK` | Describe a message attachment and the stored export files that match it. In a terminal, `--reveal --out FILE` writes the verified bytes to FILE ([attachments](docs/attachments.md)). |
| `export SESSION…` | Write sessions, documents or events as JSON (read back by `import`), Markdown, HTML or terminal text. |
| `note add\|tag\|withdraw\|list` | Notes and tags on sessions, messages, documents, tool calls, repositories and directories, with revisions; `--query QUERY` notes every row a query returns. |
| `note read NOTE` | Read one exact note revision, including superseded and withdrawn revisions. |
| `note accept\|reject NOTE` | Judge a note an agent wrote: an accepted note reaches briefs as the operator's own. `note list --agents --judged none` lists the notes waiting. |
| `save NAME QUERY` | Save a query to run as `query:NAME`; a `$word` in it is a parameter, given as `query:NAME(word=VALUE)`. |
| `watch QUERY` | Report each row that starts matching as a notice (`--for 20m` holds it first, `--notify` also shows it on the desktop). |
| `keep QUERY` | Keep a query's rows, to read back later as `result:<id>`. |
| `brief`, `takeover SESSION`, `recall WORDS` | Context for a model: the session-start brief, what to take over another session with, and a search of past work (words and, with vectors, similarity, ranked together, by pages) with a record of how it searched. |
| `continue [SESSION]` | Resume a session with its tool's own command; without one, the sessions in this directory that left something open. |
| `measure usage\|cost\|sessions\|quota\|outcomes\|failing\|drift\|unanswered\|survival\|survival-turns\|daily\|process\|rhythm` | Tokens by kind and cost, subscription quota windows, tool outcomes, commands failing now, format drift and other analyses. Unknown figures stay unknown, never 0. |
| `timeline [--date DATE]` | Recorded activity, logical-day calendar counts, this day across years and undated records ([day pages](docs/web-calendar.md)). |
| `cost [--query QUERY] [--date DATE]` | Selected-session usage and paged physical responses, optionally in one logical day, with model/day drill-downs and explicit unknown source times ([cost](docs/web-calendar.md)). |
| `live`, `notices`, `work` | Sessions running now, signal notices and the work graph. `status` also shows the last day's notices. |
| `changes [--since POSITION]` | The change feed: JSON lines naming changed sessions, notes, removals and excisions since a position, ending with the next position. |
| `sql STATEMENT` | Read-only SQL. The views `v1_sessions`, `v1_messages`, `v1_tool_calls`, `v1_links` and `v1_notes` keep their names and columns ([crates/store/src/sql_views.rs](crates/store/src/sql_views.rs)); the tables under them change with the index format. |
| `excise TARGET` | Remove a session, document or input whole, with everything parsed from it ([excision](docs/excision.md)). |
| `embed preview\|build\|status` | Opt-in embeddings for semantic search, with a cost preview. |
| `setting KEY [VALUE]` | Read or set a setting. `day.zone` (an IANA zone; the machine's unless set) and `day.logical_start` (default `04:00`) define the logical day that dates, `today` and `group by day` mean. |
| `web` | Open the web reader. |
| `mcp` | Serve MCP over stdio. |
| `hooks install\|uninstall`, `hook EVENT` | Harness hooks for Claude Code, Codex (`--harness codex`) and Hermes: live state, the session-start brief and takeovers. |
| `chat send PROVIDER TEXT`, `chat list\|approve\|decline` | Have the browser extension draft a turn in a web chat, or send it with `--submit`; list chat actions; approve or decline a held send ([extension](extension/README.md)). |
| `backfill start PROVIDER`, `backfill list\|cancel` | Have the browser extension capture a web chat's older conversations; follow or stop the job. |
| `install-agent chrome` | Register the browser extension's native messaging host ([extension](extension/README.md)). |
| `schema infer ORIGIN --input ID`, `schema diff BEFORE AFTER` | Offline structure/count inference over explicit stored revisions and recursive schema comparison ([schema tools](docs/schema-tools.md)). |
| `completions SHELL` | Completion for bash, elvish, fish, PowerShell or zsh: commands and options, and while you type a query, its units, fields, values seen and recent session IDs, which the shell asks of the daemon through `complete WORD`. |

Exit codes: 0 success, 1 an error the daemon reported or an I/O error, 2 an
invalid request, 6 no daemon running.

## Try Morrow and discover the grammar

`polylogue demo tour ./morrow-tour` creates a new directory containing a
synthetic home, a separate archive and `transcript.md`. It walks through
searching and reading a session, checking tool failures, following parent
work, inspecting unknown costs, and adding notes for takeover context.
The example files are bundled in the binary. Preparation time and first
query time are reported separately. Use `--json` for the steps and results.
Run the displayed commands against the example archive by starting

```sh
tour_directory="$(realpath ./morrow-tour)"
export POLYLOGUE_ARCHIVE="$tour_directory/archive"
export POLYLOGUE_HOME="$tour_directory/home"
export XDG_CONFIG_HOME="$tour_directory/config"
polylogue daemon
```
in another terminal, and setting the same variables for each command.

`polylogue explain query --expression 'tool_calls where outcome:error'`
parses and compiles the expression without running it. The answer includes
the parsed conditions and stages, output unit and columns, ordering,
parameters and SQL in the full compiled `program`. It uses the archive's
logical day, saved queries, kept results and embedding availability. Supply
`--session ID` for `self` references. This differs from `--why`, which executes a
query and counts how many rows its conditions keep.

The MCP `explain` tool uses the same arguments. Subjects `units`, `fields`,
`syntax`, `operators` and `capabilities` describe current declarations. For example,
`polylogue explain fields --unit tool_calls --page-size 10` returns entries and
`next`; pass that value as `--cursor` with the same subject and unit.
`page_size` controls page size. The units subject lists session-field inheritance and relation prefixes; the fields subject lists each unit's own fields. Capabilities include command schemas,
read views and output formats from their registries.

## Context for agents

Each harness's start hook brings the session-start brief into the agent's
context, on by default (`polylogue setting briefing.session_start off`
turns it off). The brief carries the operator's notes on the directory and
its repository, other agents working there now, Beads claimed there this
week and still open, what the recent sessions there did (commits, pull requests,
files changed, checks, plan items left open, the last answer), checks that
failed there with no later pass (the `xtask` check and test runs devkit
records for the checkout, which AgentCTL's jobs run, and test, build and
lint commands sessions ran), and the last day's signal notices.
It stays within `briefing.budget` tokens (default 2000) and names what it
left out with the command that shows it. A session ID pasted into a prompt
brings that session's takeover context.

| Harness | Hook entries |
|---|---|
| Claude Code | `polylogue hooks install` writes them into `settings.json`. |
| Codex | `polylogue hooks install --harness codex` writes them into `hooks.json`. |
| Hermes | In `config.yaml`: `hooks: {pre_llm_call: [{command: "polylogue hook pre_llm_call --harness hermes"}]}` and `hooks_auto_accept: true`; the brief comes with the first turn. |

A signal is a note of kind `signal` whose body names a watched query, such
as `{"name": "agent waiting", "query": "sessions where live:waiting", "for": "20m"}`;
`polylogue watch` writes one. While the daemon's first pass catches the
index up, and whenever a query's answer is incomplete, a watch is held as it
was rather than read as rows appearing or leaving.
Each notice it fires shows in `status`, the next briefs and `notices`; a
rule the operator writes also shows it as a desktop notification through
`notify.command` (default `notify-send`), within its quiet hours and rate
limit.

## Search

Words in a query match full text and rank the rows by relevance. With
embeddings built, `near:TEXT` ranks rows by similarity to TEXT: the query's
other conditions choose the rows first and every row they keep is ranked,
and with words in the same query both orders are fused (reciprocal rank
fusion). The answer counts rows without a vector (`unknown.near`); with
`near:` alone they follow the ranked rows, and in hybrid search they still
rank by full text. A count or grouping after `near:` computes no similarity
unless an earlier selection used the ranking. `| union (…)` adds another
query's rows; when either side is ranked, the rows rank by fusing every
list's order, and `found_in`
names the lists that ranked each row. `| fuse (…)` ranks the rows also by
another query's order without adding any. Ranked answers page by cursor like
any other. `recall` is one such query: every word in prose, every word in
tool output and similarity to the whole text, with the current directory's
sessions ranked in one more list.

The daemon keeps warm what common answers need: the vectors in memory, the
vectors of recent `near:` texts, recent answers by read position, and
completion data, each dropped when a write changes what it was read from.
`status` shows what it holds under `warm`.

## Reading

Credentials in session text (API keys, tokens, private keys) are masked in
every output; the stored bytes are not changed.

A budgeted read (`read SESSION --budget N`) returns a page sized for a model
and a cursor in `next`; pass it as `--cursor` for the next page. Query pages
work the same way: the cursor holds the last row shown.

`read SESSION --with-parents` reads a session together with the history it
continues from (a fork's parent up to the fork point, and so on). Missing
parents or cycles keep what is available readable and mark the result
`parents.complete: false` with the issues.

Sessions show the title the tool recorded, or else one made from the first
prompt (`display_title` in JSON).

`first_seen` is when Polylogue first had a message's bytes (a session's:
its first message's). `evidence_until:TIME` limits messages, blocks, tool
calls, events and notes to what Polylogue had by then, read with today's
parsers; a session's figures are not recomputed as they stood then.

Text that is not a conversation (claude.ai memories, project instructions
and project documents, Hermes transcripts) is stored as documents:
`polylogue 'documents where origin:claude-ai "phrase"'` searches them and
`read document:ID` reads one. Their author stays unknown unless the source
says. Hermes activity records are events, read with `read event:ID`.

The web reader shows sessions, documents, a usage and cost page (`/cost`,
the same report as `measure usage`) and the links between related sessions.
Sessions and documents page their content. Message links open at the selected
message, and the session outline pages prompt, failure and branch references.
See [the shared reader contract](docs/web-reader.md) for API and access details.

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
- `crates/sinex`: primary mode on Sinex, behind the `sinex` feature.
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

## Sinex primary mode

With the `sinex` feature (on in the Nix package),
`$XDG_CONFIG_HOME/polylogue/sinex.toml` selects the mode. A missing file means
standalone. With `mode = "primary"`, Sinex holds the archive and Polylogue is
one of its producers:

```toml
mode = "primary"
nats = "nats://127.0.0.1:4222"
# The NATS user is the producer's name, `polylogue` unless `producer` says
# otherwise. The password stays in its own file, never in this one.
password_file = "/run/agenix/sinex-nats-polylogue"
# Sinex's store root; Polylogue writes the bytes it sends under it, so its
# user needs the store's group.
store = "/var/lib/sinex/daemon/store"
```

If the configuration is malformed or Sinex cannot be reached at start, the
daemon starts without collecting: `status.mode` shows `state: blocked`, the
configuration path and the error, reads and queries work, and archive writes
return `sinex_unavailable`. Fix the cause and restart the daemon.
[docs/architecture.md](docs/architecture.md#sinex-primary-mode) describes
how primary mode stores the archive and how a standalone archive moves to it.

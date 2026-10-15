# Command line

All examples run against the Morrow example archive
(`examples/archive/FACTS.md`) with the clock fixed at **2026-10-08 12:00
Europe/Warsaw** (tests set `POLYLOGUE_NOW`). Message texts quoted here are
part of the example archive; the raw files are written to contain them.

This file is the one authoritative list of the CLI: grammar, verbs,
commands and options. Other files may use only what is listed here. Command
and verb names are final when slice 1 ships and are not renamed afterwards:
there are no compatibility aliases (principles), and the Python version's
renames caused much of the operator's 45% failure rate at the shell.

## Grammar

```
polylogue                              the overview (section "Bare polylogue" below)
polylogue [QUERY] VERB [args] [options]  run a query, then a verb on the matched set
polylogue QUERY [options]              same, with the default verb `list`
polylogue find QUERY [VERB …]          the same, with the first word read as a query even if it is a command name
polylogue COMMAND [args] [options]     commands that are not about a matched set
```

One rule decides how arguments are read:

1. If the first argument is a command name, it is that command.
2. Otherwise the query is the arguments before the verb, and the verb is
   the first verb name followed only by that verb's own arguments and
   options (several shell arguments are joined with spaces; quote anything
   containing `|`, `(`, `)` or `>` for the shell). An unquoted verb name
   anywhere else in an unquoted multi-word query (`polylogue how to read
   files`) is refused with exit 2 and the quoted form as the suggestion,
   never split into a query and a verb (`query.md`, principle 2). The query
   language is `query.md`: plain words search text, `field:value` filters.
3. A query whose first unquoted word is a command or verb name, or a near
   miss of a command, verb or unit name (one typing mistake for a word of
   4–6 letters, two for 7 or more, none under 4, counted as
   Damerau–Levenshtein distance on the typed word, so a swapped pair of
   letters is one mistake; `query.md`, principle 2), is refused with a
   suggestion and exit 2. A unit name itself starts a query (`sessions |
   group by origin | count`), and the references `last`, `latest` and
   `self` are never near misses (`did you mean "status"? to search for the word,
   use: polylogue find stauts`). `find` differs only here: the word after
   it is read as a query even when it is a command name or a near miss;
   rule 2 still decides where the verb starts.
4. A verb with no query before it acts on its default scope: `brief` and
   `continue` on the current directory's work (sessions whose recorded
   repository is the current directory's git repository, plus sessions
   whose working directory is the current directory or below it, matched by
   whole path components and exactly, `query.md`, Fields); every
   other verb on all sessions.
5. A directory or takeover briefing, `continue` and the references `last`
   and `latest` never include the session that is asking (`self brief`,
   the one briefing about the asking session, is the exception by name): when the CLI runs inside an agent harness, the harness's
   session ID variable (`sources/README.md`) names that session, and MCP knows it
   as `self`. "The previous session here" is therefore the previous one, not
   the one asking.

A reference alone (`user-data.md`, references) is a query matching that one
thing: `polylogue claude-code:7f3a2c10-0002 read`. It drops the default
filters for that match (subagents, hidden, inherited, abandoned branches;
`query.md`, Defaults), so a subagent's or a hidden session's reference
reaches it. That includes a bare native ID or prefix pasted from any
harness (`polylogue 7f3a2c10-0002 read`), when the word has an ID's shape
(`query.md`, what a bare word is), and a provider URL (`polylogue
https://chatgpt.com/c/68f0a1b2-… read`).

## IDs and row numbers

- **Short IDs.** Terminal output and MCP results show the shortest unique
  prefix of each native ID, at least 8 characters: `claude-code:7f3a2c10-0002`,
  `claude-code:7f3a2c10-0002/a7f3a2c1` (a subagent: its parent's short ID,
  "/", and at least 8 characters of its agent ID), `codex:01a20000-0005`,
  `chatgpt:68f0a1b2`. A native ID with a qualifier (`/` a subagent, `~side-` an older
  sidechain, `@profile-` a second install of the harness) shortens each
  part separately: the owning session's short ID, the qualifier, then the
  shortest unique prefix of the rest, at least 8 characters. A message's short ID is its session's short ID plus the shortest unique
  prefix of its native ID within the session. Any unique prefix is accepted
  as input everywhere. CLI `--json` and the HTTP API always carry full IDs.
- **Row numbers.** Terminal lists number their rows. `@N` means row N of
  the last list printed in this terminal (remembered per terminal;
  `$XDG_RUNTIME_DIR/polylogue/<key>/` holds socket, token, port and rows,
  where `<key>` is the first 12 hex digits of BLAKE3 over the archive
  directory's resolved path), and `@N.M` hit M of row N in a search.
  `@N` is the only row reference; `last` is not a row but the previous
  session here (rule 5). `polylogue @2 read`, `polylogue @1.2 note "…"`.

## Verbs (act on the matched set)

| Verb | Does | Options | Slice |
|---|---|---|---|
| `list` | one line per row (session, message, tool call, …); order and selection are query stages (`| sort by cost desc | limit 5`), not flags | `--fields a,b`, `--budget TOKENS` (stop when the output would exceed it, saying how many rows were left out) | 1 |
| `read` | render sessions, messages or any reference (`rendering.md`) | `--view transcript\|dialogue\|summary\|context\|lineage\|changes\|live\|raw`, `--format terminal\|markdown\|html\|json`, `--to stdout\|pager\|browser\|clipboard\|FILE`, `--tools full\|headers\|none`, `--thinking full`, `--harness full`, `--inherited`, `--branches`, `--from MESSAGE\|TIME`, `--until MESSAGE\|TIME\|compaction[:N]` (exclusive: up to and not including that message or time; TIME as in `query.md`, with one difference: a bare `10:37` means that time within the session being read, its first occurrence at or after the session's start, not today; `2026-04-06T08:00` is absolute), `--output-lines N` (a tool output longer than N lines shows its first and last lines around the marker `… M lines skipped (polylogue block:<id> read)`, as `rendering.md` writes it; a presentation budget the reader asks for, `query.md`, four bounds), `--last N`, `--around [MESSAGE]` (default: the matched message), `--at MESSAGE` (for `--view context`), `--all-matches`, `--for-others`, `--turns` (mark turn boundaries, slice 3), `--variant LANG` (slice 12) | 1 (views `transcript`, `summary`, `dialogue`, `raw`); 2 (`live`: state and process tree; tool calls linked to child processes in 6); 6 (`context`, `lineage`); 11 (`changes`) |
| `follow` | print one line per new message or tool call in the matched sessions until interrupted, and one line per live-state change (`→ waiting: permission for Bash`) (`polylogue running:true follow`, `polylogue @1 follow`) | `--tools none` | 2 |
| `brief` | a briefing for continuing the matched work (`user-data.md`, briefings); with `--task "HINT"`, a recall: what the archive knows that bears on a task, across all repositories when no query is given, within the query's sessions when one is (`polylogue 'repo:morrow path:src/parse.rs' brief --task "weekday arithmetic"`; `user-data.md`, recall; the MCP `brief` tool's `query` with `task`) | `--budget TOKENS`, `--accepted-only` (operator-authored items and agent items the operator accepted; slice 7), `--task "HINT"` (slice 6), `--keep` | 2 |
| `analyze` | a fixed report over the matched set (below) | `--by KEY[,KEY]` with keys `origin\|model\|day\|week\|month\|repo\|cwd\|tool\|action` (two keys give one row per pair), `--cost`, `--tools`, `--time`, `--windows` (subscription windows, `analysis.md`) | 3 |
| `tag`, `untag` | add or remove a tag on each matched session or row | `TAG` (`key:value` allowed) | 7 |
| `outcome` | set how each matched session ended (`user-data.md`) | `solved\|partial\|abandoned` | 7 |
| `note` | attach a note to each matched item | `"TEXT"` (the value; JSON for a structured kind, checked against its shape), `--kind KIND` (any kind in `user-data.md` that is written directly: not `withdrawal`, `hidden`, `title`, `outcome` or `tag`, which have their own verbs), `--key KEY`, `--valid-until TIME`, `--check QUERY --expect rows\|no_rows\|'count OP N'\|'AGG OP N'` (claims only; `user-data.md`, checked claims), and one flag per relation: `--blocks REF`, `--answers REF`, `--corrects REF`, `--challenges REF`, `--derived-from REF` | 7 |
| `revise` | write a new revision of each matched note (`user-data.md`, revisions): same kind, key and targets unless given; revising a note that is no longer current is refused with the current revision named, so a stale edit is re-read and merged rather than forked | `"TEXT"`, `--key KEY`, `--valid-until TIME`, `--retarget REF…` | 7 |
| `keep` | store the result as it is now, as `result:<id>` (`user-data.md`, kept results) | `--name NAME` | 7 |
| `hide`, `unhide` | hide matched sessions from results without deleting anything (`user-data.md`) | | 7 |
| `title` | set the operator's title for each matched session (a title note; `user-data.md`) | `"TEXT"` | 7 |
| `withdraw` | withdraw the matched notes with one withdrawal note (`user-data.md`, withdrawal); withdrawing a withdrawal restores | `--reason TEXT` | 7 |
| `continue` | list resume candidates, numbered (`user-data.md`); `continue N` runs candidate N's resume command; a candidate whose session is running is refused (`refused`, `session_running`) unless `--anyway` | `--all`, `--anyway` | 6 |
| `export` | write matched sessions in the export format (`export-format.md`), with a manifest saying which query selected them, when, and whether the archive was up to date | `--to DIR`, `--format json\|jsonl\|markdown`, `--profile full\|markdown\|wiki` (slice 11), `--with-blobs`, `--no-inherited`, `--no-notes`; with Markdown, `--view dialogue\|transcript\|changes` and `--output-lines N` make a compact bundle (for example `polylogue 'repo:sinex' export --format markdown --view dialogue --to sinex-history/`, every Sinex session as the operator's prompts and the model replies next to them, for reading by another model) | 1 |
| `excise` | permanently remove matched content (`storage/archive-db.md`, how excision works) | `--yes` | 7 |

Options every query accepts:

- `--json`: one JSON document, the result fields of `query.md` (Results),
  and nothing else on stdout. This is the result envelope on every verb and
  command; `read --format json` and `export --format json` instead write
  export documents (`export-format.md`), one per session.
- `--jsonl`: JSON lines and nothing else on stdout: first a header object
  with the result fields other than `rows` (`outcome`, `unit`, `count`,
  `notes`, `up_to_date`, `view`, `resolved`, `query`, `read_as`, `unknown`,
  `why`), then one row per line, then a last line `{"next_cursor": …}`
  (`null` when the result is finished). Large results stream this way.
- `--page-size N`: rows per result page (`query.md`, four bounds); never a
  selection, so `count` still covers every row.
- `--cursor C`: continue from a `next_cursor`; the query, sort and unit
  come from the cursor (`query.md`, continuation).
- `--why`: the condition-by-condition `why` list (`query.md`, Results).
- `--reveal`: show credential-shaped spans unmasked, refused unless stdout
  is a terminal and no harness environment is present (`behavior.md`,
  privacy; `refused`, `operator_only`).

`explain QUERY` (a command, below) shows how a query is understood and how
many rows each clause keeps: it counts read-only within the read time limit
and never writes, builds an index or calls a model; there is no second
spelling.

**`analyze` is shorthand.** Each report is a fixed set of pipelines over the
matched sessions (`query.md`, stages), joined on the grouping key;
`explain` prints them. Cost and tool counts read every branch (`active:*`,
`query.md`, defaults), as session cost does. `--by model --cost`, for
example, is
`messages where written_by:model active:* | group by model | agg count_distinct:session.id (split into the `sessions` and `subagents` columns by `session.kind`), sum:input_tokens, sum:output_tokens, sum:cache_read_tokens, sum:cache_write_tokens, sum:cost`.
The `sessions` column counts top-level sessions in every report; work done
by subagents is shown in a `subagents` column where a group has any.
It exists because the common reports combine several units (sessions,
prompts, tool calls, cost) in one table, which a single pipeline cannot;
each of its pipelines is an ordinary query, so an agent reaches the same
figures through MCP `search` with the pipeline `explain` prints.
Counting is by the time of each message and tool call: with `since:` or
`until:` in the query, only messages and tool calls inside the window are
counted, and `--by day` puts each on its own day (a session active on two
days counts on both).

**`excise`** needs an interactive terminal outside an agent harness, even
with `--yes`; without `--yes` it prints what would be removed and asks. It
resolves the query once on the client, shows the scopes, and on confirmation
sends exactly those scopes in one request; it never re-runs the query, so
rows that arrived in between are not removed by surprise. `export --to` writes to a temporary name and renames over the
previous export only when the new one is complete.

**`--archive DIR`** on any command names a different archive directory;
without it, the default location is used (`behavior.md`, concurrency). No
environment variable changes which archive a command acts on.
Agents can hide sessions but cannot excise them.

## Commands (not about a matched set)

| Command | Does | Slice |
|---|---|---|
| `status` | archive and daemon state, sources, unrecognized content, parse problems; warns when Claude Code's effective `cleanupPeriodDays` (managed, user and project settings; default 30) is low, since Claude Code deletes older transcripts at startup and the blob store holds only what was collected before that; `--agents`: whether each installed harness is wired to Polylogue (MCP entry, skill, hooks, recent calls counted from archived tool calls naming Polylogue's MCP tools; the MCP server keeps no call log of its own) | 1 (`--agents`: 7) |
| `import PATH…` | store files so the daemon parses them: an export (ZIP or JSON), a Polylogue export, an archive of provider files, or a directory (`sources/README.md`, importing a directory); `--format NAME` when structure alone cannot tell (`openai-chat`, `otel-genai`); `--wait` waits for parsing; `--dry-run` shows how each file would be recognized and what it would yield (format, sessions, messages) without storing anything | 1 |
| `import --notes FILE` | writes a batch of notes from a JSON-lines file, one note per line with the MCP `note` fields, targets checked first, withdrawable as one batch (`user-data.md`, importing notes in bulk); `note` is only a verb | 7 |
| `tags [WORD]` | list tags with counts; with WORD, only tags containing it | 7 |
| `query save NAME QUERY` | save a query as `query:NAME` (`user-data.md`, saved queries); `query list`; `query delete NAME` writes a withdrawal note for that saved query (`archive.db` only gains rows) | 7 |
| `explain [QUERY]` | describe the query language, or how a query is understood with each clause's row count (read-only counting within the read time limit; `query.md`, Results, `why`) | 1 (the grammar; field help grows with each slice) |
| `judge` | go through open claims by default, or the note kinds a query selects: accept, reject or defer, one key each (`user-data.md`, judgment); `--blind` hides author, model and origin, shuffles the order and records it on the judgment, and reveals them after the verdict; `--sample N QUERY` judges a random sample of N pinned note revisions a query matches (a pass's blind sample, `user-data.md`) | 7 |
| `config schema` | print the compiled static configuration JSON schema for package/module generation; works without a daemon, opens no archive and performs no runtime setting changes (`decisions/process.md`) | 1 |
| `setting [KEY [VALUE]]` | show all settings, one, or set one (`setting subscription.anthropic max-20x`, `setting day.zone Europe/Warsaw`; plan names in `analysis.md`); each shows who set it and when; turning on a setting that sends archive data out (the embedding provider, `fetch-links`) requires an interactive operator terminal and is refused for agent and `script` authors (`user-data.md`) | 3 |
| `install-agent HARNESS` | add Polylogue's MCP entry, skill and hooks to Claude Code or Codex config (Hermes from slice 5, through its memory or hook interface); `install-agent chrome` (slice 4) registers the browser extension's native messaging host instead: it writes the host manifest naming `polylogue browser-host` and the extension's ID as the only allowed origin into Chrome's `NativeMessagingHosts` directory (`extension-and-clients.md`); records what it will change before writing; `--remove` deletes exactly the entries it added (the backups are for undoing by hand); `--read-only` leaves out the MCP `note` and `keep` tools; `--instructions` adds the one line to the global instructions file (`mcp-and-hooks.md`) | 2 |
| `auth drive`, `auth embeddings` | `drive`: authorize read access to the Google Drive folder that holds AI Studio chats (`sources/google.md`); `embeddings`: store the embedding provider's API key, read from stdin or `--key-file PATH` (the NixOS module passes a secrets file); both write to `$XDG_STATE_HOME/polylogue/credentials/` (`principles.md`); `--revoke` | 5 (drive), 9 (embeddings) |
| `daemon` | run the daemon in the foreground; `--until-idle` exits once caught up; `daemon --until-idle --json` ends with one JSON object: `{"inputs_stored", "inputs_parsed", "sessions_written", "transactions", "per_format": {FORMAT: {"inputs_parsed", "records_read"}}, "per_input": [{"path", "records_read"}], "elapsed_ms"}` | 1 |
| `mcp` | run the MCP server on stdio | 1 |
| `hook EVENT --harness claude-code\|codex [--brief]` | without `--brief` (the recording entry, installed for every event): append the event to the hook spool; with `--brief` (the briefing entry, installed for session start and prompt submission): ask the daemon for the briefing and print it to inject, appending only its `delivery` line, never a second event line (`mcp-and-hooks.md`) | 2 |
| `browser-host` | the browser extension's native messaging host (`extension-and-clients.md`) | 4 |
| `completions SHELL` | print the shell completion script | 3 |
| `cite REF` | print a short citation line (reference, title, time) for pasting into a report, PR or git note (`export-format.md`, citations) | 7 |
| `open [REF]` | open the web reader, at a reference if given (slice 10) | 10 |
| `day [DATE]` | the day page: sessions, prompts, tool calls, cost and notes of one day, in time order (slice 10) | 10 |
| `changes --since GEN:COUNTER:ARCHIVE` | the change feed as JSON lines: sessions changed, new notes and excisions in `local_seq` order (`storage/archive-db.md`, local order; local to this archive and never a key; a merged or imported note is reported when it arrives, whatever its ID), removed sessions, and a "generation changed, read everything again" line when the index was rebuilt (`storage/index-db.md`, change counter and generation); each line carries the position to resume from (`GEN:COUNTER:ARCHIVE`). Read-only; the Python client and Sinex (`decisions/sinex.md`) may follow the archive with it. Polylogue shows changes; pushing notifications (to a phone, a desktop, a chat) is Sinnix's, which reads this feed; Polylogue has no notification channel of its own; an interrupting signal hands its notice to the configured notifier, Sinnix's (`behavior.md`, no notifications). | 7 |
| `input cat ID` | one stored input's bytes to stdout (ID: BLAKE3 hex for a complete input, UUIDv7 for a growing one), with every piece verified against its hash first. Read-only | 1 |
| `page …` | pages (slice 12; `user-data.md`) | 12 |
| `schema infer`, `schema unread` | offline schema inference over stored inputs, and the fields stored inputs carry that no parser reads (`analysis.md`, schema inference) | 8 |
| `embed estimate` | the cost of embedding what has no vector yet, before anything is sent (`analysis.md`, embeddings) | 9 |
| `demo` | build a synthetic archive from the Morrow world to try Polylogue on (`presentation.md`) | 8 |
| `sql 'SELECT …'` | run a read-only SQL query, executed by the daemon on a read-only connection over the documented, versioned SQL views in `storage/index-db.md` (sessions, messages, tool calls, notes, facts); the tables behind them stay internal. Its output is masked like every CLI output (credential-shaped spans; `behavior.md`, privacy), so SQL is no way around the masking, and agents may use it like any other read. For heavy analysis, `--jsonl` output is loaded elsewhere; nothing outside the daemon opens the archive's files | 3 |
| `running` | the live sessions: runs the saved query `query:running` that ships with Polylogue (`running:true`, listed with the fields id, live, since, repo, checkout, model, open tool, cost, task), which the operator can read, copy or change like any saved query (`user-data.md`, saved queries) | 2 |
| `statusline` | one short line for a terminal or harness status bar: running agents, sessions waiting on you, usage in the current subscription window; answered by the daemon within the hook budget | 3 |

## Output rules

- Columns in terminal lists are padded to the widest value among the rows
  shown, left-aligned for text and right-aligned for numbers and costs; a
  title is cut with `…` to fit the terminal width. Piped output (not a
  terminal) uses the same columns without colour.
- Terminal times are in the operator's zone (setting `day.zone`, default the system zone; the same zone logical days use, `query.md`, times);
  JSON times are UTC with `Z` (`README.md`, results and errors). Dates
  within the current year omit the year. Durations are written `8.2s`,
  `3m 43s`, `1h 05m` (zero seconds left out: `7m`).
- Costs are shown in USD to the cent; "—" means unknown, never zero. A
  session row's cost is the session's own; the summary view adds its
  subagents' cost on the same line, and `analyze` includes subagents
  because they cost money.
- Counts exclude inherited messages (`decisions/lineage.md`) unless asked.
- Top-level sessions only by default; `kind:*` or `kind:subagent` includes
  subagents. Search hits inside a subagent are shown under its parent.
- Titles are the display title (`model.md`, Session) and its source;
  derived ones are dimmed and cut at 60 characters (`rendering.md`).
- When the index is not up to date, a line on stderr says so
  (`archive is catching up: 12 inputs waiting`), and JSON carries
  `"up_to_date": false`. Results are still shown. Inputs stored before
  their format has a parser (`no_parser_yet`) are not waiting: `status`
  lists them as "stored, parser arrives in slice N", and they do not make a
  result out of date.

### `status --json`

One object; every figure says whether it was measured (law 46, rule R5),
and the lists are the parts the law tests compare between archives
(`plan/testing/laws.md`, comparing two views):

```json
{"archive": "/home/op/.local/share/polylogue", "daemon": "running", "up_to_date": true,
 "figures": [
   {"name": "sessions", "value": 8, "state": "measured"},
   {"name": "inputs_waiting", "value": 0, "state": "measured"},
   {"name": "embeddings", "value": null, "state": "unmeasured", "reason": "not enabled"},
   {"name": "location:/home/op/.codex/sessions", "value": 2, "state": "measured"}],
 "unrecognized": [["claude-code-jsonl", "record:atis-latch", 1]],
 "inputs": [],
 "problems": []}
```

- `figures`: `{name, value, state, reason}`, `state` one of `measured`,
  `measured_absent` (looked, found none: the value is 0 or empty) and
  `unmeasured` (could not look: `value` is null and `reason` says why).
  An unreadable location is an `unmeasured` figure naming the path, never
  zero inputs (`"value": null, "state": "unmeasured", "reason": "permission
  denied"`); Morrow has none.
- `unrecognized`: `(format, what, count)` triples from the `unrecognized`
  table, summed over inputs (`storage/index-db.md`).
- `inputs`: `(format, status, detail, count)` for the input chains whose
  `parse_state` status is not `done` (or `companion`), grouped: `parsing`,
  `no_sessions`, `no_parser_yet` with its detail. Grouped, because slice 1
  alone stores about 944,000 restored spool files with `no_parser_yet`.
- `problems`: `(path, status, detail)` for every chain that is `refused` or
  `failed`, the parse problems, which are few and each worth naming. The
  path, not the input ID, because an append-only input's ID differs between
  two runs.

## Examples

### Bare `polylogue`

Run from `/home/op/morrow`:

<!-- test: surfaces-1 slice:4 -->
```console
$ polylogue
Polylogue · 8 sessions (7 top-level) · last activity Oct 07 18:05
daemon    running · up to date
sources   claude-code 4 · codex 2 · chatgpt 1 · claude-ai 1
last 14 days  7 sessions · 17 prompts · 37 tool calls · $6.20 API-eq. known (2 sessions unknown)

recent in /home/op/morrow
1  Oct 06 09:30  codex:01a20000-0006        Review parser (try strict mode)   6 msgs  $0.12
2  Oct 05 16:00  codex:01a20000-0005        Review parser                     9 msgs  $0.22
3  Oct 03 14:00  claude-code:7f3a2c10-0004  Date parsing                     10 msgs  $1.59

notice    1 unrecognized record type (polylogue status for details)
```

Blocks that appear only when they have content (all from slice 2, when the
daemon watches files and records hook events):

- **waiting on you**, first: sessions whose live state is `waiting`
  (`model.md`, Live state). One line each: short ID, title, what it waits
  for, and since when.
- **running now**, one line per session whose `running` field is true
  (`query.md`) and that is not already shown under waiting on you: short ID, live state, title, the working directory's last
  path component, the open tool call or in-progress todo item, cost so far,
  and the last tool call with its outcome. When two live sessions share a
  worktree, a line says so (`⚠ 2 agents in /realm/worktree/x`). Nothing is
  running at the example clock.
- **window**, when a subscription is set (`setting subscription.anthropic
  max-20x`): the current usage window, for example
  `window    Anthropic 5h: $31 API-equivalent so far · resets 14:10 · week 61% (reported)`.
  A limit is shown only as the harness reported it (Codex percentages,
  Claude Code reset times); none is estimated (`analysis.md`, subscription
  windows).
- **recent in DIR**, when the current directory is the working directory of
  some sessions.

### Search

<!-- test: surfaces-2 slice:4 -->
```console
$ polylogue chrono
3 sessions · 8 matching messages

1  Oct 02 10:00  claude-code:7f3a2c10-0002  Date parsing             18 msgs  API-eq. $2.85
   .1  10:20  ↳ subagent claude-code:7f3a2c10-0002/a7f3a2c1 (Explore)
              model  Compare the chrono and time crates for relative date arithmetic. …
   .2  10:26  ↳ subagent claude-code:7f3a2c10-0002/a7f3a2c1 (Explore)
              model  chrono offers NaiveDateTime plus Duration and weekday helpers; …
   .3  10:56  you    let's use chrono, not time
   .4  11:07  model  Switched to chrono. Tests, clippy and build are clean.
   .5  11:15  model  chrono covers it too: NaiveDate::parse_from_str for ISO dates. …
2  Oct 07 18:00  claude-ai:0b9d4e6c         README wording            4 msgs  —
   .1  18:04  you    Shorter, and mention chrono.
   .2  18:05  model  morrow parses human-written dates into chrono types and rejects …
3  Oct 04 21:00  chatgpt:68f0a1b2           Date format edge cases    6 msgs  —
   .1  21:09  model  dateutil guesses month-first; chrono refuses. Refusing by default …
```

`polylogue @1.3 read --around` opens #2 around the operator's message.

What the example shows:

- Session #4 has no hits of its own. Its copies of #2's messages are
  inherited and excluded, and the compaction summary is not a message.
- The ChatGPT hit on the abandoned branch ("chrono refuses ambiguous input")
  is excluded, because inactive messages are excluded by default
  (`query.md`, defaults).
- The subagent's hits are shown under its parent. Its first message is the
  instruction #2's model wrote for it, so it is `written_by: model` and
  plain search matches it.
- Sessions are ordered by relevance. Tests compare relevance-ordered results
  as sets; only `sort by` fixes an order.

<!-- test: surfaces-3 slice:4 -->
```console
$ polylogue chrono --json
{"outcome":"ok","unit":"sessions","count":3,"next_cursor":null,"notes":[],"up_to_date":true,
 "view":"…","resolved":"…","query":"chrono","unknown":"…",
 "read_as":"Sessions whose messages mention chrono (subagents counted with their parent), over all sessions in this archive, up to now.",
 "rows":[
  {"id":"claude-code:7f3a2c10-0002-4000-8000-000000000002","title":"Date parsing",
   "started":"2026-10-02T08:00:00.000Z","messages":18,"cost":2.85,
   "hits":[
    {"message":"claude-code:7f3a2c10-0002-4000-8000-000000000002/a7f3a2c1000000003:c3000000-0000-4000-8000-000000000001",
     "session":"claude-code:7f3a2c10-0002-4000-8000-000000000002/a7f3a2c1000000003","via":"subagent","written_by":"model",
     "at":"2026-10-02T08:20:00.000Z",
     "snippet":"Compare the chrono and time crates for relative date arithmetic. …","marks":[[12,18]]},
    {"message":"claude-code:7f3a2c10-0002-4000-8000-000000000002/a7f3a2c1000000003:c3000000-0000-4000-8000-000000000006",
     "session":"claude-code:7f3a2c10-0002-4000-8000-000000000002/a7f3a2c1000000003","via":"subagent","written_by":"model",
     "at":"2026-10-02T08:26:00.000Z",
     "snippet":"chrono offers NaiveDateTime plus Duration and weekday helpers; …","marks":[[0,6]]},
    {"message":"claude-code:7f3a2c10-0002-4000-8000-000000000002:c2000000-0000-4000-8000-000000000013",
     "written_by":"human","at":"2026-10-02T08:56:21.000Z","snippet":"let's use chrono, not time","marks":[[10,16]]},
    "… 2 more hits"]},
  "… 2 more sessions"]}
```

(`"… n more"` strings stand for elided entries in this example only.)

### Read a session

Tool calls are collapsed to one line each, outcome first after the target.
`--tools full` expands them; `--tools none` hides them. The example is not a
terminal (it runs under the test harness), so the whole session is printed
rather than paged (`rendering.md`, ranges).

<!-- test: surfaces-4 slice:7 -->
```console
$ polylogue claude-code:7f3a2c10-0002 read
Date parsing
claude-code · Oct 02 10:00–11:15 (1h 15m) · /home/op/morrow · branch main
claude-fable-5-1 · 18 messages · 3 prompts · 13 tool calls (1 error) · $2.85 API-eq.
tags project:morrow

10:00 you
  Add parse_relative() for phrases like "tomorrow at noon" and "in 3 days".

10:06 model
  Let me look at the current code first.
  ▸ Read   src/lib.rs                         ok
  ▸ Read   Cargo.toml                         ok

10:19 model
  ▸ Agent  Explore: Survey date crates        ok   7m   → claude-code:7f3a2c10-0002/a7f3a2c1 (6 msgs)

10:30 model
  ▸ Edit   src/parse.rs                       ok
  ▸ Edit   src/lib.rs                         ok

10:37 model
  ▸ Bash   cargo test                         error  exit 101   3m 43s
           test parse_dates::tomorrow_at_noon ... FAILED

10:45 model
  thinking (10 words)
  ▸ Edit   src/parse.rs                       ok
  ▸ Bash   cargo test                         ok   3m 44s

10:52 model
  All 6 tests pass. parse_relative handles "tomorrow at noon" and "in N days" using the time crate.

10:56 you                                     ◆ decision: decided: chrono, not time
  let's use chrono, not time

11:00 model
  ▸ Edit   Cargo.toml                         ok
  ▸ Read   src/parse.rs                       ok
  ▸ Bash   cargo test                         ok   3m 43s
  ▸ Bash   cargo clippy                       ok   3m 43s
  ▸ Bash   cargo build                        ok   3m 43s

11:07 model
  Switched to chrono. Tests, clippy and build are clean.

11:11 you
  Which crate does parse_absolute need?

11:15 model
  chrono covers it too: NaiveDate::parse_from_str for ISO dates. I'll add it next session.
```

Tool results are messages written by the harness. They are folded into the
call lines above them, so 12 of the 18 messages (all but the 6 tool results)
have their own heading. A
failed call shows the first line of its output that names the failure.
Durations are shown for shell and subagent calls, where they measure the
command's run time; for file tools the time is mostly the model's own, so it
is left out (`rendering.md`). The synthetic example archive spaces records
evenly, so its durations are long. The `◆` text is the note attached to that
message. Notes and tags come in slice 7; before that the same output has no
`tags` line and no `◆` marks (slice 1 checks that version,
`examples/archive/expected/slice-1/read-2.txt`).

### Summary view

<!-- test: surfaces-5 slice:7 -->
```console
$ polylogue claude-code:7f3a2c10-0002 read --view summary
Date parsing  ·  claude-code:7f3a2c10-0002
when      Oct 02 10:00–11:15 (1h 15m)
where     /home/op/morrow · branch main
model     claude-fable-5-1
messages  18 (3 typed by you)
tools     Bash 5 (1 error) · Edit 4 · Read 3 · Agent 1
tokens    input 2,000 · output 21,100 · cache read 900,000 · cache write 70,000
cost      $2.85 API-eq.  (+ $0.16 in 1 subagent)
lineage   subagent claude-code:7f3a2c10-0002/a7f3a2c1 · continued by and briefed into claude-code:7f3a2c10-0004
notes     decision "decided: chrono, not time" (written 11:00, on the 10:56 message) · tag project:morrow
first     Add parse_relative() for phrases like "tomorrow at noon" and "in 3 days".
```

Before slice 7 there is no `notes` line (`examples/archive/expected/slice-1/summary-2.txt`).

### Cost by model

<!-- test: surfaces-6 slice:4 -->
```console
$ polylogue since:2026-10-01 analyze --cost --by model
model              sessions  subagents   input   output   cache read  cache write  API-eq.
claude-fable-5-1          3          0   4,700   44,100    1,700,000      140,000   $5.70
claude-opus-5-5           0          1     400    3,000       60,000       15,000   $0.16
gpt-6.1-sol               2          0  48,000    9,000      180,000          n/a   $0.33
gpt-6                     1          0       —        —            —            —       —
(not recorded)            1          0       —        —            —            —       —
total                     7          1                                              $6.20
2 sessions have unknown cost (subscription chats without token counts).
The claude.ai export does not record the model, so session #8 is grouped
under "(not recorded)".
Rows are rounded separately; the total is computed before rounding.
```

Subagents are included in cost analysis, since they cost money.

### By day

<!-- test: surfaces-7 slice:4 -->
```console
$ polylogue since:2026-10-01 analyze --by day
day     sessions  subagents  prompts  tool calls  API-eq.
Oct 01         1          0        3           6  $1.26
Oct 02         1          1        3          17  $3.01
Oct 03         1          0        3           8  $1.59
Oct 04         1          0        3           0      —
Oct 05         1          0        2           3  $0.22
Oct 06         1          0        1           3  $0.12
Oct 07         1          0        2           0      —
total          7          1       17          37  $6.20
```

`sessions` counts top-level sessions and `subagents` the subagent sessions;
tool calls and cost include subagents (Oct 02 includes the Explore
subagent's 4 tool calls and $0.16).

<!-- test: surfaces-8 slice:4 -->
```console
$ polylogue since:2026-10-01 analyze --by day --json
{"outcome":"ok","unit":"report","count":7,"next_cursor":null,"notes":[],"up_to_date":true,
 "view":"…","resolved":"…","query":"…","unknown":"…",
 "read_as":"Sessions, prompts, tool calls and cost per logical day since Oct 01, over all sessions in this archive, up to now.",
 "group_by":"day",
 "rows":[
  {"day":"2026-10-01","sessions":1,"subagents":0,"prompts":3,"tool_calls":6,"sum_cost":1.262,"unknown_cost":0,"partial_cost":0},
  {"day":"2026-10-02","sessions":1,"subagents":1,"prompts":3,"tool_calls":17,"sum_cost":3.0106,"unknown_cost":0,"partial_cost":0},
  {"day":"2026-10-03","sessions":1,"subagents":0,"prompts":3,"tool_calls":8,"sum_cost":1.59,"unknown_cost":0,"partial_cost":0},
  {"day":"2026-10-04","sessions":1,"subagents":0,"prompts":3,"tool_calls":0,"sum_cost":null,"unknown_cost":1,"partial_cost":0},
  {"day":"2026-10-05","sessions":1,"subagents":0,"prompts":2,"tool_calls":3,"sum_cost":0.216,"unknown_cost":0,"partial_cost":0},
  {"day":"2026-10-06","sessions":1,"subagents":0,"prompts":1,"tool_calls":3,"sum_cost":0.117,"unknown_cost":0,"partial_cost":0},
  {"day":"2026-10-07","sessions":1,"subagents":0,"prompts":2,"tool_calls":0,"sum_cost":null,"unknown_cost":1,"partial_cost":0}],
 "total":{"sessions":7,"subagents":1,"prompts":17,"tool_calls":37,"sum_cost":6.1956,"unknown_cost":2,"partial_cost":0}}
```

JSON carries exact values; only terminal output rounds. A report's JSON has
the usual result fields (`query.md`, results) plus `group_by` and `total`.

`--by origin` works the same way (a subagent counts under its own origin,
which is its parent's): claude-code $5.86 API-eq. (subagent included), codex $0.33,
chatgpt and claude-ai unknown.

### Tagging

<!-- test: surfaces-9 slice:7 -->
```console
$ polylogue origin:chatgpt tag research
tagged 1 session: chatgpt:68f0a1b2 (Date format edge cases)
```

<!-- test: surfaces-10 slice:7 -->
```console
$ polylogue tag:research
1  Oct 04 21:00  chatgpt:68f0a1b2  Date format edge cases   6 msgs  —
```

### A note on a message

<!-- test: surfaces-11 slice:7 -->
```console
$ polylogue codex:01a20000-0005:… note "next friday bug found here"
noted message codex:01a20000-0005:… (Review parser, Oct 05 16:25)
```

`polylogue @1 note "…"` (or `polylogue last note "…"`) attaches to row 1 of the last list
printed in this terminal.

### Resume candidates

<!-- test: surfaces-12 slice:6 -->
```console
$ cd /home/op/morrow && polylogue continue
resume candidates for /home/op/morrow
1  Oct 03 14:00  claude-code:7f3a2c10-0004  Date parsing
   open: unanswered_prompt
   last: Clean: clippy passes, all 9 tests pass, and the change is committed as …
   claude --resume 7f3a2c10-0004-4000-8000-000000000004
2  Oct 06 09:30  codex:01a20000-0006        Review parser (try strict mode)
   last: Strict mode is in: Options { strict: true } rejects same-day weekday …
   codex resume 01a20000-0006-7000-8000-000000000006
3  Oct 05 16:00  codex:01a20000-0005        Review parser
   last: Noted on the message above: next friday bug in weekday_offset().
   codex resume 01a20000-0005-7000-8000-000000000005
1 more: polylogue continue --all
```

Ranking and the exact "open" signals are defined in `user-data.md` (resume
candidates): open candidates come first, each with its signals named
(`unanswered_prompt`, `unfinished_tool_call`, `open_note`, `outcome_partial`,
`question`) on a line under the title. #4 is open (its last message is a
prompt with no reply), so it comes first; the rest follow by recency. #2
does not appear because #4 continues it. The
"last:" line is the final model message, shortened; Polylogue does not
summarize. `polylogue continue 1` runs the first candidate's command.

**A session that is still running is not resumed twice.** A candidate whose
`running` field is true (`query.md`, the one definition of running now) is
shown as "running now (pid 41237, in
/realm/worktree/x)" instead of with a resume command, and `continue N` on it
refuses with that line, because a second harness process on the same
session would split it. `--anyway` runs the command regardless, and the
result is recorded like any other fork. A candidate whose state is `stale`
(the process died without an end event) resumes normally; `unknown` counts
as running, since a second harness process on a session that may still be
written would split it.

### Import

<!-- test: surfaces-13 slice:4 -->
```console
$ polylogue import ~/Downloads/chatgpt-export-2026-10-04.zip
stored ~/Downloads/chatgpt-export-2026-10-04.zip (ChatGPT export, 1 conversation)
the daemon will parse it; use --wait to wait
```

<!-- test: surfaces-14 slice:4 -->
```console
$ polylogue import ~/Downloads/chatgpt-export-2026-10-04.zip --wait
already stored (same bytes); 1 session, 0 new
```

### Status

<!-- test: surfaces-15 slice:4 -->
```console
$ polylogue status
daemon        running (pid 4123) since Oct 08 08:00 · up to date
archive       ~/.local/share/polylogue · blobs 88 KB · index 0.4 MB
inputs        11 stored · 0 waiting · 0 failed
sessions      8 (7 top-level) · 73 messages (2 on abandoned branches) · 18 inherited copies not counted

source        location                          inputs  sessions  last new
claude-code   ~/.claude/projects                     5         4  Oct 03 15:10
codex         ~/.codex                               3         2  Oct 06 09:50
chatgpt       import                                 1         1  Oct 04 21:20
claude-ai     import                                 1         1  Oct 07 18:05
hooks         ~/.local/share/polylogue/hooks         1         —  Oct 03 14:42

unrecognized  claude-code-jsonl  record type "atis-latch"   1×  first seen Oct 02
parse problems none
```

Inputs are files: Claude Code's four session files plus the subagent's
`.meta.json`; Codex's two rollouts plus `state_5.sqlite`; the two exports;
and from slice 2 the one hook spool file (`examples/archive/FACTS.md`, hook
spool), which yields no sessions of its own: its prompt line and delivery
line join #4.

During a first build or a rebuild, `status` shows the rate, bytes and
inputs done and waiting, and an estimate only while the rate has been
steady (within about 20% over the last few minutes); otherwise it says "rate
varying" instead of a time, because rate-based estimates were more than 50%
off under competing disk load (Sinex's measurements on `/realm`). It also
names what is running now, which a rate from finished work hides: a commit
or sync in progress and its age ("a commit has been running 15 s"), the
stage holding the single writer ("writer busy with the derive step"), or
pacing ("paced for IO"). It tells idle (nothing to do) from stalled
(work waiting, no step running, no progress for several minutes): a long
sync or a busy writer is shown as what it is, never as a stall. A build is finished only
when everything is up to date, derived products included; a fast parse that
leaves derived work waiting is not done. An archive whose files cannot be
opened or read is reported as unknown, never as up to date.

A freshly created archive with no inputs is `ok`: zero counts and the
provider locations it watches; "no sessions yet" is not a problem. A
location that exists but cannot be read is listed as unavailable with its
error and is not treated as missing (`sources/README.md`, how sources work). With
Sinex configured (`decisions/sinex.md`), `status` adds a line with the mode,
pending inputs and rows, the age of the oldest pending one and the last
error, in plain words. While Sinex is unreachable the pending count is
unmeasured, because nothing local records what Sinex received
(`decisions/sinex.md`, transport), so the line says so and adds only what
was retained in the pending area since the last successful check ("sinex primary · Sinex
unreachable since 14:02 · remote pending unmeasured · 1 input retained pending
since the last check at 13:55"); that is a normal state, never reported as success and never as an
error code. While Sinex's ingress is full the line says "sending, waiting for
Sinex to catch up" with the count of revisions not yet sent; that is normal
too (`decisions/sinex.md`, initial transfer). Retained in the pending area, confirmed by Sinex and interpreted locally are separate facts: a newest unconfirmed revision never contributes to local query results. Already confirmed revisions remain readable, with pending work named and `up_to_date: false`.

When something failed, it is listed with the input, the reason and where to
look:

<!-- test: surfaces-16 slice:1 -->
```console
parse problems
  claude-code-jsonl  ~/.claude/projects/-home-op-morrow/9a1b….jsonl
                     1 line is not valid JSON (byte 48210, counted as undecodable); the rest is parsed
```

An incomplete last line is not a problem: it is a file being written, and
its complete lines are parsed while the rest waits for the file to grow
(law 3; `sources/claude-code.md`).

### Empty result explained

<!-- test: surfaces-17 slice:3 -->
```console
$ polylogue origin:codex tool:Edit --why
no matches
origin:codex → 2 of 5
tool:Edit → 0 of 2 (Codex sessions use: exec_command 5, apply_patch 1)
```

One line per entry of the result's `why` list (`query.md`, Results); the
exit code is 1.

## Exit codes

| Code | Outcome or error kind |
|---|---|
| 0 | `ok`: success; for queries, at least one match |
| 1 | `empty`: the query ran and nothing matched |
| 2 | `invalid_query`, `not_yet_available`, `ambiguous`: the request is wrong as written (the message points at the problem and gives a suggestion) |
| 3 | `refused`: a valid request the archive declines (the `reason`s in `query.md`, Results; the reason is printed), and an excision declined at its confirmation |
| 4 | `not_found`, `conflict`, `io`: the request named something absent, collided with an earlier write, or failed on the machine (file unreadable, database locked too long) |
| 5 | `degraded`: the query ran but part of what it needed is missing; rows, if any, are printed |
| 6 | `not_running`: the daemon is not running (the one-line message says how to start it) |
| 7 | `timeout`: the read passed its time limit; the narrower suggestion is printed |

Every outcome and error kind of `query.md` (Results) has exactly one code.
"Not up to date" on its own is a warning: results are printed and the exit
code reflects them.


## Completion and interactive selection
- Completion (bash, zsh, fish) is generated from the query grammar and
  the live archive: field names after a space, values after a colon
  (`origin:` → origins present, `tag:` → existing tags, `repo:`/`cwd:` →
  values in the index, `model:` → models seen), session IDs by recency with
  their titles as descriptions. It asks the daemon, which answers from
  warm state in under 50 ms. With the daemon down, completion still offers
  what the grammar alone knows (field names, operators, verbs, flags) in
  under 50 ms, without opening the archive or starting anything; values that
  need the archive are simply absent (the Python daemon-off completion took
  3.5 s per keystroke, `polylogue-vtgpi`).
- `polylogue QUERY read` with several matches and an interactive terminal
  opens a fuzzy picker (titles, dates, first prompt) instead of rendering
  all of them; `--all-matches` renders all.
- `read --to pager` (the default on a terminal for long output) uses `$PAGER`.
- Colors follow the terminal's palette (16 ANSI colors only, so pywal and
  similar themes apply); `NO_COLOR` turns them off.

Bead and audit IDs behind each section: `evidence/beads/spec-sources-g3.md`.

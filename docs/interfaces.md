# Interfaces

Polylogue has one table of operations, and every surface is generated from it: the command line, the daemon's socket, HTTP and MCP. This page owns that table as Polylogue uses it, the command-line grammar and the rule that keeps the command set small, latency classes and Polylogue's budgets, the daemon's states as the command line sees them, warm state, completion and its catalog file, the picker, row addressing (`@N`, `@last`), footers and affordances, output, Polylogue's failure kinds and exit codes, and help.

The generic mechanics (the `Operation` trait, the failure-to-exit-and-HTTP table, request and answer envelopes, deadline classes, SSE framing) are the common [`command-surface`](https://github.com/Sinity/polylogue-sinex-common/blob/master/docs/command-surface.md) crate's. The list of operations is generated into [reference/commands.md](reference/README.md). The query language is in [query.md](query.md); what a caller sends about itself and how its author is computed is in [assertions.md](assertions.md#authorship); writing out with `read --to` is in [reading.md](reading.md#writing-out); MCP as the agents' channel is in [agents.md](agents.md#mcp-server); the web is in [web.md](web.md).

## Command table

- **One typed operation table** drives every surface. Each row declares the operation's arguments, answer type, latency class, failure kinds, one-sentence help, its reason for existing ([below](#discipline)) and one to four examples on the Morrow archive. The CLI, socket, HTTP and MCP (with `outputSchema`) are generated from it; a new operation is a row, not code on each surface.
- **Answers are typed** and rendered from their types ([reading.md](reading.md)); each has one terminal printer and one JSON form, and no surface has features the others cannot reach.
- **The daemon is required.** The command line is a thin client: it never opens archive files, and the daemon is the only writer. Ruling, 2026-10-27: "the CLI is supposed to be thin client for the daemon, which is supposed to have advanced caching, indexing whatnot to enalbe the CLI to be snappy" (cc:ee52794c-8a9e-43b6-9fbc-9749c7e78e8f:dad489d4-a1b8-48bd-bbc0-26262f34b262). Ruling, 2026-09-14: "there is no requirement or need for CLI to ever work 'standalone'. Daemon is the standard, not some sort of optional addon" (cc:652ae14f-a036-496b-abfa-42e7574c9ec9:b8a68d64-772f-4d8c-8e95-435ce601d245). Ruling, 2026-05-03: "we do not do non-daemonic writes" (cc-queued:d227b344-a871-4be1-bdf0-30c59ff1db7d@2026-05-03T18:55).
- **No refusal depends on who the caller is.** No act is refused because the caller is an agent or has no terminal; the only terminal-dependent behaviours are the picker-based ones, and each names its non-interactive form ([Picker](#picker)).
- There is no REPL or terminal reader; the interactive elements in a terminal are completion, `@N`, `next`, the pager and the picker, and interactive reading is the web reader. View data is typed, so a terminal reader stays a thin addition.

## Grammar

```
polylogue                                  orientation (contents: operations.md#status)
polylogue QUERY [VERB [args]]…             a query, then verbs on its rows; without a verb, the rows
polylogue find QUERY [VERB …]              the same, with the first word read as a query whatever it is
polylogue COMMAND [args]                   a command that is not about a query's rows
```

Query first, then what to do with the rows; there is no `do` or `query` verb in front.

1. If the first argument is a command name, it is that command.
2. Otherwise the first argument must carry a **query signal**: a field condition (`repo:x`), a unit name (`sessions …`), a reference (an ID shape, a provider URL, `@N`, `@last`, `last`, `latest`, `self`, a typed reference such as `query:running`), or a single argument holding whitespace or query punctuation (`polylogue 'rate limit'`). `find` forces the query reading.
3. **An unsignalled first word is refused,** exit 2, within the local budget, with the `find` form and nothing else; command names are not guessed. Ruling, 2026-07-05: "typo hint is actually unnecessary given we do not want bare query support anyway" (cc:900c6128-99a3-4986-8ed4-d5d7aacb38fe:e5f52f29-85e8-4cfc-9088-b5ea79b7bccc).
   ```console
   $ polylogue chrono
   polylogue: "chrono" is not a command or a query
   to search for the word: polylogue find chrono
   ```
   Suggestions for misspelled fields and stages inside a query are the query compiler's ([query.md](query.md#errors)).
4. After the query, a verb is the first verb name followed only by that verb's own arguments and options. Several verbs may follow in order (`… assert tag x read`). An unquoted verb name inside an unquoted multi-word query is refused with the quoted form.
5. A reference alone is a query for that one thing and drops the query defaults.
6. A verb with no query acts on its default scope, declared per verb in the table (`pack` and `resume` on the current directory's work; everything else on all sessions). `last`, `latest` and the default scopes never include the calling session.
7. Options are recognised anywhere on the line, before or after the query and verbs; `--` ends option parsing for a query that starts with `-`.

Names are fixed once, with no aliases. `plg` and `plog` are shell-level names the package installs, not command aliases.

### Discipline

A command or verb exists only when what it does cannot be expressed as a query plus an existing verb, or as a shipped saved query that completion offers. Each row states that reason, and `explain capabilities` prints it. Ruling, 2026-03-16: "I don't understand why would we have separate subcommands which do strictly a subset of another command" (cc-reject:d202c788-0043-4643-b7bf-1a9dac780739:a505ddb7-d4a3-441b-9c48-f386ad327765). Ruling, 2026-05-25: "randomly introduce new top-level command for that?" (cc-queued:c2404522-7bba-4de3-936d-e08deb14262c@2026-05-25T00:40). Ruling, 2026-07-29: "probably should be in CLI, unless clear reason not to, but CLI itself is supposed to be disciplined, not sprawling" (cc-queued:d6f4b0c3-840b-47b6-bd57-5f1bd6c9c7d4@2026-07-29T01:12).

So listing is the bare query; the running list, notices and resume candidates are the shipped saved queries `query:running`, `query:notices` and `query:resume_candidates`; a day is a query with the day's time condition; reports are the `| measure` and `| agg` stages; writing out is `read --to`; citing is `read --format citation`; opening in the browser is `read --to browser`; attachment bytes are `block:<message>#<n> read --to FILE`; the stored input behind a row is `trace`. Derived work is the daemon's, so there are no build, rebuild or repair commands ([architecture.md](architecture.md#principles)).

### Verbs

Verbs act on a query's rows.

| Verb | Does | Semantics |
|---|---|---|
| `read` | render rows; with `--to`, write them out | [reading.md](reading.md) |
| `follow` | one line per new row or live-state change until interrupted | [reading.md](reading.md), [agents.md](agents.md#live-sessions) |
| `trace` | the provenance chain of a reference back to its stored input, its bytes and its reading version | [reading.md](reading.md) |
| `assert KIND [KEY] [VALUE]` | write one assertion of KIND on each row | [assertions.md](assertions.md) |
| `revise`, `withdraw` | a new revision of each assertion row (refused on a stale base), or a withdrawal | [assertions.md](assertions.md#currency) |
| `judge` | on a terminal, one-key verdicts over the rows in the picker (`--blind` hides author, model and origin, shuffles, and records that on the judgment); off a terminal, exit 2 naming `judge accept\|reject\|defer` with explicit rows | [assertions.md](assertions.md#judgment) |
| `keep` | store the result as `result:<id>` | [query.md](query.md#query-objects) |
| `save NAME` | label the query's hash with a name | [query.md](query.md#query-objects) |
| `watch` | write a watch rule over the query | [query.md](query.md#watches) |
| `pack [NAME]` | build a pack under a budget and record a delivery | [agents.md](agents.md#packs) |
| `ping TEXT`, `send TEXT`, `approve`, `deny`, `interrupt`, `resume` | address a session, or act on a live one through its harness | [agents.md](agents.md#live-sessions) |
| `pick` | open the picker over the rows and print the chosen references, one per line | [below](#picker) |
| `compare DIMENSION` | a comparison of the matched references | [assertions.md](assertions.md#judgment) |
| `variant` | write a variant of a reference | [assertions.md](assertions.md#variants) |
| `add REF…`, `move POS`, `toc`, `checkout DIR`, `checkin DIR` | page operations | [pages.md](pages.md) |
| `applies ENV` | whether a method's conditions hold in an environment | [assertions.md](assertions.md#memory) |

### Commands

Commands are not about a query's rows.

| Command | Reason |
|---|---|
| `status [--line]` | health of the daemon and archive, not a query unit ([operations.md](operations.md#status)) |
| `explain [QUERY \| SUBJECT]` | describes the language or how a query is read, without running it |
| `import PATH…` | takes files, not rows ([capture.md](capture.md#import)) |
| `setting KEY VALUE` | writes one key; reading settings is the `settings` unit |
| `sql 'SELECT …'` | read-only SQL through the daemon ([query.md](query.md#read-only-sql)) |
| `changes --since POS` | a position-based feed for programs |
| `next` | the following page of the last list here ([Addressing](#addressing)) |
| `web [PATH]` | opens the workbench at a view that is not a reference |
| `auth drive` | a credential flow, not rows |
| `daemon [--until-idle] [--json]`, `mcp`, `hook EVENT`, `hooks print\|status`, `install-agent HARNESS [--remove]`, `browser-host`, `completions SHELL`, `demo` | local process entry points and installers ([operations.md](operations.md#installing-into-harnesses)) |
| `schema infer\|unread` | daemon requests over stored inputs ([archive.md](archive.md#format-drift)) |
| `verify` | the on-demand read-only check ([operations.md](operations.md)) |
| `sinex cutover` | the one mode transition ([primary-mode.md](primary-mode.md#cutover)) |

## Latency classes

Each table row declares a latency class, and the class applies on every surface (CLI, MCP, HTTP, web). Budgets are end to end (process start to the last byte of the first screen, through the real binary and socket), with a warm daemon and a real-shaped archive under ordinary load, at p95. The deadline each request carries comes from its class, in the `command-surface` request envelope.

| Class | Members | Budget (p95) | Deadline sent |
|---|---|---|---|
| **local** | `--help`, `--version`, `completions`, grammar-only completion, `explain` of the grammar, refusals of malformed or unsignalled input | ≤ 15 ms | none (no socket) |
| **status** | bare `polylogue`, `status`, `status --line`, `pack` with nothing to include | ≤ 100 ms; `status --line` ≤ 30 ms, because it runs in harness status bars | 80 ms |
| **completion** | Tab in any shell, MCP `completion/complete`, the web query box | ≤ 50 ms per keystroke, daemon up or down | none for values (catalog file); 30 ms for the count preview |
| **read** | `REF read`, `@N read`, one-row references, indexed lists (`origin:`, `repo:`, `since:`, `tag:` …), `resume`, `pack takeover`, `next` | ≤ 100 ms to the first screen | 250 ms |
| **search** | word and phrase search, `pack recall`, `similar:` with a stored vector | ≤ 250 ms first page | 2,000 ms |
| **aggregate** | `group by`, `| measure`, `| agg`, facet counts, `explain QUERY` with clause counts | ≤ 400 ms | 400 ms |

Ruling, 2026-05-12: "I expect invocation of something so trivial as help to be IDK 100ms … querying also should be subsecond" (cc-queued:64f29439-f42d-4128-88d5-1bc327cfdbf4@2026-05-12T01:49). A missed budget is a defect in the code, not a reason to relax the number. Budgets are measured as paired benchmark runs ([testing.md](testing.md#benchmarks)); no test asserts elapsed time.

**Deadlines are interactive defaults.** The CLI's socket timeout is the deadline plus 50 ms, so a hung daemon never hangs a shell. Any caller lifts a deadline (`--deadline MS`, `--deadline none`; `deadline: null` over MCP and HTTP); deadline behaviour and cancellation are in [query.md](query.md#deadlines). A timed-out aggregate prints the narrower query the planner can serve and the same query with the deadline lifted:

```console
$ polylogue 'tool_calls where outcome:error | group by tool | count'
polylogue: timeout after 400 ms in stage "group by" (1.2 M rows scanned)
narrower:  polylogue 'tool_calls where outcome:error since:30d | group by tool | count'
complete:  polylogue 'tool_calls where outcome:error | group by tool | count' --deadline none
```

Over MCP the timeout answer carries both as `next` entries.

### The daemon's states

| Daemon | Command line |
|---|---|
| busy with a first build or re-parse | interactive reads keep their budgets on the reserved read path; answers over data not yet indexed say `up_to_date: false` on stderr |
| starting (lock held, socket not bound) | waits up to the class deadline, printing "daemon starting" once on stderr, then exit 6 |
| down | the local class and completion (from the catalog file) work; everything else exits 6 with one line naming the archive and the installed service command (`systemctl --user start polylogue`), or `polylogue daemon` when no service is installed. Nothing starts a daemon implicitly |
| hung | the socket timeout ends the command, exit 7 |

## Warm state

The daemon keeps derived state warm so the classes hold. Each item is keyed by what it depends on, so no answer from before a write is served after it; none is a second archive, and each is rebuilt from the archive after a restart (`status` says `warming` until they are built). An LRU may evict items; it never limits answers.

| Warm item | Serves | Keyed by | Refreshed |
|---|---|---|---|
| catalog (memory, and published as a file) | completion, short IDs, reference resolution, picker candidates without a query | archive write position | incrementally from each committed batch |
| sorted session ID index with titles, last activity, origin, repository, first prompt | short IDs, prefixes, `latest`, the picker | write position | batch delta |
| status snapshot | bare `polylogue`, `status` | write position and live-state change | after each batch and each hook event |
| compiled plans | every query | normal form, definitions and options | on definition change |
| first pages, next pages and counts | repeated lists, `next`, footers, `explain`, previews | question hash and read position | evicted on write |
| saved-query names and kept-result shapes | compiling | write position | on user write |
| loaded vectors (int8 in memory, f32 on disk) and stored query vectors | `similar:` | vector write counter | on vector write |
| search vocabulary | fuzzy recall | write position | batch delta |
| rendered summaries of recent sessions | picker preview | session, read position, width | evicted on write to that session |

## Completion

Completion is advanced and works with the daemon down. Ruling, 2026-05-01: "CLI in particular _CERTAINLY_ requires advanced autocomplete, otherwise it is going to be ~pointless. Ideally that includes integrated fuzzyfinding somehow" (chatgpt:69f4b473-d540-83eb-b0ad-ad26c9ea32b1:06788182-5f1d-4763-9ee8-139dad3aed1e). Ruling, 2026-09-14, on completion with the daemon down: "yes. everything is supposed to be usable." (cc:652ae14f-a036-496b-abfa-42e7574c9ec9:473cc86c-de9a-40aa-bd32-30659f66bd71).

- **It runs in the CLI process,** against a catalog file the daemon publishes, so it never waits on the daemon. The socket `complete` request serves MCP `completion/complete`, HTTP and the web query box from the same catalog in the daemon's memory.
- **The grammar-position completer** is the shared `query-language` function: it takes the whole query text and a cursor offset and returns candidates with their replace range. Polylogue supplies the catalog.
- **Never** does completion run a search, open the archive or write.

| Position | Candidates |
|---|---|
| first word | command names with help, unit names, references (`last`, `latest`, `self`, `@last`, `@N` rows of the last list with their titles), shipped and saved query names, recent sessions by short ID with title |
| condition start | fields valid for the current unit, `session.` fields, `exists`, `count(`, `seq(`, `query:`, `result:`, `similar:`, `NOT`, `-` |
| after `field:` | closed sets; values present with counts (origins that have sessions, repository, project, cwd by path component, model, harness, tool, MCP server, tag and `tag:key:`, branch); `query:` names with parameters; `result:` IDs with names |
| after a comparison field | operators; time words (`today`, `7d`, `this week`, month names) |
| after `|` | stage names; after `group by` and `sort by`, the unit's fields and aggregate columns |
| inside `with` | child units and their fields |
| after a complete query, outside quotes | verbs valid for the query's unit, with help |
| a verb's options | its flags and closed values; `--view` lists views valid for the unit; `--to` targets; file paths for `--to FILE` and `import` |
| a reference argument | sessions by short ID with title, then rows of the last list; typed prefixes |
| `setting KEY VALUE` | keys with current values; closed values |

- **Order:** grammar candidates in declaration order, values by count descending, sessions by last activity. Ruling, 2026-01-24: "completions for id should likely be ordered by recency" (cc:1d6c0b33-94c7-4a2a-9957-347e5383f5c5:0e773aa3-b6f8-4cce-af2a-3f775d18e2f9). There are no fixed caps: the catalog holds every session and value, and the shell shows what fits.
- **Descriptions** (zsh, fish): `origin:codex  4,927 sessions`; `7f3a2c10  next friday returns the same day · 2h ago`; `group by  one row per value of a field`. Bash shows values only.
- **Title matching** at a reference position (zsh, fish): a word that is not a prefix of any ID is matched against titles with the fuzzy matcher, and those sessions' IDs are offered with the title as description.
- **Quoting:** the completer rebuilds query text and cursor offset from the shell's words and the open quote, and inserts candidates without closing the quote.
- **Count preview:** when the text before the cursor is a complete query, zsh and fish show one line that is not inserted, `→ 1,234 sessions`. The count is asked with a 30 ms deadline and left out when not available in time, never estimated.
- **Daemon down:** candidates come from the last published catalog, marked with its age in zsh and fish (`values as of 16:09 (daemon not running)`); bash has no message channel and shows the values unmarked. Without a catalog file, only grammar candidates.

### The catalog file

- **Location:** `$XDG_CACHE_HOME/polylogue/<archive key>/catalog`, so it survives a reboot before the daemon starts.
- **Contents:** a header (format version, archive key, write position, time written); values with counts per completable field; saved query names with parameters and help; kept results (ID, name, unit, row count); settings keys and values; sessions (full ID, shortest unique prefix, title, origin, repository, last activity, first prompt), sorted by ID with a second order by last activity.
- **Format:** versioned binary with a fixed header and offset tables, memory-mapped. A catalog of another format version is ignored, leaving grammar-only completion.
- **Freshness:** updated incrementally from each committed batch, written by temporary file, sync and rename, at most every 2 s while writes continue. Candidates may trail by that interval; query answers never do.
- It is derived and harmless to delete.

### Registration

`polylogue completions SHELL` prints Polylogue's own script for bash, zsh and fish. The script calls the program by the name it was invoked as, so `plg`, `plog` and wrappers bound to another archive complete against their own archive; it registers every installed name; it passes words, cursor index and open-quote state. zsh groups candidates under headings and uses messages for the preview and the catalog's age. The package installs the scripts into the shells' completion directories.

## Picker

One embedded picker, with fuzzy matching and an inline viewport below the prompt that leaves scrollback intact; it needs no external program.

- It opens only when stdin and stdout are terminals and no machine format (`--json`, `--jsonl`) is asked for. Running inside a harness is not a condition of its own: a harness's tool shell has no terminal.
- Candidates stream page by page as the user scrolls; matching keeps up with typing over every session.
- The preview pane shows the highlighted row as `read` would (a session summary, a message with one message of context either side, an assertion with its targets, a saved query's definition and last count), fetched from the daemon at the pane's width with a 100 ms deadline. A reply for a row no longer highlighted is never drawn.

| Situation | Candidates | Enter |
|---|---|---|
| a verb that takes one row (`read`, `resume`, `pack takeover`, `assert` on an ambiguous target) and a query that matched several | the query's rows | runs the verb on the chosen rows; Tab selects several |
| `pick` | the query's rows, or every session when there is no query | prints the chosen references |
| `query:` at a reference position with no name | saved and shipped queries | inserts the name |
| `judge` | the rows | one-key verdict without leaving the list |

A query matching exactly one row, or a unique prefix, never opens the picker. `--all-matches` acts on all matches; `--first` takes the first row in the query's order explicitly; the first row is never taken silently.

**Off a terminal,** a one-row verb with several matches exits 2 `ambiguous` with the count, the candidates (short ID, time, title) and the exact calls (`polylogue 'QUERY' read --all-matches`, `polylogue @N read`), and remembers the rows so `@N` works next. With `--json`: `{"outcome":"ambiguous","count":3,"rows":[…],"next":[…]}`.

**Shell widget:** `completions zsh|fish` also defines a widget, bound to `Alt-p`, that runs `polylogue pick` and inserts the chosen references at the cursor.

### Keys

The picker and the web use the same letter for the same act wherever both offer it; the shared key table is the common [`web-shell`](https://github.com/Sinity/polylogue-sinex-common/blob/master/docs/web-shell.md)'s, and Polylogue's web bindings are in [web.md](web.md). The picker uses Ctrl forms for letters that would otherwise type into its query.

| Act | Picker |
|---|---|
| move | `↑`/`↓`, `Ctrl-j`/`Ctrl-k` |
| multi-select | `Tab` |
| accept, open | `Enter` |
| print reference to stdout | `Ctrl-p` |
| open in the web reader | `Ctrl-o` |
| copy reference | `Ctrl-y` |
| toggle preview | `Ctrl-/` |
| cancel | `Esc` |

## Addressing

The reference forms are [references.md](references.md#forms)'s. The command line adds local forms that name rows of the last list printed:

| Form | Means |
|---|---|
| `@3` | row 3 of the last list printed here |
| `@3.2` | hit 2 of row 3 of the last search |
| `@2-5`, `@1,4`, `@*` | rows 2 to 5; rows 1 and 4; every row of the last list as printed |
| `@last` | the last result printed here: its `result:<id>` when kept, else its `query:<hash>` at its read position |

- **"Here"** is the terminal, or, for a caller with no terminal, the calling session (its harness session ID), so agents' tool shells get `@N` and `@last` per session. A caller with neither is refused with "rows are remembered per terminal or session; use the ID".
- `@` forms are rewritten after argument parsing, and only in arguments of reference or query kind; free text (assertion values, titles, setting values) is kept as typed.
- `@N` sets travel as a list parameter, not as text, and row numbers never leave the command line: the request carries the resolved references.
- **The row file**, per terminal or session and per archive, holds the list's query hash and text, unit, read position, the displayed rows' references with hit references, and the next cursor. `next` prints the following page with continued row numbers; a cursor invalidated by a rebuild answers `stale_cursor` and offers to run the query again.

### Footers and affordances

Every list, search and aggregate printed for a person ends, on stderr, with what the page is, its handle and the most useful next command:

```console
50 of 1,234 sessions · query:3f9a1c2e · next: polylogue next
```

When the result is kept, the footer names `result:<id>` instead. Stdout stays rows only, so pipes receive rows; `--json` carries `query`, `result` and `next` in the answer.

Every write verb prints what it wrote (the count and the assertion references) and the exact command that undoes it (`polylogue assertion:… withdraw`).

One declared **affordance table** per verb lists the next actions offered after its output (after a list: `@1 read`, `next`, `--why`; after an empty result: `--why` and the "where else" notes; after `keep`: `@last read`). The CLI prints the first as the footer's `next:`, MCP returns them as `next`, and the web shows them as actions; no surface offers an action outside the table.

## Output

- **Machine formats:** `--json` (one document, the answer envelope of [query.md](query.md#answers)) and `--jsonl` (header, rows, then `{"next_cursor": …}`) on every command; `--fields a,b` selects columns; `--page-size N`, `--cursor C`. A command's JSON schema is generated from its answer type (MCP `outputSchema` uses the same) and published by `explain capabilities --json`.
- **Colour and decoration** only when stdout is a terminal, `NO_COLOR` is unset and `TERM` is not `dumb`; stderr is judged on its own. There is no `--plain` flag, because detection decides. Only the 16 ANSI colours are used, so the terminal's palette applies.
- **Width:** the terminal's; 100 columns for pipes, so piped columns are stable.
- **Tables:** columns sized by display width; numbers right-aligned; unknown shown as `—`, never `0`. When a table is wider than the terminal, columns drop in a declared order per unit (for sessions: cwd, origin, message count; row number, time, short ID, title and cost stay), then the title is cut with `…`. Row numbers are plain and dimmed.
- **Values:** times local to the logical day's zone, with the year omitted in the current year; durations `3m 43s`; costs `$2.85`, unknown `—`.
- **Pager:** on a terminal, output longer than the screen goes to `$PAGER` (default `less -FRX`). The first page is written as soon as it arrives, later pages are requested only as the pager consumes earlier ones, and quitting the pager cancels the rest.
- **Archived text:** C0 and C1 control characters other than tab and newline, and bidirectional overrides, are shown escaped (`\x1b`) in every terminal output, the picker preview included. This protects the terminal and alters no stored text.
- **Errors:** one printer for every failure: kind, position with a caret when there is one, and a runnable `try:` line, shell-quoted, with archived IDs quoted so a pasted ID cannot inject. With `--json` the error is `{"error":{"kind","message","position","suggestion","retryable"}}` on stdout and nothing on stderr.

```console
$ polylogue 'sessions | grup by origin'
polylogue: invalid_query at 11: unknown stage "grup"
  sessions | grup by origin
             ^^^^
try: polylogue 'sessions | group by origin'
```

## Exit codes

The exit and HTTP table is the common `command-surface`'s, shared with Sinex; each failure kind maps to exactly one code. Polylogue's kinds:

| Code | Polylogue kinds |
|---|---|
| 0 | `ok` (rows, or a measure) |
| 1 | `empty`: the read succeeded and matched nothing |
| 2 | `invalid_query`, `unknown_field`, `bad_value`, `not_a_query`, `bad_request`, `ambiguous`, `stale_cursor`, `not_built` |
| 3 | `refused`, with the operation's stated reason |
| 4 | `not_found`, `conflict`, `io` |
| 5 | `degraded`: an `incomplete` answer, rows printed, part of what was needed missing |
| 6 | `not_running`: the daemon is down |
| 7 | `timeout`, `cancelled` |

Exit 78 is used only by the daemon process itself, when it refuses to open an archive whose stored format it does not read.

## Help

- Each table row carries one sentence (shown in lists), an optional description (`COMMAND --help`), its reason, its latency class and one to four examples, each a command line on the Morrow archive with its first output lines.
- `explain` describes the language and how a query is read ([query.md](query.md#explain)); `explain capabilities` lists the table with each row's reason.
- An empty result says what to change (`--why`, the "where else" notes) and exits 1.
- Bare `polylogue` is orientation, in the status class; its contents are [operations.md](operations.md#status)'s.
- There is no manual or tutorial command: help is learned in place, `demo` builds a Morrow archive to try things on, and the agent manual is generated from the same table ([agents.md](agents.md#agent-manual)).

## Laws

- **mode-parity**: every Morrow example gives identical rows, totals and refusals through CLI `--json`, the socket, HTTP and MCP `structuredContent`, because one daemon assembly answers all.
- **unsignalled-word-refused**: `polylogue chrono` exits 2 within the local budget with the `find` form and no command suggestion; `polylogue find chrono` and `polylogue 'rate limit'` search.
- **table-rows-declared**: every command-table row has a reason, a latency class and at least one example, and every example runs on Morrow and matches its golden.
- **options-anywhere**: `polylogue 'sessions | limit 1' --json` equals `polylogue --json 'sessions | limit 1'`; `polylogue find query --help` prints help, never a search.
- **class-budgets**: each latency class meets its budget at p95 through the real binary on a real-shaped archive, in paired benchmark runs.
- **deadline-timeout**: a request past its deadline ends `timeout`, exit 7, never `ok` with a partial page, and no read for it is left running in the daemon 100 ms later; closing the connection frees the read within 100 ms.
- **deadline-lift**: `--deadline none` and MCP `deadline: null` run a query that times out by default to completion; a default timeout of an aggregate prints the narrower query and the lifted command.
- **no-caller-refusal**: no refusal depends on the caller's harness or author; each terminal-only behaviour, off a terminal, names the non-interactive form that does the same act.
- **completion-matrix**: the position matrix (position × unit × shell) passes through `COMPLETE=bash|zsh|fish` on the empty and the Morrow archive.
- **completion-daemon-down**: with the daemon down, catalog values are offered within 50 ms and zsh shows the catalog's age; the completing process opens no file under the archive directory.
- **completion-under-writes**: while a writer commits one batch per second, no keystroke returns zero candidates for a position that has values.
- **candidates-parse**: every candidate, spliced at its replace range, parses (property test over generated queries and offsets).
- **completion-surfaces-agree**: MCP `completion/complete` returns the same candidates as the shell for the same text and offset.
- **wrapper-completes-own-archive**: a wrapper bound to its own archive completes values from that archive; `plg` completes.
- **picker-in-terminal**: in a PTY on Morrow, typing, moving, Tab and Enter act on the chosen rows; a late preview for a row no longer highlighted is never drawn.
- **ambiguous-off-terminal**: off a terminal or with `--json`, an ambiguous one-row verb exits 2 with candidates and exact calls, and `@N` works next.
- **key-parity**: every act in both the picker table and the web key table has the same letter.
- **at-forms-only-in-references**: `REF assert note "see @2"` stores `see @2`; `@2-4 assert tag x` tags exactly rows 2, 3 and 4; in a harness tool shell without a terminal, `@N` resolves against the calling session's last list.
- **footer-handle**: every list, search and aggregate printed for a person carries its `query:<hash>` in the footer, and `result:<id>` when kept; `@last read` reproduces the same rows.
- **next-continues**: `next` after a page prints the following page with continued numbers; after a rebuild it answers `stale_cursor`.
- **write-undo**: every write verb prints its undo command, and running it restores the previous current state.
- **output-gallery**: every command's terminal output on Morrow at widths 60, 80, 120 and 200, with `NO_COLOR=1` and piped, is checked on a terminal-cell model: no ANSI in a pipe, no line over the width, no traceback, no exit code outside the table, and colour never changes the text.
- **exit-code-per-kind**: one invocation per failure kind exits with its table code.
- **control-characters-inert**: a fixture with `\x1b]0;pwned\x07` in a title and a message renders inert in lists, reads and previews.
- **long-read-first-screen**: `read` of a 10,000-message session shows its first screen within the read budget, and quitting the pager ends the command within 100 ms with no read left running.

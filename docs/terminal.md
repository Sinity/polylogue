# Terminal interface

The picker, row addressing, output and help present the
[operation table and grammar](interfaces.md) in a terminal.

## Picker

One embedded picker, with fuzzy matching and an inline viewport below the prompt that leaves scrollback intact; it needs no external program.

- It opens only when stdin and stdout are terminals and no machine format (`--json`, `--format json|jsonl`) is asked for. Running inside a harness is not a condition of its own: a harness's tool shell has no terminal.
- Candidates stream page by page as the user scrolls; matching keeps up with typing over every session.
- The preview pane shows the highlighted row as `read` would (a session summary, a message with one message of context either side, an assertion with its targets, a saved query's definition and last count), fetched from the daemon at the pane's width with a 100 ms deadline. A reply for a row no longer highlighted is never drawn.

| Situation | Candidates | Enter |
|---|---|---|
| a verb that takes one row (`read`, `resume`, `pack takeover`, `assert` on an ambiguous target) and a query that matched several | the query's rows | runs the verb on the chosen rows; Tab selects several |
| `pick` | the query's rows, or every session when there is no query | prints the chosen references |
| `query:` at a reference position with no name | saved and shipped queries | inserts the name |
| `judge` | the rows | one-key verdict without leaving the list |

A query matching exactly one row, or a unique prefix, never opens the picker. `--all-matches` acts on all matches; `--first` takes the first row in the query's order explicitly; the first row is never taken silently.

**Off a terminal,** a one-row verb with several matches exits 2 `ambiguous` with the count, the candidates (short ID, time, title) and the ready calls (`polylogue 'QUERY' read --all-matches`, `polylogue @N read`), and remembers the rows so `@N` works next. With `--json`, the shared error answer has `outcome: "error"`, kind `ambiguous`, candidates and ready calls.

**Shell widget:** `completions zsh|fish` also defines a widget, bound to `Alt-p`, that runs `polylogue pick` and inserts the chosen references at the cursor.

### Keys

The picker and the web use the same letter for the same act wherever both offer it; the shared key table is the common [`web-shell`](https://github.com/Sinity/polylogue-sinex-common/blob/master/crates/web-shell/README.md)'s, and Polylogue's web bindings are in [web.md](web.md). The picker uses Ctrl forms for letters that would otherwise type into its query.

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

Every write verb prints what it wrote (the count and the assertion references) and the command that undoes it (`polylogue assertion:… withdraw`).

One declared **affordance table** per verb lists the next actions offered after its output (after a list: `@1 read`, `next`, `--why`; after an empty result: `--why` and the "where else" notes; after `keep`: `@last read`). The CLI prints the first as the footer's `next:`, MCP returns them as `next`, and the web shows them as actions; no surface offers an action outside the table.

## Output

- **Machine formats:** `--json` selects structured output. Finite daemon answers use the answer envelope of [query.md](query.md#answers); local commands use their declared output. Reads also accept `--format jsonl`. Query rows accept `--fields a,b`, `--limit N` (rows per page) and `--cursor C`; reads use `--page-tokens N` and `--cursor C`. A command's JSON schema is generated from its answer type and published by `explain capabilities --json`; MCP tools carry no output schema.
- **Outcome and notes:** the operation that assembles an answer decides its outcome (`ok`, `empty`, `degraded`, `error`) and its notes; every surface carries both unchanged, and no surface infers an outcome from the value. On a terminal the notes go to stderr after the value.
- **Colour and decoration** only when stdout is a terminal, `NO_COLOR` is unset and `TERM` is not `dumb`; stderr is judged on its own. There is no `--plain` flag, because detection decides. Only the 16 ANSI colours are used, so the terminal's palette applies.
- **Width:** the terminal's; 100 columns for pipes, so piped columns are stable.
- **Tables:** columns sized by display width; numbers right-aligned; unknown shown as `—`, never `0`. When a table is wider than the terminal, columns drop in a declared order per unit (for sessions: cwd, origin, message count; row number, time, short ID, title and cost stay), then the title is cut with `…`. Row numbers are plain and dimmed.
- **Values:** times local to the logical day's zone, with the year omitted in the current year; durations `3m 43s`; costs `$2.85`, unknown `—`.
- **Pager:** on a terminal, output longer than the screen goes to `$PAGER` (default `less -FRX`). The first page is written as soon as it arrives, later pages are requested only as the pager consumes earlier ones, and quitting the pager cancels the rest.
- **Archived text:** C0 and C1 control characters other than tab and newline, and bidirectional overrides, are shown escaped (`\x1b`) in every terminal output, the picker preview included. This protects the terminal and alters no stored text.
- **Errors:** one printer for every failure: kind, position with a caret when there is one, and a runnable `try:` line, shell-quoted, with archived IDs quoted so a pasted ID cannot inject. With `--json` the answer has `outcome: "error"` and an `error` object with `kind`, `message`, `position`, `suggestion`, `candidates` and `retryable` on stdout and nothing on stderr; `candidates` lists what an ambiguous reference matched, each on its own line for a person.

```console
$ polylogue 'sessions | grup by origin'
polylogue: invalid_query at 11: unknown stage "grup"
  sessions | grup by origin
             ^^^^
try: polylogue 'sessions | group by origin'
```

## Help

- Each table row carries one sentence (shown in lists), an optional description (`COMMAND --help`), its reason, its latency class and one to four examples, each a command line on the Morrow archive with its first output lines.
- `explain` describes the language and how a query is read ([query.md](query.md#explain)); `explain capabilities` lists the table with each row's reason.
- An empty result says what to change (`--why`, the "where else" notes) and exits 1.
- Bare `polylogue` is orientation, in the status class; its contents are [operations.md](operations.md#status)'s.
- `manual` prints the embedded agent card, guide and generated operation reference without a daemon; `demo tour DIR` builds a Morrow archive to try things on ([agents.md](agents.md#agent-manual)).

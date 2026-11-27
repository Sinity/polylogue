# Terminal interface

What a person sees in a terminal: the orientation screen, the picker, the editor, row numbers, footers and next lines, refusals, output, the cheatsheet and help topics, `--explain`, and the defaults a person would guess. The grammar these present is [interfaces.md](interfaces.md); completion is [completion.md](completion.md).

Nothing here needs a manual. Every mechanism teaches the same grammar in the same words and shows a runnable line, never a description of one; every line any of them prints is parsed by the real line parser in a test.

## The orientation screen

Bare `polylogue` on a terminal answers within the status budget, from the daemon's status snapshot and the catalog:

```console
$ polylogue
polylogue · healthy · 41,212 sessions · up to date 3 s ago · claude 5h 61 % (resets 16:40) · codex week 34 %
waiting on you
  @1  codex        sinex      permission: apply_patch crates/store/src/writer.rs   4m
running
  @2  claude-code  polylogue  working · Bash cargo xtask check --changed           12m
  @3  claude-code  sinnix     idle                                                  2m
recent here (polylogue)
  @4  10:02      Replace opaque vocabulary in one pass   claude-code  $4.10
  @5  yesterday  Store text bodies compressed in groups  codex        —
next: polylogue @1 answer allow · polylogue @2 read --follow · polylogue 'repo:here since:7d'
Tab completes every position · polylogue help is one screen
```

- The screen is the shipped page `page:home` rendered for a terminal ([pages.md](pages.md#templates-and-shipped-pages)); revising that page changes the screen. Rows new since the reader's seen mark ([reading.md](reading.md#ranges)) are marked.
- **Pages changed since you looked** (`edits` with `since:seen`) and **Open items** (`open`) follow `recent here` when they have rows ([page-templates.md](page-templates.md#shipped-pages-and-templates)).
- Each section is a shipped saved query's listing (`waiting`, `running` and `recent`, the last narrowed to `here`; [saved-queries.md](saved-queries.md#shipped-saved-queries)), so the same rows are one word away; rows are numbered, so row numbers work at once.
- The `next:` line is built from what is on screen: answer the first waiting session, follow the first running one, this repository this week. With nothing waiting it offers a recent session's `read` and the `brief` of a recent phrase.
- The last line shows while the archive has recorded fewer than about twenty runs by the operator, counted from the `runs` unit with no separate state.
- **Daemon down:** one line naming the service and how to start it, then `polylogue demo DIR`; exit 6.
- **Empty archive:** which harness locations are watched and found, a `polylogue install HARNESS` line for each harness found but not wired, `polylogue import PATH` and `polylogue demo DIR`.
- **Off a terminal**, bare `polylogue` prints the `status` block ([operations.md](operations.md#status)).

## The picker

One embedded fuzzy picker, with no external program, in an inline viewport below the prompt that leaves scrollback intact. It opens

- for `pick` in a line, over the selection's rows;
- for a verb with no selection, over its usual rows ([interfaces.md](interfaces.md#a-verb-with-no-selection));
- for a one-row verb given several rows on a terminal;
- for Alt-p in any shell line, inserting the chosen references at the cursor (the key `install zsh` and `install fish` bind, [installing.md](installing.md#installing)).

It opens only when standard input and a display output are terminals and no machine format (`--json`, `--format json|jsonl`) is asked for; running inside a harness is not a condition of its own, since a harness's tool shell has no terminal. It never opens for one row and never takes a row silently. An explicit `pick` may send its references to a pipe while drawing on terminal standard error, as the Alt-p shell binding does. Automatic pickers still require terminal standard output. The line runner prints the selected references once; drawing and explicit Ctrl-p output stay on the display terminal.

```
 polylogue 'repo:sinex since:7d' pick › back off                        → 212 sessions
   1  Nov 12  Back off on 429 from the provider API      sinex  claude-code
 › 2  Nov 11  Store text bodies compressed in groups     sinex  codex
 ─ preview ──────────────────────────────────────────────────────────────────
 Store text bodies compressed in groups
 codex · Nov 11 21:02–23:40 · ~/sinex · 41 messages · 22 tool calls · $3.10
 Enter tag reviewed · Tab mark · ^r read · ^o browser · ^y copy · ^p print · ^e query · ^/ preview · Esc
```

- **Filtering.** The prompt filters fuzzily over titles, first prompts and repositories, ignoring letter case, from the session index the CLI reads from the catalog. `Ctrl-e` switches it to query mode: the text is a query, the list runs again as it changes, stale replies are dropped and the count updates. Candidates stream page by page as the user scrolls.
- **A page on its way** shows `loading` where the count goes and no rows, and Enter waits for it; a page with no rows shows its count, `0 rows`. A page that could not be read shows `failed` there and the reason on the line above the footer. The highlight starts on a new page's first row, or on its last row when the page was reached by moving up.
- **The footer always shows the keys**, generated from one key table. Enter runs the line's next verb with its words, shown in the footer (`Enter tag reviewed`; `Enter read` for `polylogue read`). When nothing follows `pick`, Enter prints the chosen references, one per line, which is what Alt-p inserts. `Ctrl-r` reads the highlighted row in the pager and returns to the picker.
- **The preview** shows the highlighted row as `read --view overview` would, at the pane's width; a reply for a row no longer highlighted is never drawn. Archived text is shown with control characters escaped.
- **Off a terminal or with `--json`**, a line that needs the picker is refused `ambiguous` with the candidates (short ID, time, title) and ready lines (`polylogue SESSION resume`, `polylogue @N read` on a terminal); it never dumps every candidate into a pipe. With `--json` the shared error reply carries kind `ambiguous`, the candidates and the ready lines.

### Keys

The picker and the web use the same letter for the same act wherever both offer it, except marking (`Tab` in the picker, `x` on the web) and rejecting a judged annotation (`Ctrl-x` in the picker, `r` on the web's review card), whose picker letters are taken by the prompt's reading and mark keys. Movement and copying take their letters from the common [`web-shell`](https://github.com/Sinity/polylogue-sinex-common/blob/master/crates/web-shell/README.md)'s key table; the picker's own table adds its reading, printing, mode and preview keys and drives both dispatch and the footer. Polylogue's web bindings are in [web.md](web.md#keys). The picker uses Ctrl forms for letters that would otherwise type into its prompt.

| Act | Key |
|---|---|
| move | `↑`/`↓`, `Ctrl-j`/`Ctrl-k` |
| mark several | `Tab` |
| run the line's next verb, or print the chosen references | `Enter` |
| read the highlighted row in the pager | `Ctrl-r` |
| open in the browser | `Ctrl-o` |
| copy the reference | `Ctrl-y` |
| print the reference and stay | `Ctrl-p` |
| query mode | `Ctrl-e` |
| toggle the preview | `Ctrl-/` |
| cancel | `Esc` |

Over annotations, three more keys make the one-key judging loop: each records a judgment of the highlighted annotation at its shown revision and moves to the next row; moving past a row writes nothing, and each judgment prints its undo line when the picker closes ([judging.md](judging.md)).

| Act, over annotations | Key |
|---|---|
| accept | `Ctrl-a` |
| reject | `Ctrl-x` |
| defer | `Ctrl-d` |

Accept and defer are the web's `a` and `d` with Ctrl; reject is `Ctrl-x` because `Ctrl-r` reads. Every other write is typed on the line: `polylogue review pick judge accept` judges the chosen annotations.

## The editor

`edit` with no text on a terminal opens the selected annotations, or a page as its Markdown working copy, in `$VISUAL` or `$EDITOR`, and checks in what you save when the editor exits (`polylogue page:flaky-ci edit`); the format, the check-in and refused paragraphs reopening with their reasons are [page-working-copy.md](page-working-copy.md#editing-in-your-editor). Off a terminal `edit` needs its text, and the refusal names the `read --to` and `import` lines that work there.

## Row numbers

Row numbers name rows of the last list printed on this terminal. The other reference forms are [references.md](references.md#forms)'s.

| Form | Means |
|---|---|
| `@3` | row 3 of the last list printed on this terminal |
| `@3.2` | hit 2 of row 3 of the last search printed on this terminal: the `span:` of that hit ([references.md](references.md#forms)) |
| `@2-5`, `@1,4` | rows 2 to 5; rows 1 and 4 |
| `@all` | every row shown |

- **Terminal only.** Row numbers resolve when standard input and output are a terminal. Elsewhere (pipes, agents' tool shells) `@3` is refused `row_numbers_off_terminal` with "use the references printed above": parallel subagents share one harness session ID, so a list remembered per session would let one act on another's rows. Every reply carries short references, so agents lose nothing.
- **The row file**, per terminal and archive, holds the last list's question, the shown rows' references and when they were shown. Only list replies replace it: a listing, a page of picker candidates once it has arrived and `ambiguous` candidates. Reads, writes and acts never do. A list that cannot be written removes the previous one, so a row number never names a row of an older list.
- A write or act that used row numbers prints `rows of '<question>' shown <n> min ago` before its outcome.
- Row numbers are rewritten only in the selection position (and in `judge`'s word), never inside text. They travel as resolved references: row numbers never leave the command line, and the daemon refuses a row number with "row numbers are resolved by the command line".

## Footers and next lines

Every list, read and write ends, on standard error, with what it was and up to three runnable next lines from the verb's affordance table:

```
20 of 143 sessions · words in prose and tool output · ranked · query:3f9a1c2e
next: polylogue @1 read · polylogue 'rate limit' pick read · polylogue 'rate limit' save
```

- A default that removed rows is named whenever it removed any (`3 subagent sessions folded into their parents`); a plain-word search says it was one.
- After a write: what was written, the recorded author, and the undo command ([annotations.md](annotations.md#the-write-path)).
- After an act: who it reached, by which channel, and that it cannot be undone.
- After an empty reply (exit 1): how many rows each condition kept and the nearest line that returns rows.
- When the rows are kept, the footer names `result:<id>`.

Standard output stays rows only, so pipes receive rows; `--json` carries `query`, `result` and `next` in the reply. One declared affordance table per verb lists the next actions offered after its output; the CLI prints them as `next:`, MCP returns them as `next` calls ([agents.md](agents.md#next-calls)), and the web shows them as actions. No surface offers an action outside the table.

## Refusals

One printer for every failure: what is wrong, a caret under it when there is a position, and one `try:` line that runs as shown, shell-quoted so a pasted ID cannot inject. Each refusal of the grammar is exit 2 and answers within the local budget. Command and verb names are never guessed; near-miss suggestions exist only inside a query, where the catalog knows the candidates.

| Line | Reply |
|---|---|
| `polylogue repo:sinex since:7d read` | `"since:7d" is not a verb; a query is one argument` · `try: polylogue 'repo:sinex since:7d' read` |
| `polylogue 'repo:x' tag later flaky` | `"flaky" is not a verb; tag takes one word` · `try: polylogue 'repo:x' tag later` |
| `polylogue @1 note flaky clock` | `note takes one text argument` · `try: polylogue @1 note 'flaky clock'` |
| `polylogue status code 500` | `status takes no words` · `to search: polylogue 'status code 500'` |
| `polylogue 'repo:x' tagg later` | `"tagg" is not a verb` and the verbs valid on sessions, one line each |
| `polylogue note 'x'` | `note needs rows first` · `polylogue @1 note 'x'` · `polylogue 'QUERY' pick note 'x'` |
| `polylogue @1 tag` | `tag needs a name` · `try: polylogue @1 tag NAME` · the tags in use with counts |
| `polylogue - note -` | `standard input can be read once per line` |
| `polylogue 'sessions \| grup by origin'` | a caret under `grup` · `try: polylogue 'sessions \| group by origin'` (the shared query language's suggestion) |
| a write over 214 query rows without `--all` | `214 sessions match; to tag all: polylogue 'repo:x' tag later --all`, the first rows, and `pick` as the way to choose |
| a one-row verb over several rows, off a terminal | the candidates (short ID, time, title) and `polylogue SESSION resume` lines |
| `@3` off a terminal | `row numbers work on a terminal; use the references printed above` |
| `polylogue page:flaky-ci edit` off a terminal | `edit needs its text off a terminal` · `try: polylogue page:flaky-ci read --to .page/` · then `polylogue import .page/flaky-ci.md` |
| `--blind` on a selection that is not a kept result | `blind judging needs kept rows: polylogue '@1,2' save` |
| a timeout (exit 7) | `narrower:` when a shorter query exists, and `complete: … --deadline none`, both runnable ([latency.md](latency.md#writes-and-deadlines)) |
| a stale cursor or a re-parsed session | what changed and the line to run again |
| an MCP call with a misplaced argument | the corrected call, as a `next` call |

```console
$ polylogue 'sessions | grup by origin'
polylogue: invalid_query at 11: unknown transform "grup"
  sessions | grup by origin
             ^^^^
try: polylogue 'sessions | group by origin'
```

With `--json` the reply has `outcome: "error"` and an `error` object on standard output and nothing on standard error; `candidates` lists what an ambiguous reference matched, each on its own line for a person.

## Output

- **Machine formats:** `--json` selects the typed reply. Finite daemon replies use the reply envelope of [query.md](query.md#replies); local commands use their declared output. Listings accept `--fields a,b`, `--page-size N`, `--cursor C` and `--budget N`; reads take `--budget N` and `--cursor C`. Each operation's JSON schema is generated from its reply type and published by `help capabilities`; MCP tools carry no output schema.
- **Outcome and notes:** the operation that assembles a reply decides its outcome (`ok`, `empty`, `degraded`, `error`) and its notes; every surface carries both unchanged, and no surface infers an outcome from the value. On a terminal the notes go to standard error after the value.
- **Colour and decoration** only when standard output is a terminal, `NO_COLOR` is unset and `TERM` is not `dumb`; standard error is judged on its own. There is no `--plain` option, because detection decides. Only the 16 ANSI colours are used, so the terminal's palette applies.
- **Width:** the terminal's; 100 columns for pipes, so piped columns are stable.
- **Tables:** columns sized by display width; numbers right-aligned; unknown shown as `—`, never `0`. When a table is wider than the terminal, columns drop in a declared order per unit (for sessions: working directory, origin, message count; row number, time, short ID, title and cost stay), then the title is cut with `…`. Row numbers are plain and dimmed.
- **Values:** times local to the logical day's zone, with the year dropped in the current year; durations `3m 43s`; costs `$2.85`, unknown `—`.
- **Pager:** on a terminal, output longer than the screen goes to `$PAGER` (default `less -FRX`). The first page is written as soon as it arrives, later pages are requested only as the pager consumes earlier ones, and quitting the pager stops the reading.
- **Archived text:** C0 and C1 control characters other than tab and newline, and bidirectional overrides, are shown escaped (`\x1b`) in every terminal output, the picker preview included. This protects the terminal and alters no stored text.

## Help

`polylogue help` prints the cheatsheet: one terminal screen (at most 24 lines at 100 columns), generated from the operation table, the catalog's top values and each row's examples; local, in the local class, with no daemon.

```text
polylogue 'QUERY' [VERB …]   the quotes hold which rows; the words after say what to do
polylogue REF [VERB …]       one thing: @3  latest  self  here  7f3a2c10  a chat URL  query:NAME  result:ID
polylogue read · resume      pick from recent sessions, or from sessions that ended open
polylogue brief              what to know here; SESSION brief: take it over; 'words' brief: recall
look      read [--view V] [--follow]   pick   brief
annotate  note 'TEXT' --as KIND   tag NAME   edit ['TEXT']   withdraw   judge accept|reject|defer   save [NAME] --watch
act       send 'TEXT'   answer allow|deny|'TEXT'   interrupt   resume [--in HARNESS]
queries   'rate limit'   'repo:sinex since:7d'   'tool_calls where outcome:error | group by tool | count'
saved     waiting · running · recent · unfinished · review · notices · yours: failing …
pages     page:NAME read [--view contents]   page:NAME edit (your editor)   ROWS save page:NAME --section S [--as FORM]
commands  status · start 'TEXT' · import PATH|chatgpt · set [KEY VALUE] · web · embed · install · demo · dev
rows      @3  @2-5  @1,4  @all   (this terminal)     a plain word searches; Tab completes everything
any line  --explain (how it is read, runs nothing) · --dry-run (what it would do) · --all (every matched row)
try       polylogue 'repo:polylogue since:7d'  ·  polylogue @1 note 'jittered backoff' --as decision
more      polylogue help VERB · help query · help fields · help views · help kinds · help keys · help --all
```

- The `queries`, `saved` and `try` lines are filled from the catalog, so they return rows on this archive; without a catalog the sheet uses the Morrow values.
- **`polylogue help VERB`** (and `VERB --help`): the sentence, its words and options with their allowed values, its forms (for `brief`: alone, of a session "take it over", of a search "recall", of other rows), its usual rows if any, and two to four examples with their first output lines on Morrow (hand-written goldens, never regenerated from output and never run at build time: [testing.md](testing.md#goldens)). `help COMMAND` is the same for a command.
- **A word from the vocabulary's removed list** ([vocabulary.md](vocabulary.md#removed-words)) answers with the one line that does it now: `help takeover` prints `takeover is brief of a session: polylogue SESSION brief`, so a word a person remembers still finds its way.
- **Topics:** `help query` (the grammar and the query language in one screen), `help fields [UNIT]`, `help units`, `help transforms`, `help views`, `help kinds`, `help keys`, `help saved`, `help capabilities` page the generated declarations; `help unbuilt` prints the list of what is designed and not built. Over MCP `help(topic)` pages them at about 1,000 tokens a page.
- **`help --all`** is the full reference and the agents' skill file: every verb, command, option, unit and field, with each row's reason. `help capabilities` publishes every operation's argument and reply schemas, the arguments the MCP tool list leaves out included, and `help unbuilt` lists what is designed and not built yet, each with its page ([interfaces.md](interfaces.md#operation-table)). `polylogue help --to browser` (or `--format html|markdown --to FILE`) gives a printable page.
- `help` lists the first-word rule in one line: a word that is also a command runs the command, and inner quotes search for it.

## Explain

`--explain` on any line prints how it is read and what it would do, and runs nothing:

```console
$ polylogue 'repo:sinex since:7d' tag needs-tests --explain
first word: query text, rule 8
selection:  sessions in repository sinex active in the last 7 days, main sessions only, hidden ones excluded
defaults:   main sessions only (lift with 'kind:*'), hidden excluded (lift with 'hidden:*')
unit:       sessions · verbs: read pick brief note tag save send answer interrupt resume
step 1:     would tag each matched session needs-tests; needs --all if more than one matches; undo with withdraw
```

- It names the rule that read the first word ([interfaces.md](interfaces.md#the-first-word)), reads the selection back in words, lists the defaults it applied and how to lift each, names the unit and the verbs valid on it, and says per step what it would do with its words.
- For the two optional words it says which reading the line got: `polylogue page:flaky-ci edit read --explain` prints `step 1: edit with no text: opens page:flaky-ci in your editor` and `step 2: read`, and `polylogue annotation:01JB7Q2M edit 'read.' --explain` prints `step 1: edit with the text "read."`.
- On a command it says what the command would do with its arguments: `polylogue import chatgpt --explain` prints `would ask the browser extension to fetch chatgpt conversations the archive lacks`.
- A line that does not parse is explained up to where it fails, with the refusal's `try:` line.
- The grammar part needs no daemon; reading the selection back in words asks the daemon's compiler without running the query.
- `--dry-run` instead resolves the selection and reports counts and per-step effects ([interfaces.md](interfaces.md#steps)).
- Over MCP it is `explain: true` on any call, and the reply carries the parsed query with spans, which the web uses for facet editing.

## Defaults a person would guess

- The default unit is sessions: a session with its continuations and subagents ([sessions.md](sessions.md#logical-sessions)).
- A word search covers prose and tool input and output; the footer says what was searched; it ranks by relevance, ties going to the more recent message, and says so.
- Lists are newest first and fill one screen (20 sessions, 50 other rows); the pager takes over for more.
- Reads show the transcript with tool calls on one line and thinking folded.
- `here` is the repository with its worktrees, not the directory.
- One-row verbs never guess between rows; writes and acts never touch several query rows without `--all`; every write prints its undo line.
- Times are local, with the year dropped in the current year; unknown is `—`, never 0.
- Colour and the pager follow whether the output is a terminal.
- The same option has the same name on every verb ([interfaces.md](interfaces.md#option-names)).

## The first five minutes

For a person, on the Morrow archive:

1. `polylogue` shows the orientation screen with numbered rows and a `next:` line of three commands.
2. Running the third, `polylogue 'repo:here since:7d'`, gives a numbered table whose footer offers `polylogue @1 read`.
3. `polylogue @1 read` shows the transcript in the pager; `q` leaves it.
4. Typing `polylogue 'repo:here since:7d' ` and pressing Tab lists the verbs with one line each; choosing `pick read` opens the picker with its keys in the footer.
5. `polylogue help` shows the one screen, with examples drawn from this archive; Tab after an opening quote then teaches the fields with their values and counts.

Nothing in this sequence needs a word the screen did not show. An agent's first minute is in [agents.md](agents.md#how-agents-get-going).

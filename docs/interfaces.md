# Interfaces

Polylogue has one table of operations, and every surface is generated from it: the command line, the daemon's socket, HTTP and MCP. This page owns that table, the line grammar (selections, verbs, steps and commands), each verb's words, the rule that keeps the set of words small, option names, and failure kinds and exit codes. Latency classes, deadlines, the daemon's states as the command line sees them and warm state are [latency.md](latency.md).

The [terminal interface](terminal.md) owns what a person sees: the orientation screen, the picker, row numbers, footers, refusals, output, `help` and `--explain`. [Completion](completion.md) owns completion and its catalog file. Selections and the query language are [query.md](query.md), references [references.md](references.md), reading [reading.md](reading.md), annotations and the write path [annotations.md](annotations.md), live sessions [session-control.md](session-control.md), `brief` [agent-context.md](agent-context.md), the three MCP tools [agents.md](agents.md#mcp-server) and the web [web.md](web.md). The generic mechanics (the `Operation` trait, the failure-to-exit-and-HTTP table, request and reply envelopes, deadline classes, SSE framing) are the common [`command-surface`](https://github.com/Sinity/polylogue-sinex-common/blob/master/crates/command-surface/README.md) crate's.

## Principles

- **A selection says which rows, a verb says what to do, a view says how to show it.** A new need is met by the query language, a view or an option before a verb or command is added.
- **One grammar on every surface.** The command line, MCP and HTTP carry the same selection, verbs and argument names; the web is a client of the same operations and shows the line for every view.
- **Agents and the operator are equal callers.** Same verbs, same replies; only the recorded author differs ([authorship.md](authorship.md)).
- **Pages never cut silently.** Every budgeted reply names what it left out and the line or call that continues.
- **Plain names, fixed once, no aliases.** One word per thing on every surface ([vocabulary.md](vocabulary.md)). `plg` and `plog` are shell names the package installs, not aliases inside Polylogue.

## Operation table

- **One typed operation table** drives every surface. Each row declares the operation's words, latency class, failure kinds, one-sentence help, its reason for existing ([Discipline](#discipline)), its affordances (the next lines offered after it, [terminal.md](terminal.md#footers-and-next-lines)) and one to four examples on the Morrow archive, and names the operation that answers it. The operation carries the request and reply types and decides from the typed arguments how a request is treated, so the row is declared once with its operation and a row without one cannot be written. A new operation is a row and its operation, not code on each surface.
- **Arguments are declared on request types.** Each field of a row's request type is one argument on every surface: its doc comment is its help, its serde default its default, and its type gives the value kind, requiredness and choices. The schema key `argument_form` (`positional`, `words` or `rest`) places a field on the command line; without it a boolean is a flag, a list a repeated `--name` whose values may be comma-separated, and anything else `--name VALUE`. The key `unlisted` leaves a field out of the MCP tool list (`explain capabilities` still lists it). Options acting on the calling machine (`--to`, `--out`, `--exec`) are typed structs declared the same way and never enter the request. Help shows a row's examples under "Examples:". `--cursor` and `--budget` are envelope options on every command.
- **One request format.** The socket, HTTP and MCP carry the shared request: the operation, its typed arguments, the request ID, the deadline and the options beside them. One function answers it on every transport, on the thread that accepted it: it checks the envelope (a request ID, only the options Polylogue takes, well-formed steps), decodes the arguments once, decides the request's policy from them (the smallest tool that may run it, whether it changes the archive, how long it may read, which readers it may take, whether it pages or takes a token budget), computes its one read limit, and runs it. The hooks and the browser extension's host send the same requests for rows nobody types; these sit under the hidden `internal` command (`internal hook`, `internal host`, `internal browser-host`, `internal complete`, `internal advisory`, `internal delivery-event`, `internal mail-claim`, `internal permission`, `internal chat claim|report|answer|attachment`, `internal import claim|report`, `internal conversations`, `internal stored-files`, `internal selection-note`) and never appear in help or completion.
- **CLI values have one parser.** The shared `command-surface` argument declarations supply value kinds, required fields and choices for each operation. The line parser of this page builds the request envelope, the JSON output selection and the calling-process options from them. Domain catalogs keep the choices of named measures, help topics and hosted harnesses. File preparation, stream framing and daemon peer negotiation remain product behaviour. Local commands and calling-process options do not become daemon operations or request fields. The global options (`--json`, `--request`, `--request-id`, `--deadline`, `--cursor`, `--budget` and the deployment selections of [installing.md](installing.md#configuration)) are declared once, and one scan of the process arguments reads them and the invocation's shape (a line, a command, help, a hook, the browser host or a completion callback) for the command line, completion and the daemon's reading of a caller's command line ([authorship.md](authorship.md)).
- **Archive selection preserves path aliases.** An explicit archive path is made absolute without expanding its symlinks; the archive's identity is resolved separately, so a short Unix-socket path stays usable.
- **Ready calls are complete requests.** A line printed as a next step runs as shown. MCP groups the rows into three tools nested by access ([agents.md](agents.md#mcp-server)); a call is `{operation, arguments, request_id?, deadline?, budget_tokens?, cursor?}` with `operation` defaulting to `rows`. HTTP takes the `rows` arguments on `POST /api/v1/rows` and each command on `/api/v1/<command>` ([web.md](web.md#routes)); GET encodes arguments and the deadline as query values, and runs only a request whose policy reads and changes nothing; any other is refused with the route to POST. Every surface checks the same typed arguments. The API answers with the typed reply; the web's reader routes render the same typed reply and its next calls as HTML.
- **HTTP GET has one argument parser for finite replies and feeds.** The shared declarations decode both plain query pairs and complete generated requests, keeping request identity, deadlines and argument JSON distinct. Duplicate scalar or envelope fields are refused. `dev feed` resumes with its `--after` position, not a page cursor; feeds refuse token budgets and unsupported request options before production. A completed feed position is sent only with the final completion event.
- **Every reply is zero or more events and exactly one final answer.** On the socket a request is one JSON line and every reply uses the shared server-sent-event framing: the events, then one `answer` frame carrying the shared answer; a finite reply is the `answer` frame alone. On HTTP a reply that ended without events is one `application/json` body with the answer's status, and one whose first event was written is `text/event-stream` with the same frames. A transport learns that a reply streams when its first event is written. A failure after the first event is the final answer with an error outcome; there is no separate error event. A connection that closes before its answer means the daemon went away, and clients report it as `not_running`. While a request runs, its events are sent without waiting for the client, and what a slow client has not taken waits in memory and then in a file in the archive's staging directory; the rest is sent at the client's pace once the request has answered and holds no reader.
- **Finite transfers** are a begin event, record events and a completion event from one read. A client writes a directory bundle into place, and `dev feed` advances its saved position, only after the completion event. `read --follow` streams with the same framing until stopped.
- **A streamed line preserves step order.** A line containing any streamed read is one SSE response. Each read emits its own `begin`, `record` and `complete` events at that step. A `steps` event carries a typed `Answer<LineAnswer::Steps>` with each completed step's reply and undo lines exactly once. A streamed read's reply is `streamed` with its actual `ReadComplete`; a followed read's reply is `followed` with its final marks. These acknowledge already delivered records and do not contain another initial page. The final `answer` event carries the remaining failed or not-run steps and the whole line's empty, successful or degraded outcome. Clients finish a line at `answer`, rather than at an individual read's `complete`. A failed step prevents subsequent reads and writes.
- **A mutating line retains its first selection on retry.** Before its first step, the existing authored request journal stores the accepted line body and resolved references as `resolve_line`. The line request ID owns that body; each step retains its own derived request ID. A retry recovers the same references, including pending primary-mode requests, before rerunning any query. Reusing the line ID with different arguments or envelope options is a conflict. The journal record has no annotation effect; it uses the same preservation and confirmation path as other authored requests.

- **Access belongs to the request.** A line's access is its highest step's, decided from the typed arguments: `set` without a value and `import` of paths are not acts. MCP tool hints describe the tool; the same operation and arguments have the same access on every surface.
- **Replies are typed** and rendered from their types ([reading.md](reading.md)); each has one terminal printer and one JSON form, and no surface has features the others cannot reach. The operation keeps its declared reply type through execution and deadline handling. JSON serialization belongs to the transport or local wire boundary; the daemon does not serialize a reply and decode it back.
- **The daemon is required.** The command line is a thin client: it never opens archive files, and the daemon is the only writer. Completion, `help`, grammar refusals and `--explain` of the grammar run in the CLI from its own tables and the catalog file, without the daemon.
- **No refusal depends on who the caller is.** No act is refused because the caller is an agent or has no terminal; the only terminal-dependent behaviours are the picker, row numbers and `edit` with no text, and each refusal off a terminal names the line that works there ([terminal.md](terminal.md#the-picker)).
- There is no REPL or terminal reader; the interactive elements in a terminal are completion, row numbers, the pager, the picker and the editor that `edit` with no text opens, and interactive reading is the web reader.

## Grammar

```text
polylogue                                       the orientation screen (terminal.md)
polylogue COMMAND [ARGS] [OPTIONS]
polylogue SELECTION [OPTIONS] [VERB [WORD] [OPTIONS]]…
polylogue VERB [OPTIONS]                        only read, resume (the picker over their usual rows) and brief (this directory)
```

The quotes hold which rows; the words after them say what to do.

- **The selection is exactly one shell argument**: a quoted query (`'repo:sinex since:7d'`, `'tool_calls where outcome:error | group by tool | count'`), a reference, a row number or range, a saved query name, a unit name, a single plain word, or `-` ([Selections](query.md#selections)). A query is never spread over several arguments.
- **After the selection come verbs.** A verb is its name, then exactly its declared number of words ([Verb words](#verb-words)), each one shell argument, then its own options. The next argument that is neither an option nor owed to the current verb is the next verb. Because every verb's word count is fixed, a line parses left to right with one word of lookahead and no separator word.
- **The two optional words:** `save` takes a name or nothing, and `edit` takes its text or nothing. The word after either is its argument unless it is absent, starts with `-` (other than `-` alone, which reads standard input), or is a verb name. Saved names never equal a verb or command name ([saved-queries.md](saved-queries.md#saved-names)), so `save` is never ambiguous; a note whose whole new text is one verb word is written `edit 'read.'` or `edit -`. `--explain` shows which reading a line got ([terminal.md](terminal.md#explain)).
- **No verb lists the rows.** A selection alone lists them.
- **Options belong where they stand.** Options before the first verb belong to the listing (`--page-size`, `--fields`, `--why`, `--to`, `--cursor`); options after a verb belong to that verb; the line options (`--json`, `--deadline`, `--request-id`, `--dry-run`, `--explain`, `--wait`, `--all`) go anywhere, on commands too. An option a verb or command does not declare is refused (exit 2) with the ones it does declare. `--` before the selection takes the next argument literally as the selection, including a command or saved name, or a leading `-`. Before an owed verb word it takes that word literally, so `edit -- read` writes the text `read`; the next verb still starts the next step.
- **Free text is one argument.** `note`, `send`, `edit`, `answer` and `start` take their text as one shell argument, quoted when it has spaces, or `-` to read it from standard input. The text of `note` and `edit` is Markdown with references ([page-text.md](page-text.md#markdown-with-references)). `edit` with no text on a terminal opens the selected annotations, or a page, in your editor ([page-working-copy.md](page-working-copy.md#editing-in-your-editor)). A line reads standard input at most once: a `-` selection and a `-` text in one line are refused.

These executable examples use synthetic session and annotation references:

```bash
polylogue 'sessions'
polylogue 'sessions where repo:morrow' pick read
polylogue session:claude-code:example read --view dialogue --budget 1000
polylogue session:claude-code:example brief
polylogue session:claude-code:example note 'Investigate the latency.' --as finding
polylogue session:claude-code:example tag reviewed
polylogue annotation:example edit 'Confirmed the finding.'
polylogue annotation:example withdraw --reason 'Superseded finding.'
polylogue annotation:example judge accept --reason 'Matches the result.'
polylogue 'sessions where repo:morrow' save morrow-review
polylogue session:claude-code:example send 'Please check the result.'
polylogue session:claude-code:example answer allow
polylogue session:claude-code:example interrupt
polylogue session:claude-code:example resume --fork
```

### The first word

The CLI decides what the first argument is from its own tables and the catalog file, with no daemon round trip (local class). The first matching rule wins:

1. **Nothing**: the orientation screen on a terminal, the `status` block otherwise ([terminal.md](terminal.md#the-orientation-screen)).
2. **`-`**: references read from standard input, one per line.
3. **A command name** (the twelve [commands](#commands) and `internal`): that command. `--help` and `-h` are `help`.
4. **A verb name**: `read` and `resume` act on their usual rows through the picker, and `brief` gives the brief for this directory ([A verb with no selection](#a-verb-with-no-selection)); any other verb is refused with the forms that work, exit 2.
5. **A saved query name**, your own first, then the shipped ones ([saved-queries.md](saved-queries.md#shipped-saved-queries)): `polylogue waiting` is `polylogue query:waiting`. A saved query with parameters is written `'query:NAME(key=value)'`.
6. **A unit name** (`sessions`, `messages`, `tool_calls`, `annotations`, `results`, …): every row of that unit, with its defaults.
7. **A reference shape**: a typed prefix (`session:`, `codex:`, `annotation:`, `page:` with an optional `#ANCHOR` such as `page:flaky-ci#findings`, `query:`, `result:`, `run:`, `role:`, `file:`, …), a pasted provider URL, a bare ID prefix of at least eight hexadecimal characters and dashes, a row number form (`@3`, `@3.2`, `@2-5`, `@1,4`, `@all`), a member alias (`@reviewer`: `@` and a name starting with a letter, `all` excepted; [pages.md](pages.md#members)), or a reserved word (`latest`, `self`, `here`) ([references.md](references.md#forms)).
8. **Query text**: an argument holding whitespace, a field condition (`word:…`) or query punctuation (`|`, `"`, `(`, `)`, `~`, `*`, `,`).
9. **Any other single word**: a search for that word in prose and tool input and output, ranked; the footer says `searched the word "chrono"`.

Only the twelve command names, the thirteen verb names, the unit names and saved names are taken from plain-word search. A word that is also a command runs the command; searching for it needs inner quotes: `polylogue '"status"'`. Saving a name that an earlier rule would capture is refused with the reason, so a saved name is always reachable as a first word. Completion groups first-word candidates by rule and says which rule a typed word meets ([completion.md](completion.md#positions)); `--explain` names the rule ([terminal.md](terminal.md#explain)).

### A verb with no selection

| Line | Usual rows | On a terminal | Off a terminal |
|---|---|---|---|
| `polylogue read` | `recent` | the picker; Enter reads the chosen rows | refused, with the candidates and the lines `polylogue latest read` and `polylogue SESSION read` |
| `polylogue read --follow` | `running` | the picker; Tab marks several; Enter follows them | refused the same way |
| `polylogue resume` (also `resume --in HARNESS`) | `unfinished` (sessions here that ended open) | the picker over one row, each row with why it is open | refused the same way |
| `polylogue brief` | — | the brief for this directory (`--cwd DIR` for another), no picker | the same |

The picker never takes a row silently and never opens for a single match: one matching row runs directly. Writes and acts always need a selection.

## Verb words

Each verb takes a fixed number of words after it, then its options; only `save`'s name and `edit`'s text are optional. A verb's words always complete, so a person learns them by pressing Tab.

| Verb | Words after it | Options |
|---|---|---|
| `read` | 0 | `--view`, `--show`, `--hide`, `--from`, `--until`, `--first`, `--last`, `--around`, `--diff`, `--alternate`, `--beside`, `--blind`, `--seed`, `--physical`, `--neutral-names`, `--follow`, `--until-idle`, `--at`, `--align`, `--format`, `--to`, `--budget`, `--cursor` |
| `pick` | 0 | — |
| `brief` | 0 | `--cwd`, `--for`, `--budget`, `--to` |
| `note` | 1: `TEXT` | `--as KIND`, `--key`, `--parts FILE\|-`, `--evidence`, `--replies-to`, `--supports`, `--challenges`, `--qualifies`, `--corrects`, `--derived-from`, `--blocks`, `--scope`, `--valid-until`, `--check`, `--expect`; pages: `--page`, `--section`, `--level`, `--template`, `--param`, `--layout`, `--about`, `--role`, `--brief`; web-chat titles: `--rename` |
| `tag` | 1: `NAME` or `NAME:VALUE` | — |
| `edit` | 0 or 1: `TEXT` | `--key` |
| `withdraw` | 0 | `--reason` |
| `judge` | 1: `accept`, `reject`, `defer`; on several rows the preferred row (a reference, a row number, or a blind label `A`, `B`, …), an order (`'B>A>C'`), `tie`, `incomparable`, `abstain` or `insufficient-evidence` | `--on DIMENSION`, `--seed`, `--reason` |
| `save` | 0 or 1: `NAME` or `page:NAME` | `--watch`, `--hold`, `--notify`, `--denominator`; on a page: `--section`, `--after`, `--fixed`, `--live`, `--kept`, `--template`, `--as FORM` (`card`, `excerpt`, `text`, `full`, `table`, `list`, `count`, `value`, `chart`, `compare`) |
| `send` | 1: `TEXT` | `--about`, `--attach`, `--page`, `--section` |
| `answer` | 1: `allow`, `deny`, or quoted text | `--reason`, `--standing` |
| `interrupt` | 0 | — |
| `resume` | 0 | `--fork`, `--anyway`; `--in HARNESS`, and with it `--at`, `--cwd`, `--stop-source` |

Line options on every line: `--json`, `--deadline MS|none`, `--request-id ID`, `--dry-run`, `--explain`, `--wait D`, `--all`, `--fresh`. Listing options, before the first verb: `--page-size N`, `--fields a,b`, `--why`, `--to`, `--cursor`, `--budget`. The same concept has the same option name on every verb and command that has it ([Option names](#option-names)). Boolean line and listing flags accept explicit values, such as `--all=false` and `--why=true`.

## Steps

- **What flows is the selection**: its query and its rows, resolved once at the read position of the first step. Every verb acts on those same rows in the same order; nothing a verb produces feeds the next verb, so a write cannot change which rows a later verb sees (`polylogue 'tag:later' withdraw read` reads the rows that were tagged). A line with one step streams; resolving once costs one pass over the keys only when a line has two or more steps.
- **`pick` is the one verb that narrows** the rows, to those chosen in the picker. It needs a terminal; a line containing it is sent as two requests (resolve, then the remaining steps over the chosen references). `pick` as the last verb prints the chosen references, one per line.
- **`save` takes the question as well as the rows**: the selection's query text and the rows this line resolved ([saved-queries.md](saved-queries.md#save)); `save page:NAME` places the rows, or with `--live` the question, on a page, each shown in its default form or the one `--as FORM` names ([pages.md](pages.md#forms)).
- **`--wait D`** makes a listing return when rows match or the time passes (`polylogue 'session:codex:019d0f2a live:idle' --wait 30m`), and makes `import PROVIDER` return when its job finishes or the time passes. It raises the line's deadline to the wait; an answer that ran out of time is `empty` with the same line as its next line. Over MCP one call waits at most 50 seconds ([agents.md](agents.md#waiting)).
- **A grouped or measured selection** (`| group by`, `| measure`, `| count`) has rows that are not archive things: only the listing, `read`, `brief` and `save` accept it; other verbs refuse before anything runs.
- **One-row verbs** (`resume`, with or without `--in`) given several rows open the picker on a terminal and are refused `ambiguous` off it, with the candidates as ready lines. `brief` over several rows is the brief of those rows ([agent-context.md](agent-context.md#brief)), never a guess at one.
- **Writes and acts on several rows need `--all`.** When the selection is a query (a quoted query, a saved name, a unit, a plain word) and it matches more than one row, a line with any write or act step refuses before its first step (`needs_all`): it prints the count, the first rows, and the same line with `--all` appended. Row numbers and ranges, references, `-`, kept results and rows chosen in `pick` are sets the user named and need no flag. There is no yes/no prompt: a line behaves the same on every surface and when run again from shell history.
- **`send` to several sessions** reaches the running ones; ended ones are listed as skipped, each with the line that sends to it alone, because resuming starts a harness process and spends money ([session-control.md](session-control.md#send)).
- **Steps run in order, each to completion.** A write commits before the next step starts, so a `read` after it shows it. Each write step is its own request and transaction; the line is not one transaction, because acts reach harnesses and browsers that cannot be rolled back.
- **A failing step stops the line.** Earlier steps keep their effects; the reply lists every step (done, failed with its error, not run), prints the undo line of each committed write, and exits with the failed step's code.
- **Output.** Rendering steps write to standard output in order; footers, notes and undo lines go to standard error.
- **`read --follow` is the last step** of any line that has it.
- **`--dry-run`** covers the whole line: the selection is resolved and counted, and each step reports what it would do (the annotations it would write and any refusal it would meet; each recipient and channel of an act; the command `resume` would run; the brief size and harness `resume --in` would start), writing and reaching nothing. `pick` still opens its picker under `--dry-run`, since choosing writes nothing. `--explain` reads a line back without running even the selection ([terminal.md](terminal.md#explain)).

## Verbs

Verbs act on the selected rows. Each row states why it exists.

| Verb | Does | Why it is a verb | Defined in |
|---|---|---|---|
| *(none)* | list the rows | — | [query.md](query.md) |
| `read` | show the rows' content (sessions, pages, anything); `--follow` keeps printing what their sessions do; `--view trace` shows where something came from | rendering is not a query; every view, range and destination is an option of the one verb | [reading.md](reading.md), [pages.md](pages.md#reading), [session-control.md](session-control.md#following) |
| `pick` | choose rows in the picker; the verbs after it act on the chosen ones | a person's choice cannot be written as a query | [terminal.md](terminal.md#the-picker) |
| `brief` | packaged context: alone, for this directory; of a session, to take it over; of search words, what is already known; of a page, the page; of other rows, what matters in them | it packages under a budget, which no view does | [agent-context.md](agent-context.md#brief) |
| `note TEXT` | write text about each row, of a kind (`--as decision`, `finding`, `translation:pl`, `page`, `heading`, …); on a page selection it writes discussion about the page unless `--section` places it on the page or it is `--as heading` or `--as page` | a write | [annotations.md](annotations.md#note), [pages.md](pages.md#writing) |
| `tag NAME` | label each row (`NAME` or `NAME:VALUE`) | the most common write, with its own completion of tags in use | [annotations.md](annotations.md#tag) |
| `edit [TEXT]` | write a new revision of each annotation; with no text on a terminal, open them, or a page, in your editor and check in what you save | a write on annotations | [annotations.md](annotations.md#edit), [page-working-copy.md](page-working-copy.md#editing-in-your-editor) |
| `withdraw` | take each annotation back; nothing is deleted | a write on annotations | [annotations.md](annotations.md#withdraw) |
| `judge VERDICT` | accept, reject or defer each annotation; or record which of several rows is better, or their order | a write on annotations or rows | [judging.md](judging.md) |
| `save [NAME]` | name the question and keep this line's rows; put the rows on a page (`save page:NAME`, `--as FORM` for how they show); checkpoint a page; `--watch` tells you of changes | it acts on the question as well as the rows | [saved-queries.md](saved-queries.md#save), [pages.md](pages.md#writing) |
| `send TEXT` | send text to each session, web chats and member aliases included | an act that reaches a session | [session-control.md](session-control.md#send) |
| `answer VALUE` | respond to what a session is waiting for: `allow`, `deny` or quoted text | an act | [session-control.md](session-control.md#answer) |
| `interrupt` | stop a session's current turn | an act | [session-control.md](session-control.md#interrupt) |
| `resume` | reopen a session where it ran (`--fork` forks it); `--in HARNESS` continues its work in another harness | an act that starts a harness | [session-control.md](session-control.md#resume) |

Listing is the selection alone; the running list, the waiting list, notices and the sessions that ended open are shipped saved queries; a day is a query with a time condition; reports are queries with `| measure`; writing out is `read --to` (a file replaces its target only once complete; a new file gets the mode a new file normally gets under the caller's umask, and an existing file keeps its mode); citing is `read --format citation`; opening in the browser is `read --to browser`; attachment bytes are `attachment:MESSAGE#N read --to FILE`; the stored records behind a row are `read --view raw`, and where a quote, a paste or a file line came from is `read --view trace`; the table of contents of a page is `read --view contents`; pinning is `save page:pinned`. Derived work is the daemon's, so there are no build, rebuild or repair commands ([architecture.md](architecture.md#principles)).

## Commands

Commands are not about selected rows.

| Command | Form | Why it is a command |
|---|---|---|
| *(bare)* | `polylogue` | where things stand and what to type next ([terminal.md](terminal.md#the-orientation-screen)) |
| `help` | `help [VERB\|TOPIC] [--all] [--to browser\|FILE] [--format html\|markdown]` | teaches the words; local and generated ([terminal.md](terminal.md#help)) |
| `status` | `status [--line] [--vectors] [--verify [--manifest FILE]]` | health of the daemon and archive, not rows ([operations.md](operations.md#status)) |
| `start` | `start TEXT [--in HARNESS\|WEBCHAT \| --role NAME] [--cwd DIR] [--attach FILE]… [--model M] [--effort E] [--project P] [--submit]` | starts something that has no row yet ([session-control.md](session-control.md#start)) |
| `web` | `web` | opens the workbench ([web.md](web.md)) |
| `import` | `import PATH… [--origin NAME] [--withdraw-missing]` · `import PROVIDER [--since TIME] [--cancel] [--wait D]` | takes files or a provider, not rows ([capture.md](capture.md#import), [extension.md](extension.md#importing-a-provider)) |
| `set` | `set [KEY [VALUE]] [--machine NAME]` | lists, shows or changes a setting ([installing.md](installing.md#configuration)) |
| `embed` | `embed [--query 'QUERY'] [--yes]` | spending money is an explicit act ([search.md](search.md#semantic-ranking)) |
| `install` | `install TARGET [--remove] [--dry-run] [--instructions] [--extension-id ID] [--program NAME]` | wires Polylogue into a harness, Chrome, a shell or Google Drive ([installing.md](installing.md#installing)) |
| `demo` | `demo DIR` | builds the Morrow sample archive ([installing.md](installing.md#demo)) |
| `daemon` | `daemon [--until-idle] [--no-sources]` | the background service ([operations.md](operations.md#the-daemon)) |
| `mcp` | `mcp` | the MCP server over stdio ([agents.md](agents.md#mcp-server)) |
| `dev` | `dev sql STATEMENT` · `dev schema infer\|unread\|diff …` · `dev feed --after POSITION` | read-only SQL over the `v1_*` views ([query.md](query.md#read-only-sql)), parser maintenance ([archive.md](archive.md#schema-tools)), the change feed for programs ([operations.md](operations.md#the-change-feed)) |

### Discipline

A verb or command exists only when what it does cannot be said as a selection plus an existing verb, a view, an option, or a shipped saved query that completion offers. Each row states that reason, and `help --all` prints it.

Each first-level word takes a word away from plain-word search, so a command earns the first level only when a person types it as an act of its own. Explaining a line is the line option `--explain`, because it is a way of running any line. Fetching a web chat's history is `import PROVIDER`, because both bring outside conversations into the archive and only the channel differs. The deep storage check is `status --verify`, the deep form of "is my archive healthy". Shell completion and the Google Drive sign-in are `install` targets, because each wires Polylogue into something once. SQL, parser maintenance and the change feed sit under `dev`, because they are not everyday acts. Context for a directory, a session or a search is one `brief`, because the selection's form already says which. Pages add no verb or command group: every page act is the same verbs on a `page:` selection, so a group would be a second spelling of each. Keeping rows on a page is `save page:NAME`, because a page is one more place to keep rows; a checkpoint is `save` on the page, as freezing any selection is; the table of contents is a view; writing a page in your editor is `page:NAME edit`, because `edit` takes its text or nothing as `save` takes its name or nothing; and agents and scripts take a page's working copy out with `read --to` and bring it back with `import`, the way files already leave and enter.

## Option names

One name per concept, on every verb and command that has the concept.

| Concept | Name | Where |
|---|---|---|
| rows per page | `--page-size N` | the listing (`\| limit` is a transform) |
| tokens per page | `--budget N\|none` | `read`, `brief`, the listing |
| continue a page | `--cursor C` | every paged reply |
| how long to wait | `--deadline MS\|none` | every line |
| act on every row of a query | `--all` | lines with a write or act |
| show what would happen | `--dry-run` | every line; `install` |
| how a line is read | `--explain` | every line and command |
| where the CLI writes | `--to stdout\|pager\|FILE\|DIR\|clipboard\|browser` | the listing, `read`, `brief`, `help` |
| document encoding | `--format F` | `read`, `help` |
| the reply as JSON | `--json` | everything |
| keep printing as it happens | `--follow`, with `--until-idle` | `read` |
| who will continue | `--for SESSION` | `brief` |
| directory | `--cwd DIR` | `brief`, `start`, `resume --in` |
| where a session runs | `--in HARNESS\|WEBCHAT` | `start`, `resume` |
| the kind of a note | `--as KIND` | `note` |
| how rows show on a page | `--as FORM` | `save page:NAME` |
| how a page enters briefs | `--brief off\|contents\|full` | `note --as page` |
| the dimension of a comparison | `--on DIMENSION` | `judge` |
| a reason | `--reason TEXT` | `withdraw`, `judge`, `answer` |
| input structure | `--origin NAME` | `import` of files |
| a feed position | `--after POSITION` | `dev feed` |
| changed since (time) | `--since TIME` | `import` of a provider, `dev schema infer` |
| how long a watched row must match | `--hold D` | `save --watch` |
| deep storage check | `--verify` | `status` |
| time | `since:` and `until:` in the query | every query |
| why rows matched or not | `--why` | the listing |
| columns | `--fields a,b` | the listing |
| another rendering | `--alternate KIND\|none` | `read` |
| a relation to another record | `--replies-to`, `--supports`, `--challenges`, `--qualifies`, `--corrects`, `--derived-from`, `--blocks`, `--evidence`, `--scope` | `note` |
| the session alone | `--physical` | `read` |
| labels for excerpts | `--neutral-names` | `read` |
| as it stood | `--at TIME\|POSITION` | `read` |
| where this reader stopped | `--from seen`, and `since:seen` in queries | `read` |
| other compare alignments | `--align prompt\|lineage\|time\|file` | `read --view compare` |
| wait for rows or an import | `--wait D` | every listing; `import PROVIDER` |
| a page's place | `--page NAME`, `--section S`, `--after SLOT`, `--level N` | `note`, `save`, `send` |
| a page's shape | `--template T`, `--param KEY=VALUE`, `--layout linear\|board\|table` | `note --as page`; `--template` also on `save page:NAME` |
| slot modes | `--fixed`, `--live`, `--kept` | `save page:NAME` |
| the subject or topic | `--about REF` | `note --as page`, `send` |
| check-in deletions | `--withdraw-missing` | `import` of a page's working copy |
| reopen differently | `--fork`, `--anyway` | `resume` |
| send a new web chat | `--submit` | `start --in WEBCHAT` |
| scope a setting | `--machine NAME` | `set` |
| also the instructions | `--instructions` | `install` |
| rename in the provider | `--rename` | `note --as title` on a web-chat session |

`--since TIME` is named like the query's `since:` field and means the same thing: a time. `--json` is the typed reply itself; `--format json` is the bundle encoding of a reading ([reading.md](reading.md#formats-and-destinations)).

## Exit codes

The exit and HTTP table is the common `command-surface`'s, shared with Sinex; each failure kind maps to exactly one code. A line exits with its failed step's code.

| Code | Polylogue kinds |
|---|---|
| 0 | `ok` (rows, a measure, a completed step list) |
| 1 | `empty`: the read succeeded and matched nothing |
| 2 | invalid input: `invalid_query`, `unknown_field`, `bad_value`, `bad_request`, `not_a_verb`, `too_many_words`, `not_a_query`, `not_yet_available`, `ambiguous`, `needs_all` (a write or act over several query rows without `--all`), `row_numbers_off_terminal`, `stale_cursor`, `not_built` |
| 3 | `refused`, with the operation's stated reason (an interrupt of a session Polylogue does not host, a blind judgment on a reference) |
| 4 | `not_found`, `conflict` (a stale base, a late answer, an archive lock another process holds), `io` (unreadable stored data as `store`, a corrupt blob as `blob_corrupt`), `internal` (a state no request can cause: a Polylogue bug, reported with the request ID) |
| 5 | `degraded`: rows printed but part of what was needed missing, an operation that stopped early, or a write pending in primary mode |
| 6 | `not_running`: the daemon is down, or it replied in a shape this build does not read (`daemon_protocol_mismatch`: a daemon from another build; restart it) |
| 7 | `timeout` (a read deadline, or a database busy past its wait as `busy`), `cancelled` |

Each failure has one code, and each code one kind. `retryable` is true when the same request can succeed later unchanged: the condition passes by itself (a busy database, an index replaced or being built, a deadline under load) or the daemon was stopping or the request was cancelled. A store failure is classified by what happened: a stopped writer or closed read pool is `not_running` and retryable, a held archive lock is `conflict` and retryable, a stored format this build does not read is `refused`, an index being built is `not_yet_available`, a corrupt blob is `io` with code `blob_corrupt`, stored data this build cannot read back is `io` with code `store`, and an unreachable Sinex is `sinex_unavailable`. `bad_request` and the other exit-2 kinds mean the caller can fix the request by changing it. A refusal the caller can correct carries a typed reason: one of a closed set declared with the failure codes, each with its name, its code, the fields it names and one sentence, which is the failure's message. Callers compare the reason's name and fields, not the sentence; the browser host's error answers carry it as `reason` (`{name, fields}`). Every `stale_cursor` failure suggests the same read without its cursor.

With `--json` the reply has `outcome: "error"` and an `error` object (`kind`, `message`, `position`, `suggestion`, `hint`, `candidates`, `retryable`) on standard output; over MCP the suggestion is a ready call in `next`. A failure before any command runs leaves by the same table: an unreadable configuration file or working directory is `io` (4), a configuration value the binary cannot take is `bad_value` (2), and arguments it cannot read (not UTF-8, a deployment option without its value) are `bad_request` (2). The daemon process also uses exit 1 for a run that failed or a held lock, 74 for a storage sync failure, and 78 when it refuses an archive whose stored format it does not read ([operations.md](operations.md#the-daemon)); any other failure to start serving exits by the table, 4 for an archive directory it cannot create or open. Hook processes exit 0 on failure so that they never block their harness. Help and version output exit 0, and a resumed harness's exit is its own.

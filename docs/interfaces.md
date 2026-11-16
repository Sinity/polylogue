# Interfaces

Polylogue has one table of operations, and every surface is generated from it: the command line, the daemon's socket, HTTP and MCP. This page owns that table as Polylogue uses it, the command-line grammar and the rule that keeps the command set small, latency classes and Polylogue's budgets, the daemon's states as the command line sees them, warm state, and Polylogue's failure kinds and exit codes. [Completion](completion.md) owns completion and its catalog file; the [terminal interface](terminal.md) owns the picker, row addressing (`@N`, `@last`), footers and affordances, output and help.

The generic mechanics (the `Operation` trait, the failure-to-exit-and-HTTP table, request and answer envelopes, deadline classes, SSE framing) are the common [`command-surface`](https://github.com/Sinity/polylogue-sinex-common/blob/master/crates/command-surface/README.md) crate's. The query language is in [query.md](query.md); what a caller sends about itself and how its author is computed is in [assertions.md](assertions.md#authorship); writing out with `read --to` is in [reading.md](reading.md#writing-out); MCP as the agents' channel is in [agents.md](agents.md#mcp-server); the web is in [web.md](web.md).

## Command table

- **One typed operation table** drives every surface. Each row declares the operation's arguments, latency class, failure kinds, one-sentence help, its reason for existing ([below](#discipline)) and one to four examples on the Morrow archive, and names the handler that answers it. The handler carries the request and answer types, so the row is declared once with its handler and an operation without one cannot be written. The CLI, socket, HTTP and MCP are generated from it; a new operation is a row and its handler, not code on each surface.
- **One request format.** The socket, HTTP and MCP carry the shared request: the operation, its typed arguments, the request ID, the deadline and the options beside them. The hooks and the browser extension's host send the same requests, for rows people do not type (`delivery-event`, `mail-claim`, `permission`, `chat claim`, `chat report`, `chat decide`, `chat attachment`, `backfill claim`, `backfill report`, `conversations`, `stored-files`, `selection-note`); they are left out of help. `changes` delivers a finite batch of change events; `follow` keeps streaming until stopped.
- **CLI values have one parser.** The shared `command-surface` argument declarations supply value kinds, required fields and choices for each typed operation, including local commands and streams. Its `parse_cli` constructs the request envelope, JSON output selection and calling-process options. Clap presents those same arguments, preserves command placement, help, shell completion and lexical errors, and owns OS-valued startup paths with configuration. Domain catalogs keep the choices of named measures, explain subjects and hosted harnesses. File preparation, stream framing and daemon peer negotiation remain product behavior. Local commands and calling-process options do not become daemon operations or request fields.
- **Archive selection preserves path aliases.** An explicit archive path is made absolute without expanding its symlinks; canonical archive identity is resolved separately. CLI adapters and the Morrow tour preserve declared aliases so a short Unix-socket path stays usable.
- **Ready calls are complete requests.** The CLI form of a row is its command words, with `--request JSON` for a complete argument object (not mixed with operation flags), `--request-id ID` and `--deadline MS|none`; `find` is the explicit query form. MCP groups the rows into five tools ([agents.md](agents.md#mcp-server)), and a call takes `{operation, arguments, request_id, deadline}`, `operation` being the row's command name. HTTP takes `{operation, arguments, request_id, deadline}` on the row's route; GET encodes `arguments` and `deadline` as JSON query values. Named context operations also take optional envelope `budget_tokens` (`--budget N`), as described in [Budgets](agent-context.md#budgets). MCP tool-family selection and envelope decoding use the same shared declaration parser as the Common MCP transport; the product proxy keeps authenticated grants, parent transport deadlines, typed row checks and rendering. Every surface checks the same typed arguments. On the read route, `Accept: text/html` renders the same typed page and its ready calls.
- **HTTP GET has one argument parser for finite answers and feeds.** The shared declarations decode both plain query pairs and complete generated requests, keeping request identity, deadlines and argument JSON distinct. Duplicate scalar or envelope fields are refused. The `changes` feed resumes with its `since` argument, not a page cursor; feeds refuse token budgets and unsupported request options on CLI, socket and HTTP before production. `follow` keeps its socket exposure. A completed feed position is sent only with the final completion event.
- **Finite transfers use the shared server-sent-event framing** on the socket and HTTP: a begin event, record events and a completion event from one read. A failure before HTTP headers uses the shared failure status; a later failure ends with an error event and no completion. A client writes a directory bundle into place, and `changes` advances its saved position, only after the completion event.
- **Access belongs to the request.** The declared operation determines access from its typed arguments. MCP tool-family hints describe the family; the same operation and arguments have the same access on every surface.
- **Answers are typed** and rendered from their types ([reading.md](reading.md)); each has one terminal printer and one JSON form, and no surface has features the others cannot reach. The handler keeps its declared result type through execution and deadline handling. JSON serialization belongs to the transport or local wire boundary; the daemon does not serialize an answer and decode it back to its declared type.
- **The daemon is required.** The command line is a thin client: it never opens archive files, and the daemon is the only writer.
- **No refusal depends on who the caller is.** No act is refused because the caller is an agent or has no terminal; the only terminal-dependent behaviours are the picker-based ones, and each names its non-interactive form ([Picker](terminal.md#picker)).
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
3. **An unsignalled first word is refused,** exit 2, within the local budget, with the `find` form and nothing else; command names are not guessed.
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

A command or verb exists only when what it does cannot be expressed as a query plus an existing verb, or as a shipped saved query that completion offers. Each row states that reason, and `explain capabilities` prints it.

So listing is the bare query; the running list, notices and resume candidates are the shipped saved queries `query:running`, `query:notices` and `query:resume_candidates`; a day is a query with the day's time condition; reports are the `| measure` and `| agg` stages; writing out is `read --to`; citing is `read --format citation`; opening in the browser is `read --to browser`; attachment bytes are `block:<message>#<n> read --to FILE`; the stored input behind a row is `trace`. Derived work is the daemon's, so there are no build, rebuild or repair commands ([architecture.md](architecture.md#principles)).

### Verbs

Verbs act on a query's rows.

| Verb | Does | Defined in |
|---|---|---|
| `read` | render rows; with `--to`, write them out | [reading.md](reading.md) |
| `follow` | one line per new row or live-state change until interrupted | [reading.md](reading.md), [agents.md](agents.md#live-sessions) |
| `trace` | the source chain of a reference back to its stored input, its bytes and its reading version | [reading.md](reading.md) |
| `assert KIND [KEY] [VALUE]` | write one assertion of KIND on each row | [assertions.md](assertions.md) |
| `revise`, `withdraw` | a new revision of each assertion row (refused on a stale base), or a withdrawal | [assertions.md](assertions.md#current-revisions) |
| `judge` | on a terminal, one-key verdicts over the rows in the picker (`--blind` hides author, model and origin, shuffles, and records that on the judgment); off a terminal, exit 2 naming `judge accept\|reject\|defer` with explicit rows | [assertions.md](assertions.md#judgment) |
| `keep` | store the result as `result:<id>` | [query.md](query.md#query-objects) |
| `save NAME` | label the query's hash with a name | [query.md](query.md#query-objects) |
| `watch` | write a watch rule over the query | [query.md](query.md#watches) |
| `pack [NAME]` | build a pack under a budget and record a delivery | [agent-context.md#packs](agent-context.md#packs) |
| `ping TEXT [--live]`, `answer approve\|deny [--standing]`, `interrupt`, `continue` | address a session, or act on a live one through its harness; `send` is `ping --live` | [session-control.md#live-session-control](session-control.md#live-session-control) |
| `pick` | open the picker over the rows and print the chosen references, one per line | [below](terminal.md#picker) |
| `compare DIMENSION` | a comparison of the matched references | [assertions.md](assertions.md#judgment) |
| `variant` | write a variant of a reference | [assertions.md](assertions.md#variants) |
| `add REF…`, `move POS`, `toc`, `checkout DIR`, `checkin DIR` | page operations | [pages.md](pages.md) |
| `applies ENV` | whether a method's conditions hold in an environment | [assertions.md](assertions.md#memory) |

### Commands

Commands are not about a query's rows.

| Command | Reason |
|---|---|
| `status [--line]` | health of the daemon and archive, not a query unit ([operations.md](operations.md#status)) |
| `explain SUBJECT [--expression QUERY] [--unit UNIT]` | describes the language or how a query is read, without running it |
| `import PATH…` | takes files, not rows ([capture.md](capture.md#import)) |
| `setting [KEY [VALUE]]` | lists the settings catalog, reads one key or writes one key |
| `sql 'SELECT …'` | read-only SQL through the daemon ([query.md](query.md#read-only-sql)) |
| `changes --since POS` | a position-based feed for programs |
| `next` | the following page of the last list here ([Addressing](terminal.md#addressing)) |
| `web` | opens the workbench at a view that is not a reference |
| `auth drive` | a credential flow, not rows |
| `daemon [--until-idle] [--no-sources] [--json]`, `mcp`, `hook EVENT`, `hooks print\|status`, `install-agent HARNESS [--remove]`, `browser-host`, `completions SHELL`, `demo` | local process entry points and installers ([operations.md](operations.md#installing-into-harnesses)) |
| `schema infer\|diff\|unread` | infer and unread inspect stored inputs through the daemon; diff compares explicit schema files locally ([archive.md](archive.md#format-drift)) |
| `verify [--manifest FILE]` | the on-demand read-only check ([operations.md](operations.md#verification)) |
| `sinex cutover` | the one mode transition ([primary-mode.md](primary-mode.md#cutover)) |

### Available operations

The operation grammar above defines the intended interface. The current
binary's `--help`, per-command help and `explain capabilities --json`,
together with generated MCP schemas, describe the operations it provides.

## Latency classes

Each table row declares a latency class, and the class applies on every surface (CLI, MCP, HTTP, web). Budgets are end to end (process start to the last byte of the first screen, through the real binary and socket), with a warm daemon and a real-shaped archive under typical load, at p95. The deadline each request carries comes from its class, in the `command-surface` request envelope.

An MCP `_meta.deadline_ms` absolute bound travels separately from the unchanged shared request as the transport's own `parent_deadline_ms`; the daemon intersects it with the operation lifetime. An HTTP `x-polylogue-deadline-ms` header bounds the whole request; a stream ends at the earlier of that bound and its operation's deadline, and without the header a `null` deadline has no limit. A request already past its bound starts no new write; a write the writer has accepted keeps its effects and acknowledgment.

| Class | Members | Budget (p95) | Deadline sent |
|---|---|---|---|
| **local** | `--help`, `--version`, `completions`, grammar-only completion, `explain` of the grammar, refusals of malformed or unsignalled input | ≤ 15 ms | none (no socket) |
| **status** | bare `polylogue`, `status`, `status --line`, `pack` with nothing to include | ≤ 100 ms; `status --line` ≤ 30 ms, because it runs in harness status bars | 80 ms |
| **completion** | Tab in any shell, MCP `completion/complete`, the web query box | ≤ 50 ms per keystroke, daemon up or down | none for values (catalog file); 30 ms for the count preview |
| **read** | `REF read`, `@N read`, one-row references, indexed lists (`origin:`, `repo:`, `since:`, `tag:` …), `resume`, `pack takeover`, `next` | ≤ 100 ms to the first screen | 250 ms |
| **search** | word and phrase search, `pack recall`, `similar:` with a stored vector | ≤ 250 ms first page | 2,000 ms |
| **aggregate** | `group by`, `| measure`, `| agg`, facet counts, `explain QUERY` with clause counts | ≤ 400 ms | 400 ms |

Help answers in about 100 ms and queries in under a second. A missed budget is a defect in the code, not a reason to relax the number. Budgets are measured as paired benchmark runs ([testing.md](testing.md#benchmarks)); no test asserts elapsed time.

The read-class `advisory` operation has a stricter preparation cap of 20 ms, intersected with its caller’s bound on every surface. Hooks send a 20 ms request and emit nothing on failure; `--deadline none` does not lift this optional-advisory cap. The accepted request envelope and identity remain unchanged.

Pure writes, including imports, await their committed result without a read deadline. Mixed operations keep deadlines for their read preparation and end that phase when they hand the mutation to the writer: a preparation that has run out refuses the handoff, and a handed-over write answers its committed result however long the commit takes, never a timeout. Classification follows the accepted arguments, so reading a setting remains bounded while writing it keeps its result. The `x-polylogue-deadline-ms` header above still applies.

**Deadlines are interactive defaults.** A read whose row declares no latency class, called with the default deadline, stops after 30 seconds (`DEFAULT_READ_TIMEOUT_MS`); `--deadline none` lifts it. The CLI's socket timeout is the deadline plus 50 ms, so a hung daemon never hangs a shell. Any caller lifts a deadline (`--deadline MS`, `--deadline none`; `deadline: null` over MCP and HTTP); deadline behaviour and cancellation are in [query.md](query.md#deadlines). A timed-out aggregate prints the narrower query the planner can serve and the same query with the deadline lifted:

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
| busy with a first build or re-parse | interactive answers keep their budgets on the reserved executor place and read connection; answers over data not yet indexed say `up_to_date: false` on stderr |
| starting (lock held, socket not bound) | waits up to the class deadline, printing "polylogue: waiting for the daemon to finish opening the archive" once on stderr, then exit 7 if the deadline passes |
| down | the local class and completion (from the catalog file) work; everything else exits 6 with one line naming the archive and `polylogue daemon`. Nothing starts a daemon implicitly |
| hung | the socket timeout ends the command, exit 7 |

Because one daemon assembly answers every transport, the same question gives identical rows, totals and refusals through CLI `--json`, the socket, HTTP and MCP.

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

## Exit codes

The exit and HTTP table is the common `command-surface`'s, shared with Sinex; each failure kind maps to exactly one code. Polylogue's kinds:

| Code | Polylogue kinds |
|---|---|
| 0 | `ok` (rows, or a measure) |
| 1 | `empty`: the read succeeded and matched nothing |
| 2 | `invalid_query`, `unknown_field`, `bad_value`, `not_a_query`, `bad_request`, `ambiguous`, `stale_cursor`, `not_built` |
| 3 | `refused`, with the operation's stated reason |
| 4 | `not_found`, `conflict`, `io` |
| 5 | `degraded`: an `incomplete` answer (rows printed, part of what was needed missing), an operation that stopped early, or a write pending in primary mode |
| 6 | `not_running`: the daemon is down, or it answered in a shape this build does not read (`daemon_protocol_mismatch`: a daemon from another build; restart it) |
| 7 | `timeout`, `cancelled` |

The daemon process also uses exit 1 for a failure or a held lock, 74 for a storage sync failure, and 78 when it refuses an archive whose stored format it does not read ([operations.md#the-daemon](operations.md#the-daemon)). Hook processes exit 0 on failure so that they do not block their harness. Clap uses exit 2 for argument errors and 0 for help and version output.

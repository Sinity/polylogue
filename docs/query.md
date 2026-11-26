# Query

Polylogue asks every question through one query language: listings, reports, the searches inside briefs, watches and facets are all queries, and every surface replies to the same query the same way. This page owns selections, Polylogue's catalog and compiler (units, fields, defaults, times, transforms), paging, deadlines, replies, facets and read-only SQL.

The grammar, its spelling, the AST's three-valued semantics, time expressions, the logical day and the cursor type are shared with Sinex and owned by the common [`query-language`](https://github.com/Sinity/polylogue-sinex-common/blob/master/crates/query-language/README.md) crate, whose tests run against Polylogue's catalog. How a line is read, verbs, chains and exit codes are [interfaces.md](interfaces.md); questions, runs, saved queries, kept results and watches are [saved-queries.md](saved-queries.md); text and semantic search are [search.md](search.md); references are [references.md](references.md); measures are [measures.md](measures.md); queries that mix Polylogue and Sinex data are [primary-mode.md](primary-mode.md#cross-system-queries).

## Selections

A selection is the first argument of a line and says which rows the verbs after it act on. It is always one shell argument; which form a first word is, and in what order the forms are tried, is [interfaces.md](interfaces.md#the-first-word). A selection is one of:

- **a quoted query** in the language on this page, transforms included: `polylogue 'repo:polylogue since:7d'`, `polylogue 'tool_calls where outcome:error | group by tool | count'`;
- **a reference** ([references.md](references.md#forms)): `polylogue codex:019d0f2a read`, a bare ID prefix, a pasted provider URL, `annotation:<id>`, `query:NAME`, `result:<id>`, `run:<id>`, `role:NAME`;
- **a reserved word**: `latest`, `self` or `here` ([references.md](references.md#reserved-words));
- **a row number or range** on a terminal, such as `@3`, `@2-5` or `@all` ([terminal.md](terminal.md#row-numbers));
- **a saved query name** as the first word, your own first, then the shipped ones: `polylogue waiting` runs `query:waiting` ([saved-queries.md](saved-queries.md#shipped-saved-queries));
- **a unit name** as the first word: `polylogue tool_calls` lists every tool call, with the unit's defaults;
- **a single plain word**, which searches for that word in prose and tool input and output, ranked: `polylogue chrono` is the search for "chrono", and its footer says `searched the word "chrono"` ([search.md](search.md#the-plain-word));
- **`-`**: references read from standard input, one per line, so `grep`, `fzf` and `xargs` compose (`polylogue 'tool_calls where outcome:error' --fields id`, cut with `head`, then piped into `polylogue - read`).

A reference alone drops the query defaults, because it names exactly that thing; how short a reference may be typed and how replies print it is [references.md](references.md#forms).

## Principles

- **One expressive language instead of small features.** No surface re-implements filtering, totals or paging; a new need is met by the language or a transform, not by a command.
- **Unknown is not zero; scopes are strict.** An unknown value matches neither a condition nor its `NOT`. A query never widens itself, and never falls back to searching its own text.
- **No arbitrary limits.** Page sizes are defaults, deadlines can be lifted by any caller, and nothing truncates a valid reply.
- **Discoverable without shrinking the language.** `--explain`, `read_as`, completion and runnable corrections teach the language; every string an agent might pass returns rows, a refusal with a correction, or a typed error.
- **Agents are callers like the operator,** with the same powers, deadlines and replies.

## Grammar

Polylogue uses the shared grammar. The shapes it accepts:

```
[UNIT [where CONDITIONS]] [with ATTACHED] [| TRANSFORM]...    tool_calls where outcome:error | group by session | count
CONDITIONS [with ATTACHED] [| TRANSFORM]...                   origin:codex since:7d with messages(id,role)[last:1]
SESSION-CONDITIONS | UNIT [where CONDITIONS] [| TRANSFORM]... tag:project:morrow | tool_calls where action:shell
```

- Conditions side by side are AND; parentheses group everywhere; without a unit, the unit is `sessions`.
- **Atoms:** `field:value`; `field OP scalar`; `field between A and B`; `field ~ "text"` (contains); a text term (bare word or `"phrase"`, `word*` prefix, `word~` approximate, see [search.md](search.md)); `exists UNIT(…)`; `count(UNIT [where …]) OP N`; `FIELD in (QUERY)`; `in:result:<id>`; `seq(step -> [next|within D] step …)`; `query:<hash>` or `query:NAME[(p=v,…)]`; `similar:TEXT|REF`; a reference.
- **Values:** `word`, `"quoted"` with backslash escapes, `(a|b|c)`, `*` (names the field without restricting it and drops its default), `unknown`.
- **Comparisons:** `> >= < <= = !=`; durations `ms s m h d w`; money `2.90` or `2.90usd`. An amount is read exactly from its decimal spelling into signed 64-bit nano-USD, half ties rounded away from zero; an amount outside that range is refused, never saturated. There is one spelling: the compact `field:>=N` is refused with `field >= N` as its suggestion.
- **Depth:** nesting has no limit. Parsing, binding, compiling and evaluation grow their stack as a query nests (`query_language::depth::deeper`), so a deep query is answered or refused for what it says, never for how deep it is. Two negations cancel, and a set inside a set adds its values. Saved-query expansion refuses a cycle and names the chain.

### Errors

Every query error carries `kind`, `message`, `position` (byte offset, shown with a caret), `suggestion` (a corrected query that runs as given, whenever one can be formed), `hint`, `candidates` (the matches of an ambiguous reference, or the known fields close to an unknown one) and `retryable`. Polylogue's error kinds and their exit codes are [interfaces.md](interfaces.md#exit-codes). An error found inside a saved query's body is reported at the position of its name, and its message names the position in the saved query's text. A unit or field that is part of the design but has no producer yet is refused `not_built`, never an empty result.

### References inside queries

A reference names exactly one thing and drops the reachability defaults (subagent, hidden, duplicate, active path) for that match. A query that is only an exact annotation revision (`annotation:01JB7Q2M@2`) reads the `annotations` unit, since the revision is one row; every other lone reference is a condition on the default unit. An ID-shaped word or unique prefix resolves to exactly one session or message, or is `not_found` or `ambiguous` naming the matches; it never silently becomes search text. A field condition on a session ID (`id:`, `parent:`, `root:` and the other session-reference fields) names at most one session: a prefix two share is `ambiguous`, and an ID no session has is a condition no row meets, so `sessions where id:X` answers `empty` (exit 1), its negation keeps every session, and a waiting listing on it keeps waiting until the session arrives. `self` is the calling session; `latest` is the most recently active main session that also meets the query's other session conditions (`repo:x latest` is the newest session in `x`), never the calling session; both resolve inside the query when it runs, so a saved query using them follows new sessions. Row numbers never reach a query: the command line resolves them to references first ([terminal.md](terminal.md#row-numbers)).

## Units and fields

Each unit declares, once, its rows, key and identity, default filters, default order, search behaviour and relations (`session.`, `message.`, `call.`, `child.`, `next.`, `parent.`, `recipient.`). Each field declares its kind (match, substring, words, subtree, number, money, time, duration, bool, ref), its closed value set taken from the model's stored names (a condition on it compares the stored value, so the column's indexes serve it), its unknown word, a one-line help and the phrase `read_as` uses. Entity-reference fields declare their target independently of their storage unit. A raw provider ID or mixed-reference value has no fixed entity target. An ID list holds its list shape and declares only its members' target. `help units` and `help fields [UNIT]` page the declarations.

| Unit | Rows |
|---|---|
| `sessions` | logical sessions by default ([sessions.md](sessions.md#logical-sessions)); `physical:true` lists physical sessions |
| `messages`, `blocks`, `tool_calls`, `documents`, `attachments`, `events`, `session_links`, `turns`, `files` | as parsed ([sessions.md](sessions.md)); `session_links` are the links between sessions ([sessions.md](sessions.md#session-links)) |
| `pastes` | paste spans: content hash, source of the span, state, message and session; `group by hash` gives repeats |
| `annotations` | annotations ([annotations.md](annotations.md)) |
| `alternates` | alternates and their parts ([alternates.md](alternates.md)) |
| `deliveries` | deliveries ([agent-context.md](agent-context.md#deliveries)) |
| `feed` | the change feed as rows ([operations.md](operations.md#the-change-feed)) |
| `pages`, `slots` | pages and the items placed on them ([pages.md](pages.md)) |
| `queries`, `runs`, `results` | questions and saved queries, runs and kept results ([saved-queries.md](saved-queries.md#query-objects)) |
| `import_jobs`, `chat_actions` | what the browser extension was asked to do: provider imports and turns drafted or sent in web chats ([extension.md](extension.md)) |
| `prices` | the price catalog's rows: model, effective date, rates and source ([measures.md](measures.md#the-price-catalog)) |
| `live_intervals` | the intervals of each session's activity, wait and stall, with the event that closed each ([live-sessions.md](live-sessions.md)) |
| `processes` | harness processes |
| `commits` | commit and pull-request links |

A unit whose rows have no producer yet (`pages`, `slots`, `prices`, `commits`, `pastes`, `attachments`) is declared with its columns and answers `not_built`.

`import_jobs` (id, provider, since, state, listed, captured, unchanged, failed, position, detail, created_at, updated_at, claimed_by) and `chat_actions` (id, provider, conversation, model, text, kind `draft` or `submit`, state, phase, needs_approval, created_at, updated_at, claimed_by, detail) read the archive's job and action rows, newest first. A row's `state` is computed from its events at the query's `now`, by the same rule the writer applies: a claim held past its lifetime (3 minutes without a report) reads as `waiting` for an import job and `queued` for a chat action, or `outcome_unknown` once that action may have been sent. `claimed_by` names the extension instance holding an unexpired claim; `detail` is the latest event's, and `position` the latest reported list position: `polylogue 'import_jobs where state:running'`, `polylogue 'chat_actions where state:awaiting_approval'`.

Raw tool-call failures are queryable as recorded outcomes: `tool_calls where outcome:error`. These rows report tool results without classifying a command as a verification check.

Field rules:

- **Session-field shorthand.** In a row query, a session field without `session.` means `session.<field>` when the row unit has no field of that name; `--explain` shows the full form.
- **String matching** is case-insensitive. `title`, `path`, `command` and `input` match substrings; `output` and `thinking` match words; `cwd` and `checkout` match subtrees (whole path components, case-sensitive, `_` and `%` literal); everything else matches whole values.
- **Two models in one session** match either; `group by model` counts the session under each.
- **Counts and content fields** include `project`, `has:code` and `code_lang`, per-role counts (`human_words`, `model_words`, `thinking_messages`), `since_session:<ref>`, and `time_quality` and `precision` on every time.
- **Live and open fields on sessions:** `running`, `waiting`, `live_rank`, `live_since` and `last_event_at` ([session-control.md](session-control.md)); `ends_open` (an unanswered prompt, an unfinished tool call, an open todo or handoff, a partial outcome); `triage_score` ([saved-queries.md](saved-queries.md#shipped-saved-queries)).
- **Triage contributions:** `unrecovered_failures`, `failure_streak`, `outlier`, `corrections` and `open_items` expose the facts summed by `triage_score`; `unrecovered_failures` counts commands whose latest run failed, a command being recovered only by a later run of it that passed. `corrections` and `open_items` count annotations about the session or anything inside it: a message, a block, a tool call, or a span of a message or block. The outlier comparison uses the same repository's preceding 30 days, excluding future sessions. A `triaged` tag suppresses the score only while it is newer than the session's activity.
- **Cost fields:** every assistant message carries its cost and its tokens per token kind; sessions carry `cost` (their own responses) and `cost_with_parents` (what the session shows, inherited responses included) ([measures.md](measures.md#cost-and-usage)). A day's usage or one model's responses are queries over these: `polylogue 'messages where role:assistant model:gpt-5 since:today'`.
- **`month_day`** on every timed unit (`11-09`) selects one logical day of the year across years, using the request's day zone and day start: `polylogue 'sessions where month_day:11-09'`. A sequence number in `feed` is a position, not a date.
- **Annotation fields**, on `annotations` and as conditions on any unit through `kv:`: `kind`, `key`, `value`, `author` (`operator`, `unattributed`, a session ID), `agent`, `via`, `about`, `evidence`, `scope`, `origin`, `current`, `conflict`, `withdrawn`, `judged`, `judged_by`, `expired`, `changed_since`, `waiting`, `pass`, `rubric`, `on` (a comparison's dimension), `answered` (a question with a current reply), `open` (a question, goal or problem whose work is open, or anything blocked; [pages.md](pages.md#open-items-and-the-decision-box)), `request` (the request that wrote it, for the undo line of a many-row write), `root` (its first revision, so `root:X withdrawn:*` is a note's history), `links_to` (what links to a reference) and `grounding` (`cited`, `changed`, `uncited` or `none`; [page-text.md](page-text.md#grounding)). On any unit: `tag:<name>[:<value>]`, `title:`, `kv:<kind>.<key>=<value>`, `hidden:`, `referenced_by:<ref>`. Grain is strict: an annotation about a session never matches its messages.
- **Page fields** on `pages`: `name`, `title`, `layout`, `template`, `about` (the subject), `scope`, `brief`, `shows` (a reference a slot shows), `slots` and `sessions` (counts), `open_items`, `last_change`, `linked_from` (how many pages place, include or link to it; `linked_from:0` finds orphaned pages), `citations`, `changed_sources` (citations whose source changed, went missing or was withdrawn), `uncited` (paragraphs that need a citation) and `proposals` (text awaiting the operator's judgment; [pages.md](pages.md#authorship-and-proposals)). On `slots`: `page`, `parent` (the heading it sits under), `pos`, `mode` (`current` or `fixed`), `form`, `shows`, `shows_kind` (the kind of target), `heading`, `is_query`, the live or kept query, `target_state` and `grounding`. `polylogue 'pages where shows:codex:019d0f2a'` lists the pages that place a session; `polylogue 'pages where "zone change"'` matches page titles and the text of the annotations a page shows ([pages.md](pages.md#finding-pages)).
- **Where a row is placed, linked or cited.** On every unit, the field `pages` lists the pages that place a row, `cited_by` the citations of the row or of something inside it (a citation of a span counts on its message and, rolled up, its session), and `group by page` counts a row under each page. `referenced_by:page:X` matches the rows page X places, at the unit's grain: `polylogue 'messages where referenced_by:page:flaky-ci'` gives the messages its slots show. Nested pages are followed and cycles cut; a live query slot places its query, not the rows it shows. Every link of an annotation to a target, with its role, is a row of the designed unit `links` (columns `record`, `role`, `target`, `target_kind`, `target_state` and `page`; roles `about`, `evidence`, `mentions`, `cites`, `embeds`, `slot_of`, …): `polylogue 'links where target:codex:019d0f2a role:cites'`, `polylogue 'links where role:cites target_state:changed'`.
- **`applies:REF`** on method annotations is computed: `applicable`, `not_applicable` or `unassessed` for the scope REF names ([annotations.md](annotations.md#memory)).
- **`since:seen`** matches rows changed since this reader's seen mark on them ([reading.md](reading.md#ranges)).
- **Query fields** on `queries`: `name` (set on saved queries) and `saved_at`.
- **Alternate fields:** `alternate:true` admits alternate text into text search; `lang:<code>` matches alternate parts in that language and messages by their derived `lang`.
- **Names shared with Sinex** retain their meaning per product: `seen_by:` is Polylogue's as-of field, and Sinex's "as of" is a different concept. Polylogue does not store historical query membership at earlier read positions: `| as of POSITION|TIME` replies `not_yet_available`. `seen_by:` selects records received by then (annotations by their authored `created_at`, [annotations.md](annotations.md#reading)), read with current parsers and prices.

Readings that are queries rather than views, and the rule that a row of a word search is read from its first hit, are listed in [reading.md](reading.md#views).

## Defaults

Each unit binds its default filters once, when its source is bound (a unit, `query:…` or `result:<id>`). A default is dropped only by a top-level AND condition on its field, or by `F:*`; a condition under OR, NOT, `exists`, `seq` or a saved query never drops one. After binding, everything further only narrows. A saved relation binds the unit's defaults by its own top-level conditions wherever it stands. As the source, or as a top-level condition, it stands in for the unit's defaults, so the caller's conditions only narrow it: `query:x hidden:*` cannot show a session `x` hid. A field whose value is unknown, such as `active` for a message whose path is unknown, matches neither `active:true` nor `active:false`; `active:unknown` names it.

| Unit | Default |
|---|---|
| `sessions` | logical sessions: continuations, subagents and forks fold into their root, and measures include the whole logical session; not stubs. A top-level `physical:true`, `kind:`, `id:` or structure field, or a reference, lifts it |
| listings and search over every unit with a session | not `hidden`; measures and population counts include hidden sessions and say how many |
| `messages`, `blocks`, `tool_calls` | not duplicate copies; on the active path, or path unknown |
| `tool_calls` | wrapper calls whose children are counted are excluded |
| `annotations` | current and not withdrawn; withdrawal records and read marks stay out unless `kind:` explicitly selects them |
| `alternates` | current and not withdrawn; `current:`, `root:` or `withdrawn:` selects revision history |
| `feed` | read marks stay out unless `annotation_kind:` explicitly selects them |

`queries where name:*` selects named questions; unnamed questions remain available in `queries` without that condition.

A reference drops the reachability defaults for its own match. A group or sort key naming a field leaves its default in place. Rows included because their path is unknown are counted in `notes`. No default concerns deletion, because Polylogue has none ([architecture.md](architecture.md#deletion)). A footer names every default that removed rows ([terminal.md](terminal.md#footers-and-next-lines)).

Session conditions match physical sessions before their matches fold into logical roots. `why` counts each condition on that physical population; the logical-session default separately shows the logical count against the physical count. A root with unresolved ancestry stands on its own until its original relationship evidence resolves it.
Different physical sessions can make one logical session match both `cwd:/a` and `NOT cwd:/a`. Use `physical:true` when comparing complementary populations of physical rows.

## Times

Time expressions (relative, calendar, natural-language dates, bounds, daylight-saving rules) and the logical day are the shared language's. Polylogue's use of them:

- A query resolves its times once, against one `now` and zone, recorded in `resolved`. A cursor carries them and a kept result records them; a saved query and a watch resolve them again on each run.
- Every time Polylogue reads carries `time_quality` and `precision` ([sessions.md](sessions.md#times)), in the closed sets shared with Sinex.
- A tool call's `result_at` is the time of its result or completion notice. `reported_duration` is the harness's own duration in milliseconds, unknown when absent; it is separate from the elapsed `duration` of [measures](measures.md), which needs both times on one clock.
- **Coarse times.** A coarse or bounded time satisfies a time condition only when its whole period does, fails when none of its period does, and is otherwise `unknown`. A row whose time is unknown never matches a time condition, is counted in `notes`, and `group by day` puts it in an `unknown` group.
- Period groups (`day()`, `week()`, `month()`) follow the logical day, and each says `period: complete|partial`. Their labels are common's, the same as Sinex's: `2026-11-09`, the ISO week `2026-W46`, `2026-11`.
- Time columns in replies are RFC 3339 in the logical-day zone with its offset (`2026-11-09T09:12:00+01:00`), so tables show local clock times and the instant stays exact.
- `last_active` is the latest of the messages' known source times. Its possible period is the maximum of their whole periods: a recorded period `[0,10)` and a point at 5 give `[5,10)`. Its source quality is known only when every possible latest message has that quality; a combination of different source qualities leaves the derived quality unknown. Calendar bounds are never inferred from the reader's zone. A time condition reads stored times only of rows inside its bounds and of the few rows whose time is a period, which an index names; an instant outside the bounds is decided by its millisecond column.
- An order relation's `at when` and `at start` anchor the outer row at its start; `at end` anchors it at its end. The inner row retains its source period and start. `match.distance` is the signed time from that outer point to the inner start, in seconds. Coarse periods still have to satisfy the whole-period relation and window.
- An order condition chooses its nearest satisfying matching row by default. `pick first by E [asc|desc], …` chooses by explicit inner-row sort keys instead, with unknown values last and identity as the final tie-breaker; the outer row appears once, in its existing order. `abs(match.distance)` compares exact nanosecond distances between source times when choosing a time matching row; the public signed distance remains seconds. All operands of a condition read the relation entering it. After the complete predicate filters those rows, exactly one positive order condition in its top-level AND exposes its matching row; OR, NOT, and multiple positive order conditions preserve the input matching row namespace.

## Related rows, transforms, attached rows, combining

Conditions over related rows, transforms after `|`, attached rows and combining queries are [transforms.md](transforms.md).

## Reading a query back

- `--explain` on a line ([terminal.md](terminal.md#explain)) compiles the selection without reading rows and returns the parsed query with spans; the defaults applied and which conditions dropped them; units; definitions; resolved times; the ranking in effect (none, relevance, similarity, fused, and which lists); `query:<hash>`; and `read_as`, the query read back in plain language from each field's, operator's and transform's declared phrase. With `--json` the reply also carries the compiled plan.
- `--why`, a listing option, also counts within the aggregate deadline: first each default the unit applied, as `{clause, matches, of}` where `clause` names the conditions that lift it and `matches of of` is the rows it retains of the rows without it; then per condition `{clause, group, matches, of, hint}`, nested by OR, NOT and parenthesis group, with no limit on entries; for a ranked reply, each shown row's rank in each list and its contribution. For an empty reply it adds the nearest relaxation: the time bound that would admit rows, the closed-set values present, the closest field values.
- Every reply carries `read_as`.
- `--explain` and completion never write, build, embed or call a model.

## Paging

- **Four bounds stay apart.** Selection (`limit`, `[last:k]`) is in the question. The page size (`--page-size`) is how much travels now (default 20 sessions or 50 other rows, the same on every surface). A presentation budget (`--budget`) cuts display, never a figure's population. The deadline bounds execution.
- **Default order.** A unit's default order puts rows without a time after the dated ones. An index carries that order's leading terms for sessions, messages, tool calls and events, so their first page reads only its own rows.
- **Key first.** A page of a unit's rows is read in two steps: the selection's keys and order values are ordered and paged, then only the page's rows are read. Sorting and paging touch keys, never the columns of rows off the page.
- For tool-call queries, a named session resolves to one indexed session key.
- **Cursors** (`--cursor`) hold the last row's order-key values with row identity as the final key, the resolved `now`, the random seed, the index generation, the first page's read position, and a hash of the resolved question (normal form, fields, caller, zone, day start, ranking options, vector write counter). They are opaque and integrity-checked.
- **Continuation is by key,** never by offset, and no read transaction is held between pages. When the archive changes between pages, no row whose key did not change is skipped or repeated, and `notes` say "the archive changed since the first page", listing rows that moved before the cursor's position.
- **Refusals:** a cursor from another question is `invalid_query`; one from an earlier index generation, or a ranked cursor after a vector write, is `stale_cursor`, never restarted silently. Only the page size may change between pages.
- A page may be empty and still carry a cursor; only a page without `next` is the end. An enumeration that must not change is saved first ([saved-queries.md](saved-queries.md#kept-results)), then paged.
- The count is computed once per question and read position, and reused by later pages.
- A listing that should wait for rows takes the line option `--wait D` ([interfaces.md](interfaces.md#steps)).
- On a terminal the pager requests each page as it consumes the last; elsewhere every page ends with the line or call that continues it ([terminal.md](terminal.md#output)).

## Deadlines

- Every request carries a deadline from its latency class ([latency.md](latency.md#latency-classes)). Deadlines are defaults for interactive display, not limits on what may be asked.
- **Any caller can lift the deadline,** agents included: `--deadline MS` or `--deadline none` on the command line, `deadline: null` over MCP and HTTP. No valid query is refused for time.
- A read past its deadline ends `timeout`, never `ok` with a partial page. The reply names the running transform and rows scanned, a narrower query when one exists ([latency.md](latency.md#writes-and-deadlines)), and the same query with the deadline lifted.
- A read stops within 100 ms when its deadline passes or its caller cancels or disconnects (MCP cancellation, a closed HTTP connection or socket, Ctrl-C); no abandoned read goes on running in the daemon.
- Waiting for vectors is part of the request's read, bounded by its deadline and cancellation ([warm state](latency.md#warm-state)).
- A read's deadline covers the whole read, from waiting for a read connection and opening it through the last statement. A deadline that passes before any statement runs ends the read the same way, past its deadline.
- Background work never blocks an interactive read; one read connection is reserved for latency-bound requests (hook briefs, `status`, completion) ([operations.md](operations.md#idle-and-steady-state-work)).

## Replies

Every surface returns the shared typed reply: `outcome`, a `value` on success or `error` on failure, and the request and continuation fields. The query document inside `value` has this shape:

```json
{
  "unit": "sessions", "shape": "rows",
  "columns": [{"name": "id", "kind": "text", "reference": "session"}],
  "rows": [], "count": 3, "next": null, "ranked": true,
  "query": "query:9f2c…", "run": "run:…", "result": null, "pre_registered": false,
  "definitions": [], "read_as": "…", "why": null,
  "notes": [], "incomplete": [],
  "unknown": {"similar": 0, "cost": 2, "unparsed_inputs": 0},
  "up_to_date": true,
  "read_position": {"index_instance": "…", "index": 1234, "archive": 87, "cursor": "…"},
  "resolved": {"now": "…Z", "zone": "Europe/Lisbon", "day_start_minutes": 240},
  "positions": null
}
```

- The document carries no outcome of its own: the reply's outcome ([terminal.md](terminal.md#output)) says `ok`, `empty` or `degraded`, and a measurement is the document with `measure` and `data` in place of the row fields. Errors are `{"outcome":"error","error":{…}}`. `positions` is filled only for cross-system replies ([primary-mode.md](primary-mode.md#cross-system-queries)).
- `empty` comes only from a read that succeeded and matched nothing; a failure never produces `empty`.
- A reply assembled from parts (the two sides of a hybrid search, the lists fused in the brief of a search, cross-system parts) is `incomplete` if any part failed, naming each.
- `up_to_date` is false while files have unparsed growth, inputs wait, derived products are behind, or the index is held; `notes` say which.
- `unknown.unparsed_inputs` counts stored inputs the query's origins cannot read yet; a reply that would be `empty` is then `incomplete`.

## Facets

A facet count is `group by F | count` over the same filtered population as the reply, never the visible page, in two views: `scoped` (under the query) and `global` (under the defaults alone), with IDF and lift per value computed from the two (`| measure facets`, [measures.md](measures.md#sessions-work-and-comparison)). A facets request runs several in one read transaction, after the first page; a count past its deadline is reported deferred, never zero. Presentation is the web's ([web.md](web.md)).

## Acting on rows

Verbs after the selection act on the rows it resolved, once, in order: reading, briefing, notes, tags, judgments, saving and acts on live sessions ([interfaces.md](interfaces.md#verbs), [interfaces.md](interfaces.md#steps)). A write over several rows of a query needs `--all`, writes one annotation per row in one request, and reports the committed count with `query:<hash>` and its undo line; a reviewed population is named with `in:result:<id>`. There is no export verb, because writing out is `read --to`, and no delete verb, because Polylogue has no deletion.

## Read-only SQL

`polylogue dev sql STATEMENT` runs one read-only statement against `index.db` with `archive.db` attached, through versioned views (`v1_sessions`, `v1_messages`, `v1_tool_calls`, `v1_links`, `v1_annotations`): `polylogue dev sql 'select count(*) from v1_sessions'`. It is an escape hatch outside the language: a need met repeatedly only through `dev sql` is a gap in the language. Over MCP it is the `dev sql` operation of every tool ([agents.md](agents.md#mcp-server)).

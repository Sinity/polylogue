# Query

Polylogue asks every question through one query language: listings, reports, recall, watches, facets and the searches inside briefs are all queries, and every surface answers the same query the same way. This page owns Polylogue's catalog and compiler: units, fields, defaults, times, stages, query objects, paging, deadlines, answers, facets, watch semantics, acting on rows and read-only SQL.

The grammar, its spelling, the AST's three-valued semantics, time expressions, the logical day and the cursor type are shared with Sinex and owned by the common [`query-language`](https://github.com/Sinity/polylogue-sinex-common/blob/master/docs/query-language.md) crate, together with the laws that are properties of the language. Text and semantic search are in [search.md](search.md); how the command line reads arguments, completion and exit codes are in [interfaces.md](interfaces.md); references are in [references.md](references.md); measures in [measures.md](measures.md); queries that mix Polylogue and Sinex data in [primary-mode.md](primary-mode.md#cross-system-queries).

The product catalog comes from the unit and field registries used by the compiler. Parsing, grammar completion and discovery use that vocabulary. Observed repositories, models and other values extend completion suggestions; they do not turn an open field into a closed set. The product compiler consumes the shared tree directly and refuses a shared form whose execution it does not implement with `not_built`.

## Principles

- **One expressive language instead of small features.** No surface re-implements filtering, totals or paging; a new need is met by the language or a stage, not by a command. Ruling, 2026-06-17: "The whole point of expressive query DSL is to make such funcitonality alwasy within reach. I don't want tons of specific weird little features" (chatgpt:6a2ea930-87a8-83eb-adb7-4cf38ccaa6a3:3f4ec96c-1c94-48aa-9ae6-ceeaaed5eddc). Ruling, 2026-07-02: "dont build one-off silos, build things that powerfully compose" (cc-queued:3437c3ea-5b13-45b4-9b88-8786223f0fa7@2026-07-02T18:20).
- **Unknown is not zero; scopes are strict.** An unknown value matches neither a condition nor its `NOT`. A query never widens itself, and never falls back to searching its own text.
- **No arbitrary limits.** Page sizes are defaults, deadlines can be lifted by any caller, and nothing truncates a valid answer.
- **Discoverable without shrinking the language.** `explain`, `read_as`, completion and runnable corrections teach the language; every string an agent might pass returns rows, a refusal with a correction, or a typed error.
- **Agents are callers like the operator,** with the same powers, deadlines and answers.

## Grammar

Polylogue uses the shared grammar. The shapes it accepts:

```
[UNIT [where CONDITIONS]] [with ATTACHED] [| STAGE]...      tool_calls where outcome:error | group by session | count
CONDITIONS [with ATTACHED] [| STAGE]...                     origin:codex since:7d with messages(id,role)[last:1]
SESSION-CONDITIONS | UNIT [where CONDITIONS] [| STAGE]...   tag:project:morrow | tool_calls where action:shell
```

- Conditions side by side are AND; parentheses group everywhere; without a unit, the unit is `sessions`.
- **Atoms:** `field:value`; `field OP scalar`; `field between A and B`; `field ~ "text"` (contains); a text term (bare word or `"phrase"`, `word*` prefix, `word~` approximate, see [search.md](search.md)); `exists UNIT(…)`; `count(UNIT [where …]) OP N`; `FIELD in (QUERY)`; `in:result:<id>`; `seq(step -> [next|within:D] step …)`; `query:<hash>` or `query:NAME[(p=v,…)]`; `similar:TEXT|REF`; a reference.
- **Values:** `word`, `"quoted"` with backslash escapes, `(a|b|c)`, `*` (names the field without restricting it and drops its default), `unknown`.
- **Comparisons:** `> >= < <= = !=`; durations `ms s m h d w`; money `2.90` or `2.90usd`. There is one spelling: the compact `field:>=N` is refused with `field >= N` as its suggestion.
- **Depth:** parenthesis and nesting depth is checked against 64 before evaluation, because a parser recursion has to stop somewhere; deeper is `invalid_query` with the reason. Saved-query expansion refuses a cycle and names the chain. No other depth or size limit exists.

How the first word of a command line is read (command, query signal, `find`) is in [interfaces.md](interfaces.md#grammar).

### Errors

Every query error carries `kind`, `message`, `position` (byte offset, shown with a caret), `suggestion` (a corrected query that runs as given, whenever one can be formed), `hint` and `retryable`. Polylogue's query error kinds and their exit codes are listed in [interfaces.md](interfaces.md#exit-codes). A unit or field that is part of the design but has no producer yet answers `not_built`, never an empty result.

### References inside queries

A reference names exactly one thing and drops the reachability defaults (subagent, hidden, duplicate, active path) for that match. The reference forms and their resolution are in [references.md](references.md#forms); the command line's own forms (`@N`, `@last`) are in [interfaces.md](interfaces.md#addressing). An ID-shaped word or unique prefix resolves to exactly one session or message, or is `not_found` or `ambiguous` naming the matches; it never silently becomes search text. `self` is the calling session; `last` and `latest` are the most recently active main session in scope, never the calling session.

## Units and fields

Raw tool-call failures are queryable as recorded outcomes: `tool_calls where outcome:error`. These rows report tool results without classifying a command as a verification check.

Each unit declares, once, its rows, key and identity, default filters, default order, search behaviour and relations (`session.`, `message.`, `call.`, `child.`, `next.`, `parent.`, `recipient.`). Each field declares its kind (match, substring, words, subtree, number, money, time, duration, bool, ref), its closed value set taken from the model's stored names, its unknown word, a one-line help and the phrase `read_as` uses. The full listing will be rendered into [planned reference/query-units.md](reference/README.md#planned-pages).

| Unit | Rows |
|---|---|
| `sessions` | logical sessions by default ([sessions.md](sessions.md#logical-sessions)); `--physical` or `physical:true` lists physical sessions |
| `messages`, `blocks`, `tool_calls`, `documents`, `attachments`, `events`, `links`, `turns`, `files` | as parsed ([sessions.md](sessions.md)) |
| `pastes` | paste spans: content hash, source of the span, state, message and session; `group by hash` gives repeats |
| `assertions` | assertions ([assertions.md](assertions.md)) |
| `variants` | variants and their parts ([assertions.md](assertions.md#variants)) |
| `deliveries` | deliveries ([agents.md](agents.md#deliveries)) |
| `changes` | the change feed as rows |
| `queries`, `runs`, `results` | query objects, query runs and kept results ([below](#query-objects)) |
| `processes`, `commits` | harness processes; commit and pull-request links (`not_built` until their producers exist) |

Field rules:

- **Session-field shorthand.** In a row query, a session field without `session.` means `session.<field>` when the row unit has no field of that name; `explain` shows the full form.
- **String matching** is case-insensitive. `title`, `path`, `command` and `input` match substrings; `output` and `thinking` match words; `cwd` and `checkout` match subtrees (whole path components, case-sensitive, `_` and `%` literal); everything else matches whole values.
- **Two models in one session** match either; `group by model` counts the session under each.
- **Counts and content fields** include `project`, `has:code` and `code_lang`, per-role counts (`user_words`, `assistant_words`, `tool_use_messages`, `thinking_messages`, `paste_messages`), `since_session:<ref>`, and `time_quality` and `precision` on every time.
- **Assertion fields**, on `assertions` and as conditions on any unit through `kv:`: `kind`, `key`, `value`, `author`, `by_operator`, `agent`, `unattributed`, `via`, `about`, `evidence`, `scope`, `origin`, `current`, `conflict`, `withdrawn`, `judged`, `judged_by`, `expired`, `changed_since`, `waiting`, `pass`, `rubric`. On any unit: `tag:<name>[:<value>]`, `title:`, `kv:<kind>.<key>=<value>`, `hidden:`, `referenced_by:<ref>`. Grain is strict: an assertion about a session never matches its messages.
- **Variant fields:** `variant:true` admits variant text into text search; `lang:<code>` matches variant parts in that language and messages by their derived `lang`.
- **Names shared with Sinex** keep their meaning per product: `evidence_until:` is Polylogue's as-of field, and Sinex's "as of" is a different concept.

## Defaults

Each unit binds its default filters once, when its source is bound (a unit, `query:…` or `result:<id>`). A default is dropped only by a top-level AND condition on its field, or by `F:*`; a condition under OR, NOT, `exists`, `seq` or a saved query never drops one. After binding, everything further only narrows.

| Unit | Default |
|---|---|
| `sessions` | main sessions (subagents roll up into their logical session); not stubs |
| listings and search over every unit with a session | not `hidden`; measures and population counts include hidden sessions and say how many |
| `messages`, `blocks`, `tool_calls` | not duplicate copies; on the active path, or path unknown |
| `tool_calls` | wrapper calls whose children are counted are excluded |
| `assertions`, `variants` | current and not withdrawn |

A reference drops the reachability defaults for its own match. A group or sort key naming a field keeps its default. Rows kept because their path is unknown are counted in `notes`. No default concerns deletion, because Polylogue has none ([architecture.md](architecture.md#deletion)).

## Times

Time expressions (relative, calendar, natural-language dates, bounds, daylight-saving rules) and the logical day are the shared language's. Polylogue's use of them:

- A query resolves its times once, against one `now` and zone, recorded in `resolved`. A cursor carries them and a kept result keeps them; a saved query and a watch re-resolve on each run.
- Every time Polylogue reads carries `time_quality` and `precision` ([sessions.md](sessions.md#times)), in the closed sets shared with Sinex.
- Tool-call `result_at` is the native result or completion-notification time. `reported_duration` is the harness's own duration, in milliseconds, and stays unknown when absent. It is separate from the elapsed `duration` defined in [measures](measures.md); that field requires the declared same-clock interval and its reason before it enters the product catalog.
- **Coarse times.** A coarse or bounded time satisfies a time condition only when its whole period does, fails when none of its period does, and is otherwise `unknown`. A row whose time is unknown never matches a time condition, is counted in `notes`, and `group by day` puts it in an `unknown` group.
- Period groups (`day()`, `week()`, `month()`) follow the logical day, and each says `period: complete|partial`.

## Related rows, stages, attached rows, combining

### Conditions over related rows

- `exists child(C)`: one child satisfies all of C; three-valued.
- `count(UNIT [where C]) OP N`: a count is always known, so it and its `NOT` partition the rows.
- `FIELD in (QUERY | group by F)`: the value is one of the other query's grouped values of the same type.
- `seq(A -> [next] B -> [within:5m] C)`: tool calls in session order. Matches are the earliest non-overlapping; a step's own conditions narrow before matches are chosen; `within` needs both times, the later strictly after; `| matches` returns each match as rows. `seq all(…)` (every, overlapping match) and `gap_steps`/`gap_time` on match rows are part of the language.

### Stages

Stages apply in the order written:

- `sort by F [asc|desc], …`: unknowns last in either direction; row identity is the final tie-break. `sort by random[(seed)]` keeps its seed in the cursor.
- `group by K1, K2 | count` or `| agg …`: keys are fields and `day()`, `week()`, `month()`.
- `agg count, sum:F, avg:F, min:F, max:F, pNN:F, count_distinct:F, histogram:F, corr:F,G`: each value is summed at its owner's grain; unknown and partial values are reported as `unknown_F` and `partial_F`; percentiles are nearest-rank.
- `| where C` after grouping filters groups.
- `limit N`, `offset N` are selections and change the question; there is no maximum.
- `distinct`; `matches`; `| UNIT [where C]` (rows of another unit within the matched sessions); `fields a,b` (projection; `--fields` on the command line).
- `| measure WHAT` hands the selection to a measure ([measures.md](measures.md)).
- Further stages are part of the language and answer `not_built` until built: a per-group limit (`| group by origin | first 5`), derived keys (`program(command)`, `fingerprint(text)`), `| series`, `| extract number before "MB/s" as rate`, `| cluster by similar [k:20]`, and `let` names within one query.

### Attached rows

`QUERY with messages(id, role, text)[role:assistant, last:2], tool_calls(…)` returns each parent row with one list per child spec. Inside the brackets, conditions filter first, then `first:` and `last:` take that many per parent in the child's order. Attaching never changes which parents are returned. There is no page-wide ceiling, and attached text is never cut.

### Combining queries

- `Q | union (R)`, `| intersect (R)` and `| except (R)` combine two queries of the same unit; `| fuse (R)` adds R's order as one more ranking without adding rows.
- A chain of `| union` and `| fuse` stages at one level is one reciprocal-rank fusion over all operands ([search.md](search.md#fusion)); `intersect` and `except` keep the left order; an unranked operand contributes its own order; `found_in` names the lists that ranked each row.
- Grains must match; a mismatch is `invalid_query` suggesting the explicit lift (`| sessions`). Combining applies before a terminal aggregate.

## Query objects

Queries, their runs, kept results, figures and findings are stored objects with identities, so a question and its answer can be cited, reused and compared over time. Ruling, 2026-07-09: "what if Polylogue tracked its own use? … saving a query in a database, and returning that as a result -- with proper ids … THIS. THIS IS WHAT I MEANT" (cc-queued:3b0038ec-234c-44b8-bb88-fe222b44fe0f@2026-07-09T20:06).

### Identity

- **`query:<hash>`** is the BLAKE3 hash of the query's normal form: the parsed AST after resolving surface references to their stored forms and saved-query names to the hashes they resolved to, with operands sorted where order carries no meaning and spelling variants removed. Two texts with one normal form have one hash. The normal form and the hash function belong to the shared language, so Sinex hashes its queries the same way.
- `query:<hash>` accepts the full 64-hex hash or a unique prefix of at least 12 hex characters. Saved-query names cannot be all hex.
- Relative times stay as expressions in the normal form (`since:7d`, not an instant); each run records what they resolved to.

### Names and composition

- **Names point at hashes.** A name is a `saved_query` assertion ([assertions.md](assertions.md#kinds)) with key = name and value `{text, hash, denominator?}`. Saving under an existing name revises that assertion, with the ordinary currency rules ([assertions.md](assertions.md#currency)); `query:NAME` resolves to the hash of the current revision, and two current revisions under one name are `ambiguous`.
- **Composition edges are recorded at compile time** in `query_edge(query, uses, how)`, where `how` is `saved` (a name and the hash it resolved to), `result` (a kept result used as source), `combine` or `in` (a sub-query). "Which queries use this one" and "what did this query depend on" are queries over the `queries` unit.

### Runs

- Every executed query writes a run: `query_run(id, query, at, resolved, read_at, author, via, outcome, count, exact)`, with the author computed as for every authored record ([assertions.md](assertions.md#authorship)). `explain` and completion write none.
- `query(hash, normal_form, first_run)` holds each distinct query once.
- Runs are written by the writer in its ordinary groups, never inside the read, so a read never waits on its run row.
- Every answer carries `query: "query:<hash>"` and `run: "<run id>"`; a kept answer also carries `result:<id>`.

### Saved queries

- **Shapes.** A saved query whose text is a relation (a unit and conditions, no stages) can stand where a condition can, over the same unit: at the top level it is the source, bound with its own defaults and times; under OR or NOT it means membership. A saved query with stages or `similar:` is a shaped result, which can be read and followed by stages only.
- **Parameters:** `$word` in the text is a parameter, given as `query:NAME(word=VALUE)`. Values substitute in the parsed tree, never as text; a missing or unused parameter is an error naming it.
- **Recorded definitions:** every answer and kept result names each name it used, with the hash and saved-query revision it resolved to (`definitions`). Revising a saved query never changes what an earlier kept result says it ran.
- **Shipped saved queries** are data in the code, listed beside the operator's own and without an author: `running`, `live`, `notices`, `changes`, `triage`, `needs-review` (non-operator assertions not yet judged by the counted judges), `open-questions`, `unjudged-agent-notes`, `resume_candidates`. A saved query of one's own with the same name takes precedence; withdrawing it brings the shipped one back. Shortcuts are shipped saved queries, never text builders.
- **`query:triage`** orders sessions across all repositories by a visible sum of named facts, each shown with its contribution: a failed command never followed by a success of the same command; three or more consecutive failures of one normalized command; cost or duration at least three times the repository's 30-day median; three or more operator corrections; open questions and blockers. Tagging a session `triaged` removes it; later activity brings it back.

### Kept results

`QUERY keep [--name N]` (and `keep` over MCP and HTTP, `--keep` on recall) stores the answer as it is now under `result:<uuidv7>`. It records:

- `query:<hash>`, the run and the definitions;
- the resolved bounds (`now`, zone, day start, resolved times) and `read_at`;
- **exactness**: whether `count` was exact, and whether the answer was complete or `incomplete`, with the missing parts;
- columns with kinds and references, each row's values and references, and assertions pinned to their revisions;
- for `exists` or `seq`, the match rows; for a ranked answer, each row's rank and per-list contributions;
- the version of every derivation the answer read.

Rules:

- **Ranking is kept, or the keep is refused.** Keeping a ranked answer whose ranking is incomplete is `refused` with the missing list named; the caller can keep the unranked membership by removing the ranking term.
- **Reading back** `result:<id>` (or `#<row>`, `.matches`) reads the stored rows, never a re-run; only stages may follow (`result:<id> | group by origin | count`). Re-running is running `query:<hash>`. Where a derivation version changed since, the read says so ("computed under prices `<hash>`, now `<hash>`").
- **Membership:** `in:result:<id>` restricts any query of the result's unit to its rows; a bulk write over an exact reviewed population uses it.
- **One kept result, many uses:** assertions, page slots, briefs, watch baselines and written-out bundles cite the same `result:<id>`; nothing copies rows.
- **Every row is stored,** in `kept_row(result, position, values_json, refs_json)`, so paging a large result reads by key. Keeping an answer that cannot be read within one deadline keeps it in steps, each its own read transaction, recording every position read.
- A name is a display label on the result, never a reference.
- Kept results are user state, not assertions; in primary mode they travel as `polylogue.kept_result` records, and runs as `polylogue.query` and `polylogue.query_run` records ([primary-mode.md](primary-mode.md)).

### Figures

A **figure** is a saved `| measure` or `| agg` query with a declared denominator query: a `saved_query` assertion whose value carries `denominator: query:<hash>`. Running a figure answers the value with its denominator count, its unknown count and its exactness; keeping it keeps all three. A figure without a denominator is refused at save.

```console
$ polylogue 'sessions since:30d | measure cost by day' save cost-30d --denominator 'sessions since:30d'
$ polylogue query:cost-30d keep
```

### Findings

A **finding** is an assertion of kind `finding` that cites the `result:` or `query:` it rests on through a `derived_from` link. `polylogue result:<id> assert finding "…"` writes one about the result with that link; `polylogue result:<id> --stack` lists the findings resting on a result.

### Pre-registration

A run or kept result is **pre-registered** when a `saved_query` assertion naming its `query:<hash>` (or a `claim` or `forecast` whose check names it) was committed before the hash's first run on this archive. Answers and kept results carry `pre_registered: true|false` with the registering assertion. Order is the archive's commit order, never a clock; in primary mode, Sinex's commit order.

## Explain

- `explain QUERY` compiles without reading rows and returns the parsed query with spans; the defaults applied and which conditions dropped them; units; definitions; resolved times; the ranking in effect (none, relevance, similarity, fused, and which lists); `query:<hash>`; and `read_as`, the query read back in plain language from each field's, operator's and stage's declared phrase. With `--detail full` it adds the compiled plan.
- `explain` alone describes the language; `explain units|fields [UNIT]|stages|operators|capabilities` pages the declarations.
- `--why`, on any query, also counts within the aggregate deadline: per condition `{clause, group, matches, of, hint}`, nested by OR, NOT and parenthesis group, with no limit on entries; for a ranked answer, each shown row's rank in each list and its contribution. For an empty answer it adds the nearest relaxation: the time bound that would admit rows, the closed-set values present, the closest field values.
- Every answer carries `read_as`.
- Explain and completion never write, build, embed or call a model.

## Paging

- **Four bounds stay apart.** Selection (`limit`, `[last:k]`) is in the question. `page_size` is how much travels now (default 20 sessions or 50 other rows, the same on every surface). A presentation budget cuts display, never a figure's population. The deadline bounds execution.
- **Cursors** hold the last row's order-key values with row identity as the final key, the resolved `now`, the random seed, the index generation, the first page's read position, and a hash of the resolved question (normal form, fields, caller, zone, day start, ranking options, vector write counter). They are sealed: opaque and integrity-checked.
- **Continuation is by key,** never by offset, and no read transaction is held between pages. When the archive changes between pages, no row whose key did not change is skipped or repeated, and `notes` say "the archive changed since the first page", listing rows that moved before the anchor.
- **Refusals:** a cursor with an invalid or retired seal, or from another question, is `invalid_query`. An authenticated cursor whose read context is no longer available, including a ranked cursor after a vector write, is `stale_cursor`. The index keeps its private seal key across daemon restarts; rebuilding retires that key. Neither refusal restarts the query silently. Only the page size may change between pages.
- A page may be empty and still carry a cursor; only a page without `next` is the end. An enumeration that must not change is kept first, then paged.
- The count is computed once per question and read position, and reused by later pages.

## Deadlines

- Every request carries a deadline from its latency class ([interfaces.md](interfaces.md#latency-classes)). Deadlines are defaults for interactive display, not limits on what may be asked.
- **Any caller can lift the deadline,** agents included: `--deadline MS` or `--deadline none` on the command line, `deadline: null` over MCP and HTTP. No valid query is refused for time.
- A read past its deadline ends `timeout`, never `ok` with a partial page. The answer names the running stage and rows scanned, a narrower query, and the same query with the deadline lifted.
- A read stops within 100 ms when its deadline passes or its caller cancels or disconnects (MCP cancellation, a closed HTTP connection or socket, Ctrl-C); no abandoned read keeps running in the daemon.
- Background work never blocks an interactive read; one read connection is reserved for latency-bound requests (hook briefs, `status`, completion).

## Answers

Every surface returns one answer shape, rendered from types:

```json
{
  "outcome": "rows",
  "unit": "sessions", "shape": "rows",
  "columns": [{"name": "id", "kind": "text", "reference": "session"}],
  "rows": [], "count": 3, "next": null, "ranked": true,
  "query": "query:9f2c…", "run": "…", "result": null, "pre_registered": false,
  "definitions": [], "read_as": "…", "why": null,
  "notes": [], "incomplete": [],
  "unknown": {"similar": 0, "cost": 2, "unparsed_inputs": 0},
  "up_to_date": true,
  "read_at": {"generation": "…", "index": 1234, "archive": 87, "vectors": 167943},
  "resolved": {"now": "…Z", "zone": "Europe/Lisbon", "day_start_minutes": 240},
  "positions": null
}
```

- `outcome` is `rows`, `empty`, `incomplete` or `measure`; errors are `{"error": {…}}`. `positions` is filled only for cross-system answers ([primary-mode.md](primary-mode.md#cross-system-queries)).
- `empty` comes only from a read that succeeded and matched nothing; a failure never produces `empty`.
- An answer assembled from parts (the two sides of a hybrid search, recall lists, cross-system parts) is `incomplete` if any part failed, naming each.
- `up_to_date` is false while files have unparsed growth, inputs wait, derived products are behind, or the index is held; `notes` say which.
- `unknown.unparsed_inputs` counts stored inputs the query's origins cannot read yet; an answer that would be `empty` is then `incomplete`.

## Facets

A facet count is `group by F | count` over the same filtered population as the answer, never the visible page, in two views: `scoped` (under the query) and `global` (under the defaults alone), with IDF and lift per value computed from the two. A `facets` request runs several in one read transaction, after the first page; a count past its deadline is reported deferred, never zero. Presentation is the web's ([web.md](web.md)).

## Watches

A watch is a signal rule over a query ([operations.md](operations.md#notices)); the rule schema and the evaluator's semantics (changes, baselines, cooldowns, thresholds, catching up, downtime) are the common [signals](https://github.com/Sinity/polylogue-sinex-common/blob/master/docs/signals.md) design. What Polylogue's query side supplies:

- `watch [--name N] QUERY` keeps the current answer as the baseline (a kept result) and re-evaluates as data arrives, re-resolving relative times.
- Changes are reported as `membership` (rows joined or left), `ranking` or `top_k`, never one for another.
- A reading taken while the index catches up, or an incomplete answer, is held as incomplete, never reported as a change.
- Occurrences are keyed by logical session, so a re-parse never fires a watch again and one logical session fires once.
- A watch over assertions compares by kind, target, key and value.
- The rule is stored as a `signal` assertion. A watch only notifies; it never starts or resumes an agent ([agents.md](agents.md#pings)).

## Acting on rows

Verbs act on the matched set: `read` (and `read --to`, [reading.md](reading.md#writing-out)), `keep`, `assert`, `revise`, `withdraw`, `judge`, `ping` and the others in [interfaces.md](interfaces.md#command-table). A bulk write over all matches resolves the query in the write's own transaction through the same compiled plan, writes one assertion per row, and reports the committed count with `query:<hash>`; an exact reviewed population is named with `in:result:<id>`. There is no export verb, because writing out is reading, and no delete verb, because Polylogue has no deletion.

## Read-only SQL

`sql` runs read-only statements against `index.db` with `archive.db` attached, through versioned views (`v1_sessions`, `v1_messages`, `v1_tool_calls`, `v1_links`, `v1_assertions`). It is an escape hatch outside the language: a need met repeatedly only through `sql` is a gap in the language.

## Laws

The language's own laws (filter composition, three-valued filters, grouped rows, match order, closed sets, command words, the page is not the answer, coarse times, namespace binding) are the common `query-language` suite, run against Polylogue's catalog. Mode parity across surfaces is in [interfaces.md](interfaces.md#laws). Polylogue's own:

- **definition-reuse**: a saved query selects the same rows by name and inline; a result kept under revision A still names A after revision B.
- **attachment-keeps-parents**: attaching rows (`with`) never changes which parents are returned.
- **defaults-bound-once**: on Morrow, `tool_calls where outcome:error (action:shell OR NOT action:shell)` equals `tool_calls where outcome:error`; a condition under OR never drops a default.
- **reference-drops-defaults**: a reference alone resolves to the same row as its full ID and is returned even when a default would exclude it.
- **pasted-strings-answer**: every pasted string returns rows or a correction: a session ID prefix, a provider URL, `"unbalanced`, `reop:morrow` (suggesting `repo`), `messges cargo` (suggesting `messages`).
- **depth-and-spelling-refusals**: depth 65 is refused with its reason; `field:>=3` is refused suggesting `field >= 3`.
- **last-excludes-caller**: `last` called from inside a session never returns that session.
- **times-resolved-once**: a paged query keeps its `now` across pages; DST days of 23 and 25 hours and 01:30 on the previous logical day group correctly; a pinned table of natural-language phrases resolves as listed.
- **coarse-time-unknown**: a day-precision time is `unknown` under `since:` a mid-day instant of that day and true when the whole day is after it.
- **query-hash-normal-form**: two spellings with one normal form have one `query:<hash>`; a changed condition changes it; a name resolves to the hash of its current revision, and two current revisions under one name are `ambiguous`.
- **every-run-recorded**: every run writes one `query_run` with the computed author; `explain` and completion write none.
- **composition-edges**: a query using `query:a` and `in:result:r` records both edges, and "what uses `query:a`" lists it.
- **pre-registration-order**: a saved query committed before the hash's first run marks the kept result `pre_registered: true`; one saved after marks it `false`.
- **kept-result-stable**: a kept result reads back unchanged after new sessions arrive and after a re-parse; `.matches | agg count` equals the match count at keep time; a 100,000-row result pages with per-page cost independent of its size.
- **ranked-keep-or-refuse**: keeping a fused answer stores ranks and contributions; keeping one whose vector side is missing is refused naming the list.
- **figure-needs-denominator**: a figure without a denominator is refused at save; a kept figure reports value, denominator count and unknown count.
- **kept-result-derivation-versions**: a kept result records derivation versions and reports the change after a price-table change.
- **paging-stable-under-writes**: paging across concurrent writes never repeats or skips an unchanged row; a cursor across a rebuild is refused with `invalid_query` after its seal key retires; random-order pages never overlap.
- **deadline-timeout-and-lift**: a read past its deadline ends `timeout` naming a narrower query and the lifted form, and is cancelled in the daemon within 100 ms; the same query with `--deadline none`, or `deadline: null` over MCP called as an agent, completes.
- **disconnect-cancels**: a client disconnect cancels the read.
- **answer-schema**: one JSON-schema test of the answer holds across all surfaces; `empty` never results from a failure.
- **read-as-complete**: every field, operator and stage has a reading phrase; `read_as` for every Morrow example is pinned; `--why` on an empty Morrow query names the clause that matched nothing and the known values.
- **facets-equal-group-by**: facet counts equal their `group by` queries.
- **watch-change-kinds**: a watched `similar:` query reports rows joining and the top five changing as separate changes; a reading during catch-up reports nothing; a re-parse of the same input never fires again.
- **bulk-write-membership**: a bulk write over a query writes exactly one assertion per row of the query's membership at write time.

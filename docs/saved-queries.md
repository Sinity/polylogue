# Saved queries

A question and its reply are stored objects with identities, so they can be cited, reused and compared over time. This page owns those objects: questions, runs, saved queries, kept results, figures, findings, pre-registration and watches, and the verb `save` that writes them, rows on a page and page checkpoints included. The query language is [query.md](query.md); the reference grammar is [references.md](references.md); signal rules and notices are [operations.md](operations.md#notices); the line grammar and the `--all` rule are [interfaces.md](interfaces.md#steps).

## Query objects

| Object | Reference | What it is | Unit |
|---|---|---|---|
| question | `query:<hash>` | the normal form of a query; every query has one | `queries` |
| run | `run:<id>` | one evaluation: question, read position, author, count; holds no rows | `runs` |
| saved query | `query:NAME`, `query:NAME@N` | a name for a question, with revisions | `queries` (field `name`) |
| kept result | `result:<id>`, `result:NAME`, `result:<id>#<row>`, `result:<id>.matches` | the rows of one run, stored, never run again | `results` |

Composition edges are recorded at compile time in `query_edge(query, uses, kind)`, where `kind` is `saved` (a name and the hash it resolved to), `result` (a kept result used as source), `combine` or `in` (a sub-query). "Which queries use this one" and "what did this run depend on" are queries over `queries`.

## Identity

- **`query:<hash>`** is the BLAKE3 hash of the query's normal form and declared parameter types. The normal form resolves surface references to their stored forms and saved-query names to the hashes they resolved to, sorts operands where order carries no meaning, and removes spelling differences. Its explicit encoding is independent of the Rust AST's serialization layout. Equal normal forms and parameter declarations have equal hashes. The normal form and the hash function belong to the shared language, so Sinex hashes its queries the same way.
- `query:<hash>` accepts the full 64-hex hash or a unique prefix of at least eight hex characters; replies print the shortest unique prefix, as footers do (`query:3f9a1c2e`).
- Relative times stay as expressions in the normal form (`since:7d`, not an instant); each run records what they resolved to in its stored body.
- `query(hash, normal_form, first_run)` holds each distinct question once.

## Runs

- Executed queries and their computed authorship are permanent records in `archive.db`. Saving a result is a separate action that keeps the selected rows. Explaining a query and completing an input line create no run.
- A listing commits the accepted inputs before evaluating its first page and commits the outcome before returning the reply. A continuation uses the same run identity and creates no new run. The reply exposes the question and run references in `resolved.query` and `resolved.run`. Refusals during preparation create no run. A failed execution records its error. If the writer cannot acknowledge the history, the caller receives an error. An interruption after the beginning commits can leave a pending run.
- A query evaluated to accept a Save, picker selection, or other line population records that evaluation. A kept result identifies the run that selected its population, when there was one, and records the position of its stored rows. Hydrating accepted identities and copying an already kept result do not begin another query run. A kept reply refers to its stored rows as `result:<id>`.
- A query that reads a population, streams records, or starts a follow records the accepted evaluation. Data frames can arrive while the query is running. The stream's completion frame follows the archive's acknowledgement. Follow refreshes and watch checks observe changing data for an existing subscription. They do not start new user runs. A completed direct paged reading can commit its execution inputs and outcome together.
- A run is read as `polylogue run:01JD7Q4M read`: its query ID (`query_id`), read position (`read_position`), author, outcome and count. A run holds no rows; reading the rows again is running its query, and reading the same rows is reading a kept result. A normal run read returns one run card. Its next command runs the immutable query only when that query object is available; an object-read run has no invented query ID. A beginning without a completion is pending, an absent run is not found, and a completion whose beginning has not arrived is unavailable. `--view raw` shows the checked original beginning and optional completion instead of a card.
- The unit `runs` lists them: `polylogue 'runs where author:operator since:1d'`. Filter completed runs with `outcome:ok` or `outcome:error`. The outcome object carries `state` with the same word and, for an error, its `kind`.
- The shared language's `answer:<id>` form is refused with `try: polylogue run:<id> read`.

## Save

`save [NAME]` acts on the question as well as the rows: it names the selection's question and stores the rows this line resolved as a kept result, in one act.

One Save step has one request and transaction. Its definition, kept result and optional watch commit together; a failure leaves none of those effects. A retry returns the original kept result and the actual definition and watch revisions accepted by that request, including revisions reused from earlier Saves. Undo names only revisions that the Save created. The kept result uses the request ID; new annotations use that ID when there is one, or its numbered parts when there are several. Reused revisions retain their original IDs.

**What `save NAME` writes**, on any selection (a reference or row numbers become their references' query text, `session:a OR session:b`):

1. the saved query `NAME`, pointing at the selection's question (an annotation of kind `saved_query`; a new revision when `NAME` exists and the question differs);
2. a kept result of the rows this line resolved (the same rows every other step of the line sees), labelled `NAME`, recording the question, the read position and the time;
3. with `--watch`, a watch on `query:NAME` ([Watches](#watches)), with this kept result as its baseline. `--hold D` and `--notify` refine it.

The statement that produced the selection determines what is retained. A keyed unit selection keeps its fixed row identities and hydrates their current columns when the Save step runs, so an earlier tag or note in the line is visible in the kept result. Its original predicate is not evaluated again. An empty keyed selection keeps its original unit and columns and stays empty across later Saves in the line. A derived relation retains the values resolved before the first step, including aggregates and projections that still carry native row keys. For example, a projection of annotation counts stays fixed even if a preceding Save creates another annotation. A later `read` still hydrates the line's fixed entity references at that read step. A named measurement is retained as one summary row with its actual `measure` text and `data` value. A paged measurement follows its normal authenticated cursor to completion under the original binding and request deadline; the retained data contains every page, with each global summary recorded once.

`--denominator 'QUERY'` makes the saved query a [figure](#figures).

It prints every part, with the line that undoes it:

```console
$ polylogue 'tool_calls where outcome:error command~"cargo test" since:7d' save failing --watch
saved   query:failing (revision 1): runs again each time you use it
kept    result:01JD7Q4M (34 rows, 2026-11-16T14:02:00Z), now result:failing
watch   notices when a row starts matching query:failing
undo    polylogue 'annotation:01JD7Q4M.1@01JD7Q4M.1 OR annotation:01JD7Q4M.2@01JD7Q4M.2' withdraw --all --request-id 01JD7Q4M-undo
```

Over MCP the same step is `polylogue_write {"arguments": {"selection": "tool_calls where outcome:error since:7d", "steps": [{"verb": "save", "name": "failing", "watch": true}]}}`.

**`save` with no name** stores the rows only and prints `result:<id>`; this is how a set of rows is fixed for later use, for blind judging ([judging.md](judging.md#blind)) or a pass's population. `--watch` without a name is refused: `a watch needs a name: polylogue 'QUERY' save NAME --watch`.

**Saving a saved query by its own name.** When the selection is `query:NAME` (or `NAME` as the first word) and the name given is the same `NAME`, the question is the saved query's current definition, so no revision is written and only a new kept result is: `polylogue failing save failing` records today's rows under the same name. Saving `query:NAME` under another name saves a question that uses `NAME`, fixed to the revision it resolved to ([Saved definitions](#saved-definitions)).

**What later references mean.**

| Reference | Means | Changes over time? |
|---|---|---|
| `query:NAME` | the saved query's current revision, run at the read position of the line that uses it | yes: it runs again, and follows re-saves |
| `query:NAME@N` | revision N's question, run now | rows change; the text does not |
| `result:<id>` | the stored rows of one kept result | never |
| `result:NAME` | the newest kept result saved under `NAME`, resolved when the line runs; the reply prints the `result:<id>` it resolved to | moves on each re-save |
| `in:query:NAME`, `in:result:<id>` (conditions) | membership in the question now, or in the stored rows | as above |

Inside a saved query's definition, `result:NAME` is fixed at save time to the `result:<id>` it meant then.

`query:NAME@N` selects revision N in the uniquely current annotation family for NAME. A name recreated after withdrawal has a new family; older families remain addressable by their exact annotation references and question hashes. Without a unique current family, colliding historical revision numbers are ambiguous.

**Listing.** `polylogue saved` (a [shipped saved query](#shipped-saved-queries)) lists saved queries with name, revision, text, newest kept result and whether each is watched. `polylogue 'results where name:failing'` lists every kept result of that name, newest first. `polylogue query:failing read` identifies the selected definition in its header, then pages its text, revisions, kept results, watch and dependents before the selected rows, with the line that re-saves each named dependent. Long texts continue through the normal read fragments.

**Re-saving `NAME`.** A different question writes a new revision (earlier ones stay readable as `query:NAME@N`); the same question writes no revision. Either way a new kept result is written and `result:NAME` moves to it; earlier kept results stay listed, since nothing is deleted. A watch on the name stays on, follows the new revision, and takes the new kept result as its baseline, so a re-save never floods notices.

**Withdrawing** the saved query (`polylogue query:failing withdraw`) makes the name unknown again, or restores the shipped query of that name, and withdraws its watch with it. This selects the authored family rather than its result rows ([annotations.md](annotations.md#withdraw)). Kept results are immutable records and stay readable by ID.

**Shipped names.** A saved query of your own with a shipped name takes precedence; withdrawing yours restores the shipped one.

**Freezing and starring.** Freezing rows is `save` with no name; starring a row is `tag star` ([annotations.md](annotations.md#tag)); pinning a row is saving it to the shipped page `pinned` (`polylogue @2 save page:pinned`).

### Saving to a page

`save page:NAME` places the selection's rows on page NAME, one slot per row, creating the page when it does not exist; `--section S` and `--after SLOT` place them, `--fixed` shows exact revisions, `--live` places the question as one live query cell, `--kept` places a newly kept result as one kept cell that keeps its question beside it so it can run again, and `--template T` shapes a new page. `--as FORM` says how the slots show what they hold (`card`, `excerpt`, `text`, `full`, `table`, `list`, `count`, `value`, `chart`, `compare`); without it each kind of target takes its default form ([pages.md](pages.md#forms)). Saving a row the page already shows in that section moves its slot. The rules are [pages.md](pages.md#writing); live and kept cells are [pages.md](pages.md#queries-on-pages).

```console
$ polylogue 'sessions where repo:morrow since:7d' save page:flaky-ci --live --section 'Recent work'
$ polylogue @1-6 save page:reading/agents --as card
$ polylogue 'tool_calls since:30d | measure usage' save page:report --kept --as chart --section Results
```

A [figure](#figures) on a page is saved as a value: `polylogue errors-30d save page:report --as value` shows its value, denominator, unknown count and exactness. A figure's rows are not archive things, so without `--live` it goes on as a kept cell showing the figure as it stood when saved; with `--live` it runs each time the page is read.

`save` on a page itself, `polylogue page:flaky-ci save`, writes a **checkpoint**: a kept result of unit `slots` with every reference resolved to an exact revision ([pages.md](pages.md#time-and-checkpoints)). `polylogue page:flaky-ci save --watch` watches the page's changes, named after the page ([Watches](#watches)). So one verb keeps a question, its rows, rows on a page, and a page's checkpoint.

## Saved names

A name is lowercase letters, digits and `-`, starting with a letter. Saving a name is refused when it equals a command, a verb, a unit, a reserved word (`latest`, `self`, `here`) or a typed reference prefix, or when it is entirely hexadecimal, or contains `:`: each would be read by an earlier first-word rule or as an ID ([interfaces.md](interfaces.md#the-first-word)), so a saved name is always reachable as a first word, and `page:NAME` after `save` is never a name. A shipped name may be reused, and yours wins. Two current revisions under one name are a conflict ([annotations.md](annotations.md#current-revisions)), and `query:NAME` is then refused as `ambiguous`, listing both.

## Shipped saved queries

Shipped saved queries are data in the code (`crates/model/src/saved_queries.rs`), listed beside your own without an author, resolved by name, completed by the shell ([completion.md](completion.md#positions)), shown on the orientation screen and in `help` ([terminal.md](terminal.md#the-orientation-screen)), and overridden by a saved query of your own with the same name. Their plans are compiled ahead of the first use ([latency.md](latency.md#warm-state)).

| Name | Text |
|---|---|
| `running` | `sessions where running:true \| sort by live_rank, last_event_at desc` |
| `waiting` | `sessions where waiting:true \| sort by live_since` (a permission request, a question, or a held web-chat turn) |
| `recent` | `sessions \| sort by last_event_at desc` |
| `unfinished` | `sessions where repo:here ends_open:true \| sort by last_event_at desc` (an unanswered prompt, an unfinished tool call, an open todo or handoff, a partial outcome) |
| `notices` | `deliveries where sender:signal \| sort by at desc` |
| `review` | `annotations where NOT author:operator judged:unknown \| sort by at desc` |
| `questions` | `annotations where kind:question answered:false` |
| `triage` | `sessions where triage_score > 0 \| sort by triage_score desc` |
| `saved` | `queries where name:* \| sort by saved_at desc` |
| `open` | `annotations where kind:(question|goal|problem) open:true` (open questions, goals and problems) |
| `edits` | `annotations where referenced_by:page:* \| sort by at desc` (changes to pages, their slots and the annotations they show, newest first, with author and page; what changed since you looked is `polylogue 'annotations where referenced_by:page:* since:seen \| sort by at desc'`) |
| `wanted` | `links where target_kind:page target_state:waiting \| group by target \| count` (page names linked or placed that do not exist, with how many link to each) |
| `changed-sources` | `links where role:cites target_state:(changed missing withdrawn) \| group by page` (citations whose source changed, went missing or was withdrawn) |
| `uncited` | `annotations where referenced_by:page:* grounding:uncited` (text on pages written by anyone but the operator with no citation) |

Pages add grouping by page to `open`; the page-specific questions below `open` join it when page links and slots are available.

Shipped pages (`home`, `index`, `pinned`, `cost`) and page templates sit beside the shipped saved queries, built in and overridden the same way ([pages.md](pages.md#templates-and-shipped-pages)).

`recent` and `unfinished` are the usual rows of `read` and `resume` with no selection ([interfaces.md](interfaces.md#a-verb-with-no-selection)).

**`triage_score`** orders sessions across all repositories by a visible sum of named facts, each shown with its contribution: a failed command never followed by a success of the same command; three or more consecutive failures of one normalized command; cost or duration at least three times the repository's 30-day median; three or more operator corrections; open questions and problems. Tagging a session `triaged` removes it; later activity brings it back.

## Saved definitions

- **Names point at hashes.** A saved query is a `saved_query` annotation ([annotations.md](annotations.md#kinds)) with key = name and value `{text, hash, denominator?}`; `query:NAME` resolves to the hash of the current revision.
- **Shapes.** A saved query whose text is a relation (a unit and conditions, no transforms) can stand where a condition can, over the same unit: at the top level it is the source, bound with its own defaults and times; under OR or NOT it means membership. A saved query with transforms or `similar:` is a shaped result, which can be read and followed by transforms only.
- **Parameters:** `$word` in the text is a parameter, given as `'query:NAME(word=VALUE)'`. Values substitute in the parsed tree, never as text; a missing or unused parameter is an error naming it.
- **A saved definition holds its executable object and the dependencies it had when saved.** Running a saved hash loads that object, its typed source-output schemas and the definitions it captured; it does not resolve its old dependency names against their current revisions. Saving `query:mine` again therefore never changes a saved query that uses it. Execution requires the object and every saved definition to be stored.
- **Only named definitions are read.** A query reads the saved queries, saved questions and kept results it names (and those their definitions name), never the others, so its cost does not grow with how many are stored. A damaged stored definition fails the queries that name it with its storage error; it is never read as empty.
- **Recorded definitions:** every reply and kept result names each name it used, with the hash and revision it resolved to (`definitions`). A new revision never changes what an earlier kept result says it ran.

## Kept results

A kept result stores a reply as it is now under `result:<uuidv7>`. It records:

- `query:<hash>`, the run and the definitions;
- the resolved bounds (`now`, zone, day start, resolved times) and the read position (`read_position`);
- **exactness**: whether `count` was exact, and whether the reply was complete or `incomplete`, with the missing parts;
- columns with kinds and references, each row's values and references, and annotations pinned to their revisions;
- for `exists` or `seq`, the match rows; for a ranked reply, each row's rank and per-list contributions;
- the version of every derivation the reply read.

Rules:

- **The ranking is stored whole, or `save` is refused.** Saving a ranked reply whose ranking is incomplete is `refused` before anything is written, naming the missing list; removing the ranking term saves the unranked membership.
- **Reading back** `result:<id>` (or `#<row>`, `.matches`) reads the stored rows. A listing over those rows records its own run and can apply transforms (`polylogue 'result:01JD7Q4M | group by origin | count'`). Running the original question again is running `query:<hash>`. Where a derivation version changed since, the read says so ("computed under prices `<hash>`, now `<hash>`").
- A kept result stores the producer's bound logical declarations alongside its physical columns, so same-name `text(id) as id` and `text(cwd) as cwd` go on matching as text. Declarations do not grant access to source fields, periods or positions. Normalized questions and their saved definitions record each source declaration, and reading a saved question checks its recorded declaration against the immutable result it names.
- Transforms over a kept result use its captured shape, columns and values, including a shaped result with no unit rows. Missing fields and source times or positions are not fetched from current rows. Captured row identity is used when present; otherwise each stored item has an identity from the result, part and stored ordinal, so equal values remain distinct when paged.
- **Membership:** `in:result:<id>` restricts any query of the result's unit to its rows; a write over a reviewed population uses it, and a kept result as the selection needs no `--all`, because its rows are a set the user named.
- **One kept result, many uses:** annotations, briefs, watch baselines, blind comparisons and written-out bundles cite the same `result:<id>`; nothing copies rows.
- **Every row is stored,** in `kept_row(result, position, values_json, refs_json)`, so paging a large result reads by key. Saving a reply that cannot be read within one deadline stores it in steps, each its own read transaction, recording every position read.
- An annotation about `result:<id>` is about the kept result itself, not about its rows.
- Kept results are user state, not annotations; in primary mode they travel as `polylogue.kept_result` records, and questions and runs as `polylogue.query` and `polylogue.query_run` records ([primary-mode.md](primary-mode.md)).

## Figures

A **figure** is a saved `| measure` or `| agg` query with a declared denominator query: a `saved_query` annotation whose value carries `denominator: query:<hash>`. Running a figure replies with the value, its denominator count, its unknown count and its exactness; a kept result of it records all three. Saving a `| measure` or `| agg` query with `--denominator` and no name is refused, since a figure is a named question.

```console
$ polylogue 'tool_calls where outcome:error since:30d | agg count()' save errors-30d --denominator 'tool_calls since:30d'
$ polylogue errors-30d save errors-30d
```

The second line records today's figure under the same name with no new revision.

## Findings

A **finding** is an annotation of kind `finding` that cites the `result:` or `query:` it rests on through a `derived_from` link. `polylogue result:01JD7Q4M note 'retries cluster after deploys' --as finding` writes one about the kept result, with that link; `polylogue 'annotations where kind:finding links_to:result:01JD7Q4M'` lists the findings resting on it.

## Pre-registration

A run or kept result is **pre-registered** when a question naming its `query:<hash>` was written before the hash's first run on this archive: a `claim` or `finding` whose `--check` names it ([annotations.md](annotations.md#checked-claims)), or a `saved_query` annotation brought in through `import` of annotation lines. `save` registers its question in the same act as its first run, so a save is not a pre-registration. Replies and kept results carry `pre_registered: true|false` with the registering annotation. Order is the archive's commit order, never a clock; in primary mode, Sinex's commit order.

## Watches

A watch is a saved query that tells you when rows start matching: `save NAME --watch` writes a signal rule (an annotation of kind `signal`) on `query:NAME`, which puts a notice in `status`, in the next briefs and in `polylogue notices` for each row that starts matching, with the line's kept result as the baseline. `--hold 20m` means a row must go on matching that long before it counts; `--notify` also sends a desktop notification. Both options require `--watch`. A rule the options cannot state is written with `note --as signal` on the saved query ([annotations.md](annotations.md#note)). The rule schema, routing and the evaluator's behaviour (changes, baselines, cooldowns, thresholds, catching up, downtime) are [operations.md](operations.md#notices) and the common [signals](https://github.com/Sinity/polylogue-sinex-common/blob/master/crates/signals/README.md) design. What the query side supplies:

- The watch evaluates `query:NAME` again as data arrives, resolving relative times each time, and follows re-saves of the name.
- Changes are reported as `membership` (rows joined or left), `ranking` or `top_k`, never one for another.
- A reading taken while the index catches up, or an incomplete reply, is held as incomplete, never reported as a change.
- Occurrences are keyed by logical session, so a re-parse never fires a watch again and one logical session fires once.
- A watch over annotations compares by kind, target, key and value.
- **A page watch**, `polylogue page:flaky-ci save --watch`, is named after the page and watches `annotations where referenced_by:page:flaky-ci`: its slots and the annotations they show, so a revision counts as a change, and so does a citation on the page whose source changed ([pages.md](pages.md#time-and-checkpoints)).
- A watch only notifies; it never starts or resumes a session ([session-control.md](session-control.md)).
- Withdrawing the saved query withdraws its watch; withdrawing only the `signal` annotation stops the watch and leaves the name.

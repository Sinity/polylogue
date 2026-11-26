# Vocabulary

One name per concept. Code, SQL, CLI help, JSON keys, errors, comments, tests
and docs use these words for these things. Before naming something, look for
it here: use the listed word, or add the new concept here in the same change.
The values of closed sets (kinds, states, link kinds) are listed in the
generated page of the thing they belong to, or on the area page that owns them.

## Stores and stored inputs

- **archive**: `archive.db` and the segment log; what cannot be recomputed.
- **index**: `index.db`; everything parsed and derived, built from the archive.
- **full-text files**: the files holding the index's full-text entries, split by text key, in the directory under the archive root that `index.db` names (`search-<id>/`); they leave with the index file that names them.
- **archive key**: the first 12 hex digits of BLAKE3 over the archive directory's
  resolved path; it names the archive's runtime directory, reader cookie and
  completion cache directory.
- **schema identity**: a short hash of an index file's version, format identity,
  tables, views and triggers.
- **index rebuild**: writing a new index in the private file
  `index-rebuild/build.db`, renamed `ready.db` when it ends and then over
  `index.db`; the first index of an archive, or one replacing an index this
  build cannot use. Its reason is `index-rebuild/reason.json`, written first. It runs in named **steps** (reading stored files, sorting texts,
  writing tables, indexing, assembling sessions, indexing sessions, joining
  sessions across files, finishing indexes, search index, installing), which
  status and the log list by those names. Assembling sessions writes each
  session whose readings one file states; joining sessions across files
  writes, from the index, the sessions whose readings several files state or
  that copies name.
- **index files**: the pair `status` shows of the index: the installed index
  (none, current, or unusable with why) and the rebuild (none, writing with
  its reason, or ready).
- **archive identity**: the random ID `archive.db` gets when it is created,
  which every index of that archive records.
- **embeddings store**: message and annotation vectors for semantic ranking, kept apart
  because vectors cost money or time to make.
- **segment log**: append-only files of compressed frames holding every blob;
  a **segment** is one such file.
- **blob**: bytes stored under their hash.
- **byte home**: where blob bytes are kept: the segment log, or a directory
  of one file per hash. To **put** bytes is to make them durable there and
  known to the archive; to **open** them is to read a range back.
- **input**: one file as Polylogue tracks it: one location's history, or one
  imported file.
- **revision**: one distinct content of an input.
- **chain**: revisions of an input where each one appends to the one before,
  its **previous revision**.
- **extent**: a range of a blob placed at an offset in a chain.
- **chunk**: one content-defined piece of a document its source saves whole,
  stored as a blob of its own so a revision stores only the chunks it changes.
- **observation**: one time the collector found something at a path; a file
  that disappears is an observation, never a deletion.
- **observation kind**: what an observation found: `first`, `appended`,
  `rewritten`, `deleted`, `back` or `imported`. Distinct from the three file
  classes of coverage.
- **container**: an input that holds other inputs as its members: a provider's
  export (ZIP or tar) or one compressed file. A member is an input of its own,
  addressed `container!/member`.
- **observed state**: what one read saw of a file or database (size, inode,
  write-ahead log), used to tell whether it changed.
- **export** and **member**: a provider's data export, stored whole, and one
  file inside it.
- **migration**: a numbered, data-preserving step that brings a store from one
  version to the next.
- **position**: how far a consumer has applied a store or stream; saved in
  the same transaction as what it applied.
- **commit position**: the index's write counter at the transaction that
  committed a row's source; such a row is **committed** at that position.
- **fixed position**: a commit position held as one consistent read point
  while several reads use it.
- **freshness**: how far the index is behind its sources, or in primary mode
  behind Sinex's latest position.

## Where inputs come from

- **origin**: the tool, export or site a file comes from (`claude-code`,
  `chatgpt`, …); each has one **origin declaration**.
- **root**: a tool's home directory that Polylogue scans.
- **location**: a declared path pattern under a root.
- **collected**, **ignored**, **unrecognized**: the three classes every file at
  a known location falls into; ignored files carry a recorded reason.
- **coverage**: per root, how many files are in each class (a pass has its
  own **pass coverage**, below).
- **arrival**: how an origin's files reach Polylogue: a live location, a spool
  or `import`.
- **provider**: the company or site whose model answered.
- **harness**: the agent program that runs a model and writes session files
  (Claude Code, Codex, Gemini CLI).
- **tool**: what a model calls in a session (`Bash`, `exec_command`). The
  programs above are origins or harnesses, never "tools", except in **tool
  database**: a tool's own SQLite file, stored as a **row dump** that keeps
  rows the tool later deleted.
- **spool**: an append-only file Polylogue's own writers fill (hook events,
  browser captures, live events). Hook and browser spools are synced per
  line; live-event appends are not ([live-sessions.md](live-sessions.md#live-state)).
- **sources**: everything the daemon reads inputs from on its own: its roots'
  locations, the spools, tool databases and collectors. `daemon --no-sources`
  reads none of them; an `import` is not a source.
- **hook event**: one event a harness reports to `polylogue internal hook`, spooled whole.
- **process identity**: a harness process's pid, start time and boot ID.
- **capture**: one conversation saved by the browser extension.
- **capture read**: one read of original bytes from a provider location, with
  its byte range, hash and route.
- **provider import**: `import PROVIDER`, the browser extension fetching a web
  chat's conversations the archive lacks or holds older copies of; one
  **import job** per provider, listed in the unit `import_jobs`.
- **chat action**: a turn the extension sends, a new conversation it drafts
  or submits, or a rename it makes in a web chat on a user's request (`send`,
  `start --in chatgpt`, `note --rename`); the unit `chat_actions`.
- **native ID**: the ID an origin itself gives a session or message.
- **content ID**: the ID of a message whose source gives none: its session, a
  hash of its role and content, and its occurrence among equal messages
  (`id_kind` `content`, against `native`).
- **content name**: a content ID without its occurrence number
  (`<session>~<role>-<hash>`); a parser writes a content-named message as its
  content name and its record's position, and the number is settled when the
  chain's reading is formed.
- **response key**: the key a source gives the records of one response written
  as several records (Claude Code's `message.id`); a chain's rows with one
  session and key are one message.
- **alias row**: a record that is not a message but stands in a chain of
  parents, saying which name it stands for.
- **linked material**: a page or file a conversation cites, fetched and stored.
- **Python Polylogue**: the earlier Python implementation; `import` reads
  its vector stores, whose vectors were paid for.

## Parsing

- **parser**: reads one origin's chains into parsed rows; one per origin.
- **parse version**: a parser module's version; changing it re-parses the
  chains that module read.
- **input record**: one entry in an input (a JSON line, an export item, a
  row). Each input record ends **used**, **skipped** (with a reason) or **unreadable**: its
  **record outcome**.
- **record wrapper**: the outer JSON of a harness record around its message.
- **stated**: a value as a source states it, before it is checked (a stated
  source, a stated attachment, a stated parent).
- **parsed end**: the furthest byte a parse of a chain reached.
- **field accounting**: every leaf of every record is read, kept, or counted
  as unused.
- **golden**: expected output of a parser, or of a CLI or MCP reply, written
  by hand from knowledge of the format, never from the code's own output.
- **wire shape**: per origin, path and type, when a field was first and last
  seen and how often.
- **format drift**: new, changed or vanished fields in a provider's format.
- **record**: `archive.db` with the bytes it cites: what was captured and
  authored, which cannot be recomputed. The index is a function of it.
- **chain parse**: what a parser, at its parse version, makes of one chain. It
  is owned by its chain; the **shown rows** of a session are composed from
  the parses that state it. **Reading** is what `read` produces.
- **covered chain**: a chain a newer chain of the same input is known to
  begin with all of; it has no reading and is shown as `repeated`.
- **chain state**: what the index holds of one chain: no parse, parsed to a
  length under a parse version, or failed there with its reason. With what
  the archive says of the chain it decides whether the chain is **due** and
  how it is shown: `parsing`, `repeated`, `no_parser`, `done`, `waiting`,
  `refused` or `failed`.
- **stale part**: a part of a chain's parse that does not start where the
  chain's stored state ends; it is refused and the chain stays due.
- **upkeep** and **build**: keeping the installed index current as the record
  grows, and making an index from nothing; they give the same index.
- **re-parse**: reading a chain's stored bytes again with a newer reading
  version; it is not deletion.
- **user submission**: a user request together with its optional pinned read,
  trusted catalog context, and caller deadline and cancellation.

## Sessions

- **session**, **message**, **block**, **tool call**, **attachment**,
  **document**: the parsed things. A message is stored once, in the session
  that originated it.
- **duplicate message**: a message that a capture repeats from another route
  of the same session; it names the shown version in `duplicate_of` and is
  left out of every population (see the shown version of repeated turns in
  `docs/measures.md`).
- **logical session**: the root of a session's fork, spawn, continuation and
  resume tree; the default unit in listings, counts and measures. Its members
  are **physical sessions**.
- **session link**: an explicit relationship between sessions (`spawn`,
  `fork`, `continuation`, `resume`, `reference`); the unit `session_links`.
- **known parents**: the parent sessions known for a session whose ancestry
  reaches no root yet.
- **compaction boundary**: where a harness replaced earlier context with a
  summary; the summary is a message.
- **`written_by`**: who wrote a message (`human`, `model`, `harness`, `tool`,
  `unknown`). **`prompt_kind`**: what kind of prompt a user-role message is.
- **time quality** and **precision**: how a stored time was obtained and how
  exact it is. Unknown times stay unknown. **Time accuracy** is the pair with
  an optional upper bound, stored beside each source time.
- **paste**: a pasted range of a prompt.
- **sidecar**: a file read into its session, never as a session of its own.
- **gone**: a session its source no longer holds: absent from a complete
  newer export of its account, or removed from the Drive folder. It keeps its
  rows; the session field `source` reads `gone`.
- **session alias**: the ID a copy of a session carries when the copy has no
  native ID (an AI Studio Takeout's content name), recorded by the route that
  knows the session's own ID; the index shows the copy as that session.
- **route**: one way a conversation reached the archive: `live`, `capture`,
  `recorded`, `export`, `copy` or `page_text`. **Route precedence** says, per
  field group, whose value is shown; the reading that supplies the content is
  the message's **main route**.
- **field group**: the fields of a message taken together from one route
  (content, time, model, usage, stop reason, parent, production).
- **message completion**: whether a reading's message is `finished`, `unknown` or
  `in_progress`.
- **order time** and **session order**: the latest source time of a message
  and the messages before it in one input's reading; a message's place in its
  session, from its readings' order times, inputs and positions
  ([sessions.md](sessions.md#messages)).
- **availability**: whether an attachment's bytes are `present`, `unfetched`,
  `failed` or `unavailable_at_source`.
- **byte match**: a saved check that an export member's stored bytes hash to
  an attachment's stated hash.
- **visit start**: the position where a streaming read starts, including
  whether it walks forward or backward.
- **message position**: a message's ordinal in a selection and its row key
  when known.
- **message source**: the selected blocks, commit position and ordinal
  returned beside a message by a sourced visit.

## References and queries

- **reference**: a name for exactly one thing (`session:codex:…`,
  `message:…`, `annotation:…@…`, `query:NAME`, `self`); one parser reads them
  on every surface.
- **reserved words**: `latest` (the most recently active main session
  matching the line, never the caller), `self` (the calling session) and
  `here` (the current repository with its worktrees).
- **selection**: the first argument of a line: a quoted query, a reference, a
  row number, a saved query name, a unit name, a plain word or `-`.
- **query**: the question in quotes, saying which rows. **unit**: what the
  rows are (`sessions`, `messages`, `tool_calls`, `annotations`, …).
  **field**: a property a query filters on (`repo:`, `since:`, `outcome:`).
  **condition**: a field with a value.
- **transform**: a step after `|` inside a query (`group by`, `sort by`,
  `measure`).
- **plain-word search**: a single plain first word, read as a search for that
  word in prose and tool input and output.
- **field selection**: a transform or default that changes which fields a row
  shows, keeping its row identity.
- **question**: `query:<hash>`, the identity of a query's normal form. (The note kind `question`, `note --as question`, is a different thing: an open question someone wrote.)
- **run**: one evaluation of a query (`run:<id>`): its question, read
  position, author and count; it holds no rows.
- **saved query**: a question saved under a name, re-run each time it is used
  (`query:NAME`, or the name as the first word); a **shipped saved query**
  comes with Polylogue.
- **kept result**: the rows of one run, stored and never re-run
  (`result:<id>`, `result:NAME`).
- **watch**: a saved query that tells you when rows start matching
  (`save NAME --watch`).
- **figure**: a saved measure query with a declared denominator query.
- **finding**: an annotation that rests on a result or query.
- **pre-registration**: a claim or finding whose `--check` names a question
  before that question's first run (saving does not pre-register).
- **page size** and **cursor**: replies come in pages; a cursor continues from
  the last row shown. A page of rows is never the reply's limit.
- **deadline**: how long an operation may run before it stops and names a
  narrower query.
- **latency class**: an operation's declared speed budget.
- **logical day**: the day boundary every date means: a named zone and a start
  time (`DaySetting`).
- **facet**: a read-time grouped count.
- **search leg**: the full-text part of a compiled query.
- **semantic ranking**: ordering rows by meaning (`similar:`).
- **fusion**: ranking rows by their places in several ordered lists.
- **warm state**: derived state the daemon keeps between requests, each item
  keyed by what it depends on (`docs/latency.md`).
- **status snapshot**: what `status` reads from the index, read after each
  commit; a status reply **answers from** the snapshot, an index build, or
  nothing read yet (`docs/operations.md`).

## Reading

- **reading**: what `read` produces: rows, view, show and hide, range and
  presentation at one read position, as one typed document.
- **view**: which content a reading shows (`transcript`, `dialogue`,
  `overview`, `context`, `tree`, `changes`, `live`, `raw`, `compare`,
  `contents`, `trace`).
- **trace**: where something came from: the message a quote or paste first
  appeared in, the tool call that last wrote a file line, and the bytes behind
  a record (`read --view trace`), read from the **trace data** sets.
- **seen mark**: where a reader last stopped, per target (`read --from seen`,
  `since:seen`); it advances only when an item was shown to that reader.
- **show** and **hide**: how much of each item kind a reading shows
  (`--show tools:full`).
- **range**: which part of a session a reading shows (`--from`, `--until`,
  `--first`, `--last`, `--around`).
- **indicator**: a one-line marker in place (continues, compacted, branches).
- **inherited message**: a message shown from a parent's copied prefix.
- **value state**: how a shown value stands: known, unknown, partial, inferred.
- **budget**: how many tokens a page of a reading, brief or listing may use;
  what it leaves out is named, with the line that continues.
- **format**: how a reading is encoded (`markdown`, `html`, `jsonl`,
  `citation`, …). **destination**: where the CLI writes it (`--to clipboard`,
  `browser`, a file, a directory).
- **bundle**: what `read --to DIR` writes: typed records that `import` reads
  back.
- **alternate**: another rendering of content (a translation,
  transliteration, simplification or summary) bound to an exact source
  revision, shown in place of the original when enabled and always marked; an
  **alternate part** maps to one original message, block, span or (on a page)
  annotation.
- **blind**: a comparison shown with labels `A`, `B` instead of references and
  with identifying fields removed.
- **as it stood**: `read --at TIME|POSITION`, rendering annotations and
  pages from the revisions current then.
- **neutral names**: `--neutral-names`, labelling the human `user` instead of
  `you` in excerpts shared with others.

## Authored records

- **annotation**: anything people or agents wrote about the archive: notes,
  tags, judgments, comparisons, saved queries, watches, settings; an authored
  statement that a key has a value for what it is about, with evidence and
  scope.
- **kind**: the closed type of an annotation (`note`, `tag`, `decision`,
  `title`, `judgment`, `memory`, `setting`, …).
- **note**: text written about something, of a kind (`note TEXT --as KIND`).
- **tag**: a label, optionally with a value (`rate-limits`, `priority:high`).
- **link role**: how an annotation points at a reference (`about`, `evidence`,
  `scope`, `origin`, a relation such as `replies_to` or `revises`, or, for
  references in its text, `mentions`, `cites` and `embeds`).
- **stack**: annotations about annotations, joined by typed relations.
- **current revision**: a revision that is not withdrawn and has no unwithdrawn
  descendant through `revises`; two current revisions of one record are a
  **conflict**. A write names its **base**, the revisions it revises; one
  naming an outdated base is **stale** and refused.
- **withdrawal**: an authored statement that takes annotations back; nothing
  is deleted.
- **undo line**: the line a write prints that withdraws what it wrote.
- **author**: who wrote a record (`operator`, `agent:<session>`, `agent:<origin>`, `agent`,
  `script:<unit>`, `unattributed`), computed by the daemon. **via**: the
  channel the write arrived on.
- **judgment**: a verdict on one exact revision (`accept`, `reject`,
  `defer`); a **comparison** records which of two or more rows is better on
  one dimension.
- **rubric** and **pass**: the declared schema of a bulk annotation run, and
  the run itself, with its pass coverage (which kept rows have a value or
  abstention) and abstentions; a **rubric value** is one
  rubric field's value for one row.
- **ranking**: an order of rows computed from comparisons
  (`| measure ranking`); **agreement**: how a pass's values match the
  judgments a reader counts (`| measure agreement`).
- **checked claim**: a claim that carries a query checking it.
- **marker**: a `::kind(args)` line in a transcript, read as an annotation by
  the message's author.
- **annotation lines**: a file of JSON annotation writes brought in with
  `import`, checked line by line and written as one request.
- **memory**: annotations an agent's memory tool writes, through the
  **memory-tool adapter**; a **method** is a procedure with checkable
  conditions.
- **setting**: a changeable value from the settings catalog, stored as an
  annotation with its history (`set`).
- **role**: a named kind of agent with a harness and instructions
  (`role:reviewer`); a **role instance** is a session running them.

## Pages

- **page**: a named, ordered arrangement of the archive: sections and slots
  (`page:NAME`); its **subject** is what it is about, and its **layout** is
  `linear`, `board` or `table` ([pages.md](pages.md#page)).
- **slot**: one placed item on a page: text, a heading, an excerpt, an embed,
  a query cell or included text. Never called a block, which is a part of a
  message ([pages.md](pages.md#slot)).
- **section**: a heading slot and what sits under it (`page:NAME#ANCHOR`)
  ([pages.md](pages.md#sections-and-order)).
- **form**: how a slot shows what it holds: `card`, `excerpt`, `text`,
  `full`, `table`, `list`, `count`, `value`, `chart`, `compare`
  (`save page:NAME --as FORM`) ([pages.md](pages.md#forms)).
- **link**: a reference in text that goes there (`[text](REF)`)
  ([page-text.md](page-text.md#links)).
- **citation**: a footnote saying a sentence rests on an exact source passage, carrying
  the quoted words, source and hash (`[^1]`) ([page-text.md](page-text.md#citations)).
- **excerpt**: captured text quoted exactly on a page, with its source line
  ([page-text.md](page-text.md#citations)).
- **embed**: a thing shown on a page, rendered from the archive at read
  (`![caption](REF "form")`) ([page-text.md](page-text.md#embeds)).
- **inclusion**: another annotation's or page section's text shown as part of
  this page, in step with its source ([page-text.md](page-text.md#inclusions)).
- **live cell**: a query on a page, run when the page is read
  ([pages.md](pages.md#queries-on-pages)).
- **kept answer**: a query cell fixed to a kept result
  ([pages.md](pages.md#queries-on-pages)).
- **value**: a number from a query or figure inside a sentence, fixed when
  written unless written `live value` ([page-text.md](page-text.md#values-in-sentences)).
- **cite by quote**: giving a source and a quotation; Polylogue finds and
  stores the exact span ([page-text.md](page-text.md#cite-by-quote)).
- **quote check**: `✓` or `≠`, whether a quoted phrase occurs in the source it
  cites ([page-text.md](page-text.md#the-quote-check)).
- **source changed**: a citation's or excerpt's source text differs from what
  was cited ([page-text.md](page-text.md#when-a-source-changes)).
- **re-anchored**: the cited words were found again after a re-parse moved
  them ([page-text.md](page-text.md#when-a-source-changes)).
- **grounding**: whether a paragraph's statements are cited: `cited`,
  `changed`, `uncited`, `none` ([page-text.md](page-text.md#grounding)).
- **citation needed**: the mark on agent prose with no citation
  ([page-text.md](page-text.md#grounding)).
- **proposed**: text by anyone but the operator that he has not judged, shown
  with accept and reject ([pages.md](pages.md#authorship-and-proposals)).
- **tracked change**: a proposed revision shown as a word diff over the
  accepted text ([pages.md](pages.md#authorship-and-proposals)).
- **backlinks**: everything that places, links, cites or includes an object,
  shown as **Cited and linked** on its reading and as the web's Backlinks
  panel ([page-text.md](page-text.md#backlinks)).
- **active page**: the page a workbench tab is collecting into with `+`
  ([web-pages.md](web-pages.md#citing-while-reading)).
- **working copy**: a page as one Markdown file (`read --to`, or what `edit`
  opens), checked in with `import` ([page-working-copy.md](page-working-copy.md)).
- **generated region**: rows or included text written into a working copy
  for reading, never checked in ([page-working-copy.md](page-working-copy.md#the-format)).
- **red link**: a link to a page that does not exist yet; it attaches when the
  page is created ([pages.md](pages.md#finding-pages)).
- **checkpoint**: a page kept as it stands (`page:NAME save`, giving
  `result:<id>`); its permanent link ([pages.md](pages.md#time-and-checkpoints)).
- **discussion**: notes about a page or its parts that are not placed on it
  ([pages.md](pages.md#discussion)).
- **template**: a page others are made from (`page:template/investigation`),
  with **parameters** bound by `--param`; a **shipped page** (`page:home`,
  `page:index`, `page:pinned`, `page:cost`) comes with Polylogue
  ([pages.md](pages.md#templates-and-shipped-pages)).
- **member**: an alias (`@reviewer`) for a session, optionally on a page and
  with a role. An export's **member** (below, under stored inputs) is a file
  inside it; the context always says which.
- **pin**: putting rows on `page:pinned` (`save page:pinned`).
- **open items** (of a page): the questions, goals and problems it shows that
  are still open, and anything blocked; the **decision box**: the current
  decisions about its subject. Both are computed when read.
- **staleness mark**: `written before 2 findings on this page`, on a note
  older than later findings or decisions on its page; **circle marker**:
  `returned to the position of revision 1`, on a decision back at an earlier
  value.
- **context bundle**: a page from the `bundle` template, the context a
  recipient should read, given with `brief`.

## Agents

- **brief**: packaged context under a budget: alone, for a directory; of a
  session, to take it over (its **takeover**); of search words, what is
  already known (its **recall**); of other rows, what matters in them.
  "Takeover" and "recall" are words help and completion use for two forms of
  `brief`.
- **left-out list**: the closing section of a brief: what is whole,
  shortened, expandable or left out, with the line that reads each.
- **delivery**: the record that a brief, message or notice was handed to a
  session, with its outcome; **used** when the recipient later refers to it.
- **advisory**: a one-line pointer shown before a prompt touches a judged
  target.
- **skill file**: `help --all`, the full reference of every operation,
  installed with the hooks and carried as the brief's first section.
- **live state**: whether a session is `working`, `waiting`, `idle`,
  `stale`, `ended` or `unknown`.
- **send**: deliver text to sessions. **answer**: reply to what a session is
  waiting for (`allow`, `deny` or text). **interrupt**: stop a session's
  current turn. **resume**: reopen a session in its own harness; with `--in`,
  continue its work in a new session of another harness.
- **web chat**: a conversation on ChatGPT, Claude.ai, Grok, Gemini, AI Studio
  or Codex Cloud, captured through the browser extension; chat actions reach
  only ChatGPT, Claude.ai and Grok ([extension.md](extension.md)).
- **held web-chat turn**: a turn Polylogue wrote into an existing web chat,
  waiting for the operator's `answer allow`.
- **peers**: other live sessions in the same repository, or in the same
  working directory when there is no repository identity.
- **task**: a session's open claim in a task tracker.
- **delegation**: a parent session handing work to a child session.

## Derived data

- **derivation**: a product computed from parsed rows, declared with its
  construct, unit, denominator and unknown rule.
- **session summary**: a derivation of one session that is not a column of
  one of its rows (its counts, spans, values, live facts, title, lineage),
  computed from the session's assembled rows by the summary worker for the
  sessions marked due.
- **fold**: a session summary computed over the session's rows in order,
  resumed from a state stored after an earlier row.
- **transcript line**: one layout item a transcript of a session can show
  (a message header, a block, an in-place event), with its token estimates
  and the counts before it; a session summary pages read from.
- **validity kind**: how a derivation's validity is shown: `exact`,
  `parameterized`, `extracted` or `judged`.
- **measure**: a computation a pipeline cannot express, named after `| measure`
  (`usage`, `quota`, `survival`, `contrast`, …).
- **population**: which responses a usage figure counts.
- **money kind**: where a cost comes from (`api_equivalent`,
  `provider_reported`, `subscription_share`, `manual`, `estimated`).
- **clustered time**: the union of active intervals with short gaps closed.
- **party**: whose time a span of a session counts as (`tool`, `model`,
  `human`, `unattributed`).
- **token kind**: a category of tokens a provider counts or charges
  (uncached input, cache read, cache write, output, reasoning).
- **command class**: the extracted class of a normalized shell command, used
  only for analysing agent tool use.
- **fail-then-pass**: a failing run followed by a passing run of the same
  command.
- **open item**: unfinished work in a repository, with the record that
  resolves it.

## The daemon and its surfaces

- **worker**: work following committed changes. It is **running**, **failing** while a transient failure waits for another commit, or **halted** when repeating the work cannot end its failure.

- **daemon**: the one process on the archive, the background service; it
  holds the lock, runs the two writers and serves every client.
- **archive writer**: the one thread that commits `archive.db`'s record:
  captured inputs whose bytes are put, authored records and the request log.
- **index writer**: the one thread that writes `index.db`, from parses of
  bytes the archive has committed.
- **position**: `archive.db`'s write counter after a commit; the rows a commit
  wrote carry it.
- **stop request**: the one request a process makes to stop, carried by the
  archive it opened; every read, wait and reconstructible write observes it.
  A request's **cancellation** (its client hung up, its stream was dropped) is
  a child of it.
- **operation table**: every operation, once; the CLI, socket, HTTP and MCP
  are generated from it.
- **line**: one invocation: a selection and its verbs, or a command, with
  options. **step**: one verb of a line.
- **verb**: a word after the selection that says what to do with the rows;
  **command**: a first word that is not about selected rows (`status`,
  `import`, …).
- **line option**: an option that goes anywhere on a line (`--json`,
  `--deadline`, `--request-id`, `--dry-run`, `--explain`, `--wait`, `--all`).
- **usual rows**: the saved query a verb with no selection picks from
  (`recent` for `read`, `unfinished` for `resume`).
- **reply**: what an operation returns. **next call**: a ready call or line
  offered after a reply, from the verb's **affordance table**.
- **dry run**: `--dry-run`: resolve the selection and report what a line
  would do, doing nothing. **explain**: `--explain`: say how a line is read and
  what it would do, running nothing.
- **failure kind**: a stable name for why an operation failed, mapped to an
  exit code and an HTTP status.
- **cause**: what would end a failure: `transient` (passes by itself),
  `content` (these bytes under this reading version), `request` (the caller
  changes it), `environment` (the machine), `stored_data` (stored data this
  build cannot read back), `bug` or `stopping`. A pass reports each
  **failure** with its subject, cause, code and message; only a `content`
  failure is recorded as a chain's failed parse state.
- **orientation screen**: what bare `polylogue` shows.
- **cheatsheet**: the one screen `polylogue help` prints.
- **status**: figures and a verdict about the daemon and archive.
- **halt state**: work that cannot proceed and is reported, not retried.
- **notice**: a delivery a signal rule fired; an **interrupting** one is also a
  desktop notification. A **shipped rule** is defined in code. A rule's
  **pending deliveries** are those it fired that are not yet delivered.
- **picker**: the embedded chooser with fuzzy matching, a preview and its keys
  in the footer.
- **row number**: `@3`, row 3 of the last list printed on this terminal.
- **completion**: what Tab offers at the cursor, with descriptions.
- **catalog file**: the completion data the daemon publishes, so completion
  works without asking it.
- **scope index**: the daemon's in-memory map of main sessions by the working
  directories they recorded, with repository, activity, stored message count,
  live facts and task; it answers which sessions are under a directory, live
  there or hold a task there without reading the index.
- **change feed**: `dev feed`, the position-based stream of changes programs
  read; the unit `feed`.
- **workbench**: the web UI.
- **host**: the browser extension's native messaging host (`internal
  browser-host`).
- **per-session host**: the process that runs one hosted agent, holds its
  stdio and serves its socket (`internal host`).

## Primary mode

- **mode**: `standalone`, or `primary` on Sinex.
- **material**: Sinex's unit of stored bytes; Polylogue's input revisions
  become materials in primary mode.
- **kind declaration**: the immutable schema and rules of one version of a
  Sinex event type; Polylogue owns its `polylogue.*` declarations.
- **kind version**: the numbered declaration a producer event selects.
- **recall class**: the class a kind declaration gives its events for recall.
- **confirmed stream**: Sinex's ordered log of committed changes. Polylogue's
  reader applies from it what other hosts recorded and what Sinex's
  lifecycle decided. A recreated stream is a new **stream instance**; its
  newest committed position is Sinex's **latest position**.
- **Sinex event** (or **confirmed event**): one of Polylogue's `polylogue.*`
  events as Sinex holds it after confirming it.
- **acknowledgment**: Sinex's answer to a frame Polylogue sent: admitted,
  kinds registered, replayed or refused.
- **holding directory**: in primary mode, where bytes stay on the host until
  Sinex acknowledges them.
- **shipper**: in primary mode, the part that sends the record to Sinex in
  commit order.
- **record offset**: the byte position of a record inside a material.
- **snapshot**: the current state Sinex sends a new consumer before its
  deltas, in bounded pages.
- **completion marker**: the stream message saved with an applied position,
  named on reconnect so delivery resumes after it; the last snapshot page
  carries the first one.
- **write identity**: the identity of a prepared write, saved with it and
  reused on retry.
- **move to Sinex**: the one-way move of a standalone archive to primary mode
  by attaching the archive to Sinex in place, through deployment
  configuration; nothing is imported or parsed again.

## Development

- **Morrow**: the synthetic example archive in `examples/morrow/`, which
  `demo` builds and the examples use.
- **fault point**: a named point where a test can fail, panic, stop or
  **hold** a write or a read; a held point blocks until the test releases it.
- **scenario**: an end-to-end test against the real daemon, CLI and MCP.
- **paired run**: a base and a change benchmarked together on one sample.
- **row struct**: a store row declared with `row!`: its fields are its
  columns, its statements are declared beside it, and its byte form is what
  sorted files hold ([store boundary](architecture.md#store-boundary)).

## Removed words

These words are not part of the surface; each was renamed, and the row names
what says it now. `polylogue help WORD` prints the same line. Words whose job
another line does are in [set-aside.md](set-aside.md#words-another-line-does).

| Word | Now |
|---|---|
| `sx:` | `sinex:<kind>/<id>` |
| `query`, `list` (commands) | the selection alone: `polylogue 'QUERY'` |
| `live` (as a list) | the saved query `running` |
| `follow` | `read --follow` |
| `takeover` (as a command) | `polylogue SESSION brief` |
| `recall` (as a command) | `polylogue 'WORDS' brief` |
| `revise` | `edit` |
| `keep` | `save` with no name; on a page, the checkpoint `page:NAME save` |
| `watch` (as a verb) | `save NAME --watch` |
| `continue` (as a verb) | `resume` |
| `hand-off` | `resume --in HARNESS` |
| `chat send`, `chat approve`, `chat decline`, `chat list` | `send`, `start --in chatgpt`, `answer allow`, `answer deny`; the unit `chat_actions` |
| `backfill` | `import PROVIDER`; the unit `import_jobs` |
| `attachment` (command) | `attachment:… read` |
| `explain` (command) | `--explain`, `help query` |
| `setting`, `settings` | `set` |
| `verify` | `status --verify` |
| `completions` | `install zsh`, `install bash`, `install fish` |
| `sql`, `schema`, `changes` (commands) | `dev sql`, `dev schema`, `dev feed` |
| harness and browser setup | `install TARGET`, `install TARGET --remove` |
| `auth drive` | `install drive` |
| `demo tour` | `demo DIR` |
| `manual` | `help --all` |
| `fetch` | `import PROVIDER` |
| `timeline`, `cost`, `measure NAME` (commands) | queries with `\| measure` |
| `trace` (as a verb) | `read --view trace` |
| `toc` | `read --view contents` |
| `checkout`, `checkin` | `page:NAME edit` in an editor; `read --to DIR` and `import DIR/NAME.md` |
| `add`, `move` (on a page) | `save page:NAME [--section S] [--after SLOT]` |
| `pin` | `save page:pinned` |
| `analyze` (experiments) | `\| measure experiment` |
| `applies` (as a verb) | the field `applies:REF` on methods |
| fold | `--show` and `--hide` |
| `links` (the unit of links between sessions) | `session_links`; `links` is the unit of link rows |
| answer (for a query run) | run: `run:<id>`, the unit `runs` |
| `ranking:[…]`, `insufficient_evidence` (as typed comparison verdicts; the stored value keeps the underscore) | `judge 'B>A>C'`, `judge insufficient-evidence` |
| `--limit`, `--out`, `--output`, `--for-session`, `--print`, `--exec`, `--harness`, `--for-others`, `--page-tokens`, `--output-lines`, `--since seen` | `--page-size`, `--to`, `--for`, `--dry-run`, `--in`, `--neutral-names`, `--budget`, `--show tools:N`, `--from seen` |
| `last` (reference) | `latest` |
| `--format` on `import` | `--origin` |
| `--since` on the change feed | `--after` |
| `no_parser_yet`, `partial` (parse states) | `no_parser`; a chain whose earlier bytes are missing has no state of its own |
| stopped attempt (a failure bound to the executable) | a failed chain state, tried again only when its bytes or parse version change |

# Vocabulary

One name per concept. Code, SQL, CLI help, JSON keys, errors, comments, tests
and docs use these words for these things. Before naming something, look for
it here: use the listed word, or add the new concept here in the same change.
The values of closed sets (kinds, states, link kinds) are listed in the
generated page of the thing they belong to, or on the area page that owns them.

## Stores and stored inputs

- **archive**: `archive.db` and the segment log; what cannot be recomputed.
- **index**: `index.db`; everything parsed and derived, built from the archive.
- **embeddings store**: message vectors for semantic ranking, kept apart
  because vectors cost money or time to make.
- **segment log**: append-only files of compressed frames holding every blob;
  a **segment** is one such file.
- **blob**: bytes stored under their hash.
- **input**: one file as Polylogue tracks it: one location's history, or one
  imported file.
- **revision**: one distinct content of an input.
- **chain**: revisions of an input where each one appends to the one before.
- **extent**: a range of a blob placed at an offset in a chain.
- **observation**: one time the collector found something at a path; a file
  that disappears is an observation, never a deletion.
- **export** and **member**: a provider's data export, stored whole, and one
  file inside it.
- **migration**: a numbered, data-preserving step that brings a store from one
  version to the next.
- **position**: how far a consumer has applied a store or stream; saved in
  the same transaction as what it applied.

## Where inputs come from

- **origin**: the tool, export or site a file comes from (`claude-code`,
  `chatgpt`, …); each has one **origin declaration**.
- **root**: a tool's home directory that Polylogue scans.
- **location**: a declared path pattern under a root.
- **collected**, **ignored**, **unrecognized**: the three classes every file at
  a known location falls into; ignored files carry a recorded reason.
- **coverage**: per root, how many files are in each class.
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
  browser captures), synced per line.
- **hook event**: one event a harness reports to `polylogue hook`, spooled whole.
- **process identity**: a harness process's pid, start time and boot ID.
- **capture**: one conversation saved by the browser extension.
- **backfill**: the extension capturing a web chat's older conversations.
- **chat action**: a turn the extension drafts or sends in a web chat on a
  user's request.
- **native ID**: the ID an origin itself gives a session or message.
- **linked material**: a page or file a conversation cites, fetched and stored.

## Parsing

- **parser**: reads one origin's chains into parsed rows; one per origin.
- **reading version**: a parser module's version; changing it re-parses the
  chains that module read.
- **record**: one entry in an input (a JSON line, an export item, a row). Each
  record ends **used**, **skipped** (with a reason) or **unreadable**: its
  **record outcome**.
- **field accounting**: every leaf of every record is read, kept, or counted
  as unused.
- **golden**: expected parser output written by hand from knowledge of the
  format, never from the parser's own output.
- **wire shape**: per origin, path and type, when a field was first and last
  seen and how often.
- **format drift**: new, changed or vanished fields in a provider's format.
- **versioned row**: every reading of a parsed row is kept; the **shown
  version** is the current one in a total order.
- **re-parse**: reading a chain's stored bytes again with a newer reading
  version; it is not deletion.

## Sessions

- **session**, **message**, **block**, **tool call**, **attachment**,
  **document**: the parsed things. A message is stored once, in the session
  that originated it.
- **logical session**: the root of a session's fork, spawn, continuation and
  resume tree; the default unit in listings, counts and measures. Its members
  are **physical sessions**.
- **session link**: an explicit relationship between sessions (`spawn`,
  `fork`, `continuation`, `resume`, `reference`).
- **compaction boundary**: where a harness replaced earlier context with a
  summary; the summary is a message.
- **`written_by`**: who wrote a message (`human`, `model`, `harness`, `tool`,
  `unknown`). **`prompt_kind`**: what kind of prompt a user-role message is.
- **time quality** and **precision**: how a stored time was obtained and how
  exact it is. Unknown times stay unknown.
- **paste**: a pasted range of a prompt.
- **sidecar**: a file read into its session, never as a session of its own.
- **route**: one way a conversation reached the archive (live file, capture,
  export). **Route precedence** says, per field, whose value is shown.
- **availability**: whether an attachment's bytes are `present`, `unfetched`,
  `failed` or `unavailable_at_source`.

## References and queries

- **reference**: a natural-key name for anything (`session:…`, `message:…`,
  `assertion:…@…`, `query:…`); one parser reads them on every surface.
- **query**, **unit** (what a query lists), **field**, **condition**,
  **stage** (`| …`).
- **query signal**: what makes a first argument a query rather than a command.
- **query object**: `query:<hash>`, the identity of a query's normal form.
- **run**: one recorded execution of a query.
- **saved query**: a name pointing at a query's hash; a **shipped saved query**
  comes with Polylogue.
- **kept result**: `result:<id>`, a query's rows kept as they were.
- **figure**: a saved measure query with a declared denominator query.
- **finding**: an assertion that rests on a result or query.
- **pre-registration**: saving a query before its first run.
- **explanation**: how a query is read and compiled, without running it.
- **page size** and **cursor**: answers come in pages; a cursor continues from
  the last row shown. A page of rows is never the answer's limit.
- **deadline**: how long an operation may run before it stops and names a
  narrower query.
- **latency class**: an operation's declared speed budget.
- **logical day**: the day boundary every date means: a named zone and a start
  time (`DaySetting`).
- **facet**: a read-time grouped count.
- **watch**: a rule that reports changes in a query's rows.
- **search leg**: the full-text part of a compiled query.
- **semantic ranking**: ordering rows by meaning (`similar:`).
- **fusion**: ranking rows by their places in several ordered lists.
- **recall**: the query that answers "what do I already know about this".
- **warm state**: what the daemon keeps in memory between requests, keyed by
  read position.

## Reading

- **read target**: what a reference or query selects for `read`.
- **view**: how a read shows its target (`full`, `dialogue`, `prompts`,
  `actions`, context, links, …).
- **indicator**: a one-line marker in place (continues, compacted, branches).
- **inherited message**: a message shown from a parent's copied prefix.
- **value state**: how a shown value stands: known, unknown, partial, inferred.
- **budget**: the size a read, pack or brief may take, in tokens.
- **bundle**: what `read --to DIR` writes: typed records that `import` reads
  back.
- **variant**: transformed content (translation, transliteration,
  simplification, summary) bound to an exact source revision; a **variant
  part** maps to one original message or block.

## Authored records

- **assertion**: an authored statement that a key has a value for a target,
  with evidence and scope.
- **kind**: the closed type of an assertion (`note`, `tag`, `highlight`,
  `claim`, `decision`, `judgment`, `memory`, `setting`, …).
- **link role**: how an assertion points at a reference (`about`, `evidence`,
  `scope`, `origin`, or a relation such as `answers` or `revises`).
- **stack**: assertions about assertions, joined by typed relations.
- **current revision**: a revision no unwithdrawn `revises` points at; two
  current revisions of one record are a **conflict**. A write naming an
  outdated base is **stale** and refused.
- **withdrawal**: an authored statement that retracts assertions.
- **author**: who wrote a record (`operator`, `agent:<session>`,
  `script:<unit>`, `unattributed`), computed by the daemon. **via**: the
  channel the write arrived on.
- **judgment**: a verdict on one exact revision; a **comparison** judges two or
  more references on one dimension.
- **rubric** and **pass**: the declared schema of a bulk annotation run, and
  the run itself, with its coverage and abstentions.
- **checked claim**: a claim that carries a query checking it.
- **marker**: a `::kind(args)` line in a transcript, read as an assertion by
  the message's author.
- **memory**: assertions an agent's memory tool writes, through the
  **memory-tool adapter**; a **method** is a procedure with checkable
  conditions.
- **setting**: a changeable value from the settings catalog, stored as an
  assertion with its history.

## Pages

- **page**: a container of slots, rendered in a **layout** (linear, board,
  wiki, table).
- **slot**: one placed reference, live query, heading or nested page; it shows
  the current revision (**follow**) or one exact revision (**fixed**).
- **member**: a session that belongs to a page under an alias.
- **role**: reusable instructions plus a harness; a **role instance** is a
  session running them.
- **toc**: a page's generated table of contents, an agent's starting point.
- **check-out** and **check-in**: editing a page as files, refused on a stale
  base.
- **template**: a page whose headings are copied into new pages
  (**investigation**, **experiment**).
- **checkpoint**: a page revision with every slot fixed.
- **ping**: an explicit address plus text, delivered to a session.

## Agents

- **pack**: prepared context for a recipient and purpose, built under a
  budget by one pack builder; **brief**, **takeover** and **recall** are named
  packs.
- **entry map**: the closing section of a pack: what is whole, shortened,
  expandable or left out.
- **delivery**: one hand-over of prepared content into a model's context, with
  its outcome; **used** when the recipient later refers to it.
- **advisory**: a one-line pointer shown before a prompt touches a judged
  target.
- **agent manual**: the generated description of every operation, installed
  with the hooks and carried as the brief's first section.
- **live state**: whether a session is `working`, `waiting`, `idle`,
  `stale`, `ended` or `unknown`.
- **peers**: other live sessions in the same repository or worktree.
- **task**: a session's open claim in a task tracker.
- **delegation**: a parent session handing work to a child session.

## Derived data

- **derivation**: a product computed from parsed rows, declared with its
  construct, unit, denominator and unknown rule.
- **validity kind**: how a derivation's validity is shown: `exact`,
  `parameterized`, `extracted` or `judged`.
- **measure**: a named computation over derived data (cost, quota, time,
  survival, compare).
- **population**: which responses a usage figure counts.
- **money kind**: where a cost comes from (`api_equivalent`,
  `provider_reported`, `subscription_share`, `manual`, `estimated`).
- **clustered time**: the union of active intervals with short gaps closed.
- **command class**: the extracted class of a normalized shell command, used
  only for analysing agent tool use.
- **fail-then-pass**: a failing run followed by a passing run of the same
  command.
- **open item**: unfinished work in a repository, with the evidence that
  resolves it.

## The daemon and its surfaces

- **daemon**: the one process on the archive; it holds the lock, runs the
  writer and serves every client.
- **writer**: the one thread that appends and commits.
- **command table**: every operation, once; the CLI, socket, HTTP and MCP are
  generated from it.
- **failure kind**: a stable name for why an operation failed, mapped to an
  exit code and an HTTP status.
- **orientation**: what bare `polylogue` prints.
- **status**: figures and a verdict about the daemon and archive.
- **halt state**: work that cannot proceed and is reported, not retried.
- **notice**: a delivery a signal rule fired; an **interrupting** one is also a
  desktop notification. A **shipped rule** is defined in code.
- **picker**: the embedded fuzzy selector; `@N` names row N of the last list.
- **catalog file**: the completion data the daemon publishes.
- **workbench**: the web UI.
- **host**: the browser extension's native messaging host.

## Primary mode

- **mode**: `standalone`, or `primary` on Sinex.
- **material**: Sinex's unit of stored bytes; Polylogue's input revisions
  become materials in primary mode.
- **kind declaration**: the immutable schema and rules of one version of a
  Sinex event type; Polylogue owns its `polylogue.*` declarations.
- **kind version**: the exact numbered declaration a producer event selects.
- **confirmed stream**: Sinex's ordered stream of committed events, which
  Polylogue applies to `index.db`.
- **write identity**: the durable identity of a prepared write, reused on retry.
- **cutover**: the one-way move of a standalone archive to primary mode.

## Development

- **Morrow**: the synthetic example archive in `examples/morrow/`.
- **law**: a property a page states and tests check, named `<page>/<key>`.
- **fault point**: a named point where a test can stop a write.
- **scenario**: an end-to-end test against the real daemon, CLI and MCP.
- **paired run**: a base and a change benchmarked together on one sample.

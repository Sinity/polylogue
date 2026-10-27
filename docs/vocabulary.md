# Vocabulary

One name per concept. Code, SQL, CLI help, JSON keys, errors, comments, tests
and docs use these words and no others for the same thing. Before naming
something, look for it here: use the listed word, or add the new one here in
the same change.

## Stored data

- **archive**: `archive.db` and `blobs/`, what cannot be recomputed. The
  connection to it is the archive connection.
- **migration**: a numbered SQL step that brings `archive.db` from one
  version to the next; opening an older archive applies the ones it lacks.
- **index**: `index.db`, everything parsed and derived. Rebuilt from the
  archive when deleted or when `INDEX_VERSION` changes. Each build is an
  **index generation**; the change feed's positions name one.
- **embeddings**: `embeddings.db`, purchased vectors, kept apart from the
  index because they cost money to make.
- **blob**: bytes stored by their BLAKE3 hash, inline in `archive.db` when
  small, otherwise as a file under `blobs/`. Its row's state is `writing`,
  `present` or `deleting`; the file is written only after the row exists.
- **input**: one file as Polylogue tracks it: one location's history, or one
  imported file.
- **revision**: one distinct content of an input.
- **chain**: revisions of an input where each one appends to the one before.
- **snapshot revision**: a revision that holds a whole copy, such as a
  tool database's rows at one moment, instead of appending to a chain.
- **extent**: a range of a blob placed at an offset in a chain.
- **appended bytes**: the bytes one revision step adds to its chain, stored
  as one blob.
- **observation**: one time the collector found something at a path.
- **machine**: a computer, by `/etc/machine-id`; its host name is a label.
- **export ZIP** and **member**: an imported ZIP, stored whole, and one file
  inside it (`observation_zip`). A file found under an imported directory is
  just a file.
- **import**: one `polylogue import` run; the files it stored share its ID.
- **extracted blob**: bytes decoded from inside a record, such as inline
  media, stored with the record's byte range. **Inline** means only a blob
  kept inside `archive.db`.
- **write counter** (`commit_seq`): incremented once per archive transaction;
  orders the change feed.
- **stored**: kept in the archive. Prefer no word at all: an input is stored
  by definition.

## Where inputs come from

- **origin**: the tool, export or page a file comes from (`claude-code`,
  `chatgpt`, ...). Each has one **origin declaration** (`OriginSpec`) holding
  everything Polylogue knows about it.
- **arrival**: how an origin's files reach Polylogue: `live` (a location the
  tool writes), `spool` (files Polylogue's own writers put down for the loop)
  or `import`. When one session arrives several ways, the live one is shown.
- **provider**: the company or site whose model answered (Anthropic,
  OpenAI, chatgpt.com).
- **harness**: the agent program that runs a model and writes session files
  (Claude Code, Codex, Gemini CLI). A session's `harness` names it.
- **tool**: what a model calls in a session (`Bash`, `exec_command`). The
  programs above are named by origin or harness, never "tool", except in
  **tool database**.
- **native ID**: the ID an origin itself gives a session or message. Chrome's
  **native messaging** is the only other use of "native".
- **web chat**: a provider's chat pages, which the browser extension captures
  from. **Chat action**: a turn the operator asks the extension to draft or
  submit in one.
- **collector**: code that fetches an origin's files into its spool (the AI
  Studio Drive collector).
- **spool**: a directory Polylogue's own writers fill and the loop reads:
  the hook spool, the browser capture spool, the collectors' spools.
- **tool database**: a tool's own SQLite file. Polylogue stores its rows as a
  **row dump** (JSON lines) up to the **synced prefix** named by the tool's
  checkpoint, and parses the dump.

## Builds and prepared writes

- **first build**: building an empty index from all available inputs. The
  writer delays search entries and secondary indexes until the build ends.
  **Bulk mode** means this first-build write mode, with deferred indexes and
  syncing; it ends by making the index durable. Prefer "first build".
- **prepared batch** (`store::staging::PreparedBatch`): parsed rows together
  with private files of new bytes, ready for the writer to store. The
  `store::staging` module prepares bytes off the writer thread; its
  **stored input** (`StoredInput`) identifies where a committed batch's
  bytes can be read, including their chain and length.
- **Morrow**: the synthetic example archive in `examples/morrow/`, laid out
  by Testkit and read through the real loop in tests.

## Reading sources

- **format**: which parser reads an input.
- **parser**: turns an input's bytes into sessions, messages, documents and
  events. Its output is **parsed rows**. What it remembers between records
  is its **parser state**.
- **record**: one entry in an input: a JSON line, an export item, a database
  row. Every record has a **byte range** in its chain. Each record ends
  **used** (it became rows), **skipped** (understood and set aside, with a
  reason) or **unreadable** (not a record the parser can read; its bytes stay
  stored). The **record counts** add up to the records read.
- **capture**: one conversation saved by the browser extension, appended to
  the provider's **capture log** as a frame: a header holding its length and
  BLAKE3, then its bytes. A **backfill job** captures a web chat's older
  conversations; the extension holds it by a **claim** that expires, and a
  chat action is handed out the same way.
- **diagnostic**: a note from a parser about a record it could not fully
  read, with a `reason`.
- **usage report** and **quota report**: token use or subscription quota a
  tool reported without naming the response it belongs to. Each carries its
  own `report_id`, shared by every copy.

## What the index holds

- **session**, **message**, **block**, **tool call**, **event**, **link**,
  **document**, **attachment**, **turn**, **work graph**: the parsed things. A
  message seen in several inputs is stored once, with one **shown version**.
- **duplicate**: a message another route of its session holds under a
  different ID, matched by exact evidence; the session shows the other one.
- **parents**: the sessions a session continues or was forked from, whose
  messages a read `--with-parents` shows first.
- **record position**: a message's place among the messages its record
  holds; it orders messages that share an input and byte offset.
- **derivation**: a product computed from parsed rows, with a rule version
  (cost, the work graph, session facts, measures).

- **parsed** (index table): links each input chain to the sessions its
  parser produced, so a session can be rebuilt from all its inputs.
- **unused** (index table): counts skipped records, unreadable records and
  fields the parser did not read, per input chain and format.
- **run by script** (`run_by_script`): a command run inside a script call
  rather than through its own tool call. **Counts as action**
  (`counts_as_action`) excludes a script call from action counts when its
  commands are already counted separately.

## What people and agents add

- **note** (with **tag**, **title** and the other note kinds): a note's
  revisions share its **first note** (`first_note`); a **withdrawal** takes
  them all out.
- **kept result**: a query's rows kept as they were when it ran, so a note or
  briefing can cite them.
- **setting**: a value the operator or an agent sets, kept with its history.
- **delivery**: exactly what was put into a model's context.
- **check run**: one `xtask` check or test run, as devkit's machine-wide
  run history records it with its checkout and outcome.
- **brief**: the session-start context for a directory; it fits its
  **budget** (`briefing.budget`, tokens).
- **judgment**: the operator's `accepted` or `rejected` on one revision of
  an agent's note, itself a note of kind `judgment`.
- **notice**: a delivery a signal rule fired; an interrupting one is also
  a desktop notification.
- **client request**: one write request, kept whole so a retry does not
  repeat it and so it can be replayed.
- **user state**: every client request and excision, in commit order: what
  `export user-state` writes and `import` replays into another archive.
- **chat action**: see "Where inputs come from".
- **excision**: stored inputs removed on purpose, with everything parsed from
  them. The request is recorded once with its **target** (an input, a session or a
  document), with the identities it removed (**excised**); a stop
  part way needs no recovery.

## The daemon

- **daemon**: owns the archive lock and serves clients over a socket.
- **loop**: compares what exists with what should exist and does the
  difference: collect, store, parse, derive. One round of it is a **pass**.
- **writer**: the one thread that writes the databases.
- **read pool**, **reader**, **reader slot**: readers see one SQLite read
  transaction each; a slot bounds how many run at once.
- **write lock**: what a register commit holds so readers start from a
  consistent pair of databases.
- **command table**: every command, once, for the CLI, MCP and HTTP.
- **warm state**: what the daemon keeps between requests so answers need
  no reload: loaded vectors, query vectors, recent answers, completion data.
- **mode**: `standalone`, or `primary` with Sinex holding the archive.
- **register**: the archive's rows for inputs and for what people and agents
  wrote: `archive.db` in standalone mode, `sinex.db` (a copy of what Sinex
  holds) in primary mode.
- **material**: Sinex's word for bytes it stores for an event.
- **part**: in primary mode, one step of an input's appended bytes as a
  Sinex material, with the register rows that place it.
- **register material**: in primary mode, a small Sinex material holding a
  change to the register that is not input bytes: a look at an input that
  found no new bytes, a request, or an excision.

## The browser extension

- **host**: the native messaging host, `polylogue browser-host`, which Chrome
  starts for the extension.
- **outbox**: captures the extension keeps until the host has stored them;
  the only thing it keeps.
- **page script**: the extension's script in a chat page, which fetches the
  conversation, the provider's conversation list and files with the page's
  sign-in.
- **list position**: where a backfill job got to in the provider's
  conversation list, in the provider's own terms.
- **in-page button**: the Polylogue button on a chat page, with its panel.
- **extension instance**: one browser profile's extension, as claims name it.

## Offline schemas

- **schema artifact**: an offline report of observed field paths, types,
  presence and counts for an explicit input/revision selection.
- **schema observer**: the pure Sources reader that counts structure without
  storing observed values; dynamic map keys share the `*` path.
- **schema diff**: added and removed paths, changed types and changed
  requiredness between explicit schema artifacts.
- **named field**: a field name forced into schema paths even when the dynamic-key
  heuristic would group it under `*`.
- **sample**: explicit inputs and revisions selected for schema observation,
  with no claim of full coverage of an origin.

## Queries and output

- **logical day**: the day in `day.zone` (the machine's time zone unless
  set), starting at `day.logical_start` (default `04:00`). Dates, `today`
  and `group by day` use these boundaries rather than midnight.

- **query**, **unit** (what a query lists: sessions, messages, ...),
  **field**, **condition**, **measure**.
- **explanation**: a query's parsed conditions, units and compiled shape,
  without executing it; `--why` instead counts rows for its conditions.
- **tour**: ordinary commands over a separate synthetic Morrow archive,
  with their explanations and output in a transcript.
- **saved query** (`query:NAME`) and its **parameters** (`$word` in its
  text, given as `query:NAME(word=VALUE)`); the note revision a name
  resolved to is its **definition**.
- **combining stage**: `| union (…)`, `| intersect (…)`, `| except (…)` or
  `| fuse (…)`, with another query of the same unit.
- **reciprocal rank fusion**: ranking rows by their places in several
  ordered lists; `found_in` names the lists that ranked a row.
- **page** and **cursor**: a page ends with a cursor holding the last key
  shown; the next page reads the rows after it.
- **block slice**: one page's character range of a masked projected block,
  preserving the original message identity and source block index.
- **outline**: paged prompt, tool failure, compaction and branch references,
  selected without reading message bodies.
- **read start**: a message anchor or the latest selected message, resolved by
  the daemon before issuing continuation cursors.
- **view**: how a session is shown: `full`, `dialogue`, `prompts`,
  `actions`.
- **read transaction**: what one answer read; a kept result records it as
  `read_at`.
- **page end**: the masking state at the end of a page, cached so the next
  page need not read the messages before it again.
- **parent row**: in `| matches`, the row a matching child belongs to
  (`parent_id`).
- **budget**: the size a page for a model, or a brief, may take.
- **masking**: hiding credentials in output; stored bytes never change.
- **change feed** and **position**: `changes --since POSITION`.

## Words not used

| Not | Use |
|---|---|
| retain, retained, retention | stored, kept, or nothing |
| custody, reservation, reserve (bytes) | pending blob row |
| publication, publish | commit, make visible, send to Sinex, write lock |
| admission, admit | insert, reader slot |
| retire, retirement | delete, replace |
| canonical | archive (connection), sorted (JSON), exact |
| candidate | match, possible |
| interpretation | format, parsed rows |
| backing, backend | mode |
| unfinished | pending |
| legacy, Archive*N*, old*N* | older (for an outside format), or delete |
| basis | source bytes |
| control, proof (tests) | test |
| qualify, qualification | test, or nothing |
| lease | claim |
| witness, authority, closure, frozen, donor | say what it is |
| snapshot (for reads or captures) | read transaction, capture |
| carry | parser state |
| acquisition | import |
| route (how an input arrives) | arrival |
| piece | appended bytes |
| companion, carrier | read with its main input, hook spool file |
| framed, disposition, materialized | records, record outcome, used |
| preferred (version) | shown |
| container (ZIP) | export ZIP |
| logical, lineage, ancestry, inherited (sessions) | parents |
| convergence, converge | loop, pass |
| law (for a test) | test, property |
| degraded | incomplete |
| world (tests) | test archive, test daemon |
| owner (query rows) | parent row |
| checkpoint (read pages) | page end |
| logical export | row dump |
| receipt (search) | search summary |
| receiver (of browser captures) | host |
| envelope, bundle (captures) | capture |
| staging, staged (in the extension) | outbox, or nothing |
| freshness | changed, captured again |
| fidelity | complete, or say what is missing |
| invocation, settle, settlement | call, finished |
| ledger, mission control, intelligence, ambient surface | the popup, the in-page button, or say what it is |

Excision, epoch and sealed are kept. **Receipt** is only the provider's
record of a chat action's result, as the extension saw it.

## Session measures

- **facet**: a value counted once per selected or global session; **IDF** is
  the natural logarithm of global session count divided by its frequency.
- **cost outlook**: an estimate over explicitly configured billing terms and
  observed usage, with units, method and coverage shown.
- **pacing**: observed gaps in source message order. **Burstiness** is the
  Goh–Barabási coefficient of consecutive user-message gaps.
- **session profile**: recorded session facts together with named timing and
  classification calculations, each with its rule version.
- **message-clustered time** (`message_clustered_duration_ms`): summed intervals within
  a five-minute gap threshold. **Tool-active time**: summed timestamped paired
  tool-call/result intervals. Neither measures attention.
- **workflow shape**: the threshold label over observed tool/message counts.
  **final observed state** (`final_observed_state`): the last structural activity,
  without a task-success judgment.

- **unpaired tools past the gap** (`unpaired_tools_past_gap`): tool starts without
  a paired result more than five minutes before the last observed time.

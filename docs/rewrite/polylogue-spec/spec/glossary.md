# Glossary

The terms this spec uses in a specific sense. Everything else means what it
means in ordinary programming. Each entry points to the file that owns the
rule; a new noun in code, logs or errors needs an entry here in the same
change.

## Data

| Term | Meaning |
|---|---|
| **input** | One logical source file as Polylogue keeps it: one location's history (a session log, a spool file, a database) or one imported file. Every distinct content it had is a revision, all kept. Its ID is the hex BLAKE3 of its bytes for an imported file that cannot change, otherwise a UUIDv7 (`storage/blobs.md`). |
| **piece** | The bytes one append added to a growing file, stored as one blob; small pieces are joined once the file is quiet. |
| **revision** | One distinct content of an input, kept for good; its bytes are given by extents. |
| **revision chain** | The revisions of one input that each extend the one before (a prefix of the next) and share one set of extents; a rewrite starts a new chain, and so does a head rewrite in a format with a mutable first record, whose chain reuses the previous one's blob ranges and subsumes it (`storage/blobs.md`). Not "line", which in this spec means only a line of a JSONL file. |
| **record location** | Where a record's bytes are: input, revision chain and byte range. The citation unit; offsets alone are not enough, because a rewrite's new chain puts other bytes at the same offsets (`model.md`, kinds of sameness). |
| **provider ID / provisional ID** | A message ID built from the provider's own ID, or one Polylogue computed because the provider gave none (fallback IDs, capture turn IDs). Provisional IDs are stable across re-parsing and arrival order but can renumber when a provider edit drops an earlier duplicate (`model.md`, Identity). |
| **preferred version** | The version of a message the merged session shows and counts: finished before in progress, then the most faithful route, then the most complete, then the hash tie-break. An archival policy, not a claim about the provider's latest state; the other versions stay readable (`model.md`, Identity). |
| **quiet time** | How long a growing file must go without changes before it counts as settled (15 minutes): then its small pieces are joined, its full hash is checked, and a spool file may be deleted. Every storage rule that depends on it refers to this name (`storage/blobs.md`). |
| **live window** | How recent hook evidence or file growth must be for a session to count as live (15 minutes; `model.md`, Live state). A separate constant from the quiet time, though equal today. |
| **extent** | A range of a blob placed at an offset in a revision: how appends, rewrites (often stored as deltas) and database page patches are all described (`storage/blobs.md`). |
| **observation** | A recorded change seen at a file's location (appended, rewritten, renamed, deleted, …), with the file's full metadata. |
| **blob** | Bytes named by the BLAKE3 of their uncompressed content; written `blake3:<hex>` in exports (`storage/blobs.md`). |
| **change classification** | Deciding what happened to a file since it was last observed: unchanged (by device, inode, size and nanosecond times), appended (sampled 4 KiB windows), replaced, rewritten, renamed or deleted, with a full hash check when the file goes quiet (`storage/blobs.md`, observing a file). |
| **companion file** | A file parsed together with its main file, such as Claude Code's `tool-results/*.txt` and a subagent's `.meta.json` (`convergence.md`; per provider in `sources/`). Claude Code's `tool-results/*.txt` companions are called side files. |
| **carried state** | What the parser keeps between appends to a growing file (open tool calls, record-to-message map, seen IDs), stored with the read position so an append is parsed alone; the Rust type is `Carry` (`sources/README.md`, `storage/index-db.md`). |
| **spool** | A directory Polylogue owns where the hook command and the browser host drop files for the daemon to collect (`sources/README.md`). |
| **origin** | Which product a session came from: `claude-code`, `codex`, `chatgpt`, … (`model.md`). |
| **provider format** | The file format of an input (`claude-code-jsonl`, `codex-rollout`, `chatgpt-export`, …). One origin can have several formats, and a format can have several vintages: dated layouts the parser tells apart from the first record (`sources/`). |
| **route** | The way an input reached Polylogue: a live provider file, a native API capture, an account export, a page capture. When routes disagree about a record, the more faithful one (the one that preserves more of what the provider had) wins (`sources/README.md`, "The same conversation through several routes"). |
| **session** | One conversation or agent run, as the provider understands it. |
| **message** | One contribution by one author inside a session: a prompt, a model response, a tool result. One model response is one message even when the provider writes it as several records (`model.md`). |
| **block** | One piece of a message's content: text, thinking, a tool call, a tool result, an attachment. |
| **written_by** | Who actually produced a message's content: `human`, `model`, `harness` or `unknown`, independent of its role (`model.md`, Message). |
| **human prompt** | A message `written_by: human` with `prompt_kind: prose`; typed slash commands and shell or memory input are `command` and counted apart (`model.md`, Message). |
| **request-bearing message** | A message produced by a model request of the session itself (Claude Code assistant records that are not `<synthetic>`, Codex model requests, …; each provider's section names the record forms). Messages `written_by: model` that arrive as input (a subagent's instruction, a peer or coordinator message) are not request-bearing. Only these have usage; "unknown usage" and "unpriced" count only them, and other messages' lack of usage is not unknown (`analysis.md` §1). |
| **harness** | An agent program that runs a model in a loop with tools: Claude Code, Codex, Gemini CLI, Hermes. |
| **tool outcome** | `ok`, `error` or `unknown`, with a reason code from the list in `model.md`. A call with no result is stored `unknown` (`no_result`) and read as `pending` while its session is live. In code `ToolOutcome`. |
| **result outcome** | A query result's status: `ok`, `empty` (a read that succeeded and matched nothing), `degraded` (some part could not be read, named) or `error`; never `empty` for a failed read (rule R2). The result's `outcome` field; in code `ResultOutcome` (`query.md`, Results). |
| **event** | Something in a session's timeline that is not a message: compaction, model switch, permission decision, hook event (`model.md`). |
| **compaction** | A harness replacing the conversation so far with a summary to free context space. Stored as an event with the summary, never as a message. **Microcompaction** removes named tool results or attachments from context; also an event. |
| **unrecognized** | A record type or field a parser does not understand, counted per input; never dropped. Record types deliberately kept only as counts are listed beside them (`storage/index-db.md`). |

## Lineage

| Term | Meaning |
|---|---|
| **link** | A relationship between two sessions: `subagent`, `continuation` or `reference`, each with a `how` (`model.md`, Link). |
| **pending link** | A link one of whose ends is not in the archive yet: a parent not collected yet or deleted by the provider first, or a child a parent names before the child's own file arrives. It is stored with the missing end's ID (`parent_id`, `child_id`) and resolves when that session arrives. |
| **inherited message** | A message a session's file copies from another session (a resume, a fork, a subagent started with a copy). Marked `inherited_from` (by the writer from the record's own evidence, by the derive step by matching; `decisions/lineage.md`) and never counted, costed or searched twice: counting follows `counted_as`, which points at the source when it is stored and, while it is not, at the one copy the counting rule chooses (`decisions/lineage.md`). "Copy" in the spec means the same. |
| **own** | Of a session's messages or cost: counted in this session (`counted_as` null): a message that is not a copy, or a copy counted here while its source is absent. "Inherited" means having `inherited_from`, so a copy can be both inherited and own; `session_facts.inherited_messages` counts the copies not counted here. |
| **lineage marks** | The `inherited_from` marks and resolved links, computed when a session is written and again when a parent arrives late (`decisions/lineage.md`). |
| **sidechain** | An older Claude Code layout where a subagent's records sit inside the parent's file, marked `isSidechain`, usually without `agentId` (`sources/claude-code.md`). |
| **spawning block** | The parent's tool-call block that started a subagent; stored as `link.spawned_by` (`model.md`, Link). |
| **tree cost** | A session's own cost plus the own cost of every subagent below it, each counted once (`analysis.md`). |
| **delegation card** | The rendered summary of a subagent call as the parent saw it: the instruction, the child's result, what the parent did next (`rendering.md`). |

## Keeping the archive current

| Term | Meaning |
|---|---|
| **convergence loop** | The daemon's single loop: find what is out of date and redo it (`convergence.md`). |
| **the writer** | The one thread in the daemon that writes all three databases: parsed sessions into `index.db` in bounded session steps, and notes, settings, imports and excisions into `archive.db` (`convergence.md`, `storage/README.md`). |
| **derived data** | Everything in `index.db`: recomputable from inputs and notes. (`embeddings.db` is filled by the loop too, but its vectors were paid for and are kept, not recomputed.) |
| **derived product** | One kind of derived data with its own version number, such as session facts (counts, tokens, cost) or lineage marks; bumping the version recomputes only that product (`convergence.md`). |
| **parser version** | A number per provider format; bumping it re-parses that format's inputs. |
| **excision** | Permanent, deliberate removal of content from everywhere Polylogue keeps it, recorded so it never comes back (`storage/archive-db.md`). |
| **generation** | A random ID chosen when `index.db` is created. With the write counter it lets anything following the change feed notice a rebuild (`storage/index-db.md`). The only meaning: Python also used the word for index directories and for one ChatGPT reply's nodes; the rewrite says "reply" there. |
| **step, round** | Scheduling words: the loop takes one bounded step per input per round (`convergence.md`, fair stepping). Not "turn", which means a prompt and its reply, and not "pass", which means an annotation pass. |
| **view** | The consistent state one query reads (index generation and counter, `archive.db` write counter, highest `embedded.id`; `storage/README.md`, one view per read). Display choices are "rendering views" (`rendering.md`) and SQL views are "SQL views"; the bare word means the first. The CLI option `--view` and the MCP `view` argument name a rendering view; that is the one exception, and a result reports the rendering it used as `rendered_as`, never `view`. |
| **warm state** | What the daemon keeps in memory to answer fast: compiled query plans, recent results and paging anchors, embedding vectors; keyed by the view, never a source of truth (`behavior.md`, warm state). |
| **session facts** | The derived product holding per-session counts, tokens, cost, display title and tree cost (`storage/index-db.md`, `session_facts`). |
| **request ID** | The UUIDv7 a client puts on every write request, reused on retry; the written rows take their IDs from it, and `client_request` keeps it with the request's content hash so a resend writes nothing and a reused ID with other content is refused (`decisions/process.md`, writes carry their own ID). |
| **change feed** | `polylogue changes --since GEN:COUNTER:ARCHIVE`: sessions changed, notes added and excisions since a point, for outside consumers (`storage/index-db.md`, `surfaces/cli.md`). |
| **Sinex mode: off, primary** | `off` (the default): the local archive is authoritative and performs no Sinex work; in *primary* Sinex is the source of truth (`decisions/sinex.md`). **Backed mode** means primary. Shadow qualification is a finite comparison run, not a third mode. A standalone archive still uses its Polylogue daemon; the CLI never opens archive files directly (`decisions/process.md`). |
| **local order** | `local_seq`: this archive's write counter at the transaction that inserted a note, excision or revision; assigned locally, never copied in a union, never a key. Sinex's replicated frontier is stated in it (`storage/archive-db.md`, local order; joint #29). |
| **pending area** | In `primary`, the only local copy of source bytes: inputs Sinex has not yet confirmed, each deleted once `polylogue.have` answers complete (`decisions/sinex.md`; joint #27). |
| **placement** | Sending an input to Sinex on the same host by writing its blob file into Sinex's incoming directory and announcing its digest, instead of copying it through NATS (`decisions/sinex.md`; joint #27). |
| **publish step** | The loop step that sends Sinex the inputs, archive rows and excisions it does not yet hold (`convergence.md`). |

## Notes and agents

| Term | Meaning |
|---|---|
| **note** | Anything a person or agent adds: tag, note, title, decision, claim, judgment, saved query, page (`user-data.md`). |
| **marker** | A `::kind(args): body` line an agent or the operator writes inside a message, turned into a note on ingest (`user-data.md`). |
| **reference** | A stable address of something in the archive: `session:…`, `message:…`, `block:…`, `note:…`, `query:…`, `result:…`, `page:…`. Also the third link kind, for a session that names another. |
| **kept result** | A query result saved as it was at a moment, so citations to it do not change when the archive does. |
| **briefing** | A compact document with references, injected into an agent's session by a hook. A *directory briefing* covers recent work, decisions and open questions where the agent starts; a *takeover briefing* covers one named session the agent is picking up (`user-data.md`). |
| **ending** | How a session's end looks from its records: `open` (a question asked, a call unanswered, a failure unresolved) or `clean`, with the reason named (`user-data.md`, resume). |
| **page** | (later) A named, ordered collection of references, notes and live queries that the operator and agents work on together. A page of query results is a **result page**, always called that. |
| **result page** | How much of a query's answer travels in one response: 20 rows for sessions and 50 for every other unit and for grouped rows by default, set by `page_size` (CLI `--page-size`), continued by cursor; never a selection, so `count` covers every row (`query.md`, four bounds). Not a **page**, the workbench object. |
| **ping** | (later) A message sent from a page to a specific agent session, which Polylogue delivers by resuming that session. |
| **watch** | A saved query re-evaluated as data arrives against a kept result, reporting which of its candidates, ranking, selection or page changed (`user-data.md`, watched queries). "Watched query" means the same. |
| **signal** | (later) A watch with a delivery rule (`fires`, `for`, `deliver`, `interrupt`, quiet hours, rate limit), in the rule shape both projects share; each project signals only over its own facts (`user-data.md`, signals; `polylogue-sinex-joint/prototype/signals/`). Elsewhere the word keeps its operating-system sense; an ending's evidence is its reason. |
| **distillate** | A note kind: a model-written standalone version of a session or message range, targeting the exact messages it covers; a read shows it stale once the session grows past them or a covered message changes, and a pass under its rubric writes the next revision (`user-data.md`, note kinds). |
| **investigation** | (later) Work on one question in pages, built from what exists: saved queries for its questions and measures, kept results for what they returned, claims and judgments for findings, deliveries for the context given; no table of its own (`user-data.md`, pages, workspaces and investigations). |
| **workbench** | (later) The shared working surface of pages, investigations and pings (slice 12, `plan/slices.md`). |
| **delivery** | One occasion Polylogue handed context to a session or a signal notice to the operator, with what it included, what it left out and whether it arrived; recorded from the hook spool (`model.md`, Delivery). |
| **advisory** | One line a hook injects into an agent's context before a tool call or with a prompt, pointing at recorded failures or a judged note; rare by rule, with a cooldown read from earlier deliveries (`surfaces/mcp-and-hooks.md`). A signal is a standing rule that notifies the operator or supplies a briefing's watching section. |
| **run** | A unit of work a harness or job runner tracked across sessions: a task and its attempts (`model.md`, Run). |
| **process identity** | The triple (boot ID, PID, start time) naming one operating-system process, so a reused PID or a reboot never makes a new process look like an old one (`sources/README.md`, hooks). |
| **process spool** | The daemon's own spool of process-tree changes under harness processes, written only on change and stored like any input (`sources/README.md`). |
| **live state** | What a session is doing now (`working`, `waiting`, `idle`, `ended`, `stale`, `unknown`), computed at read time from stored hook and process facts and the current time, never written by a timer; it carries `until`, when it would change with no new fact (`model.md`, Live state). |
| **peers** | The other live sessions in the same repository or worktree as a session, found by query (`model.md`, Live state). |
| **live task** | The job or Bead a session is working on, from a job variable in its hook lines or a claim command it ran, never from wording (`model.md`, Live state). |
| **work graph** | The view joining sessions, runs, tool calls, files, commits and notes; not a store (`model.md`). |
| **variant** | A translation, summary or other derived version of some text, kept beside the original with how it relates (`model.md`, Variant). |
| **turn** | One human prompt and everything up to the next human prompt on the active path (`query.md`, turns). Codex's `turn_id` is Codex's own term. |
| **qualifier** | The part of a native ID after the session's own ID that names a sub-session or an install: `/<agentId>` (a subagent), `~side-…` (an older sidechain), `@profile-<16 hex>` (a second Hermes install; `sources/others.md`). Short IDs shorten each part separately (`surfaces/cli.md`). |
| **subtree** | The match type of the scope fields `cwd` and `checkout`: a directory and everything below it, by whole path components and exactly as written (`query.md`). Not the field `path`, which matches substrings. |
| **interrupt budget** | The setting `signal.interrupts_per_hour`: how many interrupting notices (handed to the configured notifier) agent-written signal rules may raise per hour together; past it they become plain notices. The operator's own rules are outside it (`user-data.md`, signals). |
| **shipped saved query** | A named query that comes with Polylogue as data in its code, such as `query:running`; read like the operator's own, overridden by saving one under the same name (`user-data.md`, saved queries). |
| **capture envelope** | The small header the browser host wraps around a captured provider response (`sources/README.md`). The rewrite uses "envelope" only for this; a record's own outer fields are its wrapper, and a query result's status is its result outcome. |
| **match rows** | For an `exists` or sequence match, the rows that made it match, returned by `| matches` at pair grain (`query.md`, one relation, several uses). Called "witnesses" until 2026-10-12. |
| **witness** | One source's record of something, in the operator's sense: several routes to one conversation are separate witnesses, merged in derivation (`principles.md` #5). Never a query term (the joint query law "match order"). |
| **binding** | Resolving a query's source (a unit, a saved query, a kept result) and its defaults once, before anything composes with it; only a top-level condition on a default's field, or `F:*`, drops that default (`query.md`, Defaults, binding). Not Sinex's "source binding", which is a configured input in its catalog. |
| **candidates, ranking, selection, page** | The rows a query's conditions select (`count` is their number); the order a relevance or `near:` ranking puts them in; a `limit` or window that is part of the question; and how many rows travel now (`query.md`). |

## Building and testing

| Term | Meaning |
|---|---|
| **slice** | One vertical piece of the build plan, usable on its own (`plan/slices.md`). |
| **law** | A property every archive must satisfy, tested on the example archive and generated ones (for example: ingesting the same input twice changes nothing; `plan/testing/laws.md`). |
| **reference model** | A small in-memory evaluator of the query language used only by tests, to check the SQL compiler against (`plan/testing.md`). |
| **Morrow archive** | The small invented example world every spec example runs against (`examples/archive/FACTS.md`). |
| **native ID** | The ID a provider gave a session, message, call or attachment, kept exactly (`model.md`, Identity). IDs Polylogue composes from several native IDs join them with `/`; IDs it computes from content start with `~`. |
| **active path** | The branch of a conversation tree the provider marks as current (its marked leaf back to the root); other branches are kept but off the path. Without a marked leaf, branch messages' status is unknown (`model.md`, Message). |
| **attached file** | A provider file read as its own input and attached to the session or project it names (`history.jsonl`, `todos/`), unlike a companion file, which is parsed with its main file. |
| **parse state** | Per input: how far its chains are parsed (`parsed_upto`, the carried state) and a status, `parsing`, `done`, or why it produced nothing: `no_sessions`, `refused`, `failed`, `no_parser_yet` (`storage/index-db.md`, `parse_state`). Not the parser's return value, which is the Rust type `ParseOutcome` (`plan/slice-01.md`). |
| **status figure** | One reading in `polylogue status`: measured, measured absent, or unmeasured with the reason (`behavior.md`). |
| **stuck, idle, stalled** | An input is idle when caught up and stuck when it has unprocessed bytes or a pending failure and no advance for 10 minutes; the loop is stalled when it has work and has committed nothing for 10 minutes (`behavior.md`, `convergence.md`). |
| **embedding key** | Everything that makes two embeddings comparable: provider, model, dimensions, input type, normalization; vectors are compared only under one key. Formerly "model recipe"; renamed because Sinex's "recipe" is an analysis plan (`analysis.md`). |

## Annotation, time and briefing terms

| Term | Meaning |
|---|---|
| **evidence cutoff** (`evidence_until:`) | Polylogue's historical query: the evidence it had up to T, read with today's parsers, prices and excisions. Sinex's "state as of T" is a different question (its committed interpretation then); the two names stay distinct so similar syntax never implies the same meaning (`query.md`). |
| **time quality, precision** | Each provider time's `time_quality` (`intrinsic`, `file_mtime`, `unknown`) and `precision` (`instant` to `year`, or `bounded` with an end), in Sinex's vocabulary; a coarse time stands for its whole period (`model.md`, Time; joint glossary). |
| **coverage** | What a snapshot lists: an account export's coverage is the whole account, so a session absent from a later complete export was deleted upstream (`sources/chat-exports.md`; joint glossary, removed at source). "Field coverage" (which schema fields a parser reads) is a different, qualified use. |
| **check** | A query attached to a claim, with the result it expects (`rows`, `no_rows`, a count comparison or an aggregate comparison). Its **check state** (`holds`, `fails`, `unknown`, or `unchecked` when there is none) is computed at read time, never stored, and is not a judgment (`user-data.md`, checked claims; joint #16). |
| **rubric** | A note giving an annotation pass its instructions, the query it covers, and the note kinds, keys and values it may write; the writer refuses anything else (`user-data.md`). |
| **pass** | One agent's annotation run under a pinned rubric revision; its notes carry `derived_from` to the rubric and are judged together on a blind sample (`user-data.md`). |
| **logical day** | The day a person means: date D runs from D at `day.logical_start` (default 04:00) to the same time on D+1, in `day.zone`. Used by time words, day grouping and briefing ages (`query.md`, times; joint #15). |
| **read back** | The plain-language sentence a query is rendered into from declared phrases, returned as `read_as` (`query.md`, errors and explain; joint #17). |
| **machine section** | The part of a session-start briefing that Sinex fills with machine context, at most a declared share of the budget (`user-data.md`, briefings; joint #18). |
| **record outcome** | What a provider did with one framed record: `Used`, `Ignored` (a kind from its closed list) or `Unrecognized`; the framework counts them (`sources/README.md`, the framework owns iteration). In code `RecordOutcome`. |

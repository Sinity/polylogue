# Assertions

Everything people and agents add to the archive is an assertion, except variants, which have their own records here too. This page owns the record and its tables, kinds, links and currency, authorship of every authored record, reading, judgment, comparisons and forecasts, rubrics and passes, checked claims, variants, markup, memory and methods, and the one write path. Pages are built from assertions and are described in [pages.md](pages.md); the reference grammar is [references.md](references.md); how agents receive assertions in briefs is [agents.md](agents.md#brief); how a web view edits them is [web.md](web.md).

Assertions are the substrate of everything authored, and there are no separate tables per kind of mark. Ruling, 2026-06-20: "assertions are the substrate, the specifc user mark tables are supposed to be wiped out" (codex:019ecc6e-b42c-73b1-865a-6a59d352de2c@2026-06-20T20:08:14Z).

## The record

An assertion is an authored statement that a **key** has a **value** for a **target**. It records its author and the channel it came through, may carry evidence, a scope and an origin, may carry an expiry, and is revisable and addressable, so other assertions can be about it.

Captured content (sessions, messages, blocks, tool calls, bytes) is evidence and never an assertion. Derived values (costs, detected language) are not assertions; neither are settings' defaults, kept results, deliveries or variants. "Annotation" is the everyday word for assertions in general; "note" names only the free-text kind. The type is `Assertion`, the query unit `assertions`, the verb `assert`.

The tables live in `archive.db`:

```sql
CREATE TABLE assertion (
  id          TEXT PRIMARY KEY,  -- request ID (UUIDv7) or a derived ID (markup); '<request>.<n>' for the n-th row of one request
  root        TEXT NOT NULL,     -- the first revision of this assertion; equals id for a first revision
  kind        TEXT NOT NULL,     -- closed list, see Kinds
  key         TEXT,              -- meaning per kind
  value       TEXT,              -- text, or JSON checked against the kind's shape
  author      TEXT NOT NULL,     -- computed, see Authorship
  via         TEXT NOT NULL,     -- mcp | cli | web | extension | hook | markup | import | daemon
  machine     TEXT NOT NULL,
  created_at  INTEGER NOT NULL,  -- the writer's clock: display, age and as-of reads; never revision order
  valid_until INTEGER,           -- optional; after it the assertion is shown as expired and left out of briefs
  commit_seq  INTEGER NOT NULL   -- this archive's write counter: change feed and send order only
) STRICT;

CREATE TABLE assertion_link (
  assertion   TEXT NOT NULL REFERENCES assertion(id),
  role        TEXT NOT NULL,     -- see Links
  ref         TEXT NOT NULL,     -- a reference in its stored form
  seen_hash   BLOB,              -- BLAKE3 of the referenced text when written; filled on arrival for a waiting target
  PRIMARY KEY (assertion, role, ref)
) STRICT;
CREATE INDEX assertion_link_ref ON assertion_link(ref, role);
CREATE INDEX assertion_kind_key ON assertion(kind, key);
CREATE INDEX assertion_root     ON assertion(root);
```

- **One link table.** Target, evidence, scope, origin and every relation are the same fact: this assertion points at that reference, in this role.
- **Nothing computed is stored.** Currency, conflicts, withdrawal, judgment state, target state and work-item state are computed when read, so they can never disagree with the links they come from.
- **Nothing expires by itself.** `valid_until` changes how an assertion is shown; it deletes nothing. Polylogue has no deletion ([architecture.md](architecture.md#deletion)); withdrawal is itself an authored statement.

## Kinds

Kinds are one closed declaration in the `model` crate. Each states the key's meaning, the value's shape, how many current values it allows, what it may be about, whether markup may write it, and whether briefs read it. An unknown kind is refused; a new kind is a change to that declaration. Annotation is constrained by these declared shapes so that it is useful rather than free-form. The operator, 2026-07-30: Polylogue "should usefully constrain, rather than having the system be like blank piece of paper" (journal 2026-07-30 20:17).

| Kind | Key | Value | Current values | About | Markup |
|---|---|---|---|---|---|
| `note` | — | text | many | anything | `::note` |
| `tag` | tag name | optional `{value?, confidence?}` | one per (about, key, value) | anything | `::tag` |
| `highlight` | colour name | optional text | many | a `span:` | `::highlight` |
| `title` | — | text | one per about | a session | `::title` |
| `outcome` | — | `solved`, `partial` or `abandoned` | one per about | a session | `::outcome` |
| `hidden` | — | optional reason | one per about | a session | — |
| `claim` | optional label | text, optional `check` (see Checked claims) | many | anything | `::claim` |
| `finding` | optional label | text | many | anything; `derived_from` the `result:` or `query:` it rests on | `::finding` |
| `decision` | optional label | text, optional `{alternatives, constraints, reverse_if}` | many | anything | `::decision` |
| `lesson`, `question`, `goal`, `problem`, `blocker`, `handoff`, `correction` | optional label | text | many | anything (`correction`: exactly what it corrects) | `::<kind>` |
| `forecast` | event name | `{probability, horizon, resolves_by}` | many | anything | `::forecast` |
| `judgment` | `accepted`, `rejected` or `deferred` | optional reason; `blind` details | many | exactly one assertion at an exact revision | — |
| `comparison` | dimension | `{verdict, rubric?}` | many | two or more references | — |
| `rubric` | name | `{instructions, unit_query, fields, abstain, confidence}` | one per name | — | — |
| `pass` | — | `{rubric, population: result:<id>, counts}` | many | — | — |
| `annotation` | a rubric field, or `abstain` | the field's value, or the reason for abstaining | one per (about, rubric revision, key, author) | one row of the pass's population | `::annotation(rubric=…, key=…)` |
| `memory` | path | file text | one per (scope, key) | — (`scope` link) | — |
| `method` | name | `{inputs, conditions: [{name, check}], steps, expected, stop}` | one per name | — (optional scope) | — |
| `saved_query` | name | `{text, hash, denominator?}` | one per name | — | — |
| `signal` | name | a signal rule ([operations.md](operations.md#notices)) | one per name | — | — |
| `page`, `slot`, `member`, `role`, `ping` | see [pages.md](pages.md) | | | | |
| `withdrawal` | — | optional reason | many | — (`withdraws` links) | — |
| `setting` | setting name | the setting's value ([operations.md](operations.md#configuration)) | one per name | — | — |

- **Marks.** Starring, pinning and other marks are tags. "Pin" stays the word for a mark; a page slot that shows an exact revision is `fixed` ([pages.md](pages.md#layouts)).
- **`hidden`** takes a session and its rows out of default listings, search results, briefs and resume lists; `hidden:true` or `hidden:*` brings them back. Measures and population counts still include hidden sessions and say how many ("includes 3 hidden sessions"), because a display preference never changes a measured population. It is a reading preference, written and withdrawn with the ordinary `assert` and `withdraw` verbs.
- **A summary is a variant,** not a kind (see Variants). A pass is the one way to group bulk annotation.

## Links

| Role | Meaning |
|---|---|
| `about` | what the assertion is about (one or more) |
| `evidence` | what supports it, kept apart from what it is about |
| `scope` | where it applies (`repo:`, `cwd:`, `page:`) |
| `origin` | where it came from: the message a marker was in, the pass it belongs to |
| `revises` | it is a new revision of the named revision |
| `withdraws` | it withdraws the named revision, root, variant revision or withdrawal |
| `answers`, `challenges`, `supports`, `qualifies`, `corrects`, `derived_from`, `blocks` | typed relations between assertions |
| `slot_of`, `member_of`, `role_of` | page structure ([pages.md](pages.md)) |

**Stacks.** An assertion may be about another assertion at an exact revision, through one typed relation, so a claim, a question about it and an answer that challenges it form a stack. `read assertion:X --stack` shows everything that links to any revision of X, transitively, each with author, revision, currency and relation; a page renders the same tree in place.

**Repetition is not corroboration.** `supports` counts distinct roots reached through `derived_from`: three summaries of one source are one source, and variants never count.

## Currency

- **Current revision.** Revision R of root X is current when R is not withdrawn and no non-withdrawn assertion links `revises assertion:X@R`. It is a pure function of the `revises` and `withdraws` links, never of `commit_seq` or time, so every machine computes the same answer from the same links.
- **Conflict.** Two current revisions of one root are a conflict. Both are shown with their authors; a single-valued read matches neither value; `conflict:true` finds them; `status` counts them. A revision that `revises` both resolves it.
- **Stale base.** A write from the CLI, MCP or HTTP whose `revises` names a revision that is no longer current is refused as `stale`, naming the current revision or revisions. Standalone, the check runs inside the write's transaction. In primary mode it runs against the head as of the applied stream position plus this host's writes that are prepared but not yet confirmed, so two quick revisions on one host, or the memory adapter's `create` followed by `str_replace`, see each other ([primary-mode.md](primary-mode.md)).
- **Markup and imports never refuse; they fork** into a visible conflict, so the order of arrival cannot change the result.
- **Across machines** no host sees another's unconfirmed writes; concurrent revisions become two heads by the same rule. No winner is chosen by timestamp.
- **A revision inherits** the kind, key and links of its base unless the write gives new ones.

**Withdrawal.** A `withdrawal` links `withdraws` to one or more assertion revisions, roots (all their revisions), variant revisions or roots, or other withdrawals. An assertion is withdrawn while a non-withdrawn withdrawal names it or its root. Withdrawing the newest revision makes the one it revised current again; restoring is withdrawing the withdrawal. One withdrawal may name many assertions, so a whole pass is withdrawn at once. Withdrawn assertions leave results, briefs, pages and currency unless a query asks for them (`withdrawn:true`, `withdrawn:*`).

**Exact references hold.** Judgments, questions, `fixed` slots, kept results and deliveries store `assertion:X@R`. A new revision never moves them, and a judgment of R says nothing about R+1.

**Cardinality is enforced at write** from the kind declaration. Writing a single-valued kind (`title`, `outcome`, `hidden`, `rubric`, `saved_query`, `memory` per scope, `page` per name) from a command revises the current one instead of adding a second root. Markup of a single-valued kind never revises: a command-written value beats every marker, and among markers in force the latest in message order along the session's lineage wins, then block index, then byte offset. A `tag` write is skipped when the target already has a current tag with that key and value; the reply says "already tagged".

## Authorship

Every authored record in Polylogue (assertions, variants, withdrawals, judgments, settings, kept results, deliveries, chat actions, backfill requests) stores who wrote it and how it arrived. Agents are primary users with no fewer powers than the operator; what distinguishes the operator is authorship, which the daemon computes and no caller supplies. Ruling, 2026-07-03: "I want agents to have more power / capabilites rather than less, more affordances" (cc-queued:30f5e650-7033-4423-9f91-2b66ebd1e6b2@2026-07-03T13:20:44Z).

### The record

- **`author`** is one of:
  - `operator`;
  - `agent:<session>`, the most specific session known, for example `claude-code:agent-a7f3a2c1000000003` for a Claude subagent or `codex:<thread>`;
  - `agent:<origin>`, when the harness is known and its session is not, for example `agent:gemini-cli`;
  - `script:<unit>`, a named non-interactive program, named by its cgroup unit (a systemd unit or a scope);
  - `unattributed`, when there is no evidence either way.
- **`via`** is the channel the write arrived on: `mcp`, `cli`, `web`, `extension`, `hook`, `markup`, `import` or `daemon`. There is no evidence blob and no confidence score.

**No surface accepts an author from the caller.** A request that names an author is refused as `invalid`. A `--session` option or `session` argument names a subject or a recipient, never the author.

### How each channel decides

The client process gathers its identity and sends it as `caller` in the request envelope: harness, session ID, tool call ID, harness pid, whether a controlling terminal is present, and the systemd unit or scope it runs in. The daemon's one function decides the author from `caller`, `via` and the hook spool, and every answer that used an identity names it under `caller`.

| Channel | Author |
|---|---|
| **MCP**, the agents' primary channel (`polylogue mcp`, a child of the harness) | At start: `CLAUDE_CODE_SESSION_ID` gives `claude-code:<id>`; `CODEX_THREAD_ID` gives `codex:<id>`; `HERMES_SESSION_ID` is resolved through Hermes's state database; otherwise the parent executable names the harness, giving `agent:<origin>`. When hooks are installed, the hook spool's latest `SessionStart` for the server's parent pid wins over the start-up variable, so resumed and cleared sessions are followed. Per call, for Claude Code: `_meta["claudecode/toolUseId"]` is joined to the spooled `PreToolUse` with that `tool_use_id`; when that event carries `agent_id`, the author is `claude-code:agent-<agent_id>`, otherwise its `session_id`. MCP writes are never `operator`. |
| **CLI**, the second channel | Harness identities come from the environment (`CLAUDE_CODE_SESSION_ID` with `CLAUDE_PID`; `CODEX_THREAD_ID` with `CODEX_TOOL_CALL_ID`; `HERMES_SESSION_ID`; `GEMINI_CLI=1`, which has no session). With several present, the nearest harness ancestor in the process tree wins; with several and no harness ancestor, `unattributed`. A Claude identity is refined to a subagent when the spool holds exactly one open `PreToolUse` Bash event from that `CLAUDE_PID` whose command runs `polylogue`; otherwise the main session stands, which is correct as the root of the work. With no harness identity, a controlling terminal gives `operator`; no terminal gives `script:<unit>` when the process's cgroup names a unit, else `unattributed`. |
| **Web UI and HTTP API** | `operator`, `via: web`, with no inspection of the browser. The web UI is the operator's channel; agents are directed to MCP. |
| **Browser extension** | `operator`, `via: extension`, for acts in the extension's own UI. Acts it performs for a request (a chat send, a backfill) carry the requester's author from the request record ([extension.md](extension.md#authorship-from-the-extension)). |
| **Hooks** | records a hook writes for a session are `agent:<that session>` or `daemon`, never `operator` |
| **Markup** | the author of the message the marker is in (see Markup) |
| **Import of user state** | each record keeps its stored author |
| **Launched work** (job runners, schedulers, scripts) | the CLI rule: `script:<unit>` from the cgroup, or the requesting session when the launcher forwards the harness variables. Polylogue recognises no job runner by name and requires no forwarding. |

Harness variable names are harness internals. A test pins the variables seen by a probe of each supported harness version; a renamed variable degrades the author to `agent:<origin>` or to the terminal rule, and never silently to `operator` except through an interactive terminal.

How captured messages record who wrote them (`written_by`, `prompt_kind`) is a separate fact of the parsed model: [sessions.md](sessions.md#message-authorship).

### What the author changes

- **No capability depends on the author.** Every author may write every kind, judgments and settings included. Settings that spend money or send data out of the machine are no exception; changes by anyone but the operator are listed in `status` until the operator next looks (default; open operator question).
- "Operator" means exactly `operator`; `agent:*`, `script:*` and `unattributed` are claims.
- Readers show author and `via` on every record. Query fields: `author`, `by_operator` (exactly `operator`), `agent` (any `agent:*`), `unattributed`, `via`. Unattributed records are labelled "author unknown", never shown as the operator's.
- **Claiming.** The operator makes an unattributed or mis-attributed record their own by an ordinary revision authored `operator`. The original stays in history; there is no re-attribution verb.
- Briefs and signals read the author: how briefs treat agent assertions is [agents.md](agents.md#brief); which rules may interrupt is [operations.md](operations.md#notices).

### MCP completeness

Every write in this page is a command-table operation and therefore an MCP tool: assert, revise, withdraw, judge, compare, keep, variant, page edits, ping, memory and setting. Only local-only commands (the daemon, hook installation, completions and similar) stay CLI-only, so an agent never needs the CLI to write.

### What authorship does not guarantee

Authorship is for honest attribution, not a security boundary: all agents run as the operator's user, and the threat it addresses is mistakes, not adversaries. Not guaranteed: deliberate impersonation (unsetting variables, typing into a terminal an agent opened, driving the web UI, forging `caller`); an agent using the operator's channels in good faith (recorded as `operator`); the operator running `polylogue` inside a harness's shell mode (recorded as that session, and claimable); exact subagent attribution for two concurrent CLI calls from one Claude process (recorded as the main session); Gemini sessions (harness only, `agent:gemini-cli`); detached work whose launcher does not forward the harness variables.

## Reading

- **Key/value reads.** `kv(about, kind, key)` is the set of current, non-withdrawn values; a single-valued kind gives one value or `conflict`. In the query language: `tag:<name>`, `title:`, `kv:<kind>.<key>=<value>` on any unit, matched through `about` links at that unit's grain or through a span inside it, and the `assertions` unit ([query.md](query.md#units-and-fields)).
- **Grain is strict.** An assertion about a session does not match its messages or tool calls; one about a block matches its message only through `kv:` at message grain with `within:`. A session-level `correction` marks no tool call corrected.

**Target state.** Each `about` and `evidence` link is shown in one of four states:
- **current**: the target exists and its text hash equals `seen_hash`;
- **changed since**: the hash differs (a better version merged, a re-parse produced different text), shown as "the text has changed since this was written";
- **waiting**: well formed, not stored yet, its session live or incomplete; it attaches on arrival with no rewrite and fills `seen_hash` then;
- **missing**: the session is complete and the ID never appeared; `status` counts these.

**Work-item state** is computed from links, never from text:

| State | Rule |
|---|---|
| question **answered** | a current assertion `answers` it |
| goal **achieved** | an `accepted` judgment, by the judges the reader counts, of an assertion that `answers` it |
| item **blocked** | a current assertion `blocks` it; the block ends when the blocker is answered or withdrawn; blocked never closes a goal |
| **decided** | a current `decision` is about it; deciding does not answer an open question |
| **withdrawn** | see Currency |
| **unresolved since T** | none of the above, stated with how far capture reaches |

A later answer is tied to an earlier item only by an explicit link (a marker argument or an `answers` link); similar wording never ties them.

**Backlinks, history and as-of.**
- Reading any reference lists what links to it: assertions by role, slots, kept results, deliveries and variants (`referenced_by:` in queries).
- `read assertion:X --history` lists revisions, withdrawals, judgments and conflicts in the order of the `revises` graph; ties are shown by `(created_at, id)`.
- `evidence_until:T` reads each assertion at the revision current among records with `created_at ≤ T`; pages use it for `--at`. `created_at` is the writer's clock, so as-of reads across machines are approximate and say so.

## Judgment

Judgment is optional and never a gate: most agent assertions will never be seen by a person, and nothing waits for one. Ruling, 2026-07-19: "I presume vast majority of agent-authored annotations ('assertions') will not even be seen by a human" (cc:af12164b-a2fc-42cb-a548-22277c0875a2:ce6a2395-1934-42dc-8ff5-f909be411f74).

- A `judgment` is an assertion about one exact revision, keyed `accepted`, `rejected` or `deferred`, with an optional reason.
- **Any author may judge,** and the judgment records its true author.
- **Readers choose whose judgments count.** `judged:` and `judged-by:` default to the operator's judgments; `judged-by:agent:<…>` and `judged-by:*` widen it. The operator's judgment of a revision is the latest non-withdrawn operator judgment of exactly that revision. Operator-authored assertions need none.
- No score, check result or count of agent judgments accepts anything; there is no promotion.
- `judge` walks open items (assertions not by the operator, of the kinds named, whose current revision has no judgment by the counted judges), one key per verdict on a terminal.

**Blind and sampled judging.**
- `judge --blind` hides author, model, origin and earlier verdicts until the verdict is given, shuffles the order, records `{order, hidden}` in the judgment's `blind` value, and reveals afterwards. It is labelled "hidden fields", not a guarantee of blindness.
- `judge --blind --sample N --seed S QUERY` draws a seeded random sample; the seed is stored on each judgment so the sample can be drawn again.
- **Agreement** of a pass with a declared reference (by default the operator's judgments) is reported per pass with its denominator, counting excluded, incompatible and conflicting items. Only judgments of the same revision under the same rubric revision are compared. Labelled samples are private judgment assertions in the archive; only the figures leave it.

**Comparisons.** A `comparison` judges two or more references on one dimension. Its verdict is `prefer:<ref>`, `ranking:[…]`, `tie`, `incomparable`, `abstain` or `insufficient_evidence`; the last two add no preference. Rankings are computed at read (win rate and a Bradley-Terry fit), each rank with its number of comparisons, and uncertain pairs are listed for judging, because choosing between two things is easier than scoring one.

**Forecasts.** A `forecast` names an event, a probability, a horizon and what resolves it; the resolution is an assertion that `answers` it with the outcome. Calibration uses only forecasts written before their event resolved (by `commit_seq` on one archive, otherwise by the resolving evidence's time) and is reported with its count.

## Passes and bulk writes

Systematic agent annotation is a **pass**: key/value assertions governed by a rubric, recording what was covered and where the annotator abstained. Ruling, 2026-07-09: an agent annotates "in particular ways, then maybe query the annotations it makes … we are allready supposed to basically support it, via assertions" (cc:3b0038ec-234c-44b8-bb88-fe222b44fe0f:389179b6-472e-4a33-b0eb-e3b0209a0ac2).

1. A `rubric` declares instructions, a unit query, and fields with types, enums or ranges, plus whether abstaining, a confidence and a rationale are allowed.
2. A pass first keeps its population as `result:<id>` ([query.md](query.md#query-objects)), then writes one `pass` assertion, the `origin` of everything that follows, and one `annotation` per row and field, each `derived_from` the rubric's exact revision.
3. A value the rubric does not allow is refused and returned, not stored. An abstention is an `annotation` keyed `abstain` with its reason.
4. **Coverage** is the kept rows minus the rows with an annotation or abstention, computed at read.
5. Withdrawing a pass's whole set is one withdrawal.
6. A marker cannot open a pass; `::annotation` names a rubric revision and a pass the same author opened.
7. **Shipped rubrics:** the predecessor's six annotation schemas (`delegation.discourse`, `seed.activity`, `seed.goal-event`, `seed.outcome-evidence`, `seed.knowledge-artifact`, `seed.reusability`) ship as rubric data, resolved by name like shipped saved queries; a rubric of one's own with the same name takes precedence.

**Bulk writes.** Any write over a query or a kept result writes **one assertion per matched row**, in one transaction and one request, with IDs `<request>.<n>`; the reply gives the committed count, the query's `query:<hash>` and its resolved bounds. An assertion **about** `result:<id>` is about the result itself, which is different from assertions on its rows.

## Checked claims

A `claim` or `finding` may carry `check: {query, expect}`, where `expect` is `rows`, `no_rows`, `count OP N` or `agg OP N`. The check is evaluated at read over the current archive and shown beside the assertion everywhere as `holds`, `fails` (with the rows it returned), `unknown` (the answer was incomplete or depends on data not collected) or `unchecked`.
- A check is a fact about the archive, not a verdict; `holds` accepts nothing.
- `status` counts operator-accepted claims whose check now fails.
- A `claim` or `question` may carry `outcomes: [{observation, implies}]`: the possible results of the next check and what each would mean.
- "Claim" in Polylogue means this kind and nothing else. A task someone has taken in a task tracker is a "task".

## Variants

A variant is transformed content: a translation, transliteration, simplification or summary. It is not an assertion, and assertions may be about it. Its purpose is that agents can render anything in the archive for the reader, such as translating a session the operator wants to read in Polish. (ruling 2026-07-04, codex:019f2e28-d759-7283-8d83-cf147b04bf75@2026-07-04T18:33)

```sql
CREATE TABLE variant (
  id          TEXT PRIMARY KEY,
  root        TEXT NOT NULL,       -- first revision of this rendition
  base        TEXT,                -- the revision this one revises (same root); NULL for a first revision
  source      TEXT NOT NULL,       -- exact reference: session, message, range, block, span, assertion@rev, page@rev, variant@rev
  source_hash BLOB NOT NULL,       -- BLAKE3 of the source text at that revision
  kind        TEXT NOT NULL,       -- translation | transliteration | simplification | summary
  from_lang   TEXT, to_lang TEXT,  -- BCP 47
  author      TEXT NOT NULL, via TEXT NOT NULL,
  method      TEXT,                -- free text or a reference: model, prompt, rubric used
  machine     TEXT NOT NULL, created_at INTEGER NOT NULL, commit_seq INTEGER NOT NULL
) STRICT;

CREATE TABLE variant_part (
  variant     TEXT NOT NULL REFERENCES variant(id),
  n           INTEGER NOT NULL,    -- order in the variant
  of          TEXT NOT NULL,       -- the source child it renders: a message, block or span inside `source`
  of_hash     BLOB NOT NULL,
  relation    TEXT NOT NULL,       -- translates | transliterates | simplifies | summarizes | omits | expands | reorders
  body        BLOB NOT NULL,       -- zstd text
  PRIMARY KEY (variant, n)
) STRICT, WITHOUT ROWID;
```

Revisions use `base`; withdrawal is an ordinary `withdrawal` that `withdraws variant:<root>[@<rev>]`; currency follows the rule under Currency with `base` in place of `revises`. Finer alignment inside one part is expressed by span references in assertions about the part.

- **Hierarchical.** A variant whose source is a session consists of parts, one per rendered block, each naming its original block in `of`; its messages are the groups of parts by their original message. A message variant is the same at message grain. A summary is one or more parts whose `of` is what the producing agent chose (the session, a range of messages, a block), with relation `summarizes`; the agent declares that mapping when it writes the variant. Reading `variant:V` renders the original's structure with parts swapped in; unmapped children show "untranslated" and `omits` parts show "omitted".
- **Bound to its source revision.** Revisions and variants are separate dimensions: a page can exist in Polish and English, each with its own revisions. Correcting a rendition writes a revision with the same source. When the source changes (`source_hash` or an `of_hash` differs), the variant shows "rendition of an earlier text"; translating again is a new variant.
- **Assertions get variants too.** The source may be `assertion:X@R`. An agent translating an annotated session writes one session variant plus one variant per assertion it renders, each pointing back at its original. On a variant view, assertions about the variant itself are shown; for an assertion about an original child, its current variant in the view's language is shown when one exists, otherwise a count with a link ("2 notes in the original"). The original's text is never shown as if it were about the variant.
- **Variants never count** as support, votes or judgment.
- **Enabled variants are a setting** that every reader follows. `reading.variants` holds an ordered list of keys (`translation:pl`, `simplification`). The operator, 2026-07-04: "'enable' such alternates, particular keys. If enabled, then we are supposed to render with content swapped … (also clearly indicating this happened)" (journal 2026-07-04 18:56). Every reader (CLI `read`, the web reader, pages, `read --to DIR`) renders each part swapped for the first enabled, current, non-stale variant, visibly marked, with the original one action away; `read --variant KEY` overrides it for one read. One renderer serves terminal, web and files ([reading.md](reading.md#views)).
- **Search.** Variant text is indexed separately and matches only with `variant:true` or `lang:<code>`. A message's detected language is a derived field (`message.lang`), not a variant.
- **Writers** are agents, through the `variant` operation (body from a file or stdin, parts as JSON). Polylogue never calls a model to produce or judge content: variants, labels and summaries are written by agents in their own harnesses.

## Markup

Agents and the operator write assertions and pings inside ordinary messages, as an alternative to tool calls. Ruling, 2026-07-03: "an annotation protocol for the agents, to directly output as text not tool calls" (cc:30f5e650-7033-4423-9f91-2b66ebd1e6b2:027fee03-005e-46fe-9d0f-eb7c62ed4b8a).

```ebnf
marker = "::" kind [ "(" args ")" ] [ ":" " " body ] ;
args   = arg { "," [ " " ] arg } ;
arg    = [ name "=" ] value ;          (* the first unnamed argument is the key *)
value  = bare | quoted ;
body   = rest of the line, then lines indented by at least 2 spaces ;
ping   = "@[" reference "]" " " body ;
```

Named arguments: `about=` (by default the session for `title` and `outcome`, otherwise the containing message), every relation role (`answers=`, `revises=`, `challenges=`, …), `evidence=`, `scope=`, `page=` and `section=` (which also add a slot on that page), `key=`, `span=` for `::highlight`, and `rubric=` and `pass=` for `::annotation`. Any other argument makes the marker "not applied". Pings are described in [pages.md](pages.md#ping-records).

**Where markers count.** At the start of a line, after at most three spaces; only in text written by a person or a model. Never inside fenced or indented code, on a quoted line, after `\::`, in thinking, tool calls, tool results, attachments, inherited copies, harness-written text or known pasted spans. Text inside a code fence or a stored prompt is data.

**Extraction and identity.**
- Markers are extracted when the message is parsed (the `sources` crate), into assertions authored by the message's author, with `via: markup` and the message as `origin`.
- The ID is derived from the origin, the provider message ID, the BLAKE3 of the marker line and its occurrence among identical lines in that message. Position, arrival, parser and reading version never enter it, so re-parsing, re-importing or a copy in a resumed session finds the existing row and creates nothing.
- A marker assertion is in force while the current parse of its message still contains it; otherwise it shows "marker no longer present" and leaves results like a withdrawn one. Assertions that revise, answer or cite it remain.
- **Success is silent.** Failures (unknown kind, unresolvable reference, wrong value shape, a rubric refusal) are counted in `status`, listed by a query, and shown as "N markup errors" on the page a marker named.
- Instructions inside a body are content; markup does nothing beyond writing its assertion or ping.

## Memory

**Project memory is assertions:** decisions, lessons, handoffs, problems, questions and findings scoped to a `repo:` or `cwd:`, or about sessions there. Briefs read them by scope ([agents.md](agents.md#brief)); each cites its origin, and nothing decays except through an explicit `valid_until`. A later session confirms an entry by revising it with a `supports` link to the confirming message; a contradiction is a `challenges` assertion shown beside the entry.

**The memory-tool adapter** maps a harness's memory tool onto `memory` assertions: key = path, value = text, `scope` = the repository or directory (none for global). `view` is a query; `create` is a write or revision; `str_replace` and `insert` are a revision on the current base, refused as `stale` when the base moved; `delete` is a withdrawal; `rename` is a revision with the new key. It is a thin mapping on the MCP side, so an agent's memories become queryable across sessions, machines and agents with no second store. Harness memory files (`CLAUDE.md`, `AGENTS.md`, a harness's memory directory) are captured inputs with their revision history ([capture.md](capture.md)); Polylogue does not inject them, and an assertion cites one by its `document:` reference.

**Methods** are `method` assertions with checkable conditions. `method:X applies ENV` evaluates each condition's check with ENV as scope: all hold gives `applicable`, any fails gives `not_applicable`, otherwise `unassessed`, naming the deciding check; an empty list is `unassessed`. Applicable (the assessment), adopted (an accepted judgment) and used (a delivery that included it, or an assertion `derived_from` it) are three separate facts.

## Write path

Every write is a request `{id, machine, at, write}`, executed by the daemon's writer in its own small transaction:
1. **Idempotency.** The same request ID with the same content hash returns the original outcome (`repeat: true`); with other content it is refused as `conflict`. The client chooses the ID once per user action and reuses it on retry.
2. **Author**, computed as described under Authorship.
3. **Resolution.** References resolve; a query target runs in the same read transaction as the write's checks.
4. **Validation:** the kind's shape, cardinality, the rubric's allowance, stale base, an experiment's frozen assignment ([pages.md](pages.md)).
5. **Insert** with `commit_seq`.
6. **Reply** with the committed IDs, the count, the recorded author and `via`, and for query targets `query:<hash>` and the resolved bounds. Refusals carry their kind: `stale`, `ambiguous`, `invalid/<shape>`, `not_allowed_by_rubric`, `conflict`.

Markup assertions are the only assertions the parse step writes, and only for derived IDs not already present.

**Verbs.** Every write is one verb after a reference or query (`polylogue [REF | QUERY] VERB [ARGS]`, [interfaces.md](interfaces.md#grammar)); there are no shorthand commands that do a subset of another.

| Verb | Acts on | Arguments |
|---|---|---|
| `assert KIND [KEY] [VALUE]` | each matched row, one assertion per row | `--evidence REF`, `--scope REF`, one flag per relation role (`--answers`, `--challenges`, `--supports`, `--qualifies`, `--corrects`, `--derived-from`, `--blocks`), `--valid-until`, `--check QUERY --expect E`, `--rubric REF --pass REF`, `--page NAME --section S`, `--template T` |
| `revise [VALUE]` | each matched assertion revision | `--key`, `--about REF…` |
| `withdraw [KIND [KEY]]` | the matched assertions, or the current `KIND KEY` assertions about the matched rows | `--reason` |
| `judge [accept\|reject\|defer]` | the matched revisions, or interactively over open items | `--reason`, `--blind`, `--sample N`, `--seed S`, `--judged-by` |
| `compare DIMENSION` | the matched references | `--prefer REF`, `--ranking`, `--tie`, … |
| `variant` | a reference | `--kind`, `--from`, `--to`, `--base VARIANT`, parts from a file or stdin |
| `applies ENV` | a method | — |
| `import FILE` | — | lines of assertion writes |

Page verbs are listed in [pages.md](pages.md#agents-and-pages). Every write prints what it committed, the author it recorded, and the command that undoes it; `--json` gives the MCP structure.

**Importing assertion lines.** `polylogue import FILE` with JSON lines of assertion writes checks each line first; bad lines are reported with their line numbers and not written; good lines share one request and one `pass` (without a rubric) as their `origin`, so one withdrawal removes them. Moving user state between archives (`read --to DIR`, then `import`) is described in [primary-mode.md](primary-mode.md#moving-user-state).

**The predecessor's assertions** are a declared older format of `import`. Their kinds map onto the list above (mark, star and pin to `tag`; metadata to `tag` with a value; caveat to `note` with `qualifies`; suppression to `hidden`; recall pack and workspace to `page`; annotation batch to `pass`; comparative judgment to `comparison`). Authors map to the nearest honest author: script-written records to `script:<name>`, Codex sessions to `agent:codex:<id>`, and records stored under the operator's account name but written by agents to `unattributed`, never `operator`. Old identifiers and fields go into the value under `predecessor`, superseding links become `revises`, and all of them share one `pass` origin.

## Storage

Standalone, `assertion`, `assertion_link`, `variant` and `variant_part` live in `archive.db` and are backed up with it; there is no separate file of user state. In primary mode every authored record travels to Sinex as a `polylogue.*` record and is applied locally from the confirmed stream like every other record ([primary-mode.md](primary-mode.md)); currency follows the same rule on every machine.

Performance rules: currency, conflict and withdrawn state are computed by indexed lookups on `assertion_link(ref, role)` and `assertion(root)`, never by scanning all assertions per row; an assertion write never waits behind a first build for more than one writer group; markup extraction costs a declared, measured share of parsing throughput; `toc` and the other assertion reads meet the latency classes in [interfaces.md](interfaces.md#latency-classes).

## Laws

- **kinds-round-trip**: every kind writes and reads back its key, value and links through the daemon, the `assertions` unit and `read assertion:`; a malformed value and an unknown kind are refused with the shape error (property test over generated assertions).
- **links-by-role**: an assertion with `about`, `evidence`, `scope` and `origin` links reads back each in its role, and `kv:` matches only `about` links at the unit's grain.
- **highlight-changed-since**: a `highlight` on a `span:` written from the web reader and one from `::highlight` both read back on the message; after a better version of the message merges, both show "changed since".
- **waiting-target-attaches**: a reference written before its message is stored attaches on arrival with no rewrite and fills `seen_hash`; on a completed session whose ID never appears, it counts as missing.
- **stale-base-refused**: two revisions of one base through the daemon: the second is refused `stale`, naming the first. The law runs standalone and in primary mode against a test Sinex with the first revision still unconfirmed; the memory adapter's `create` then `str_replace` on the unconfirmed `create` succeeds, and a second `str_replace` on the old base is refused.
- **concurrent-revisions-conflict**: the same two revisions written on two archives and imported into one give a conflict; `conflict:true` matches; a single-valued read matches neither; a revision of both resolves it.
- **withdraw-and-restore**: with R1←R2, withdrawing R2 makes R1 current; withdrawing that withdrawal makes R2 current; withdrawing the root hides the family except under `withdrawn:*`.
- **exact-reference-holds**: a judgment and a question naming R1 stay on R1 after R2; `judged:` on R2 is unknown.
- **currency-ignores-order**: permuting `commit_seq` and `created_at` on a generated history gives the same set of current revisions.
- **cardinality**: two `title` writes give one root with two revisions; a `::title` marker never changes a command-written title; `assert tag research` twice writes one assertion.
- **stack-shown**: an agent claim, the operator's question about it at R1 and the agent's answer that `challenges` R1 all appear in `--stack`, the question still on R1 after R2.
- **work-item-state**: `::blocker(blocks=assertion:g@1)` makes goal g blocked; answering the blocker unblocks it; g is achieved only after an accepted judgment of an answer.
- **repetition-not-support**: two summaries `derived_from` one assertion count as one support; a variant of a claim changes no support or judgment count.
- **mcp-never-operator**: MCP never records `operator`; with `toolUseId` joined to a spooled `PreToolUse` carrying `agent_id`, the author is that subagent.
- **nearest-harness-wins**: Codex run inside Claude, with both sets of variables, is attributed to the nearest harness ancestor.
- **terminal-author**: the CLI with a terminal and no harness records `operator`; with no terminal and no harness, `script:<unit>` or `unattributed`.
- **web-and-extension-author**: web and extension acts record `operator`; an extension chat send records its requester's author.
- **caller-author-refused**: any request naming an author is refused; an agent's judgment is stored with the agent as author and does not count under the default `judged-by:operator`.
- **pass-coverage**: a pass over a kept 20-row result with 3 abstentions reports coverage 20/20 and 17 values; a disallowed value is refused; withdrawing the pass removes all 20 in one withdrawal.
- **bulk-one-per-row**: a query-targeted `assert tag` over five sessions writes five assertions in one request and replies 5; withdrawing one leaves four.
- **import-bad-line**: an import file with one unresolvable line writes the others and reports that line.
- **blind-judging-hides**: `judge --blind` shows no author, origin or model before the verdict, and the stored judgment records the shuffle.
- **agreement-denominator**: agreement over a sample of 20 with 3 abstentions reports n = 17, excluded 3.
- **comparison-ranking**: six comparisons over four items give Bradley-Terry ranks with counts; `abstain` adds no edge.
- **checked-claim**: the Morrow claim that an edit made the tests pass, with a sequence check, reads `holds`; without the passing run it reads `fails`; on an incomplete answer, `unknown`.
- **marker-idempotent**: one `::finding(answers=assertion:q@1)` arriving twice through two routes, and again after a reading-version re-parse, is one assertion.
- **marker-placement**: a marker inside a fence, a tool result or a quoted line creates nothing; an unknown kind increases the markup-error count in `status`, and nothing is echoed.
- **marker-order-independent**: extraction in reverse input order gives identical assertions and the same single-valued winner.
- **variant-renders-structure**: a Polish session variant with 12 block parts renders the session's structure in Polish with marks; a block without a part shows untranslated; a summary with one part of the session renders as a summary; editing the source marks it as a rendition of an earlier text.
- **variant-view-assertions**: on the Polish view, an English note on block 3 is not shown; its Polish variant is, pointing to the original; with no variant, "1 note in the original" is shown.
- **variants-enabled-everywhere**: `reading.variants = [translation:pl]` gives identical swapped output in `read`, the web reader and `read --to DIR`.
- **variant-search-separate**: a Polish word matches nothing without `variant:true` or `lang:pl`.
- **predecessor-import**: the predecessor's assertions import with none authored `operator`, and one withdrawal removes them all.

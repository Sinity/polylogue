# Annotations

Everything people and agents write about the archive is an annotation: notes, tags, judgments, comparisons, saved queries, watches and settings. This page owns the record and its tables, kinds, links and current revisions, reading annotations, the one write path and the write verbs `note`, `tag`, `edit` and `withdraw`, annotation lines, checked claims, memory and methods. Neighbouring pages own the rest: who wrote a record is [authorship.md](authorship.md); `judge`, comparisons, forecasts, rubrics and passes are [judging.md](judging.md); alternates (translations and other renderings, written with `note --as`) are [alternates.md](alternates.md); annotations written inside messages are [markup.md](markup.md); `save` and saved queries are [saved-queries.md](saved-queries.md#save); pages, slots and members are [pages.md](pages.md); settings are [installing.md](installing.md#configuration); the reference grammar is [references.md](references.md); how briefs read annotations is [agent-context.md](agent-context.md#the-brief-for-a-directory).

Annotations are the one record of everything authored; there are no separate tables per kind of mark.

## The record

An annotation is an authored statement that a **key** has a **value** for what it is **about**. It records its author and the channel it came through ([authorship.md](authorship.md)), may carry evidence, a scope and an origin, may carry an expiry, and has revisions and a reference (`annotation:<id>[@revision]`), so other annotations can be about it.

Captured content (sessions, messages, blocks, tool calls, bytes) is what annotations point at, never an annotation. Derived values (costs, detected language) are not annotations; neither are settings' defaults, kept results, deliveries or alternates. "Note" names only the free-text kind. The query unit is `annotations`.

The tables live in `archive.db`:

```sql
CREATE TABLE annotation (
  id          TEXT PRIMARY KEY,  -- request ID (UUIDv7) or a derived ID (markup); '<request>.<n>' for the n-th row of one request
  root        TEXT NOT NULL,     -- the first revision of this annotation; equals id for a first revision
  kind        TEXT NOT NULL,     -- closed list, see Kinds
  key         TEXT,              -- meaning per kind
  value       TEXT,              -- text, or JSON checked against the kind's shape
  author      TEXT NOT NULL,     -- computed, see authorship.md
  via         TEXT NOT NULL,     -- mcp | cli | web | extension | desktop | hook | markup | import | daemon
  machine     TEXT NOT NULL,
  created_at  INTEGER,           -- original authored time; NULL when unknown; never revision order
  valid_until INTEGER,           -- optional; after it the annotation is shown as expired and left out of briefs
  commit_seq  INTEGER NOT NULL   -- this archive's write counter: change feed and send order only
) STRICT;

CREATE TABLE annotation_link (
  annotation  TEXT NOT NULL REFERENCES annotation(id),
  role        TEXT NOT NULL,     -- see Links
  ref         TEXT NOT NULL,     -- a reference in its stored form
  seen_hash   BLOB,              -- BLAKE3 of the referenced text when written; filled on arrival for a waiting target
  quote       TEXT,              -- JSON, links to a span and citations: the exact text with up to 32 characters before and after, and the container's parse version when written
  PRIMARY KEY (annotation, role, ref)
) STRICT, WITHOUT ROWID;
CREATE INDEX annotation_link_ref ON annotation_link(ref, role, annotation);

CREATE TABLE revision_state (    -- an index kept by the writing transaction, never a source of truth
  id          TEXT PRIMARY KEY,
  root        TEXT NOT NULL,
  current     INTEGER NOT NULL,
  withdrawn   INTEGER NOT NULL
) STRICT, WITHOUT ROWID;

CREATE TABLE annotation_unique_key (    -- the unique keys each revision holds, by digest
  key_hash    BLOB NOT NULL,
  id          TEXT NOT NULL REFERENCES annotation(id),
  PRIMARY KEY (key_hash, id)
) STRICT, WITHOUT ROWID;
CREATE INDEX annotation_kind_key ON annotation(kind, key);
CREATE INDEX annotation_root     ON annotation(root);
```

- **One link table.** What it is about, evidence, scope, origin and every relation are the same kind of row: this annotation points at that reference, in this role.
- **Nothing derived is stored except two indexes.** `revision_state` holds which revisions are current and withdrawn, and `annotation_unique_key` which unique keys each revision holds. The write that adds a record recomputes them for exactly the roots it can affect, in the same transaction, so they always equal the rule over the links. Conflicts, judgment state, target state and work-item state are computed when read.
- **Nothing expires by itself.** `valid_until` changes how an annotation is shown; it deletes nothing. Polylogue has no deletion ([architecture.md](architecture.md#deletion)); withdrawal is itself an authored statement.
- **Original authored time is optional.** Other writes use the accepting request's captured time; imports keep the recorded time, or `null` when none was recorded. An undated annotation is current by the same link and slot rules as a dated one. Time-filtered and as-of reads exclude undated annotations, whose standing at that instant cannot be established; unbounded reads include them and show an unknown time. An import's observed time describes arrival, not authorship.

## Kinds

Kinds are one closed declaration in the `model` crate. Each states the key's meaning, the value's shape, how many current values it allows, what it may be about, whether markup may write it, whether briefs read it, and which verb writes it. An unknown kind is refused; a new kind is a change to that declaration. Annotation is constrained by these declared shapes so that it is useful rather than free-form. After `note … --as`, completion lists the kinds with what each is for ([completion.md](completion.md#positions)).

| Kind | Key | Value | Current values | About | Written by | Markup |
|---|---|---|---|---|---|---|
| `note` | — | text | many | anything | `note` | `::note` |
| `tag` | tag name | optional `{value?, confidence?}` | one per (about, key, value) | anything | `tag` | `::tag` |
| `highlight` | colour name | optional text | many | a `span:` | `note --as highlight` | `::highlight` |
| `title` | — | text | one per about | a session | `note --as title` (with `--rename`, also renamed in the provider for a web chat) | `::title` |
| `outcome` | — | `solved`, `partial` or `abandoned` | one per about | a session | `note --as outcome` | `::outcome` |
| `hidden` | — | optional reason | one per about | a session | `note --as hidden` | — |
| `claim` | optional label | text, optional `check` ([Checked claims](#checked-claims)) | many | anything | `note --as claim` | `::claim` |
| `finding` | optional label | text | many | anything; `derived_from` the `result:` or `query:` it rests on | `note --as finding` | `::finding` |
| `decision` | optional label | text, optional `{alternatives, constraints, reverse_if}` | many | anything | `note --as decision` | `::decision` |
| `lesson`, `question`, `goal`, `problem`, `handoff`, `correction` | optional label | text | many | anything (`correction`: exactly what it corrects) | `note --as KIND` | `::<kind>` |
| `forecast` | event name | `{probability, horizon, resolves_by}` | many | anything | `note --as forecast` | `::forecast` |
| `judgment` | `accepted`, `rejected` or `deferred` | optional reason | one per about and author | exactly one annotation at an exact revision | `judge` | — |
| `comparison` | dimension | `{verdict, blind?}` | many | two or more references | `judge` | — |
| `rubric` | name | `{instructions, unit_query, fields, abstain, confidence, rationale}` | one per name | — | `import` | — |
| `pass` | — | `{rubric, population: result:<id>, counts}` | many | — | `import` | — |
| `rubric_value` | a rubric field, or `abstain` | `{value, confidence?, rationale?}`: the field's value, or the reason for abstaining | one per (about, rubric revision, key, author) | one row of the pass's population | `import` | `::rubric_value(rubric=…, pass=…, key=…)` |
| `memory` | path | file text | one per (scope, key) | — (`scope` link) | `note --as memory`, the memory-tool adapter | — |
| `method` | name | `{inputs, conditions: [{name, check}], steps, expected, stop}` | one per name | — (optional scope) | `import` | — |
| `seen` | the target's reference | `{position}` | one per (key, author) | — | reads shown to their reader ([reading.md](reading.md#ranges)) | — |
| `role` | name | `{instructions, harness}` ([session-control.md](session-control.md#roles)) | one per name | — | `import` | — |
| `page` | page name | `{title, layout: linear\|board\|table, template?, params?, brief: off\|contents\|full}` ([pages.md](pages.md#page)) | one per name | its subject (optional) | `note --as page`, `save page:NAME`, `edit` and `import` of its working copy | — |
| `slot` | — | `{parent?, position, mode: current\|fixed, form?, heading?, query?, caption?, x?, y?, w?, h?}` ([pages.md](pages.md#slot)) | many (`slot_of` one page) | what it shows (absent for headings and live queries) | `save page:NAME`, `note --section`, `note --as heading`, `edit` and `import` of the page's working copy | `page=`, `section=`, `::heading`, `::slot` |
| `member` | alias | — | one per (key, page) | a session | `note ALIAS --as member` | — |
| `saved_query` | name | `{text, hash, denominator?}` | one per name | — | `save` | — |
| `signal` | name | a signal rule ([operations.md](operations.md#notices)) | one per name | — | `save --watch`, `note --as signal` | — |
| `withdrawal` | — | optional reason | many | — (`withdraws` links) | `withdraw` | — |
| `setting` | setting name | `{machine, value}` ([installing.md](installing.md#configuration)) | one per (name, host) for a host-scoped key, one per name for an archive-scoped key | — | `set` | — |

- **Kinds keyed by name** (`rubric`, `method`, `role`, `pass`) have a name as their identity and come in through `import` of annotation lines ([Annotation lines](#annotation-lines)); names are keys, never selections. Pages are written by verbs (`note`, `save`, `edit`) and by `import` of a working copy: the page's name comes from its `page:NAME` reference or the working copy's header, and a member's alias is the note's text, so names stay keys there too ([pages.md](pages.md#writing)).
- **Settings.** A setting's `machine` is its target host for a host-scoped key and null for an archive-scoped key; `value` has the catalog's type for that key. The machine that wrote the setting stays the annotation's own `machine`. Two current revisions for one key and host are a conflict, shown with both references; the default does not resolve it.
- **Marks.** Starring and other marks are tags: starring is `tag star`.
- **`hidden`** takes a session and its rows out of default listings, search results, briefs and the `unfinished` list; `hidden:true` or `hidden:*` brings them back. Measures and population counts still include hidden sessions and say how many ("includes 3 hidden sessions"), because a display preference never changes a measured population. It is written with `note 'why' --as hidden` and taken back with `withdraw`.
- **A summary is an alternate,** not a kind ([alternates.md](alternates.md)). A pass is the one way to group bulk annotation.

Seen marks contain `{position: {item, at}}`, where `item` is the last displayed item's reference or null, and `at` is the captured `INDEX_INSTANCE:INDEX:ARCHIVE` position. A mark belongs to one target and computed author. Within the current index generation it advances by archive counter, then index counter, then the actual item's reading order. Repeating or moving backward writes no request or mark. A mark from the current generation replaces an older-generation mark; a late write from a stale generation is refused. Reading marks provide no undo command and are omitted from normal annotation and feed listings. Displaying readers submit them; a read that merely writes a pipe or file does not.

## Links

| Role | Meaning | Option |
|---|---|---|
| `about` | what the annotation is about (one or more): the selected row | — |
| `evidence` | what supports it, kept apart from what it is about | `--evidence` |
| `scope` | where it applies (`repo:`, `cwd:`) | `--scope` |
| `origin` | where it came from: the message a marker was in, the pass or import it belongs to | — |
| `revises` | it is a new revision of the named revision | written by `edit` |
| `withdraws` | it withdraws the named revision, root, alternate or withdrawal | written by `withdraw` |
| `replies_to`, `challenges`, `supports`, `qualifies`, `corrects`, `derived_from`, `blocks` | typed relations between annotations; `replies_to` may also name the `delivery:` of a sent message | `--replies-to`, `--challenges`, `--supports`, `--qualifies`, `--corrects`, `--derived-from`, `--blocks` |
| `slot_of`, `member_of`, `role_of` | a slot's page; a member's page and role ([pages.md](pages.md#the-model)) | `save page:NAME`, `--page`, `--role` |
| `mentions` | the annotation's text links to the reference (`[text](REF)`, `<REF>`) | written from the text |
| `cites` | a sentence of the text rests on that exact place (a footnote `[^1]: "quote" <REF>`); `quote` holds the cited words with their context and `seen_hash` their hash | written from the text |
| `embeds` | the text shows a value or includes other text from the reference (`![caption](REF "form")`) | written from the text |

**References in text become link rows.** A note's text is Markdown with references ([page-text.md](page-text.md)). The write that stores the text parses it and writes one `mentions`, `cites` or `embeds` row per reference in the same transaction, so the link rows always equal what the text says; the text stays the source of truth. A reference that does not resolve yet is a waiting link; an ambiguous one refuses that write. `evidence` says what supports the annotation as a whole; `cites` says which sentence rests on which words.

**Stacks.** An annotation may be about another annotation at an exact revision, through one typed relation, so a claim, a question about it and a reply that challenges it form a stack. What links to a note is a query: `polylogue 'annotations where links_to:annotation:01JB7Q2M'` lists everything that links to any revision of it, each with author, revision, whether it is current, and relation.

**Repetition is not corroboration.** `supports` counts distinct roots reached through `derived_from`: three summaries of one source are one source, and alternates never count.

## Current revisions

- **Current revision.** Revision R of root X is current when R is not withdrawn and no revision reachable from R through incoming `revises` links (its descendants, transitively) is not withdrawn. Withdrawn revisions are thus transparent: in R1←R2←R3, withdrawing R2 leaves R3 current and R1 not. It is a pure function of the `revises` and `withdraws` links, never of `commit_seq` or time, so every machine computes the same answer from the same links. The rule and its functions are shared with Sinex through polylogue-sinex-common's `authored` crate.
- **Conflict.** Two current revisions of one root are a conflict. Both are shown with their authors; a single-valued read matches neither value; `conflict:true` finds them; `status` counts them. A revision that `revises` both resolves it.
- **Several families in one slot are a conflict too.** Records that arrive through the stream or a bundle skip the write-time uniqueness check, so two hosts can each write the first `title`, setting or mark for one target. Every reader treats that slot as conflicting (`conflict:true`, `has:conflict`, the displayed title, settings and their history, which reports the interval as unknown, saved queries and marks). The next write into the slot without bases revises every head of every family, and a revision whose bases name every head of every family is accepted; either resolves the conflict into the first family by root. A revision of only one of them is refused as `conflict`, naming the others.
- **Stale base.** An `edit` whose base is no longer current is refused for that row as `conflict`, naming the current revision or revisions; the other rows commit. Standalone, the check runs inside the write's transaction. In primary mode it runs against the head as of the applied stream position plus this host's writes that are prepared but not yet confirmed, so two quick revisions on one host, or the memory adapter's `create` followed by `str_replace`, see each other ([primary-mode.md](primary-mode.md)).
- **Markup and imports never refuse; they fork** into a visible conflict, so the order of arrival cannot change the result.
- **Across machines** no host sees another's unconfirmed writes; concurrent revisions become two heads by the same rule. No winner is chosen by timestamp.
- **A revision inherits** the kind, key and links of its base unless the write gives new ones.

**Withdrawal.** A `withdrawal` links `withdraws` to one or more annotation revisions, roots (all their revisions), alternate revisions or roots, or other withdrawals. An annotation is withdrawn while a non-withdrawn withdrawal names it or its root. Withdrawing the newest revision makes the one it revised current again; restoring is withdrawing the withdrawal. A withdrawal is never edited: an edited withdrawal would go on withdrawing what the old one named, so `edit` of a withdrawal is refused. One withdrawal may name many annotations, so a whole pass or a whole request is withdrawn at once. Withdrawn annotations leave results, briefs and current revisions unless a query asks for them (`withdrawn:true`, `withdrawn:*`).

**Exact references hold.** Judgments, questions, kept results and deliveries store `annotation:X@R`. A new revision never moves them, and a judgment of R says nothing about R+1.

**Uniqueness is enforced at write** from the kind declaration. Each kind declares a uniqueness key, a list of components (`about`, `key`, `value`, `scope`, `author`, `rubric`, `machine`; the kind is always part of it), or none for a kind that may have many values; a write without bases goes into the slot its key gives, as a revision of every current head of the record that holds it. Writing a single-valued kind (`title`, `outcome`, `hidden`, `rubric`, `saved_query`, `memory` per scope) revises the current one instead of adding a second root. Markup of a single-valued kind never revises: a value written by a verb beats every marker, and among markers in force the latest in message order along the session's lineage wins, then block index, then byte offset. A `tag` write is skipped when the row already has a current tag with that key and value; the reply says "already tagged". A write over more than one row never writes a single-valued kind (`title`, `outcome`, `hidden`): it is refused as `single_valued_bulk`, and systematic labelling goes through a rubric field instead.

## Reading

- **Key/value reads.** `kv(about, kind, key)` is the set of current, non-withdrawn values; a single-valued kind gives one value or `conflict`. In the query language: `tag:<name>`, `title:`, `kv:<kind>.<key>=<value>` on any unit, matched through `about` links at that unit's grain or through a span inside it, and the `annotations` unit ([query.md](query.md#units-and-fields)).
- **Grain is strict.** An annotation about a session does not match its messages or tool calls; one about a block matches its message only through `kv:` at message grain with `within:`. A session-level `correction` marks no tool call corrected.
- **History and links are queries.** A note's history is `polylogue 'annotations where root:annotation:01JB7Q2M withdrawn:*'`: revisions, withdrawals, judgments and conflicts in the order of the `revises` graph, ties shown by `(created_at, id)`. The undo command carries the exact revisions a request created.
- **Backlinks.** Reading any reference lists what links to it: annotations by role, pages that place, link, cite or include it, kept results, deliveries and alternates (`referenced_by:` and `cited_by` in queries, the `links` unit; [page-text.md](page-text.md#backlinks)).
- **As-of.** `seen_by:T` reads each annotation at the revision current among records with `created_at ≤ T`. `created_at` is the writer's clock, so as-of reads across machines are approximate and say so.

**Target state.** Each link to a target (`about`, `evidence`, `mentions`, `cites`, `embeds`) is shown in one of six states, computed when read and never stored:
- **current**: the target exists and its text hash equals `seen_hash`. When a re-parse moved the cited text's offsets, the stored quote and its context are found again in the new reading and the link stays `current`, shown `✓ re-anchored`; the stored reference stays as written;
- **changed**: the hash differs and the quote is not found again (a better version merged, a re-parse produced different text), shown `≠ source changed` with the text as cited and the text now;
- **waiting**: well formed, not stored yet, its session live or incomplete; it attaches on arrival with no rewrite and fills `seen_hash` then;
- **missing**: the session is complete and the ID never appeared; `status` counts these;
- **withdrawn**: the target is a withdrawn annotation; the quote stays shown;
- **outside**: a `url:` or other target outside the archive, which is never fetched and so never checked.

How each state shows on a page, per relation, is [page-text.md](page-text.md#when-a-source-changes).

**Grounding** (field `grounding` on annotations and on text slots: `cited`, `changed`, `uncited` or `none`) says whether text rests on citations. It is computed from the `cites` and `evidence` links and their target states when read, and marks text without gating any write ([page-text.md](page-text.md#grounding)).

**Proposals and tracked changes** are readings of judgments, never a state of the record or a gate. Text whose current revision is by anyone but the operator and has no judgment by the counted judges shows as proposed; an unjudged revision of text whose earlier revision was accepted shows as a tracked change over the accepted text. The current revision is still the newest one for every query and view ([pages.md](pages.md#authorship-and-proposals)).

**Work-item state** is computed from links, never from text:

| State | Rule |
|---|---|
| question **answered** (field `answered`) | a current annotation `replies_to` it |
| goal **achieved** | an `accepted` judgment, by the judges the reader counts, of an annotation that `replies_to` it |
| item **blocked** | a current annotation `blocks` it; the block ends when the blocking annotation is replied to or withdrawn; blocked never closes a goal |
| **decided** | a current `decision` is about it; deciding does not answer an open question |
| **withdrawn** | see [Current revisions](#current-revisions) |
| **unresolved since T** | none of the above, stated with how far capture reaches |

A later reply is tied to an earlier item only by an explicit link (a marker argument or `--replies-to`); similar wording never ties them. The shipped saved query `questions` lists the open ones.

## The write path

Every write step is a request `{id, machine, at, write}`, executed by the daemon's writer in its own small transaction:

1. **Idempotency.** The same request ID with the same content hash returns the original reply (`repeat: true`); with other content it is refused as `conflict`. The CLI makes a fresh request ID per invocation; an MCP or HTTP call without one gets an ID derived from its author, tool, operation and arguments, so a re-issued call never writes twice ([agents.md](agents.md#retries-never-write-twice)).
2. **Author**, computed as [authorship.md](authorship.md) describes.
3. **Resolution.** References resolve; the selection was resolved once for the whole line ([interfaces.md](interfaces.md#steps)).
4. **Validation:** the kind's shape, cardinality, the rubric's allowance, a stale base.
5. **Insert** with `commit_seq`. A step over many rows writes one annotation per row in one request, numbered `<request>.1`, `<request>.2`, …; a large write commits in steps under one request ID so the single writer never stalls live capture.
6. **Reply** with the committed references, the count, the recorded author and channel, and **the undo line**: a declared `withdraw` command over the exact created revisions, passed on standard input for a bulk write. Existing annotations reused by the write get no new undo; a retry retains the original undo. Refusals carry their kind: `conflict`, `ambiguous` and `needs_all` as codes, and as typed reasons ([interfaces.md](interfaces.md#exit-codes)) `bad_value` with the kind and the shape it broke, `not_allowed_by_rubric` with the field and the part refused (the value, an abstention, a confidence or a rationale), `single_valued_bulk` with the kind and the row count, and `slot_taken` with the holders.

Undo is the safety for named rows; a query matching several rows also needs `--all` first ([interfaces.md](interfaces.md#steps)). An annotation **about** `result:<id>` is about the kept result itself, which is different from annotations on its rows. Markup annotations are the only ones the parse step writes, and only for derived IDs not already present.

### note

`note TEXT [--as KIND]` writes text about each selected row; `--as` names the kind (default `note`). `TEXT` is one shell argument, or `-` for standard input. It is Markdown with references: links, footnote citations, embeds and values, whose references become link rows in the same write ([page-text.md](page-text.md#markdown-with-references)).

| Kinds | Rows | `TEXT` | Example |
|---|---|---|---|
| `note`, `claim`, `finding`, `decision`, `lesson`, `question`, `goal`, `problem`, `handoff`, `correction` | what it is about | the text | `polylogue @2 note 'flaky because the clock is real' --as finding` |
| `title` | a session | the title | `polylogue latest note 'Back off on 429' --as title` |
| `outcome` | a session | `solved`, `partial` or `abandoned` | `polylogue latest note solved --as outcome` |
| `hidden` | a session | why it is hidden | `polylogue @4 note 'test run' --as hidden` |
| `highlight` | a `span:` reference | the colour | `polylogue span:message:01JB7Q2M#c=10-80 note yellow --as highlight` |
| `translation:LANG`, `transliteration:LANG`, `simplification`, `summary` | a message, or a session or page with `--parts FILE\|-` (one part per message or per annotation the page shows, as lines) | the text, or `''` with `--parts` | `polylogue @3 note '' --as translation:pl --parts parts.jsonl` |
| `signal` | the selection's question, not its rows (one rule; no `--all`) | the rule as JSON, any shape the shared signals schema accepts | `polylogue query:waiting note '{"for":"20m","notify":true}' --as signal` |
| `page` | a `page:NAME` reference | the title; `--template`, `--param`, `--layout`, `--about`, `--scope`, `--brief` set the rest | `polylogue page:flaky-ci note 'Why is CI flaky on Fridays?' --as page --template investigation` |
| `heading` (a heading slot) | a page | the section's title; `--level N`, `--section S` | `polylogue page:flaky-ci note 'Risks' --as heading` |
| `member` | a session | the alias; `--page P`, `--role R` | `polylogue codex:019d0f2a note reviewer --as member --page flaky-ci --role reviewer` |
| `forecast`, `memory` (the row is the scope), `rubric_value` | as declared | JSON | — |

- Relation options take references: `--evidence`, `--replies-to`, `--supports`, `--challenges`, `--qualifies`, `--corrects`, `--derived-from`, `--blocks`, `--scope`; `--valid-until TIME`; `--check QUERY --expect E` for a checked claim; `--key KEY` for kinds keyed by name.
- The four alternate kinds write an alternate record, not an annotation ([alternates.md](alternates.md)).
- **On a page.** `note` on a `page:` selection writes discussion about the page, placing nothing, unless `--section S` places the note on the page or it is `--as heading` or `--as page`; on `page:NAME#ANCHOR` it writes about that section or slot (`polylogue page:flaky-ci#7q2m note 'which test?' --as question`). On any other selection `--page NAME --section S` writes the note about its rows and places it too. A placed note and its slot are one request ([pages.md](pages.md#discussion)).
- **Renaming a web chat.** `polylogue chatgpt:6a2e91c0 note 'New title' --as title --rename` writes the title and asks the browser extension to rename the conversation in the provider, an act recorded like every act ([extension.md](extension.md#chat-actions)); without `--rename` the title is Polylogue's alone.

### tag

`tag NAME` or `tag NAME:VALUE` writes the `tag` kind on each selected row.
The first colon separates the name from a text value, including any later
colons; without a colon the value is absent.

A tag the row already has with that value is skipped and the reply says
"already tagged". If every row is already tagged, no annotation write request
is created. A retry returns its original written and skipped rows, even if
the current tags have since changed.

Completion after `tag` lists the tags in use with counts, so tags converge.
A uniform label over a query is one line:
`polylogue 'sessions where repo:sinex ends_open:true' tag needs-look --all`.

### edit

`edit TEXT` writes a new revision of each selected annotation with the new text, keeping kind, key and links unless `--key` changes the key; the link rows its text implies are rewritten with it. A stale base is refused for that row, naming the current revision; the other rows commit. The reply says `revision 3; earlier revisions kept`.

`edit` with no text, on a terminal, opens the selection in your editor: one note as its text, several notes as one file with a marker each, a page as its working copy, a page name that does not exist as a new working copy. What you save is checked in when the editor exits ([page-working-copy.md](page-working-copy.md#editing-in-your-editor)). Off a terminal and over MCP, `edit` without text is refused with the `read --to` and `import` lines.

### withdraw

`withdraw [--reason TEXT]` withdraws each selected annotation (a root withdraws its family, a revision only itself), alternate or saved query; withdrawing the withdrawal restores. Nothing is deleted.

A single unparameterized saved-query reference such as `query:NAME` in a line containing `withdraw` names the current authored saved-query family, including its watch, rather than the question's result rows. The whole line uses that record selection, and a retry retains its original family even if the name was withdrawn or recreated. Shipped queries without an authored definition cannot be withdrawn. The query parser preserves literal existing names, including quoted names outside the spelling rule for new saves. Parameterized, conditioned, revisioned and pipelined query expressions still select results: `polylogue 'query:selected-notes | sort by id' withdraw --all` withdraws those result annotations while keeping the saved definition.

On rows that are not annotations or alternates, the refusal gives an executable line selecting annotations about the first such row, with the reason preserved (`polylogue 'annotations where about:session:codex:019d0f2a' withdraw --all`). It reports the selected row count; the proposed line addresses that one row.

### judge

`judge accept|reject|defer` records a verdict on each selected annotation, and `judge ROW --on DIMENSION` records which of several rows is better; both are [judging.md](judging.md).

### Annotation lines

A batch of different annotations about different rows is a file of annotation lines brought in with `polylogue import FILE`: each JSON line names its kind, key, value, links and optional request ID; each line is checked first, bad lines are reported by number and not written, and good lines share one request and one origin, so the returned undo command withdraws just the revisions they created. Moving user state between archives (`read --to DIR`, then `import`) is [primary-mode.md](primary-mode.md#moving-user-state).

## Checked claims

A `claim` or `finding` may carry `--check QUERY --expect E`, where `E` is `rows`, `no_rows`, `count OP N` or `agg OP N`. The check is evaluated at read over the current archive and shown beside the annotation everywhere as `holds`, `fails` (with the rows it returned), `unknown` (the reply was incomplete or depends on data not collected) or `unchecked`.

- A check is a fact about the archive, not a verdict; `holds` accepts nothing.
- `status` counts operator-accepted claims whose check now fails.
- A `claim` or `question` may carry `outcomes: [{observation, implies}]`: the possible results of the next check and what each would mean.
- "Claim" in Polylogue means this kind and nothing else. A task someone has taken in a task tracker is a "task".

## Memory

**Project memory is annotations:** decisions, lessons, handoffs, problems, questions and findings scoped to a `repo:` or `cwd:`, or about sessions there. Briefs read them by scope ([agent-context.md](agent-context.md#the-brief-for-a-directory)); each cites its origin, and nothing decays except through an explicit `valid_until`. A later session confirms an entry by editing it with a `supports` link to the confirming message; a contradiction is a `challenges` annotation shown beside the entry.

**The memory-tool adapter** maps a harness's memory tool onto `memory` annotations: key = path, value = text, `scope` = the repository or directory (none for global). `view` is a query; `create` is a write or revision; `str_replace` and `insert` are a revision on the current base, refused as `conflict` when the base moved; `delete` is a withdrawal; `rename` is a revision with the new key. It is a thin mapping on the MCP side, so an agent's memories become queryable across sessions, machines and agents with no second store. Harness memory files (`CLAUDE.md`, `AGENTS.md`, a harness's memory directory) are captured inputs with their revision history ([capture.md](capture.md)); Polylogue does not inject them, and an annotation cites one by its `document:` reference.

**Methods** are `method` annotations: a procedure with inputs, checkable conditions, steps, an expected result and a stop rule, brought in by `import`. Adopted (an accepted judgment) and used (a delivery that included it, or an annotation `derived_from` it) are separate facts.

**Where a method applies** is the computed field `applies:REF` on method annotations: each condition's check is evaluated against the scope REF names (`here`, a `repo:` or `cwd:`), giving `applicable` when every condition holds, `not_applicable` when one fails, and `unassessed` when a check cannot be decided from what was captured. `polylogue 'annotations where kind:method applies:here'` lists the methods that apply in this repository; reading a method shows each condition with its result; the brief for a directory lists the methods that apply there ([agent-context.md](agent-context.md#the-brief-for-a-directory)).

## Storage

Standalone, the annotation and alternate tables live in `archive.db` and are backed up with it; there is no separate file of user state. In primary mode every authored record travels to Sinex as a `polylogue.*` record and is applied locally from the confirmed stream like every other record ([primary-mode.md](primary-mode.md)); current revisions follow the same rule on every machine.

Performance rules: current, conflict and withdrawn state are read from `revision_state`, which the writing transaction recomputes for the affected roots only, never by scanning all annotations per row; an annotation write never waits behind a first build for more than one writer group; markup extraction costs a declared, measured share of parsing throughput; annotation reads meet the latency classes in [latency.md](latency.md#latency-classes).

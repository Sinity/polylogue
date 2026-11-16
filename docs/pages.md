# Pages

A page is a named, ordered arrangement over the archive: a group of sessions, messages, annotations, live queries, kept results and other pages, arranged so the current state of a piece of work reads without scrolling back through transcripts. A project overview, an investigation, a decision record, a wiki article, a workspace and a bundle of context are the same arrangement under different templates. People and agents both work through pages, and pages are optional: someone who never makes one loses nothing but the arrangement, and nothing else in Polylogue needs one.

This page owns the page model (pages, slots, sections, order, modes, red links), members and aliases, open items and the decision box, staleness and circle markers, layouts, templates and shipped pages, how the verbs build and read pages, time and checkpoints, the working copy, pages as briefs and bundles, investigations, experiments, reply placement and the laws. Everything on a page is an annotation ([annotations.md](annotations.md)); there are no page tables. The units `pages` and `slots` and their fields are [query.md](query.md#units-and-fields); the views are listed in [reading.md](reading.md#views); the page markers are [markup.md](markup.md#pages); the web's page views are [web-views.md](web-views.md#pages).

## The model

- **A page holds references, never copies.** Its content is the archive itself, shown in place, so a page never goes stale against what it cites and reading it never calls a model.
- **`page`** is an annotation keyed by the page's name, with value `{title, layout: linear|board|table, template?: page:<root>@<rev>, params?: {name: value}}`. It may link `about` its **subject**: a `question` for an investigation, a `decision` for a decision record, `repo:` or `cwd:` for a project page, a session or a file. It may carry `scope` links (`repo:`, `cwd:`), which the brief uses to find pages for a session. One page is current per name.
- **Names** are lowercase words joined by `-`, with `/` for namespaces (`morrow/architecture`); names under `template/` are templates. A name is resolved when a line is read; stored references hold the root (`page:<root>`), so renaming the page, a revision with a new key, breaks no reference. Withdrawing a page takes it out of listings, and every reference to it still resolves, shown as withdrawn; withdrawing the withdrawal restores it.
- **`slot`** is one placed item, an annotation linked `slot_of page:<root>`, with value `{parent?, pos, mode, heading?: {text, level}, query?, caption?, x?, y?, w?, h?}`. There are three kinds of slot:
  - a **reference slot** links `about` what it shows: any reference ([references.md](references.md#forms)), including a session, a message range, a block, a span, an annotation, an alternate, a kept result, a file at a commit, a `url:`, an `sx:` reference or another page;
  - a **query slot** holds `query` (query text or `query:<hash>`). It is live while it has no `about` link and shows the query's current rows; it is frozen when `about` names the `result:` kept from it, and the query stays beside it so the slot can be kept again;
  - a **heading slot** holds `heading`; headings are the page's sections.
- **The section tree.** A slot's `parent` is the heading slot it sits under; no parent means the top of the page. Renaming a section is one revision of its heading; moving a heading carries its children in order; two sections may share a title without becoming one. Lines and markers name a section by its title (`--section Findings`, `section=Findings`), resolved when written to the current heading slot with that text, case-insensitively. A title that matches no heading creates one at the end of the page; two matches are refused `ambiguous`, listing both.
- **Order.** `pos` is a fractional key from the common `authored::order`, so inserting between two siblings never rewrites either. Two writers inserting into the same gap get distinct keys; equal keys, which only imports can produce, sort by slot root.
- **Modes.** `current` shows the current revision of an annotation or alternate, or a waiting reference once it arrives. `fixed` shows exactly the revision named in `about` (`annotation:<root>@<rev>`). Captured content has no revisions, so its mode records only the editor's intent; its target state still shows "changed since" when a re-parse changed the text ([annotations.md](annotations.md#reading)).
- **Edits are annotation edits.** Editing the text a slot shows revises that annotation, so it changes on every page that shows it, and the reply says so. Moving a slot revises the slot; removing a slot withdraws the slot only, and what it showed survives. A move and a text edit by two writers both survive, because they touch different roots; two edits of one root on one base are a conflict ([annotations.md](annotations.md#current-revisions)). Placing changes only the page: one annotation can appear on several pages.
- **Red links.** A page name not yet created is a waiting reference: a slot showing `page:not-yet` is shown waiting and attaches, with no rewrite, when the page is created.
- **Writing on a page.** Prose written on a page is a note (scope `page:<root>`) and a slot showing it, in one request. The page is never a text of its own.

### Members

A **member** is an alias for a session: a `member` annotation whose key is the alias, `about` the session, optionally linked `role_of role:<root>` and `member_of page:<root>`. Without a page it is a global alias. Roles are [session-control.md](session-control.md#roles).

- `polylogue codex:019d0f2a note reviewer --as member --page flaky-ci --role reviewer` names the session `@reviewer` on `flaky-ci` as an instance of `role:reviewer`; the note's text is the alias, so aliases stay keys.
- **`@reviewer` is a selection** ([interfaces.md](interfaces.md#the-first-word)): it resolves to the global member of that alias, else to the one page member of it; several page members with one alias and no global one are refused `ambiguous`, listing their sessions. In a bracketed send written by a session that is itself a member of pages, those pages' members are tried first ([markup.md](markup.md#sends)).
- **An alias follows the session's successor.** It resolves to the newest session reached from its session through `continuation` and `resume` links, never across a fork ([sessions.md](sessions.md#session-links)), at the moment something is sent. Rebinding the alias later never changes the recipient of an earlier send.
- On a page, each member shows its resolved session's live state, its role, and its last activity on the page (the newest slot or shown annotation it wrote).

### Open items and the decision box

Both are computed when the page is read, from links ([annotations.md](annotations.md#reading)), never stored.

- **Open items** are the `question`, `goal` and `problem` annotations the page's slots show, or that are scoped to the page, whose work-item state is open (not answered, achieved, decided or withdrawn), plus anything blocked through a `blocks` link. Each carries "unresolved since T" with how far capture reaches. The shipped saved query `open` lists them across every page ([saved-queries.md](saved-queries.md#shipped-saved-queries)).
- **The decision box** shows the current `decision` annotations about the page's subject, or about the page itself when it has no subject. Each shows its value, alternatives, constraints and `reverse_if`, its author, the judgments of the judges the reader counts, its revision history, its circle marker and whether its revisit check currently matches. With no decision it shows `no decision yet`. A decision does not answer the subject question.

### Staleness and circle markers

Reading a page marks two things from stored revisions, with no model involved:

- **Staleness.** A `current`-mode note whose shown revision was written before later findings or decisions were placed on the same page is marked `written before 2 findings and 1 decision on this page`. Revising the note clears the mark. A cited text that changed is the separate target state "changed since".
- **Circle marker.** A decision or note whose current value equals an earlier, non-adjacent revision of the same root is marked `returned to the position of revision 2 (Nov 06)`. A → B → A is marked; A → B → C is not.

## Layouts

One set of slots renders in three layouts, chosen by the page's `layout`:

- **linear**: slots in tree order under their headings, like a wiki article; the terminal always renders this layout;
- **board**: slots placed by `x`, `y`, `w`, `h`, which only this layout reads;
- **table**: slots as rows, with their kinds, authors and target states.

A live query slot shows the query's current rows; a frozen query slot or a slot showing a kept result shows the stored rows, marked `stored rows, not re-run`. A nested page is a reference slot showing `page:<root>`. `polylogue page:flaky-ci read --to browser` opens the page in the workbench in its own layout ([web-views.md](web-views.md#pages)).

## Building and reading with the verbs

Page work uses the same verbs as everything else. `save` keeps rows, and a page is one more place to keep them; notes go onto a page the way markers put them there; page files go out and in with `read --to` and `import`; the table of contents is a view. No verb exists for pages alone.

| To | Line |
|---|---|
| make a page from a template | `polylogue page:flaky-ci note 'Why is CI flaky on Fridays?' --as page --template investigation` |
| put rows on it, one slot per row | `polylogue 'messages where "flaky" repo:morrow since:7d' pick save page:flaky-ci --section Evidence --fixed` |
| put a live query on it | `polylogue 'sessions where repo:morrow since:7d' save page:flaky-ci --live --section 'Recent work'` |
| write a note on it | `polylogue page:flaky-ci note 'weekday arithmetic used >= instead of >' --as finding --section Findings` |
| add a section | `polylogue page:flaky-ci note 'Risks' --as heading` |
| read it | `polylogue page:flaky-ci read` (linear, in the pager) |
| the table of contents an agent starts from | `polylogue page:flaky-ci read --view contents` |
| what changed since you last looked | `polylogue page:flaky-ci read --view changes --from seen` |
| the page as it stood | `polylogue page:flaky-ci read --at 2026-11-10` |
| edit it as a file | `polylogue page:flaky-ci read --to .page/` then `polylogue import .page/flaky-ci.md` |
| move a slot | `polylogue annotation:01JC7Q2M save page:flaky-ci --section Findings --after annotation:01JD4K8P` |
| remove a slot (what it showed stays) | `polylogue annotation:01JC7Q2M withdraw` |
| checkpoint it | `polylogue page:flaky-ci save` |
| hand it to an agent | `polylogue page:flaky-ci brief --for codex:019d0f2a` |
| name a member and ask it | `polylogue @reviewer send 'fold the findings into the overview' --page flaky-ci --section Overview` |
| be told when it changes | `polylogue page:flaky-ci save --watch` |
| pin something | `polylogue @2 save page:pinned` |
| list pages; find where a row is cited | `polylogue pages` · `polylogue 'pages where shows:codex:019d0f2a'` |

### Writing

- **`note TEXT --as page`** on `page:NAME` writes the page record: TEXT is its title; `--template T`, `--param KEY=VALUE` (repeatable), `--layout linear|board|table` and `--about REF` (the subject) set the rest. It creates the page, filling a red link, or revises it.
- **`note` on a page selection** places the note on that page, under `--section S`, or at the end of the page without it. On any other selection, `--page NAME --section S` writes the note about its rows and also places it. `--as heading` writes a heading slot, TEXT being its title, with `--level N` (default 2) and `--section S` for the heading it sits under. `--as member` is [Members](#members). The note and its slot are one request, so the undo line withdraws both.
- **`save page:NAME`** on a selection writes one slot per row: `--fixed` shows exact revisions, `--section S` and `--after SLOT` place them. `--live` places the selection's question as one live query slot instead, and `--kept` keeps the rows as a new kept result and places that as one slot. Saving a row the page already shows in that section moves its slot rather than placing it twice, and the selection may be a slot itself. A page that does not exist is created with its name as its title, from `--template T` when given. The word `page:NAME` after `save` is never a saved query's name, since saved names cannot contain `:` ([saved-queries.md](saved-queries.md#saved-names)).
- **`withdraw`** on a slot removes it from its page; on the page record it withdraws the page.
- **Agent edits apply directly**, attributed and reversible like every write, with no proposal step ([authorship.md](authorship.md)). An agent can set up a page's sections at the start of a task and fill them while it works, by lines or by markers in its own messages; a marker aimed at a page appears there live, and success is silent.

### Reading

- **`read`** of a page renders it linearly: headings, slots in order with what each shows, open items, the decision box, members with their live state, and the pages and annotations that link to it. Alternates follow the reader's setting for every shown annotation ([alternates.md](alternates.md)).
- **`--view contents`** is the generated table of contents, an agent's starting point on a page the way `AGENTS.md` is for a repository: the title, the members, each slot with its reference and one line (kind, author, state, title or first line), the open items, the lines that read further, and the position it read at. It is computed at each read, never stored or written by hand. Over MCP it is also the resource `polylogue://page/<root>/contents` ([agents.md](agents.md#resources)).
- **`--view changes --from POSITION|seen [--until POSITION]`** is the page diff: slots added, moved (old and new heading and neighbours), withdrawn and restored; shown annotations revised (with the text change), withdrawn or judged; targets whose text changed; and live query slots whose rows changed, when the page is watched. It is a function of the records between two positions, and an empty diff is empty. `seen` is the reader's seen mark ([reading.md](reading.md#ranges)).
- **`--at TIME|POSITION`** renders the page from the revisions current then. A position is exact on one archive; a time uses each writer's clock, so across machines it is approximate and says so.
- Every page read reports the position it read at, so the next `--from POSITION` is cheap.
- A template with parameters renders its live slots for the page's bound parameters.

## Time and checkpoints

- **Checkpoint.** `save` on a page selection writes a kept result of unit `slots`: the page's slots in tree order, each reference resolved to an exact revision and each live query slot kept as its own nested result. It prints `result:<id>`. Reading that result renders the page as it was, marked `checkpoint, stored rows, not re-run`; a slot can show a checkpoint like any kept result, and a finding can rest on one (`--derived-from result:<id>`). `page:<root>@<rev>` names a revision of the page record only (title, layout, subject, parameters).
- **Watching a page.** `polylogue page:flaky-ci save --watch` writes a watch named after the page over `annotations where referenced_by:page:flaky-ci`: slots and shown annotations, compared by kind, target, key and value, so revisions count as changes. It delivers notices like any watch ([operations.md](operations.md#notices)) and never starts or resumes an agent.

## The working copy

`read --to FILE.md`, or `read --to DIR` writing `DIR/<name>.md`, writes the page as one Markdown file with a marker before each block naming its slot and base revision:

```markdown
<!-- polylogue page 01JA7M3K name=flaky-ci position=48390: edit text, move blocks, add paragraphs; import checks it in -->
# Why is CI flaky on Fridays?

<!-- slot 01JB2D9F heading -->
## Evidence

<!-- slot 01JC7Q2M shows message:codex:019d0f2a:msg_01a2 fixed · captured text, read-only -->
> "next friday" evaluated on a Friday returns the same day

<!-- slot 01JD4K8P shows annotation:01JE5R1T@3 current · finding · also on 1 other page -->
weekday arithmetic used >= instead of >; strict mode fixes it

## Open questions
Does the parser cache survive a zone change?
```

`polylogue import .page/flaky-ci.md` recognises the header and checks the file in, as one request:

- a changed note becomes a revision on the base its marker names; when that base has moved, that block alone is refused `conflict`, naming the current revision, and the other blocks commit;
- a moved block becomes a slot move, its new heading and position computed from the markers around it;
- a new paragraph without a marker becomes a note and a slot under the heading above it, and a new `##` line a heading;
- an edited quote of captured text is refused for that block;
- a missing block is withdrawn only with `--withdraw-missing`.

The file is a working copy, never a second store; the per-block refusal is the same per-line refusal annotation lines get ([annotations.md](annotations.md#annotation-lines)).

## Templates and shipped pages

A template is a page named `template/<name>`. Instantiating it (`--template T` on `note --as page` or `save page:NAME`) copies its heading and query slots into the new page, each copy linked `derived_from` its template slot at that revision, and records the template's exact revision on the new page. Copies do not follow later template edits, so there is one source of structure per page. **Parameters** are the saved-query parameters ([saved-queries.md](saved-queries.md#query-objects)): a template declares `params: {repo: null, since: "30d"}`, its query slots use `$repo` and `$since`, and a page binds them with `--param repo=morrow`. Empty sections render `empty — add`; a filled Checks section is not a test result.

Shipped pages and templates are built into the binary and resolved by name, with no write at daemon start. Revising one writes an ordinary page under the same name, which then wins; the built-in stays readable as `page:NAME@builtin`.

| Page | What it is |
|---|---|
| `page:home` | the orientation screen: bare `polylogue` is this page rendered for a terminal ([terminal.md](terminal.md#the-orientation-screen)) |
| `page:pinned` | what `save page:pinned` collects; pinning a row is saving it here |
| `page:cost` | the Cost view as a page ([web-views.md](web-views.md#cost)) |

| Template | Sections, in order |
|---|---|
| `investigation` | Goal, Assumptions, Alternatives considered, Plan, Evidence, Findings, Open questions, Decision |
| `decision-record` | Context, Options, Decision, Consequences, Revisit when |
| `plan` | Goal, Assumptions, Alternatives considered, Plan, Steps, Checks, Expected artifacts, Decision |
| `project-understanding` | Overview, Components, and live query slots Decisions (decisions scoped to `$repo`), Open questions, Recent work (sessions in `$repo` since `$since`) |
| `experiment` | Hypothesis, Units, Arms, Assignment, Exposure, Outcomes, Analysis, Decision |
| `weekly-review`, `project-story`, `day-story`, `feature-story` | live query slots over `$repo` and `$since` (or `$day`, `$feature`): sessions, decisions, findings, costs and failing checks of the period |
| `workspace` | the web's saved tabs and lanes ([web.md](web.md#workspaces)) |
| `bundle` | a context bundle: the slots a recipient should read, with the page's title as its purpose and the manifest the brief prints |

## Pages elsewhere

- **Briefs.** `polylogue page:NAME brief` is the page as a brief: its contents view, open items and decision box under the brief's budget, and its slots' targets in budgeted depth. A page from the `bundle` template is a named context bundle, so `polylogue page:release-context brief --for codex:019d0f2a` builds it for that session. The session-start brief carries a "Pages here" block ([agent-context.md](agent-context.md#pages-here)).
- **Queries.** The units `pages` and `slots` list pages and their slots; `referenced_by:page:X` on any unit matches the rows a page places, at that unit's grain; the field `pages` on every unit lists the pages that cite a row ([query.md](query.md#units-and-fields)).
- **Markup.** `page=` and `section=` on any marker write the annotation and its slot in one request; `::heading` and `::slot` place sections and references ([markup.md](markup.md#pages)).
- **Sends.** `send … --page P --section S` asks for the reply in that section ([Reply placement](#reply-placement)).
- **Primary mode.** Pages and slots travel to Sinex as authored records like every annotation, and `sx:` slots resolve through Sinex ([primary-mode.md](primary-mode.md)).

### Reply placement

`polylogue @reviewer send 'fold the findings into the overview' --page flaky-ci --section Overview` records, on the delivery, the page and heading where a reply belongs. An annotation that replies to that delivery (`--replies-to delivery:<id>`, or `replies_to=` in a marker) and names no page of its own is placed in that section by the same request, authored by the reply's author. A marker's explicit `page=` wins. One reply is placed once: the slot's ID derives from the reply's ID, so re-parsing writes nothing new.

## Investigations

An investigation is a page from the `investigation` template whose subject is a `question`; it may start as a question and a selection of sessions and messages. There is no separate investigation record. Four facts stay separate on it: whether a reference still resolves to the text it cited (its target state), whether something supports a claim (`supports` links), the editorial choice to place it (the slot), and standing acceptance (an accepted judgment of a `lesson` or `method`). An intact reference does not show that a claim is true.

## Experiments

An experiment is a page from the `experiment` template, for real comparisons only.

- Its assignment is a `decision` on the page with value `{units: result:<id>, arms, method: list|seeded:<seed>, assignment: {unit: arm}}`, written before any exposure. A revision of the assignment after any assigned unit's exposure is refused.
- Outcomes are `outcome` annotations or judgments per unit. `polylogue 'slots where page:exp-1 | measure experiment'` reports, per arm, the interval `[satisfied/n, (satisfied+unresolved)/n]`, never a point estimate that drops unresolved units ([measures.md](measures.md#experiment)).
- Polylogue's own output uses causal words only about a page with a fixed assignment.

## Laws

Each law is a property with a test on the Morrow archive through the real daemon.

| Law | Property |
|---|---|
| page-holds-references | an investigation built from the template with a message slot, a kept result, a live query and a `::finding(page=…, section=Findings)` marker stores one page record, heading slots, reference slots and the finding, and no copy of the quoted message text |
| pages-optional | with no page in the archive every other behaviour is unchanged, the brief has no "Pages here" block, and the index has no table or index for pages alone |
| slot-edits | moving a slot writes one slot revision; removing the Evidence slot keeps the annotation it showed; editing a note through one page changes it on every page that shows it |
| section-tree | renaming a section writes one revision; moving a heading carries its children in order |
| section-on-demand | `::finding(page=X, section=Nope)` creates heading `Nope` at the end of X and places the finding under it; a second marker with `section=nope` reuses it |
| order-stable | a thousand random concurrent inserts from two writers, including into one gap, give a total order and rewrite no existing slot's position |
| checkout-checkin | `read --to`, an edit of one block and `import` write one revision; with that block's base moved it alone is refused; a moved block becomes a move, a new paragraph a note slot, an edited quote is refused |
| page-as-of | `read --at` a position before the finding shows Findings empty; `--at TIME` is labelled approximate when records from two machines are involved |
| page-diff | `read --view changes --from POSITION` lists exactly the changes after that position |
| page-checkpoint | after `page:X save` gives `result:R`, reading `result:R` reproduces the page exactly after later edits to its slots, its notes and the archive, and nothing runs again |
| contents-generated | `--view contents` lists every slot with its reference and one line, changes when a slot is added, ends with the position, and is never stored |
| rename-keeps-references | renaming a page leaves every stored `page:` reference and slot resolving to it |
| red-link | a slot showing `page:not-yet` is waiting, and attaches when that page is created, with no rewrite |
| subject-decision-box | the decision box shows exactly the current decisions about the subject; recording one does not answer the subject question |
| open-items-computed | answering a question on a page removes it from open items, and withdrawing the answer brings it back, with nothing stored about the page |
| circle-marker | a decision revised A → B → A shows `returned to the position of revision 1`; A → B → C shows nothing |
| staleness | a `current`-mode note written before a later finding was placed on its page shows `written before 1 finding`; revising the note clears it |
| reply-placement | a marker replying to a delivery sent with `--section Findings`, without `page=`, writes one slot under Findings; re-parsing writes none |
| alias-follows-successor | an alias whose session was resumed resolves to the resumed session, never across a fork; an earlier send keeps its recipient |
| watch-never-starts | a page watch firing while its member's session has ended produces a notice and no harness process |
| referenced-by-grain | `messages where referenced_by:page:X` returns the messages of a message-range slot and none of a session slot's messages; a page that nests itself terminates |
| template-params | `project-story` with `--param repo=morrow` and with `--param repo=other` gives two pages whose live slots answer different populations; editing the template changes neither |
| experiment-interval | an arm with 1 satisfied, 1 unsatisfied and 2 unresolved of 4 reports [1/4, 3/4]; revising the assignment after an exposure is refused |
| same-on-every-surface | each page write through the CLI, MCP, markup and the web produces the same records apart from `via` |

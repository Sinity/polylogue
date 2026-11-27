# Pages

A page is a named, ordered arrangement of the archive: sections, and under them slots, each holding one thing. A slot holds a paragraph of text, an excerpt of captured text, a live or kept query, a figure, a session or tool call shown as a card, or another page's text. Text is Markdown, and inside it a sentence can link, cite the exact words it rests on, show a value from a query, or include another note or section. A page is how the current state of a piece of work reads without scrolling back through transcripts: a project overview, an investigation, a decision record, a guide agents read at session start, a weekly account, a comparison, a reading list and a context bundle are the same arrangement under different templates.

People and agents build pages with the same verbs. Pages are optional: someone who never makes one loses nothing but the arrangement, and nothing else in Polylogue needs one. Reading a page never calls a model, so a page costs nothing to read.

This page owns the page model (pages, slots, sections, order, modes, forms), writing and reading pages with the verbs, query cells, authorship and proposals, time and checkpoints, names and renames, finding pages, members, reply placement and the laws. Its companions:

- [page-text.md](page-text.md): the text of a page, Markdown with references, and the four relations (link, citation, embed, inclusion), citations and their checks, backlinks, what happens when a source changes, and grounding.
- [page-working-copy.md](page-working-copy.md): the page as one Markdown file, editing it in your editor with `edit`, check-in and its conflicts, importing Markdown, and writing pages out as files.
- [page-templates.md](page-templates.md): templates and their parameters, the shipped pages and templates, investigations and experiments.
- [web-pages.md](web-pages.md): pages in the web workbench, citing while reading, editing in place and the history view.

Everything on a page is an annotation ([annotations.md](annotations.md)); there are no page tables. The units `pages`, `slots` and `links` and their fields are [query.md](query.md#units-and-fields); the views are listed in [reading.md](reading.md#views); the page markers are [markup.md](markup.md#pages).

## The model

A page holds references, never copies of captured text. Its content is the archive itself, shown in place, and when a cited source changes the page shows it ([page-text.md](page-text.md#when-a-source-changes)). Every page, slot and shown note is an annotation with revisions and a computed author ([authorship.md](authorship.md)); nothing on a page is deleted.

### Page

- **The record.** `page` is an annotation keyed by the page's name, with value `{title, layout: linear|board|table, template?: page:<root>@<rev>, params?: {name: value}, brief: off|contents|full}`. It may link `about` its **subject** (a `question` for an investigation, a `decision` for a decision record, `repo:` or `cwd:` for a project page, a session or a file) and carry `scope` links (`repo:`, `cwd:`) that briefs use to find pages for a session. One page is current per name.
- **Names** are lowercase words joined by `-`, with `/` to group names (`morrow/architecture`); names under `template/` are templates. The title is free text and is what readers see. A name is resolved when a line is read; stored references hold the root (`page:<root>`), so renaming a page breaks no reference ([Names and renames](#names-and-renames)).
- **`brief`** says how the page enters briefs for its scope: `off`, `contents` (the default: listed with its open items and changes) or `full` (its text, within the brief budget) ([Pages elsewhere](#pages-elsewhere)).
- **Lifecycle.** A page is created, revised and withdrawn. Withdrawing takes it out of listings; every reference to it still resolves, shown as withdrawn; withdrawing the withdrawal restores it. `page:<root>@<rev>` names a revision of the page record itself (title, layout, subject, parameters, brief).
- **Red links.** A page name not yet created is a waiting reference. A link or slot naming `page:not-yet` shows as a red link and attaches, with no rewrite, when the page is created.

### Slot

`slot` is one placed item: an annotation linked `slot_of page:<root>`, whose `about` link names what it holds (any reference, [references.md](references.md#forms)), with value `{parent?, position, mode: current|fixed, form?, heading?: {text, level}, query?, caption?, x?, y?, w?, h?}`. What kind of slot it is follows from its fields:

| Slot | Fields | Holds | Shown as (default form) |
|---|---|---|---|
| text | `about` an authored annotation, form `text` | a note, finding, decision, question, claim … written on or brought to the page | a paragraph in the reading column, with its citations |
| heading | `heading` | a section title | a heading; the section is the heading and what sits under it |
| excerpt | `about` a span, message, block, tool call output or file lines | captured text quoted exactly | a quotation with its source line and check mark |
| embed | `about` a session, message range, tool call, attachment, kept result, figure, `url:`, `sinex:<kind>/<id>` reference or page | a thing shown in place | a card, a table, a value, a chart, a transcript range |
| query | `query`: query text, `query:<hash>` or `query:NAME(params)`; `about` empty while live, the `result:` kept from it when kept | a live question or its kept answer | a table, list, count, value or chart, with its run |
| included text | `about` another annotation, a page section or a page, form `text`, its home elsewhere | text that lives elsewhere, in step with its source | text flowing in the column, its origin in the margin |

- **Writing on a page** writes the note and its slot in one request; the note carries `scope page:<root>`, so the page is its home. A note written elsewhere and placed on a page is an inclusion ([page-text.md](page-text.md#inclusions)).
- **One paragraph, one slot.** A paragraph, list, quotation, table or fenced block is one top-level Markdown element, and a page's prose is a sequence of text slots each holding one element. Each paragraph can then be cited, included, judged, revised and moved on its own. A note with several elements (a long handoff note placed on a page) stays one slot.
- **Modes.** `current` shows the current revision of what the slot holds, and a waiting reference once it arrives. `fixed` shows exactly the revision in `about` (`annotation:<root>@<rev>`, `alternate:…@<rev>`, a page section at a checkpoint). Captured content has no revisions, so its mode records the writer's intent; it still shows `source changed` when a re-parse changed the cited text.
- **Edits are annotation edits.** Editing the text a slot shows revises that annotation, so it changes on every page that shows it, and the reply says so. Moving a slot revises the slot; removing a slot withdraws the slot only, and what it showed survives. A move and a text edit by two writers both survive, because they touch different roots; two edits of one root on one base are a conflict ([annotations.md](annotations.md#current-revisions)). One annotation can appear on several pages.

### Sections and order

- **The section tree.** A slot's `parent` is the heading slot it sits under; no parent means the top of the page. Renaming a section is one revision of its heading; moving a heading carries its children in order; two sections may share a title without becoming one.
- **Naming a section.** Lines and markers name a section by its heading text (`--section Findings`, `section=Findings`), resolved when written to the current heading slot with that text, case-insensitively. A title that matches no heading creates one at the end of the page; two matches are refused `ambiguous`, listing both.
- **Anchors.** `page:NAME#ANCHOR` addresses a section or slot: ANCHOR is a heading's slug (`#open-questions`) or a slot's short ID (`#7q2m`). It is stored as `page:<root>#<slot root>` and printed with the current slug, so renaming a heading breaks no link ([references.md](references.md#forms)).
- **Order.** `position` is a fractional key from the common `authored::order`, so inserting between two siblings never rewrites either. Two writers inserting into the same gap get distinct keys; equal keys, which only imports can produce, sort by slot root.

### Forms

A form says how a slot shows what it holds. Each kind of target has a default and a closed set of forms. `save page:NAME --as FORM` and the title part of an embed (`![…](REF "FORM")`) set it. A page never cuts silently: a form that shows part of something says how much it left out and links the rest.

| Target | Default form | Other forms | Shows |
|---|---|---|---|
| annotation (note, finding, decision, question …) | `text` | `card` | `text` flows in the column with its citations; `card` frames it with kind, author and state |
| span, block | `excerpt` | `card` | the exact text, a source line, the quote check |
| message | `excerpt` | `card`, `full` | the whole message text; over 40 lines, head and tail with the exact omitted count and **read whole** |
| message range | `full` | `card` | a dialogue reading of the range (`--view dialogue`), within the slot's share of the budget |
| session | `card` | `full`, `excerpt` | card: title, origin, when, cost, outcome, notes, live state; `full` the dialogue view; `excerpt` its first prompt and last reply |
| tool call | `card` | `full` | the one-line form; `full` the family body ([transcripts.md](transcripts.md)) |
| attachment | `full` | `card` | images inline; other files as a chip with **Download** |
| kept result, `result:<id>#<row>` | `table` | `list`, `count`, `chart`, `compare`, `value` | the stored rows, marked `stored rows, not run again` |
| query (live) | `table` | `list`, `count`, `chart`, `value` | the rows at this read, its run and position; `pending` until it answers |
| figure (a saved query with a denominator) | `value` | `chart`, `table` | value, denominator, unknown count, exactness ([saved-queries.md](saved-queries.md#figures)) |
| page | `card` | `text` | `card`: title, open items, last change; `text` includes the whole page |
| page section `page:X#anchor` | `text` | `card` | the heading and what sits under it |
| `file:PATH@COMMIT#La-Lb` | `excerpt` | `card` | the lines with their numbers |
| `url:` | `card` | — | the stored text of the link; reading never fetches it |
| `sinex:<kind>/<id>` | `card` | `excerpt` | resolved through Sinex in primary mode, `unavailable` otherwise |

`compare` is the side-by-side view ([reading.md](reading.md#compare-and-blind)); on a kept result it may be read `--blind`.

## Writing

Page work uses the same verbs as everything else. `edit` opens a page in your editor; `save` keeps rows, and a page is one more place to keep them; `note` writes on a page; files go out and in with `read --to` and `import`; the contents is a view. No verb or command group exists for pages alone: every page act is a verb on a `page:` selection.

| To | Line |
|---|---|
| open a page in your editor (a new name makes a new page) | `polylogue page:flaky-ci edit` |
| edit one note, or several, in your editor | `polylogue annotation:01JE4K8P edit` · `polylogue 'annotations where kind:lesson repo:here' edit` |
| make a page from a template without the editor | `polylogue page:flaky-ci note 'Why is CI flaky on Fridays?' --as page --template investigation` |
| put a search hit on it as an excerpt | `polylogue 'next friday'` then `polylogue @2.1 save page:flaky-ci --section Evidence` |
| put rows on it, chosen in the picker | `polylogue 'messages where "flaky" repo:morrow since:7d' pick save page:flaky-ci --section Evidence` |
| put sessions on it as cards | `polylogue @1-6 save page:reading/agents --as card` |
| put a live question on it | `polylogue 'sessions where repo:morrow since:7d' save page:flaky-ci --live --section 'Recent work'` |
| put a kept chart on it | `polylogue 'tool_calls since:30d \| measure usage' save page:report/first-build --kept --as chart --section Results` |
| write a short paragraph on it | `polylogue page:flaky-ci note 'weekday arithmetic used >= instead of >' --as finding --section Findings --evidence span:message:codex:01a20000-0005:msg_01a20000a11#c=88-121` |
| write cited prose from a file | `polylogue page:flaky-ci note - --section Findings` with the Markdown on standard input |
| get citation lines to paste | `polylogue @2-4 read --format citation --to clipboard` |
| add a section | `polylogue page:flaky-ci note 'Risks' --as heading` |
| move a slot | `polylogue page:flaky-ci#7q2m save page:flaky-ci --section Findings --after page:flaky-ci#4hq1` |
| remove a slot (what it showed stays) | `polylogue page:flaky-ci#7q2m withdraw` |
| check out and in (agents, scripts) | `polylogue page:flaky-ci read --to .page/` then `polylogue import .page/flaky-ci.md` |

- **`edit`** with no text, on a terminal, writes the selection to a private temporary file as its working copy (a page; one note as its text; several notes as one file), opens `$VISUAL` or `$EDITOR`, and checks the file in when the editor exits. On a page name that does not exist, it opens a new working copy whose header lists the templates. Refused parts reopen with their reasons. Off a terminal and over MCP it is refused with the `read --to` and `import` lines ([page-working-copy.md](page-working-copy.md#editing-in-your-editor)). With text, `edit` revises each selected annotation as everywhere ([annotations.md](annotations.md#edit)).
- **`note TEXT --as page`** on `page:NAME` writes the page record: TEXT is its title; `--template T`, `--param KEY=VALUE` (repeatable), `--layout linear|board|table`, `--about REF` (the subject), `--scope REF` and `--brief off|contents|full` set the rest. It creates the page, filling a red link, or revises it.
- **`note` on a page selection** writes on the page when `--section S` names where it goes: the note and its slot, in one request, so the undo line withdraws both. Without `--section` it writes about the page, which is [discussion](#discussion). `--as heading` writes a heading slot, TEXT being its title, with `--level N` (default 2) and `--section S` for the heading it sits under. On any other selection, `--page NAME --section S` writes the note about its rows and also places it. `--as member` is [Members](#members).
- **TEXT is Markdown with references.** Links, footnote citations, embeds, values and query cells written in a note's text become link rows in the same write; a citation may give a source and a quotation instead of a span, and the write finds the exact span ([page-text.md](page-text.md)). `-` reads the text from standard input.
- **`save page:NAME`** on a selection writes one slot per row. `--as FORM` sets the [form](#forms); `--fixed` shows exact revisions; `--section S` and `--after SLOT` place them; `--live` places the selection's question as one live query slot instead; `--kept` keeps the rows as a new kept result and places that as one slot. `@N.M` selects the `span:` of hit M of row N, so a search hit goes on as an excerpt of exactly the matched text. Saving a row the page already shows in that section moves its slot rather than placing it twice, and the selection may be a slot itself. A page that does not exist is created with its name as its title, from `--template T` when given. The word `page:NAME` after `save` is never a saved query's name, since saved names cannot contain `:` ([saved-queries.md](saved-queries.md#saving-to-a-page)).
- **`withdraw`** on a slot removes it from its page; on the page record it withdraws the page.
- **Agents write the same way**, attributed and reversible like every write; their writes apply directly, and the page marks their text proposed until the operator judges it ([Authorship and proposals](#authorship-and-proposals)). A marker aimed at a page in an agent's own output appears there live ([markup.md](markup.md#pages)).

### Discussion

Discussion is annotations about a page or its parts that are not placed on it: challenges, qualifications, questions and replies. `note` on `page:X` without `--section` (and not `--as heading` or `--as page`) writes about the page and places nothing; on `page:X#ANCHOR` it writes about that section or slot. Discussion shows in the margin beside what it is about and in the page's discussion panel ([web-pages.md](web-pages.md#the-page-view)), and in the terminal as a count under the slot with the line that reads it.

- `polylogue page:flaky-ci note 'should this cover Codex too?' --as question`
- `polylogue page:flaky-ci#7q2m note 'which test?' --as question`

## Reading

- **`read`** of a page renders it in the linear layout: the header (title, template, subject, the counts `14 citations · 1 source changed · 2 need citations · 6 proposals`), then headings and slots in order with what each shows in its form, the decision box, open items, members with their live state, and the **Cited and linked** block of everything that places, links, cites or includes the page ([page-text.md](page-text.md#backlinks)). Citations show as footnotes under their slot; included text flows with its origin; stubs render `empty — add`. Alternates follow the reader's setting for every shown and included annotation ([alternates.md](alternates.md)).
- **`--view contents`** is the generated contents, an agent's starting point on a page the way `AGENTS.md` is for a repository: the title, the members, each slot with its anchor, reference and one line (kind, author, state, grounding, title or first line), the open items, the lines that read further, and the position it read at. It is computed at each read, never stored or written by hand. Over MCP it is also the resource `polylogue://page/<root>/contents` ([agents.md](agents.md#resources)).
- **`--format markdown`** of a page is its working copy ([page-working-copy.md](page-working-copy.md#the-format)); `--format citation` of any selection prints footnote definitions ready to paste ([page-text.md](page-text.md#citation-lines)).
- **Paging.** A page reads within its budget like any reading; each embed and inclusion renders within its own share of the page budget and says what it left out; over MCP a page read pages at 4,000 tokens with a cursor. A page with 50,000 slots pages its contents and its rendering by cursor; nothing truncates.
- Every page read reports the position it read at, so the next `--from POSITION` is cheap. `polylogue page:flaky-ci read --to browser` opens it in the workbench in its own layout ([web-pages.md](web-pages.md)).

### Layouts

One set of slots renders in three layouts, chosen by the page's `layout`:

- **linear**: slots in tree order under their headings, like a wiki article; the terminal always renders this layout;
- **board**: slots as cards placed by `x`, `y`, `w`, `h`, headings as frames, which only this layout reads;
- **table**: slots as rows with kind, target, form, mode, target state, author, grounding and last change.

### Open items and the decision box

Both are computed when the page is read, from links ([annotations.md](annotations.md#reading)), never stored.

- **Open items** are the `question`, `goal` and `problem` annotations the page's slots show, or that are scoped to the page, whose work-item state is open (not answered, achieved, decided or withdrawn), plus anything blocked through a `blocks` link. Each carries "unresolved since T" with how far capture reaches. The shipped saved query `open` lists them across every page ([saved-queries.md](saved-queries.md#shipped-saved-queries)).
- **The decision box** shows the current `decision` annotations about the page's subject, or about the page itself when it has no subject. Each shows its value, alternatives, constraints and `reverse_if`, its author, the judgments of the judges the reader counts, its revision history, its circle marker and whether its revisit check currently matches. With no decision it shows `no decision yet`. A decision does not answer the subject question.

### Staleness and circle markers

Reading a page marks two things from stored revisions, with no model involved:

- **Staleness.** A `current`-mode note whose shown revision was written before later findings or decisions were placed on the same page is marked `written before 2 findings and 1 decision on this page`. Revising the note clears the mark. A cited text that changed is the separate state `source changed`.
- **Circle marker.** A decision or note whose current value equals an earlier, non-adjacent revision of the same root is marked `returned to the position of revision 1 (Nov 06)`. A → B → A is marked; A → B → C is not.

## Queries on pages

A query on a page is a slot: the question as text (a ```` ```query FORM ```` fenced code block in Markdown, [page-text.md](page-text.md#embeds)) or a saved query with parameters (`![](query:failing(repo=$repo) "table")`).

- **A live cell** runs each time the page is read, at the page read's position, concurrently with the other cells and under its own deadline. It shows `pending` until it answers and never holds the page back. It shows its run, its `query:<hash>` and its row count, and **open as search** goes to the full listing. Page parameters bind `$name` ([page-templates.md](page-templates.md#templates)).
- **A kept answer** is the same cell fixed to stored rows: `about` names the `result:<id>`, and the question stays beside it. **Keep** (`save page:X --kept`) stores the current rows; **run again** keeps a new result and moves the slot to it, the old one staying in history and readable; **compare with now** lists rows added and gone against a fresh run without writing anything.
- **Forms**: `table` (the default), `list` (references with one line each), `count`, `value` (a figure: value, denominator, unknown count, exactness), `chart` (measures and aggregates drawn as the Analysis view draws them; on a terminal a one-line spark of the series and the table) and `compare` (kept results only). A measure or figure on a page is a cell: `polylogue errors-30d save page:report/first-build --as value` places it kept, since a figure's rows are not archive things, and `--live` keeps it running ([saved-queries.md](saved-queries.md#saving-to-a-page)).
- **A value inside a sentence is fixed when the sentence is written**, so the sentence keeps saying what it said ([page-text.md](page-text.md#values-in-sentences)).
- **Cost and speed.** A cell costs its query's latency class. Results are cached by query hash and archive position and dropped when the change feed touches their sources ([warm state](latency.md#warm-state)), so reopening a page runs nothing that has not changed. A `similar:` cell whose vectors are not built shows `not built` with the `embed` line; spending stays an explicit act.
- **Saved queries.** Revising a saved query changes every live cell that names it, as it changes every other use; a cell that must not move names `query:NAME@N` or keeps its rows. A saved query's reading lists the pages that show it (`polylogue 'links where target:query:failing'`).

## Authorship and proposals

Authorship is computed and shown on every slot ([authorship.md](authorship.md)). Agents may write anything the operator may; their writes apply directly, nothing waits for a judgment, and a judgment is never a gate. What a page adds is a display of what the operator has not yet shaped:

- **Proposed.** A text slot whose current revision is authored by anyone but the operator and has no judgment from the counted judges (`judged_by:`, default the operator) renders **proposed**: a dashed rule, the author badge, **accept** (`judge accept` on that revision), **reject** (`judge reject`) and **edit** (an operator revision, which makes the text his; the agent's stays in history). This includes text the operator asked an agent to write: his judgment, not his request, is the record of acceptance.
- **Tracked change.** When an earlier revision of the same annotation was accepted and the current one is proposed, the page shows the accepted text with the proposed change as an inline word diff, with accept and reject beside it. Briefs and exports carry the accepted text and say `1 proposed change`. This is a reading of existing judgments, not a second current revision: the current revision is still the newest, and `read --view changes` and queries see it.
- **Rejected** text folds to one line (`rejected by operator · show`) and stays in history and in `withdrawn:false` queries; withdrawing it is a separate act.
- **The page's review queue** is `review` narrowed to the page: `polylogue 'annotations where referenced_by:page:flaky-ci NOT author:operator judged:unknown' pick judge accept`, where the picker's judge keys record accept, reject and defer row by row ([terminal.md](terminal.md#keys)). The web header's `6 proposals` walks the same set ([web-pages.md](web-pages.md#proposals)).
- **The operator's own text** is never marked proposed and never flagged uncited: his statements are the record ([page-text.md](page-text.md#grounding)).

An agent sees how its text fared in `read` and in its next brief: `your 6 paragraphs on page:flaky-ci: 4 accepted, 1 rejected, 1 proposed` ([agent-context.md](agent-context.md#pages-here)).

## Time and checkpoints

Every slot and every shown annotation has its own revisions; a page's history is computed from them, never stored.

- **`--view changes --from POSITION|seen [--until POSITION]`** is the page diff: slots added, moved (old and new heading and neighbours), withdrawn and restored; shown annotations revised (with a word diff), judged or withdrawn; citations whose source changed; and live query slots whose rows changed where a baseline exists (a watch or a checkpoint). It is a function of the records between two positions, and an empty diff is empty. `seen` is the reader's seen mark ([reading.md](reading.md#ranges)).
- **`--at TIME|POSITION`** renders the page from the revisions current then. A position is exact on one archive; a time uses each writer's clock, so across machines it is approximate and says so. One slot's history is `annotation:<root> read`.
- **Checkpoint.** `save` on a page selection writes a kept result of unit `slots`: the page's slots in tree order, each reference resolved to an exact revision and each live cell kept as its own nested result. It prints `result:<id>`. Reading that result renders the page as it was, marked `checkpoint, stored rows, not run again`; a slot can show a checkpoint like any kept result, and a finding can rest on one (`--derived-from result:<id>`). A checkpoint is the permanent link to a version; `--at` and the web's `?at=` are views, not references.
- **Staleness against new findings** is the marker above; a page's text written before a source changed shows the source change on its citation ([page-text.md](page-text.md#when-a-source-changes)).
- **Watching a page.** `polylogue page:flaky-ci save --watch` writes a watch named after the page over `annotations where referenced_by:page:flaky-ci`: slots and shown annotations, compared by kind, target, key and value, so revisions count as changes, and a citation whose source changed is one of its events. It delivers notices like any watch ([operations.md](operations.md#notices)) and never starts or resumes an agent.

## Names and renames

- **One name per page.** Renaming is a revision of the page record with a new key: change `page:` in the working copy's header (`polylogue page:old-name edit`), or revise the record with `--key`, its text being the title (`polylogue page:old-name edit 'Why is CI flaky on Fridays?' --key flaky-ci`). `note TITLE --as page` on a new name is never a rename; it makes a second page.
- **Old names keep working.** A name that a page carried in an earlier revision, and that no current page carries, resolves to that page, and the reply says `page:old-name was renamed to page:flaky-ci`. Completion offers current names only. This is the name's history, not an alias: there is no second name per page.
- **Namespaces.** `/` groups names for listing and for the web's left rail; it carries no access rule and no inheritance (`polylogue 'pages where name:morrow/*'`).
- **Categories** are tags on pages; a category's index is a query, which a page can show as a live cell: `polylogue page:flaky-ci tag topic:parsing`, `polylogue 'pages where tag:topic:parsing'`.

## Finding pages

| To | Line |
|---|---|
| every page, by prefix and tag, with recent edits and wanted pages | `polylogue page:index read` |
| list pages | `polylogue pages` |
| recent changes to pages, newest first, with author and page | `polylogue edits` · `polylogue 'annotations where referenced_by:page:* since:seen | sort by at desc'` |
| wanted pages: names linked or placed that no page carries, with how many link to each | `polylogue wanted` |
| orphaned pages: no page places, includes or links them | `polylogue 'pages where linked_from:0'` |
| pages whose citations changed, and agent text with no citation | `polylogue changed-sources` · `polylogue uncited` |
| pages that mention words, in titles or in the text they show | `polylogue 'pages where "zone change"'` |
| what links to, places or cites a page or anything else | `polylogue 'links where target:page:flaky-ci'` · `polylogue 'pages where shows:codex:019d0f2a'` |
| templates | `polylogue 'pages where name:template/*'` |

`edits`, `wanted`, `changed-sources` and `uncited` are shipped saved queries ([saved-queries.md](saved-queries.md#shipped-saved-queries)). A plain-word search also names the pages that match ([search.md](search.md)). The watchlist is a watch on each page, and the home page's section Pages changed since you looked is the same query with `since:seen`.

## Templates and shipped pages

A template is a page named `template/<name>` whose heading and query slots a new page copies; shipped pages (`page:home`, `page:index`, `page:pinned`, `page:cost`) and the shipped templates are built into the binary. Templates, their parameters, the shipped set, investigations and experiments are [page-templates.md](page-templates.md).

## Pages elsewhere

- **Briefs.** `polylogue page:NAME brief` is the page as a brief: its contents view, open items and decision box, then its text and its slots' targets in budgeted depth. Pages in a session's scope enter its session-start brief by their `brief` setting: `full` pages as their text, every line cited, after the operator's own notes and before recent sessions (a page that does not fit is given as its contents plus the line that reads it); `contents` pages listed with open items, changes since that role last looked and `N sources changed`; `off` pages not at all ([agent-context.md](agent-context.md#pages-here)). Briefs carry accepted text, with `N proposed changes` where an agent's revision is waiting.
- **Agents.** An agent uses a page when the page is in its scope or named to it (in a send, a brief or the task it was given). It writes with the same steps as everything else (`note` with Markdown text, `page` and `section`; `save` with `name: "page:X"`, `section` and `as`; `import` of a working copy it wrote with its file tools), drafts by citing what it relies on, and sees its text proposed until judged ([agents.md](agents.md), [page-text.md](page-text.md#grounding)). `polylogue://page/<root>/contents` is its starting point on any page; `--view changes --from seen` is the re-entry read.
- **Context bundles.** A page from the `bundle` template is a named context bundle, so `polylogue page:release-context brief --for codex:019d0f2a` builds it for that session.
- **Queries.** The units `pages`, `slots` and `links` list pages, their slots and every reference written in text or placed; `referenced_by:page:X` on any unit matches the rows a page places, at that unit's grain (nested pages followed, cycles cut); `pages` and `cited_by` on every unit list the pages that place a row and the citations of it or inside it ([query.md](query.md#units-and-fields)).
- **Judgments** act on text slots' annotations; a judging pass over a page's text is a pass whose population is `polylogue 'annotations where referenced_by:page:flaky-ci' save` ([judging.md](judging.md)).
- **Alternates.** A page read follows the enabled alternates for every shown and included annotation, marked where they replace the original. Translating a page is one request: `polylogue page:flaky-ci note '' --as translation:pl --parts parts.jsonl`, each part's `of` a shown annotation. A page in two languages is one page with alternates, never two pages ([alternates.md](alternates.md)).
- **Markup.** `page=` and `section=` on any marker write the annotation and its slot in one request; `::heading` and `::slot` place sections and references; a marker's body is Markdown with references, its footnote definitions as indented continuation lines ([markup.md](markup.md#pages)).
- **Sends.** `send … --page P --section S` asks for the reply in that section ([Reply placement](#reply-placement)).
- **Primary mode.** Pages, slots and their notes travel to Sinex as authored records like every annotation, applied from the confirmed stream; `sinex:<kind>/<id>` references link, cite and embed Sinex events and materials, resolved through Sinex ([primary-mode.md](primary-mode.md)).

## Members

A **member** is an alias for a session: a `member` annotation whose key is the alias, `about` the session, optionally linked `role_of role:<root>` and `member_of page:<root>`. Without a page it is a global alias. Roles are [session-control.md](session-control.md#roles).

- `polylogue codex:019d0f2a note reviewer --as member --page flaky-ci --role reviewer` names the session `@reviewer` on `flaky-ci` as an instance of `role:reviewer`; the note's text is the alias, so aliases stay keys.
- **`@reviewer` is a selection** ([interfaces.md](interfaces.md#the-first-word)): it resolves to the global member of that alias, else to the one page member of it; several page members with one alias and no global one are refused `ambiguous`, listing their sessions. In a bracketed send written by a session that is itself a member of pages, those pages' members are tried first ([markup.md](markup.md#sends)).
- **An alias follows the session's successor.** It resolves to the newest session reached from its session through `continuation` and `resume` links, never across a fork ([sessions.md](sessions.md#session-links)), at the moment something is sent. Rebinding the alias later never changes the recipient of an earlier send.
- On a page, each member shows its resolved session's live state, its role, and its last activity on the page (the newest slot or shown annotation it wrote).

## Reply placement

`polylogue @reviewer send 'fold the findings into the overview' --page flaky-ci --section Overview` records, on the delivery, the page and heading where a reply belongs. An annotation that replies to that delivery (`--replies-to delivery:<id>`, or `replies_to=` in a marker) and names no page of its own is placed in that section by the same request, authored by the reply's author. A marker's explicit `page=` wins. One reply is placed once: the slot's ID derives from the reply's ID, so re-parsing writes nothing new.

## Speed

Budgets of page operations on a warm daemon, at the 95th percentile, beside the classes of [latency.md](latency.md#latency-classes):

| Operation | Budget | How |
|---|---|---|
| contents view of a 300-slot page | 50 ms | one range read of the page's `slot_of` links, the in-memory map of current revisions, one batched title read per kind |
| page open to first paint, 300 slots, 60 citations | 150 ms | the same, plus batched text reads of the cited and excerpted messages, from which citation states are computed; live cells stream in after |
| a live cell | its query's latency class | concurrent, under its own deadline, `pending` until it answers; cached by query hash and position |
| a paragraph write with five references | 30 ms to commit; within 1 s in every open view | the text is parsed and its link rows written in the write's transaction; the change feed reaches the views |
| cite by quote within one session | 100 ms | the session's text index narrows to candidate messages, then an exact match over their text |
| check-in of a 300-slot working copy with 20 changed elements | 500 ms | parse, compare with `base` by slot ID, one request with numbered writes |
| `changed-sources` over all citations | the search class | each message stores the hash of its text when written and each citation the hash of the cited text (`seen_hash`), so changed citations are found by comparison and only those are read |
| `--at` and the page diff | 300 ms, plus 100 ms per 1,000 changes | revisions current at a position, from the revision graph |
| a marker with `page=` during ingest | 1 ms | the page's heading map in memory |

Nothing is hashed on a read except the cited spans themselves, and no citation state is stored.

## Laws

Each law is a property with a test on the Morrow archive through the real daemon.

| Law | Property |
|---|---|
| page-holds-references | an investigation built from the template with a message slot, a kept result, a live query and a `::finding(page=…, section=Findings)` marker stores one page record, heading slots, slots and the finding, and no copy of the quoted message text |
| pages-optional | with no page in the archive every other behaviour is unchanged, the brief has no "Pages here" block, and the index has no table or index for pages alone |
| slot-edits | moving a slot writes one slot revision; removing the Evidence slot keeps the annotation it showed; editing a note through one page changes it on every page that shows it |
| section-tree | renaming a section writes one revision; moving a heading carries its children in order |
| section-on-demand | `::finding(page=X, section=Nope)` creates heading `Nope` at the end of X and places the finding under it; a second marker with `section=nope` reuses it |
| order-stable | a thousand random concurrent inserts from two writers, including into one gap, give a total order and rewrite no existing slot's position |
| contents-generated | `--view contents` lists every slot with its reference and one line, changes when a slot is added, ends with the position, and is never stored |
| page-as-of | `read --at` a position before the finding shows Findings empty; `--at TIME` is labelled approximate when records from two machines are involved |
| page-diff | `read --view changes --from POSITION` lists exactly the changes after that position |
| page-checkpoint | after `page:X save` gives `result:R`, reading `result:R` reproduces the page exactly after later edits to its slots, its notes and the archive, and nothing runs again |
| rename-keeps-references | renaming a page leaves every stored `page:` reference and slot resolving to it |
| rename-redirects | a page's earlier name resolves to it with `renamed to`; completion offers only the current name |
| anchor-survives-rename | `page:X#findings` written before the heading is renamed still opens that section |
| red-link | a slot showing `page:not-yet` is waiting, and attaches when that page is created, with no rewrite |
| discussion-not-placed | `note` on `page:X` without `--section` writes about the page and places nothing |
| subject-decision-box | the decision box shows exactly the current decisions about the subject; recording one does not answer the subject question |
| open-items-computed | answering a question on a page removes it from open items, and withdrawing the answer brings it back, with nothing stored about the page |
| circle-marker | a decision revised A → B → A shows `returned to the position of revision 1`; A → B → C shows nothing |
| staleness | a `current`-mode note written before a later finding was placed on its page shows `written before 1 finding`; revising the note clears it |
| proposal-tracked-change | an agent revision of an accepted paragraph renders as a tracked change over the accepted text; accepting it renders plain |
| reply-placement | a marker replying to a delivery sent with `--section Findings`, without `page=`, writes one slot under Findings; re-parsing writes none |
| alias-follows-successor | an alias whose session was resumed resolves to the resumed session, never across a fork; an earlier send keeps its recipient |
| watch-never-starts | a page watch firing while its member's session has ended produces a notice and no harness process |
| referenced-by-grain | `messages where referenced_by:page:X` returns the messages of a message-range slot and none of a session slot's messages; a page that nests itself terminates |
| same-on-every-surface | each page write through the CLI, MCP, markup and the web produces the same records apart from `via` |

The laws of page text (links from text, citations, embeds, inclusions, grounding) are in [page-text.md](page-text.md#laws); those of the working copy and the editor in [page-working-copy.md](page-working-copy.md#laws).

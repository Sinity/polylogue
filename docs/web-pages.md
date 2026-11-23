# Pages on the web

The workbench is where reading and citing happen together: a page is built while reading sessions, and edited in place as Markdown. The routes are `/pages` (the page list) and `/page/NAME` (one page). The model is [pages.md](pages.md), the text and its relations [page-text.md](page-text.md); this page gives their web form. The frame, keys and live behaviour shared by every view are [web.md](web.md).

Every action on a page is a line: the view posts it, shows the committed records and the undo line from the reply, and the footer shows the CLI line and the MCP call of what is on screen:

```text
polylogue page:flaky-ci read --view contents
polylogue_read {"arguments": {"selection": "page:flaky-ci", "steps": [{"verb": "read", "view": "contents"}]}}
```

## The page list

**`/pages`** is the listing of the unit `pages` (`polylogue pages`): title with `page:NAME`, template, subject, slots, members, open items, grounding counts (citations, sources changed, need citations, proposals), last change (`finding from codex:019d0f2a · 14:02`) and whether it is watched, grouped by name prefix and tag. Filters are conditions of the listing: `pages where shows:REF` finds the pages that place a row, `pages where linked_from:0` the pages nothing links to, `pages where "zone change"` the pages whose title or text holds the words ([pages.md](pages.md#finding-pages)). Shipped pages (`home`, `index`, `pinned`, `cost`) and templates (names under `template/`) are listed apart, each marked built-in until revised. **New page** asks for a title, a name and a template and writes the page record (`polylogue page:flaky-ci note 'Why is CI flaky on Fridays?' --as page --template investigation`), with **Parameters** for a parametric template (`--param repo=morrow`), **Subject** (`--about REF`), **Scope** (`--scope repo:morrow`) and **Brief** (`--brief full`).

## The page view

**`/page/NAME`** is the page's `read`; its query string carries the `read` step's argument names (`/page/flaky-ci?view=contents`, `/page/flaky-ci?at=48390`, `/page/flaky-ci?view=changes&from=seen`).

```
┌ Polylogue  Home  Running(2·1)  Search  Notes  Pages  Review  Saved  Cost  Status  [query…]   Collecting to page:flaky-ci › Evidence ✕ ┐
├─ contents ──────────┬─ investigation · page:flaky-ci · by operator · changed 2h ago ──────────────┬─ margin ─────────────────────┬─ inspector ───────┤
│ Goal             ·  │ Why is CI flaky on Fridays?                                                │                              │ Backlinks         │
│ Evidence         3  │ linear · board · table · contents · history      since seen ✕              │                              │  placed on 1 page │
│ Findings       ● 2  │ 14 citations · 1 source changed · 2 need citations · 6 proposals ▸         │                              │  linked from 3    │
│ Open questions   1  │ EVIDENCE                                                             + ⋮   │                              │ Decision          │
│ Decision            │ ┊ > Found a bug: "next friday" evaluated on a Friday returns the same day… │                              │  none yet         │
│                     │ ┊ — model · codex:…0005 · Oct 05 16:25 ✓                                   │                              │  record decision  │
│ Open items          │ ┊ ┌ live · tool_calls where command~"cargo test" outcome:error … · 3 ─┐    │                              │ Members           │
│  ? zone change      │ ┊ └──────────────────────────────────────────────────────────────────┘    │                              │  @reviewer ended  │
│                     │ FINDINGS                                                                   │                              │ History           │
│ pages/morrow        │ ╎ finding · codex:…0006 · proposed            accept  reject  edit         │ ¹ "uses >= where it should   │  r: 14 changes    │
│  architecture       │ ╎ weekday_offset() compared with >= where it should use >; strict mode¹    │   use >" — model, codex:…0005│  compare…         │
│  flaky-ci           │ ┃ The parser has no clock of its own; every relative date goes through     │   16:25 ✓                    │                   │
│                     │ ┃ weekday_offset().                       — from page:morrow/architecture  │ ◆ challenges · @parser:      │                   │
│                     │                                                                            │   strict mode breaks one test│                   │
├─────────────────────┴────────────────────────────────────────────────────────────────────────────┴──────────────────────────────┴───────────────────┤
│ polylogue page:flaky-ci read          polylogue_read {"arguments":{"selection":"page:flaky-ci","steps":[{"verb":"read"}]}}      position 48390     │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

- **Header:** template, `page:NAME`, created by and when, the subject above the sections (a question, a decision, a repository, a session or a file, linked); the counts line (`14 citations · 1 source changed · 2 need citations · 6 proposals ▸`, each a link to the rows it counts); the switcher **linear · board · table** with the **contents** view and the **history** panel beside it; **Checkpoint**, **Permanent link**, **Watch**, **Brief**, **Working copy**, **Collect here** and the copy menu. The page's own `layout` is the default; the switcher changes the presentation only, and **Make this the page's layout** writes it (`note TITLE --as page --layout board`).
- **Reading column**, 75 to 90 characters wide: headings in tree order, each with **+** to add to that section; an empty section renders as one line `empty — add`. Slots render from the archive at read time in their forms ([pages.md](pages.md#forms)), each with its mode (`current`, or a pin glyph and `fixed at r3`) and its state. Text slots read as prose with superscript citation markers; an excerpt is a quotation with its source line and check mark; embeds are framed (a card, a table with `live · N rows` or `stored rows, not run again` and **Run again**, **Keep**, **Compare with now**, a chart, a code view of file lines with **History** and **Current**); included text has a tinted margin bar and its origin on hover; a red link reads `page:not-yet · waiting`. `url:` targets show their written title and are never fetched; `sinex:<kind>/<id>` targets resolve through Sinex in primary mode and show `unavailable` otherwise.
- **Margin**, at 1,440 px and wider: citations as sidenotes beside their markers, each with its quote, source line and state; and discussion (annotations about a slot's note that are not placed on the page: challenges, qualifications, questions, replies) as sidenotes with author badges ([pages.md](pages.md#discussion)). Narrower, footnotes gather at each section's end and discussion opens from a count.
- **Left rail**: the contents with counts per section and the open items, then the pages sharing the name prefix.
- **Inspector**: **Backlinks** (what places, links, cites or includes this page, exact rows first, then rows inside, [page-text.md](page-text.md#backlinks)); the **decision box** (`no decision yet` with **Record decision**, which writes a `decision` about the subject placed in the Decision section: `polylogue annotation:01JA9X2B note 'strict weekday mode' --as decision --page flaky-ci --section Decision`); **Members**, each alias with its resolved session, live state, role and last activity on the page (`@reviewer → codex:019d0f2a · working · reviewer · last on this page 09:50`), with **Add member** ([pages.md](pages.md#members)); **Page details** (template, subject, scope, brief, parameters); **History**.
- **References** everywhere open nested hover popups from the shared reference reader, with an icon per kind of target; a hover never runs a model or changes anything. Clicking a citation opens the reader at the highlighted span.
- **Open items** each show "unresolved since" and link to their item ([pages.md](pages.md#open-items-and-the-decision-box)). **Staleness and circle markers** are dim marks in place (`written before 2 findings on this page`, `returned to the position of revision 1`).
- **Send composer:** a member or session, the text, and the section where the reply belongs (`polylogue @reviewer send 'fold the findings into the overview' --page flaky-ci --section Overview`); delivery is [session-control.md](session-control.md#send)'s, and the reply is placed in that section ([pages.md](pages.md#reply-placement)).
- **Live.** A page view streams its slots and shown annotations; live cells, figures and text update in place, and fixed slots never change. Agents' markers aimed at the page appear under their section as they are written, authored by the message's author. Updates never move the reader: a change above the viewport shows `N changes · show`. Unsent text in an editor survives live updates.

The **Backlinks** panel is also on the session reader and on document, message and note pages ([web-reader.md](web-reader.md)), as `links where target:REF` for the object shown.

## Citing while reading

A tab can have an **active page**: set by **Collect here** on a page or from the palette, shown in the header as `Collecting to page:flaky-ci › Evidence ✕`, and held as view state in the tab, which is all the browser holds ([web.md](web.md#what-the-browser-holds)). In every reader and listing, `+` (and **+ page** on hover) places the focused thing in the active page's current section:

| Focused | Placed |
|---|---|
| a message header | the message, as an excerpt |
| a text selection | a span excerpt; the selection toolbar has **+ page** beside highlight, note, claim and send |
| a tool call line | its card; with part of its output selected, a span of the output |
| a result row | the row; a row of a live query first keeps the rows, then places `result:<id>#<row>` |
| a session row | its card |
| a search hit | the span of its snippet |

`Shift-+` asks first: the page (recent first, and **New page from this** with a template), the section, the form, and **cite in a paragraph…**, which appends a footnote citing the focused thing to a chosen paragraph instead of placing a slot. Each act shows its line with undo: `polylogue span:message:codex:01a20000-0005:msg_01a20000a11#c=0-96 save page:flaky-ci --section Evidence`.

Several search hits marked with `+` and no active page offer **New page from these**: a template, a question, and the marked hits placed in its Evidence section.

## Copy and paste

Copying text in the reader also puts its span reference on the clipboard, as a workbench clipboard type beside the plain text. Pasting it into a page paragraph offers **cite** (the paste becomes a quotation in the sentence with a footnote citing the span) or **plain text**. Pasting a reference or a provider URL makes a link, or a card when it is pasted alone on a line.

## Editing

- **In place.** Each text slot edits in place (`i`, or a click): the paragraph's Markdown source, with a live preview beside it, saved as one revision with `Ctrl-Enter` (`polylogue annotation:01JE4K8P edit TEXT` under the hood, [annotations.md](annotations.md#edit)). The source is exactly the text people and agents edit in files ([page-working-copy.md](page-working-copy.md)).
- **A stale base** shows the current text beside yours with **apply on top**, which rebases your change onto the current revision for you to save.
- **New paragraphs.** `Enter` on an empty line after a slot starts a new text slot there; the composer at a section's end does the same, with a kind select (note, finding, question, decision, …): `polylogue page:flaky-ci note 'weekday arithmetic used >= instead of >' --as finding --section Findings`. **Add section** writes a heading (`note 'Risks' --as heading`).
- **Pickers.** `[[` opens the page and reference picker and inserts a link; `[^` opens the citation picker (the page's excerpts, what this tab collected, the current selection, or a search) and inserts the footnote with its quote; `/` inserts a heading, excerpt, query cell, embed or inclusion.
- **Included text** says `this text is also on page:morrow/architecture (2 pages)` before saving; a fixed inclusion offers **switch to current** or **edit the source** instead of an editor ([page-text.md](page-text.md#inclusions)).
- **Working copy.** **Working copy** offers `read --to flaky-ci.md` as a download, and **Import** takes the edited file back as `import`: the reply's refused, removed and ignored parts are listed with their reasons, and the rest commit ([page-working-copy.md](page-working-copy.md#check-in)).

## Moving and layouts

- **Linear:** drag handles on slots, or `m` then the arrow keys and `Enter`, write one slot move (`polylogue page:flaky-ci#7q2m save page:flaky-ci --section Findings --after page:flaky-ci#4hq1`). **Remove** withdraws the slot (`withdraw`), and what it showed stays.
- **Board:** the same slots as cards placed by their `x`, `y`, `w` and `h`, headings as frames; dragging a card, or moving it with the arrow keys, writes one revision of that slot. The board exports and imports JSON Canvas files.
- **Table:** one row per slot with kind, target, form, mode, target state, author, grounding and last change, for moves and withdrawals in bulk.
- **Contents:** exactly `read --view contents`, each line a link.

## Proposals

Proposed text ([pages.md](pages.md#authorship-and-proposals)) has a dashed rule, its author badge, and **accept**, **reject** and **edit**; a tracked change shows the proposed revision as a word diff over the accepted text with the same buttons. Rejected text folds to `rejected by operator · show`. The header's `6 proposals ▸` walks them in order with the judge keys of the shared key table, each key writing `judge accept`, `judge reject` or `judge defer` on the shown revision.

## History

The history panel lists the page's change positions (each position where a slot or a shown annotation changed, with author and time). The scrubber renders the page as it stood at any of them (`read --at POSITION`); an approximate time across machines is labelled approximate. Choosing two positions shows the page diff between them (`read --view changes --from POSITION --until POSITION`): slot moves, additions, withdrawals, word diffs and sources that changed. **Since I last looked** is `--from seen`. **Permanent link** writes a checkpoint (`polylogue page:flaky-ci save`), an explicit act and never a side effect of opening a URL, and copies the URL of its `result:<id>`. **Watch** writes `polylogue page:flaky-ci save --watch`. Reading a page here moves the reader's seen mark only when the web reports that the items were on screen.

## Keys

Keys follow the shared key table ([web.md](web.md#keys)). The page view adds:

| Key | Does |
|---|---|
| `+` | place the focused thing on the active page; `Shift-+` asks for page, section, form, or a paragraph to cite in |
| `i` | edit the focused text slot |
| `m` | move the focused slot (arrows, then `Enter`) |
| `[`, `]` | previous and next section |
| `v` then a letter | layout or view: `t` linear, `b` board, `a` table, `n` contents, `h` history, the letters of the shared table |
| `Ctrl-Enter` | save the open editor as one revision |

Lines: `polylogue pages`, `polylogue page:flaky-ci read --view changes --from seen`, `polylogue page:flaky-ci read --at 2026-11-10`, `polylogue 'messages where "flaky" repo:morrow since:7d' save page:flaky-ci --section Evidence --fixed`, `polylogue page:flaky-ci#7q2m withdraw`, `polylogue import flaky-ci.md`.

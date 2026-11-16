# Page text

The text of a page is Markdown with references. Inside a paragraph a sentence can do four things with an address: link to it, cite it as the exact evidence the sentence rests on, embed it to show it in place, or include its text as part of the page. This page owns that text grammar, the four relations, citations and their checks, embeds and values, inclusions, backlinks on every object, what each relation does when its source changes, and grounding. The page model is [pages.md](pages.md); the file form of a page is [page-working-copy.md](page-working-copy.md); reference forms are [references.md](references.md).

## Markdown with references

A note's text, and so a text slot's, is CommonMark with footnotes, in which four existing Markdown forms carry the meanings below, plus the product's markup prefix for kinds (`::decision: …`, [markup.md](markup.md)). The same text is accepted by `note TEXT`, `edit TEXT`, the MCP `note` and `edit` steps, marker bodies, the working copy and the web editor, and it is what `read --format markdown` prints. One grammar and one parser serve every writer and renderer.

- **References in text become link rows** in the same write: role `mentions` for a link, `cites` for a citation (with its quote and hash), `embeds` for an inline value or an inclusion ([annotations.md](annotations.md#links)). The text stays the source of truth and the rows are written with it, as markers in messages become annotations. A reference that does not resolve yet is a waiting link (for a page, a red link); an ambiguous one refuses that paragraph with the candidates.
- **Citations travel with their text.** A note's footnote definitions are part of its stored text, so the note carries its evidence wherever it is shown or included. Stored labels are local to the note (`[^1]`); a page rendering or working copy numbers them page-wide and maps them back on check-in.
- **Any reference form** can appear: a session, message, span, block, tool call, kept result or row, annotation, page or section, `file:`, `url:`, `sx:` ([references.md](references.md#forms)).

## The four relations

| | Link | Citation | Embed | Inclusion |
|---|---|---|---|---|
| **Means** | go there | this sentence rests on that evidence | show that thing here, rendered at read | include that text here, in step with its source |
| **Holds** | any reference | an exact place in evidence: a span, message, block, tool call, result row, file lines, an annotation revision, a document, `url:`, `sx:` | captured things, queries, kept results, figures, cards of anything | authored text: an annotation, a page section, a whole page |
| **Markdown** | `[text](REF)` or `<REF>` | `[^1]` in the sentence, `[^1]: "quote" <REF>` below it | `![caption](REF "form")`; a value in a sentence `![…](REF "value")`; a question as a ```` ```query ```` block | `![](REF "text")` on its own line, REF an annotation or `page:X[#anchor]` |
| **Stored** | link row `mentions` | link row `cites` with quote, context and hash | a slot (`about` or `query`, a form); inside text, link row `embeds` | a slot whose `about` is the annotation or page, form `text` |
| **Terminal** | the text underlined, its short reference dimmed after it, a terminal hyperlink to the workbench | `¹` in the text; the slot's footnotes under it: quote, source line, state mark | a boxed card, table or value with its reference line | the text flowing with a `┃` bar and a dimmed `— from page:morrow/architecture#parser` |
| **Web** | underlined, an icon per kind of target, a nested hover popup; a wanted page in red | superscript; a sidenote in the right margin, or at the section's end on narrow screens; quote and source on hover; a click opens the reader at the highlighted span | a framed card, table or chart, its footer with the reference, run and position | text in the column with a tinted margin bar; origin and "also on 2 pages" on hover; a pin glyph when fixed |
| **When the source changes** | the target's state only | re-anchored, or `source changed` with both texts | live: shows the new; kept: never changes | current: follows; fixed: `source revised` with a diff |

What separates them is what the reader is meant to do: follow a link, trust or check a citation, look at an embed, read an inclusion as part of the page. A link never draws content; a citation always shows the exact words it rests on; an embed is framed as an object; included text is not framed, because it reads as the page's own prose, and its origin is one hover away.

A working-copy paragraph using all four:

```markdown
The parser was reviewed in [the review session](codex:01a20000-0005); see <page:morrow/architecture#parser>.
weekday_offset() compared with >= instead of >.[^1] Errors fell to ![34 of 1,210](result:01JD7Q4M "value").

[^1]: "uses >= where it should use >" — model, codex:01a20000-0005, 5 Oct 2026 16:25 CEST <span:message:codex:01a20000-0005:msg_01a20000a11#c=88-121> b3:9c1e2f4a
```

## Links

`[text](REF)` or `<REF>` links to anything with a reference. A link to a page that does not exist is a red link: it is stored waiting and attaches when the page is created, and the shipped query `wanted` lists such names with how many link to each ([pages.md](pages.md#finding-pages)). A link to a section (`page:X#anchor`) is stored by the section's slot root, so renaming the heading breaks nothing. Pasting a bare reference or a provider URL into the web editor makes a link ([web-pages.md](web-pages.md#editing)).

## Citations

A citation says a sentence rests on exact evidence: `[^1]` in the sentence and a footnote definition below the paragraph giving the quoted words and the source. One link row is stored per cited place: role `cites`, the reference as written (normally a `span:`), the quoted text with up to 32 characters of context either side, the BLAKE3 hash of the cited text when written, and the reading version of the container at that time. The quote and context are what let a citation be found again after a re-parse moves offsets.

### Cite by quote

A footnote or an excerpt may give a container and a quotation instead of a span:

```markdown
[^2]: "returns the same day" <codex:01a20000-0005>
```

The write finds the quotation in the container's text and stores the exact span. Matching treats any run of whitespace as one space, straight and curly quotes as equal, and `…` (or `...`) as a gap whose pieces must appear in order; nothing else is folded. One match is pinned and the footnote is rewritten with the span reference, the source label and the hash. Several matches are refused with the first candidates as span references; none is refused with `this text is not in codex:01a20000-0005` and the closest passage. The container may be a session, message, message range, tool call, document or kept result row.

Citing by quote needs no ID typed beyond the container. On a terminal it is the way to make a precise citation: quote the words, or embed a whole message in the working copy and trim the quotation to what is needed ([page-working-copy.md](page-working-copy.md#check-in)).

### The quote check

A quotation inside prose ("…" of four words or more) followed directly by a citation marker must occur in the cited source. The reader marks it `✓`, or `≠ these words are not in the source`. It catches a sentence that misquotes what it cites, which no structural check otherwise would.

### How a citation shows

Numbering is page-wide; a source cited several times is one footnote marked `cited 3×`. The footnote shows the quote, the source line (author, origin, local time with zone, reference) and a state mark:

| Mark | `target_state` | Means |
|---|---|---|
| `✓` | `current` | the cited text is there as cited |
| `✓ re-anchored` | `current` | a re-parse moved the text and the quote and context were found again; the stored reference stays as written |
| `≠ source changed` | `changed` | the source's text differs from what was cited; both texts are shown |
| `? waiting` | `waiting` | the target is not stored yet; the citation attaches and gets its hash when it arrives |
| `✗ not found` | `missing` | the target never appeared, and its session is complete |
| `withdrawn` | `withdrawn` | the cited annotation was withdrawn; the quote stays |
| `○ outside, not checked` | `outside` | a `url:` or other source Polylogue does not hold |

States are computed when the page is read, from the text the page reads anyway to show the quote; nothing about them is stored. In the terminal the footnotes sit under their slot; on the web they are sidenotes ([web-pages.md](web-pages.md#the-page-view)).

### What a citation proves

A citation of captured text proves what it quoted three ways: the hash of the cited text in the link row; the stored input bytes the message was read from, which are never deleted (`polylogue span:message:codex:01a20000-0005:msg_01a20000a11#c=88-121 read --view trace` walks record → input record → stored bytes → reading version, [reading.md](reading.md#trace)); and the reading version recorded with the citation. When a later reading of the same bytes yields different text, the page shows both, the text as cited and the text now. The quote and hash preserve what was cited whether or not the earlier reading can be run again.

### Citation lines

`read --format citation` prints, for any selection, footnote definitions ready to paste: one line per item, and a query of messages one line per hit.

```text
[^1]: "next friday evaluated on a Friday returns the same day" — model, codex:01a20000-0005, 5 Oct 2026 16:25 CEST <span:message:codex:01a20000-0005:msg_01a20000a11#c=12-66> b3:4be1c07d
```

`polylogue @2-4 read --format citation --to clipboard` puts three on the clipboard; `polylogue 'messages where "zone" repo:morrow' read --format citation` gives one per hit, which is how an agent gathers what it will cite ([reading.md](reading.md#formats-and-destinations)).

## Embeds

An embed shows a thing in place, rendered from the archive at read.

- **A block embed is a slot.** A paragraph holding only `![caption](REF "form")` is a slot with `about` REF and that form; the caption is shown above it. A question written out in full is a query block, the page's live cell:

  ````markdown
  ```query table
  tool_calls where outcome:error command~"cargo test" since:7d | group by session | count
  ```
  ````

  A saved query with parameters embeds as `![](query:failing(repo=$repo) "table")`. Forms are [pages.md](pages.md#forms); cells are [pages.md](pages.md#queries-on-pages).
- **What is live and what is fixed follows the reference.** `query:` is run at read (live); `result:` is the stored rows (fixed); a captured thing renders from the archive and shows `source changed` when re-parsed; a session card shows the session's current state. A query slot can be kept (`save page:X --kept`, or **Keep** on the web) and keeps its question beside the result so it can run again.
- **Budgets.** Each embed renders within its own share of the page budget: a table shows its first page (20 sessions or 50 other rows) and `N more · open as search` with the line; a transcript range pages like any reading. Nothing is cut silently.

### Values in sentences

`![](query:errors-30d "value")` inside a sentence is kept when the sentence is written: the write keeps a result, stores `result:<id>` in the text, and the reply says `fixed 1 value as result:01JD7Q4M`. A sentence states a fact at a time; a number that moves under it would make the sentence false without anyone editing it. `"live value"` keeps it live, and the reader tags it `live`. In the working copy the alt text of a value is the value Polylogue rendered (`![34 of 1,210 (2.8%)](result:01JD7Q4M "value")`), readable in any Markdown viewer and ignored on check-in; for every other form the alt text is the caption.

A report states its numbers this way: `First builds fell from ![41 min](result:01JD7Q4M "value") to ![17 min](result:01JE2B9C "value").`, with the live version of the same query in its own slot below, labelled live.

## Inclusions

An inclusion shows another annotation's or page section's text as part of this page, in step with its source.

- **What can be included:** one annotation (`![](annotation:01JE4K8P "text")`), a page section (`![](page:morrow/architecture#parser "text")`, the heading and everything under it), a whole page (`![](page:morrow/glossary "text")`), and a span of an annotation revision (`span:annotation:<root>@<rev>#c=…`, shown as an excerpt, since a span is fixed).
- **Placing a note that lives elsewhere is an inclusion.** The mechanism is the slot; the form `text` and a home other than this page make it one. Its home is shown: `from page:morrow/architecture`, or `note about codex:01a20000-0005` for a note written about a session.
- **Current or fixed.** `current` follows the source's current revision and marks `revised since you looked` from the reader's seen mark. `fixed` shows the named revision and, once the source moves on, `source revised (r3 → r5) · show diff · take r5`.
- **Editing included text edits its source.** The editor says `this text is also on page:morrow/architecture (2 pages)` before saving; the revision changes it everywhere it is included. A fixed inclusion is read-only: editing one is refused with the two choices, switch it to current or edit the source.
- **Citations come along**, because they are in the included text. Included headings are shifted below the including heading's level. Alternates apply to included text like any other.
- **Cycles.** A section that would include itself, directly or through other pages, renders `(included above: page:morrow/architecture#parser)` at the first repeat. Nesting is otherwise unlimited and pages within the budget.
- **Withdrawn sources never vanish.** An included annotation that is withdrawn shows its last text dimmed with `withdrawn by codex:01a20000-0006 on 12 Nov: "superseded"` until someone withdraws the slot.

## Backlinks

Every object's reading ends with a **Cited and linked** block, and the workbench has a **Backlinks** panel ([web-pages.md](web-pages.md#the-page-view)): the pages that place it, the notes and pages that link to it, the citations of it or inside it, and its inclusions. Exact rows come first, then rows inside it: a citation of a span in a message is listed on the message and, rolled up, on its session as `2 citations inside this session`. A session's overview lists `cited on page:flaky-ci (2)`.

The query form is the unit `links` (record, role, target, target_kind, target_state, page), and every unit has `cited_by` beside `pages` and `referenced_by:` ([query.md](query.md#units-and-fields)):

- `polylogue 'links where target:codex:019d0f2a role:cites'`
- `polylogue 'links where role:cites target_state:changed'`
- `polylogue 'links where page:flaky-ci target_state:changed'` is where fixing a page's citations starts.

## When a source changes

Polylogue deletes nothing, so a source never disappears: it changes, is revised or withdrawn, or turns out never to exist.

| Event | Link | Citation | Live embed | Kept embed | Current inclusion | Fixed inclusion |
|---|---|---|---|---|---|---|
| a re-parse moves the text's offsets | — | the quote and context are found again: `✓ re-anchored` | shows the new reading | rows unchanged; `computed under reading X, now Y` | — | — |
| a re-parse changes the text | — | `≠ source changed`: the text as cited and the text now | the new text | unchanged | — | — |
| the annotation is revised | — | a citation names an exact revision: `✓`, plus `newer revision exists` | — | — | follows; `revised since you looked` | `source revised (r3 → r5)` with a diff and **take r5** |
| the annotation is withdrawn | marked `withdrawn` | `cited text withdrawn by …`; the quote stays | — | — | the last text dimmed with the withdrawal line | the same |
| a target never appears (its session complete) | `✗ not found` | `✗ not found`; the quote stays | `not found` | — | `not found` | — |
| a target is not stored yet | waiting (for a page, a red link) | `? waiting`; attaches and gets its hash on arrival | `waiting` | — | `waiting` | — |
| a query's rows change | — | — | shows the new rows | unchanged; **compare with now** lists rows added and gone | — | — |
| a page is renamed or withdrawn | still resolves; the old name redirects; withdrawn is marked | — | — | — | follows; withdrawn dimmed | — |

A citation of a target that is not stored yet gets its hash when the target arrives. Changes reach people through what already exists: the page header's counts, `read --view changes` ([pages.md](pages.md#time-and-checkpoints)), a page watch (a citation whose source changed is one of its events), the shipped query `changed-sources`, and the brief of any session in the page's scope (`page:morrow/architecture: 1 source changed since cited`).

## Grounding

A page cannot judge whether text is true; it shows which text is grounded.

- **`grounding`** is a field on text slots and annotations: `cited` (has citations, all current), `changed` (a cited source changed, went missing or was withdrawn), `uncited` (no citation and no `evidence` link, written by anyone but the operator, of a kind that states something: note, claim, finding, decision, lesson, correction, handoff), or `none` (headings, questions, goals and the operator's own text).
- **Shown**: an `uncited` paragraph ends with a small `citation needed`; the page header counts `14 citations · 1 source changed · 2 need citations`; anyone may mark a particular paragraph, the operator's included, with `tag citation-needed`.
- **Queries**: `polylogue uncited` (agent text on pages with no citation) and `polylogue changed-sources` (citations whose source changed, went missing or was withdrawn, grouped by page) are shipped ([saved-queries.md](saved-queries.md#shipped-saved-queries)).
- **Never a gate.** Uncited text is written, shown and briefed, marked. Removing a citation is allowed and the reply reports it ([page-working-copy.md](page-working-copy.md#check-in)).

An agent drafting a page works so that its grounding is visible: `brief` of the topic, read the hits, `read --format citation` on what it relies on, write prose whose every claim carries a footnote, check in. The reply lists the citations pinned, the quotations refused (`not in the source`) and the paragraphs left uncited ([agents.md](agents.md)).

## Laws

Each law is a property with a test on the Morrow archive through the real daemon.

| Law | Property |
|---|---|
| text-is-source-of-links | the link rows of an annotation equal the references its text parses to, after every write and import |
| cite-by-quote | a footnote giving a session and a quotation stores the exact span; an ambiguous quotation is refused with candidates; a quotation not in the source is refused |
| quote-check | a quoted phrase before a citation marker is marked `✓` when it occurs in the cited span and `≠` when it does not |
| citation-reanchors | a re-parse that shifts offsets leaves the citation `✓ re-anchored`; one that changes the text shows `≠` with both texts |
| hash-on-arrival | a citation of a not-yet-stored message is `waiting`, and gets its hash and state when the message arrives |
| value-fixed-in-prose | a `value` embed of a query inside a paragraph is stored as a kept result; `live value` stays live |
| citations-travel | a note with footnotes included on a second page shows the same footnotes there |
| inclusion-edits-source | editing included text revises its source, which changes on every page including it; a fixed inclusion refuses the edit |
| withdrawn-inclusion-visible | withdrawing an included note shows its last text dimmed with the withdrawal on every page including it |
| inclusion-cycle-cut | a page including itself through another page renders `included above` at the first repeat and terminates |
| backlinks-everywhere | a citation of a span in a message is listed in that message's and, rolled up, its session's backlinks |
| grounding-computed | agent prose with no citation is `uncited`; adding a citation makes it `cited`; a changed source makes it `changed`; operator text is `none` |

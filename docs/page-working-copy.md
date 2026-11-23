# The page working copy

A page's working copy is the page as one Markdown file: what `read --format markdown` prints for a page, what `read --to FILE.md` writes, what `edit` opens in your editor, and what `import FILE.md` checks in. It reads as plain Markdown (GitHub, Obsidian and static site tools show it sensibly), diffs line by line, survives any editor and any agent's file tools, and round-trips exactly. The file is a working copy, never a second store. This page owns editing in your editor, the format and its grammar, check-in and its conflicts, importing other Markdown, and writing pages out as files. The page model is [pages.md](pages.md); the text grammar inside the file is [page-text.md](page-text.md).

## Editing in your editor

`edit` takes its text or nothing ([interfaces.md](interfaces.md#verb-words)). With no text, on a terminal, it writes the selection to a private temporary file, opens `$VISUAL` or `$EDITOR`, and checks the file in when the editor exits:

| Selection | The file |
|---|---|
| a page | its working copy |
| one note | its text, with one slot comment |
| several notes | one file with a slot comment before each note's text |
| a page name that does not exist | a new working copy whose header lists the templates; saving creates the page |

- `polylogue page:flaky-ci edit` opens the page; `polylogue annotation:01JE4K8P edit` one note; `polylogue 'annotations where kind:lesson repo:here' edit` several.
- **An unchanged file writes nothing.** A changed file is checked in as one request, exactly as `import` does, and the reply is printed after the editor closes.
- **Refused parts reopen.** When any part is refused, the editor opens the file again with the refusal beside each refused element and your text kept below it:

  ```markdown
  <!-- refused: conflict, the current revision is r4 by codex:01a20000-0006; your text is kept below -->
  ```

  The parts that were not refused are already committed; closing the file unchanged leaves the refused ones unwritten.
- **Off a terminal, and over MCP,** `edit` without text is refused with the two lines that do the same with a file: `polylogue page:flaky-ci read --to .page/` and `polylogue import .page/flaky-ci.md`. Agents and scripts use those.
- A note whose whole new text is one verb word is written `edit 'read.'` or `edit -`; `--explain` shows which reading a line got ([terminal.md](terminal.md#explain)).

## The format

````markdown
---
page: flaky-ci
title: Why is CI flaky on Fridays?
template: investigation
about: annotation:01JA7Q2MXK4V9R3T8N6P0WZ5C1
layout: linear
brief: contents
scope: [repo:morrow]
base: page 01JA9XK2MV revision 4 position 48390
---

<!-- slot 1ab2 heading -->
## Evidence

<!-- slot 9kd2 excerpt fixed -->
> Found a bug: "next friday" evaluated on a Friday returns the same day instead of a week later.
> — model, codex:01a20000-0005, 5 Oct 2026 16:25 CEST <span:message:codex:01a20000-0005:msg_01a20000a11#c=0-96> b3:4be1c07d

<!-- slot 3fq7 query -->
```query table
tool_calls where command~"cargo test" outcome:error repo:morrow since:7d
```
<!-- generated: run 01JE5M2Q at position 48390, 3 rows; not checked in -->
| time | session | command | outcome |
|---|---|---|---|
| Oct 05 16:14 | codex:01a20000-0005 | cargo test | error · exit 101 |
<!-- /generated -->

<!-- slot 1ab3 heading -->
## Findings

<!-- slot 7q2m r3 · finding · by codex:01a20000-0006 · proposed -->
::finding: weekday_offset() compared with `>=` where it should use `>`; strict mode fixes it.[^1]

[^1]: "uses >= where it should use >" — model, codex:01a20000-0005, 5 Oct 2026 16:25 CEST <span:message:codex:01a20000-0005:msg_01a20000a11#c=88-121> b3:9c1e2f4a

<!-- slot 4hq1 r2 · note · by operator · also on page:morrow/architecture -->
The parser has no clock of its own; every relative date goes through [`weekday_offset()`](file:src/parse.rs@4e1d2c9#L40-L58).

<!-- slot 2mn8 text -->
![](page:morrow/architecture#parser "text")
<!-- generated: page:morrow/architecture#parser at position 48390; edit it there -->
### Parser
The parser turns phrases into dates…
<!-- /generated -->

<!-- slot 1ab4 heading -->
## Open questions

::question: Does the parser cache survive a zone change?
````

The last paragraph has no slot comment: someone just wrote it, and check-in makes it a question and its slot.

## Grammar

- **Header.** YAML front matter with `page` (the name; changing it renames the page), `title`, `template` (honoured only when creating), `about`, `layout`, `brief`, `scope`, `params` and `tags`; `base` is written by Polylogue and names the page root, its revision and the position the file was read at. A file without `base` creates a page, named by `page`, else by the file name; naming a page that exists is refused with its `edit` line. Unknown keys are refused, listing the known ones. The title lives in the header, so the body's top-level sections are `##`; a `#` line is refused with that reason.
- **Slot comments.** An HTML comment on its own line directly before an element: `<!-- slot ID [rN] [WORD]… -->`. Check-in reads the slot ID (a short prefix unique within the page), the base revision `rN`, `fixed` or `current`, and `N parts` (the element count of a note with several elements); every other word (kind, author, `proposed`, `also on …`) is written for the reader and ignored. Comments are invisible in rendered Markdown.
- **Text.** A paragraph, list, quotation, table or fenced block is one element, and one element is one slot unless its slot comment says `N parts`. A paragraph may open with the markup prefix `::kind(args): ` for kinds other than `note`; the arguments are the markup's (`reverse_if=`, `replies_to=`, `evidence=` …, [markup.md](markup.md)).
- **Links, citations, embeds and inclusions** are the Markdown forms of [page-text.md](page-text.md#the-four-relations). A slot's footnote definitions follow its element, before the next slot comment; labels are unique within the file and numbered page-wide.
- **Excerpt.** A quotation whose last line is a source line, `— LABEL <REF> [b3:HASH]`. The label and hash are written by Polylogue and ignored on check-in.
- **Embed or inclusion.** A paragraph holding only `![caption](REF "form")`.
- **Query.** A fenced block whose info string is `query [FORM] [kept result:ID]`; its content is the question.
- **Generated regions.** `<!-- generated: … -->` … `<!-- /generated -->` after an embed, query or section inclusion: rows, cards and included sections, written for reading and never checked in.
- **Conflicts.** [Conflicts](#conflicts) have their own region form.

## Check-in

`polylogue import .page/flaky-ci.md` recognises the header and checks the file in as one request, committed or refused element by element, with one undo line for everything it wrote. `edit` does the same when the editor exits.

| In the file | Check-in |
|---|---|
| text under a slot comment changed | a revision on `rN`; if `rN` is no longer current, that element alone is refused `conflict`, naming the current revision |
| a `::kind` prefix changed | a revision with the new kind |
| an element with no slot comment | a new annotation (kind from its prefix) and a slot, under the heading above, between its neighbours |
| a heading line added, renamed or re-levelled | a heading slot written or revised |
| a slot comment in a new place | a slot move: new parent and position from its neighbours |
| a slot comment gone | nothing; the reply lists `2 slots missing, not withdrawn`; `--withdraw-missing` withdraws those slots, and what they showed survives |
| `fixed` ↔ `current` changed | a slot revision of its mode |
| an excerpt's quote trimmed | the excerpt narrows to the trimmed text's span (contiguous, or with `…` gaps in order); text that is not in the source is refused: `this text is not in codex:01a20000-0005` |
| a new quotation with a source line | cite by quote ([page-text.md](page-text.md#cite-by-quote)): a new excerpt slot |
| a footnote's quote edited, or a footnote giving a container and a quote | resolved by quote within that container |
| a footnote reference removed from text | committed; the reply lists `slot 7q2m: removed citation [^1] (codex:01a20000-0005)`, and the paragraph's grounding is computed again |
| an embed's form changed | a slot revision |
| a query's text changed; `kept result:ID` removed; `kept` added | a new question; back to live; a new kept result now |
| a value's alt text | ignored: it is the rendered value |
| a generated region edited | ignored; the reply says so and names where to edit (the source page) |
| a conflict region left with one text | one revision revising both heads |
| the header edited | a revision of the page record (a changed `page` renames it) |

The reply lists what was written and what was not:

```text
page:flaky-ci · 4 revisions, 1 move, 1 new question · 3 citations pinned
refused 1: slot 4hq1 conflict, the current revision is r3 by codex:01a20000-0006
removed citation: slot 7q2m [^1] (codex:01a20000-0005)
1 paragraph has no citation (shown as citation needed)
ignored: edits in 1 generated region; edit page:morrow/architecture#parser there
```

The reply also prints an undo command carrying the exact written references on standard input to `polylogue - withdraw --all`.

Reading a page out and checking it in unchanged writes nothing. Per-element refusal is the same per-line refusal annotation lines get ([annotations.md](annotations.md#annotation-lines)).

## Conflicts

Two current revisions of one annotation (written on two machines from one base, [annotations.md](annotations.md#current-revisions)) are written into the working copy as a conflict region:

```markdown
<!-- slot 7q2m conflict: keep one text -->
<!-- r3 by operator -->
weekday_offset() compares with `>=`; strict mode fixes it.
<!-- r3 by codex:01a20000-0006 -->
weekday_offset() compared with `>=` where it should use `>`.
<!-- /conflict -->
```

Leaving one text (edited or not) and checking in writes one revision that revises both heads. A conflict region left with both texts is refused for that slot.

## Importing Markdown

`import FILE.md` of a file without Polylogue's header makes a page from it, named by the file:

- headings become sections and paragraphs notes, one slot each;
- Markdown links to other files imported in the same request become page links; `[[Name]]` and `[[Name|text]]` become page links, red until those pages exist;
- URLs become `url:` links;
- footnotes become citations of outside sources, shown `○ outside, not checked`, unless they name Polylogue references, which resolve and are checked like any citation;
- front-matter `title` and `tags` become the title and tags; front-matter `aliases` are ignored with a note, since a page has one name;
- local images are not copied; their links stay as text.

`import DIR` imports every `.md` file in the directory as a page and resolves links among them. This is how an existing Markdown wiki or notes folder comes in: `polylogue import ~/notes/parser.md`, `polylogue import ~/vault/`. A wiki whose articles cite its own source pages by URL keeps those citations as outside `url:` citations, marked not checked. Nothing is imported unless someone runs `import`.

## Writing pages out

- **A static site.** `read --to DIR --format html` of one or more pages writes each page as a self-contained HTML file (inline CSS, no scripts): links between written pages are relative; citations are footnotes carrying the quote, source label, reference and hash; included sections are inlined; popups for written targets are CSS-only disclosures. Links to anything not written stay as their references in text, and cited sessions are not written unless they are selected. `polylogue 'pages where name:morrow/*' read --to site/ --format html`.
- **A Markdown folder.** `read --to DIR --format markdown` writes each page's working copy with relative links between written pages. The front matter (`title`, `tags`) is what Quartz, Obsidian and other static site tools read, so one Markdown form serves editing and publishing, and pages can be written into an existing Markdown site as a folder of it: `polylogue 'pages where name:morrow/*' read --to site/polylogue/ --format markdown`.
- **Local files only.** Writing pages out writes files in the named directory and nowhere else; nothing is uploaded or sent. Each file is named after its page (`morrow/architecture` becomes `morrow/architecture.html`) and carries no session ID, user name or machine name in its name. Labels follow `--neutral-names` ([reading.md](reading.md#neutral-names)): the human is shown as `you`, or as `user` with the flag. The text is quoted as stored, with nothing masked, as in every reading; publishing the files anywhere is a separate act of whoever holds them.
- Agents and scripts check a page out with `read --to .page/` and back in with `import .page/flaky-ci.md`; a page exported as Markdown can be checked in the same way while its header carries `base`.

## Laws

Each law is a property with a test on the Morrow archive through the real daemon.

| Law | Property |
|---|---|
| checkout-checkin | `read --to`, an edit of one element and `import` write one revision; with that element's base moved it alone is refused `conflict`; a moved slot comment becomes a move, a new paragraph a note slot |
| working-copy-roundtrip | reading a page to Markdown and checking it in unchanged writes nothing; every form in [The format](#the-format) survives the round trip |
| excerpt-trim-narrows | trimming an excerpt's quote in the working copy narrows the span; adding a word not in the source is refused |
| edit-opens-editor | `edit` with no text on a terminal checks in what the editor saved and reopens refused elements with their reasons; off a terminal it is refused with the `read --to` and `import` lines |
| markdown-import | a plain Markdown file with headings, a wikilink and a footnote becomes a page with sections, a red link and an outside citation |

# Transcripts

How a reading is composed and rendered: the layout every format shares, composed transcripts, branches, compaction and other in-place events, the context and tree views, message headers, tool calls, thinking, attachments, value states, text as stored, rendering for models, what the web reader embeds, and the transcript lines that pages come from. What `read` takes (views, show and hide, ranges, budgets, formats, destinations) is [reading.md](reading.md); the parsed model is [sessions.md](sessions.md).

## Layout

```
selection ─► rows in order ─► view and show/hide ─► layout (reading document) ─► format ─► destination
                └── the page cursor binds selection, view and content
```

| Phase | Input → output | Where |
|---|---|---|
| Resolve | the selection → a read target | the one reference parser ([references.md](references.md)) and the query compiler |
| Select | target, range and parent options → ordered message keys with their source session, plus completeness issues | `store` (metadata only; bodies are read when laid out) |
| Filter | selection, view and show/hide → visible items | one definition of each view's rules, used by the store's kind filter and the renderer |
| Lay out | visible items → the reading document | `render` |
| Encode | the reading document → bytes in one format | `render`, one encoder per format |
| Page | the reading document stream and a budget → a page with `next` and `previous` | `reading`, using `render`'s pager |
| Deliver | bytes → standard output, pager, file, directory, clipboard, browser | the CLI only |

The `reading` crate composes sessions, prepares layout facts and walks records in either direction. It returns typed pages and streams over the store's reader. The daemon adapts requests and supplies ranking, freshness, the writer and ready follow-up calls through `ReadServices`; process hosting stays in the daemon.

A reference keeps its whole ID, whitespace, pipes and quotes included. To run query transforms on a referenced row, write a unit query, for example `messages where message:"claude-code:opaque | id" | sort by id`.

**The reading document** is the one typed structure every encoder, the pager and the web fragment consume, so pairing, indicators and value states are written once:

```text
ReadDocument = Header(SessionHeader | LogicalHeader | DocumentHeader | EventHeader | RecordHeader | PageHeader), Item*, Footer
Item = MessageHeader{time, author, model?, annotations[], alternate?, id, inherited?, branch?}
     | Text{source text}
     | Thinking{words, encrypted | redacted | text}
     | Call{tool, target, outcome, exit?, duration?, error_line?, input?, output?, child?: DelegationCard, edit?: EditView}
     | Attachment{name, media_type, size, supplied_by, availability, blob?}
     | Other{type, data?}
     | Indicator{kind, text, expand: ReadyCall}
     | Omission{count, unit, why, more: ReadyCall}
     | Card(link node, changes row, live process, physical session, …)
     | Slot{ref, form, mode, author, grounding, proposed?, item: Item}
     | Excerpt{quote, source, state}
     | Inclusion{origin, mode, Item*}
     | Footnote{label, quote, source, state, count}
     | Cell{query, form, run?, rows | pending}
Footer = CitedAndLinked[], More(ReadyCall)?
```

- A `Call` holds its result. Pairing happens once, in layout, by the stored call and result relation, never by serialization order; a result whose call is on another page is paired by reading the call's row by key. If a transcript range omits the call, the pair appears at the selected result, using the recorded call header. The input stays with the call's position.
- Copied messages are resolved within the selected message's physical session before the prefix relation is expanded; unrelated session edges add no per-message graph walk. Copied members keep their original source coordinates, and prefixes fixed at a branch point keep its bounds.
- Every item carries its source reference (message and block), so the pager can cut between items and start again there. Pages carry no other state. A message header's reference is its `message:` reference; the containing-message and block fields are set only on block items.
- A `ReadyCall` stores one complete typed request: operation, arguments, request ID, deadline, token budget, cursor, freshness and format. Each `next`, `previous` or other action carries it once. Clients send it directly; the operation table derives CLI lines, MCP calls and HTTP links only where those forms are shown. Every reading ends with ready calls.
- **One line per indicator**, in every format: a mark, then the text: `↳` continues, `⑂` branches, `⟲` compacted and microcompacted, `⏸` rate limited, `↺` away summary, `·` the rest. Token counts everywhere use one rule: below 1,000 the number, then `k`, `M` or `B` with one decimal (two for `B`): `182.0k`, `14.5k`, `1.5M`.
- Indicator kinds are a closed set: `continues`, `branches`, `compacted`, `microcompacted`, `rate_limited`, `model_switch`, `away_summary`, `harness`, `turn`, `link_issue`, `stale`, `alternate`.
- **Pages** lay out as a `PageHeader` (title, template, counts of citations, changed sources, uncited paragraphs and proposals) followed by one `Slot` per placed item in section order, each wrapping the item its [form](pages.md#forms) shows. A text slot's footnotes follow it, numbered page-wide, a source cited several times being one footnote with its count ([page-text.md](page-text.md#how-a-citation-shows)); a live query cell is `pending` until it answers and never holds the page back. The `markdown` encoder writes a page as its [working copy](page-working-copy.md#the-format); the terminal and HTML encoders draw each relation as [page-text.md](page-text.md#the-four-relations) describes.
- **The footer** of every reading is the **Cited and linked** block, headed `Cited and linked` in the text, Markdown and HTML formats: the pages that place the object, the notes and pages that link to it, citations of it or inside it, and inclusions of it, exact rows first, then rows inside it rolled up ([page-text.md](page-text.md#backlinks)).

## Composed transcripts

[Session links](sessions.md#session-links) and their `copies_prefix` flag are part of the parsed model. A child whose link copies a prefix (a fork, a resume, a copied spawn) is stored thin, holding only its own messages.

- **Default:** one indicator stands for the inherited messages: `↳ continues claude-code:7f3a2c10-0002 (18 earlier messages; --show inherited to show)`. The same default holds on the CLI, MCP and the web.
- **`--show inherited`** shows the parent's composed prefix up to the branch point first, each message marked inherited (a dim `↳` gutter in the terminal, a collapsed section in HTML).
- **Composition rules:**
  - `fork`, `resume` and copied `spawn`: the parent's composed prefix through the branch point;
  - `continuation`: the parent's messages are history before the context boundary, not the model's context (the [context view](#the-context-view) shows what the model had);
  - `spawn` without a copied prefix adds no parent messages;
  - `reference` links never compose;
  - shared ancestors are counted once; the walk is iterative with no depth limit;
  - cycles, missing parents, invalid parent IDs, missing copied messages, missing branch points, conflicting parents, and copied messages with no recorded or with conflicting relative order become `link_issue` indicators and `parents.issues[]` in JSON, with `kind` `cycle`, `missing_parent`, `invalid_parent`, `missing_copied_message`, `missing_branch_point`, `conflicting_parents`, `unknown_order` and `conflicting_order`. They are never resolved by choosing one link. Only the two order kinds can fail a reading: a read that needs the undetermined order of copied messages fails with `incomplete_parents`, naming the issue.
- Every message row carries `source_session` and `inherited`. The header shows `10 own + 18 inherited`.
- A copy whose source session is not stored shows as the child's own message, marked by one line before the first message, `(repeated from a session not in the archive, counted as own)`. It is no link issue; a parent that is missing from a link that copies nothing is.

## Branches

- Messages off the active path (edits, regenerations) are hidden in `transcript` and `dialogue`. Where a branch leaves the path one indicator appears: `⑂ 2 other versions of this reply (--show branches to show)`.
- `--show branches` shows abandoned versions indented under their split point, each labelled `branch N of M`.
- When the active path is unknown, every version is shown in source order with `⑂ active version unknown: 3 versions`. Unknown never collapses to "off the path".
- A branch whose parent message is missing shows standalone with `parent not stored`.
- Header counts cover every branch: `18 messages (3 on other branches)`.

## Compaction and other in-place events

The compaction summary is a message (`written_by: harness`, `prompt_kind: compaction_summary`) and the boundary is an event linked to it ([sessions.md](sessions.md#compaction-boundaries)).

- At the boundary the transcript shows one `compacted` indicator from the boundary event (`⟲ compacted (182.0k → 14.5k tokens) · auto`), then the summary message headed `[harness] compaction summary`, collapsed to its first line and shown whole with `--show harness:full`. Unknown token counts show `⟲ compacted (tokens not recorded)`; a boundary whose summary is not stored shows `summary not recorded`.
- Other events are one dim line at their position: `⟲ 3 tool results removed from context (−40.0k tokens)` (microcompaction), `⏸ rate limited until 14:10`, `↺ away summary`, `model → <name>`.
- `injected_context` messages are one dim line of their first 80 characters, whole with `--show harness:full`; a matched delivery shows its reference.
- An event without a message position goes at its time position when known, otherwise under "events with unknown position" at the end.

## The context view

The context view reconstructs what the model had at the selected message from the captured harness records, in their recorded order:

1. the composed parent prefix for `fork`, `resume` and copied `spawn` children ([Composed transcripts](#composed-transcripts));
2. the newest compaction boundary before the message: its summary message, then what the boundary records as kept (Codex `replacement_history`), then the messages after it;
3. `injected_context` messages at their positions;
4. microcompactions, which remove the tool results they name (Claude Code's `microcompactMetadata.compactedToolIds`, stored in the microcompaction's index entry) when the results came before them.

- At each boundary the view lists the references it evicted.
- Positions are in the composed transcript. A child with no compaction of its own sees its parent's composed prefix unchanged; a parent's compaction reaches a child only when the child's source says so.
- The newest boundary that fits is chosen before it is checked. An incomplete newest boundary gives the available prefix with `context.complete: false` and the reason, and never falls back to an older summary.
- System prompts and tool definitions are not recorded; the view begins `not recorded: system prompt, tool definitions`.
- `brief` composes context with this same function.
- The selection is the message (`polylogue message:codex:019d0f2a:m41 read --view context`); its context includes the earlier prefix, up to that message, with the same recorded parent branch cut. A session selection means its latest message. Paged `--at TIME|POSITION` selects the annotations separately ([reading.md](reading.md#ranges)); continuations bind the accepted message, so later appends cannot move that anchor. `--diff MESSAGE` or `--diff compaction[:N]` (just before the last or the Nth-last boundary) shows what left the context between the two places, as omission lines, and what entered, as the reading.

## The tree view

- The view starts at the known root of the shared lineage. When ancestry is unresolved it shows every known parent and cycle entry point as a starting position, without naming a root. Every node carries its root, depth and any pending, conflict or cycle issue, with all its recorded structural parents. The selected session is marked `▶`; a session absent from the archive shows unavailable values. Own and inherited counts and costs stay distinct.
- `reference` links are cross-links and never join structural ancestry. Every recorded alternative keeps its kind, reason, branch point, spawning call and copied-prefix observation, including edges to a node already shown. Copying content creates no provider relationship. A cycle shows a cross-link and is never expanded twice.
- Only link rows are read, never bodies. Cycle detection is iterative; a diamond is not a cycle; several roots stay; conflicting parents are reported; a known own count survives an unknown composed count.
- Pages walk the tree depth first, with the selected session on every page. A page's cursor holds the walk's place: the nodes being walked with how many of each one's links were taken, and the earlier nodes that more than one link reaches, each with the link that showed it. A page, forward or back, reads the links of its own nodes and of the nodes above them, never those before it; a later link to a node already shown is a cross-link where the walk reaches it, and the total is known once a page reaches the end of the walk. A tree cursor is bound to one archive-wide revision of the session links: a visible relationship or its stored order changing anywhere makes it stale, while message appends and unchanged re-parses do not. Cross-link steps obey the page budget too.
- JSON carries the same node and edge cards as the text and HTML encoders; an unavailable node has no invented lineage value. The web lineage panel is this view's HTML fragment.

## Message headers

- Each message starts with one header line: local time (`HH:MM`; `--:--` when unknown, never invented), then the author, labelled from [`written_by`](sessions.md#message-authorship), never from the provider's role. Shown times carry their recorded quality and precision (`09:12 (captured, minute)`); a bounded time shows its recorded upper bound as a clock time in the same zone (`09:12 (captured, bounded until 09:13)`). The date is its own line before the first message and before each message that starts a later day. A reading for a model (see below) writes every time with its date and offset instead.

  | `written_by` | Label to a person | Label to a model |
  |---|---|---|
  | `human` | `you` (`user` with `--neutral-names`) | `operator` |
  | `model` | `model`, or the model name when the session used several | the same |
  | `harness` | `[<harness>]`, the harness's name (`[claude-code]`) | the same |
  | `tool` | shown in its call line | the same |
  | `unknown` | `unknown author` | the same |

  A delegated task prompt (`prompt_kind: model`) is therefore labelled as model-written.
- **Annotations on a message** include current, unexpired annotations about the message or a span of its original message/block text. An annotation linked both ways appears once; private reading marks and internal saved-query, signal, setting and withdrawal records are omitted. These annotations (notes, tags, highlights, judgments; [annotations.md](annotations.md)) appear right-aligned as `◆ kind: text` (`+N` for more), each with its author, and are listed under the message with their judgments while `annotations` is shown. Unjudged agent annotations are shown with their author; what a brief carries is [agent-context.md](agent-context.md#the-brief-for-a-directory)'s rule.
- **Human text authorship:** typed text has a plain left rule; a paste span is shaded with one line saying where it came from (`pasted · <source of the span>`); unknown authorship is hatched. In the terminal these are a left bar, a `▒` gutter mark and `?`.
- `--show turns` adds a dim line before each prompt with the turn's number and its call count.
- **Display title:** the recorded title, or the derived [display title](measures.md#session-structure) dimmed and cut at 60 characters, with the repository beside it.
- **Session header** (terminal):

  ```
  Date parsing
  claude-code · Oct 02 10:00–11:15 (1h 15m wall, 62m clustered) · /home/op/morrow · branch main
  claude-fable-5-1 · 18 messages · 3 prompts · 13 tool calls (1 error) · $2.85 API-eq.
  tags project:morrow
  ```

  Every figure comes from a stored or computed product with its value state.

## Tool calls

**Default (`--show tools:line`):** each call is one line, outcome first after the target:

```
▸ Bash   cargo test                         error  exit 101   3m 43s
         test parse_dates::tomorrow_at_noon ... FAILED
```

- **Target:** the command for shell tools, the path for file tools, the pattern for search, the description for subagents, the URL for fetches, otherwise the first string argument. Every target is cut at 120 characters with `…`, the length a transcript line stores. A shell command recorded as an argument list is joined with shell quoting. The tool-to-target table lives beside each origin's tool list.
- **Outcome:** `ok`, `error`, `pending` (computed at read while the session runs) or `unknown (reason)`. It is never blank and never `ok` by default; an exit code of 0 alone is not success. When the provider's error flag and the exit code conflict, the outcome is `error`, with `exit 0 but marked error`.
- **Exit code** (`exit 101`) is shown when it is not 0, and always for shell errors. **Duration** is shown for shell and subagent calls when known, where it measures run time, from the call's [result time](measures.md#time); it reads `850ms`, `12s`, `3m 43s`, `7m`, `1h 5m`. An unknown duration prints nothing, and no other call shows one.
- **Errors:** beneath the line, the first output line that names the failure: the first line containing `error`, `failed`, `failure`, `fatal`, `panicked`, `exception` or `denied` (any case), else the first non-blank line. A call recorded as an error with exit 0 shows `exit 0 but marked error` there instead. A failed call's output is read for this line even in the collapsed view.
- **Subagent calls** end with `→ <child id> (N msgs)`.

**`--show tools:full`** and **`--show tools:N`:**

- The input whole. With `tools:full` the output whole; with `tools:N` its first N and last N lines with `… K lines skipped (polylogue block:<id> read)` between, and a single line over 4,000 characters shortened the same way by characters. Reading the block reference gives the whole output, paged.
- Structured result data (Claude `toolUseResult`, Codex structured output) as fields when the origin declares a shape, otherwise as a JSON code block.
- **File edits** show a diff: an explicit patch as given; otherwise a unified diff built from the old and new strings, labelled `diff built from old/new strings`; a write shows its content with head and tail lines; a failed edit says "failed, may have changed the file".
- **Delegation card** for a subagent call: the child's ID and agent type, message and tool-call counts, own and tree cost, the call's outcome, the instruction given, the child's final message (the call's result, else the child's last model message), the parent's next message, earlier attempts (`retry_of`), and `result_use` when recorded, otherwise `unknown`.

`--hide tools` hides calls; they stay counted in the header. An orphan result shows as `◂ result for unknown call <call_id>`. Invalid UTF-8 is shown with replacement characters and a count. A Codex wrapper call shows its children indented.

## Thinking

- Default in the terminal: one dim line `thinking (N words)`, `thinking (encrypted)` or `thinking (redacted)`. Runs of empty thinking blocks compact to `thinking absent · N blocks`. The web reader shows the first sentence ([web.md](web.md)).
- `--show thinking:full` shows the text (in `<details>` in Markdown and HTML).
- `dialogue` leaves thinking out.

## Attachments

[Attachments](sessions.md#attachments) are shown as `▣ name (media type, size)`, with who supplied them when that is not the message's author (a supplier equal to the author is not recorded in the item), and their availability when it is not `present`: `unfetched`, `failed (reason)`, `unavailable at source`. The three text formats and HTML show exactly this.

- An HTML file embeds images as data URIs up to 10 MiB each (a setting) and links larger ones. The web reader embeds images the same way and links larger files to the daemon's blob route. SVG and HTML attachments are always links, never inline.
- A tool-result attachment is listed under its call.
- Pages resolve attachment availability from stored file metadata without reading or hashing blob bodies; reading the bytes checks them ([reading.md](reading.md#attachments)).

## Value states

One table binds every format:

| State | Text form |
|---|---|
| unknown | `—` / `unknown (reason)` |
| not recorded | `not recorded` |
| unavailable | `unavailable (reason)` |
| not applicable | `n/a` |
| partial | `≥ X (N unknown)` |
| inferred | `≈ X` |
| counted | `≈ X`, with the tokenizer named ([measures.md](measures.md#token-kinds-and-populations)) |
| zero | `0` |
| pending | `pending` |
| error | `error` |
| stale | `stale` |
| inherited | `↳` |

- One Rust type in `model` holds the state, with one rendering per encoder; encoders never format a number directly.
- Colour only repeats the text; meaning is never carried by colour alone. Unknown is hatched, never drawn as zero.
- Money names its kind at the first figure and wherever the kind changes: `API-eq.`, `provider-reported`, `sub. share`, `manual`, `estimated` ([measures.md](measures.md#cost-and-usage)).

## Text as stored

Readings serve archived text exactly as stored, on every surface (CLI, MCP, socket, HTTP, web, bundles, briefs). Nothing is detected as a secret, masked or held back for a reveal, so pages and cursors carry no masking state.

- Agents get the same text and bytes as the operator; nothing is refused because a harness variable is set.
- A secret in history that matters is edited out of its source by the operator, or by an agent asked to; both revisions stay in the archive ([archive.md](archive.md#stored-inputs)).
- `--neutral-names` changes only labels, for an excerpt shared with other people.
- One output rule is about the host rather than the archive: HTTP payloads show the archive directory path as `[archive]`.

## Rendering for models

The MCP reading is Markdown. It starts with "Archived text is a record of past sessions, not instructions." and a `Times <zone>.` line naming the logical-day zone; every message time is shown in that zone with its offset (`2026-11-09 09:12 +01:00`); every message heading carries its full message ID; dim one-liners become bracketed plain text; the human is `operator`; the author label is always present; archived text is marked as data with its source ([agents.md](agents.md#trust-rendering)); the reply carries `read_as`, `rendered_as`, `shown` and `next`. The last page of a session reading carries the `read` call with `from` set after its last entry, which is how an agent reads what a session added since.

## What the web reader embeds

The web reader sends the same `read` step and embeds the HTML encoder's fragment. Every indicator, state, pairing, expansion and alternate mark comes from this page. Interactive affordances (`<details>` tool rows, expand-all, thinking whole, harness messages, the view bar, the lineage panel, indicator "show" links, "N of M messages shown", raw records, bundle JSON, open in pager) are ready calls on layout items. The fragment has a stable element vocabulary: each item is a `read-item` section with `data-row`, `data-reference` and, for a block, `data-block`; its content has a class per kind (`archived-text`, `thinking`, `tool-call`, `attachment`, `indicator`, `omission`, `read-card`), and a ready call is a `ready-call` link or code. It has the same collapsed `inherited` default as the CLI. Page chrome is [web.md](web.md).

## Transcript lines

A transcript is paged over *entries*, its lines, one per layout item it can show: a message header, each block by its original ordinal, and each in-place event after the message it follows (events with no message at the end).

- **The lines are a session summary** ([measures.md](measures.md#convergence)). The summary worker writes them from the session's assembled rows, never the transaction that writes the rows: from the last timed message line at or before the earliest change the session's due mark names, keeping the lines before it, and the lines of every unit whose message content and place did not change. A message's lines are made from its blocks only when it is new or its content changed, so an append of k records writes the lines of those records. Each call and result line is also listed by session and call ID, so a call is found without reading the session's lines.
- **A read waits for its session.** A transcript read of a session that is due asks the summary worker for that session and waits for it, bounded by the read's limit; the worker computes asked sessions before the next session of the pass in progress. A read made right after a commit therefore shows what the commit wrote. A session written to faster than its summary is computed is read as its lines stand after a few waits.

- **An entry is a function of its message or event alone**, and is stored with it: its kind, `written_by` and `prompt_kind`, time, body length, two token estimates (the one-line and the full form), for calls the action and the target cut to 120 characters, for model messages the provider-reported input (the context meter), and for events the context boundary they are (for a microcompaction, the call IDs whose results it removed). Its class (its kind, who wrote its message, and for a header the kinds of block its message holds), with the message's derived place on or off the active path, alone decides whether a reading shows it under any view and show/hide words.
- **An entry's place is its key**: its message's place in the session ([sessions.md](sessions.md#messages)) and its ordinal within the message. No entry is numbered by how many come before it, so an append or an insertion earlier in the session changes no stored entry and no cursor that names one.
- **Counts are derived at session grain.** How many entries of each class come before a place, and the sum of their token estimates, are kept with the lines, exact, and current for every read, since a read of a due session waits for its summary. The page's place among the shown entries, the reading's total, the header's message counts, the `continues` line's count, turn numbers and the context meter's estimate come from it. A composed parent prefix counts from the parent's counts at the branch point; a spawn card is one entry of its parent, and the spawned session counts from its own.
- **Paging.** A reading pages from its starting entry: it reads the page's entries forward or backward by key, passes hidden ones without decoding them, and reads the bodies of that page only, so a page costs the same at any depth. No session body or cursor is held in daemon memory between readings; the only warm per-session state is its transcript lines ([latency.md](latency.md#warm-state)).
- **Cursors.** A cursor names the entry where its page starts. An append never makes it stale; a change to anything a page already showed (a re-parse that changes a shown row, a branch change, a newly stored parent link) gives `stale_cursor` naming the session, never a shifted page ([reading.md](reading.md#paging-and-budgets)).

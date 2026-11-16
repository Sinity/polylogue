# Web reader

Session, document and message views use the [workbench frame](web.md#frame)
and the shared [reading pipeline](reading.md).

## Reader views

### Session reader `/s/SESSION`

The most important surface. Message rendering and read views are [reading.md](reading.md)'s; this section gives their web forms.

**Header.** Title (a derived title dimmed with its structural label); a mono line `claude-code:7f3a2c10-0002 · Oct 02 10:00–11:15 (1h 15m) · /home/op/morrow · branch main`; a line `claude-fable-5-1 · 18 messages · 3 prompts · 13 tool calls (1 error) · $2.85 (+ $0.16 in 1 subagent) · tag project:morrow`; a live chip when live; the **provider link** (`open in chatgpt.com`) when the origin records the session's URL. Actions: **Star** `s`, **Pin** `P`, **Annotate** `a`, **Add to bundle** `+`, **Continue elsewhere**, **Copy as CLI command**. The view switcher lists the read views (transcript · dialogue · summary · context · links · changes · live · raw), plus the flow panel, the report and share. Toggles: expand all tools, thinking in full, harness messages, include inherited, show branches, turns. A variant switch appears when variants exist.

**Rails.** The inspector, and a left rail, shown by `m` or automatically above 200 messages, with the **outline** (human prompts, tool failures, compactions, branch points, decisions and other notes, paged) and a **minimap** beside the scrollbar built from time buckets, authors and errors; click to jump. The outline shows no guessed phases.

**Message forms.**

| Element | Web form |
|---|---|
| Message header | `HH:MM` (with the date when the range spans days) · author (`you`, model name, `[claude-code]`, `injected context` with its delivery link, `unknown author`) · the first note right-aligned (`◆ decision: decided: chrono, not time · operator`) with `+N`; clicking expands all notes inline; at 1,440 px and wider notes also sit as margin sidenotes. The operator's messages on `--surface-sunk`. Permalink on the time; hover and focus show the action rail. |
| Text | Markdown rendered, code highlighted, archived HTML inert |
| Thinking | one dim line `thinking (N words)` with the first sentence; full text with the toggle; `thinking (encrypted)` or `(redacted)` |
| Tool call | one line, outcome first: `▸ Tool  target  outcome  [exit N]  [duration]`; on error, the first output line naming the failure beneath; the harness's result message folds into the call line, and a footer says how many were folded. Expanding shows the family body below |
| Durations | shell and subagent calls, when known; never file or search tools |
| Subagent call | `▸ Agent  Explore: Survey date crates  ok  7m  → claude-code:…/a7f3a2c1 (6 msgs)`; expanded, the **delegation card**: child, type, counts, own and tree cost, instruction, final answer, the parent's next message, earlier attempts |
| Harness messages | one dim line; full with the toggle |
| Compaction | in place: `⟲ compacted (182k → 14.5k tokens) · 14:40 · manual · show summary`; the summary is a message shown on request, and the boundary lists what it evicted and kept ([sessions.md](sessions.md#compaction-boundaries)) |
| Branches | `⑂ 2 other versions of this reply · show`; **show branches** indents them under the split |
| Inherited prefix | `↳ continues claude-code:7f3a2c10-0002 (18 earlier messages; show)` |
| Continuation | `↪ continued by claude-code:7f3a2c10-0004 (Oct 03 14:00, resume — repeats these 18 messages, adds 10)` |
| Other events | dim lines: rate limit, away summary, model switch |
| Turns | dividers with burst size, latency, tool calls and cost; abandoned turns say so |
| Pastes | typed text plain; a pasted span shaded with its line saying where it came from; unknown authorship hatched; a paste over 20 lines folds as `paste · 214 lines · copy · copy typed-only` |
| Attachments | images inline; other files `▣ name (media type, size)` with **Download** and **Reveal in file manager** (daemon rows using the desktop's services); SVG and HTML as links, never inline; the availability state and who supplied it ([sessions.md](sessions.md#attachments)) |

**Tool families.** Each recognised tool family has its own expanded body, and every family still collapses to the one line above. A call's family comes from the tool-family table of the derived data ([measures.md](measures.md#catalog)); an unrecognised tool uses the generic body.

| Family | Expanded body |
|---|---|
| Shell (`Bash`, `exec_command`, …) | the command highlighted, working directory, exit code, duration, stdout and stderr each with first and last 100 lines and `… N lines skipped (open block)`; when the command ran through a declared wrapper (a job or task runner named in configuration), the wrapped command, the wrapper's job ID as the transcript shows it, and the exit status. This command and exit status are the verification record |
| File read | path (opens in the editor at the range), line range, line count, the first lines |
| File edit, write, patch | path; a unified diff with context; whether the edit applied; a failed edit says "may have changed the file" |
| Search (`Grep`, `Glob`, ripgrep in a shell) | pattern, path scope, hit count, hits as `file:line` links with the matching line |
| Web fetch, web search | URL with title and status; for a search, the query and the result list |
| Subagent, task | the delegation card |
| Todo, plan | a checklist with each item's state and which items changed |
| MCP call | server and tool, arguments and result summarised; calls to Polylogue's own MCP server link the objects they returned |
| Notebook edit | cell, kind, diff |
| Image view, screenshot | the image inline |
| Ask user, permission | the question, the answer and who answered (`operator` via web, terminal or harness) |

**Read views.** The views and their names are [reading.md](reading.md#views)'s; on the web, *summary* is a definition list, *context* shows what the model had at a selected message (parents → boundaries → tail), *links* is the tree of linked sessions with own and tree cost, *changes* lists files edited (graded known, possible, unknown), commands with outcomes, commits and pull requests, *live* is this session's Running row with follow and controls, and *raw* is the stored input records, inert. The **flow panel** renders the `delegations` and `deliveries` units as the work tree of [Running](web-views.md#running-running), and the **report** composes summary, changes, the session's tool calls and its claims; neither is a read view, because both are compositions of query units.

**Reading pages.** The reader reads a session in pages through the `read` row with a budget ([reading.md](reading.md#budgets)); the web default is 8,192 tokens. Paging links are First · Previous · Next · Latest; anchors are `/s/ID?message=MID`; changing the view starts a new read at the page's message anchor. The content cursor and the links cursor are independent URL parameters; the outline is the `turns` reading of the session, and the failure of an optional panel is shown beside the content. A stale cursor offers "Read changed · refresh from this message"; pages are never joined across a stale cursor. Long sessions keep a window of the DOM and fetch by anchor when scrolling past it.

**Interactions.**
- **Action rail** per message on hover and focus: **Copy**, **Annotate**, **Decision**, **Tag**, **Star**, **Add to page or bundle**, **Compare from here**, **Continue from here**, **Raw**, **Source**, **Permalink**.
- **Range selection** (shift-click two headers, or `x` on each): **Share excerpt**, **Add range to page**, **Copy range as Markdown**, **Copy as context**.
- **Find in session:** a field in the view bar runs a message query scoped to the session, jumps by anchor and steps with `n`/`N`; matches inside folded blocks open a match window. The browser's own find sees only loaded pages.
- **Outcome:** `1`/`2`/`3` and a select in the facts.
- **Open in editor** (`o`): the session's working directory, or a tool line's file at its line, through the desktop's services; disabled with a reason when the path does not exist on this machine.

**Continue elsewhere,** on the session and on every message:
- **Resume this session** in its own harness (for a live session, the [controls](#live-session-control-panel) instead): runs the harness's resume command headlessly with an optional prompt, or opens it interactively in the operator's terminal; the full command is shown before it runs ([agents.md](agents.md)).
- **Continue from here in another harness:** a new session seeded with the effective context at that message (`context --at MESSAGE`) and the message text, built by the pack builder and recorded as a delivery. It is never labelled "resume" and creates no lineage link by itself: a `reference` link appears only when the new session's transcript shows the delivery.
- **Copy as prompt:** the same pack, copied. **Takeover brief:** the takeover pack for this session, copied or sent ([agent-context.md#takeover](agent-context.md#takeover)).

**Context bundles.**
- A bundle is a selection of targets (messages, ranges, paste spans, attachments, lineage facts, notes, kept results, query results) held as a page with the `bundle` template, so saving it is saving a page, and it stores references, never copies.
- **Copy as context** on a selection, a bundle or a page asks the pack builder for the rendered pack in Markdown, plain text or JSON with its **manifest**: every included reference with its `seen_hash`, what was included in full, what can be expanded, and what was left out and why. The pack carries its delivery ID in its header line.
- Copying records a delivery with state `prepared` and recipient `clipboard`; when a session's text later carries that delivery ID, delivery use records `used` with that message ([agent-context.md#deliveries](agent-context.md#deliveries)). Sending the bundle to a session (ping or continue) records a `delivered` delivery.
- While a bundle is open, the inspector's **Bundle** panel is its composer: items in order, drag or `+` to add, `Alt-↑/↓` to reorder, the manifest preview, size in tokens, and the target (clipboard, a harness, a session by address).

**Variants.** When the session has variants ([assertions.md](assertions.md#variants)), the header shows `≈ en` and a variant select; the enabled-variants setting chooses the default, and every rendered variant carries a visible mark. **Side by side** aligns original and variant message by message through the variant's mapping. Notes on a variant are the translated notes, each linking its original.

**Related sessions.** An inspector panel lists sessions ranked by `related:REF`; while that product is unavailable (no recall figure, or model calls off), the panel shows `unavailable: <reason>`.

Rows: `read` with budget, slices, anchors, cursors and views (the outline is `turns where session:REF` read); `attachment` (descriptors, bytes, show in file manager); `assert`; `trace`; `pack`; `open-editor`; the live-session control rows.

### Live-session control panel

Shown under the transcript's end in the reader of a live session and inline in each Running row. The operations and their per-harness mechanisms are [agents.md](agents.md#live-sessions)'s; this is their web form.
- **Composer:** a text field (`i`) with **Send**. While the session is working, Send steers it (the harness decides whether the input queues or interjects); while it is waiting or idle, Send answers. The composer shows which channel will carry the text and, for an ended session, the full resume command it will run.
- **Pending permission:** the request (tool, input in full) with **Approve** and **Deny**, and the harness's "always" variant where it has one.
- **Interrupt**, **Resume** (ended or stale sessions) and **Ping** (with an address and an optional `about:` target).
- **Branch from here:** from any message, start a new run seeded from that point (the harness's own fork where it has one, otherwise a new session seeded with `context --at MESSAGE`), shown as a sibling in the work tree once its transcript shows the link.

Each control shows its availability per harness, with the reason when disabled ("Codex: no approval channel from outside the terminal", "hooks not installed"). Every act is recorded with its author (`operator`, `via: web`) and appears in the session's flow as a delivery or control event. Each control is the same row as its CLI and MCP form.

### Session report `/s/SESSION/report`

Header `Session report · computed from stored facts`; title; ID line; **Open transcript**, **Share an excerpt**. Cards: **Cost** (`$2.85` · `+ $0.16 in 1 subagent`), **Turns** (`3` · `3 prompts · 18 messages`), **Tool calls** (`13` · `1 error, fixed later`), **Files touched** (`3` · `all edits known`). **Timeline:** a time axis, turn bars labelled by their prompt, a subagent band, tool-call marks by outcome with labels on errors (`cargo test · exit 101 · 3m 43s`), claim marks with their state, a legend; drag to zoom. A **Tool calls** table (time, tool ×N, target, outcome, duration; error rows tinted). The rail holds **Files touched** (path, edits, certainty), **Agent notes, checked** (claim with its check badge, the check, target, author, `read why it said so`), **Lineage and notes**, and compactions and operator interventions when present. "Fixed later" is derived (a failing call followed by a success of the same command), never taken from text. Every mark and row opens the reader at its message.

Rows: `read` with views `summary` and `changes`, the session's `tool_calls`, and the `claim` assertions on it with their check state.

### Share an excerpt `/s/SESSION/share`

A message list with a checkbox per message and note. **Before saving** lists what is included (messages, notes, their check states). The text is written as stored, with nothing flagged, masked or redacted; paths, names and addresses stay. **Save as one HTML file** runs the `read` row with the selected references, `--format html` and `--to` the operator's download directory, resolved by the daemon ([reading.md](reading.md#writing-out)). The confirmation names the written path and counts. The file is self-contained: inline CSS, no scripts, no archive path. Sharing ends at the local file; carrying an excerpt into a web chat is a [chat action](extension.md#chat-actions).

### Document, message and note pages

- `/d/DOCUMENT`: title, purpose, authorship, text, the stored inputs it was read from; `full` and `dialogue` views; paging; the inspector.
- `/m/MESSAGE` redirects to the reader anchor.
- `/n/NOTE`: the revision (kind, author, `via`, target links, body, relations, check state for claims); **revision history** with diffs; a **conflict** view when two heads are current ([assertions.md](assertions.md#current-revisions)), resolved by writing one revision that revises both; withdrawn state; **Referenced by**; judgments on this revision with their authors; actions **Revise**, **Withdraw**, **Judge**, **Add to page**. Revised and withdrawn revisions stay readable.

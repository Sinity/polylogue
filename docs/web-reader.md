# Web reader

Session, document and note views use the [workbench frame](web.md#frame) and the shared [reading pipeline](reading.md). Each is the `read` verb on its selection: the route passes the `read` step's argument names in its query string, and the footer shows the CLI line and the MCP call that give the same reading ([web.md](web.md#routes)).

## Reader views

### Session reader `/s/SESSION`

The most important surface. Message rendering, views and show/hide are [reading.md](reading.md)'s; this section gives their web forms.

**Header.** Title (a derived title dimmed with its structural label); a mono line `claude-code:7f3a2c10-0002 · Oct 02 10:00–11:15 (1h 15m) · /home/op/morrow · branch main`; a line `claude-fable-5-1 · 18 messages · 3 prompts · 13 tool calls (1 error) · $2.85 (+ $0.16 in 1 subagent) · tag project:morrow`; a live chip when live; the **provider link** (`open in chatgpt.com`) when the origin records the session's URL. Actions: **Star** `s` (`tag star`), **Note** `a` (`note`), **Tag** `t` (`tag`), **Brief** (`brief`), **Resume** (`resume`), **Resume in another harness** (`resume --in`), **Copy CLI line**, **Copy MCP call**. The view switcher lists the views of [reading.md](reading.md#views): transcript · dialogue · overview · context · tree · changes · live · raw, plus the flow panel, the report and share. The show/hide bar has one toggle per word of the closed table ([reading.md](reading.md#show-and-hide)): tools, thinking, harness, inherited, branches, spawns, annotations, turns, context; a toggle cycles a word's levels (`tools:line`, `tools:full`) where it has them. An alternate select appears when alternates exist ([reading.md](reading.md#alternates)).

**Rails.** The inspector, and a left rail, shown by `m` or automatically above 200 messages, with the **outline** (human prompts, tool failures, compactions, branch points, decisions and other notes, paged) and a **minimap** beside the scrollbar built from time buckets, authors and errors; click to jump. The outline shows no guessed phases.

**Message forms.**

| Element | Web form |
|---|---|
| Message header | `HH:MM` (with the date when the range spans days) · author (`you`, model name, `[claude-code]`, `injected context` with its delivery link, `unknown author`) · the first note right-aligned (`◆ decision: decided: chrono, not time · operator`) with `+N`; clicking expands all notes inline; at 1,440 px and wider notes also sit as margin sidenotes. The operator's messages on `--surface-sunk`. Permalink on the time; hover and focus show the action rail. |
| Text | Markdown rendered, code highlighted, archived HTML inert |
| Thinking | one dim line `thinking (N words)` with the first sentence; full text with `thinking:full`; `thinking (encrypted)` or `(redacted)` |
| Tool call | one line, outcome first: `▸ Tool  target  outcome  [exit N]  [duration]`; on error, the first output line naming the failure beneath; the harness's result message joins the call line, and a footer says how many were joined. `tools:full` shows the family body below. Output cut to head and tail marks the omitted middle with its exact line count and a link that reads it whole |
| Durations | shell and subagent calls, when known; never file or search tools |
| Subagent call | `▸ Agent  Explore: Survey date crates  ok  7m  → claude-code:…/a7f3a2c1 (6 msgs)`; with `spawns:card`, the **delegation card**: child, type, counts, own and tree cost, instruction, final reply, the parent's next message, earlier attempts; `spawns:full` reads the child inline |
| Harness messages | one dim line; full with `harness:full` |
| Compaction | in place: `⟲ compacted (182k → 14.5k tokens) · 14:40 · manual · show summary`; the summary is a message shown on request, and the boundary lists what it evicted and kept ([sessions.md](sessions.md#compaction-boundaries)) |
| Branches | `⑂ 2 other versions of this reply · show`; `branches` indents them under the split |
| Inherited prefix | `↳ continues claude-code:7f3a2c10-0002 (18 earlier messages; show)`; `inherited` shows them |
| Continuation | `↪ continued by claude-code:7f3a2c10-0004 (Oct 03 14:00, resume — repeats these 18 messages, adds 10)` |
| Other events | dim lines: rate limit, away summary, model switch |
| Turns | with `turns`, dividers with burst size, latency, tool calls and cost; abandoned turns say so |
| Pastes | typed text plain; a pasted span shaded with its line saying where it came from; unknown authorship hatched; a paste over 20 lines shows as `paste · 214 lines · copy · copy typed-only` |
| Attachments | images inline; other files `▣ name (media type, size)` with **Download** (the attachment's `read` as bytes) and **Reveal in file manager** (through the desktop's services); SVG and HTML as links, never inline; the availability state and who supplied it ([sessions.md](sessions.md#attachments)) |
| Context meter | with `context`, the model's context size at each response against its window |

**Tool families.** Each recognised tool family has its own expanded body, and every family still collapses to the one line above. A call's family comes from the tool-family table of the derived data ([measures.md](measures.md#catalog)); an unrecognised tool uses the generic body.

| Family | Expanded body |
|---|---|
| Shell (`Bash`, `exec_command`, …) | the command highlighted, working directory, exit code, duration, stdout and stderr each with first and last 100 lines and `… N lines skipped (read whole)`; when the command ran through a declared wrapper (a job or task runner named in configuration), the wrapped command, the wrapper's job ID as the transcript shows it, and the exit status. This command and exit status are the verification record |
| File read | path (opens in the editor at the range), line range, line count, the first lines |
| File edit, write, patch | path; a unified diff with context; whether the edit applied; a failed edit says "may have changed the file" |
| Search (`Grep`, `Glob`, ripgrep in a shell) | pattern, path scope, hit count, hits as `file:line` links with the matching line |
| Web fetch, web search | URL with title and status; for a search, the query and the result list |
| Subagent, task | the delegation card |
| Todo, plan | a checklist with each item's state and which items changed |
| MCP call | server and tool, arguments and result summarised; calls to Polylogue's own MCP server link the objects they returned |
| Notebook edit | cell, kind, diff |
| Image view, screenshot | the image inline |
| Ask user, permission | the question, the reply and who answered (`operator` via web, terminal or harness) |

**Views on the web.** *overview* is a definition list (when, where, model, tools with errors, cost, how it ended, files, failing checks, notes); *context* shows what the model had at a selected message (parents → boundaries → tail), and its **What changed** control reads with `--diff` against an earlier message or a compaction; *tree* is the tree of the session's continuations, subagents and forks with own and tree cost; *changes* lists files edited (graded known, possible, unknown), commands with outcomes, commits and pull requests; *live* is this session's Running row with follow and controls; *raw* is the stored input records, inert. The **flow panel** renders the `delegations` and `deliveries` units as the work tree of [Running](web-views.md#running-and-waiting-running-waiting), and the **report** composes overview, changes, the session's tool calls and its claims; neither is a view, because both are compositions of query units.

**Reading pages.** The reader reads a session in pages through `read` with a budget ([reading.md](reading.md#paging-and-budgets)); the web's budget is 8,192 tokens a page. Paging links are First · Previous · Next · Latest, taken from the reply's `next` calls; a message's permalink is `/s/SESSION?from=message:MID`; changing the view or a show/hide word starts a new read at the page's first message. The content cursor and the flow panel's cursor are independent URL parameters; the outline is the session's `turns` listing, and the failure of an optional panel is shown beside the content. A stale cursor offers "Read changed · refresh from this message"; pages are never joined across a stale cursor. Long sessions keep a window of the DOM and read by position when scrolling past it.

**Interactions.**
- **Action rail** per message on hover and focus: **Copy**, **Note**, **Decision** (`note --as decision`), **Tag**, **Star**, **Compare from here**, **Resume from here in another harness** (`resume --in HARNESS --at MESSAGE`), **Raw**, **Source**, **Permalink**.
- **Range selection** (shift-click two headers, or `x` on each): **Share excerpt**, **Copy range as Markdown**, **Copy as context** (the range's `brief`, in Markdown).
- **Search in the session:** a field in the view bar runs a message query scoped to the session (`messages where session:REF …`), jumps by position and steps with `n`/`N`; a match inside a collapsed block opens a window around it. The browser's own search sees only loaded pages.
- **Outcome:** `1`/`2`/`3` and a select in the facts (`note solved --as outcome`).
- **Open in editor** (`o`): the session's working directory, or a tool line's file at its line, through the desktop's services; disabled with a reason when the path does not exist on this machine.

**Resume and brief** ([session-control.md](session-control.md#resume), [agent-context.md](agent-context.md#brief)):
- **Resume** reopens the session in its own harness in the directory where it ran. The browser is not a terminal, so the reader shows the command `resume` would run, with **Copy** and **Open in terminal** (through the desktop's services). A running session is not reopened: the reader offers live reading and the composer instead.
- **Resume in another harness** (`resume --in HARNESS`, from a message with `--at MESSAGE`; **Directory** sets `--cwd`; **Stop the source** sets `--stop-source`) starts a new hosted session of that harness briefed with this session's brief and the message to continue from. It first shows what it would send and through which channel (the step's dry run), then starts on confirmation. The new session records what it was briefed from; it is never labelled a resume and creates no lineage link by itself.
- **Brief** shows this session's brief in the form that takes it over (`brief of a session: takeover of …`), with **For** to name the session that will continue (`--for SESSION`) and **Copy**. A brief the operator asks for records no delivery.

**Alternates.** When the session has alternates ([alternates.md](alternates.md)), the header shows `≈ en` and an alternate select (`--alternate KEY|none`); the enabled-alternates setting chooses the default, and every rendered alternate carries a visible mark. **Side by side** (`--beside`) aligns original and alternate message by message through the alternate's parts. Notes on an alternate are the translated notes, each linking its original.

**Related sessions.** An inspector panel lists sessions ranked by `related:REF`; while that product is unavailable (no measured quality figure, or model calls off), the panel shows `unavailable: <reason>`.

Lines: `polylogue claude-code:7f3a2c10 read --view dialogue --show tools:full`, `polylogue message:01JB7Q2M read --around 5`, `polylogue claude-code:7f3a2c10 read --view context --diff compaction`, `polylogue claude-code:7f3a2c10 brief --for self`, `polylogue claude-code:7f3a2c10 resume --in codex --at message:01JB7Q2M`, `polylogue message:01JB7Q2M note 'decided: chrono, not time' --as decision`.

### Live-session control panel

Shown under the transcript's end in the reader of a live session and inline in each Running row. The acts and their per-harness channels are [session-control.md](session-control.md)'s; this is their web form.
- **Composer** (`i`): a text field with **Send** (`send TEXT`), an **About** target (`--about REF`) and the channel that will carry the text (the hosted session's connection, or its next hook event). While the session waits on a question, the composer becomes **Answer** (`answer 'TEXT'`). For an ended session, Send resumes it headlessly with the text, and the composer says that this starts a harness process.
- **Pending permission:** the request (tool, input in full) with **Allow** and **Deny** (`answer allow`, `answer deny` with an optional reason), and **Always allow** where the harness has one (`--standing`).
- **Held web-chat turn:** the turn's text with **Allow**, which releases it, and **Deny**, which cancels it.
- **Interrupt** (`interrupt`): stops the current turn of a session Polylogue hosts.
- **Read live** (`read --follow`): appends what the session does as it happens.
- **Resume** (ended or stale sessions) and **Resume in another harness**, as in the reader.

Each control shows its availability per harness, with the reason when disabled ("Codex: no approval channel from outside the terminal", "hooks not installed", "not hosted by Polylogue"). Every act is recorded with its author (`operator`, `via: web`), appears in the session's flow as a delivery or control event, and cannot be undone, which the control says. Each control sends the same step as its CLI and MCP form.

Lines: `polylogue codex:019d0f2a send 'rebase on master first'`, `polylogue codex:019d0f2a answer deny --reason 'not on main'`, `polylogue codex:019d0f2a interrupt`, `polylogue codex:019d0f2a read --follow`.

### Session report `/s/SESSION/report`

Header `Session report · computed from stored facts`; title; ID line; **Open transcript**, **Share an excerpt**. Cards: **Cost** (`$2.85` · `+ $0.16 in 1 subagent`), **Turns** (`3` · `3 prompts · 18 messages`), **Tool calls** (`13` · `1 error, fixed later`), **Files touched** (`3` · `all edits known`). **Timeline:** a time axis, turn bars labelled by their prompt, a subagent band, tool-call marks by outcome with labels on errors (`cargo test · exit 101 · 3m 43s`), claim marks with their state, a legend; drag to zoom. A **Tool calls** table (time, tool ×N, target, outcome, duration; error rows tinted). The rail holds **Files touched** (path, edits, certainty), **Agent notes, checked** (claim with its check badge, the check, target, author, `read why it said so`), **Lineage and notes**, and compactions and operator interventions when present. "Fixed later" is derived (a failing call followed by a success of the same command), never taken from text. Every mark and row opens the reader at its message.

Lines: `polylogue claude-code:7f3a2c10 read --view overview`, `polylogue claude-code:7f3a2c10 read --view changes`, `polylogue 'tool_calls where session:claude-code:7f3a2c10'`, `polylogue 'annotations where kind:claim about:session:claude-code:7f3a2c10'`.

### Share an excerpt `/s/SESSION/share`

A message list with a checkbox per message and note. **Before saving** lists what is included (messages, notes, their check states). The text is written as stored, with nothing flagged, masked or redacted; paths and addresses stay. **Neutral names**, on by default here, labels the human `user` instead of `you` (`--neutral-names`, [reading.md](reading.md#neutral-names)). **Save as one HTML file** asks the read route for the selected references with `format=html` and the browser saves the reply as a download; the confirmation names the file and counts. The file is self-contained: inline CSS, no scripts, no archive path. The same excerpt from the command line:

```console
$ polylogue 'message:01JB7Q2M OR message:01JB7Q3K' read --format html --neutral-names --to excerpt.html
```

Sharing ends at the local file; carrying an excerpt into a web chat is a `send` to that web-chat session, held for the operator's approval ([session-control.md](session-control.md#send)).

### Document, message and note pages

- `/d/DOCUMENT`: title, purpose, authorship, text, the stored inputs it was read from; the `transcript` and `dialogue` views; paging; the inspector.
- `/m/MESSAGE` redirects to the reader at that message.
- `/n/NOTE`: the revision (kind, author, `via`, target links, body, relations, check state for claims); **revision history** with diffs (`annotations where root:X withdrawn:*`); a **conflict** view when two revisions are current ([annotations.md](annotations.md#current-revisions)), resolved by writing one revision with `edit`; withdrawn state; **Referenced by** (`annotations where links_to:X`); judgments on this revision with their authors; actions **Edit** (`edit`), **Withdraw** (`withdraw`), **Judge** (`judge accept|reject|defer`). Edited and withdrawn revisions stay readable.

Lines: `polylogue annotation:01JB7Q2M read`, `polylogue annotation:01JB7Q2M edit 'honours Retry-After'`, `polylogue annotation:01JB7Q2M@2 judge accept`.

# Web views

These views use the [workbench frame](web.md#frame). Session and document
reading is described in [Web reader](web-reader.md).

## Views

### Home `/`

The orientation of bare `polylogue`, as a page:
1. The query field, focused on load; **Search**; **Copy as CLI command**.
2. Cards: **Sessions** (count, top-level count, last activity); **Last 14 days** (prompts, tool calls, errors); **Known cost** (with its money kind and the unknown count); **Waiting on you · running** (link to Running); **Quota window** (current subscription window, API-equivalent so far, reset time, reported percentage); **Notices**, only when there is one.
3. **Waiting on you**, when not empty: one row per waiting session with what it waits for and the [live-session controls](web-reader.md#live-session-control-panel).
4. **Recent sessions:** logical sessions ([sessions.md](sessions.md#logical-sessions)), each expandable into its physical sessions; columns last active · session · title · links (`fork of …`, `continues …`, `1 subagent`, `⑂ 1 abandoned branch`) · messages · cost (own cost, with `+ $X sub` muted beside it; a tree-cost column on request; the header names the money kind) · notes and tags · the provider URL when known.
5. **Recent notes:** time · kind · text · author · target.
6. **Sources:** source · location · inputs · sessions · last new; unrecognised record types and parse problems.
7. On wide screens the inspector lists saved queries and recently opened pages.

Every number links to the query that produced it. Rows: `status`; `query` per table; `query:running`; measures through `| measure` stages.

### Running `/running`

The rows are the shipped saved query `query:running`.
- Title `Running now · 2 sessions · 1 waiting on you · 15:00`.
- **Window line:** `Anthropic 5h: $X API-equivalent so far · resets HH:MM · week N% (reported)`.
- One section per repository (worktrees grouped under their repository), headed by its path, with a peer warning (`⚠ 2 agents in /home/op/morrow`) when several live sessions share it.
- One row per session: state word and since; title; reference, start, pid; what it is doing (`waiting on you: permission for Bash: cargo test --all`, or `open tool call: exec, since 14:58:30`); the **process tree** from the sampler (`codex (pid 5202) → cargo test parser (pid 5230, 14:58:32)`), each child linked to the tool call whose pre and post hook events bracket its start; model; cost so far; task (a task or job identifier the transcript or hook line names); subagents nested as a tree.
- Order: waiting, working, idle, unknown. **Stopped** lists `stale` sessions with the reason (`process gone at 13:40, no end event`).
- Footer: the state legend and `Same rows as polylogue query:running`.

**Work tree and context flow.** A row expands into its logical session's physical sessions (subagents, forks, continuations) with what flowed on each edge: subagent instruction and final answer; brief, takeover and ping deliveries (prepared, delivered, used; size; references; what was left out); compaction boundaries with what they evicted. The same graph is the reader's flow panel.

A row opens the reader at Latest with following on. Each live row carries the [live-session controls](web-reader.md#live-session-control-panel) inline, and a waiting row's pending permission shows **Approve** and **Deny** directly. **Open in terminal** appears when the harness's terminal is known through the desktop's services; otherwise the resume command is offered to copy. `x` selects rows for a stack. The view updates in place, recomputes at each state's `until`, reorders only when no row has focus, and announces "1 session now waiting" to screen readers.

### Search `/search?q=…`

- **Left rail:** facets for the unit, with whole-population counts (sessions: origin, repository, harness, model, date, live state, tag, outcome, `has:` tool_use · thinking · paste; messages add author; tool calls add tool family and outcome), with **saved queries ★** below them. Editing a facet edits the parsed condition, keeping the rest of the expression.
- **Summary line** (`3 sessions · 8 matching messages for chrono`); the **read as** line, the query read back in words; the **defaults line** (`Excluded by default: inherited copies (#4 repeats #2), messages on abandoned branches (1 hit in chatgpt:68f0a1b2). include them · explain this query`).
- **Session and message results** group by session: title; `id · date · N msgs · cost · tags`; hits `HH:MM · author — snippet` with the match marked; subagent hits as `↳ subagent claude-code:… (Explore)` under the parent; notes on a hit inline; `N more hits in this session`; row numbers `1`, `.1`, `.2` matching the CLI's `@1.3` ([terminal.md#addressing](terminal.md#addressing)).
- Other units use the shared results table with reference columns linked.
- **Also in tool output: N hits · show them** when the defaults exclude tool output.
- **Refine:** deterministic suggestions from the parsed query (narrower origin, author, a grouped count, related notes), each a link; never a model call.
- **Coverage line:** `exact`, or `ranked (semantic): candidates exact, order by similarity` ([search.md](search.md#semantic-ranking)).
- **Wide layout** (1,600 px and wider): results, a reading pane with the focused result in the reader's compact form, and the inspector. `Enter` opens the full reader.

The query field completes from the completion catalog ([completion.md#completion](completion.md#completion)), with result counts as preview. `x` selects hits or sessions; the selection bar offers **Annotate**, **Tag**, **Add to page**, **Add to bundle**, **Keep result**, **Compare** (two sessions), **Open as stack** and **Copy as context**. **Save query** names the query's hash, **Keep result** keeps its rows, **Watch** creates a signal rule ([query.md](query.md#watches)). **Annotate all matches** previews the population, then writes one note per row. The field is the query line, so a bare word follows the [grammar](interfaces.md#grammar): it is refused with the `find WORDS` form shown as the correction. Results are a snapshot with a read position; a banner "N new matches since this page was read · refresh" appears when the stream reports matching changes.

Rows: `query` (with `why`), `explain`, `complete`, `save`, `keep`, `watch`, `assert` over a query, facet counts, `refine`.

### Notes, claims, judging, triage

- **`/notes`:** filters by kind, author (operator, agents, a session, `unattributed`, `via`), target, judged state (unjudged by the operator, accepted, rejected), conflict, stale (target text changed since its `seen_hash`) and withdrawn. Rows: kind badge, body, target, author, time, judgments, check badge. **Needs review** lists current, unwithdrawn notes by other authors without an operator judgment. Bulk actions: judge, withdraw, tag. Any author's judgment is shown as that author's opinion; acceptance means an operator judgment ([assertions.md](assertions.md#judgment)).
- **`/claims`:** claims are `claim` assertions. Each shows its **check** badge (`holds`, `fails`, `unknown` with the reason, `unchecked`) with the check query, `expects rows · got N` and the failing rows linked; the verification it rests on is shown as the check command and exit status the transcript records (`cargo test · exit 101`, or a declared wrapper's `job 1234 · cargo test · exit 101`), never as the agent's words. The header reads `checks evaluated now, at read time · never stored · never a verdict`. An accepted claim that the archive now contradicts says so. "Claimed done, last check failed" is a shipped saved query and one filter here.
- **`/judge`:** the command line (`polylogue judge --blind --sample 5 'assertions where rubric:outcomes@r2'`); the rubric card; the item card (`item 4 of 5 · order shuffled · author hidden until the verdict`) with its target and last messages, `a` accept · `r` reject · `d` defer; **This pass so far** (agreement with the operator's judgments on the same rubric revision, deferred, left); **withdraw the whole pass**; **revise the rubric and re-run**. Calibration figures of agent passes against operator-judged samples are shown here.
- **`/triage`:** sessions ordered by the sum of named facts, each with its contribution; the tag `triaged` empties a row.
- **Marks:** star and pin are tags (`tag star`, `tag pin`), shown as toggles on every session and message. "Pin" keeps an item on top of lists; a page slot that holds one revision is "fixed".

Rows: `assertions` queries and `read assertion:…`; `assert`, `revise`, `withdraw`, `judge`; check evaluation in assertion reads; `triage`.

### Pages `/pages`, `/p/PAGE`

The page model, slots, members, roles and pings are [pages.md](pages.md)'s.

**`/pages`:** title with `page:name`, template, items, sessions, last change (`finding from codex …0006`); **New page** with templates; footer `Same list as polylogue page list`.

**`/p/PAGE`:**
- Header `investigation · page:morrow-next-friday · created Oct 06 09:15 by operator`; title; layout switcher **linear · board · table · table of contents · history**; a variant switch when the page has variants; Copy as CLI.
- **Subject:** the question note above the sections. **Sections** carry **Add** on each heading; an empty section renders as one line `empty — add`.
- **Slots** are rendered from the archive at read time and show **following** (latest revision) or **fixed at rev N**:
  - message, message range, session, note, block, span;
  - kept result: its rows `as of …`, "Stored rows, not re-run", **re-run now** as a separate act;
  - live query: `live · N matches`, rows that update while the page is open;
  - figure: a kept figure or a measure query rendered as a chart and its table (as in [Analysis](#analysis-analysisq));
  - file at a commit: the file at that revision in a code view; **History** browses the commits that touched it (from the repository when it is on this machine, else from captured git data, otherwise disabled with the reason) and **Current** shows today's state;
  - Sinex event or material (primary mode): resolved through the [cross-system route](primary-mode.md#cross-system-queries) and rendered from Sinex's answer with a link to Sinex's reader; standalone, the slot shows `unresolved: Sinex not configured` with the reference text;
  - external link: a URL with an operator-written title, never fetched;
  - page: an embedded child page, collapsed; heading.
- **Decision box:** `no decision yet`, a field, **record decision**, the CLI line.
- **Inspector:** **Members** (`@reviewer → codex:01a20000-0006 · last active Oct 06 09:50`; after a resume, `@parser → claude-code:7f3a2c10-0004 (via resume of …0002)`, showing that the member name followed the continuation link) with each member's role and **add member**; the **Ping** composer; the **Pings** list with delivery state (`prepared`, `delivered`, `not_sent:<reason>`, `unknown`, `used`) and reply link; **Sessions on this page**; **Agents write here with** `::finding(page=morrow-next-friday, section=Findings): …`; **What is stored**.
- **Ping composer:** member or explicit address, `about:` (whole page, a section, a slot), text; the line `will run: codex exec resume 01a20000-0006-… "…"` (for a live member, `delivered at its next hook event`); **send ping**. Sending runs it: delivery is [session-control.md#pings](session-control.md#pings)'s, and there is no copy-only default.
- **Directing the model:** an operator annotation on a slot or section that addresses a member (`@[reviewer] expand this`) is a ping like any other.
- **Board layout:** the same slots as cards on a canvas, positions stored as slot values; drag or arrow keys move a card, which writes a slot revision. **Table layout:** slots as rows with kind, target, state, author, last change. **Table of contents:** exactly `polylogue page toc PAGE`. **History:** the page's revisions with author and time, and the page as of a chosen moment by a scrubber.
- **Sidenotes and previews:** on wide screens notes about a slot sit in the margin beside it; hovering any reference shows a preview from one shared reference reader, and a hover never runs a model or changes anything.
- **Add to page from anywhere:** `+` on any target opens a picker of page, section, following or fixed.

Editing a slot's note writes a revision; moving or removing a slot changes only the page; withdrawn slots disappear and show in history. Agents' markers appear live under their section, authored by the message's author. A stale base is refused with the current head shown; two heads from two machines show as a conflict. Live-query, figure and note slots update from the stream; fixed slots never change.

**Templates** are pages, editable by the operator: **investigation** (subject question; Goal, Assumptions, Alternatives, Plan, Evidence, Findings, Open questions, Decision); **project story**; **weekly review** (a week's sessions per logical week and harness, decisions recorded, findings with checks, open items per repository with what resolved them, cost and quota of the week, an empty **Next week** section); **day story** (the logical day's sessions in time order, what each produced, interventions, cost); **feature story** (from the first mention of a subject to its landing commit or pull request, with the sessions and decisions in order); **bundle** and **workspace**. No template writes generated prose; model-written narrative of the recent past is not offered while model calls are off (default; open operator question).

### Workspaces `/w/MODE`

A workspace is a page with the `workspace` template whose slots are the open targets and whose values record mode, lane widths and focus; saving one is saving the page. An unsaved workspace lives in the URL (`/w/stack?ids=A,B,C&focus=B:msg42`).
- **Tabs:** open targets as tabs; `Ctrl-click` opens in the background; tabs show the live state and a short title.
- **Stack:** two to five token kinds, each a compact reader with its own scroll and live chip, and one inspector locked to the active lane. **Stack builders:** parent chain, siblings and forks, subagents, same repository recent, selected search results, sessions sharing a file. More lanes collapse the oldest into tabs.
- **Compare:** [below](#compare-compare).
- **Timeline:** a chronological merge of the lanes' messages, tool calls, notes and lineage events, ordered by time or by session; clock gaps and provider-relative times flagged; tool bursts collapsed (`… tool burst · 42 runs · 22m · expand`).

### Compare `/compare`

Left and right headers (title, origin, relation such as `fork @ msg 42`, counts); alignment **by lineage · by prompt · by time · by shared file**; paired rows with a gutter naming the row type (`same`, `shared paste`, `divergent`, `shared file`, `decision`) and tool outcomes per side; a **metadata diff** above (model, harness, duration, tool calls, errors, cost, outcome); a footer with the rule and counts; **Add both to page**, **Fix paired messages on a page**, **Copy comparison summary**. By lineage, shared message identity and the recorded branch point are exact; by prompt and by time, the row type is labelled as a comparison rule, never "same message". A missing side stays visible (`missing on right`). For more than two sessions, a metadata table with one column per session, compared on matched tasks with denominators. Row: `compare LEFT RIGHT --align …`.

### Timeline, days, firehose

- **`/timeline`:** a year **calendar heatmap** (sessions, prompts, cost or tool errors per logical day, with an accessible table equivalent); below it a **scrubbable time axis** with lanes per repository or harness, sessions as bars, tool errors as marks and notes as `◆`; drag selects a window, the wheel zooms hour ↔ day ↔ week; the table below lists the window's sessions and events. Timeline, table and measures share three bindings: the time window, the focused item (opened at its neighbourhood, never by loading every earlier message), and the file at a moment. The header names the zone and the logical day start ([operations.md](operations.md#configuration)).
- **`/day/DATE`:** the logical day's activity by time (each with its time source), notes and decisions, cost and usage, previous and next, **This day in other years**, **Undated records**. In primary mode a wake-to-wake view of the day, read from Sinex, appears beside it.
- **File at a moment:** a file's state graded by what is recorded: known (after a successful edit), possible (after a failed one), unknown; never by re-running commands.
- **`/firehose`:** the timeline in follow mode with no window end, over AI-session records (messages, tool calls, notes, deliveries) at full detail. The cross-source firehose is Sinex's, a followed query over its confirmed stream; in primary mode `/firehose` links to it.

Rows: `timeline` (window, grain, lanes, sessions and events); heatmap counts grouped by logical day; file at a moment from `changes` data.

### Cost `/cost`

- Title `Cost · since Oct 01` and its CLI command; group by **day · model · origin · repository**.
- Cards: **Known cost** with its money kind and the exact value in a tooltip; **Unknown** (`4 sessions — item-only Codex rollouts and subscription chats`); **Price table** in use; **Most expensive**.
- **By day** with a bar column; unknown days hatched with the word `unknown`; the total `$5.86 + 4 sessions unknown`; sessions count logical sessions, and cost includes subagents.
- **By model** (input, output, cache read, cache write, cost; a `(not recorded)` row; the Codex observation subset priced apart); **By origin**; **By repository**.
- **Most expensive sessions:** own cost, `+ $X subagent`, tool errors; paged, never a fixed top N; sessions of unknown cost listed after, counted, never ranked as zero.
- **Subscription windows:** window, API-equivalent so far, reset, reported percentage; history with rate-limit events.

Every table has its own CLI command. Dollars show two decimals in tables and the exact value in the tooltip. Unknown is never `$0` and never a zero-height bar, and a column never mixes money kinds ([measures.md](measures.md#cost-and-usage)).

### Analysis `/analysis?q=…`

Renders any `| measure` or `| agg` query, or a kept figure, as **a chart and its table**. The chart form follows the answer's shape: a series over time → bars per grain (lines for cumulative measures); a grouped count or sum → horizontal bars sorted by value; a distribution → a histogram or a percentile strip (nearest-rank percentiles); survival → a step curve; a comparison across groups → paired bars with denominators. Every chart shows its unit, denominator query, scope, the count of unknowns (hatched, never zero) and the product's validity note ([measures.md](measures.md#rules)); each bar or point opens the rows behind it. Charts are SVG rendered by the daemon from the typed answer, with the shell adding hover and selection; there is no charting script. **Keep as figure** turns the shown query into a citable figure for a page's figure slot. Shipped saved queries open here first: **where the time went** (time per session, repository or day split into model working, tool running by family, waiting on the operator and idle), **operator interventions**, and **context pressure** (context size per response and what each compaction dropped).

### Pastes, attachments, topics

- **`/pastes`:** paste spans across the archive grouped by content hash, with repeat counts, first and last use, and sessions; **find repeats**. Backed by the `pastes` unit.
- **`/attachments`:** a grid and list of attachments with who supplied them, availability state, size and session; filters by media type and origin. Backed by the `attachments` unit.
- **`/topic?q=…`:** one reading view of every message matching a query across sessions, oldest first, grouped by session with one line of context on each side.

### Project story `/p/story-REPO`

A page from the `project story` template: header `Project story · repository identity root commit [hash] · from stored facts and checked notes only`; a summary line; **Sessions per logical week** (week, Claude Code, Codex, web chats, models, cost), delegated work beside it and never mixed; **What happened** per session (date, link, title, cost, one finding with its check state, decisions, commits); **Outcomes** from sampled annotation passes with the sampling stated (an unsampled pass is labelled unsampled). Commits and pull requests come from captured git data, or from Sinex in primary mode. No generated prose.

### Status `/status`

The figures, states and verdicts are [operations.md](operations.md#status)'s; this is their page.
- **Verdict** (`healthy`, `degraded`, `unknown`) with its reasons.
- **Daemon** (pid, since, up to date); **archive** (path as `[archive]`, sizes); **inputs** (stored, waiting, failed); sessions and messages with abandoned and inherited counts; a per-source table (location, inputs, sessions, last new, **live lag**); unrecognised record types; parse problems; markers not applied; notes with missing targets; conflicts; accepted claims whose check fails; settings changed by anyone but the operator since the operator last looked; enabled signal rules with their authors; embeddings state; primary-mode position and lag.
- Every figure carries its state (`measured`, `none_found`, `not_measured`, `unavailable`); an unmeasured figure is never shown as zero or healthy.
- **Catching up:** inputs waiting, rate, and an ETA from the measured rate.
- **Own use:** query runs by surface and latency class, p50 and p95 per class against its [budget](interfaces.md#latency-classes), most-run queries and fields, reader opens by surface, time spent reading per session; local only, from the recorded query runs and the daemon's counters.
- **Notices** with their channel and delivery state ([operations.md](operations.md#notices)).

### Saved queries and kept results

Saved queries are names on query hashes, listed in the facet rail and at `/notes?kind=saved_query`; each opens `/search?q=query:NAME` live and offers **Keep result**, **Watch**, **Add to page** and **Open in analysis**. A kept result opens at `/r/RESULT` with its stored rows, `as of`, the query and bounds it cites, and **re-run now** as a separate act.

### Settings `/settings`

Settings are archive records with history and author. This page edits those a person sets through the web:
- **View preferences:** density; fold defaults (tool output and pastes over N lines); single-key shortcuts on or off; theme (system, light, dark) and the palette source; rail and panel state per view; default reader toggles.
- **The operator's facts and open choices** ([operations.md](operations.md#configuration)): logical day (read from Sinex in primary mode), subscription plan and price table, enabled variants, quiet hours and interrupt budget (Sinex's in primary mode), listen address and authentication, the optional away command.

Every change shows who made it and when.

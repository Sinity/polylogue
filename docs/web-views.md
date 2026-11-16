# Web views

These views use the [workbench frame](web.md#frame). Session and document reading is described in [Web reader](web-reader.md); the page model the Pages views draw is [pages.md](pages.md). Every view is a line or a command, and its footer shows that CLI line and the MCP call that reproduce it ([web.md](web.md#frame)); the HTTP routes behind the views are in [web.md](web.md#routes).

## Views

| View | Route | Line or command |
|---|---|---|
| [Home](#home-) | `/` | the sections of bare `polylogue`: `polylogue waiting`, `polylogue running`, `polylogue recent`; `polylogue status` |
| [Running, Waiting](#running-and-waiting-running-waiting) | `/running`, `/waiting` | `polylogue running`, `polylogue waiting`; `read --follow`; `answer`, `interrupt`, `send` |
| [Search](#search-searchq) | `/search?q=QUERY` | the listing `polylogue 'QUERY'` with `polylogue 'QUERY \| measure facets'` for the rail; `save` |
| [Session reader](web-reader.md#session-reader-ssession) | `/s/SESSION` | `polylogue SESSION read` with every view and show/hide word; `note`, `tag`, `edit`, `withdraw`, `send`, `resume` (with `--in`), `brief` |
| [Notes, claims, triage](#notes-claims-and-triage-notes-claims-triage) | `/notes`, `/claims`, `/triage` | `polylogue annotations`, `polylogue 'annotations where kind:claim'`, `polylogue triage` |
| [Review and compare](#review-and-compare-review-compare) | `/review`, `/compare` | `polylogue review`; `judge`; `polylogue result:01JD7Q4M read --view compare --blind` on a kept result; `read --view compare --align prompt\|lineage\|time\|file` |
| [Judging passes](#judging-passes-judge) | `/judge` | `\| sample`, `\| measure agreement`, forecasts' calibration, passes ([judging.md](judging.md)) |
| [Pages](#pages) | `/pages`, `/page/NAME` | `polylogue pages`; `polylogue page:NAME read` with its views and `--at`; `save page:NAME`, `note --section`, `withdraw`, `import` of a working copy |
| [Project story](#project-story) | `/page/NAME` | a page from the `project-story` template |
| [Workspaces](web.md#workspaces) | `/w/MODE`, `/page/NAME` | a page from the `workspace` template |
| [Timeline, days, firehose](#timeline-days-firehose) | `/timeline`, `/day/DATE`, `/firehose` | time conditions and `\| group by day(at)`; `polylogue recent read --follow` |
| [Cost](#cost) | `/cost` | `\| measure usage`, `quota`, `outlook`, `daily`; the shipped page `page:cost` |
| [Analysis](#analysis-analysisq) | `/analysis?q=QUERY` | any `\| measure` or `\| agg` query |
| [Pastes, attachments, topics](#pastes-attachments-topics) | `/pastes`, `/attachments`, `/topic?q=QUERY` | `polylogue pastes`, `polylogue attachments`, `polylogue 'QUERY' read` |
| [Status, Settings](#status-and-settings-status-settings) | `/status`, `/settings` | `polylogue status`; `polylogue set` (listing and writing) |
| [Saved and kept](#saved-and-kept-saved-rresult) | `/saved`, `/r/RESULT` | `polylogue saved`; `polylogue result:01JD7Q4M read`; "run again" is `query:<hash>` |

### Home `/`

The sections of bare `polylogue` ([terminal.md](terminal.md#the-orientation-screen)), as a view, from the same status snapshot. Both render the shipped page `page:home` ([pages.md](pages.md#templates-and-shipped-pages)), so revising that page changes the terminal's orientation screen and this view together:
1. **The status line:** health, session count, how up to date, the quota windows (`claude 5h 61 % (resets 16:40) · codex week 34 %`). The query field above it is focused on load.
2. **Waiting on you** (the rows of `waiting`), when not empty: one row per waiting session with what it waits for and the [live-session controls](web-reader.md#live-session-control-panel); a permission request shows **Allow** and **Deny** directly.
3. **Running** (the rows of `running`): state word and since, what it is doing, repository.
4. **Recent** (the rows of `recent`): logical sessions ([sessions.md](sessions.md#logical-sessions)), each expandable into its physical sessions; columns last active · session · title · links (`fork of …`, `continues …`, `1 subagent`, `⑂ 1 abandoned branch`) · messages · cost (own cost, with `+ $X sub` muted beside it; the header names the money kind) · notes and tags · the provider URL when known. A browser has no working directory, so where the terminal shows recent sessions *here*, the web shows all recent ones with a repository filter.
5. **Next:** the actions bare `polylogue` offers as `next:` (answer the first waiting session, read the first running one live, this week's sessions), as buttons.

Below the sections: cards **Last 14 days** (prompts, tool calls, errors), **Known cost** (with its money kind and the unknown count) and **Notices** (the rows of `notices`, only when there is one); **Recent notes** (time · kind · text · author · target); **Sources** (source · location · inputs · sessions · last new; unrecognised record types and parse problems). On wide screens the inspector lists saved queries.

Every number links to the query that produced it.

### Running and Waiting `/running`, `/waiting`

**Running** shows the rows of the shipped saved query `running`.
- Title `Running now · 2 sessions · 1 waiting on you · 15:00`.
- **Window line:** `Anthropic 5h: $X API-equivalent so far · resets HH:MM · week N% (reported)`.
- One section per repository (worktrees grouped under their repository), headed by its path, with a peer warning (`⚠ 2 agents in /home/op/morrow`) when several live sessions share it.
- One row per session: state word and since; title; reference, start, pid; what it is doing (`waiting on you: permission for Bash: cargo test --all`, or `open tool call: exec, since 14:58:30`); the **process tree** (`codex (pid 5202) → cargo test parser (pid 5230, 14:58:32)`), each child linked to the tool call whose pre and post hook events bracket its start; model; cost so far; task (a task or job identifier the transcript or hook line names); subagents nested as a tree.
- Order: waiting, working, idle, unknown. **Stopped** lists `stale` sessions with the reason (`process gone at 13:40, no end event`).
- Footer: the state legend and the line `polylogue running`.

**Work tree and context flow.** A row expands into its logical session's physical sessions (subagents, forks, continuations) with what flowed on each edge: subagent instruction and final reply; briefs (a takeover included) and sent messages, each with its delivery state (prepared, delivered, used), size, references and what was left out; compaction boundaries with what they evicted. The same graph is the reader's flow panel.

A row opens the reader at the latest message, reading live (`read --follow`). Each live row carries the [live-session controls](web-reader.md#live-session-control-panel) inline. **Open in terminal** appears when the harness's terminal is known through the desktop's services; otherwise the `resume` line is offered to copy. `x` marks rows; the selection bar offers **Send** (`send` to each; ended sessions are listed as skipped, each with the line that sends to it alone) and **Interrupt**. The view updates in place, recomputes at each state's `until`, reorders only when no row has focus, and announces "1 session now waiting" to screen readers.

**Waiting** shows the rows of `waiting`, the longest wait first, each with what it waits for and its answer in place:
- a permission request: the tool and its input in full, **Allow**, **Deny** with an optional reason, and **Always allow** where the harness has one (`answer allow --standing`);
- a question the session asked: the question and a text field whose submit is `answer 'TEXT'`;
- a held web-chat turn: the turn's text, **Allow** to release it and **Deny** to cancel it.

The first answer wins, from any surface; a late one shows what won. Lines: `polylogue waiting`, `polylogue codex:019d0f2a answer allow`, `polylogue codex:019d0f2a answer 'use the staging database'`.

### Search `/search?q=…`

- **Left rail:** facets for the unit from `| measure facets` in the listing's reply, with whole-population counts, scoped and global (sessions: origin, repository, harness, model, date, live state, tag, outcome, `has:` tool_use · thinking · paste; messages add author; tool calls add tool family and outcome), with **saved queries ★** below them. Editing a facet rewrites its condition's span in the parsed query ([web.md](web.md#routes)), keeping the rest of the query as typed.
- **Summary line** (`3 sessions · 8 matching messages for chrono`; a single plain word adds `searched the word "chrono"`); the **read as** line, the selection read back in words as `--explain` gives it; the **defaults line** (`Excluded by default: inherited copies (#4 repeats #2), messages on abandoned branches (1 hit in chatgpt:68f0a1b2). include them · explain this line`), where **include them** adds the conditions that lift those defaults.
- **Session and message results** group by session: title; `id · date · N msgs · cost · tags`; hits `HH:MM · author — snippet` with the match marked; subagent hits as `↳ subagent claude-code:… (Explore)` under the parent; notes on a hit inline; `N more hits in this session`. Opening a hit reads its session from the hit.
- Other units use the shared results table with reference columns linked.
- **Also in tool output: N hits · show them** when the defaults exclude tool output.
- **Refine:** suggestions from the parsed query (narrower origin, author, a grouped count, related notes) that come in the listing's reply, each a link; never a model call.
- **Coverage line:** `exact`, or `ranked (semantic): candidates exact, order by similarity` ([search.md](search.md#semantic-ranking)).
- **Wide layout** (1,600 px and wider): results, a reading pane with the focused result in the reader's compact form, and the inspector. `Enter` opens the full reader.

The query field is the selection of a line and completes from the completion catalog ([completion.md](completion.md#positions)), with the count as preview. A single plain word searches for that word, as on the command line ([interfaces.md](interfaces.md#the-first-word)). `x` marks hits or sessions; the selection bar offers **Note** (`note`), **Tag** (`tag`), **Save rows** (`save` with no name, giving `result:<id>`), **Compare** (`read --view compare`, two or more sessions), **Add to page** (`save page:NAME`, one slot per row, with the section and mode chosen in the `+` picker), **Pin** (`save page:pinned`) and **Copy as context** (the marked rows' `brief`). **Save query** names the question and keeps today's rows (`save NAME`); **Save and watch** adds notices of rows that start matching (`save NAME --watch`) ([saved-queries.md](saved-queries.md#save)). **Note all matches** and **Tag all matches** show the count and the first rows, then send the step with `all`. Results are a snapshot with a read position; a banner "N new matches since this view was read · refresh" appears when the stream reports rows that start matching.

Lines: `polylogue 'rate limit'`, `polylogue 'rate limit | measure facets'`, `polylogue 'rate limit' save rate-limits --watch`, `polylogue 'repo:sinex rate limit' tag rate-limits --all`.

### Notes, claims and triage `/notes`, `/claims`, `/triage`

- **`/notes`:** the `annotations` listing, its filters written as conditions: kind, author (operator, agents, a session, `unattributed`, `via`), target, judged state (unjudged by the operator, accepted, rejected), conflict, stale (target text changed since its `seen_hash`) and withdrawn (`withdrawn:*`). Rows: kind badge, body, target, author, time, judgments, check badge. Marked rows take **Judge**, **Withdraw** and **Tag**. Any author's judgment is shown as that author's opinion; acceptance means an operator judgment ([judging.md](judging.md#verdicts-on-annotations)). A note opens its page ([Web reader](web-reader.md#document-message-and-note-pages)).
- **`/claims`:** the `claim` annotations. Each shows its **check** badge (`holds`, `fails`, `unknown` with the reason, `unchecked`) with the check query, `expects rows · got N` and the failing rows linked; the verification it rests on is shown as the check command and exit status the transcript records (`cargo test · exit 101`, or a declared wrapper's `job 1234 · cargo test · exit 101`), never as the agent's words. The header reads `checks evaluated now, at read time · never stored · never a verdict`. An accepted claim that the archive now contradicts says so. "Claimed done, last check failed" is one filter here.
- **`/triage`:** the rows of the shipped saved query `triage`, sessions ordered by the sum of named facts, each with its contribution; tagging a row `triaged` (`tag triaged`) removes it.
- **Marks:** starring is a tag (`tag star`), shown as a toggle on every session and message.

Lines: `polylogue annotations`, `polylogue 'annotations where kind:claim'`, `polylogue triage`, `polylogue annotation:01JB7Q2M withdraw`.

### Review and compare `/review`, `/compare`

**`/review`** shows the rows of the shipped saved query `review`: annotations by agents and programs that the operator has not judged, newest first ([judging.md](judging.md#verdicts-on-annotations)). Each row is a card: kind, text, author and `via`, target with its last messages, relations, check state. **Accept**, **Reject** and **Defer** (`a`, `r`, `d`), each with an optional reason, judge the exact revision shown. A selector chooses whose judgments count (`judged_by:`, by default the operator). Marked cards are judged in one step, as `polylogue review pick judge accept` does on the command line. A judgment is never a gate, and the queue may stay long.

**`/compare`** reads two or more rows with the `compare` view ([reading.md](reading.md#compare-and-blind)): headers per side (title, origin, relation such as `fork @ msg 42`, counts); paired rows aligned by the chosen rule, with a gutter naming the row type (`same`, `shared paste`, `divergent`, `shared file`, `decision`) and tool outcomes per side; a **metadata diff** above (model, harness, duration, tool calls, errors, cost, outcome); a footer with the alignment rule and counts. The alignment switcher **by prompt · by lineage · by time · by shared file** is `--align prompt|lineage|time|file` ([reading.md](reading.md#compare-and-blind)): by prompt (the default) pairs matching prompts; by lineage pairs a fork and its parent from the branch point; by time interleaves by wall-clock time; by shared file puts edits to the same file side by side. Shared message identity and the recorded branch point are exact; any other pairing is labelled as the alignment rule, never "same message". A missing side stays visible (`missing on right`). For more than two rows, a metadata table with one column per row, compared on matched tasks with denominators. **Prefer** on a dimension records a comparison (`judge REF --on DIMENSION`); **Tie**, **Incomparable**, **Abstain** and **Insufficient evidence** record the other verdicts ([judging.md](judging.md)). **Add both to page** places the sides on a page (`save page:NAME`), and **Fix paired messages on a page** places the paired messages with `--fixed`.

**Compare blind** first saves the rows (`save` with no name, giving a kept result), then reads the kept result with `--view compare --blind`: the sides are labelled `A`, `B` with identifying fields removed and tool names shown by family, and the view shows the seed. **Prefer A** and **Prefer B** on a dimension send `judge A --on DIMENSION --seed N` on the kept result; the daemon maps the label back. Once the comparison is recorded, **Reveal** reads the same kept result without `--blind`. Blind reading of anything but a kept result is not offered.

Lines: `polylogue review`, `polylogue annotation:01JB7Q2M@2 judge accept`, `polylogue 'session:codex:019d0f2a OR session:claude-code:7f3a2c10' read --view compare --align file`, `polylogue 'session:codex:019d0f2a OR session:claude-code:7f3a2c10' save`, `polylogue result:01JD7Q4M read --view compare --blind`, `polylogue result:01JD7Q4M judge A --on correctness --seed 7`.

### Judging passes `/judge`

A pass is a rubric applied by an agent over a kept population; this view is where the operator checks one ([judging.md](judging.md#rubrics-and-passes)). The rows behind it are annotations, so every panel is a line.

- **Passes:** each pass with its rubric and exact revision, author, population (`result:<id>`), coverage (rows with a value or an abstention against the kept rows), abstentions with their reasons, and refused values; **Withdraw the pass** withdraws its whole set in one step.
- **Sample to judge:** **Draw a sample** keeps a seeded, reproducible sample of the pass's values (`polylogue 'annotations where pass:outcomes-r2 | sample 20 seed 7' save`), and the view prints the seed so the same sample can be drawn again. The sample is then judged one card at a time: the card shows the value, its target with its last messages, and the rubric's field and allowed values, with the annotator's identity hidden until the verdict; **Accept**, **Reject** and **Defer** (`a`, `r`, `d`) judge the exact revision shown, as on [Review](#review-and-compare-review-compare). Pairs of rows from a sample go to [Compare blind](#review-and-compare-review-compare).
- **Agreement:** the pass against the operator's judgments (`| measure agreement`), with its denominator and the excluded, incompatible and conflicting counts; a selector chooses whose judgments it is measured against (`judged_by:`). A pass with no judged sample shows `not measured`, never a figure.
- **Rankings:** over comparisons on one dimension, `| measure ranking` gives win rates and a fitted order, each rank with its number of comparisons; the uncertain pairs are listed as **Compare these** buttons, each the ready `judge` line.
- **Forecasts' calibration:** forecasts grouped by stated probability against how often they came true, counting only forecasts written before their event resolved, with the count beside every bucket ([judging.md](judging.md#forecasts)).
- **Run the rubric again:** after the rubric is revised, the same kept population is offered for a new pass, with the line that starts an agent on it; the new pass's values derive from the new revision, and agreement is shown per revision side by side.

Lines: `polylogue 'annotations where pass:outcomes-r2 | sample 20 seed 7' save`, `polylogue 'annotations where pass:outcomes-r2 | measure agreement'`, `polylogue 'annotations where kind:comparison on:correctness | measure ranking'`, `polylogue annotation:01JB7Q2M judge accept`.

### Pages

The routes are `/pages` (the page list) and `/page/NAME` (one page). The model is [pages.md](pages.md); this section gives its web form. Every action on a page is a line: the view posts it and shows the committed records and the undo line from the reply, and the footer shows the CLI line and the MCP call of what is on screen:

```
polylogue page:flaky-ci read --view contents
polylogue_read {"arguments": {"selection": "page:flaky-ci", "steps": [{"verb": "read", "view": "contents"}]}}
```

**`/pages`** is the listing of the unit `pages` (`polylogue pages`): title with `page:NAME`, template, subject, slots, members, open items, last change (`finding from codex:019d0f2a · 14:02`) and whether it is watched. Filters are conditions of the listing (`pages where shows:REF` finds the pages that cite a row). Shipped pages (`home`, `pinned`, `cost`) and templates (names under `template/`) are listed apart, each marked built-in until revised. **New page** asks for a title, a name and a template and writes the page record (`polylogue page:flaky-ci note 'Why is CI flaky on Fridays?' --as page --template investigation`), with **Parameters** for a parametric template (`--param repo=morrow`) and **Subject** (`--about REF`).

**`/page/NAME`** is the page's `read`; its query string carries the `read` step's argument names (`/page/flaky-ci?view=contents`, `/page/flaky-ci?at=48390`, `/page/flaky-ci?view=changes&from=seen`).
- **Header:** template, `page:NAME`, created by and when, the subject above the sections (a question, a decision, a repository, a session or a file, linked); the switcher **linear · board · table · contents · history**; **Checkpoint**, **Watch**, **Brief**, **Working copy** and the copy menu. The page's own `layout` is the default; the switcher changes the presentation only, and **Make this the page's layout** writes it (`note TITLE --as page --layout board`).
- **Linear:** headings in tree order, each with **+** to add to that section; an empty section renders as one line `empty — add`. Slots render from the archive at read time, each with its mode (`current`, or `fixed at rev 3`) and target state:
  - a message, message range, session, block, span, annotation or alternate, in the reader's compact form;
  - a kept result or checkpoint: its rows marked `stored rows, not re-run`, with **Run again** as a separate act beside them;
  - a live query: `live · N rows`, rows that update while the page is open;
  - a figure: a measure query or saved figure as a chart and its table, as in [Analysis](#analysis-analysisq);
  - a file at a commit: the file at that revision in a code view, **History** (the commits that touched it, from the repository when it is on this machine, else from captured git data, otherwise disabled with the reason) and **Current**;
  - an `sx:` reference: resolved through Sinex in primary mode with a link to Sinex's reader; standalone it shows `unresolved: Sinex not configured` with the reference text;
  - a `url:`: the URL with its written title, never fetched;
  - another page: embedded and collapsed; a page not yet created shows as a red link, `page:not-yet · waiting`, and attaches when the page appears.
- **Board:** the same slots as cards placed by their `x`, `y`, `w` and `h`; dragging a card, or moving it with the arrow keys, writes one revision of that slot. Only this layout reads those values.
- **Table:** one row per slot with kind, target, mode, target state, author and last change.
- **Contents:** exactly `read --view contents` ([pages.md](pages.md#reading)), each line a link.
- **History:** a scrubber over the page's change positions (each position where a slot or a shown annotation changed, with author and time); moving it renders the page as it stood (`read --at POSITION`), and an approximate time across machines is labelled approximate. Choosing two positions shows the page diff between them (`read --view changes --from POSITION --until POSITION`); **Since I last looked** is `--from seen` ([reading.md](reading.md#ranges)). Reading a page here moves the reader's seen mark only when the web reports that the items were on screen.
- **Composer:** below each section and at the end of the page, a text field with a kind select (note, finding, question, decision, …) and the target section; sending writes the note and its slot in one request (`polylogue page:flaky-ci note 'weekday arithmetic used >= instead of >' --as finding --section Findings`), and the undo line withdraws both. **Add section** writes a heading (`note 'Risks' --as heading`).
- **Decision box** and **open items**, computed at read ([pages.md](pages.md#open-items-and-the-decision-box)): `no decision yet` with **Record decision**, which writes a `decision` about the subject placed in the Decision section (`polylogue annotation:01JA9X2B note 'strict weekday mode' --as decision --page flaky-ci --section Decision`). Open items each show "unresolved since" and link to their item; the shipped saved query `open` lists them across pages.
- **Members** (inspector): each alias with its resolved session, live state, role and last activity on the page (`@reviewer → codex:019d0f2a · working · reviewer · last on this page 09:50`); after a resume the alias shows the successor it followed. **Add member** writes `note reviewer --as member --page flaky-ci --role reviewer` on the chosen session ([pages.md](pages.md#members)).
- **Send composer:** a member or session, the text, and the section where the reply belongs (`polylogue @reviewer send 'fold the findings into the overview' --page flaky-ci --section Overview`); delivery is [session-control.md](session-control.md#send)'s, and the reply is placed in that section ([pages.md](pages.md#reply-placement)). The sent messages list each delivery with its state and its reply.
- **Staleness and circle markers:** dim marks in place, `written before 2 findings on this page` and `returned to the position of revision 2` ([pages.md](pages.md#staleness-and-circle-markers)).
- **Sidenotes and previews:** on wide screens notes about a slot sit in the margin beside it; hovering any reference shows a preview from the shared reference reader, and a hover never runs a model or changes anything.

**Moving and removing.** Dragging a slot in the linear or table layout, or `Alt-↑`/`Alt-↓`, writes one slot move (`polylogue annotation:01JC7Q2M save page:flaky-ci --section Findings --after annotation:01JD4K8P`). **Remove** withdraws the slot (`withdraw`), and what it showed stays.

**Add from anywhere.** `+` on any target or selection, in any view, opens a picker of page (or a new page from a template), section and mode (current or fixed) and writes `save page:NAME --section S`, with `--fixed` for fixed and `--live` to place a query's question as a live slot instead of its rows.

**Checkpoints, watches and the working copy.** **Checkpoint** writes `polylogue page:flaky-ci save` and links the `result:<id>`; **Watch** writes `polylogue page:flaky-ci save --watch`. **Working copy** offers `read --to flaky-ci.md` as a download, and **Import** takes the edited file back as `import` ([pages.md](pages.md#the-working-copy)): blocks that were refused (a moved base, an edited quote) are listed with their reasons, and the rest commit.

**Live.** A page view streams its slots and shown annotations; live query, figure and note slots update in place, and fixed slots never change. Agents' markers aimed at the page appear under their section as they are written, authored by the message's author. Updates never move the reader: a change above the viewport shows "N changes · show". Two current revisions of one slot from two machines show as a conflict with both.

Lines: `polylogue pages`, `polylogue page:flaky-ci read --view changes --from seen`, `polylogue page:flaky-ci read --at 2026-11-10`, `polylogue 'messages where "flaky" repo:morrow since:7d' save page:flaky-ci --section Evidence --fixed`, `polylogue annotation:01JC7Q2M withdraw`, `polylogue import flaky-ci.md`.

### Project story

A page from the `project-story` template, bound to a repository (`polylogue page:morrow/story note 'Morrow story' --as page --template project-story --param repo=morrow`) and read at `/page/morrow/story`. Its slots are live queries over `$repo` and `$since`, so it shows stored facts and checked notes only, with no generated prose: the repository's identity (its root commit); **Sessions per logical week** (week, Claude Code, Codex, web chats, models, cost), with delegated work beside it and never mixed in; **What happened** per session (date, link, title, cost, a finding with its check state, decisions, commits); **Outcomes** from sampled judging passes with the sampling stated, an unsampled pass labelled unsampled. Commits and pull requests come from captured git data, or from Sinex in primary mode. The `weekly-review`, `day-story` and `feature-story` templates render the same way over a week, a day or a feature.

Lines: `polylogue page:morrow/story read`, `polylogue page:morrow/story save`.

### Timeline, days, firehose

- **`/timeline`:** a year **calendar heatmap** (sessions, prompts, cost or tool errors per logical day, with an accessible table equivalent); below it a **scrubbable time axis** with rows per repository or harness, sessions as bars, tool errors as marks and notes as `◆`; drag selects a window, the wheel zooms hour ↔ day ↔ week; the table below lists the window's sessions and events. Timeline, table and measures share three bindings: the time window (the line's `since:` and `until:`), the focused item (opened at its neighbourhood with `--around`, never by loading every earlier message), and the file at a moment. The header names the zone and the logical day start ([operations.md](installing.md#configuration)).
- **`/day/DATE`:** the logical day's activity by time (each with its time source), notes and decisions, cost and usage, previous and next, **This day in other years** (a `month_day` condition), **Undated records**. In primary mode a wake-to-wake view of the day, read from Sinex, appears beside it.
- **File at a moment:** a file's state graded by what is recorded: known (after a successful edit), possible (after a failed one), unknown; never by re-running commands. It comes from the `changes` view's data.
- **`/firehose`:** the timeline reading new rows live with no window end, over AI-session records (messages, tool calls, notes, deliveries) at full detail. The cross-source firehose is Sinex's, a query read live over its confirmed stream; in primary mode `/firehose` links to it.

Lines: `polylogue 'messages since:365d | group by day(at) | count'`, `polylogue 'sessions since:2026-10-08 until:2026-10-09'`, `polylogue recent read --follow`.

### Cost

`/cost`, the shipped page `page:cost` ([pages.md](pages.md#templates-and-shipped-pages)) rendered in the workbench; revising that page changes this view.

- Title `Cost · since Oct 01` and its line; group by **day · model · origin · repository**.
- Cards: **Known cost** with its money kind and the exact value in a tooltip; **Unknown** (`4 sessions — item-only Codex rollouts and subscription chats`); **Price table** in use; **Most expensive**.
- **By day** with a bar column; unknown days hatched with the word `unknown`; the total `$5.86 + 4 sessions unknown`; sessions count logical sessions, and cost includes subagents.
- **By model** (input, output, cache read, cache write, cost; a `(not recorded)` row; the Codex observation subset priced apart); **By origin**; **By repository**.
- **Most expensive sessions:** own cost, `+ $X subagent`, tool errors; paged, never a fixed top N; sessions of unknown cost listed after, counted, never ranked as zero.
- **Subscription windows** (`| measure quota`): window, API-equivalent so far, reset, reported percentage; history with rate-limit events. **Billing-cycle outlook** (`| measure outlook`).

Every table has its own line in the footer. Dollars show two decimals in tables and the exact value in the tooltip. Unknown is never `$0` and never a zero-height bar, and a column never mixes money kinds ([measures.md](measures.md#cost-and-usage)).

Lines: `polylogue 'sessions since:2026-10-01 | measure usage'`, `polylogue 'sessions since:2026-10-01 | measure daily'`, `polylogue 'sessions since:30d | measure quota'`, `polylogue 'sessions | measure outlook'`.

### Analysis `/analysis?q=…`

Renders any `| measure` or `| agg` query, or a saved figure, as **a chart and its table**. The chart form follows the reply's shape: a series over time → bars per grain (lines for cumulative measures); a grouped count or sum → horizontal bars sorted by value; a distribution → a histogram or a percentile strip (nearest-rank percentiles); survival → a step curve; a comparison across groups → paired bars with denominators. Every chart shows its unit, denominator query, scope, the count of unknowns (hatched, never zero) and the product's validity note ([measures.md](measures.md#rules)); each bar or point opens the rows behind it. Charts are SVG rendered by the daemon from the typed reply, with the shell adding hover and selection; there is no charting script. **Save as figure** saves the shown query under a name with its denominator (`save NAME --denominator 'QUERY'`), so it can be cited.

Lines: `polylogue 'tool_calls since:30d | measure survival'`, `polylogue 'tool_calls where outcome:error since:30d | group by tool | count' save tool-errors --denominator 'tool_calls since:30d | group by tool | count'`.

### Pastes, attachments, topics

- **`/pastes`:** paste spans across the archive grouped by content hash, with repeat counts, first and last use, and sessions; **repeats only** narrows to spans pasted more than once. Backed by the `pastes` unit.
- **`/attachments`:** a grid and list of attachments with who supplied them, availability state, size and session; filters by media type and origin. Backed by the `attachments` unit; an attachment's bytes open through `read`.
- **`/topic?q=…`:** one reading of every message matching a query across sessions, oldest first, grouped by session with one line of context on each side (`read --around 1`).

Lines: `polylogue pastes`, `polylogue attachments`, `polylogue 'messages where chrono' read --around 1`.

### Status and Settings `/status`, `/settings`

**Status.** The figures, states and verdicts are [operations.md](operations.md#status)'s; this is their view.
- **Verdict** (`healthy`, `degraded`, `unknown`) with its reasons.
- **Daemon** (pid, since, up to date); **archive** (path as `[archive]`, sizes); **inputs** (stored, waiting, failed); sessions and messages with abandoned and inherited counts; a per-source table (location, inputs, sessions, last new, **live lag**); unrecognised record types; parse problems; markers not applied; notes with missing targets; conflicts; accepted claims whose check fails; settings whose current value someone other than the operator set; watches with their authors; vectors; primary-mode position and lag.
- Every figure carries its state (`measured`, `none_found`, `not_measured`, `unavailable`); an unmeasured figure is never shown as zero or healthy.
- **Catching up:** inputs waiting, rate, and an ETA from the measured rate.
- **Own use:** runs by surface and latency class, p50 and p95 per class against its [budget](latency.md#latency-classes), most-run queries and fields, reader opens by surface, time spent reading per session; local only, from the recorded runs and the daemon's counters.
- **Notices** with their channel and delivery state ([operations.md](operations.md#notices)).

The storage check `status --verify` runs from the command line only; the view shows the line.

**Settings** are annotations with history and author, listed by `set` with their values, scopes and who set them; opening one shows its allowed values and history (`set KEY`), and a change writes it (`set KEY VALUE`). This view edits those a person sets through the web:
- **View preferences:** density; show/hide defaults (tool output and pastes over N lines); single-key shortcuts on or off; theme (system, light, dark) and the palette source; rail and panel state per view.
- **The operator's facts and open choices** ([operations.md](installing.md#configuration)): logical day (read from Sinex in primary mode), subscription plan and price table, enabled alternates (`reading.alternates`), quiet hours and interrupt budget (Sinex's in primary mode), listen address and authentication, the optional away command.

Every change shows who made it and when. Lines: `polylogue status`, `polylogue set`, `polylogue set reading.alternates translation:pl`.

### Saved and kept `/saved`, `/r/RESULT`

**`/saved`** shows the rows of the shipped saved query `saved`: name, revision, text, the newest kept result and whether it is watched ([saved-queries.md](saved-queries.md)). Each opens `/search?q=query:NAME`, run at the time of reading. Actions: **Save again** (`query:NAME save NAME`, which stores today's rows as a new kept result and moves `result:NAME` to it), **Save and watch** (`query:NAME save NAME --watch`), **Open in analysis** and **Withdraw** (the saved query's annotation withdrawn; a shipped name comes back). `polylogue query:NAME read` gives the saved query's page: text, revisions, kept results, watch and dependents.

**`/r/RESULT`** shows a kept result: its stored rows, never re-run, with when they were kept, the question and the read position. **Run again** runs its question (`query:<hash>`) now, as a separate act, beside the stored rows.

Lines: `polylogue saved`, `polylogue query:failing read`, `polylogue result:01JD7Q4M read`, `polylogue query:3f9a1c2e`.

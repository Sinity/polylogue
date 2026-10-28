# Web workbench

The workbench is the web interface the daemon serves: reading, searching, measuring and annotating the archive, working with Pages, and following and answering live agent sessions. It is one more client of the [command table](interfaces.md#command-table) and adds no semantics. This page owns its principles, visual baselines, frame, Polylogue's key bindings, inspector, live behaviour, state matrix, access, routes and every view. Shared web mechanics (visual tokens, the shared key table, palette and copy mechanics, the frame, SSE fragments, windowed lists) belong to the common [`web-shell`](https://github.com/Sinity/polylogue-sinex-common/blob/master/crates/web-shell/README.md) crate. The browser extension's own interface is [extension.md](extension.md); rendering rules for messages and read views are [reading.md](reading.md); live-session control operations are [agents.md](agents.md#live-sessions); status figures are [operations.md](operations.md#status).

## Principles

- **A full designed interface, built from the designs.** Every view below is specified to the fidelity its board draws; the [visual baselines](#visual-baselines) are built faithfully, and views without a board are drawn before they are built. Ruling, 2026-05-02: "I really want full aesthetic and functional webui" (chatgpt:69f4b473-d540-83eb-b0ad-ad26c9ea32b1:a8aa12ec-7b2b-4678-a766-b0f5e5a5a312). Ruling, 2026-10-27: "wasn't it actually designed? what happened to that? why it's so basic?" (cc:ee52794c-8a9e-43b6-9fbc-9749c7e78e8f:3c3dd59f-4560-4fee-95d4-5c43b6c94ca5).
- **A shell over the same operations.** Every action is a command-table row reachable from the CLI and MCP; there is no web-only endpoint, and every view shows the CLI command that reproduces it.
- **Where the operator works with agents.** Live sessions are followed, answered, approved, steered, interrupted, resumed and pinged from the web ([Running](#running-running), [live-session control](#live-session-control-panel)). Ruling, 2026-06-18: "ability to actually real-time interact with all sessions through (our) webui" (journal 2026-06-18 21:56).
- **Live, never polling.** Views update from the change feed. Ruling, 2026-07-03: "lag from realtime of all sorts, especially in webui (things should feel ~instant)" (cc:30f5e650-7033-4423-9f91-2b66ebd1e6b2:e72e1c54-c67d-4f23-9649-c84390ffd442).
- **Every visible fact has provenance.** A chip, figure or line opens where it came from ([inspector](#inspector-and-provenance)).
- **Pages are the centre of work, and optional.** The model is [pages.md](pages.md); the views are [Pages](#pages-pages-ppage).
- **Durable state lives in the archive.** The browser holds only view state in progress ([what the browser holds](#what-the-browser-holds)).
- **Verified by running it.** Checks drive a real browser against the real daemon. Ruling, 2026-05-03: "I do not believe you can implemtn full aesthetics as they should be, full fucniaolity, interactivity - without ever runnign thigns." (cc-queued:d227b344-a871-4be1-bdf0-30c59ff1db7d@2026-05-03T18:16:47Z).
- **Stored text is shown as it is.** No view masks, flags or reveals spans, and there is no delete action, because Polylogue has no deletion ([architecture.md](architecture.md#deletion)).
- **No dependency on one desktop.** Desktop integration (opening an editor, a file manager, a terminal, the clipboard) goes through the desktop's own services; a desktop or window manager named on this page is an example of a deployment.

## Technology

Each view is HTML rendered by the daemon with `maud` from the same typed answer the CLI renders as text, so a reloaded row and a live row come from one function. One TypeScript shell from `web-shell` handles keys, the palette, copying, SSE fragments, windowed lists for long readers, and pointer handling for the timeline and the board layout. There is no single-page application, no WebAssembly and no third-party runtime script, and every route carries a strict content security policy (`script-src 'self'`, `style-src 'self'`). Web comes first and there is no native application: an installed app window gets the desktop's notification, shortcut and window services, and the typed answers keep a native or terminal front end a thin later addition.

## Visual baselines

The boards and mockups in [`web/`](web/) are the visual baselines, drawn on the synthetic Morrow archive (clock 2026-10-08 12:00 Europe/Warsaw; the live set at 2026-10-09 15:00). The canvas boards and mockups use the joint visual tokens (IBM Plex Sans and Mono, light and dark). Where a board and this page disagree, this page wins and the board is redrawn. A visual baseline match is a review beside the named board, not a pixel test.

### Lineage

Four generations of design feed the workbench. Each later one refines the earlier ones; none replaces an earlier one whole. An element of an earlier generation still binds unless a later board draws the same thing differently or this page says otherwise, and the tables below record which board governs each element.

| Generation | Board | What it contributes that still binds |
|---|---|---|
| **MK2 "Cockpit"** (Claude Design, 2026-05-02) | [boards/mk2/](web/boards/mk2/index.html) | the cockpit layout (facet rail, result list, reader, side panel); provenance on every fact; the status strip; one keyboard model across web, CLI picker and palette; the empty, loading, degraded and failure states with a command as the way out; the daemon counting its own use; context bundles from selected messages; the data-flow invariants |
| **MK3 / MK3.1 "archive workbench"** (Claude Design, 2026-05-23) | [boards/mk3/](web/boards/mk3/index.html) | the information architecture; the search cockpit with score components and bulk selection; paste and attachment rendering; the message card and its copy menu; fold rules and match windows for huge sessions; the inspector in five selection contexts; the topology view; continue, fork and bundle flows; the tabs, stack, compare and timeline workspace modes; the palette with disabled reasons; the full state matrix; settings |
| **MK4 concept** (2026-05-24, prose, no board) | — | continue elsewhere from any message; the context composer with its provenance manifest; observed lineage, never created by a UI act; every action shown with its availability and the reason when it is unavailable. These are written into the views below |
| **The board canvas** (2026-10-12) and the static mockups (2026-10-11, 10-24) | [boards/](web/boards/), [mockups/](web/mockups/) | the current look (the joint tokens); the views they draw, on the Morrow archive. Where they draw a view an earlier board also drew, they govern |

The visual style of MK2 (graphite and slate, dark only, Inter and JetBrains Mono) and of MK3.1 (the Sinex ANSI-16 palette) is replaced by the joint tokens; their layouts, panels, interactions and keyboard model are not.

### MK3.1 artboards

| Artboard | Governs |
|---|---|
| [01 Home](web/boards/mk3/index.html#a01) | Home's operational panels: sources with their state, quick filters, live sessions, saved queries, recent bundles, the archive digest and derived freshness, the last-hour feed. Card layout and grouped results: the canvas home board |
| [02 Search cockpit](web/boards/mk3/index.html#a02) | hit rows with score components, facets scoped to the query, the selection bar with bulk actions and their disabled reasons, the query shown as its CLI command. Summary, defaults line and Refine: the search mockup |
| [03 Reader](web/boards/mk3/index.html#a03), [05 Message card](web/boards/mk3/index.html#a05) | paste spans inside a prompt, attachment cards, folded tool output with a match window, the copy menu and its variants, the inspector following the selection. Message cards, rails and lineage: the canvas reader board |
| [04 Huge session](web/boards/mk3/index.html#a04) | virtualized reading with windows above and below, scroll restore to an anchor, fold policy, search matches always visible, the minimap. The outline lists prompts, failures, compactions and branch points rather than phases |
| [06 Pastes](web/boards/mk3/index.html#a06), [07 Attachments](web/boards/mk3/index.html#a07) | how each paste kind and each attachment state renders; the pastes and attachments views |
| [08 Inspector](web/boards/mk3/index.html#a08) | the inspector for each selection context |
| [09 Topology](web/boards/mk3/index.html#a09) | the lineage view over the closed set of session links ([sessions.md](sessions.md#session-links)), with unresolved parents shown as waiting |
| [10 Flows](web/boards/mk3/index.html#a10) | continue from a message, fork from a message, and a bundle from a selection, each explicit and previewed |
| [11 Tabs](web/boards/mk3/index.html#a11), [12 Stack](web/boards/mk3/index.html#a12), [13 Compare](web/boards/mk3/index.html#a13), [14 Timeline](web/boards/mk3/index.html#a14) | the workspace modes, compare and the timeline |
| [15 User state](web/boards/mk3/index.html#a15) | marks, notes, saved queries, bundles and workspaces as the notes list shows them; marks are tags ([assertions.md](assertions.md#kinds)) |
| [16 Palette](web/boards/mk3/index.html#a16) | the palette scoped by selection, with disabled reasons, and the keyboard help |
| [17 State matrix](web/boards/mk3/index.html#a17) | populated, empty, partial, degraded and failed for every view ([state matrix](#state-matrix)) |
| [18 Settings](web/boards/mk3/index.html#a18) | density, folding, keyboard, sources and access settings |

Parts of MK3.1 that do not bind: the redaction policy, redacted paths and "copy requires confirm" (stored text is shown as it is, with nothing masked); the "needs attention" repair suggestion (`index rebuild`), since derived data converges by itself; the "repaired" topology edge (links come only from explicit markers); the "insights" label (derived data has no umbrella name, [measures.md](measures.md#rules)); and the phase outline of 04.

### MK2 elements

| Element | Governs now |
|---|---|
| three panes (facet rail, result list, reader) with a side panel | MK2, for the wide search layout with a reading pane; MK3.1 02 for the hit rows |
| facets `provider`, `repo`, `has: tool_use · thinking · paste`, saved queries in the facet rail | MK2; scoped counts from MK3.1 02 |
| side panel tabs messages · raw · provenance | MK2 for the tab set (Outline · Raw · Provenance); MK3.1 08 for the panels inside |
| provenance panel: arrival route, source path, byte cursor, hash, derived facts | MK2. The privacy line reads "local archive only"; derived facts never include guessed phases |
| reader header actions `S` star, `P` pin, `A` annotate, `R` bundle, with the equivalent CLI command beside them | MK2; star and pin are tags |
| status strip: daemon uptime, live lag per source, capture connection, index pending, key hints | MK2 |
| `⌘K` and `:` palette, `g a` for annotations | MK2 for the keys; MK3.1 16 for the palette |
| one keymap across web, CLI picker and palette ("Same verbs, same letters") | MK2 keymap, as the [keys](#keys) table. A delete key does not exist (Polylogue has no deletion) |
| empty, loading, degraded and failure states on every surface; calm persistent status, never a blocking modal; the way out is always a command the user could run | MK2 for the rule; MK3.1 17 for the matrix. The ways out are ordinary commands; there are no repair commands ([architecture.md](architecture.md#principles)) |
| the daemon counting its own use, local only, readable from every surface: query runs, reader opens by surface, time spent per session, per-component lag | MK2 telemetry, as Status's own-use panel. The counters live with the request log in `archive.db`, and nothing is purged |
| status panels: archive counts, sources with lag, capture, recent, diagnostics | MK2's terminal cockpit, as the [Status](#status-status) view and the status strip. No terminal UI is built now, and diagnostics suggest ordinary commands only |
| fuzzy selector: descriptor-driven, preview pane, multi-select, print or open the choice | MK2, for the CLI picker ([interfaces.md](interfaces.md#picker)) |
| CLI: bare overview with source bars, completion popup with counts, opening the latest session in the web reader | MK2, for the CLI ([interfaces.md](interfaces.md#completion)) |
| data-flow invariants: any surface renders anything by reference alone; nothing leaves the machine unless the user writes it out or configures a listen address | MK2, as the laws `any-surface-by-reference` and `nothing-leaves-unasked` |
| bundles from multi-select, message references without copied payloads | MK2; MK3.1 10 for the flow. The bundle's manifest carries references and content hashes |

The MK2 board is drawn for the Python predecessor, so its command and module names (`polylogued`, `ops doctor`, `--fix` targets, Python file names) are the predecessor's; the commands are those in [interfaces.md](interfaces.md#commands), and there are no repair commands. Its roadmap and inventory artboards show the predecessor's plan, not views. MK2's MCP role gate (read, write, admin) and its MCP server with direct archive access do not apply: agents get every power ([architecture.md](architecture.md#principles)), and the daemon is the only process on the archive.

### Views

| View | Baseline |
|---|---|
| [Home](#home-) | [boards/home.html](web/boards/home.html) (cards and grouped search results); [mockups/index.html](web/mockups/index.html) (recent sessions with links column, recent notes, sources) |
| [Search](#search-searchq) | [mockups/search.html](web/mockups/search.html) (summary, defaults line, Refine, hits grouped by session); grouped results also on [boards/home.html](web/boards/home.html) |
| [Session reader](#session-reader-ssession) | [boards/reader.html](web/boards/reader.html) (message cards, collapsed tool lines, delegation card, lineage, notes and views rails); [mockups/session.html](web/mockups/session.html) (toggles, compaction, branch and inherited-prefix lines) |
| [Running](#running-running) | [boards/running.html](web/boards/running.html) |
| [Session report](#session-report-ssessionreport) | [boards/report.html](web/boards/report.html) |
| [Share an excerpt](#share-an-excerpt-ssessionshare) | [boards/share.html](web/boards/share.html) |
| [Claims and judging](#notes-claims-judging-triage) | [boards/claims.html](web/boards/claims.html) |
| [Project story](#project-story-pstory-repo) | [boards/story.html](web/boards/story.html) |
| [Page](#pages-pages-ppage) | [boards/page.html](web/boards/page.html) (sections, slots, sessions on the page); [mockups/page.html](web/mockups/page.html) (members, ping composer, decision box, kept result, live query) |
| [Page list](#pages-pages-ppage) | [boards/pages.html](web/boards/pages.html) |
| [Cost](#cost-cost) | [mockups/cost.html](web/mockups/cost.html), chosen where it differs from [boards/cost.html](web/boards/cost.html) (cost by origin, most expensive sessions, the Codex observation subset priced apart) |
| Extension popup | [boards/extension-popup.html](web/boards/extension-popup.html) |
| [Workspaces](#workspaces-wmode), [Compare](#compare-compare), [Timeline](#timeline-days-firehose) | MK3.1 11–14 |
| [Pastes and attachments](#pastes-attachments-topics) | MK3.1 06–07 |
| [Notes](#notes-claims-judging-triage) | MK3.1 15 |
| [Status](#status-status) | MK2 status panels and own-use counters |
| [Settings](#settings-settings) | MK3.1 18 |

Parts of the canvas boards and mockups that do not bind, because the text decides otherwise:
- The navigation on the boards lists Home, Running, Cost and Pages; the full navigation is in the [frame](#frame).
- The share board flags paths under the home directory and lists `has:secret` blocks; sharing writes stored text unchanged with nothing flagged ([share](#share-an-excerpt-ssessionshare)).
- The Running board's footer names a `polylogue running` command; the rows are the shipped saved query `query:running`.
- Tool durations show on shell and subagent calls only; a duration on a file read in a mockup does not bind.
- The mockups' "export JSON" link is the `read` row with `--format json`; there is no export verb ([reading.md](reading.md#writing-out)).

## Views without a board

Views that no board draws are drawn in Claude Design before they are built, with at least two variants wherever layout is a real choice; the operator picks one, and the chosen board becomes that view's baseline. Ruling, 2026-10-11: "I wnat you to mkae use of claude design all over the specs etc., for all UX, workbench design, CLI etc." (cc:56c7f026-fd50-4480-b484-72f1299a534c:f8272f95-186a-4719-8145-42140988f004).

The views without a board: the heatmap, the firehose, the note page, triage, analysis, the live-session control panel, the context composer, the flow panel, and the extension's in-page UI. Their behaviour below binds; their layout does not until the board exists.

## Frame

```
┌ header ─────────────────────────────────────────────────────────────────────────────┐
│ Polylogue  Home  Running(2·1 waiting)  Notes  Pages  Timeline  Cost  Status [query…] │
├─ left rail (optional) ─┬─ main ───────────────────────────────┬─ inspector ──────────┤
│ outline / facets /     │ view content                          │ facts · lineage ·    │
│ toc / filters          │                                       │ notes · provenance … │
├────────────────────────┴───────────────────────────────────────┴──────────────────────┤
│ ● daemon 4h 12m · claude-code 0.4 s · codex 1.1 s · extension connected · index 1,210 pending · : palette  / query  ? keys │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

- **Header:** wordmark; navigation Home · Running · Notes · Pages · Timeline · Cost · Status; the persistent query field with completion on every page. The Running item carries `running · waiting` counts, so "waiting on you" is visible from every page.
- **Status strip:** daemon state and uptime; **live lag per source** (time from the newest byte a live source wrote to its commit in the archive, per origin; `—` when the source is idle; a figure state when not measured); the extension's connection (from the native host's last contact); index work pending; in primary mode, the Sinex position lag. Each item links to its Status section. The strip is the daemon state indicator: `live`, `catching up · N inputs waiting`, `disconnected · retrying`, `stopped (start: <service command>)`.
- **Main column** up to 1,200 px; transcript text 75–90 characters per line.
- **Inspector** (right rail, 280–320 px): the page's own facts by default, switching to the focused target with a "back to session" link.
- **Left rail** only where it earns its width: session outline, search facets, a page's table of contents, timeline filters.
- **Narrow widths** (under 900 px): rails become disclosures above the main column; tables scroll inside their box; there is no horizontal page scroll. Every view stays usable at 390 px.
- **Footer line** on every page: the exact CLI command for what is shown, with **Copy as CLI command**; a page with several tables has one command per table. The command is computed from the request arguments that produced the page.
- **Result handle:** every query-backed view shows its `query:<hash>` handle (and `result:<id>` when kept) beside the footer command ([query.md](query.md#query-objects)).

### Visual use

The tokens, marks of meaning, value states and chips are `web-shell`'s. Polylogue's use of them:
- **Grounds:** the operator's own messages and nested blocks sit on `--surface-sunk`.
- **Accent:** Polylogue blue `--accent-polylogue` for links, focus ring and primary buttons; it never carries a state.
- **Palette from the desktop theme:** on by default when a palette file exists (pywal; the `theme.palette` setting names another file). Grounds and ink derive from it while semantic colours and contrast checks stay; a palette that fails the contrast checks falls back to the designed one, and the theme setting says why. Direction, 2026-06-18: "rather than hardcoding some colorscheme, should be responsive (to stuff like pywal)" (journal 2026-06-18 21:48).
- **Values:** the value-state table of [reading.md](reading.md); missing values in charts use the missing-value hatch, never a zero-height bar.
- **Notes:** `◆` plus the kind word, `--note` ink on `--note-bg`.
- **Check states:** `holds`, `fails`, `unknown`, `unchecked` badges; `unchecked` outlined.
- **Live states:** the six states of [agents.md](agents.md#live-sessions), always as the word: `waiting` on `--pending-bg`, `working` on `--observed-bg`, `idle` and `unknown` on `--surface-sunk`, `stale` in `--failure-ink`, `ended` plain.
- **Authorship:** typed text plain; a pasted span shaded with one evidence line; unknown authorship hatched. Authored records show their author (`operator`, `agent:<session>` as a link, `agent:<origin>`, `script:<unit>`, `unattributed` shown as "author unknown") and `via` ([assertions.md](assertions.md#authorship)).
- **Density:** comfortable (default), compact, dense; a setting.
- **Motion:** none except the live tick and progress; off under `prefers-reduced-motion`.
- **Chips:** text first, ordered origin or harness · live state · repository, working directory, branch · lineage state · counts · cost · notes and tags; at most one row in a header.

## Keys

Navigation, selection, the palette, help and the copy menu keys come from the shared key table in `web-shell`, which the CLI picker shares ([interfaces.md](interfaces.md#picker)). Polylogue binds these as well; single-character keys can be switched off (a setting), are ignored in inputs and while text is selected, and move real DOM focus, which live updates never steal.

| Key | Action | Scope |
|---|---|---|
| `g h` `g r` `g a` `g p` `g t` `g c` `g s` | go to Home, Running, Notes, Pages, Timeline, Cost, Status | everywhere |
| `+` | add the focused target or selection to a page or a context bundle (picker of page, section, following or fixed) | everywhere |
| `y c` | copy the page's CLI command | everywhere |
| `o` | open in the editor at the session's working directory, or a tool line's file at its line | reader |
| `a` / `A` | add a note / a decision on the focused target | reader, search, notes |
| `t` / `T` | add / remove a tag | reader, lists |
| `s` / `P` | star / pin mark on the focused session or message | reader, lists |
| `n` / `N` | next / previous match of the active find | reader |
| `m` `r` `p` | inspector tab Outline / Raw / Provenance | reader |
| `v` then a letter | read view: `t`ranscript, `d`ialogue, `s`ummary, `c`ontext, `l`inks, `g` changes, `v` live, `r`aw; `f` opens the flow panel | reader |
| `f` / `F` | fold or unfold the focused block / all blocks | reader |
| `z e` `z t` `z h` `z i` `z b` | expand all tools, thinking in full, harness messages, include inherited, show branches | reader |
| `c` | compare: with one target opens the chooser, with two opens Compare | lists, reader |
| `l` | follow a live session on or off | reader |
| `i` | focus the live-session composer | reader, Running |
| `1` `2` `3` | outcome `solved` / `partial` / `abandoned` | reader, judge |

Inside the judge item card, `a`, `r` and `d` mean accept, reject and defer, and the card shows them on its buttons. No other key changes meaning by view.

**Palette** (`:` or `Ctrl-K`): every action available for the current selection, generated from the command table's rows plus the view's navigation. Each entry shows its key, its CLI form and, when disabled, the reason from the operation's availability answer. It never holds an action the command table lacks.

**Copy menu** (`y`), per target:

| Target | Default | Other forms |
|---|---|---|
| view | CLI command | URL; JSON request (`/api/v1`) |
| session | its reference | title and reference; Markdown of the current view; resume command; provider URL when the origin records one |
| message | Markdown with header (`## 10:56 you`) | plain text; typed-only (when paste spans are known); reference; permalink; **as context** |
| tool call | the command or target | input JSON; output as stored; reference |
| assertion | body | `assertion:<root>@<rev>`; CLI command to revise |
| result row | its reference | the row as TSV |
| selection | Markdown | **as context** with manifest |

When the browser cannot carry the bytes (exact bytes, non-text types), the menu offers **Put on the desktop clipboard**, a daemon row that writes them through the desktop's clipboard service.

## Inspector and provenance

The inspector shows, for the page or the focused target:
- **Facts:** ID, models, tools with counts and errors, tokens, cost with its money kind.
- **Lineage:** the compact tree of linked sessions.
- **Notes:** every note on the target, newest first, each with author, `via`, judgments and check state; **Add a note**.
- **Views:** links to the read views, the report and share.
- **Referenced by:** pages, slots, kept results and notes that point here.
- **Tabs Outline · Raw · Provenance:**
  - *Outline:* the session outline.
  - *Raw:* the stored input records behind the focused messages, as inert text, with each record's outcome (used, skipped with reason, unreadable).
  - *Provenance:* the origin and arrival route (`live`, `spool`, `import`); the source location (the provider-side path, never the archive directory); the input, revision and chain; the record's byte range in its chain and, for a live input, the committed length against the file's length; the blob hash; the parser and its reading version; the observations (machine, when found); which of the row's stored versions is shown; for derived values, the derivation, its rule version and the rows it read; for authored records, author, `via`, revision and relations; in primary mode, the Sinex material and position. Each line links where a target exists. The CLI form is `REF trace`.
- **Every chip and figure opens its provenance:** a cost chip opens the usage records and price entry it came from; a live-state chip opens its evidence; a count opens the query that produced it.

## Live behaviour

- **Transport:** one server-sent-events stream per visible tab (the `stream` row). The event ID is the change feed's position; events are sent after commit and carry rendered fragments (`append`, `replace`) or changed references (`ref`), never whole sessions. A reconnect resumes from `Last-Event-ID`; an unknown position or a new index sends `reload`. Pages declare what they follow; the server filters by reference and refuses by name a subscription it cannot evaluate.
- **Clock-driven changes:** a live state carries `until`, and the view recomputes at `until` without an event.
- **Never moves the reader:** lists hold reordering while a row has focus and show "N updates · apply"; the reader appends only while following and at the end, and otherwise shows "N new messages ↓"; an update never removes text selection or focus.
- **Hidden tabs** close the stream and refresh on return.
- **Superseded requests:** a newer selection aborts the older request, and a stale reply never overwrites a newer selection.

## Writes, availability and errors

- **Writes** are POSTs of command-table requests with a client `request_id` reused on retry. The view shows the committed row the daemon returns, and a failed write keeps the editor text. Every write from the web is authored `operator` with `via: web`, computed by the daemon; the web sends no author.
- **Availability:** every action carries `enabled`, `disabled (reason)`, `partial (reason)`, `loading` or `unavailable (reason)`, answered by the operation. A disabled action stays visible with its reason ("harness cannot interrupt this session", "needs paste evidence", "Sinex not configured").
- **Errors** say what failed, what still works and the next step, which is a command the operator could run. A failed optional panel shows `unavailable: <reason>`, never zero or empty.

## State matrix

Every view renders four states as a persistent line in the view, never a modal, each with a recovery that is a command.

| View | Empty | Loading | Degraded | Failure |
|---|---|---|---|---|
| Home | "No sessions yet" with the sources Polylogue watches and `polylogue import PATH` | cards `pending` until the status answer | "catching up · N inputs waiting · ETA" from the measured rate | daemon unreachable: the page says so and names the service command |
| Running | "Nothing running" with the last ended session | rows `pending` | a source's live lag above its normal range, named | no hooks installed for a harness: rows `unknown (no hooks)` with `polylogue install-agent HARNESS` |
| Search | "0 rows" with the defaults that excluded rows and the broader query | results stream in; the coverage line says `partial` | `up_to_date: false` banner naming what is behind | a refused query: kind, the submitted text, the corrected query |
| Reader | "No messages" (only harness records) with the raw view link | first-page skeleton | session behind its derived products: values computed on the spot or `pending` | not found: the reference and the nearest match by prefix |
| Report, Cost, Analysis | "No priced sessions in this range" | figures `pending` | `≥ X (N unknown)`, sessions behind counted | measure refused: reason and a narrower query |
| Pages | "Empty page" with templates | slots `pending` independently | a slot whose target is gone from the index: `unresolved reference` with the reference text | write conflict: two heads shown |
| Timeline, Day | "No activity on this day" | lanes `pending` | undated records listed apart | — |
| Status | — | figures `not_measured` | each degraded item with its reason | daemon down (only the strip can say it) |

In primary mode every view adds the Sinex state: `Sinex blocked` (writes waiting, reads from the local index) as a degraded line. The extension's states are in [extension.md](extension.md#states).

## What the browser holds

Only view state in progress: focus, open disclosures within a page view, follow on or off, a selection being made, and unsent editor text (in page memory, kept across live updates). Everything a person expects to persist (density, fold defaults, rail and panel state per view, the key switch, theme, the last layout of a page, workspace modes) is a `setting` record in the archive, authored and with history. Browser storage is not used for product state.

## Access

- The listen address and authentication mode are deployment configuration ([operations.md](operations.md#configuration)): by default a loopback bind with a peer-uid check; any other address needs a token file, and a non-loopback address without one refuses to start. Remote or phone access is that configuration; every view works at 390 px.
- `polylogue open` hands the token over as an `HttpOnly; SameSite=Strict` cookie through a URL fragment exchanged at `/auth`; the cookie's name distinguishes archives, because cookies do not distinguish ports. Credentials never appear in query strings or change-feed positions.
- Every route checks the Host header, the peer uid where it applies, and authentication. Writes are POST only and require an allowed Origin; command POSTs are `application/json`. There is no CORS.
- Every response carries `no-store`, `no-referrer`, `nosniff` and the content security policy. Archived markup stays inert, and reading content never fetches remote previews.
- IDs are percent-encoded once. The archive directory's path is never sent over HTTP; text shows it as `[archive]`.

## Routes

`/`, `/running` (`/live` redirects), `/search`, `/s/SESSION` (and `/report`, `/share`), `/d/DOCUMENT`, `/m/MESSAGE` (redirects to the reader anchor), `/n/NOTE`, `/notes`, `/claims`, `/judge`, `/triage`, `/pages`, `/p/PAGE`, `/w/MODE`, `/compare`, `/timeline`, `/day/DATE`, `/firehose`, `/cost`, `/analysis`, `/r/RESULT`, `/pastes`, `/attachments`, `/topic`, `/status`, `/settings`, `/stream`, and `/api/v1/…`, one route per command-table row. Each route is a view of one or more rows, and the JSON API takes the same request as the daemon socket.

## Views

### Home `/`

The orientation of bare `polylogue`, as a page:
1. The query field, focused on load; **Search**; **Copy as CLI command**.
2. Cards: **Sessions** (count, top-level count, last activity); **Last 14 days** (prompts, tool calls, errors); **Known cost** (with its money kind and the unknown count); **Waiting on you · running** (link to Running); **Quota window** (current subscription window, API-equivalent so far, reset time, reported percentage); **Notices**, only when there is one.
3. **Waiting on you**, when not empty: one row per waiting session with what it waits for and the [live-session controls](#live-session-control-panel).
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
- Order: waiting, working, idle, unknown. **Stopped** lists `stale` sessions with their evidence (`process gone at 13:40, no end event`).
- Footer: the state legend and `Same rows as polylogue query:running`.

**Work tree and context flow.** A row expands into its logical session's physical sessions (subagents, forks, continuations) with what flowed on each edge: subagent instruction and final answer; brief, takeover and ping deliveries (prepared, delivered, used; size; references; what was left out); compaction boundaries with what they evicted. The same graph is the reader's flow panel. Ruling, 2026-07-03: "how does context flow between them, what do subagents return" (cc:30f5e650-7033-4423-9f91-2b66ebd1e6b2:e72e1c54-c67d-4f23-9649-c84390ffd442).

A row opens the reader at Latest with following on. Each live row carries the [live-session controls](#live-session-control-panel) inline, and a waiting row's pending permission shows **Approve** and **Deny** directly. **Open in terminal** appears when the harness's terminal is known through the desktop's services; otherwise the resume command is offered to copy. `x` selects rows for a stack. The view updates in place, recomputes at each state's `until`, reorders only when no row has focus, and announces "1 session now waiting" to screen readers.

### Search `/search?q=…`

- **Left rail:** facets for the unit, with whole-population counts (sessions: origin, repository, harness, model, date, live state, tag, outcome, `has:` tool_use · thinking · paste; messages add author; tool calls add tool family and outcome), with **saved queries ★** below them. Editing a facet edits the parsed condition, keeping the rest of the expression.
- **Summary line** (`3 sessions · 8 matching messages for chrono`); the **read as** line, the query read back in words; the **defaults line** (`Excluded by default: inherited copies (#4 repeats #2), messages on abandoned branches (1 hit in chatgpt:68f0a1b2). include them · explain this query`).
- **Session and message results** group by session: title; `id · date · N msgs · cost · tags`; hits `HH:MM · author — snippet` with the match marked; subagent hits as `↳ subagent claude-code:… (Explore)` under the parent; notes on a hit inline; `N more hits in this session`; row numbers `1`, `.1`, `.2` matching the CLI's `@1.3` ([interfaces.md](interfaces.md#addressing)).
- Other units use the shared results table with reference columns linked.
- **Also in tool output: N hits · show them** when the defaults exclude tool output.
- **Refine:** deterministic suggestions from the parsed query (narrower origin, author, a grouped count, related notes), each a link; never a model call.
- **Coverage line:** `exact`, or `ranked (semantic): candidates exact, order by similarity` ([search.md](search.md#semantic-ranking)).
- **Wide layout** (1,600 px and wider): results, a reading pane with the focused result in the reader's compact form, and the inspector. `Enter` opens the full reader.

The query field completes from the completion catalog ([interfaces.md](interfaces.md#completion)), with result counts as preview. `x` selects hits or sessions; the selection bar offers **Annotate**, **Tag**, **Add to page**, **Add to bundle**, **Keep result**, **Compare** (two sessions), **Open as stack** and **Copy as context**. **Save query** names the query's hash, **Keep result** keeps its rows, **Watch** creates a signal rule ([query.md](query.md#watches)). **Annotate all matches** previews the population, then writes one note per row. The field is the query line, so a bare word follows the [grammar](interfaces.md#grammar): it is refused with the `find WORDS` form shown as the correction. Results are a snapshot with a read position; a banner "N new matches since this page was read · refresh" appears when the stream reports matching changes.

Rows: `query` (with `why`), `explain`, `complete`, `save`, `keep`, `watch`, `assert` over a query, facet counts, `refine`.

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
| Pastes | typed text plain; a pasted span shaded with its evidence line; unknown authorship hatched; a paste over 20 lines folds as `paste · 214 lines · copy · copy typed-only` |
| Attachments | images inline; other files `▣ name (media type, size)` with **Download** and **Reveal in file manager** (daemon rows using the desktop's services); SVG and HTML as links, never inline; the availability state and who supplied it ([sessions.md](sessions.md#attachments)) |

**Tool families.** Each recognised tool family has its own expanded body, and every family still collapses to the one line above. A call's family comes from the tool-family table of the derived data ([measures.md](measures.md#catalog)); an unrecognised tool uses the generic body. Ruling, 2026-07-03: "custom rendering of various detected standard toolcalls … Think ambitiously" (cc:30f5e650-7033-4423-9f91-2b66ebd1e6b2:060b830b-16dc-4ac6-b463-7dd5deea1830).

| Family | Expanded body |
|---|---|
| Shell (`Bash`, `exec_command`, …) | the command highlighted, working directory, exit code, duration, stdout and stderr each with first and last 100 lines and `… N lines skipped (open block)`; when the command ran through a declared wrapper (a job or task runner named in configuration), the wrapped command, the wrapper's job ID as the transcript shows it, and the exit status. This command and exit status are the verification evidence |
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

**Read views.** The views and their names are [reading.md](reading.md#views)'s; on the web, *summary* is a definition list, *context* shows what the model had at a selected message (parents → boundaries → tail), *links* is the tree of linked sessions with own and tree cost, *changes* lists files edited (graded known, possible, unknown), commands with outcomes, commits and pull requests, *live* is this session's Running row with follow and controls, and *raw* is the stored input records, inert. The **flow panel** renders the `delegations` and `deliveries` units as the work tree of [Running](#running-running), and the **report** composes summary, changes, the session's tool calls and its claims; neither is a read view, because both are compositions of query units.

**Reading pages.** The reader reads a session in pages through the `read` row with a budget ([reading.md](reading.md#budgets)); the web default is 8,192 tokens. Paging links are First · Previous · Next · Latest; anchors are `/s/ID?message=MID`; changing the view starts a new read at the page's message anchor. The content cursor, the outline cursor and the links cursor are independent URL parameters, and the failure of an optional panel is shown beside the content. A stale cursor offers "Read changed · refresh from this message"; pages are never joined across a stale cursor. Long sessions keep a window of the DOM and fetch by anchor when scrolling past it.

**Interactions.**
- **Action rail** per message on hover and focus: **Copy**, **Annotate**, **Decision**, **Tag**, **Star**, **Add to page or bundle**, **Compare from here**, **Continue from here**, **Raw**, **Provenance**, **Permalink**.
- **Range selection** (shift-click two headers, or `x` on each): **Share excerpt**, **Add range to page**, **Copy range as Markdown**, **Copy as context**.
- **Find in session:** a field in the view bar runs a message query scoped to the session, jumps by anchor and steps with `n`/`N`; matches inside folded blocks open a match window. The browser's own find sees only loaded pages.
- **Outcome:** `1`/`2`/`3` and a select in the facts.
- **Open in editor** (`o`): the session's working directory, or a tool line's file at its line, through the desktop's services; disabled with a reason when the path does not exist on this machine.

**Continue elsewhere,** on the session and on every message:
- **Resume this session** in its own harness (for a live session, the [controls](#live-session-control-panel) instead): runs the harness's resume command headlessly with an optional prompt, or opens it interactively in the operator's terminal; the exact command is shown before it runs ([agents.md](agents.md)).
- **Continue from here in another harness:** a new session seeded with the effective context at that message (`context --at MESSAGE`) and the message text, built by the pack builder and recorded as a delivery. It is never labelled "resume" and creates no lineage link by itself: a `reference` link appears only when the new session's transcript shows the delivery.
- **Copy as prompt:** the same pack, copied. **Takeover brief:** the takeover pack for this session, copied or sent ([agents.md](agents.md#takeover)).

**Context bundles.**
- A bundle is a selection of targets (messages, ranges, paste spans, attachments, lineage facts, notes, kept results, query results) held as a page with the `bundle` template, so saving it is saving a page, and it stores references, never copies.
- **Copy as context** on a selection, a bundle or a page asks the pack builder for the rendered pack in Markdown, plain text or JSON with its **manifest**: every included reference with its `seen_hash`, what was included in full, what can be expanded, and what was left out and why. The pack carries its delivery ID in its header line.
- Copying records a delivery with state `prepared` and recipient `clipboard`; when a session's text later carries that delivery ID, delivery use records `used` with that message ([agents.md](agents.md#deliveries)). Sending the bundle to a session (ping or continue) records a `delivered` delivery.
- While a bundle is open, the inspector's **Bundle** panel is its composer: items in order, drag or `+` to add, `Alt-↑/↓` to reorder, the manifest preview, size in tokens, and the target (clipboard, a harness, a session by address).

**Variants.** When the session has variants ([assertions.md](assertions.md#variants)), the header shows `≈ en` and a variant select; the enabled-variants setting chooses the default, and every rendered variant carries a visible mark. **Side by side** aligns original and variant message by message through the variant's mapping. Notes on a variant are the translated notes, each linking its original.

**Related sessions.** An inspector panel lists sessions ranked by `related:REF`; while that product is unavailable (no recall figure, or model calls off), the panel shows `unavailable: <reason>`.

Rows: `read` with budget, slices, anchors, cursors and views; `outline`; `attachment` (descriptors, bytes, show in file manager); `assert`; `trace`; `pack`; `open-editor`; the live-session control rows.

### Live-session control panel

Shown under the transcript's end in the reader of a live session and inline in each Running row. The operations and their per-harness mechanisms are [agents.md](agents.md#live-sessions)'s; this is their web form.
- **Composer:** a text field (`i`) with **Send**. While the session is working, Send steers it (the harness decides whether the input queues or interjects); while it is waiting or idle, Send answers. The composer shows which channel will carry the text and, for an ended session, the exact resume command it will run.
- **Pending permission:** the request (tool, input in full) with **Approve** and **Deny**, and the harness's "always" variant where it has one.
- **Interrupt**, **Resume** (ended or stale sessions) and **Ping** (with an address and an optional `about:` target).
- **Branch from here:** from any message, start a new run seeded from that point (the harness's native fork where it has one, otherwise a new session seeded with `context --at MESSAGE`), shown as a sibling in the work tree once its transcript shows the link.

Each control shows its availability per harness, with the reason when disabled ("Codex: no approval channel from outside the terminal", "hooks not installed"). Every act is recorded with its author (`operator`, `via: web`) and appears in the session's flow as a delivery or control event. Each control is the same row as its CLI and MCP form.

### Session report `/s/SESSION/report`

Header `Session report · computed from stored facts`; title; ID line; **Open transcript**, **Share an excerpt**. Cards: **Cost** (`$2.85` · `+ $0.16 in 1 subagent`), **Turns** (`3` · `3 prompts · 18 messages`), **Tool calls** (`13` · `1 error, fixed later`), **Files touched** (`3` · `all edits known`). **Timeline:** a time axis, turn bars labelled by their prompt, a subagent band, tool-call marks by outcome with labels on errors (`cargo test · exit 101 · 3m 43s`), claim marks with their state, a legend; drag to zoom. A **Tool calls** table (time, tool ×N, target, outcome, duration; error rows tinted). The rail holds **Files touched** (path, edits, certainty), **Agent notes, checked** (claim with its check badge, the check, anchor, author, `read why it said so`), **Lineage and notes**, and compactions and operator interventions when present. "Fixed later" is derived (a failing call followed by a success of the same command), never taken from text. Every mark and row opens the reader at its message.

Rows: `read` with views `summary` and `changes`, the session's `tool_calls`, and the `claim` assertions on it with their check state.

### Share an excerpt `/s/SESSION/share`

A message list with a checkbox per message and note. **Before saving** lists what is included (messages, notes, their check states). The text is written as stored, with nothing flagged, masked or redacted; paths, names and addresses stay. **Save as one HTML file** runs the `read` row with the selected references, `--format html` and `--to` the operator's download directory, resolved by the daemon ([reading.md](reading.md#writing-out)). The confirmation names the written path and counts. The file is self-contained: inline CSS, no scripts, no archive path. Sharing ends at the local file; carrying an excerpt into a web chat is a [chat action](extension.md#chat-actions).

### Document, message and note pages

- `/d/DOCUMENT`: title, purpose, authorship, text, the stored inputs it was read from; `full` and `dialogue` views; paging; the inspector.
- `/m/MESSAGE` redirects to the reader anchor.
- `/n/NOTE`: the revision (kind, author, `via`, target links, body, relations, check state for claims); **revision history** with diffs; a **conflict** view when two heads are current ([assertions.md](assertions.md#currency)), resolved by writing one revision that revises both; withdrawn state; **Referenced by**; judgments on this revision with their authors; actions **Revise**, **Withdraw**, **Judge**, **Add to page**. Revised and withdrawn revisions stay readable.

### Notes, claims, judging, triage

- **`/notes`:** filters by kind, author (operator, agents, a session, `unattributed`, `via`), target, judged state (unjudged by the operator, accepted, rejected), conflict, stale (target text changed since its `seen_hash`) and withdrawn. Rows: kind badge, body, target, author, time, judgments, check badge. **Needs review** lists current, unwithdrawn notes by other authors without an operator judgment. Bulk actions: judge, withdraw, tag. Any author's judgment is shown as that author's opinion; acceptance means an operator judgment ([assertions.md](assertions.md#judgment)).
- **`/claims`:** claims are `claim` assertions. Each shows its **check** badge (`holds`, `fails`, `unknown` with the reason, `unchecked`) with the check query, `expects rows · got N` and the failing rows linked; the verification it rests on is shown as the check command and exit status the transcript records (`cargo test · exit 101`, or a declared wrapper's `job 1234 · cargo test · exit 101`), never as the agent's words. The header reads `checks evaluated now, at read time · never stored · never a verdict`. An accepted claim that the archive now contradicts says so. "Claimed done, last check failed" is a shipped saved query and one filter here.
- **`/judge`:** the command line (`polylogue judge --blind --sample 5 'notes where rubric:outcomes@r2'`); the rubric card; the item card (`item 4 of 5 · order shuffled · author hidden until the verdict`) with its target and last messages, `a` accept · `r` reject · `d` defer; **This pass so far** (agreement with the operator's judgments on the same rubric revision, deferred, left); **withdraw the whole pass**; **revise the rubric and re-run**. Calibration figures of agent passes against operator-judged samples are shown here.
- **`/triage`:** sessions ordered by the sum of named facts, each with its contribution; the tag `triaged` empties a row.
- **Marks:** star and pin are tags (`tag star`, `tag pin`), shown as toggles on every session and message. "Pin" keeps an item on top of lists; a page slot that holds an exact revision is "fixed".

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
- **Ping composer:** member or explicit address, `about:` (whole page, a section, a slot), text; the line `will run: codex exec resume 01a20000-0006-… "…"` (for a live member, `delivered at its next hook event`); **send ping**. Sending runs it: delivery is [agents.md](agents.md#pings)'s, and there is no copy-only default.
- **Directing the model:** an operator annotation on a slot or section that addresses a member (`@[reviewer] expand this`) is a ping like any other.
- **Board layout:** the same slots as cards on a canvas, positions stored as slot values; drag or arrow keys move a card, which writes a slot revision. **Table layout:** slots as rows with kind, target, state, author, last change. **Table of contents:** exactly `polylogue page toc PAGE`. **History:** the page's revisions with author and time, and the page as of a chosen moment by a scrubber.
- **Sidenotes and previews:** on wide screens notes about a slot sit in the margin beside it; hovering any reference shows a preview from one shared reference reader, and a hover never runs a model or changes anything.
- **Add to page from anywhere:** `+` on any target opens a picker of page, section, following or fixed.

Editing a slot's note writes a revision; moving or removing a slot changes only the page; withdrawn slots disappear and show in history. Agents' markers appear live under their section, authored by the message's author. A stale base is refused with the current head shown; two heads from two machines show as a conflict. Live-query, figure and note slots update from the stream; fixed slots never change.

**Templates** are pages, editable by the operator: **investigation** (subject question; Goal, Assumptions, Alternatives, Plan, Evidence, Findings, Open questions, Decision); **project story**; **weekly review** (a week's sessions per logical week and harness, decisions recorded, findings with checks, open items per repository with what resolved them, cost and quota of the week, an empty **Next week** section); **day story** (the logical day's sessions in time order, what each produced, interventions, cost); **feature story** (from the first mention of a subject to its landing commit or pull request, with the sessions and decisions in order); **bundle** and **workspace**. No template writes generated prose; model-written narrative of the recent past is not offered while model calls are off (default; open operator question).

### Workspaces `/w/MODE`

A workspace is a page with the `workspace` template whose slots are the open targets and whose values record mode, lane widths and focus; saving one is saving the page. An unsaved workspace lives in the URL (`/w/stack?ids=A,B,C&focus=B:msg42`).
- **Tabs:** open targets as tabs; `Ctrl-click` opens in the background; tabs show the live state and a short title.
- **Stack:** two to five lanes, each a compact reader with its own scroll and live chip, and one inspector locked to the active lane. **Stack builders:** parent chain, siblings and forks, subagents, same repository recent, selected search results, sessions sharing a file. More lanes collapse the oldest into tabs.
- **Compare:** [below](#compare-compare).
- **Timeline:** a chronological merge of the lanes' messages, tool calls, notes and lineage events, ordered by time or by session; clock gaps and provider-relative times flagged; tool bursts collapsed (`… tool burst · 42 runs · 22m · expand`).

### Compare `/compare`

Left and right headers (title, origin, relation such as `fork @ msg 42`, counts); alignment **by lineage · by prompt · by time · by shared file**; paired rows with a gutter naming the row type (`same`, `shared paste`, `divergent`, `shared file`, `decision`) and tool outcomes per side; a **metadata diff** above (model, harness, duration, tool calls, errors, cost, outcome); a footer with the rule and counts; **Add both to page**, **Fix paired messages on a page**, **Copy comparison summary**. By lineage, shared message identity and the recorded branch point are exact; by prompt and by time, the row type is labelled as a comparison rule, never "same message". A missing side stays visible (`missing on right`). For more than two sessions, a metadata table with one column per session, compared on matched tasks with denominators. Row: `compare LEFT RIGHT --align …`.

### Timeline, days, firehose

- **`/timeline`:** a year **calendar heatmap** (sessions, prompts, cost or tool errors per logical day, with an accessible table equivalent); below it a **scrubbable time axis** with lanes per repository or harness, sessions as bars, tool errors as marks and notes as `◆`; drag selects a window, the wheel zooms hour ↔ day ↔ week; the table below lists the window's sessions and events. Timeline, table and measures share three bindings: the time window, the focused evidence (opened at its neighbourhood, never by loading every earlier message), and the file at a moment. The header names the zone and the logical day start ([operations.md](operations.md#configuration)).
- **`/day/DATE`:** the logical day's activity by time (each with its time source), notes and decisions, cost and usage, previous and next, **This day in other years**, **Undated records**. In primary mode a wake-to-wake view of the day, read from Sinex, appears beside it.
- **File at a moment:** a file's state graded by evidence: known (after a successful edit), possible (after a failed one), unknown; never by re-running commands.
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

Renders any `| measure` or `| agg` query, or a kept figure, as **a chart and its table**. The chart form follows the answer's shape: a series over time → bars per grain (lines for cumulative measures); a grouped count or sum → horizontal bars sorted by value; a distribution → a histogram or a percentile strip (nearest-rank percentiles); survival → a step curve; a comparison across groups → paired bars with denominators. Every chart shows its unit, denominator query, scope, the count of unknowns (hatched, never zero) and the product's validity note ([measures.md](measures.md#rules)); each bar or point opens the rows behind it. Charts are SVG rendered by the daemon from the typed answer, with the shell adding hover and selection; there is no charting script. **Keep as figure** turns the shown query into a citable figure for a page's figure slot. Shipped saved queries open here first: **where the time went** (time per session, repository or day split into model working, tool running by family, waiting on the operator and idle), **operator interventions**, and **context pressure** (context size per response and what each compaction dropped). Ruling, 2026-08-02: an HTML reporting skill is "best just thought of as a stopgap, and we'd build such capabilities into polylgoue webui, properly" (cc:72ace629-1777-4df9-ba71-9eb58505ea65:a622d056-3f3c-458a-b265-73eba5e904bd).

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
- **The operator's facts and open choices** ([operations.md](operations.md#configuration)): logical day (read from Sinex in primary mode), subscription plan and price table, enabled variants, quiet hours and interrupt budget (Sinex's in primary mode), listen address and authentication, the optional notification route command.

Every change shows who made it and when.

## Performance targets

| Target | Budget |
|---|---|
| A view's first readable paint, warm | ≤ 1 s; reads within the `read` class (≤ 100 ms to the first screen) |
| Reader first, middle and last page of a 100,000-message session | each within the `read` class, scrolling without loading the whole session |
| Search page | the `search` class (≤ 250 ms) |
| Cost, analysis and report figures | the `aggregate` class (≤ 400 ms by default); a figure that times out offers **Run without the time limit**, never a refusal |
| Heatmap counts by logical day, warm | ≤ 50 ms |
| From a harness's file append or hook line to its fragment in an open reader or Running; a note written from the CLI or MCP to an open page | ≤ 1 s at p95 on a warm daemon |
| Keyboard and local interaction | ≤ 100 ms |

A missed budget is a defect. Figures are paired benchmark runs ([testing.md](testing.md#benchmarks)); no test asserts elapsed time.

## Laws

Checks run a real browser over the DevTools protocol against the real daemon on the [Morrow](testing.md#morrow) archive with the clock at 2026-10-08 12:00 Europe/Warsaw (live views: the live set at 2026-10-09 15:00).

- **footer-command-reproduces-view**: every view's rows equal the rows of the CLI command its footer shows (IDs, counts, order), for every view and for a filtered `near:` query across HTTP, CLI and MCP.
- **no-web-only-capability**: no web route serves a capability that is not a command-table row.
- **web-writes-authored-by-daemon**: every write from the web is stored with author `operator` and `via: web` (or `extension`), computed by the daemon; a request carrying an author field is refused as invalid.
- **retry-writes-once**: a retried write with the same `request_id` creates one record.
- **live-update-never-moves-reader**: a live update never moves the reader, steals focus or removes a text selection; following appends only at the end.
- **stale-reply-never-wins**: a stale reply never overwrites a newer selection.
- **unknown-never-zero**: an unknown value is never rendered as zero, `$0` or a zero-height bar.
- **disabled-action-has-reason**: every disabled action shows its reason, taken from the operation's availability answer.
- **four-states-without-modal**: every view renders its four states without a modal, each with a recovery command, and no recovery is a repair command.
- **ping-delivered-once**: a ping sent from the web is delivered once, to a live session at its next context-accepting hook event or to an ended one by one run of its harness's resume command; the delivery records the concrete session and never retargets.
- **launch-mints-no-lineage**: continuing or branching from the web creates no lineage link by itself; a link appears only when the new session's transcript shows the delivery.
- **context-copy-has-manifest**: Copy as context includes a manifest listing every included reference with its `seen_hash`, and records a `prepared` delivery whose ID appears in the copied text.
- **archive-path-never-sent**: no HTTP payload contains the archive directory's path.
- **any-surface-by-reference**: every view a reference opens renders from the reference alone, with no state held by another surface; opening the same reference from the CLI's `read --to browser`, a pasted URL and a fresh browser gives the same view.
- **nothing-leaves-unasked**: nothing leaves the machine through the workbench except a file the operator saves and acts the operator takes toward a provider through the extension; the web answers only on the configured listen address with its authentication, and a non-loopback address without a token refuses to start.
- **stored-text-unchanged**: no view, file or copy masks, flags or redacts stored text; a synthetic credential in a Morrow message appears byte-identical in the reader, the share file and a copied pack.
- **preferences-in-archive**: product state persists as archive settings; clearing browser storage changes no view preference.
- **csp-on-every-route**: the CSP header is present on every route, and a browser run records no CSP violation.
- **fixed-slots-never-change**: fixed slots never change; following slots show the current revision; two heads render as a conflict.
- **pages-store-references**: pages, bundles and stories store references, never copied text; the quoted text equals the archive message.
- **key-parity-with-picker**: every key shared with the CLI picker does the same action there.

Checks on Morrow, each a test:
- Home shows 8 sessions, 7 top-level, 17 prompts, 35 tool calls, the known cost with the price table's unknowns, the `atis-latch` notice, and a links column naming the fork, continuation, subagent and abandoned branch for #6, #4, #2 and #7.
- Running at 15:00: the waiting session (permission for Bash `cargo test --all`) first with **Approve** and **Deny**; the working session with `job-morrow-7` and the `cargo test parser` child linked to its tool call; the stale session under Stopped with "process gone at 13:40"; the peer warning. At 15:15 with no events the waiting session stays waiting and the working one changes per its `until`. Approving records an act authored `operator` via web.
- Search `chrono`: 3 sessions; subagent hits under #2; the decision note on the 10:56 hit; the defaults line names #4's inherited copies and #7's abandoned branch, and "include them" adds them; a bare `chrono` typed in the field is refused with the `find chrono` form.
- Reader #2: 13 of 18 messages shown with the folded-results footer; `cargo test` reads `error exit 101 3m 43s` with the FAILED line; `Read` lines show no duration; the Edit expands to a diff; the Agent line expands to the delegation card; the provenance tab of the 10:37 call shows input, byte range, hash and parser version; `1` sets the outcome solved. #4 shows the compaction line (14:40, 182k → 14.5k) and the inherited line of 18 messages; #7 shows the branch line (21:04) and **show branches**.
- A 100,000-message synthetic session opens its first, middle and last pages within budget and keeps a selection across window moves.
- Report #2 reproduces the board's figures, and "1 error, fixed later" is derived.
- Sharing 10:37–10:52 plus the claim writes one self-contained HTML file with no archive path and the stored text unchanged, through the `read` row.
- Claims: the claim on #2 `holds`; "every test run passed" `fails` with the 10:37 row; `/judge` shows "2 of 3".
- Pages: `page new --template investigation morrow-next-friday` and adding `msg_01a20000a11` under Evidence gives the Page board; the `::finding` marker in #6's final model message appears under Findings authored by `codex:…0006` without reload; reordering leaves the note's revision unchanged; the table of contents equals the CLI's; history shows the page before the finding; a ping to an ended member runs its resume command once and lists the delivery.
- Compare #5 and #6 by lineage pairs the shared prefix as `same`, shows the fork point, and shows every message once.
- Timeline: a 02:30 message lands on the previous logical day; dragging a window supersedes the earlier request.
- Cost since Oct 01: known `$5.86 (exact 5.8626)`, 4 unknown; Oct 04 and Oct 07 hatched.
- Analysis renders a kept figure with its denominator and hatched unknowns, and a bar opens its rows.
- Each run stores screenshots of every view at 1,440 and 390 px in light and dark, an accessibility report, a keyboard-only journey, and screencasts usable in the README.

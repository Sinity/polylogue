# Web designs

The [workbench](web.md) follows these visual baselines.

## Visual baselines

The boards and mockups in [`web/`](web/) are the visual baselines, drawn on the synthetic Morrow archive (clock 2026-10-08 12:00 Europe/Warsaw; the live set at 2026-10-09 15:00). The canvas boards and mockups use the joint visual tokens (IBM Plex Sans and Mono, light and dark). Where a board and this page disagree, this page wins and the board is redrawn. A visual baseline match is a review beside the named board, not a pixel test.

### Lineage

Four rounds of design feed the workbench. Each later one refines the earlier ones; none replaces an earlier one whole. An element of an earlier round still binds unless a later board draws the same thing differently or this page says otherwise, and the tables below record which board governs each element.

| Round | Board | What it contributes that still binds |
|---|---|---|
| **MK2 "Cockpit"** (Claude Design, 2026-05-02) | [boards/mk2/](web/boards/mk2/index.html) | the cockpit layout (facet rail, result list, reader, side panel); the source of every fact; the status strip; one keyboard model across web, CLI picker and palette; the empty, loading, degraded and failure states with a line as the way out; the daemon counting its own use; context from selected messages; the data-flow invariants |
| **MK3 / MK3.1 "archive workbench"** (Claude Design, 2026-05-23) | [boards/mk3/](web/boards/mk3/index.html) | the information architecture; the search cockpit with score components and bulk selection; paste and attachment rendering; the message card and its copy menu; show/hide rules and match windows for huge sessions; the inspector in five selection contexts; the topology view; the flow that resumes a session's work in another harness; the compare and timeline views; the palette with disabled reasons; the full state matrix; settings |
| **MK4 concept** (2026-05-24, prose, no board) | — | resuming in another harness from any message; observed lineage, never created by a UI act; every action shown with its availability and the reason when it is unavailable. These are written into the views below |
| **The board canvas** (2026-10-12) and the static mockups (2026-10-11, 10-24) | [boards/](web/boards/), [mockups/](web/mockups/) | the current look (the joint tokens); the views they draw, on the Morrow archive. Where they draw a view an earlier board also drew, they govern |

The visual style of MK2 (graphite and slate, dark only, Inter and JetBrains Mono) and of MK3.1 (the Sinex ANSI-16 palette) is replaced by the joint tokens; their layouts, panels, interactions and keyboard model are not.

### MK3.1 artboards

| Artboard | Governs |
|---|---|
| [01 Home](web/boards/mk3/index.html#a01) | Home's operational panels: sources with their state, quick filters, live sessions, saved queries, the archive digest and derived freshness, the last-hour feed. Card layout and grouped results: the canvas home board |
| [02 Search cockpit](web/boards/mk3/index.html#a02) | hit rows with score components, facets scoped to the query, the selection bar with bulk actions and their disabled reasons, the query shown as its CLI command. Summary, defaults line and Refine: the search mockup |
| [03 Reader](web/boards/mk3/index.html#a03), [05 Message card](web/boards/mk3/index.html#a05) | paste spans inside a prompt, attachment cards, tool output cut to head and tail with a match window, the copy menu and its forms, the inspector following the selection. Message cards, rails and lineage: the canvas reader board |
| [04 Huge session](web/boards/mk3/index.html#a04) | virtualized reading with windows above and below, scroll restore to a message, show/hide defaults, search matches always visible, the minimap. The outline lists the session's turns rather than phases |
| [06 Pastes](web/boards/mk3/index.html#a06), [07 Attachments](web/boards/mk3/index.html#a07) | how each paste kind and each attachment state renders; the pastes and attachments views |
| [08 Inspector](web/boards/mk3/index.html#a08) | the inspector for each selection context |
| [09 Topology](web/boards/mk3/index.html#a09) | the lineage view over the closed set of session links ([sessions.md](sessions.md#session-links)), with unresolved parents shown as unresolved |
| [10 Flows](web/boards/mk3/index.html#a10) | resuming from a message in another harness (`resume --in`), forking (`resume --fork`) and a bundle from a selection (a `bundle` page), each explicit and previewed ([Web reader](web-reader.md#session-reader-ssession)) |
| [11 Tabs](web/boards/mk3/index.html#a11), [12 Columns](web/boards/mk3/index.html#a12) | the workspace modes, saved as `workspace` pages ([Workspaces](web.md#workspaces)) |
| [13 Compare](web/boards/mk3/index.html#a13), [14 Timeline](web/boards/mk3/index.html#a14) | compare with its four alignments, the timeline, and the matching workspace modes |
| [15 User state](web/boards/mk3/index.html#a15) | marks, notes, saved queries, bundles and workspaces as the notes list shows them; marks are tags ([annotations.md](annotations.md#kinds)) |
| [16 Palette](web/boards/mk3/index.html#a16) | the palette scoped by selection, with disabled reasons, and the keyboard help |
| [17 State matrix](web/boards/mk3/index.html#a17) | populated, empty, partial, degraded and failed for every view ([state matrix](web.md#state-matrix)) |
| [18 Settings](web/boards/mk3/index.html#a18) | density, show/hide defaults, keyboard, sources and access settings |

Parts of MK3.1 that do not bind: the redaction policy, redacted paths and "copy requires confirm" (stored text is shown as it is, with nothing masked); the "needs attention" repair suggestion (`index rebuild`), since derived data converges by itself; the "repaired" topology edge (links come only from explicit markers); the "insights" label (derived data has no umbrella name, [measures.md](measures.md#principles)); and the phase outline of 04.

### MK2 elements

| Element | Governs now |
|---|---|
| three panes (facet rail, result list, reader) with a side panel | MK2, for the wide search layout with a reading pane; MK3.1 02 for the hit rows |
| facets `provider`, `repo`, `has: tool_use · thinking · paste`, saved queries in the facet rail | MK2; scoped counts from MK3.1 02 |
| side panel tabs messages · raw · source | MK2 for the tab set (Outline · Raw · Source); MK3.1 08 for the panels inside |
| source panel: arrival route, source path, byte cursor, hash, derived facts | MK2. The privacy line reads "local archive only"; derived facts never include guessed phases |
| reader header actions star and note, with the equivalent CLI line beside them | MK2; starring is `tag star`; the keys are the [keys](web.md#keys) table's |
| status strip: daemon uptime, live lag per source, capture connection, index pending, key hints | MK2 |
| `⌘K` and `:` palette, `g a` for annotations | MK2 for the palette keys; the go-to keys are the [keys](web.md#keys) table's (`g n` is Notes); MK3.1 16 for the palette |
| one keymap across web, CLI picker and palette ("Same verbs, same letters") | MK2 keymap, as the [keys](web.md#keys) table. A delete key does not exist (Polylogue has no deletion) |
| empty, loading, degraded and failure states on every surface; calm persistent status, never a blocking modal; the way out is always a line the user could run | MK2 for the rule; MK3.1 17 for the matrix. The ways out are everyday lines; there are no repair commands ([architecture.md](architecture.md#principles)) |
| the daemon counting its own use, local only, readable from every surface: query runs, reader opens by surface, time spent per session, per-component lag | MK2 telemetry, as Status's own-use panel. The counters live with the request log in `archive.db`, and nothing is purged |
| status panels: archive counts, sources with lag, capture, recent, diagnostics | MK2's terminal cockpit, as the [Status](web-views.md#status-and-settings-status-settings) view and the status strip. No terminal UI is built now, and diagnostics suggest everyday commands only |
| fuzzy selector: descriptor-driven, preview pane, multi-select, print or open the choice | MK2, for the CLI picker ([terminal.md](terminal.md#the-picker)) |
| CLI: bare overview with source bars, completion popup with counts, opening the latest session in the web reader | MK2, for the CLI ([terminal.md](terminal.md#the-orientation-screen), [completion.md](completion.md)) |
| data-flow invariants: any surface renders anything by reference alone; nothing leaves the machine unless the user writes it out or configures a listen address | MK2 |
| context from multi-select, message references without copied payloads | MK2, as **Copy as context**: the `brief` of the marked rows ([web reader](web-reader.md#session-reader-ssession)) |

The MK2 board is drawn for Python Polylogue, so its command and module names (`polylogued`, `ops doctor`, `--fix` targets, Python file names) are Python Polylogue's; the commands are those in [interfaces.md](interfaces.md#commands), and there are no repair commands. Its roadmap and inventory artboards show Python Polylogue's plan, not views. MK2's MCP role gate (read, write, admin) and its MCP server with direct archive access do not apply: the three MCP tools are nested by access only for the clients' hints, every agent may use all three ([agents.md](agents.md#mcp-server)), and the daemon is the only process on the archive.

### Views

| View | Baseline |
|---|---|
| [Home](web-views.md#home-) | [boards/home.html](web/boards/home.html) (cards and grouped search results); [mockups/index.html](web/mockups/index.html) (recent sessions with links column, recent notes, sources) |
| [Search](web-views.md#search-searchq) | [mockups/search.html](web/mockups/search.html) (summary, defaults line, Refine, hits grouped by session); grouped results also on [boards/home.html](web/boards/home.html) |
| [Session reader](web-reader.md#session-reader-ssession) | [boards/reader.html](web/boards/reader.html) (message cards, collapsed tool lines, delegation card, lineage, notes and views rails); [mockups/session.html](web/mockups/session.html) (toggles, compaction, branch and inherited-prefix lines) |
| [Running](web-views.md#running-and-waiting-running-waiting) | [boards/running.html](web/boards/running.html) |
| [Session report](web-reader.md#session-report-ssessionreport) | [boards/report.html](web/boards/report.html) |
| [Share an excerpt](web-reader.md#share-an-excerpt-ssessionshare) | [boards/share.html](web/boards/share.html) |
| [Claims](web-views.md#notes-claims-and-triage-notes-claims-triage), [Review](web-views.md#review-and-compare-review-compare) | [boards/claims.html](web/boards/claims.html) |
| [Pages](web-pages.md#the-page-view): a page | [boards/page.html](web/boards/page.html) (sections, slots, sessions on the page); [mockups/page.html](web/mockups/page.html) (members, the send composer, decision box, kept result, live query) |
| [Pages](web-views.md#pages): the page list | [boards/pages.html](web/boards/pages.html) |
| [Project story](web-views.md#project-story) | [boards/story.html](web/boards/story.html) |
| [Judging passes](web-views.md#judging-passes-judge) | [boards/claims.html](web/boards/claims.html) (sampled items, agreement so far) |
| [Workspaces](web.md#workspaces) | MK3.1 11–14 |
| [Cost](web-views.md#cost) | [mockups/cost.html](web/mockups/cost.html), chosen where it differs from [boards/cost.html](web/boards/cost.html) (cost by origin, most expensive sessions, the Codex observation subset priced apart) |
| Extension popup | [boards/extension-popup.html](web/boards/extension-popup.html) |
| [Compare](web-views.md#review-and-compare-review-compare), [Timeline](web-views.md#timeline-days-firehose) | MK3.1 13–14 |
| [Pastes and attachments](web-views.md#pastes-attachments-topics) | MK3.1 06–07 |
| [Notes](web-views.md#notes-claims-and-triage-notes-claims-triage) | MK3.1 15 |
| [Status](web-views.md#status-and-settings-status-settings) | MK2 status panels and own-use counters |
| [Settings](web-views.md#status-and-settings-status-settings) | MK3.1 18 |

Parts of the canvas boards and mockups that do not bind, because the text decides otherwise:
- The navigation on the boards lists Home, Running, Cost and Pages; the full navigation is in the [frame](web.md#frame).
- On the page boards, the composer that addresses a session and its list of sent messages are the send composer (`send … --page --section`), and the table of contents is the contents view ([Pages](web-views.md#pages)). The boards draw no citations, margin, Backlinks panel, proposals or in-place editing; those follow [web-pages.md](web-pages.md).
- The share board flags paths under the home directory and lists `has:secret` blocks; sharing writes stored text unchanged with nothing flagged ([share](web-reader.md#share-an-excerpt-ssessionshare)).
- Tool durations show on shell and subagent calls only; a duration on a file read in a mockup does not bind.
- The mockups' "export JSON" link is `read --format json`; there is no export verb ([reading.md](reading.md#formats-and-destinations)).
- The boards' footers show a CLI line only; every view's footer also shows its MCP call ([frame](web.md#frame)).

## Views without a board

Views that no board draws are drawn in Claude Design before they are built, with at least two versions wherever layout is a real choice; the operator picks one, and the chosen board becomes that view's baseline.

The views without a board: the heatmap, the firehose, the note page, triage, analysis, saved and kept, the live-session control panel, the flow panel, a page's board and table layouts and its contents view and history panel, the page's margin with citation sidenotes and discussion, the Backlinks panel, in-place Markdown editing with its pickers, proposals and tracked changes, the active-page header and **+ page** while reading, the bundle composer, the judging-pass rankings and calibration panels, the trace view, and the extension's in-page UI. Their behaviour in [Web views](web-views.md) and [Web reader](web-reader.md) binds; their layout does not until the board exists.

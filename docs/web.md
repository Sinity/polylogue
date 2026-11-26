# Web workbench

The workbench is the web interface the daemon serves: reading, searching, measuring and annotating the archive, arranging work on [pages](pages.md), and reading and answering live agent sessions as they run. It is one more client of the [operation table](interfaces.md#operation-table) and adds no behaviour of its own: every view is a line or a command. This page owns its principles, frame, workspaces, key bindings, inspector, live behaviour, state matrix, access and routes. [Web designs](web-design.md) owns the visual baselines; [Web reader](web-reader.md) and [Web views](web-views.md) describe every view. Shared web mechanics (visual tokens, the shared key table, palette and copy mechanics, the frame, server-sent-event fragments, windowed lists) belong to the common [`web-shell`](https://github.com/Sinity/polylogue-sinex-common/blob/master/crates/web-shell/README.md) crate. The browser extension's own interface is [extension.md](extension.md); rendering rules for messages and read views are [reading.md](reading.md); live sessions and the acts on them are [session-control.md](session-control.md); status figures are [operations.md](operations.md#status).

## Principles

- **A full designed interface, built from the designs.** Every view is specified to the fidelity its board draws; the [visual baselines](web-design.md#visual-baselines) are built faithfully, and views without a board are drawn before they are built.
- **A shell over the same operations.** Every view is a line or a command, and every action is a verb or command reachable from the CLI and MCP; there is no web-only endpoint. Every view shows the CLI line and the MCP call that reproduce it ([footer line](#frame)).
- **Where the operator works with agents.** Live sessions are read as they happen (`read --follow`), answered (`answer`), sent to (`send`), interrupted (`interrupt`) and resumed (`resume`, `resume --in`) from the web ([Running and Waiting](web-views.md#running-and-waiting-running-waiting), [live-session control](web-reader.md#live-session-control-panel)).
- **Pages are where work is arranged, and optional.** The model is [pages.md](pages.md); the page view, citing while reading and editing in the browser are [web-pages.md](web-pages.md). Every other view works the same with no page in the archive.
- **Live, never polling.** Views update from the daemon's committed changes ([live behaviour](#live-behaviour)).
- **Every visible fact shows its source.** A chip, figure or line opens where it came from ([inspector](#inspector-and-source)).
- **State that must last lives in the archive.** The browser holds only view state in progress ([what the browser holds](#what-the-browser-holds)).
- **Verified by running it.** Checks drive a real browser against the real daemon.
- **Stored text is shown as it is.** No view masks, flags or reveals spans, and there is no delete action, because Polylogue has no deletion ([architecture.md](architecture.md#deletion)).
- **No dependency on one desktop.** Desktop integration (opening an editor, a file manager, a terminal, the clipboard) goes through the desktop's own services; a desktop or window manager named on this page is an example of a deployment.

## Technology

Each view is HTML rendered by the daemon with `maud` from the same typed reply the CLI renders as text, so a reloaded row and a live row come from one function. One TypeScript shell from `web-shell` handles keys, the palette, copying, server-sent-event fragments, windowed lists for long readers, and pointer handling for the timeline. There is no single-page application, no WebAssembly and no third-party runtime script, and every route carries a strict content security policy (`script-src 'self'`, `style-src 'self'`). Web comes first and there is no native application: an installed app window gets the desktop's notification, shortcut and window services, and the typed replies keep a native or terminal front end a thin later addition.

## Frame

```
┌ header ──────────────────────────────────────────────────────────────────────────────────┐
│ Polylogue  Home  Running(2·1 waiting)  Search  Notes  Pages  Review  Saved  Cost  Status [query…] │
├─ left rail (optional) ─┬─ main ───────────────────────────────┬─ inspector ───────────────┤
│ outline / facets /     │ view content                          │ facts · lineage ·         │
│ filters                │                                       │ notes · source …          │
├────────────────────────┴───────────────────────────────────────┴───────────────────────────┤
│ ● daemon 4h 12m · claude-code 0.4 s · codex 1.1 s · extension connected · index 1,210 pending · : palette  e query  ? keys │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
```

- **Header:** wordmark; navigation Home · Running · Search · Notes · Pages · Review · Saved · Cost · Status (Review opens [Review and compare](web-views.md#review-and-compare-review-compare)); the persistent query field with completion on every view. The Running item carries `running · waiting` counts, so "waiting on you" is visible from every page. While the tab has an [active page](web-pages.md#citing-while-reading), the header ends with `Collecting to page:flaky-ci › Evidence ✕`: the page and section that `+` places into, and the control that stops collecting.
- **Status strip:** daemon state and uptime; **live lag per source** (time from the newest byte a live source wrote to its commit in the archive, per origin; `—` when the source is idle; a figure state when not measured); the extension's connection (from the host's last contact); index work pending; in primary mode, the Sinex position lag. Each item links to its Status section. The strip is the daemon state indicator: `live`, `catching up · N inputs waiting`, `disconnected · retrying`, `stopped (start: <service command>)`.
- **Main column** up to 1,200 px; transcript text 75–90 characters per line.
- **Inspector** (right rail, 280–320 px): the view's own facts by default, switching to the focused target with a "back to session" link.
- **Left rail** only where it earns its width: session outline, search facets, a page's contents, timeline filters.
- **Narrow widths** (under 900 px): rails become disclosures above the main column; tables scroll inside their box; there is no horizontal page scroll. Every view stays usable at 390 px.
- **Footer line** on every view: the CLI line and the MCP call that reproduce what is shown, each with a copy button; a view with several tables has one pair per table. Both are computed from the request arguments that produced the view, so the CLI line, the MCP call and the HTTP request carry the same selection, steps and argument names:

  ```
  polylogue 'repo:sinex since:7d' read --view dialogue
  polylogue_read {"arguments": {"selection": "repo:sinex since:7d", "steps": [{"verb": "read", "view": "dialogue"}]}}
  ```

- **Query handle:** every query-backed view shows its `query:<hash>` (and `result:<id>` when the rows are kept) beside the footer line ([saved-queries.md](saved-queries.md)).

### Visual use

The tokens, marks of meaning, value states and chips are `web-shell`'s. Polylogue's use of them:
- **Grounds:** the operator's own messages and nested blocks sit on `--surface-sunk`.
- **Accent:** Polylogue blue `--accent-polylogue` for links, focus ring and primary buttons; it never carries a state.
- **Palette from the desktop theme:** on by default when a palette file exists (pywal; the `theme.palette` setting names another file). Grounds and ink derive from it while semantic colours and contrast checks stay; a palette that fails the contrast checks falls back to the designed one, and the theme setting says why.
- **Values:** the value-state table of [reading.md](reading.md); missing values in charts use the missing-value hatch, never a zero-height bar.
- **Notes:** `◆` plus the kind word, `--note` ink on `--note-bg`.
- **Check states:** `holds`, `fails`, `unknown`, `unchecked` badges; `unchecked` outlined.
- **Live states:** the six states of [live-sessions.md](live-sessions.md#live-state), always as the word: `waiting` on `--pending-bg`, `working` on `--observed-bg`, `idle` and `unknown` on `--surface-sunk`, `stale` in `--failure-ink`, `ended` plain.
- **Authorship:** typed text plain; a pasted span shaded with one line saying where it came from; unknown authorship hatched. Annotations show their author (`operator`, `agent:<session>` as a link, `agent:<origin>`, `script:<unit>`, `unattributed` shown as "author unknown") and `via` ([authorship.md](authorship.md)).
- **Density:** comfortable (default), compact, dense; a setting.
- **Motion:** none except the live tick and progress; off under `prefers-reduced-motion`.
- **Chips:** text first, ordered origin or harness · live state · repository, working directory, branch · lineage state · counts · cost · notes and tags; at most one row in a header.

### Workspaces

A workspace is several targets open at once: sessions, pages, kept results and searches side by side. A saved workspace is a page from the `workspace` template ([pages.md](pages.md#templates-and-shipped-pages)): its slots are the open targets in order, each slot's `w` the width of its column, and the page's parameters record the mode and the focused target, so saving a workspace is saving a page and reopening it is reading that page (`/page/NAME`). An unsaved workspace lives in the URL (`/w/columns?ids=codex:019d0f2a,claude-code:7f3a2c10&focus=claude-code:7f3a2c10`); **Save workspace** writes the page (`polylogue 'codex:019d0f2a OR claude-code:7f3a2c10' save page:morrow/desk --template workspace`).
- **Tabs** (`/w/tabs`): open targets as tabs; `Ctrl`-click opens in the background; each tab shows its live state and a short title.
- **Columns** (`/w/columns`): two to five columns, each a compact reader with its own scroll and live chip, and one inspector following the active column. Columns are filled by column builders: parent chain, siblings and forks, subagents, recent sessions in the same repository, the marked search results, sessions sharing a file. Opening more columns folds the oldest into tabs.
- **Compare** (`/w/compare`): the open columns read with the `compare` view ([Compare](web-views.md#review-and-compare-review-compare)).
- **Timeline** (`/w/timeline`): the columns' messages, tool calls, notes and lineage events merged by time or grouped by session; clock gaps and provider-relative times flagged; tool bursts collapsed (`… tool burst · 42 runs · 22m · expand`).

## Keys

Navigation, selection, the palette, help and the copy menu keys come from the shared key table in `web-shell`. The picker and the web use the same letter for the same act wherever both offer it, except marking and rejecting ([terminal.md](terminal.md#keys)): the picker's `Ctrl-r` read, `Ctrl-y` copy and `Ctrl-e` query are the web's `r`, `y` and `e` ([terminal.md](terminal.md#keys)). The picker needs the Ctrl forms because its prompt takes letters; the web binds the bare letters. Single-character keys can be switched off (a setting), are ignored in inputs and while text is selected, and move real DOM focus, which live updates never steal.

| Key | Action | Line it runs | Scope |
|---|---|---|---|
| `g h` `g r` `g s` `g n` `g p` `g v` `g k` `g c` `g t` | go to Home, Running, Search, Notes, Pages, Review, Saved, Cost, Status | — | everywhere |
| `+` | place the focused target or the marked rows on the active page's current section (a message as an excerpt, a text selection as a span excerpt, a tool call as its card, a result row as the row); with no active page, or with `Shift-+`, a picker of page (or a new one from a template), section, form and **cite in a paragraph…** ([web-pages.md](web-pages.md#citing-while-reading)) | `REF save page:NAME --section S` (`--as FORM`, `--fixed`, `--live`) | everywhere |
| `P` | pin the focused target | `REF save page:pinned` | reader, lists |
| `e` | focus the query field | — | everywhere |
| `r`, `Enter` | read the focused row | `REF read` | lists |
| `x` | mark the focused row; the selection bar acts on the marked rows | — | lists, reader |
| `y` | the copy menu; `y c` copies the view's CLI line, `y m` its MCP call | — | everywhere |
| `o` | open in the editor at the session's working directory, or a tool line's file at its line | — | reader |
| `a` / `A` | write a note / a decision on the focused target; on a page, in the focused section | `REF note TEXT`, `… --as decision`; `page:NAME note TEXT --section S` | reader, search, notes, page |
| `t` | tag the focused target | `REF tag NAME` | reader, lists |
| `s` | star or unstar the focused session or message | `REF tag star`, or the star's `withdraw` | reader, lists |
| `n` / `N` | next / previous match of the search in the session | — | reader |
| `m` `w` `p` | inspector tab Outline / Raw / Source | — | reader |
| `v` then a letter | read view: `t` transcript (linear on a page), `d` dialogue, `o` overview, `c` context, `e` tree, `g` changes, `l` live, `w` raw, `x` trace, `n` contents; `f` opens the flow panel | `REF read --view VIEW` | reader, page |
| `v b` `v a` `v h` | page layout board / table, and the page's history | `page:NAME read`, `--at POSITION` | page |
| `i` | edit the focused slot in place: its Markdown source with a live preview, saved with `Ctrl-Enter` ([web-pages.md](web-pages.md#editing)) | `REF edit TEXT` | page |
| `m` | move the focused slot: arrows, then `Enter` ([web-pages.md](web-pages.md#moving-and-layouts)) | `SLOT save page:NAME --after SLOT` | page |
| `Alt-↑` `Alt-↓` | move the focused slot before its previous or after its next sibling; on the board, the arrow keys move the focused card | `SLOT save page:NAME --after SLOT` | page |
| `z` then a letter | show or hide: `t` tools in full, `k` thinking in full, `h` harness messages, `i` inherited, `b` branches, `s` spawns, `a` annotations, `u` turns, `c` the context meter | `REF read --show WORD` | reader |
| `c` | compare: with one target opens the chooser, with two or more opens Compare | `'QUERY' read --view compare` | lists, reader |
| `l` | read live on or off | `REF read --follow` | reader |
| `i` | focus the live-session composer | `REF send TEXT` | reader, Running |
| `1` `2` `3` | outcome `solved` / `partial` / `abandoned` | `REF note solved --as outcome` | reader, review |

Inside a review card, `a`, `r` and `d` mean accept, reject and defer (`REF judge accept`), and the card shows them on its buttons. On a page, `i` edits the focused slot and `m` moves it, since a page has no live-session composer and no Outline tab; the page's other keys are [web-pages.md](web-pages.md#keys). No other key changes meaning by view. The show/hide letters name the closed word table of [reading.md](reading.md#show-and-hide); `z` with Shift hides the word instead of showing it.

**Palette** (`:` or `Ctrl-K`): every action available for the current selection, generated from the operation table's verbs and commands plus the view's navigation. Each entry shows its key, its CLI line and, when disabled, the reason from the operation's availability answer. It never holds an action the operation table lacks.

**Copy menu** (`y`), per target:

| Target | Default | Other forms |
|---|---|---|
| view | CLI line | MCP call; URL; HTTP request (`/api/v1/rows`) |
| session | its reference | title and reference; Markdown of the current view; the `resume` line; provider URL when the origin records one |
| message | Markdown with header (`## 10:56 you`) | plain text; typed-only (when paste spans are known); reference; permalink; **as context** |
| tool call | the command or target | input JSON; output as stored; reference |
| annotation | body | `annotation:<root>@<rev>`; the `edit` line |
| page | `page:NAME` | the contents view as Markdown; the page's brief (**as prompt**); the page as Markdown; the working-copy line |
| result row | its reference | the row as TSV |
| selection | Markdown | **as context**: the selection's `brief` in Markdown |

Copied text in the reader also carries its span reference, so pasting it into a page paragraph offers to cite it ([web-pages.md](web-pages.md#copy-and-paste)).

When the browser cannot carry the bytes (exact bytes, non-text types), the menu offers **Put on the desktop clipboard**, which the daemon writes through the desktop's clipboard service.

## Inspector and source

The inspector shows, for the view or the focused target:
- **Facts:** ID, models, tools with counts and errors, tokens, cost with its money kind.
- **Lineage:** the compact tree of linked sessions (the `tree` view).
- **Notes:** every annotation on the target, newest first, each with author, `via`, judgments and check state; **Write a note**.
- **Views:** links to the read views, the report and share.
- **Backlinks:** the pages that place this target, the notes and pages that link to it, citations of it or inside it, and inclusions of it, exact rows first and rows inside it rolled up (`links where target:REF`; [page-text.md](page-text.md#backlinks)).
- **Tabs Outline · Raw · Source:**
  - *Outline:* the session outline (the `turns` of the session).
  - *Raw:* the stored input records behind the focused messages, as inert text, with each record's outcome (used, skipped with reason, unreadable); the line is `REF read --view raw`.
  - *Source:* the origin and route (`live`, `capture`, `recorded`, `export`, `copy`, `page_text`); the source location (the provider-side path, never the archive directory); the input, revision and chain; the record's byte range in its chain and, for a live input, the committed length against the file's length; the blob hash; the parser and its parse version; the observations (machine, when found); which of the row's stored versions is its main route and which input each other field group came from; for derived values, the derivation, its rule version and the rows it read; for annotations, author, `via`, revision and relations; in primary mode, the Sinex material and position. Each line links where a target exists.
- **Every chip and figure opens its source:** a cost chip opens the usage records and price entry it came from; a live-state chip opens the live events behind it; a count opens the query that produced it.

## Live behaviour

- **Transport:** one server-sent-event stream per visible tab: the view's own line with `follow`, in the shared framing ([routes](#routes)). A reader or Running streams what its sessions do, as `read --follow` does; a page streams changes to its slots and the annotations it shows; a listing streams each row that starts matching, as `read --follow` does on rows that are not sessions. The event ID is the change position; events are sent after commit and carry rendered fragments (`append`, `replace`) or changed references (`ref`), never whole sessions. A reconnect resumes from `Last-Event-ID`; an unknown position or a new index sends `reload`. A selection the daemon cannot follow is refused by name.
- **Clock-driven changes:** a live state carries `until`, and the view recomputes at `until` without an event.
- **Never moves the reader:** lists hold reordering while a row has focus and show "N updates · apply"; the reader appends only while reading live and at the end, and otherwise shows "N new messages ↓"; a page shows "N changes · show" for a change above the viewport; an update never removes text selection or focus.
- **Hidden tabs** close the stream and refresh on return.
- **Superseded requests:** a newer selection aborts the older request, and a stale reply never overwrites a newer selection.

## Writes, actions and errors

- **Writes are POSTs of lines**: the `rows` arguments (selection and steps) with a client `request_id` reused on retry, so a retried write never writes twice. The view shows the committed annotations the daemon returns and the undo line from the reply; a failed write keeps the editor text. Every write from the web is authored `operator` with `via: web`, computed by the daemon; the web sends no author ([authorship.md](authorship.md)). A write over several rows of a query carries `all`, and the web says how many rows before sending it, as the CLI's `needs_all` refusal does.
- **Actions after a reply** come from the verb's affordance table, the same table the CLI prints as `next:` and MCP returns as `next` ([terminal.md](terminal.md#footers-and-next-lines)). No view offers an action outside it.
- **Availability:** every action carries `enabled`, `disabled (reason)`, `partial (reason)`, `loading` or `unavailable (reason)`, answered by the operation. A disabled action stays visible with its reason ("harness cannot interrupt this session", "needs a recorded paste", "Sinex not configured").
- **Errors** say what failed, what still works and the next step, which is a line the operator could run (the refusal's `try:` line). A failed optional panel shows `unavailable: <reason>`, never zero or empty.

## State matrix

Every view renders four states as a persistent line in the view, never a modal, each with a recovery that is a line.

| View | Empty | Loading | Degraded | Failure |
|---|---|---|---|---|
| Home | "No sessions yet" with the sources Polylogue watches, `polylogue import PATH` and `polylogue demo DIR` | sections `pending` until the status reply | "catching up · N inputs waiting · ETA" from the measured rate | daemon unreachable: the view says so and names the service command |
| Running, Waiting | "Nothing running" with the last ended session | rows `pending` | a source's live lag above its normal range, named | no hooks installed for a harness: rows `unknown (no hooks)` with `polylogue install HARNESS` |
| Search | "0 rows" with the rows each condition kept and the nearest line that returns rows | results stream in; the coverage line says `partial` | `up_to_date: false` banner naming what is behind | a refused query: kind, the submitted text, the corrected query |
| Reader | "No messages" (only harness records) with the raw view link | first-page skeleton | session behind its derived products: values computed on the spot or `pending` | not found: the reference and the nearest match by prefix |
| Pages | "No pages yet" with the templates and the line that makes one (`polylogue page:NAME note TITLE --as page --template investigation`) | slots `pending` independently | a slot whose target is gone from the index: `unresolved reference` with the reference text; a `sinex:<kind>/<id>` slot standalone: `unresolved: Sinex not configured` | a write refused on a moved base: the current revision shown beside the edit; two current revisions shown as a conflict; an `import` of a working copy lists each refused block |
| Judging passes | "No passes" with the shipped rubrics and the line that keeps a population | figures `pending` | agreement `not measured` until a sample is judged | a refused value or judgment: the reason and the revision it needs |
| Review | "Nothing to review" with the judged count | rows `pending` | judgments by other authors shown apart when the operator's are missing | a refused judgment: the reason and the revision it needs |
| Cost, Analysis | "No priced sessions in this range" | figures `pending` | `≥ X (N unknown)`, sessions behind counted | measure refused: reason and a narrower query |
| Timeline, Day | "No activity on this day" | rows `pending` | undated records listed apart | — |
| Status | — | figures `not_measured` | each degraded item with its reason | daemon down (only the strip can say it) |

In primary mode every view adds the Sinex state: `Sinex blocked` (writes waiting, reads from the local index) as a degraded line. The extension's states are in [extension.md](extension.md#states).

## What the browser holds

Only view state in progress: focus, open disclosures within a view, live reading on or off, marked rows, the tab's active page and section, and unsent editor text (in page memory, kept across live updates). Everything a person expects to persist (density, show/hide defaults, rail and panel state per view, the key switch, theme) is a setting in the archive, written with `set`, authored and with history ([Settings](web-views.md#status-and-settings-status-settings)). Browser storage is not used for product state.

## Access

- The listen address and authentication mode are deployment configuration ([installing.md](installing.md#configuration)): by default a loopback bind with a peer-uid check; any other address needs a token file, and a non-loopback address without one refuses to start. Remote or phone access is that configuration; every view works at 390 px.
- `polylogue web` (and `read --to browser`) hands the token over as an `HttpOnly; SameSite=Strict` cookie through a URL fragment exchanged at `/auth`; sign-in removes the fragment from the browser history before exchanging it. The cookie's name distinguishes archives, because cookies do not distinguish ports. Credentials never appear in query strings or change positions. When a cookie is missing or stale, sign-in explains that the daemon may have restarted and keeps the requested local path and query in an archive-specific session cookie until sign-in succeeds. The fresh tab opened by `polylogue web` returns to that page; the exchange clears the temporary cookie.
- Every route checks the Host header, the peer uid where it applies, and authentication. Writes are POST only and require an allowed Origin; POST bodies are `application/json`. There is no CORS.
- Every view and API response carries `no-store`, `no-referrer`, `nosniff` and the content security policy; the shell's static assets under `/assets/` carry the other three and revalidate (`max-age=0, must-revalidate`) instead. Archived markup stays inert, and reading content never fetches remote previews.
- IDs are percent-encoded once. The archive directory's path is never sent over HTTP; text shows it as `[archive]`.

## Routes

The HTTP API carries the same operations as the CLI and MCP, with the same argument names ([interfaces.md](interfaces.md#operation-table)):

| Route | Carries |
|---|---|
| `POST /api/v1/rows` | the `rows` arguments (`selection`, `steps`, `all`, `dry_run`, `explain`) in the shared envelope with a client `request_id`; any line, writes and acts included |
| `GET /api/v1/rows?selection=…&steps=read&view=…` | a line whose steps only read (the listing, `read`, `brief`); each step's arguments are query values under their own names (`show`, `around`, `cwd`, …) |
| `/api/v1/<command>` | a command: `status`, `help`, `set`, `import`, `start`, `embed`, `dev/sql`, `dev/schema/infer`, `dev/schema/unread`, `dev/feed`; GET for those that only read, POST for the rest |

- **One reply, two renderings.** The API returns the typed reply as JSON, or as a stream once it has events. The reader route `/rows?…` takes the same query values as `GET /api/v1/rows` and renders the same typed reply and its `next` calls as links and buttons: a read page in the reader, any other line as its text in the frame. A view's HTML is that rendering inside the frame.
- **Live reading.** A GET with `follow=true` on a `read` step streams `read --follow` with the shared server-sent-event framing until the client closes it; it is the transport of [live behaviour](#live-behaviour).
- **The calling session** for HTTP callers is an envelope field, not a query argument. HTTP callers follow MCP's rule for request IDs: a POST without one gets an ID derived from its arguments ([agents.md](agents.md#mcp-server)).
- **Not over HTTP**, as over MCP: `pick` and row numbers (they need a terminal), `--to` (the browser saves what the route returns), `status --verify`, `web`, `install`, `demo`, `daemon`, `mcp`. Each is refused with the CLI line that does it.

The workbench's view routes are presentation over those operations. A **reader route** (`/s/SESSION`, `/d/DOCUMENT`, `/n/NOTE`, `/page/NAME`) passes its selection and read options as the `read` step's argument names in the query string (`/s/codex:019d0f2a?view=dialogue&show=tools:full&around=message:…`); the reader renders the typed read reply, with no message selection, parent walk or cursor handling of its own, and its next and previous links are that reply's `next` calls. Changing the view starts a new read. `/m/MESSAGE` reads the message the same way. A **collection route** (`/`, `/running`, `/waiting`, `/search`, `/notes`, `/review`, `/judge`, `/pages`, `/saved`, `/cost`, …) renders a listing; a **workspace route** (`/w/MODE`) holds its open targets in the query string until saved as a page ([Workspaces](#workspaces)); its `layout` parameter (list, table, chart, calendar) changes the presentation and never the rows. Each view's line is in the table of [Web views](web-views.md#views). Every view uses the common `web-shell` frame, styles and browser script.

Facet editing in Search and the query field uses the parsed query with spans that `explain: true` returns: editing a facet rewrites the span of its condition and leaves the rest of the query as typed ([terminal.md](terminal.md#explain)).

Unknown view paths answer with HTTP 404 in the reader frame, naming the path and linking the available pages through the navigation. Every view shows the answer's missing reasons, notes and next calls together above its content. Degraded answers label their results as partial. Failed reads use the failure's HTTP status and show its message, hint, suggestion and the command that reproduces the view, including any next call to run without a deadline. Home reports the status reply's index rebuild state or the reason archive totals could not be read.


## Performance targets

| Target | Budget |
|---|---|
| A view's first readable paint, warm | ≤ 1 s; reads within the `read` class (≤ 100 ms to the first screen) |
| Reader first, middle and last page of a 100,000-message session | each within the `read` class, scrolling without loading the whole session |
| Search view | the `search` class (≤ 250 ms) |
| Cost, analysis and report figures | the `aggregate` class (≤ 400 ms by default); a figure that times out offers **Run without the time limit** (`--deadline none`), never a refusal |
| Heatmap counts by logical day, warm | ≤ 50 ms |
| From a harness's file append or hook line to its fragment in an open reader or Running; a note written from the CLI or MCP to an open view | ≤ 1 s at p95 on a warm daemon |
| Keyboard and local interaction | ≤ 100 ms |

A missed budget is a defect. Figures are paired benchmark runs ([testing.md](testing.md#benchmarks)); no test asserts elapsed time.

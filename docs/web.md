# Web workbench

The workbench is the web interface the daemon serves: reading, searching, measuring and annotating the archive, working with Pages, and following and answering live agent sessions. It is one more client of the [command table](interfaces.md#command-table) and adds no behaviour of its own. This page owns its principles, frame, Polylogue's key bindings, inspector, live behaviour, state matrix, access and routes. [Web designs](web-design.md) owns the visual baselines; [Web reader](web-reader.md) and [Web views](web-views.md) describe every view. Shared web mechanics (visual tokens, the shared key table, palette and copy mechanics, the frame, SSE fragments, windowed lists) belong to the common [`web-shell`](https://github.com/Sinity/polylogue-sinex-common/blob/master/crates/web-shell/README.md) crate. The browser extension's own interface is [extension.md](extension.md); rendering rules for messages and read views are [reading.md](reading.md); live-session control operations are [agents.md](agents.md#live-sessions); status figures are [operations.md](operations.md#status).

## Principles

- **A full designed interface, built from the designs.** Every view below is specified to the fidelity its board draws; the [visual baselines](web-design.md#visual-baselines) are built faithfully, and views without a board are drawn before they are built.
- **A shell over the same operations.** Every action is a command-table row reachable from the CLI and MCP; there is no web-only endpoint, and every view shows the CLI command that reproduces it.
- **Where the operator works with agents.** Live sessions are followed, answered, approved, steered, interrupted, resumed and pinged from the web ([Running](web-views.md#running-running), [live-session control](web-reader.md#live-session-control-panel)).
- **Live, never polling.** Views update from the change feed.
- **Every visible fact shows its source.** A chip, figure or line opens where it came from ([inspector](#inspector-and-source)).
- **Pages are the centre of work, and optional.** The model is [pages.md](pages.md); the views are [Pages](web-views.md#pages-pages-ppage).
- **State that must last lives in the archive.** The browser holds only view state in progress ([what the browser holds](#what-the-browser-holds)).
- **Verified by running it.** Checks drive a real browser against the real daemon.
- **Stored text is shown as it is.** No view masks, flags or reveals spans, and there is no delete action, because Polylogue has no deletion ([architecture.md](architecture.md#deletion)).
- **No dependency on one desktop.** Desktop integration (opening an editor, a file manager, a terminal, the clipboard) goes through the desktop's own services; a desktop or window manager named on this page is an example of a deployment.

## Technology

Each view is HTML rendered by the daemon with `maud` from the same typed answer the CLI renders as text, so a reloaded row and a live row come from one function. One TypeScript shell from `web-shell` handles keys, the palette, copying, SSE fragments, windowed lists for long readers, and pointer handling for the timeline and the board layout. There is no single-page application, no WebAssembly and no third-party runtime script, and every route carries a strict content security policy (`script-src 'self'`, `style-src 'self'`). Web comes first and there is no native application: an installed app window gets the desktop's notification, shortcut and window services, and the typed answers keep a native or terminal front end a thin later addition.

## Frame

```
┌ header ─────────────────────────────────────────────────────────────────────────────┐
│ Polylogue  Home  Running(2·1 waiting)  Notes  Pages  Timeline  Cost  Status [query…] │
├─ left rail (optional) ─┬─ main ───────────────────────────────┬─ inspector ──────────┤
│ outline / facets /     │ view content                          │ facts · lineage ·    │
│ toc / filters          │                                       │ notes · source …     │
├────────────────────────┴───────────────────────────────────────┴──────────────────────┤
│ ● daemon 4h 12m · claude-code 0.4 s · codex 1.1 s · extension connected · index 1,210 pending · : palette  / query  ? keys │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

- **Header:** wordmark; navigation Home · Running · Notes · Pages · Timeline · Cost · Status; the persistent query field with completion on every page. The Running item carries `running · waiting` counts, so "waiting on you" is visible from every page.
- **Status strip:** daemon state and uptime; **live lag per source** (time from the newest byte a live source wrote to its commit in the archive, per origin; `—` when the source is idle; a figure state when not measured); the extension's connection (from the host's last contact); index work pending; in primary mode, the Sinex position lag. Each item links to its Status section. The strip is the daemon state indicator: `live`, `catching up · N inputs waiting`, `disconnected · retrying`, `stopped (start: <service command>)`.
- **Main column** up to 1,200 px; transcript text 75–90 characters per line.
- **Inspector** (right rail, 280–320 px): the page's own facts by default, switching to the focused target with a "back to session" link.
- **Left rail** only where it earns its width: session outline, search facets, a page's table of contents, timeline filters.
- **Narrow widths** (under 900 px): rails become disclosures above the main column; tables scroll inside their box; there is no horizontal page scroll. Every view stays usable at 390 px.
- **Footer line** on every page: the CLI command for what is shown, with **Copy as CLI command**; a page with several tables has one command per table. The command is computed from the request arguments that produced the page.
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
- **Authorship:** typed text plain; a pasted span shaded with one line saying where it came from; unknown authorship hatched. Authored records show their author (`operator`, `agent:<session>` as a link, `agent:<origin>`, `script:<unit>`, `unattributed` shown as "author unknown") and `via` ([assertions.md](assertions.md#authorship)).
- **Density:** comfortable (default), compact, dense; a setting.
- **Motion:** none except the live tick and progress; off under `prefers-reduced-motion`.
- **Chips:** text first, ordered origin or harness · live state · repository, working directory, branch · lineage state · counts · cost · notes and tags; at most one row in a header.

## Keys

Navigation, selection, the palette, help and the copy menu keys come from the shared key table in `web-shell`, which the CLI picker shares ([terminal.md#picker](terminal.md#picker)). Polylogue binds these as well; single-character keys can be switched off (a setting), are ignored in inputs and while text is selected, and move real DOM focus, which live updates never steal.

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
| `m` `r` `p` | inspector tab Outline / Raw / Source | reader |
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

## Inspector and source

The inspector shows, for the page or the focused target:
- **Facts:** ID, models, tools with counts and errors, tokens, cost with its money kind.
- **Lineage:** the compact tree of linked sessions.
- **Notes:** every note on the target, newest first, each with author, `via`, judgments and check state; **Add a note**.
- **Views:** links to the read views, the report and share.
- **Referenced by:** pages, slots, kept results and notes that point here.
- **Tabs Outline · Raw · Source:**
  - *Outline:* the session outline.
  - *Raw:* the stored input records behind the focused messages, as inert text, with each record's outcome (used, skipped with reason, unreadable).
  - *Source:* the origin and route (`live`, `capture`, `recorded`, `export`, `copy`, `page_text`); the source location (the provider-side path, never the archive directory); the input, revision and chain; the record's byte range in its chain and, for a live input, the committed length against the file's length; the blob hash; the parser and its reading version; the observations (machine, when found); which of the row's stored versions is its main route and which input each other field group came from; for derived values, the derivation, its rule version and the rows it read; for authored records, author, `via`, revision and relations; in primary mode, the Sinex material and position. Each line links where a target exists. The CLI form is `REF trace`.
- **Every chip and figure opens its source:** a cost chip opens the usage records and price entry it came from; a live-state chip opens the live events behind it; a count opens the query that produced it.

## Live behaviour

- **Transport:** one server-sent-events stream per visible tab (the `stream` row). The event ID is the change feed's position; events are sent after commit and carry rendered fragments (`append`, `replace`) or changed references (`ref`), never whole sessions. A reconnect resumes from `Last-Event-ID`; an unknown position or a new index sends `reload`. Pages declare what they follow; the server filters by reference and refuses by name a subscription it cannot evaluate.
- **Clock-driven changes:** a live state carries `until`, and the view recomputes at `until` without an event.
- **Never moves the reader:** lists hold reordering while a row has focus and show "N updates · apply"; the reader appends only while following and at the end, and otherwise shows "N new messages ↓"; an update never removes text selection or focus.
- **Hidden tabs** close the stream and refresh on return.
- **Superseded requests:** a newer selection aborts the older request, and a stale reply never overwrites a newer selection.

## Writes, availability and errors

- **Writes** are POSTs of command-table requests with a client `request_id` reused on retry. The view shows the committed row the daemon returns, and a failed write keeps the editor text. Every write from the web is authored `operator` with `via: web`, computed by the daemon; the web sends no author.
- **Availability:** every action carries `enabled`, `disabled (reason)`, `partial (reason)`, `loading` or `unavailable (reason)`, answered by the operation. A disabled action stays visible with its reason ("harness cannot interrupt this session", "needs a recorded paste", "Sinex not configured").
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

Session, document and message URLs pass their target and read options to the read route of the command table; a browser asks that route for HTML and a program for JSON. The reader renders the typed read answer, with no message selection, parent walk or cursor handling of its own, and its next and previous links are that answer's ready calls. Changing the view starts a new read. Pages use the common `web-shell` frame, styles and browser script; sign-in removes the token fragment from the browser history before exchanging it.

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

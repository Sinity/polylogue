# Shared web reader

The browser reads through `web::ask` and the daemon's shared request types.
`web/reader.rs` owns session, document and reference routes. Templates for each
view live under `web/pages/`; `pages::layout` and `reader.css` own the shared
frame. New views can register their routes in `web.rs`, use `web::ask`, and
format the masked response through the shared renderer.

## Read pages

`read --budget N` returns one text page. `--slices` additionally returns structured masked block slices; the web reader requests them.
The budget counts estimated tokens at four Unicode characters each. It limits
page content, not the amount of history available. A zero budget makes the
same forward progress as one token. The web default is 8192 tokens. Every
continuation remains reachable through `next`.

Each slice contains `message` metadata (or null for a document), the original
zero-based source `block`, projected `kind` and `label`, a recorded tool-call `outcome` when available,
`text`, `from`, `until`
and `more`. Offsets count characters in the masked projected block before
HTML escaping. Concatenating slices for a message and source block recovers
its complete masked projected text. Empty text can accompany a page containing
only a role heading. Filtering a view preserves source block indices.
Attachments remain page-local masked descriptors in `attachments`.

`Read.start` selects `"latest"` or `{"message":"origin:opaque-id"}`. CLI, MCP
and HTTP expose this as `latest` or `message`. A start requires a budget and
cannot accompany a cursor. Message anchors must belong to the selected
session and parent view; an excluded or missing reference returns `not_found`.
Latest resolves the final message visible in the selected view and reads the
tail of its final projected block within the budget. A view with no visible
content returns an empty page. An explicit message anchor may name a message
whose blocks the view hides.

A session page reports its starting message position, selected message count,
content version and index generation. Its `cursor` reopens the resolved current
page, including the tail selected by Latest; `next` continues after it.
Document pages return their metadata, source descriptors, slices and `next`. Clients treat cursors as opaque. A cursor binds
content, view, parent inclusion and generation. Unrelated writes preserve it;
changed content or an index replacement returns `stale_cursor`. The reader
keeps an anchor in its continuation URL and offers Refresh from this message.
There is no automatic joining of pages across a stale cursor.

`/s/{id}` and `/d/{id}` render one content page and explicit First/Next links.
Latest is available for sessions; Previous uses a daemon-issued cursor and reverse key seeks to fill the page before the current start. Its Next link reopens the current page.
Changing parent inclusion opens the first page of the new selection. Changing the view starts a new read at the page's message anchor. `/m/{id}`
resolves a message's owning session and redirects to its anchor; `/n/{id}`
reads one exact revision through the shared `note read ID` operation, also
available as `/api/v1/note/read?id=ID` and over MCP. Superseded and withdrawn
revisions remain readable. The notes view can extend that route with history
and judgment controls.

Masking runs before view filtering and splitting. Direct anchors restore the
same private-key state as sequential reads, including state from selected
parents. Cursors contain hashes and positions, never private masker state.
No session body before an anchor is loaded unless a marked private-key message
is required to restore that state. Messages hidden by a view keep their metadata
positions; their bodies stay unread unless they hold PEM markers needed for
ordered masking. A session without parents uses indexed
message seeks even when parent inclusion is requested. Counts and anchor
positions still read metadata. A session page can decode a whole message,
and a document page can decode a whole block; the output budget does not cap
those allocations.

## Outline and relationships

`outline SESSION --page-size N [--with-parents] [--cursor CURSOR]` is shared by
CLI, MCP and HTTP (`/api/v1/outline`). It pages references to human prompts,
recorded tool failures, known branch points and compaction events anchored
after a message. It reads metadata, never message text. Entries contain
`message`, `session`, recorded `at`, `kinds` and compaction `events` references.
Its cursor is sealed against the selected content and index generation.

The HTML reader fetches the outline when opened (`outline=true`), and pages it with an order-key seek. It also pages the existing `self | links` query.
`outline_cursor`, `links_cursor` and the content `cursor` are independent URL
parameters. Failure of either optional panel is shown beside the content.

## Change batches

`GET /api/v1/changes?since=POSITION` streams the same finite JSON-lines response
as the socket's `changes` operation. It is declared in the command table;
the daemon spools each batch once for clients at the same position and releases
the reader before sending. Each response opens an independent reader of that shared spool. Slow clients hold only the response spool.
Every line is a response envelope with `result` and `data`. Only the final
`data.type = "complete"` advances the applied position. Retrying can repeat
references. `begin.generation_changed` and `begin.replacement` tell the view
to refresh its reference population after an index replacement.

`/assets/live.js` exports `watchChanges(subscriber)`, returning an unsubscribe
function. New subscribers receive current references through a paged query, without resetting the shared position. The optional `query` selects that population (by default current session, note and excision references); `snapshot(page)` receives each initial page, or `apply` receives its rows. Other optional callbacks are `begin`, `apply`, `complete`, `refresh` and `error`. Apply references idempotently; a failed or incomplete batch leaves
its last complete position unchanged. One finite request runs per visible tab.
Hidden tabs abort the request and timers. A 30-second refresh callback asks
the shared API about clock-driven state changes. A 401 stops requests and
asks the operator to reopen with `polylogue web`. This helper keeps only the
position in page memory and never executes commands.

## Authentication and local actions

The workbench uses the existing per-archive `HttpOnly; SameSite=Strict` cookie,
exchanged from the launcher's fragment at `/auth`; bearer headers remain
supported for existing callers. This is the accepted workbench authentication
contract and supersedes the epic's older sessionStorage/bearer-only criterion.
The cookie name distinguishes archives because cookies do not distinguish
ports. Credentials never enter query strings or change-feed positions.

All private routes use the same Host, uid and authentication checks. POSTs
also require the existing allowed Origin (or origin-less bearer rule).
Command POSTs require `application/json`; sign-in requires
`application/x-www-form-urlencoded`. Every HTTP response carries no-store,
no-referrer, nosniff and the existing restrictive CSP. Archived markup stays
inert, and reading content does not fetch remote previews.

The operator has allowed local reveal-in-file-manager and exact downloads of
stored bytes. The descriptor API still returns masked attachment information;
the foundation does not add those action controls. Sharing and per-span
credential reveal are outside this contract. Page pings and background triggers
may issue existing daemon requests only. Investigations use references for this
increment. The existing design boards need no review for this increment.

# Browser extension

The browser extension captures web chats (ChatGPT, claude.ai, Grok, Gemini, AI Studio, Codex Cloud) into the archive as the operator uses them, backfills their history, and puts Polylogue into the provider's own pages: per-message state and actions, archive search, recall and a popup. It also acts in web chats on request (chat actions). Where its captures go once stored is [archive.md](archive.md); every other capture route is [capture.md](capture.md); the web workbench is [web.md](web.md); deliveries and authorship are [agents.md](agents.md#deliveries) and [assertions.md](assertions.md#authorship); installing the native host is [operations.md](operations.md#installing-into-harnesses).

The extension is designed whole rather than adjusted. Ruling, 2026-07-09: "browser capture UI/UX just sucks. It doesn't need incremental adjustments, it needs thorough redesign" (cc-queued:3b0038ec-234c-44b8-bb88-fe222b44fe0f@2026-07-09T10:46:47Z).

## Live capture

Live capture is the core of the extension and works in about real time. Ruling, 2026-07-31: "live-append is the core thingg we do here, not some fluff. it MUST work properly, reliably, in ~realtime." (cc:38baa1de-9715-48fa-8175-f2a29d92800e:38ab559c-4d32-4cc3-8901-268b17eaed02).

| Site | Route |
|---|---|
| chatgpt.com, chat.openai.com (Codex task pages on chatgpt.com → origin `codex-cloud`) | the conversation API JSON, plus file downloads |
| claude.ai (Claude Design → origin `claude-design`) | the conversation API JSON with the full tree and all tool renderings |
| grok.com, x.com/i/grok | the conversation API JSON and responses |
| gemini.google.com | the page's own data responses, intercepted; reading the DOM is a fallback whose captures are marked *partial fidelity* and are dominated by any native route |
| aistudio.google.com | the prompt JSON the page loads (the same shape as Drive) |

The extension captures the provider's data, not its rendered page; a DOM route exists only where no data route covers the origin. Ruling, 2026-07-31: "if DOM is strictly dominated // you don't have any particular reason to believe it is ever actually necessary for a given origin, it should be removed entirely." (cc-queued:38baa1de-9715-48fa-8175-f2a29d92800e@2026-07-31T03:06:53Z).

- **Transport:** a Manifest V3 extension and the native-messaging host `polylogue browser-host`, with no token or pairing. Each capture is appended to a per-conversation capture log in the spool; the host answers only after the append is synced, naming the hash and whether it was new. An outbox of pending captures is the extension's only state.
- **When:** a capture fires when a conversation stops changing; an unchanged conversation is not captured again; open tabs are swept periodically. Capture, status and tab sync are automatic, never actions the operator takes.
- **Fidelity:** attachment and generated-file bytes (uploads, generated images, code-interpreter and sandbox files, file previews, provider assets) are fetched before their links expire, in chunks, with no size cap. Ruling, 2026-07-10: "please do fix thigns so that browser extension captures these things. I don't watnt to manually do it" (cc:3347cf34-ca12-45ae-918f-781c7f96a704:5c99f860-6e9d-4005-a224-9c08ae24bd40). Temporary chats are captured and marked as not saved by the provider. Projects, custom GPTs and Claude projects are captured with their conversations.
- **Politeness:** a conversation is fetched again only when its update time or message count changed; a 429 or 5xx stops all fetching from that provider for `Retry-After` or a jittered backoff.
- **Several browser instances** each carry an instance identity and never overwrite each other's captures.
- **Trust:** the extension trusts only responses it intercepted itself, never messages posted by the page, and acts only on the operator's real input and its own code.

## Backfill

Backfill brings in a provider's whole history in the background, independent of open tabs.
- `polylogue backfill start <provider> [--since DATE]` creates a job (an archive row); `backfill list` and `backfill cancel` manage jobs. The extension claims a job, pages through the provider's conversation list in a background tab, captures each conversation the archive lacks or holds in an older form, reports progress, and resumes after a restart.
- **Coverage:** ChatGPT's four partitions (active, starred, archived, archived-starred) with projects and custom GPTs; claude.ai within its organization, with Claude Design projects; Grok; Gemini and AI Studio through their list endpoints; Codex Cloud's task list.
- **Pacing:** one request at a time per provider, `Retry-After`, a daily budget and jittered backoff; a 403 pauses the job and raises a notice for the operator.
- A backfill requested by an agent carries the requester's computed author.

## In-page UI

The extension's main surface is inside the provider's page, blended with the provider's controls; the popup is secondary. Ruling, 2026-07-09: "the extension popup could be much bigger … there should be useful stuff injected into the page, likely - perhaps primarily so" (claude.ai:f5d2d5c9-0277-45d4-83bb-82821a96da0b:9d661ad9-5a52-4ab9-b2b8-32dc4b7a3a36). The in-page UI has no board yet and is drawn before it is built ([web.md](web.md#views-without-a-board)).

- **Per-message state and actions in the provider's own action row**, next to its copy button: the message's capture state, **Save to Polylogue** (capture now), **Annotate**, **Copy reference**, **Open in Polylogue** (the reader at that message), and the message's note count with a hover preview.
- **A summonable chip and panel:** a small chip on the page's edge, with a keyboard shortcut; the panel floats over the page and shows:
  - this conversation in the archive: state, reference, provider URL, notes, cost where known, lineage (resumed from, continued by);
  - **archive search in the page:** the query field with completion, results grouped by session, opening the reader;
  - **recall across conversations:** sessions and notes related to this conversation through references, shared pastes and the same subject query (and `related:` when that product is available);
  - **inspection:** the stored records of this conversation and their provenance;
  - **chat actions** in flight for this conversation: request, send and observed reply, kept apart.
- **Selection capture:** selecting text in a message offers **Note on selection** (a note on the exact span) and **Add to bundle**.
- **Mounting:** everything mounts in an isolated Shadow DOM, and nothing shifts the page's layout. One capture-state signal feeds every element. When the provider's page or Polylogue changes shape, the extension fails open: the provider's page keeps working and the extension says it is degraded.

## Popup

At least 640 × 600 px, laid out from the [popup board](web/boards/extension-popup.html) and extended:
- **This conversation:** title, state in plain words, counts, reference, tags; **Capture again**, **Annotate**, **Open in Polylogue**.
- **Recent activity:** the latest captured conversations across providers with time, title and state, linked both to the provider's page and to the reader.
- **Providers:** one row per provider with its state and, for *Needs attention*, what to do.
- **Chat actions:** drafted turns awaiting the operator's approval (**Approve**, **Decline**) and sends in flight.
- **Backfill:** jobs with progress.
- Footer: **Open the web reader**, pending captures, daemon connection.

## States

Each conversation is in one of six plain states, shown in the in-page indicator, the badge and the popup:

| State | Meaning |
|---|---|
| *Safe* | captured and stored |
| *Catching up* | captures pending |
| *Needs attention* | a problem the operator can fix, named |
| *Failed* | will be retried |
| *Not saved* | the provider does not keep it (a temporary chat) |
| *Partial fidelity* | stored from a weaker route and missing structure |

A view of a conversation that has not been captured says "Not captured yet".

## Provider URLs

A web-chat session's provider URL, when its origin records one, is a link in the reader header, the Running row, Home's recent sessions and the popup's recent activity. A provider URL pasted anywhere resolves to the archived session through the one [reference parser](references.md#rules).

## Chat actions

The operator or an agent asks the extension to draft or send a turn in a web chat. Direction, 2026-06-29: posting through the extension "might be really REALLY high leverage" (journal 2026-06-29 01:10).
- `polylogue chat send PROVIDER TEXT [--model M] [--conversation C] [--project P] [--attach FILE…] [--submit] [--wait DURATION]`, and the MCP `act` tool. Attachments are accepted over MCP too, because the server runs on the caller's machine.
- **Request, send and reply are separate facts.** Each action has a request ID, its requester (the request's computed author), the resulting conversation, and is a delivery to a recipient of kind `web_conversation`. It is shown as sent only when a capture contains the turn.
- A submit into an existing conversation waits for the operator's approval in the extension (`chat approve`, `chat decline`, also over MCP); the approval is authored by whoever gives it.
- `--wait` returns when ordinary capture stores the reply turn, or at the deadline with the action's state; the daemon waits on the capture event and never polls. The reply is read like any captured session.
- **Rename in the provider** is a chat action: the requester asks for a new title, the extension applies it in the provider, and the next capture confirms it.
- The contract is the same for every provider the extension can act on. Each provider's availability is declared by the extension, and an unsupported provider is refused with that reason.

## Authorship from the extension

Acts made in the extension's own UI (a selection note, approving or declining a drafted turn, capture now) are authored `operator` with `via: extension`. Acts the extension performs for a request (a chat send, a backfill, a rename) carry the requester's author from the request record. The rules for computing authors are [assertions.md](assertions.md#authorship).

## Laws

- **fetch-stops-on-rate-limit**: the extension's own fetch and scheduling code, run against synthetic provider responses for each site, stops all fetching from a provider for `Retry-After` after a 429.
- **large-attachment-chunked**: a 100 MB attachment is fetched in chunks and stored with a matching hash.
- **unchanged-tabs-add-nothing**: unchanged tabs add no spool bytes over two sweeps.
- **capture-acknowledged-after-sync**: the native host acknowledges a capture only after its append is synced; a crash after the acknowledgement loses nothing.
- **backfill-completes-with-backoff**: a synthetic backfill of 1,000 conversations in pages of 100, with a 429 at page 4, finishes with all 1,000 and honours the backoff; a cancelled job stops within one page.
- **instances-never-overwrite**: captures from two browser instances of the same conversation are both kept.
- **page-messages-ignored**: a message posted by the page's own scripts never produces a capture or an action.
- **in-page-ui-keeps-layout**: the in-page elements mount in a Shadow DOM on a fixture provider page without changing the bounding boxes of the provider's elements, and fail open when the page's structure changes.
- **chat-action-author-is-requester**: a `chat send` requested over MCP by session S records S as the action's author; the operator's approval is recorded with the operator as its author.
- **sent-only-when-captured**: a chat action is reported sent only after a capture contains the turn.

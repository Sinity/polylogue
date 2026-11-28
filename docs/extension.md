# Browser extension

The browser extension captures web chats (ChatGPT, claude.ai, Grok, Gemini, AI Studio, Codex Cloud) into the archive as the operator uses them, imports their history on request, and puts Polylogue into the provider's own pages: per-message state and actions, archive search, related conversations and a popup. It also writes turns in web chats and renames conversations on request (chat actions). Where its captures go once stored is [archive.md](archive.md); every other capture route is [capture.md](capture.md); the web workbench is [web.md](web.md); deliveries and authorship are [agent-context.md#deliveries](agent-context.md#deliveries) and [authorship.md](authorship.md); installing the messaging host (`polylogue install chrome`) is [installing.md](installing.md#installing).

The extension is designed whole rather than adjusted.

## Installation and updates

The packaged manifest's public key gives Chrome the same extension ID at every path. The package publishes that ID at `share/polylogue/extension-id`. Its manifest's `version_name` names the source revision, which the extension writes into capture provenance. The native host reports its own revision in a status reply. When both revisions are known and differ, the extension reloads at most once for the pair, then installs the page scripts in the supported tabs already open. Keep the old installation available until its popup shows zero pending captures when moving from an installation with a different ID.

## Live capture

Live capture is the core of the extension and works in about real time.

| Site | Route |
|---|---|
| chatgpt.com (Codex task pages on chatgpt.com → origin `codex-cloud`) | the conversation API JSON, plus file downloads |
| claude.ai (Claude Design → origin `claude-design`) | the conversation API JSON with the full tree and all tool renderings |
| grok.com, x.com/i/grok | the conversation API JSON and responses |
| gemini.google.com | the page's own data responses, intercepted; reading the DOM is a fallback whose captures are marked *partial fidelity* and are outranked by any route through the page's data |
| aistudio.google.com | the prompt JSON the page loads (the same shape as Drive) |

The extension captures the provider's data, not its rendered page; a DOM route exists only where no data route covers the origin.

- **Transport:** a Manifest V3 extension and the native-messaging host `polylogue internal browser-host`, with no token or pairing. Each capture is appended to a per-conversation capture log in the spool; the host replies only after the append is synced, naming the hash and whether it was new. The extension's outbox holds pending capture bytes; browser-local fingerprints and pacing state are disposable scheduling metadata.
- **When:** a capture fires when a conversation stops changing; an unchanged conversation is not captured again; open tabs are swept periodically, and a sweep fetches a tab's conversation only when its page shows something other than it showed at the conversation's last capture. A capture the operator asks for always fetches. Capture, status and tab sync are automatic, never actions the operator takes.
- **ChatGPT changes:** the page signals a change when its URL or the ordered message IDs and complete rendered message text change. Opening a page menu outside the messages leaves that signal alone. The extension fetches the conversation JSON after a change and compares its content with the last capture. For changes that do not affect the rendered messages, such as a title edit, provider import fetches a new reading when the conversation list reports a later update time.
- **Fidelity:** attachment and generated-file bytes (uploads, generated images, code-interpreter and sandbox files, file previews, provider assets) are fetched before their links expire, in chunks, with no size cap. Temporary chats are captured and marked as not saved by the provider: the capture's `session_kind` is `temporary`, and the parsed session's mode is `temporary`. Projects, custom GPTs and Claude projects are captured with their conversations.
- **Politeness:** a conversation is fetched again only when it changed: for an import, its update time in the provider's list is newer than the archive's capture; for an open tab, its page shows a change; a 429 or 5xx stops all fetching from that provider for `Retry-After` or a jittered backoff.
- One request queue per provider is shared by automatic and explicit capture, file retrieval, provider import and chat actions. Retry-After accepts seconds or an HTTP date; retryable responses end the current provider operation before another fetch. Cooldown and background budget counters survive service-worker and browser restarts in browser-local storage; neither holds captures for the archive.
- **Several browser instances** each carry an instance identity and never overwrite each other's captures.
- **Trust:** the extension trusts only responses it intercepted itself, never messages posted by the page, and acts only on the operator's real input and its own code.

## Importing a provider

`polylogue import PROVIDER` brings in a web chat's whole history in the background, independent of open tabs: every conversation the archive lacks or holds in an older form.
- `PROVIDER` is `chatgpt`, `claude-ai` or `grok` ([interfaces.md#commands](interfaces.md#commands)); `--since TIME` takes only conversations changed since then (`polylogue import chatgpt --since 2026-01-01`). Each provider has one **import job** (an archive row): starting the import again while its job runs shows that job, and `polylogue import chatgpt --cancel` stops it. `polylogue import_jobs` lists the jobs with their progress (the unit `import_jobs`). A file literally named like a provider is written `./chatgpt`.
- **Waiting for it.** `polylogue import chatgpt --wait 10m` returns when the job finishes or the time passes, with the job's progress; one that ran out of time replies `empty` with the same line as its next line ([interfaces.md](interfaces.md#steps); over MCP, [agents.md](agents.md#waiting)).
- The extension claims a job through `polylogue internal import claim`, pages through the provider's conversation list in a background tab, captures each conversation the archive lacks or holds in an older form, reports progress through `polylogue internal import report`, and resumes after a restart. The archive holds a conversation in its current form when the daemon has read a capture taken at or after the provider's update time. A conversation that is only *spooled* (a capture of it is safe in the spool and not yet read) is of no known version, so the import fetches it again and stores it only when its content changed.
- **Coverage:** ChatGPT's four partitions (active, starred, archived, archived-starred) with projects and custom GPTs; claude.ai within the organization the page has selected, with Claude Design projects (with none selected, the page is treated as signed out and nothing is captured); Grok.
- **Pacing:** one request at a time per provider, `Retry-After`, a daily budget and jittered backoff; a 401 or 403 (the provider refusing the browser's sign-in) pauses the job: it keeps its saved position and totals, is not claimed again, and its detail, shown by `import_jobs`, the popup and the extension's problem list, asks the operator to sign in and start the import again, which resumes the job from that position. The default background budget is 10,000 background provider HTTP requests, including authentication, conversation, download-link, attachment and retry requests per provider per UTC day. Exhaustion yields the saved job position for a later claim; it never discards inputs or limits live capture. The scheduler accepts a budget explicitly; its default applies when none is supplied. Each HTTP fetch obtains a permit and saves its charge before it starts.
- A `--since` cutoff ends only the current newest-first partition. ChatGPT advances to the next of its four partitions before completing the job. Progress checks include unchanged conversations, so cancellation is observed within a page. A report names the start of the current page and counts only finished pages, so a resumed job lists that page again and counts each conversation once.
- An import requested by an agent carries the requester's computed author. Over MCP it is the `import` operation of `polylogue_act` with `provider` ([agents.md#mcp-server](agents.md#mcp-server)).

## In-page UI

The extension's main surface is inside the provider's page, blended with the provider's controls; the popup is secondary. The in-page UI has no board yet and is drawn before it is built ([web-design.md#views-without-a-board](web-design.md#views-without-a-board)).

- **Per-message state and actions in the provider's own action row**, next to its copy button: the message's capture state, **Save to Polylogue** (capture now), **Annotate**, **Copy reference**, **Open in Polylogue** (the reader at that message), and the message's note count with a hover preview.
- **A summonable chip and panel:** a small chip on the page's edge, with a keyboard shortcut; the panel floats over the page and shows:
  - this conversation in the archive: state, reference, provider URL, notes, cost where known, lineage (resumed from, continued by);
  - **archive search in the page:** the query field with completion, results grouped by session, opening the reader;
  - **related conversations:** sessions and notes related to this conversation through references, shared pastes and the same subject query (and `related:` when that product is available);
  - **inspection:** the stored records of this conversation and their source;
  - **chat actions** in flight for this conversation: request, send and observed reply, kept apart.
- **Selection capture:** selecting text in a message offers **Note on selection** (a note on the exact span) and **Add to bundle**.
- **Mounting:** everything mounts in an isolated Shadow DOM, and nothing shifts the page's layout. One capture-state signal feeds every element. When the provider's page or Polylogue changes shape, the extension fails open: the provider's page keeps working and the extension says it is degraded.

## Popup

At least 640 × 600 px, laid out from the [popup board](web/boards/extension-popup.html) and extended:
- **This conversation:** title, state in plain words, counts, reference, tags; **Capture again**, **Annotate**, **Open in Polylogue**.
- **Recent activity:** the latest captured conversations across providers with time, title and state, linked both to the provider's page and to the reader.
- **Providers:** one row per provider with its state and, for *Needs attention*, what to do.
- **Chat actions:** held turns and held new conversations awaiting the operator's approval (**Allow**, **Deny**), drafted new conversations, renames, and sends in flight.
- **Imports:** provider import jobs with progress.
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

A view of a conversation that has not been captured says "Not captured yet". A conversation the archive or the spool holds is shown as held; a later capture of it that failed is reported on a line under that state, never in its place. A messaging host that is installed and fails to answer is reported as Polylogue not answering, in the panel and the popup alike.

## Provider URLs

A web-chat session's provider URL, when its origin records one, is a link in the reader header, the Running row, Home's recent sessions and the popup's recent activity. A provider URL pasted anywhere resolves to the archived session through the one [reference parser](references.md#rules).

## Chat actions

A chat action is a turn the extension writes in a web chat at a user's request; the unit `chat_actions` lists them with what happened to each (`polylogue chat_actions`). Acting in web chats needs the setting `browser.actions on`; with it off, every chat action is refused with the line that turns it on.
- **A turn into an existing conversation** is `send` on its web-chat session, or on a pasted conversation URL not yet archived: `polylogue @2 send 'and the same for the 2023 export'` ([session-control.md#send](session-control.md#send)). The turn is held for the operator's approval, and the extension writes and sends it once it is allowed; the session is then waiting, and appears in `polylogue waiting`. `polylogue @2 answer allow` releases it, `polylogue @2 answer deny` cancels it, and the popup's **Allow** and **Deny** do the same ([session-control.md#answer](session-control.md#answer)). The first answer wins and a late one is refused with what won. Over MCP, `answer` on a held turn is refused ([session-control.md#answer](session-control.md#answer) gives the rule). The approval is authored by whoever gives it.
- **A new conversation** is `start --in chatgpt|claude-ai|grok`: `polylogue start 'Summarise the attached notes' --in claude-ai --attach notes.md`. The extension drafts it in a new conversation and leaves it there. `--model`, `--effort` and `--project` choose what the provider's menus offer; without `--model` the page keeps the model it has selected, and `--effort`, chosen in the model's menu, needs `--model` ([session-control.md#start](session-control.md#start)). A turn into an existing conversation keeps that conversation's model.
- **Submitting a new conversation.** `polylogue start 'Summarise the attached notes' --in claude-ai --attach notes.md --submit` has the extension send the drafted turn when the caller is the operator. When any other author asks for `--submit` (an agent over MCP or a shell), the new conversation is held like a turn into an existing conversation: the new web-chat session waits, and the operator's `answer allow` or the popup's **Allow** submits it (default; open operator question).
- **Renaming a conversation in the provider.** The extension carries out the rename that `note … --as title --rename` queues ([annotations.md](annotations.md#note)), an act recorded as a chat action with its outcome.
- `--attach FILE` (repeatable) attaches files to a turn or a new conversation; over MCP they are paths on the daemon's host, under the rule `import` uses for paths.
- **Request, send and reply are separate facts.** Each action has a request ID, its requester (the request's computed author), the resulting conversation, and is a delivery to a recipient of kind `web_conversation`. It is shown as sent only after the host acknowledges a capture whose provider conversation JSON contains the returned user-turn ID and requested text. A missing or undelivered capture leaves the outcome unknown and the action tab open; it never triggers a second send. The reply is captured like any turn and read on the session (`read`, or `read --follow` while it arrives).
- **The extension's host rows.** The extension claims the oldest queued action (`polylogue internal chat claim`), reports its phases and outcome (`internal chat report`), gives the popup's Allow or Deny as the operator's answer (`internal chat answer`, the same answer as `answer allow|deny`) and fetches an attachment's bytes (`internal chat attachment`). These rows are hidden from help and completion ([interfaces.md#operation-table](interfaces.md#operation-table)).
- **Claim expiry:** an expired pre-send claim returns to the queue. Before entering the submitting page action, the extension reports phase `sending` and saves it; later phases and renewals cannot erase it. Expiry then leaves `outcome_unknown`, with no permission to renew, act or claim another send. The original sender may resolve it with a late `submitted` report carrying a nonempty conversation and turn, matching the requested conversation when specified. This is the trusted extension's captured-turn report, not an independent check against the archive index; capture acknowledgment can precede indexing.
- The contract is the same for every provider the extension can act on. Each provider's availability is declared by the extension, and an unsupported provider is refused with that reason.

## Authorship from the extension

Acts made in the extension's own UI (a selection note, allowing or denying a held turn, capture now) are authored `operator` with `via: extension`. Acts the extension performs for a request (a turn sent or drafted in a web chat, a rename, a provider import) carry the requester's author from the request record. The rules for computing authors are [authorship.md](authorship.md).

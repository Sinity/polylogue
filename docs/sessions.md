# Sessions

The parsed model every other area reads: sessions, messages and blocks, message authorship, times and their quality, links between sessions, compaction boundaries, attachments, pastes, sidecars, one conversation arriving by several routes, and logical sessions. How these rows are produced from stored bytes is in [archive.md](archive.md#parsing); how they are rendered is in [reading.md](reading.md). Authorship of the user's own records (notes, judgments, pages) is a different thing, in [assertions.md](assertions.md#authorship).

## Sessions

- **A session ID is `origin:native`**, with qualifiers for sub-sessions and installs (a Claude subagent, a sidechain, a Hermes profile); the qualifier spellings are in [vocabulary.md](vocabulary.md).
- **Session kinds:** `main`, `subagent`, `sidechain`, including the older Claude sidechain layout.
- **A pasted session ID or provider URL resolves to the archived session** in every surface, hooks included, through the one reference parser ([references.md](references.md#rules)).
- **Documents.** Text that is not a conversation (memory files, plans, workflow scripts, brain artifacts, shell snapshots) is a document, versioned like any parsed row and attached to its session where it has one.

## Messages

- **A message is stored once, in the session that originated it.** A child that the provider wrote with a copy of its parent's prefix stores only its own messages and points at the parent through a link with `copies_prefix` ([Session links](#session-links)); copies are recognized from the child's own evidence. Ruling, 2026-06-27: "the forked sessions could just _not have these messages themselves_" (cc:269c34a5-a94b-4ae3-b9ea-a2afea3b4241:527568f9-4b9b-4f57-9346-b097772bbda6).
- **Order is source position** (input, sequence, record position), never time.
- **Branches.** Messages carry their parent message, which gives the branch tree; the active path comes from the provider's marked leaf. An unknown path is NULL and is never collapsed into "off the path".
- **Blocks.** A message is a sequence of blocks: text, thinking, tool calls and results, attachments, and the special output parts providers produce (artifacts, canvas, projects, citations), which are parsed as blocks and documents of their message rather than left as kept attributes.
- **Thinking** keeps its text, or its encrypted or redacted state, and its word count.
- **Tool calls and results are paired by the stored call and result relation**, never by serialization order. A tool call records its action (the origin's tool mapped to a closed set), target, input, exit code, outcome (`ok`, `error`, or `unknown` with a reason; `pending` is computed at read) and the time of its result. Source-reported elapsed milliseconds are stored separately from the difference between the call and result times.

## Message authorship

Who wrote a message is separate from its role, in two closed sets and no third field:

| Column | Values |
|---|---|
| `written_by` | `human` (the user), `model`, `harness`, `tool` (rows that only carry tool results), `unknown` |
| `prompt_kind` (user-role rows) | `prose`, `command` (a typed slash command), `injected_context` (hook output, skill text, a delivered brief), `compaction_summary`, `harness` (other prompts the harness wrote), `model` (a prompt a model wrote, such as a delegated task) |

- The user role is not the human: harnesses and models write user-role rows too, and the two columns say which. A human prompt and a typed command are counted apart.
- An `injected_context` row carries the delivery reference when its text matches a recorded delivery ([agents.md](agents.md#deliveries)).
- Ruling, 2026-06-29, against a third authorship field: "some weird stopgap … the solution is not MORE complexity" (cc:51696878-0e44-4543-ab23-92abe9fe5c19:9c8d7182-aaac-4e0b-9606-1b3afa2e2c83).

## Times

- **Every time Polylogue reads is stored with its quality:** `at`, `at_quality`, `at_precision`, and `at_until` when the precision is a bound.
- `at_quality` uses the same closed set as Sinex: `intrinsic` (the source recorded the time of the thing), `captured` (Polylogue saw it live and took the time at capture), `file_mtime` (only a file's modification time is known), `operator` (stated by the user), `unknown`.
- `at_precision` is `instant` through `year`, or `bounded` with `at_until`.
- Provider and commit times are `intrinsic`. Drive and ZIP member times are `file_mtime` (Drive bounded by revision, ZIP to the second). An offset-less local time is `intrinsic` with the zone it was read in recorded. A time the source does not give is `unknown`, with no instant: nothing is placed at 1970, at midnight or at import time.
- A file's modification time is never presented as a message's own time. Queries treat coarse and estimated times by the shared coarse-times rule ([query.md](query.md#units-and-fields)).
- The index and every `polylogue.*` record sent to Sinex carry all three fields.

## Session links

Links between sessions come only from explicit markers in the sources, in one closed set of kinds; text similarity never creates a link.

| `kind` | Meaning | Typical evidence |
|---|---|---|
| `spawn` | a subagent or delegated session started by a call in the parent | Claude Task or Agent `toolUseID` and the subagent file; Codex `agent_path`; Grok subagent files |
| `fork` | a session sharing its parent's prefix up to a branch point | Claude `--fork-session`; Codex fork records |
| `continuation` | the same work continued in a new session after the parent's context was compacted or exhausted | the provider's continuation marker |
| `resume` | a provider resume of an earlier session in a new session | Claude `resumeSessionId`; Codex resume records |
| `reference` | a relationship the provider did not make, with `how`: `briefing` (a delivery's recipient), `ping` (a ping's answer) or `mention` (a pasted ID or URL in a message) | delivery rows, ping rows, the reference parser |

- **`copies_prefix`** is a separate boolean: whether the child's file holds copies of the parent's messages. It never decides the kind.
- **`branch_point`** names the parent message the child continues from; **`spawning_call`** names the parent's call block for a `spawn`.
- **`retry_of`** (a later attempt at the same delegated task) and **`result_use`** (`returned`, `referenced` or `unknown`: how the parent used the child's result) come from recorded structure only.
- **Evidence** names the field path that produced each link.
- A delivery creates a `reference` link with `how: briefing`, never a provider link.
- **Pending links.** A link whose other end is not in the archive yet is stored with that end unresolved, and resolves when it arrives, whatever the arrival order. Cycles and conflicting parents are reported at read, never resolved by choosing one link.
- Claude `agent-acompact-*` files are two different things. A replay of the main session ending in a summary is a compaction of the main session ([Compaction boundaries](#compaction-boundaries)), and its file is a sidecar of that session. A Task subagent's own compaction is a `spawn` child with `copies_prefix = false`. The parser tells them apart by the share of the file's messages that are main-session messages, and records which rule applied.

## Compaction boundaries

A compaction is a discontinuity in the model's context, not a special kind of content. Ruling, 2026-06-27: the summary "just is the message in context which replaces the previous prefix" (cc:269c34a5-a94b-4ae3-b9ea-a2afea3b4241:4eefdb12-1487-43b4-a0cc-5fc4786372b0).

- **The summary is a message** at the boundary, with `written_by: harness` and `prompt_kind: compaction_summary`. It is searchable, embedded like other prose, and counted once.
- **The boundary is an event linked to that message.** It records the tokens before and after (as reported), the trigger (`manual`, `auto`, or the harness's own value), the range of messages it evicted and what it kept. Codex `replacement_history` stays in this event and is not turned into messages.
- Encodings mapped onto it: Claude Code inline `isCompactSummary` records, main-session `agent-acompact` replays, Codex `compacted` records, claude.ai inline summaries. Providers without compaction (ChatGPT, Gemini, Antigravity) have none.
- **Microcompaction** (named tool results removed from the context) is an event naming the removed results.
- Messages before the boundary stay stored and readable; they are no longer in the model's context. Composing the context as the model saw it at a message is a read ([reading.md](reading.md#views)).

## Attachments

An attachment is a stored entity, not a flag on a block:

| Field | Values |
|---|---|
| `message`, `block`, `position` | where in the conversation it appeared |
| `supplied_by` | `operator` (an upload), `model` (a file the model produced for the user), `tool` (media a tool returned), `unknown` |
| `name`, `media_type`, `size` | as recorded |
| `availability` | `present` (bytes stored, hash matches), `unfetched`, `failed` (with the reason), `unavailable_at_source` |
| `blob` | the bytes' hash, when present |
| `source_ref` | the provider's file ID, Drive ID, path or URL |

- **"Preserved" means stored bytes whose hash matches.** Anything else states its availability.
- Who supplied a file matters: an upload and a file the model generated for download are different things. Ruling, 2026-08-31: "do ensure we do model attachments properly … there are actual user attachments … but there are also stuff that model generates" (cc-queued:62b956ba-3ea3-4693-8d71-9a36321c37a9@2026-08-31T05:20:13.932Z).
- **Every route delivers the bytes** when the source has them:

  | Origin | Source of the bytes |
  |---|---|
  | ChatGPT export | `.dat` and named members, by file ID and the export's name maps; sandbox links through a resolver ported from the predecessor; generated-image metadata |
  | ChatGPT, claude.ai, Grok, Gemini and AI Studio pages | bytes the extension fetched ([extension.md](extension.md#live-capture)) |
  | claude.ai export | `extracted_content` as a document (the export has no bytes); bytes from captures |
  | AI Studio | inline base64; `driveDocument` and `driveImage` fetched by Drive ID ([capture.md](capture.md#ai-studio-through-drive)) |
  | Claude Code | inline base64 images, `tool_result` media, `paste-cache` |
  | Codex | inline `data:` images; `local_images` files, read when the rollout line is observed, because they are screenshots in paths that disappear. This is the one read outside a root, limited to paths a parsed record names; a missing file is `unavailable_at_source` |
  | Hermes | `pastes/`, `image_cache/` |

- In a ChatGPT export, an attachment is matched to the stored export members by the native file ID and the export's naming and library sidecars; a file ID can match without a name, and several distinct matching files stay separate matches.
- `polylogue block:<message>#<n> read --to FILE` works for every origin and writes the exact bytes for any caller, through a temporary file that replaces `FILE` only once the length and hash agree ([reading.md](reading.md#writing-out)).

## Pastes

- Paste spans come from transcript markers, `history.jsonl` `pastedContents` (exact text or a hash only), `UserPromptSubmit` hook payloads, and `paste-cache/*.txt` bodies for hash-only references. They are never found by pattern-matching text.
- Each span has a boundary state: `exact`, `projected`, `whole_message_fallback` or `hash_only`.
- Polylogue owns the span; in primary mode Sinex owns the join between a clipboard event and a paste.

## Sidecars and linked materials

- **A sidecar is read into its session, never as a session:** `.meta.json` files, tool-result overflow files, Gemini tool outputs, Grok terminal logs, workflow journals, compaction replays. Ruling, 2026-09-07, on sidecars parsed as sessions: "so these clearly should not be 'sessions'" (cc-queued:73b121c4-d81f-45f0-a42b-883effbfd80a@2026-09-07T01:23:23.574Z).
- **Linked materials.** Every URL a conversation cites or an agent produces is recorded as a material observation of its message. Provider-hosted files (download links, sandbox outputs, file previews) are fetched at capture as attachments. Web pages a conversation cites are fetched too, because cited pages disappear; a setting turns this off. Every fetch goes through a guard that refuses private and loopback addresses and unsafe redirects.

## One conversation through several routes

A conversation can arrive as a live file, a browser capture, an export, a Drive revision, a Takeout and an imported old copy. This redundancy is deliberate. All routes arrive as `<origin>:<native id>` and are one session; every route's reading is stored, and the shown message is composed field by field. Ruling, 2026-07-31: "maybe 'latest wins' was sloppy idea altogether" (cc:38baa1de-9715-48fa-8175-f2a29d92800e:a36102a7-666b-480b-8f62-f282494b1932).

1. A field present in one route and absent in another is taken from where it is present.
2. Where several routes have the field, the origin's declared **route precedence for that field** decides: for example, message text from the provider's API data over the export over a page scrape; provider timestamps from the export when the capture has none; attachment bytes from whichever route stored them.
3. A finished reading wins over an in-progress one of the same route.
4. The last tie-break is the content hash, so the result never depends on arrival order or recency.

- The composition records, for each shown field, the route it came from. Each origin declares its precedence table beside its parser.
- **Routes without shared turn IDs** (a Gemini page capture and a Takeout of the same conversation) are matched by the repeated-turn rule; the matched message names the shown one (`duplicate_of`). That rule and its validity evidence are in [measures.md](measures.md#catalog).
- **A capture read from the page** rather than the provider's data is marked *partial fidelity* and is outranked by any other route for every field.

## Logical sessions

The **logical session** is the default unit of work: the root of a tree of `spawn`, `fork`, `continuation` and `resume` links, so one piece of work spread over subagents, forks, compactions and resumes reads as one thing. `reference` links never join it. Ruling, 2026-10-28: "do what makes sense UX" (cc-queued:ee52794c-8a9e-43b6-9fbc-9749c7e78e8f@2026-10-27T23:54:48.743Z).

- Listings, counts and measures show logical sessions, each expandable into its physical sessions; `--physical` lists sessions one by one ([query.md](query.md#units-and-fields)).
- References always name physical sessions exactly.
- The stored root and depth are a derived product ([measures.md](measures.md#catalog)).

## Laws

- **message-stored-once**: a resume or fork child with a copied prefix stores only its own messages and a link with `copies_prefix = true`.
- **link-kinds-exact**: fixtures for `spawn`, `fork`, `continuation`, `resume` and each `reference` `how` produce exactly that kind; a Claude `resumeSessionId` is `resume`, never `continuation`; a delivery produces `reference` with `how: briefing`.
- **pending-link-resolves**: a pending link resolves when its other end arrives, whatever the arrival order.
- **acompact-classified**: a main-session compaction replay becomes a compaction of the main session; a Task subagent's own compaction becomes a `spawn` child.
- **compaction-summary-is-message**: a compaction summary is a message with `prompt_kind: compaction_summary`, found by full-text search and counted once; its boundary event records tokens before and after, the trigger, the evicted range and what was kept.
- **every-time-has-quality**: every stored time has a quality; a ZIP member time is `file_mtime`; a missing time has no instant.
- **authorship-columns**: a tool-result row is `written_by: tool`; injected hook text is `injected_context`; a delegated task prompt is `prompt_kind: model`.
- **attachment-bytes-present**: an attachment fixture for each row of the origin table has `availability: present` with a matching hash; a missing `local_images` file is `unavailable_at_source`.
- **paste-resolves-through-cache**: a hash-only paste resolves through `paste-cache` with state `exact`.
- **sidecar-not-session**: a sidecar file adds to its session and creates no session.
- **routes-merge-by-field**: a capture plus two exports of one ChatGPT conversation give one session whose every field comes from its declared route; every arrival order gives identical shown rows; a page-scraped capture never overrides a field another route has.
- **upstream-gone-kept**: a conversation missing from a complete newer export is marked `source: gone` and keeps its rows.
- **logical-session-tree**: a logical session contains exactly the sessions joined by `spawn`, `fork`, `continuation` and `resume` links, and none joined only by `reference`.

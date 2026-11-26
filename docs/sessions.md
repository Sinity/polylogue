# Sessions

The parsed model every other area reads: sessions and what identifies them, messages and blocks, names and references, message authorship, times and their quality, links between sessions, compaction boundaries, attachments, pastes, sidecars, one conversation arriving by several routes, and logical sessions. How these rows are produced from stored bytes is in [archive.md](archive.md#parsing); how they are read and rendered is in [reading.md](reading.md) and [transcripts.md](transcripts.md). Authorship of the user's own records (notes, tags, judgments) is a different thing, in [authorship.md](authorship.md).

## Sessions

- **A message ID is `origin:native`** with the source's own ID where it gives one, otherwise a content ID ([archive.md](archive.md#readings)); authored records name messages by these IDs, so an unchanged message keeps its notes through a rewrite (for repeated equal messages, see the numbering rule there).
- **A session ID is `origin:native`**, with qualifiers for sub-sessions and installs (a Claude subagent, a sidechain, a Hermes profile); a Hermes profile is spelled `<native>@profile-<name>` ([capture.md](capture.md#live-origins)).
- **A session is everything that states its ID.** A parser states a session by the ID it forms from the record; any number of inputs, on any routes, may state the same ID, and their rows are one session ([One conversation through several routes](#one-conversation-through-several-routes)). No input is part of a session's identity, and a session exists for as long as some reading states it.
- **A session may have other names.** A session can record further IDs as its *aliases*: each Drive revision of an AI Studio prompt records the content ID that a downloaded copy of that revision would carry. An ID that exactly one session records as an alias names that session; an ID that no session or more than one session records names itself. Which it is, is decided from the alias rows present each time the sessions are composed, so a Drive prompt that arrives before or after its copies, or a second prompt that later records the same name, gives the rows a build from the same inputs gives. A content-named message of an aliased session takes the session's own ID in its name, and a reference by an alias resolves to the session it names.
- **Session kinds:** `main`, `subagent`, `sidechain`, including the older Claude sidechain layout.
- **A pasted session ID or provider URL resolves to the archived session** in every surface, hooks included, through the one reference parser ([references.md](references.md#rules)).
- **Documents.** Text that is not a conversation (memory files, plans, workflow scripts, brain artifacts, shell snapshots) is a document, versioned like any parsed row and attached to its session where it has one.

## Messages

- **A message is stored once, in the session that originated it.** A child that the provider wrote with a copy of its parent's prefix stores only its own messages. Transcript reads and stored message selections assemble the same copied members in the same order. Copied-prefix observations keep the copied messages and their source order independently of provider relationships ([Session links](#session-links)); copies are recognized from the child's own records.
- **Order.** A session's messages read in source-time order across inputs and in source position within one input. Each chain's reading gives a message an *order time* where the chain first states it: the latest source time among that message and the messages the chain stated before it in the same session. Order times therefore never decrease along a chain, a message without a time stays right after the message before it, and reading a chain in one step or in many gives the same order times. A record that restates its whole conversation (a capture frame, an export item that repeats its conversation) starts again from its own first message, and the messages it restates take their order times from it. A message is placed by the earliest (order time, input ID, position in that input) among the inputs' readings of it that have an order time, so it stays before what follows it in each of its inputs, and a reading without one never pulls it to the start; a message with no time in or before it in any of its readings comes at the session's start, in its input's order. Messages of different inputs at the same order time keep input ID order, which can put a turn before the turn that precedes it in another input only when their times are equal. The message ID breaks a remaining tie. Placement is a function of the readings and never of arrival order.
- **Branches.** Messages carry their parent message by the name the source gave it ([Names and references](#names-and-references)), which gives the branch tree; the active path comes from the provider's marked leaf. An unknown path is NULL and is never collapsed into "off the path".
- **Blocks.** A message is a sequence of blocks: text, thinking, tool calls and results, attachments, and the special output parts providers produce (artifacts, canvas, projects, citations), which are parsed as blocks and documents of their message rather than left as kept attributes.
- **Thinking** keeps its text, or its encrypted or redacted state, and its word count.
- **Tool calls and results are paired by the stored call and result relation**, never by serialization order. A tool call records its action (the origin's tool mapped to a closed set), target, input, exit code, outcome (`ok`, `error`, or `unknown` with a reason; `pending` is computed at read) and the time of its result. A duration the source reports is stored separately from the difference between the call and result times.

## Names and references

A row refers to another thing by the name its source used, and the name is joined to the thing afterwards. Nothing is written into a row because the thing it names arrived.

- **Within a chain.** Some names only make sense inside one file: the key that ties the records of one response together, the number that tells equal content-named messages apart. They are settled when the chain's reading is formed, from that chain's records alone.
- **Within a session.** A parent, a tool call and the message an event follows are joined when the session is composed, from the rows of every input that carries it. A record that is not a message but stands in a chain of parents yields an *alias row* (this name stands for that one); a parent named through aliases is the nearest message the aliases lead to. An alias or a parent that another input supplies resolves the same whichever input arrived first.
- **Across sessions.** The ends of a link, a branch point and a spawning call are kept as IDs and joined when read.
- **A name whose thing is absent is unresolved, and says so.** It is never dropped, never guessed and never pointed at something similar. When the thing arrives, from any input in any order, the name resolves.

## Message authorship

Who wrote a message is separate from its role, in two closed sets and no third field:

| Column | Values |
|---|---|
| `written_by` | `human` (the user), `model`, `harness`, `tool` (rows that only carry tool results), `unknown` |
| `prompt_kind` (user-role rows; `harness` also marks runtime notices carried under another role) | `prose`, `command` (a slash command, shell escape or memory line), `injected_context` (hook output, skill text, a delivered brief), `compaction_summary`, `harness` (other text the harness wrote: tool results, interruption notices, local command output), `model` (a prompt a model wrote, such as a delegated task) |

- The user role is not the human: harnesses and models write user-role rows too, and the two columns say which. A human prompt and a typed command are counted apart.
- An `injected_context` row carries the delivery reference when its text matches a recorded delivery ([agent-context.md#deliveries](agent-context.md#deliveries)).
- There is no third authorship field; the two columns above are the whole model.

## Times

- **Every time Polylogue reads is stored with its quality:** `at`, `at_quality`, `at_precision`, and `at_until` when the precision is a bound.
- `at_quality` uses the same closed set as Sinex: `intrinsic` (the source recorded the time of the thing), `captured` (Polylogue saw it live and took the time at capture), `file_mtime` (only a file's modification time is known), `operator` (stated by the user), `unknown`.
- `at_precision` is `instant` through `year`, or `bounded` with `at_until`.
- Provider and commit times are `intrinsic`. Drive and ZIP member times are `file_mtime` (Drive bounded by revision, ZIP to the second). An offset-less local time is `intrinsic` with the zone it was read in recorded. A time the source does not give is `unknown`, with no instant: nothing is placed at 1970, at midnight or at import time.
- A file's modification time is never presented as a message's own time. Queries treat coarse and estimated times by the shared coarse-times rule ([query.md](query.md#times)).
- The index and every `polylogue.*` record sent to Sinex carry all three fields.

## Session links

Links between sessions come only from explicit markers in the sources, in one closed set of kinds; text similarity never creates a link.

| `kind` | Meaning | Typical evidence |
|---|---|---|
| `spawn` | a subagent or delegated session started by a call in the parent | Claude Task or Agent `toolUseID` and the subagent file; Codex `agent_path`; Grok subagent files |
| `fork` | a session sharing its parent's prefix up to a branch point | Claude `--fork-session`; Codex fork records |
| `continuation` | the same work continued in a new session after the parent's context was compacted or exhausted | the provider's continuation marker |
| `resume` | a provider resume of an earlier session in a new session | Claude `resumeSessionId`; Codex resume records |
| `reference` | a relationship the provider did not make, with `how`: `briefing` (a delivery's recipient), `send` (a session resumed by a sent message) or `mention` (a pasted ID or URL in a message) | delivery rows, the reference parser |

- **`copies_prefix`** is a separate boolean: whether the child's file holds copies of the parent's messages. It never decides the kind. A copied-prefix observation creates no relationship on its own; it contributes this fact when an explicit relationship is recorded, whichever arrives first. The copied messages remain readable even when there is no provider relationship. Portable readings stream each copied message's child, original source reading and order with that message, so importing a parent's later messages never extends the child's copied history.
- **`branch_point`** names the parent message the child continues from; **`spawning_call`** names the parent's call block for a `spawn`.
- **`retry_of`** (a later attempt at the same delegated task) and **`result_use`** (`returned`, `referenced` or `unknown`: how the parent used the child's result) come from recorded structure only.
- **Evidence** names the field path that produced each link. When several sources contribute the same relation, the exported evidence is their lexicographically first field path; source location, arrival order and source preference do not change it. The relation keeps every source contribution. Different recorded branch points or spawning calls remain separate statements; no branch point is selected from a disagreement.
- A delivery creates a `reference` link with `how: briefing`, never a provider link.
- **Pending links.** A link names its ends by their session IDs. An end that is not in the archive reads as unresolved and resolves when it arrives, whatever the arrival order, without the link being rewritten ([Names and references](#names-and-references)). Cycles and conflicting parents are reported at read, never resolved by choosing one link.
- Claude `agent-acompact-*` files are two different things. A replay of the main session ending in a summary is a compaction of the main session ([Compaction boundaries](#compaction-boundaries)), and its file is a sidecar of that session. A Task subagent's own compaction is a `spawn` child with `copies_prefix = false`. Explicit markers of the session a file belongs to and recorded native message identities distinguish them; message counts alone do not establish that session. When the available markers do not establish it, the reading reports the session as unknown, with its reason.

## Compaction boundaries

A compaction is a discontinuity in the model's context, not a special kind of content.

- **The summary is a message** at the boundary, with `written_by: harness` and `prompt_kind: compaction_summary`. It is searchable, embedded like other prose, and counted once.
- **The boundary is an event linked to that message.** It records the tokens before and after (as reported), the trigger (`manual`, `auto`, or the harness's own value), the range of messages it evicted and what it kept. Codex `replacement_history` stays in this event and is not turned into messages.
- Encodings mapped onto it: Claude Code inline `isCompactSummary` records (with the `compact_boundary` record before them, whose event names the summary message), main-session `agent-acompact` replays, Codex `compacted` records, claude.ai inline summaries. A Claude Code `summary` line holding only its text titles its session and is not a compaction; one carrying a `message` is a compaction replay's summary. Providers without compaction (ChatGPT, Gemini, Antigravity) have none.
- **Microcompaction** (named tool results removed from the context) is an event naming the removed results.
- Messages before the boundary stay stored and readable; they are no longer in the model's context. Composing the context as the model saw it at a message is a read ([transcripts.md](transcripts.md#the-context-view)).

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
- Who supplied a file matters: an upload and a file the model generated for download are different things.
- **Every route delivers the bytes** when the source has them:

  | Origin | Source of the bytes |
  |---|---|
  | ChatGPT export | `.dat` and named members, by file ID and the export's name maps; sandbox links through a resolver ported from Python Polylogue; generated-image metadata |
  | ChatGPT, claude.ai, Grok, Gemini and AI Studio pages | bytes the extension fetched ([extension.md](extension.md#live-capture)) |
  | claude.ai export | `extracted_content` as a document (the export has no bytes); bytes from captures |
  | AI Studio | inline base64; `driveDocument` and `driveImage` fetched by Drive ID ([capture.md](capture.md#ai-studio-through-drive)) |
  | Claude Code | inline base64 images, `tool_result` media, `paste-cache` |
  | Codex | inline `data:` images; `local_images` files, read when the rollout line is observed, because they are screenshots in paths that disappear. This is the one read outside a root, limited to paths a parsed record names; a missing file is `unavailable_at_source` |
  | Hermes | `pastes/`, `image_cache/` |

- In a ChatGPT export, an attachment is matched to the stored export members by the native file ID and the export's naming and library sidecars; a file ID can match without a name, and several distinct matching files stay separate matches.
- `polylogue attachment:<message>#<n> read --to FILE` works for every origin and writes the exact bytes for any caller, through a temporary file that replaces `FILE` only once the length and hash agree ([reading.md](reading.md#attachments)).

## Pastes

- Paste spans come from transcript markers, `history.jsonl` `pastedContents` (exact text or a hash only), `UserPromptSubmit` hook payloads, and `paste-cache/*.txt` bodies for hash-only references. They are never found by pattern-matching text.
- Each span has a boundary state: `exact`, `projected`, `whole_message_fallback` or `hash_only`.
- Polylogue owns the span; in primary mode Sinex owns the join between a clipboard event and a paste.

## Sidecars and linked materials

- **A sidecar is read into its session, never as a session:** `.meta.json` files, tool-result overflow files, Gemini tool outputs, Grok terminal logs, workflow journals, compaction replays.
- **Linked materials.** Every URL a conversation cites or an agent produces is recorded as a material observation of its message. Provider-hosted files (download links, sandbox outputs, file previews) are fetched at capture as attachments. Web pages a conversation cites are fetched too, because cited pages disappear; a setting turns this off. Every fetch goes through a guard that refuses private and loopback addresses and unsafe redirects.

## One conversation through several routes

A conversation can arrive as a live file, a browser capture, an export, a Drive revision, a Takeout and an imported old copy. This redundancy is deliberate. All routes arrive as `<origin>:<native id>` and are one session; every route's reading is stored, and the shown message is composed field by field.

A route is one of `live` (the tool's own file or database), `capture` (the provider's data for one conversation, fetched as it is shown: the extension's capture of the page's data, a Drive revision), `recorded` (Polylogue's own hook, process and collector records), `export` (a provider export), `copy` (an imported tool file, a bundle) and `page_text` (text read from the page). Each origin declares the route of its inputs; an input brought in with `import` is an `export` when the origin is an export and a `copy` otherwise. The shown message is a function of each input's reading of it and of nothing else: an input's reading of a message is that of the newest of its chains whose reading holds it ([archive.md](archive.md#readings)). It is composed in **field groups** taken together from one reading: content (role, prompt kind, blocks and the provider fields indexed by block), time, model, usage (tokens, reported cost and duration), stop reason, parent and production.

1. A field group present in one reading and absent in another is taken from where it is present. The content group comes first from a reading whose own chain records the session as a work session (any kind but a compaction replay), when one exists.
2. Then a reading from a main session wins over a sidechain's copy of the message.
3. Where several readings have the group, the **route precedence for that group** decides: `live`, `capture`, `recorded`, `export`, `copy`, then `page_text` for every group. For example, message text comes from the provider's data over the export over a page scrape, and provider timestamps from the export when the capture has none. One table serves every origin today; an origin whose routes call for another order declares its own.
4. Between readings of the same route, a finished reading wins over one whose message completion is unknown, which wins over one still in progress. A message is finished when it is not an assistant message or carries a stop reason; an assistant message without one is in progress when its reading is not page text and its origin records stop reasons when a response ends (Claude Code, claude.ai, AI Studio, Hermes), and of unknown message completion otherwise (page text records no stop reasons).
5. The last tie-break is the content hash, so the result never depends on arrival order or recency. Between readings with identical content, the main route is the one whose input ID sorts first; nothing shown depends on it.

- The reading that supplies the content is the message's **main route**: its version is the stored reading. Messages a session holds through several routes are ordered by their source times ([Messages](#messages)). The composition records, for each other field group, the input and chain it came from (`message_route`), when that is not the main route.
- The shown rows of a session are computed from its readings whenever one of them changes, never adjusted by the difference a step made ([architecture.md](architecture.md#composition)). A change that leaves a message's composition as it was leaves its rows and every cursor into them as they were.
- A message stated under several sessions belongs to the session of the reading that supplies its content; within one input, the reading is that of the chain's first statement (the last, for a chain whose records restate whole conversations). Readings of one version that each hold some of its blocks (a sparse export) together supply every block any of them holds.
- Documents show the newest source time, then route precedence, then the content hash. Events and links show their reading by route precedence, then value hash; a link's evidence and copied-prefix observation merge over readings of the same recorded coordinates. Conflicting coordinates remain distinct statements.
- **Routes without shared turn IDs** (a Gemini page capture and a Takeout of the same conversation) are matched by the repeated-turn rule; the matched message names the shown one (`duplicate_of`). That rule and its validity evidence are in [measures.md](measures.md#session-structure).
- **A capture read from the page** rather than the provider's data is marked *partial fidelity* (the `page_text` route) and is outranked by any other route for every field; it supplies only what no other reading has.

## Logical sessions

The **logical session** is the default unit of work: the root of a tree of `spawn`, `fork`, `continuation` and `resume` links, so one piece of work spread over subagents, forks, compactions and resumes reads as one thing. `reference` links never join it.

- Listings, counts and measures show logical sessions, each expandable into its physical sessions; `physical:true` lists sessions one by one ([query.md](query.md#units-and-fields)).
- References always name physical sessions exactly.
- The stored root and depth are a derived product ([measures.md](measures.md#catalog)). Structural ancestry uses every explicit non-reference relation. Missing parents, conflicting parent identities and cycles produce typed unresolved results, with no chosen parent or cycle root. Copying content alone never changes the logical root.

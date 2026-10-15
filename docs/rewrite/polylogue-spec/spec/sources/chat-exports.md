# Chat exports and captures: ChatGPT, claude.ai, Claude Design

General rules (identity, companions, unrecognized content, merging, spools) are in [README.md](README.md).

**A complete export is a listing of the account** (coverage `account`, in the
terms both projects use: a snapshot's coverage bounds what its absence
proves; a rolling window proves nothing about items older than it, joint
glossary "removed at source"). A ChatGPT export
(`user.json` `id`) and a claude.ai export (`users.json` `uuid`, every
`batch-NNNN` file of one export present) each list the whole account at
their export time. A session present in an earlier complete export of the
same account and absent from a later one was deleted upstream: it becomes
`source: gone` with `gone_at` the later export's time (`query.md`), and it
stays readable with every stored version. An export with a batch missing is
not complete and proves no deletion. The account ID is stored on each session
(in its `attrs`) and is never shown in exports or briefings.

## ChatGPT

**Where:** account data exports, imported with `polylogue import`
(the operator's are in `/realm/account/chatgpt/*.zip`), and live browser
captures written by the extension into Polylogue's capture spool (see
"Browser captures").

Excerpt: `examples/raw/chatgpt/export-conversation.json`.

**Export vintages:**
- **Up to October 2025:** one `conversations.json` (a list), plus
  `user.json`, `message_feedback.json`, `shared_conversations.json`,
  `chat.html`, and attachment files named `file-<id>-<uuid>…` without
  extension.
- **April 2026:** `conversations-000.json` … `conversations-0NN.json` (about
  100 conversations each), attachments keeping their original extensions
  (`png`, `jpg`, `wav`, `pdf`, …), `export_manifest.json`.
- **July 2026 (reduced):** the same split files, attachments as `*.dat` with
  `conversation_asset_file_names.json` mapping attachment ids to names, plus
  `codex.json`, `library_files.json`, `export_manifest.json`. This export
  drops a great deal: nodes have no `children`; messages have only `author`,
  `content`, `create_time`, `id`, `metadata` (missing on 17%); there are no
  `tool` or `system` messages and no `code`, `execution_output`,
  `computer_output` or `tether_*` content. For the 2,094 conversations in
  both, July has 44,834 messages against April's 97,403 (every shared
  message ID has identical content). Merging by union across exports is
  therefore essential, not a refinement: importing July after April must
  keep everything April had ("The same conversation through several
  routes"), and the import report says what each export added.
- An export's attachment files and asset maps (`conversation_asset_file_names.json`,
  `library_files.json`) are companions of its conversation files: an asset
  stored later, as when an extracted export folder is imported a second time
  with more files, re-parses the conversations that name it
  (`convergence.md`, companion files; the Python route never re-bound late
  assets, `polylogue-prwkz`).
- A June 2026 change (third-party report) made some fields `null` and changed
  the citation format. A `null` where an object was expected loses only
  that field; check the newest real export before slice 4.
- Some conversation entries are empty objects with no `mapping` (364 of
  2,836 in the July 2026 export, 18 in April): skipped and counted in the
  import report as "empty entries", not errors.
- Other files are stored with the export and not parsed into sessions:
  `codex.json` (Codex Cloud tasks: `id`, `title`, `archived`, `turns`; shown
  as account records), `library_files.json` (uploads with sha256 and deletion
  times; used to name attachments and to report deleted ones),
  `export_manifest.json` (used to check the export is complete),
  `user_settings.json`, `ads.json`, `shopping.json`, `chat.html`.

**Session fields:**

| Export field | Model field | Rule |
|---|---|---|
| `conversation_id` (or `id`) | `native_id` | |
| `title` | `title` | |
| `create_time`, `update_time` (float epoch seconds) | `started_at` fallback, events | message times win when present |
| `default_model_slug` | session event `default_model` | |
| `is_archived`, `is_starred`, `memory_scope`, `gizmo_id`, project fields, `pinned_time`, `is_study_mode`, `atlas_mode_enabled`, `context_scopes`, `conversation_origin`, `voice`, `async_status`, `is_do_not_remember`, `disabled_tool_ids`, `conversation_template_id`, `gizmo_type`, `moderation_results`, `is_read_only`, `plugin_ids` | session events (`provider_state`) | |
| temporary-chat marker | `temporary=1` | |

**Messages:** `mapping` is a tree of nodes `{id, message, parent, children}`.
`current_node` names the leaf of the active branch.

| Node content | Becomes | Rule |
|---|---|---|
| node with no `message` | nothing | structural only |
| `author.role` = `user` | message `role=user` | `written_by=human`, unless `metadata.is_visually_hidden_from_conversation` or the content type is `user_editable_context` (custom instructions), then `harness` |
| `author.role` = `assistant` | message `role=assistant`, `written_by=model`, `model` = `metadata.resolved_model_slug`, else `model_slug` | `resolved_model_slug` is the model that actually answered; `requested_model_slug` is kept as an event when it differs |
| `author.role` = `system` | message `role=system`, `written_by=harness` | also `metadata.is_user_system_message`, `rebase_developer_message`, `rebase_system_message`: text the user configured, still `written_by=harness` |
| `author.role` = `tool` | message `role=tool`, `written_by=harness` | the tool name is `author.name` |
| `author.role` = `assistant` with `metadata.real_author` naming a tool (`tool:web.run`) | tool_result, `written_by=harness`, paired with its call | the record's role says assistant, but a tool wrote it. Captures only: no message in the operator's four exports carries `real_author` |
| `content_type: text` | text block | `parts` joined |
| part `audio_transcription` | text block marked as transcribed speech | the text of a spoken prompt; `real_time_user_audio_video_asset_pointer` becomes a file block |
| `multimodal_text` | text blocks plus image/file blocks | `asset_pointer` (`file-service://<id>`, `sediment://<id>`) resolved to the export's attachment file via `conversation_asset_file_names.json` or the `file-<id>` naming; audio and video pointers become file blocks |
| `thoughts`, `reasoning_recap` | thinking blocks | `reasoning_recap` keeps a marker that it is a recap |
| `code` (assistant) | tool_call (the code interpreter), `input = {code}` | |
| `code` (tool role) | tool_result text: the tool's echo of the code it ran | 204 in the April export |
| `thoughts` (tool role) | a thinking block on a harness message | 29 in the April export |
| `execution_output` (tool) | tool_result answering the preceding code call | outcome from `metadata.aggregate_result.status` (`success` → ok, `failed_with_in_kernel_exception` → error) |
| `tether_quote`, `tether_browsing_display`, `sonic_webpage` | tool_result of a browsing call | |
| `computer_output` | tool_result | |
| `system_error` | text block, `written_by=harness`, event `error` | |
| `citable_code_output` | tool_result | |
| `user_editable_context`, `model_editable_context` | message `written_by=harness`, `model_saw=1` | custom instructions and memory the model saw |
| `metadata.finished_duration_sec`, reasoning start/end | message `duration_ms`, event `reasoning_time` | the operator explicitly wants reasoning time kept. ChatGPT copies one reply's timing onto several nodes below the same user message (thoughts, tool calls, the recap, the final answer); a reply starts at the first assistant node under a user message, and each regenerated reply is its own reply. One timing is kept per reply, from the first of: a node with `finished_duration_sec`; a node with both reasoning start and end (their difference); a node with a legacy `durationMs`, which counts only when the same reply has one of the other fields. Among equal nodes: a `reasoning_recap` node, then one with both start and end, then one with `end_turn: true`, then the smallest node ID; export order is never used. The chosen node's message gets `duration_ms` and the event; the reply's other nodes get none (if the chosen node is not emitted as a message, the event goes on an emitted message of the same reply, checked in slice 4). A bare `durationMs` in a reply with no other timing stays that message's own duration (example `examples/sources/chatgpt-export/repeated-reasoning-time/`, to be written in slice 4; `plan/testing.md`, layer 2) |
| `metadata.ada_visualizations` (charts and tables the code interpreter produced, with `file_id`) | one file block per entry, resolved by `file_id` like other attachments | separate from `metadata.attachments` and from sandbox links; all three are collected |
| an assistant message whose `recipient` names a tool, or that has `metadata.command` | tool_call | name = `recipient`, else `metadata.command`. Input: the text parsed as JSON when it parses; else `{"args": metadata.args}` when present; else `{}`, only when `metadata.command` is present on an assistant text message (a call with no arguments, such as `computer.initialize`). A message with a bare `recipient` and prose that is not JSON stays a text message. `recipient: "all"` is the default audience, never a tool name, so such a message stays text (or a code result, when it carries one) even when its text parses as JSON (`polylogue-dktc4`) |
| Deep Research, canvas, async tasks in `metadata` | events with their payloads; canvas documents become file blocks | |
| Codex Cloud tasks embedded in `metadata` | events `codex_cloud_task` | |
| `metadata.thinking_effort`, `reasoning_status`, `finish_details`, `request_id`, `turn_exchange_id` | kept on the message (`finish_details` → `stop_reason`) | |
| `metadata.search_result_groups` | kept on the browsing tool_result's `data_json`: the sources the answer drew on | |
| message `channel` (`final`, `analysis`, `commentary`) | kept in `attrs`; `analysis` content becomes a thinking block | the channel decides what the user saw; no export carries `analysis` (captures do) |
| `status: finished_partial_completion` | a finished version (it beats `in_progress`, merge rule 2), `stop_reason: partial` | 339 in the exports |
| `weight` | kept in `attrs`; a weight of 0.0 does not hide a message (its meaning is undocumented; only `is_visually_hidden_from_conversation` hides) | |
| `end_turn: true` (assistant) | confirms the end of a reply, the boundary the duration rule computes | |
| `metadata.branching_from_conversation_id` | a `continuation` link, `how: fork`, from that conversation | 17 in April; the July export keeps only the parent's title (54), which identifies nothing, so it is a session event and no link |
| `targeted_reply_source_message_id` (with the quoted range and label) | `quotes` in the message's `attrs`: the quoted message, range and label (a relation inside one session, not a link, since links join two sessions; `model.md`, Link) | |
| a hashed tool name (`a8km123`, `t2uay3k.sj1i4kz`) | kept as the tool name; a table beside the parser may map known ones to readable names, and an unmapped one stays as given, never guessed | |
| image-generation calls (`metadata.dalle`, image-gen tool messages) | tool_call with the prompt the model sent as its input, and the image as a file block on the result | the prompt is otherwise lost |
| any other content type | `other` block with the type name, counted under unrecognized | |

**Tool-answer presence is structural.** Resolve `author.role: tool` or a tool-valued `metadata.real_author` and the native parent/call relationship before applying the body-content rows above. A tool-authored node answering a known native call emits a `tool_result` even when it has only fallback `content.text`, empty `parts`, an explicitly empty string or no textual body. Use the recorded parts text, or supplied fallback `content.text` when the parts carry no textual payload (including `parts: []`); preserve absent versus empty output and unsupported content rather than inventing text. Retain any thinking/attachment metadata alongside the answer. Status follows `model.md`: a recognized provider status or exit code can establish success/error, but authorship, parentage and empty text cannot. Without status the result is `unknown` / `result_without_status`.

Native node coordinates locate the parent and evidence; they are not automatically canonical message IDs or call IDs. Correlate only through the provider's declared identity/parent rules, never fabricate a tool call from the result's prose or tool name. An explicitly recorded result with no established call stays unlinked. The slice-4 parser examples independently enumerate native calls and answers, including fallback-only and empty bodies (`plan/testing.md`, layer 2); preserving a tool message's text without its established result block is incomplete normalization.

**Branches:** every node is a message. Nodes not on the path from
`current_node` to the root get `on_active_path=false`. `parent` follows the
general rule (`README.md`, "Parents skip records that are not messages").
When an export and a capture name different `current_node`s, the merge rules
choose (`README.md`, rule 6). A message with `status: in_progress` is an
unfinished version (rule 2).
Sibling order (which branch is first, second, …) is the provider's
`children` order. A form without `children` (the July 2026 export, a
capture that kept only parent links) is rebuilt from `parent` links, its
siblings ordered by `create_time`, then node ID; `on_active_path` still
comes only from `current_node`, and a node whose path cannot be decided is
unknown (NULL), never marked active by default (`polylogue-dv53f`: a capture
fallback marked every turn active and first, losing the branches).

**Message identity:** the node id. When a conversation repeats a node ID, a
`parent` naming it means its nearest earlier occurrence in the mapping.

**Branches into new conversations.** "Branch in new chat" copies the
source's nodes, IDs included, into a new conversation (20 groups of
conversations in the April 2026 export share message-bearing node IDs, the
largest pair about 600 nodes). A node ID that occurs in two conversations is
a copy. The conversation whose `create_time` is earlier (then the smaller
conversation ID) is the source. The later conversation gets a
`continuation` link, `how: fork`, evidence `shared node id`, with the last
shared node on its active path as the branch point, and its shared messages
get `inherited_from`, so they are counted once.

**Usage:** exports carry no token counts. Usage stays unknown; cost views show
these sessions as "subscription, cost unknown" (`analysis.md`).

**Attachments:** attachment bytes stay inside the export ZIP, which is
stored whole as one input; a file/image block refers to its member as
`zip:<export input>/<member>` (`storage/blobs.md`; the block's `member`
column) and reading it decompresses that member from the stored ZIP; nothing
is extracted or stored twice. An attachment named in a conversation but missing from the
export keeps the block with `blob_hash` empty and a reason.

- **Sandbox links.** A `sandbox:/mnt/data/<name>` link in an assistant
  message is an attachment the code interpreter produced. Every one is kept,
  with no per-message limit: with
  its bytes when the export or a capture carries them, otherwise as a file
  block with `blob_hash` empty and the reason "not in export". A link names only a file; it is matched to a
  `library_files.json` entry in stages, stopping at the first stage with
  exactly one match: (1) same message ID and same name; (2) same
  conversation and same name; (3) the only entry anywhere with that name.
  If a stage finds several entries, or no stage finds one, the block has no
  bytes and the reason `ambiguous_name` or `not_in_export`; the stage that
  matched is kept in the block's `data_json`. A link is never joined to a
  file of its message with a different name.
- **Same result alone or in a bundle.** A conversation produces the same
  messages, attachments and IDs whether it arrives as its own file or inside
  a bundle of conversations.

**Collections.** ChatGPT projects (`gizmo_id` of a project, and the project
fields of the conversation) and custom GPTs (`gizmo_id` of a GPT) map to the
session's `collection` (provider ID and name; `model.md`), so "everything in
my ChatGPT Rust project" is a query. Names come from
the export or a capture of the project; an unnamed collection keeps its ID.

**Nested metadata: what is mapped, kept or ignored.** The export's per-message
`metadata` holds dozens of nested groups. Each group is listed once in a
table in this section, built in slice 4 from how often each appears in the
operator's real exports, and marked *mapped* (with its target field),
*kept* (copied as given into the message's `attrs`), or *ignored* (with the
reason, for example a UI flag with no meaning for the conversation). Groups
known now: code block indices and per-tool metadata (*kept*),
`content_references` and citations (*kept*, also used to resolve link text),
`finish_details` and `request_id` (*mapped*, see above). A group that is not
in the table is counted as unrecognized like any unknown field.

**Quirks:**
- Old exports use `durationMs`-style timing fields; new ones use
  `finished_duration_sec`. Both map to `duration_ms`.
- A conversation present in several exports is the same session; see
  "The same conversation through several routes".
- Ten loose capture files written by the Python version
  (`polylogue_capture_kind: browser_llm_session`) sit in the account
  folders. They are Python-era capture envelopes, imported and read like the
  old archive's captures ("Browser captures": two carry the provider's
  payload, eight only a normalized session). One
  `recovery-shared-page-*.messages.json` file is imported so its bytes are
  kept, but no parser exists for a single file; it is listed as an
  unrecognized input.

## claude.ai and Claude Design

**Where:** claude.ai account exports (`claude-ai-data-*.zip`, imported by
command) and live browser captures.

Excerpts: `examples/raw/claude-ai/export-conversation.json`,
`design-chat.json`.

**Export contents:** `users.json`, `memories.json`, `conversations.json`,
`projects/<uuid>.json` (project documents), `design_chats/<uuid>.json` (from
2026).

**claude.ai conversations** (origin `claude-ai`):

| Field | Becomes | Rule |
|---|---|---|
| `uuid` | `native_id` | |
| `name` | `title` | |
| `summary` | session value `provider_summary` (one value per session, decided across routes by rule 6, `README.md`; not an event) | |
| `created_at`, `updated_at` | `started_at` fallback when no message has a time; `updated_at` is the provider time for rule 6 (`README.md`) | |
| (no project field) | — | conversations carry no project reference in any era; `projects.json` (2025) and `projects/<uuid>.json` (2026) are stored as inputs and shown as account records, not linked to conversations |
| `chat_messages[].uuid` | message id | |
| `chat_messages[].sender` = `human` / `assistant` | `role=user, written_by=human` / `role=assistant, written_by=model` | |
| `chat_messages[].content[]` blocks `text`, `thinking`, `tool_use`, `tool_result` | blocks of the same kinds | `tool_use.id` ↔ `tool_result.tool_use_id`; `is_error` → outcome. The October 2025 export has neither ID, so there calls and results are paired by order within the message. Each block's `start_timestamp` / `stop_timestamp` gives its duration; thinking `summaries`, `cut_off`, `truncated`, `signature`, `hidden` are kept; tool `integration_name`, `mcp_server_url`, `is_mcp_app`, `tool_identifier` and result `structured_content` are kept; text `citations` are kept in the message's `attrs` |
| a message with no blocks, text or files (503 in the July 2026 export) | a message with no blocks, so tree links through it hold; never counted as a prompt or reply; counted in the import report | |
| `thinking` with `thinking_hidden: true` and no text | a thinking block marked hidden, like encrypted thinking | 9 in 2026 exports |
| block `token_budget` (`remaining`) | event `token_budget` | |
| block `voice_note` (`title`, `text`) | text block marked as spoken | |
| `chat_messages[].text` (older exports without `content`) | text block | |
| `chat_messages[].attachments` (extracted text) | file block with the extracted text; `blob_hash` empty | extracted text is not the original file; the block says so |
| `chat_messages[].files` | file block | bytes only when the export or a capture includes them |
| `parent_message_uuid` | `parent` | branches from edits/retries: the active path ends at the provider's leaf marker (`current_leaf_message_uuid`, which browser captures carry and an export carries where present); without one, messages off the shared trunk have active-path status unknown (`model.md`, the active path), never a guessed latest leaf. The root's parent is the sentinel `00000000-0000-4000-8000-000000000000`, which means no parent. Exports before 2026 have no `parent_message_uuid`: those conversations are linear in file order. |
| compaction summaries in captures | event `compaction`, `model_saw=1` | |

**memories.json:** stored as an input; shown as account memory, not a
session.

**Claude Design** (origin `claude-design`): `design_chats/<uuid>.json` with
`{uuid, title, project, created_at, updated_at, messages}`. Each message is
`{uuid, role, created_at, content}`, where `content` is an object holding
`role`, `contentBlocks` (`text`, `thinking`, `tool_call` with `toolCall`,
`error`, `user_interjection`), `attachments` (`file`, `image`, `text`,
`skill`, `folder`; `skill` attachments are harness-injected text,
`written_by=harness`), `authorName`, `authorAccountUuid`, `pill` and
`turnInputTokens`. `project` becomes a session event. `role` `user` →
`role=user, written_by=human` (`user_interjection` blocks too); `assistant`
→ `role=assistant, written_by=model`; `authorName` and `authorAccountUuid`
are kept in `attrs`; any other role is `written_by: unknown`.

**Usage:** claude.ai exports have none (unknown). Claude Design's
`turnInputTokens` (on 37 of 95 messages here) is per-turn input usage,
marked as the provider's count; output stays unknown.

**Collections.** claude.ai projects (`project_uuid`, with the project name
from the export's projects file) and Claude Design projects map to the
session's `collection` when a capture carries them: no conversation in the
operator's four claude.ai exports has a project field, so for exports the
collection is unknown, never inferred.

**Compaction summaries.** A message carrying `compaction_summary` marks the
point where claude.ai condensed the conversation. It becomes a `compaction`
event at that message, holding the summary text and its start and stop
times, `model_saw=1`. The summary is never inserted as a message, which
would shift positions and break branched conversations; the `context` view
(`rendering.md`) shows it as the summary in force.

**Conversation-level and message-level attachments**. An
export lists some files both on the conversation and on a message. A
conversation-level attachment without an ID takes the ID of the single
message-level attachment with the same name and media type; when that
message-level row has the bytes, it is the one kept, field by field. Two rows
whose bytes differ stay two files. **Retrieved knowledge** (check in the slice-4 field scan): `knowledge` content parts (documents a
model retrieved) become `tool_result` blocks of a retrieval call, with the
source named in `data`, not attachments: they are what the model read, not files a person gave it.

**Repeated message IDs.** None of the operator's four exports repeats a message ID within a conversation or across conversations of one export (51,009 message records, measured 2026-10-12). A repeat in a later export would follow the general rule for one input (`model.md`, Identity): versions of one message, the last in record order winning field by field, with a conflict recorded and shown in `status`; no special ID form.

**Across exports.** The four exports (October 2025, April, June and July
2026) hold the same conversations in successive formats, and most of the
differences between two exports of a conversation are not edits: 1,426 of
2,861 consecutive pairs differ while `updated_at` is unchanged. Measured
2026-10-12:

- Later exports add fields to the same records: `parent_message_uuid`,
  tool `id`/`tool_use_id`, `mcp_server_url`, `structured_content`,
  `signature` and file `file_uuid` from April 2026; result `links` from
  June; `hidden_in_chat`, `tool_identifier`, `approval_key_legacy`,
  `thinking_hidden` from July. Rule 2's "more fields present" picks the
  newer copy, and rule 3 fills fields only one side has.
- **Volatile fields** (left out of version hashes and comparisons, rule 2):
  the `uuid` of each item inside a `tool_result`'s `content`, which is new
  in every export (712, 808 and 808 changes); `display_content.json_block`
  (32). The conversation `summary` is regenerated without changing
  `updated_at` (17): it is a session value, decided by rule 6.
- The June export drops empty text blocks (536 messages, joined text
  unchanged); rule 2 does not count empty text blocks.
- Conversations deleted on claude.ai stay: 2 are only in the October 2025
  export and 1 is missing from July 2026; their sessions come from the
  older exports, which are their only copy.

Provenance (Bead and audit IDs by section) and the Python-version lessons behind some rules: `evidence/spec-provenance-model-sources.md`.

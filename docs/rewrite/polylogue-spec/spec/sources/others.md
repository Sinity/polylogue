# Other providers and planned sources

General rules (identity, companions, unrecognized content, merging, spools) are in [README.md](README.md).

## Hermes

**Where:** `~/.hermes/state.db` (SQLite), `~/.hermes/sessions/session_*.json`,
`~/.hermes/sessions/request_dump_*.json`, ATIF trajectory exports
(`trajectory-*.json`), `verification_evidence.db`.

Not collected: `~/.hermes/hermes-agent/`, Hermes's own program checkout, including
the prompt templates its optional skills ship (`optional-skills/*/templates/`,
7.5 MB). They are program files, not the operator's sessions; the Python
version stored them only because its detection read any JSON it found.

Excerpt: `examples/raw/hermes/state-db-rows.json`. The prototype parser on the operator's data, and what Hermes's own code says about compaction, rewrites, steers and the session documents: `evidence/hermes-real-accounting.md`.

`state.db` is a live SQLite database, collected as snapshots (read through
`sqlite_dbpage` in one read transaction) and stored as page patches over a base, like Codex's databases
(`storage/blobs.md`, Provider databases); the parser reads its `sessions` and `messages`
tables from a snapshot, and reads the layout version from its
`schema_version` table, keeping one mapping per version.

| Source | Becomes | Rule |
|---|---|---|
| `sessions.id` | `native_id` = `<sessions.id>@profile-<first 16 hex of BLAKE3 of (machine ID, the install root's path, normalized lexically, as recorded in the first observation of that install's `state.db` input)>` | one install's `session_*.json` and `state.db` qualify identically, so one conversation never becomes two sessions and two installs never collide. Later moves of the root keep the key because the input keeps its first observation; a root first seen on another machine is another install. Hermes hook and lifecycle events name the raw `sessions.id`; the hook writer records in the line the Hermes home directory it ran under, and the event joins the session `<id>@profile-<key>` of that directory. A line with no directory joins only when exactly one install has a session with that raw ID; otherwise it is counted as unmatched. No pattern match is used |
| `sessions.source` (`cli`, `subagent`, `tool`) | `kind`: `cli` → `main`; `subagent` or `tool` → `subagent`; any other value → `main`, counted in the slice-5 field scan | on this machine only `cli` (105) and `subagent` (1) occur; `tool` is kept as unobserved (`evidence/gemini-hermes-real-survey.md`) |
| `sessions.parent_session_id` | a link whose kind is read in order: (1) the child's `model_config._branched_from` set → `continuation`, `how: fork`; (2) `model_config._delegate_from` set, or `sessions.source` = `tool` → `subagent`, `how: fresh`; (3) the parent's `end_reason` is `compression` or `compaction` → `continuation`, `how: after_compaction` | rows matching none are counted in the slice-5 field scan. Sessions joined by rule 3 form a compression chain; a session in the chain with no messages is not written (a session needs positive evidence), and its child's link points at the nearest ancestor with messages, with that ancestor's last message as the branch point |
| `sessions.model`, `title`, `cwd`, `git_*`, `git_repo_root` (the worktree root: `checkout` when it differs from `cwd`'s, never `repo_root`, which is the root commit), start/end | session fields; `model` is also each model message's model (messages have no model column) | times are REAL Unix seconds with fractions, not milliseconds. `message_count` and `tool_call_count` count only the active set after a compaction, so they are never checked against the messages |
| `sessions.end_reason` | event `session_end` | `compression` also drives rule 3 above |
| `sessions.system_prompt` | message `role=system, written_by=harness` | |
| `sessions` token and cost columns (including `estimated_cost_usd`, `actual_cost_usd`, `cost_status`, `cost_source`) | session event `provider_usage` | per-message counts win when present |
| `messages.role` = user / assistant / tool | messages with the same roles, in `messages.id` order (Hermes's insertion order; `timestamp` order inverts 2 adjacent pairs) | tool messages are `written_by=harness`; a message starting `[CONTEXT COMPACTION` is the compressor's summary, `written_by=harness` |
| message IDs | a tool message: its `tool_call_id`; a message with tool calls: `calls-<first call ID>`; any other: the fallback ID (content hash), never a rowid or a position | `replace_messages` (used by `/retry`, `/undo` and older `/compress`) deletes a session's rows and inserts them again, so rowids move; the call-based IDs also match the session documents' messages. A second message with the same call-based ID in one session is a copy (below) and appends `-copy<n>` |
| a row repeating an earlier message | a copy: `inherited_from` the first occurrence (with `from_session` when that is a continuation ancestor's), never counted again | Hermes's in-place compaction (`archive_and_compact`) marks the active rows `compacted = 1` and inserts the compacted list (summary plus retained messages) as new rows, and a later compaction archives those copies too; older versions started a child session holding the same retained copies. A row with a tool call ID or tool calls is a copy when its key (call ID, or the list of call IDs) appeared earlier in the session or its continuation ancestors; a row without calls is a copy when its role and text appeared earlier and a neighbouring row is a copy or the summary. A copy whose text differs (a tool result rewritten to a compressor placeholder, `[terminal] ran `…` -> exit 0, 47 lines output`) keeps its own text with `attrs.copy = "rewritten"`. Event `compaction` sits at the first row of each run of copies, or at the summary. Measured: 1,020 copies of 7,698 rows, 297 from an ancestor, 236 rewritten; 308 own human prompts where counting copies gives 421 |
| `messages.content` | text block | |
| a steer at the end of a tool result: `[OUT-OF-BAND USER MESSAGE — …]` … `[/OUT-OF-BAND USER MESSAGE]`, or the older `\n\nUser guidance: <text>` | a human prose message right after the tool message, ID `<tool message ID>-steer<n>`, `attrs.delivered_in = "tool_result"`; the text is removed from the result's output | the user typing while the agent works; Hermes appends it to the next tool result (`agent/prompt_builder.py`). 15 rows use the older form, none the marker |
| `messages.tool_calls` (JSON) | tool_call blocks: `function.name`, call ID `id` (equal to `call_id` in all 3,584), input `function.arguments` decoded from its JSON string (4,100 of 4,100 are strings) | |
| `messages.tool_call_id`, `tool_name` | tool_result pairing by call ID; the tool's name comes from the paired call (`tool_name` is empty on 2,811 of 4,101 tool rows) | tool content is read as its leading JSON value, plain or behind the prefix `\x00json:` (none at schema 19, still read); harness notes may follow the object (`[Hint: Results truncated …]`, `[Subdirectory context discovered: …]`, `[Tool loop warning …]`, 103 rows) and stay in the output. Of 4,101 results: 3,538 a plain object, 103 an object with notes, 409 compressor placeholders, 51 other text. From the object: `output` is the output; the outcome is read in order: a non-empty string `error` → `error` (a null `error`, 1,503 rows, is absent); a boolean `success` → `ok` or `error`; an integer `exit_code` → `ok` when 0, else `error`; `status` (246 rows) by its closed table: `success`, `unchanged` → `ok`; `error`, `not_found` → `error`; `timeout` → `error` `timed_out`; `blocked` → `error` `denied`; `interrupted`, `killed` → `error` `interrupted`; `running`, `dispatched` → `unknown` `background_never_reported`; any other value → `unknown` with the value kept (`exited` always carries `exit_code`, which decides first; `exit_code` is kept beside an `error` or `success` verdict). `unknown`: an `error` that is boolean, empty or zero, or a `success`/`exit_code` of the wrong type → reason `status_field_unreadable`; prefixed content whose JSON does not decode → `json_marker_not_decoded`; a placeholder `[<tool>] …` → `output_not_recorded`; no signal at all → `result_without_status` (1,149 own results, mostly file reads and searches, which carry no verdict) |
| `messages.reasoning*`, `codex_reasoning_items` | one thinking block from `reasoning_content`, else `reasoning` (equal in all 2,079 rows holding both; 6 hold only `reasoning_content`); Codex reasoning items as encrypted thinking blocks with their summary text, mapped like Codex's | |
| a message marked `_empty_recovery_synthetic` | `written_by=harness` | written by Hermes to recover from an empty reply |
| `messages.codex_message_items` (JSON) | the message's items from Codex-backed models (reasoning items as thinking blocks, tool calls), mapped like Codex response items; a text item equal (after trimming) to the message's content or one of its lines is not added again, a mere substring is kept | kept whole in `attrs` when a part has no mapping |
| `messages.finish_reason` | `stop_reason` | |
| `messages.compacted = 1` (with `active = 0`) | message kept on the active path; the rows inserted after it that repeat it are copies (above) | condensed by compression, not rewound: all 1,434 rows with `active = 0` on this machine have `compacted = 1`. Several compactions leave no marker between them; the runs of copies are the boundaries |
| `messages.active = 0` with `compacted = 0` | `on_active_path=false` | a rewind (none observed; `sessions.rewind_count` exists) |
| `session_*.json` documents | same session (merged by session id); an edit format (`README.md`, rule 5); a message whose call-based ID is a `state.db` message of another session of the same install is a copy of it, and a session made mostly of such copies is linked `continuation`, `how: fork`, evidence `shared call IDs`, to the session holding most of them | the database is preferred where they disagree. Hermes writes the document from the agent's in-memory list: optional (`sessions.write_json_snapshots`, off by default in current Hermes), content cleaned and secrets redacted, never shrinking, and written under the current session ID, so after `/branch` or `/compress` the whole list lands under the new ID. Measured (`evidence/hermes-real-accounting.md`): of the 67 documents with no `state.db` row, 16,055 of 19,015 call-bearing messages are database messages of other sessions; merged by session ID alone they would be 67 extra conversations. Messages matching nothing (3,284 call-bearing, 1,515 text) are kept: some are the only copy of rows `replace_messages` deleted before Polylogue collected `state.db`. Times in `session_*.json` and `request_dump_*.json` (`session_start`, `last_updated`) are local times without an offset (measured: 2 h ahead of `state.db`'s epoch values in Warsaw summer time). They are read with the collecting machine's recorded zone and flagged `local_naive` (`model.md`, Time). Their messages carry no time at all (none of 28,025), so message times come only from `state.db`, and a session only in JSON (67 of 167) has just its two session times. One document is carried through a compression chain, so its `session_start` is often the chain's first session's start (33 of 100), not this session's: a session's start never comes from the JSON when `state.db` has its row |
| ATIF trajectories | same session's tool calls and timings | ATIF tool-call mapping has not been checked against a real export with tool calls; until it is, ATIF-only tool data is marked `"source": "atif"` in the result block's `data` |
| `request_dump_*.json` | stored as inputs; not parsed | raw API dumps, useful for inspection only |

## Antigravity

**Where:** `~/.gemini/antigravity/`: `conversations/<id>.pb` (protobuf, 44
here), `brain/<id>/` (agent-written Markdown artifacts with `*.metadata.json`
and versioned copies `*.resolved.N`); and `~/.gemini/antigravity-cli/`:
`conversations/<id>.db` (SQLite trajectory stores, 10 here),
`conversation_summaries.db`, `history.jsonl`, `brain/`
(`evidence/gemini-hermes-real-survey.md`).

Excerpts: `examples/raw/antigravity/`.

- `conversation_summaries.db` (under `antigravity-cli/`, 44 rows) and
  `state.vscdb` (SQLite, about 1 MB; not under `~/.gemini` on this machine, its
  location is still to be found) are collected as database snapshots (`storage/blobs.md`, Provider databases). The `.pb`
  conversation files are rewritten whole on each change, so each change is a
  new revision, stored once the file is quiet for 30 seconds.
- **Conversations** are protobuf files readable only through Antigravity's
  local language server (`ConvertTrajectoryToMarkdown`). Polylogue stores the
  `.pb` files as inputs. When the language server is available, the daemon
  asks it for the Markdown rendering, stores that rendering as a derived
  input linked to the `.pb`, and parses the Markdown into messages (user /
  assistant turns as headed sections; tool steps as tool calls when the
  rendering marks them). The files are encrypted (8.0 bits per byte, no
  compression header; the same for `implicit/*.pb`), so nothing but the
  language server reads them. The server also offers
  `GetCascadeTrajectorySteps`, which returns the same `Step` messages the CLI
  stores hold; when it answers, the daemon uses it instead of the Markdown
  rendering and the messages get their stable `step-<idx>` IDs at once (not
  yet tried against a running server). These messages come from a rendering, not from the
  provider's records, so they are named `<session id>:~md<renderer
  version>-<hash>-<n>` and replaced as a set whenever the rendering changes;
  notes and references attach to the session, not to these messages, until
  `GetCascadeTrajectorySteps` answers and gives the stable IDs
  `<session id>:step-<idx>`. A conversation is an edit format (`README.md`, rule 5). When it is not available, the session exists with
  its metadata and an event saying the transcript could not be read; nothing
  is guessed. (The `Step` messages decode with the schema below; what only
  the language server can do is decrypt the files, so reading them without it
  would need the key it uses, which has not been looked for.) If Polylogue starts the language server
  itself, it binds it to a Unix socket or gives it a random per-run token,
  and stops it when reading ends; it never leaves it listening on loopback
  TCP without authentication. Every Antigravity file is stored from slice 5 (principle 5); brain documents and state snapshots are parsed then, and IDE conversations (the encrypted `.pb` files) are parsed once the language server's `GetCascadeTrajectorySteps` has been tried against a running server (decision: `decisions/README.md`).
- **Brain artifacts** (`task.md`, `implementation_plan.md`,
  `walkthrough.md`, …) with metadata `{artifactType, summary, updatedAt,
  version}` become file blocks on a session event `artifact`, one per
  version, attached to the conversation with the same id.
- **Trajectory stores** (`antigravity-cli/conversations/<id>.db`: tables
  `trajectory_meta`, `steps`, `gen_metadata`, `executor_metadata`,
  `parent_references`; 399 steps in 10 files here) are collected as database
  snapshots and parsed: a session per database, messages from `steps` with
  IDs `<session id>:step-<idx>`, never renumbered when earlier steps are
  skipped. They are decoded with Antigravity's own schema
  (`evidence/antigravity-steps-decoded.md`, prototype
  `prototypes/antigravity-steps/`):
  - **The schema comes from the installed language server.** Its binary
    carries every `.proto` file's descriptor; Polylogue extracts them at run
    time into a descriptor set cached by the binary's BLAKE3 hash, and never
    ships them. Decoding uses runtime descriptors (in Rust, `prost-reflect`),
    so a new Antigravity version needs a new extraction, not a code change.
    Fields the descriptor set does not name are kept as raw bytes in `attrs`
    and counted per field number (9 field numbers here, from version skew
    between the CLI that wrote the stores and the installed server).
  - **`step_payload` is a whole `gemini_coder.Step`** (type, status,
    metadata and the one step-specific field); `step_type` is the
    `exa.cortex_pb.CortexStepType` enum, so the mapping is on enum names, not
    measured codes. Without an extracted schema, the stores are kept and the
    session carries an event saying the steps could not be decoded.
  - `USER_INPUT` → user message (`user_response`; `written_by=human` when
    `metadata.source` is `USER_EXPLICIT`, else `harness`).
  - `PLANNER_RESPONSE` → model message: `response` as text, `thinking` or
    `raw_thinking` as thinking (`thinking_redacted` → encrypted thinking),
    `tool_calls` (`id`, `name`, `arguments_json` decoded) as tool calls,
    `stop_reason` (enum name), usage from `metadata.model_usage`
    (`input_tokens`, `output_tokens`, `cache_read_tokens`,
    `cache_write_tokens`, `thinking_output_tokens`), the provider's
    `message_id` in `attrs`.
  - Every tool step (`RUN_COMMAND`, `VIEW_FILE`, `GREP_SEARCH`,
    `LIST_DIRECTORY`, `CODE_ACTION`, `GENERIC`, and any other type not listed
    here) → a tool message holding the result, paired by
    `metadata.tool_call.id` with the model message's call (180 of 180 pair).
    Outcome: `RUN_COMMAND` by `exit_code` (0 → `ok`); a step whose status is
    `ERROR`, `CANCELED` or `INTERRUPTED` → `error`; `DONE` → `ok`; any other
    status → `unknown`.
  - `SYSTEM_MESSAGE` → harness message (an `agent_message` names its
    sender and recipient as `from_name`); `CHECKPOINT` → event `compaction`
    (`user_intent`, `user_requests`, `included_step_index_end`, `model_saw`);
    `CONVERSATION_HISTORY` → event; `ERROR_MESSAGE` → event `error`.
  - Times: `metadata.created_at` and `completed_at` (protobuf Timestamps,
    on all 399 steps).
- **Summaries and prompts**: `conversation_summaries.db` gives each
  conversation's title, workspace (`workspace_uris`), agent name and parent
  (`parent_conversation_id`, `nesting_depth`, a `subagent` link with evidence
  `summary_parent`); `antigravity-cli/history.jsonl` (27 lines:
  `conversationId`, `display`, `timestamp`, `workspace`) is a prompt log,
  used like Claude Code's `history.jsonl`.

## Grok (web)

**Where:** account export (imported by command) and browser captures.

Format: `{"conversations": [{conversation: {id?, title, create_time…},
responses: [{response: {sender, message, createTime, …}}]}]}`; timestamps in
MongoDB extended JSON (`{"$date": {"$numberLong": "…"}}`).

| Field | Becomes |
|---|---|
| `conversation.id` when present, else hash of the opening response's text and time (the opening response is the one with the earliest `createTime`; the tie rule is fixed in slice 5; the title is not part of the ID, and a later export that adds an earlier response changes the opening response, which the import report counts) | `native_id` |
| `conversation.title` | `title` |
| `response.sender` = `human` / `assistant` | `role=user, written_by=human` / `role=assistant, written_by=model` |
| `response.message` | text block |
| `response.createTime` | `at` |
| `response.model` when present | `model` |

No native message ids: the fallback rule (`model.md`, Identity). An export
conversation is an edit format (`README.md`, rule 5), and fallback-ID
messages from an export lose to a capture route with provider IDs for the
same conversation (rule 1). An export conversation without `conversation.id`
is the same session as a capture when their first human message has the same
text (whitespace normalized) and the same time to the minute; the export's
hashed ID is then recorded as an alias of the capture's session. Tiny volume (6 sessions).

## Grok CLI

**Where:** `~/.grok/sessions/<url-encoded cwd>/<session id>/` with
`updates.jsonl`, `events.jsonl`, `chat_history.jsonl`, `summary.json`,
`usage.json`, `rewind_points.jsonl`, `system_prompt.txt`,
`tool_definitions.json`, `prompt_context.json`, `signals.json` (and
`*.lock`, `resources_state.json`, `announcement_state.json`,
`title_refresh_idx`), plus `prompt_history.jsonl` per cwd folder and
`sessions/session_search.sqlite`. 727 files in 17 sessions today; **not
ingested by the Python version**. The prototype parser on the operator's
data: `evidence/grok-cli-real-accounting.md`.

Excerpt: `examples/raw/grok-cli/updates-head.jsonl`.

| Source | Becomes | Rule |
|---|---|---|
| folder name | `native_id`; the url-decoded parent folder is `cwd` | |
| `updates.jsonl` line: `{method, timestamp, params: {sessionId, update}}` | one record; `method` is `session/update` (ACP) or `_x.ai/session/update` (Grok's extensions, 478 of 7,634) | `timestamp` is integer Unix **seconds** |
| `user_message_chunk` | user text; consecutive chunks join into one message, ID `prompt-<_meta.promptIndex>`, `model` from `_meta.modelId` | `written_by=human`, except a chunk with `_meta.hideFromScrollback` (23 of 47, system reminders and task notifications the harness inserted), which is `written_by=harness` |
| `agent_thought_chunk`, `agent_message_chunk`, `tool_call` | one model response: the run of thought chunks (thinking), message chunks (text) and tool calls up to the first result, fallback ID | ACP has no message boundaries; a run is one model call (794 responses against 785 `loop_started` events). The tool's name is `_meta["x.ai/tool"].name`, its input `rawInput` |
| `tool_call_update` with `status` | a tool message, ID the call ID, holding the result: text from `content` parts, else `rawOutput` | outcome: `failed` → `error`; otherwise `rawOutput.exit_code` decides (0 → `ok`, else `error`; 27 `completed` shell results exit non-zero), except a `GrepSearch` with exit 1 and `match_count` 0, which found nothing (`ok`, 55 such); `completed` with no exit code → `ok`; a backgrounded command (`rawOutput` with `pid` and `status`) → `unknown` `background_never_reported` until `task_completed` settles it |
| `tool_call_update` without `status` | ignored kind (`tool_call_update without result`) | restates the call's title, kind and locations (2,041) |
| `task_backgrounded` (`task_id`, `tool_call_id`), `task_completed` (`task_snapshot` with `task_id`, `exit_code`, `explicitly_killed`, times) | events; the completion sets the backgrounded call's outcome and exit code (killed → `error` `interrupted`) | joined by `task_id`; 43 of 43 report back. `background_tasks` snapshots are an ignored kind |
| `turn_completed` (`prompt_id`, `stop_reason`, `elapsed_ms`, `usage`) | event `turn_usage`; `stop_reason` on the turn's last model message; the usage is per message only when `usage.modelCalls` is 1 | the turn's usage spans all its model calls (`costUsdTicks` is the provider's cost) |
| `subagent_spawned` | link `subagent` from `parent_session_id` to `child_session_id` (`effective_context_source`: `new` → `fresh`, `forked` or `resumed` → `copied`), evidence `subagent_spawned`, the record in `attrs` | all 14 children are sessions in the same tree; `subagent_finished` is an event |
| `hook_execution`, `plan` (`model_saw`), `goal_updated`, `session_recap`, `retry_state` (event `error`), `auto_compact_started`, `compaction_checkpoint`, `auto_compact_completed` (event `compaction`, `model_saw`) | events | |
| `events.jsonl` (companion) | `tool_completed` gives the result's `duration_ms` (1,988 of 2,043 results); `permission_resolved` gives event `permission` with `decision` and `wait_ms`; `turn_*` and `goal_*` lines are events | `phase_changed`, `loop_started`, `first_token`, `tool_started` and `permission_requested` are an ignored kind (restated). Its `outcome` is the harness's verdict on the call, not the command's: it says `success` for 58 non-zero exits |
| `chat_history.jsonl` | checks and fills message boundaries when `updates.jsonl` is incomplete | `updates.jsonl` wins where both exist. It is what the model saw: it holds 95 user entries against 47 user chunks, 68 of them with `synthetic_reason` (`system_reminder`, `task_completed`, `project_instructions`, `notification_drain`, `compaction_meta`, `goal_summary`); a synthetic entry is never a human prompt |
| `summary.json` | `title` (`generated_title`), `checkout` (`git_root_dir`, the worktree root; `repo_root` is the root commit, `model.md`), `branch` (`head_branch`), `commit` (`head_commit`), session event `provider_summary` | |
| `usage.json` (`session`, `turns`) | session event `provider_usage` | per turn, like `turn_completed` |
| `rewind_points.jsonl` (`prompt_index`, `file_snapshots`, `after_snapshots`) | event `file_checkpoint` per prompt | one per prompt (46), the files' state for undo; it does not say a rewind happened, so it sets no `on_active_path` |
| `system_prompt.txt` | message `role=system, written_by=harness`, ID `system-prompt` | |
| `tool_definitions.json` | session event `tools_available`, `model_saw=1` | |
| `prompt_history.jsonl` per cwd folder (`prompt`, `session_id`, `timestamp`, `is_bash`) | a prompt log, used like Claude Code's `history.jsonl` | 9 lines |
| `signals.json`, `prompt_context.json` | session event `provider_state` | harness telemetry (line counts, latencies, models used) and the prompt's inputs |
| `session_search.sqlite`, `*.lock`, `resources_state.json`, `announcement_state.json`, `title_refresh_idx`, `~/.grok/worktrees.db`, `~/.grok/downloads/` | not collected | Grok's own index, locks, UI state and binaries |

The ACP update format is shared with other agent CLIs; the prototype's
`Updates` parser takes the ACP kinds and a list of extension kinds, so
another ACP-speaking harness is a small addition.

## Planned sources

- **OpenTelemetry GenAI span files** (`otel-genai`): OTLP-JSON files from a
  folder the user drops them into; spans with `gen_ai.*` attributes become
  sessions (trace), messages (prompt/completion events) and tool calls. No
  network receiver. Rules from the Python parser:
  only traces with at least one `gen_ai.*` span become sessions (other spans
  in them are kept as topology; traces without one are counted and dropped);
  a resource is named by `service.name`, else `resource-<first 16 hex of a
  hash>` over its attributes excluding `process.*`, `host.*`, `container.*`,
  `k8s.pod.*`, `k8s.container.*`, `os.*`, `telemetry.*`,
  `service.instance.*` and keys that are or end in `version`; a span seen
  several times keeps one copy (by schema rank, start time, canonical form,
  schema URL) and records the others as `otel_conflicting_span_id` events;
  a conversation ID is taken only when all copies agree; spans under a schema
  URL other than the GenAI convention's (or none) are evidence only; each
  request repeats the conversation, so the overlap with the stored tail
  (prefix-suffix match allowing a pinned system message) is dropped, a final
  input-only turn is never a replay, and a tool call is appended once per call
  ID; a span's parent is looked up in its own resource first, else in the one
  resource of the trace that holds it; spans without a conversation are
  adopted by the ancestor claim with the smallest (start, span ID,
  conversation, resource); usage on a span with no messages is still recorded
  as a usage event.
- **Generic OpenAI chat arrays**: `[{role, content, tool_calls?}]` documents
  as written by aider, Cline/Roo, OpenHands, open-webui, inspect-ai logs;
  imported by command with `--format openai-chat`, since there is no location
  or structure to tell them apart reliably.
- **Codex AppServer event stream**: approvals, plan changes and tool
  execution events, once a stable source exists; starting and resuming
  sessions through it later (slice 13).
- **Meta Muse Code**, once a verified native file exists to build the parser
  from (slice 5).
- **pi** (the `pi` coding agent): JSONL sessions under
  `~/.pi/agent/sessions/<mangled cwd>/<time>_<UUIDv7>.jsonl`. One trial
  session exists on the operator's machine (2026-09-23), so its location is
  known and the file is kept when collection of that location starts; a
  parser is built when the operator uses it for real work (no slice yet).
- **Claude Code Web runs**: a Claude Code variant, through their exported or
  synced transcripts (slice 5).
- **A2A** (the agent-to-agent protocol): task streams and Agent Cards as an
  import format, once a real producer exists (slice 13).

Each is a new parser module plus a row in the origins table; nothing else in
the model changes.

Provenance (Bead and audit IDs by section) and the Python-version lessons behind some rules: `evidence/spec-provenance-model-sources.md`.

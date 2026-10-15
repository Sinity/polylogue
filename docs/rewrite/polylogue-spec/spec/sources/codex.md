# Codex

General rules (identity, companions, unrecognized content, merging, spools) are in [README.md](README.md).

## Codex

**Where:**
- `~/.codex/sessions/YYYY/MM/DD/rollout-<timestamp>-<thread id>.jsonl` — one
  file per thread (median 1.4 MB, max 599 MB, 103 files over 100 MB). A
  `rollout-*.jsonl.zst` file, if one appears (a third-party report says some
  Codex versions compress rollouts; none of the 9,696 here are), is read the
  same way after zstd decompression.
Codex's SQLite databases are collected as snapshots (WAL databases through
`sqlite_dbpage` with no temporary copy, rollback-journal ones through the
backup API), stored as page patches over a base (`storage/blobs.md`,
Provider databases):

- `~/.codex/state_5.sqlite` (217 MB) — threads (with the user-set `name`),
  spawn edges, thread attachments (pull requests, worktrees), projects.
- `~/.codex/goals_1.sqlite` — `thread_goals`.
- `~/.codex/thread_history_1.sqlite` — the paginated history projection, including `thread_items`, turns and projection checkpoints. Retain its independent item payloads and timing evidence. Until a schema-specific mapping is implemented, report `no_parser_yet`; never substitute a checkpoint or row count for content verification. A supported schema maps its decoded items to messages, blocks and tool outcomes before merging with rollout evidence (Message identity and Reconciliation below); a native projection item is not necessarily one message.
- `~/.codex/memories_1.sqlite` — Codex's memory-generation jobs (`jobs`,
  `stage1_outputs`, `consolidation_progress`): stored as an input, no parser
  yet (`no_parser_yet`). The memory documents are files (below).
- `~/.codex/queue_1.sqlite` — pending user submissions: the native queue stores a thread ID, item ID, timestamps and serialized payload. Retain snapshots with `no_parser_yet` until the payload schema is supported. Queued, edited or removed submissions are input evidence, not proof of a completed turn; never synthesize assistant responses or count queue revisions as new prompts.
- **Not collected, deliberately:** `~/.codex/logs_2.sqlite`, the diagnostic log. `status` lists it as excluded by rule. This exclusion does not apply to the submission queue.
- `~/.codex/history.jsonl` — prompt history: one line per prompt the
  operator submitted, `{"session_id", "text", "ts"}` (about 19,900 lines).
  Each line becomes an event `prompt_history` on the session it names.
- `~/.codex/session_index.jsonl` — `{"id", "thread_name", "updated_at"}`
  per thread: a thread title source, used when `state_5.sqlite` has no
  name for the thread.
- `~/.codex/memories/` — Codex's memory documents: `MEMORY.md`,
  `memory_summary.md`, `raw_memories.md`, `rollout_summaries/*.md`
  (Codex-written summaries of past work, named by time and topic, not by
  thread), `archive/` (superseded versions) and `extensions/`. All are stored as inputs and
  shown as account memory, like Claude Code memory files; they are not
  parsed into sessions. `memories/.git/` is not read.
- `~/.codex/transcription-history.jsonl` and `~/.codex/dictation-history/`
  — dictation history. Both are empty on this machine (Oct 2026), so their
  format is unknown. They are watched and stored when non-empty, with parse
  status `no_parser_yet` until a parser is written from real data; dictated prompts
  are the operator's own words, so they are worth keeping.

Excerpts: `examples/raw/codex/rollout-2026-head.jsonl`,
`rollout-2025-legacy-head.jsonl`, `compacted-record.jsonl`.

**Format vintages:**
1. **August 2025 ("legacy")**: the first line is a bare header
   `{"id", "timestamp", "instructions", "git": {commit_hash, branch, repository_url}}`,
   followed by `{"record_type": "state"}` markers and bare records with no
   shape: `{"type":"message", role, content:[{type:"input_text"|"output_text", text}]}`,
   `function_call`, `function_call_output`, `reasoning`. These files have no
   prompt echoes and no per-record times: a `role=user` message is a typed
   prompt (`written_by: human`) unless its text is one of the harness's
   tagged wrappers (`<environment_context>`, `<user_instructions>`, and the
   other tags listed below), and a model request runs from the first
   assistant-side item (reasoning, message, call) to the next user-side item
   (a user message or a call output). Every message's time is the header's
   `timestamp` for the first and unknown for the rest, never invented.
2. **October 2025 on**: every line is a wrapper
   `{"timestamp", "ordinal", "type", "payload"}`. The first record is
   `session_meta`.
3. **2026 additions**: `event_msg` subtypes `item_completed`, `token_count`,
   `task_started`, `task_complete`, `turn_aborted`, `agent_message`,
   `thread_settings_applied`, `thread_goal_updated`, `context_compacted`;
   `response_item` subtypes `custom_tool_call`, `agent_message`,
   `tool_search_call`, `web_search_call`; records `turn_context`,
   `token_usage_record`, `world_state`, `inter_agent_communication_metadata`.

The parser recognizes the vintage from the first line.

**Session fields:**

| Provider field | Model field | Rule |
|---|---|---|
| `session_meta.payload.id` (legacy: header `id`) | `native_id` | the thread id |
| `session_meta.payload.cwd` | `cwd` | |
| `session_meta.payload.git` (`commit_hash`, `branch`, `repository_url`) | `commit`, `branch`, `repo` | `repo` is the repository URL as recorded |
| `originator` (`codex_cli_rs`, `codex-tui`, `codex_exec`), `cli_version` | `harness`, `harness_version` | |
| `thread_source` / `source.subagent.thread_spawn` present | `kind=subagent` | |
| `threads.name` in `state_5.sqlite` | `title` | the name the user set (65 of 9,535 threads), merged with other routes by `sources/README.md` rule 6; `session_index.jsonl` `thread_name` is a display-title source of its own (`index`, `model.md`), below the provider's `title`. `threads.title` is the first user message, not a title, and is never used as one. |
| `session_meta.payload.base_instructions` | event `base_instructions`, `model_saw=1` | the system prompt the model saw; an event like `world_state`, not a message, so message counts are the conversation's (Morrow's counting rules agree) |
| `session_meta` `context_window`, `runtime_workspace_roots`, `thread_source` | session events | |
| `turn_context` `collaboration_mode`, `personality`, `effort`, `timezone`, `user_instructions`, `workspace_roots`, `permission_profile` | kept in the `turn_context` events | the settings in force per turn |
| `state_5.sqlite` `thread_attachments` (`pull_request`: `url`, `headBranch`; `worktree`) | `pr_link` rows (evidence `codex`) and session events | ready-made links for git correlation |
| `turn_context` (`model`, `cwd`, effort, approval policy) | events `turn_context`; `model` on following assistant messages | |

**Record types.** The items one model request produced (`reasoning`,
`function_call`, `custom_tool_call`, `web_search_call`, `tool_search_call`,
an assistant `message`, up to the `token_count` event that closes the
request) form one model message (`decisions/one-message-per-response.md`;
86% of requests have no assistant `message` item to carry their usage).
Its ID is the ID of its first item that is not `reasoning` (or that item's
fallback ID; the reasoning item's own ID only when the request has nothing
else), because rewrites drop reasoning items and the ID must survive a
lossy rewrite, or merging would split one request into two messages; its
blocks are the items' contents in record order. Each output item
(`function_call_output`, `custom_tool_call_output`) is a `role=tool`,
`written_by=harness` message holding one tool_result block.

| Record | Becomes | Rule |
|---|---|---|
| `response_item` `message`, `role=user` | message `role=user` | a harness wrapper (text starting with a tag other than `<image`, or with `# AGENTS.md`: environment context, instructions, `<user_instructions>`) → `harness`; otherwise the session's `session_meta.source` decides: `cli` or `vscode` → `human`, prose; `exec` → `harness`, the generated prompt of a non-interactive run (like Claude Code's `sdk`; the originator is kept in `attrs`); `subagent` → `model`, the parent thread's instruction, `from_session` the parent; an unknown source → `unknown`. A matching echo (below) confirms a typed prompt when Codex kept one, but is not required: Codex's rewrites drop the echoes (2 of 200 sampled files had any), so a rule that needed them classed 3.4% of the operator's prompts as typed (`evidence/codex-real-accounting.md`). |
| `response_item` `message`, `role=assistant` | part of the request's model message, `role=assistant`, `written_by=model` | `output_text` blocks → text; `phase` (`commentary` or `final_answer`, on 89% of assistant messages) is kept in the message's `attrs`, and the `dialogue` view keeps final answers |
| `response_item` `message`, `role=developer` / `system` | message `role=system`, `written_by=harness` | |
| `response_item` `reasoning` | thinking block of the request's model message | `summary` text when present; `encrypted_content` kept as an encrypted thinking block. A reasoning item repeated with the same ID and a growing `summary` is one item in versions; the last wins (`model.md`, Identity) |
| `response_item` `function_call` | tool_call block | `name`, `call_id`, `arguments` (JSON string, parsed); shell-like calls (`shell`, `shell_command`, `exec_command`) set `command`; `write_stdin` continues a running command (see "Tool calls and outcomes"); the multi-agent tools (`spawn_agent`, `send_message`, `wait_agent`, `followup_task`, `close_agent`, `send_input`, `list_agents`, `wait`) are tool calls whose `action` is `subagent` |
| `response_item` `function_call_output` | tool message with one tool_result block | paired by `call_id` |
| `response_item` `custom_tool_call` / `_output` | tool_call / tool_result | free-form `input`; `apply_patch` input is patch text |
| `response_item` `web_search_call`, `tool_search_call` | tool_call (and result when present: a `tool_search_output` with the same `call_id`) | |
| `response_item` `agent_message` / `event_msg agent_message` | a message in **the file's own session** (the record sits in the recipient's file): `role=user` when `recipient` is the file's agent path, else `role=assistant`; `written_by=model`; `from_name` = `author`; `from_session` = the thread in the same spawn tree whose `thread_spawn.agent_path` equals `author` (`/root` is the tree's root thread) | the sender's own tool call stays its tool call, and no second message is made there. 1,324 of 1,659 sampled records are in root threads with `recipient: /root`. Inter-agent payloads are often encrypted; kept as such |
| `event_msg item_completed` (`UserMessage`, `AgentMessage`, `Reasoning`, `CommandExecution`, `FileChange`, `McpToolCall`, `WebSearch`, `ImageView`, `Plan`, `SubAgentActivity`, `CollabAgentToolCall`, `ContextCompaction`, `Extension`) | the corresponding message or block when the same item is not already present as a `response_item`; for tool items, the outcome of the call with the same `call_id` (see "Tool calls and outcomes"); `ContextCompaction` → event `compaction`; `Plan` → event `plan` (the todo state briefings show); `SubAgentActivity`, `CollabAgentToolCall` → events | item ids identify duplicates across the two record families (`item.id` equals the call's `call_id` for 64% of tool items) |
| `event_msg token_count` | usage of the request's model message | see "Usage"; `rate_limits` (98% of them) → event `rate_limit` for the subscription windows (`analysis.md`); `info: null` carries no usage |
| `token_usage_record` (2026-09 on) | not a usage source | it repeats what `token_count` reports for the same turn; using both would count twice, so only `token_count` is used and this record is kept as an event for checking |
| `inter_agent_communication_metadata` | kept with the `agent_message` it accompanies | paired 1:1 |
| `event_msg task_started`, `task_complete`, `turn_aborted` | events `turn_start`, `turn_end`, `turn_aborted` | `task_complete.last_agent_message` is not a separate message; `duration_ms`, `time_to_first_token_ms` and `turn_aborted.reason` are kept for time analysis |
| `compacted` (`replacement_history`, or older `message`) | event `compaction` with the replacement text, `model_saw=1` | the 2026 fields (`window_id`, `window_number`, `previous_window_id`, `latest_token_usage_record`, `retained_context`, …) are kept in the payload; they chain compaction windows and give the token count at compaction |
| `event_msg context_compacted` | event `compaction` | |
| `event_msg thread_goal_updated` | event `goal` | |
| `world_state` | event `world_state` | snapshots of AGENTS.md and environment the model saw; `model_saw=1` |
| `turn_context`, `thread_settings_applied` | events | |
| legacy `record_type: state` | ignored | carries no content |
| `event_msg` `thread_rolled_back` | event `rollback` (`num_turns`) | the last `num_turns` turns before it leave the active path |
| `retained_context` | event `retained_context` (`verified_answer`) | |
| `realtime_item` | events `realtime_session_started` / `realtime_session_closed` | |
| `function_call_output` without `call_id` (namespaced tool output: `id`, `name`, `namespace`) | tool result, paired by `id` when a call carries that `id`, otherwise counted as an unpaired result | |
| an empty rollout file | no session, not a refusal | 3 of 9,697 files |
| anything else | unrecognized | |

**Message identity:** the rollout normalizer uses the native IDs of the
records its grouping rules above identify as messages; otherwise it uses the
fallback rule of `model.md`, computed per record family. A canonical
completed item or SQLite UI item is not automatically a message ID: its
representation coordinates are `(thread_id, turn_id, item_id)`. Keep that
scope and the contributing input references even when its contents map to
several normalized objects. Generated UI IDs such as legacy `item-N` values
are representation identities, not provider message identities; they do not
establish correspondence to a rollout message. The wrapper's `timestamp` and
`ordinal` are volatile (rewrites change them) and never enter a fallback hash. A typed
prompt is written both as a `response_item` message and as an echo (`event_msg
user_message` or `item_completed UserMessage`). They are one message when
their texts are equal and they belong to the same turn: the echo is the
first unpaired echo with equal text after the `response_item` and before
the next `response_item` user message. The message keeps the
`response_item`'s ID. Timestamps and the echo's `item-N` ID never decide
pairing (a 1 ms difference between the two records is common). The rule
applies across revisions of the rollout as within one. Equal text alone
never merges two prompts: the same prompt sent twice is two turns.

**Tool calls and outcomes:**
- Paired by `call_id`.
- Outcome and exit code, first source that has them:
  1. an `item_completed` item of type `CommandExecution`, `FileChange` or
     `McpToolCall` with the same `call_id`: `exit_code`, `status`
     (`completed` → ok, `failed` → error), `duration`, `command`, `cwd`;
  2. a structured `exit_code` / `is_error` in the output object, including
     older `shell` and `apply_patch` output, which is a JSON-encoded string
     `{"output": …, "metadata": {"exit_code": N, "duration_seconds": …}}` to
     be decoded;
  3. the header lines Codex writes on free-form results: `Exit code: N` /
     `Wall time: …`, and for `exec_command` the header block `Chunk ID … /
     Wall time … / Process exited with code N / Original token count …`
     (with a TTY, a `Command: …` line comes first). The exit line is a
     header, not the last line;
  4. pre-migration files only: `exec_command_end` / `patch_apply_end`
     events;
  5. a code-mode `exec` wrapper's first output line, which Codex formats
     itself: `Script completed` → ok; `Script failed` → error; `aborted by
     user` → error, reason `interrupted`; `Script running with cell ID N` is
     not a result yet (like `Process running` below);
  6. `apply_patch`'s own fixed result lines, which it writes instead of an
     exit code: output starting `Success.` → ok; output starting with its
     failure line (`Failed to apply patch`, `apply_patch verification
     failed`) → error. These are lines the tool itself formats, so reading
     them is structure, not meaning (rule R18).
  Exit code 0 → `ok`, non-zero → `error`. `Command blocked by PreToolUse
  hook: …` → `error`, reason `blocked_by_hook`. Nothing found → `unknown`,
  reason `result_without_status`.
- **Running commands.** An `exec_command` result saying `Process running
  with session ID N` is not a result. Later `write_stdin` calls carrying the
  same session ID are continuations of that command, not separate commands:
  their output is appended to it, and the one that reports `Process exited
  with code N` sets its outcome. Until then the command is `unknown`,
  reason `no_result`, read as pending while the session is live, even after
  later model responses: `passed_without_result` never applies to it, since
  its result can still arrive. A command still running when the session
  ends has no result (`model.md`, tool outcome).
- **Code mode:** an `exec` call runs a program. It is one tool call
  (action `code`) whose result is its output list. Child calls are made only
  from records that state an operation: each `item_completed` item
  (`CommandExecution`, `FileChange`, `McpToolCall`) written between the
  wrapper call and its output becomes a child tool_call/tool_result pair
  with the item's own ID, its command and its outcome. Nothing is derived
  from the program text or from output positions (call sites matched
  output elements in only 23% of 4,463 sampled calls; rule R76).
  A child belongs to the wrapper its item names, when it names one; else to
  the one wrapper open when it was written; when several wrappers are open
  at once (parallel calls) and the item names none, the child is kept and
  counted with its wrapper unknown, never assigned to the most recent (over
  the operator's 9,697 rollouts, 26,832 of 568,364 children, 4.7%, in 11
  files; Morrow #5 has one such case).
  Counts use the children for every wrapper that had one: its owned
  children, or an unowned child that arrived while it was open (the child is
  counted once); a wrapper with neither is counted itself
  (otherwise every command would count twice); `tool_calls where action:code` lists wrappers. This is
  the usual shape: on this machine `exec` is nearly every Codex tool call in
  2026 rollouts (9,617 of 10,026 in one October session).

**Usage:** `token_count` events report both "last request" and running
totals. A `token_count` event is a new request only when its
`total_token_usage` differs from the previous event's in the same input; an
event repeating the previous running total adds no usage and is kept as an
event for checking (7.7% of events repeat; summing them all overcounts by
5.1%). Per-request counts are the `last_token_usage` of new-request events,
on that request's model message; running totals are never summed with them,
and the sum of per-request usage equals the final `total_token_usage` or
the input is listed in `status` with the difference.
`cache_write_input_tokens` (in 83% of them) is `not_charged` (`analysis.md`, token rule 3: OpenAI bills no cache writes): it is not stored as cache-write tokens and adds nothing to cost; in 58,078 recent records on this machine it was always 0, and a nonzero value is kept in `attrs` and counted in `status`, never priced. `cached_input_tokens` is a subset of
`input_tokens`: stored as cache read, with input stored as
`input_tokens - cached_input_tokens`. `reasoning_output_tokens` → reasoning, a part of `output_tokens`, never added to it.

**Lineage** (kinds and `how`: `model.md`, Link):
- **Subagent.** `session_meta.payload.source.subagent.thread_spawn` (or the
  top-level `parent_thread_id`) with `parent_thread_id`, `depth`,
  `agent_path`, `agent_nickname`, `agent_role` → link `subagent` (4,879
  rollouts, half of all Codex sessions); the attributes go in
  `link.attrs_json`. `state_5.sqlite` `thread_spawn_edges` (4,131 edges,
  `status` `open` or `closed`) is a second source for the same links. The
  spawning tool call in the parent is the spawn call whose output names the
  child thread ID; **to verify in slice 2** (exact call name and output
  field). Until then the link is stored without a spawning block and
  `evidence` says so. A spawned thread replays its parent only when it
  carries `forked_from_id` or `subagent_history_start_ordinal`; one with
  neither starts empty (`how: fresh`; 25 of 25 checked). Items before
  `subagent_history_start_ordinal` are copies by Codex's own statement
  (review threads carry it too); when no parent is named they get
  `copy_origin: unknown` and no `inherited_from`, are counted nowhere, and
  the link stays pending with evidence `subagent_history_start_ordinal`.
- **Forks.** `forked_from_id` (1,007 rollouts). On a spawned thread (1,006 of
  them) it sets the subagent link's `how`; on a thread that is not spawned
  it is a `continuation` link, `how: fork`. The parent's conversation reaches
  the child in one of three forms; a parser handles all three and counts any
  other shape as unrecognized:
  1. **Replayed prefix** (March 2026, and some later forks in paginated
     history, one replaying 26,051 items): the child starts with copies of
     the parent's items, re-stamped with the fork time and **without item
     ids**. They are matched to the parent by content: the longest run of
     leading child items whose type, role and content (or call id, arguments
     and output) equal the parent's items in order.
     `session_meta.subagent_history_start_ordinal` (when present) marks
     where the child's own records start; the content match never extends
     past it. Matched messages get `inherited_from`; `how: copied`.
  2. **Compacted history** (July 2026): the child's first records are a
     `compacted` record whose `replacement_history` holds the parent's
     conversation. It is a compaction event at the start of the child
     (`model_saw=1`), not messages; nothing is inherited; `how: summarized`.
  3. **Context not written down** (September 2026 on, first record
     `thread_settings_applied`): the file records no parent content, but
     Codex started the child from the parent's context (the fork flag says
     so). `how: copied` describes how the child got its context, whether or
     not the file holds the copy; `attrs_json.context = "none_recorded"`
     says the copy is not in the file, so nothing can be marked
     `inherited_from`.
  Some July 2026 child files stamp every record with one timestamp; their
  message times are stored as recorded, and durations inside such a session
  are `unknown` rather than zero.
- **Resume.** Current `codex resume` appends to the same rollout: growth,
  not a link. Older Codex wrote the resumed conversation into a new
  rollout that starts with its own `session_meta` and then repeats the
  parent's original `session_meta`, which carries another thread ID. Such
  a file is a `continuation` with `how: resume` from that thread when the
  repeated header's timestamp is not later than the file's own and the two
  headers have the same `cwd` or the same repository URL; the repeated
  records are matched to the parent as in "Replayed prefix" and marked
  `inherited_from`. A second header that fails this test is counted as
  unrecognized and creates no link. This follows the general lineage rules
  (copied messages plus a continuation link), so it is a technical rule,
  not a product choice (the rollouts with two headers
  are counted and the test checked against them in slice 2).
- Every spawn edge's child had a rollout on this machine; a child named only
  in `thread_spawn_edges` would get a pending link and no session until its
  rollout appears.

**What an incremental parse carries forward.** A rollout grows by appends,
and an append parses only the new records, from a carry
(`convergence.md`, parse). Unlike Claude Code's, Codex's later records
change messages already written:

- a request's message takes its ID from its first item that is not
  `reasoning` (so the ID can change after the message was written), joins
  later items until its `token_count`, and takes that usage;
- a reasoning item repeated with a longer summary replaces its block;
- a running command's result takes the outcome a later `write_stdin`
  reports;
- an `exec` wrapper's message gains its `item_completed` children until the
  wrapper's output;
- an unanswered `write_stdin` call's message is marked a continuation when
  its output arrives;
- an echo marks its prompt as typed.

So the carry holds, besides the session's fields: the request and echo
state; the call table (each call's tool and, for `write_stdin`, its process
session); the reasoning item and running-command positions; open wrappers
and their children; outcomes `item_completed` stated before their output;
the fallback-ID counter; and the **unsettled messages** in full. A message
is unsettled from the first one a later record can still change: the open
request, the prompt waiting for its echo, an open wrapper's call, a running
command's result, an unanswered `write_stdin` call, the last model message
(which an unattached `token_count` reaches) and the last message (which
events follow). If that last model-written message is a subagent prompt
still waiting for a possible typed echo, retain its preceding eligible model
message too. A matching echo can reclassify the prompt as human; an unattached
`token_count` arriving afterward must still reach the preceding model message,
as it would in a whole parse. That predecessor cannot be settled merely
because the unresolved prompt temporarily displaced it. These are logical
retention requirements, subject to the addressable-state and step bounds in
`convergence.md`, not a requirement to copy the retained prefix into every
continuation. Messages before the unsettled boundary are settled.

An append's parse returns the unsettled and new messages, and the writer
writes the session again from the first unsettled one; everything before
it stays as stored. A record that would change a settled message (a
reasoning item repeated after its message settled, seen in no real rollout
yet) makes the append refuse, and the rollout is parsed whole, which gives
the same rows. A rewritten rollout's older chains do not block appends to
its newest chain: once they are parsed and unchanged, an append merges its
records against the stored messages (`convergence.md`, appends); only the
rewrite itself, when the new chain first arrives, rebuilds the session
from every contributing chain.

Measured on the largest real rollout (599 MB, 148,022 records, carry
3.4 MB): a 29 MB append parses in 0.8–2.5 s and a 5–45-record append in
0.08–0.19 s per run, against 11–14 s for building the whole file, and the
index after eight appends equals a fresh build
(`evidence/codex-rewrite-deltas.md`).

**Codex state databases as inputs.** These are live SQLite files that Codex
keeps changing, stored as page snapshots (above; `decisions/raw-retention.md`).
The parser reads the tables it knows (`threads`, `thread_spawn_edges`,
`thread_attachments` in `state_5.sqlite`; `thread_goals` in
`goals_1.sqlite`) from a snapshot; goals become session events.

### Reconciliation and preservation

Treat rollout bytes, thread catalog, paginated history, migration ledger and prompt history as separate evidence. Resolve stored relative rollout paths against the configured Codex home. Enumerate readable rollouts independently of the catalog: a failed migration may leave usable bytes without a registered thread. Keep a prompt-only session reference pending when no conversation source is found; its prompt does not prove an assistant response ever existed.

The first `session_meta` header owns the file's session identity. Later headers can describe inherited history and must not replace it. Apply `subagent_history_start_ordinal` in the native ordinal domain before counting the child's own completed items. An inherited-only rollout can legitimately have no paginated items; retain its copied context without inventing missing child turns.

**Map representations before union.** Each supported schema declares how native completed items correspond to the rollout normalizer's messages, blocks and tool outcomes, including any native identity scope and request-boundary evidence it uses. Apply that correspondence before the generic message union (`sources/README.md`). For example, one completed command item can contain both a command and its output, while the rollout represents a call block and a separate tool-result message. Treating each as another message would duplicate the call or result. A UI item ID alone does not identify the model request that groups several assistant-side items; a schema that lacks that boundary cannot claim to reconstruct it.

Retain unmapped items and mismatching payloads with their source coordinates and provenance, and report the affected normalization coverage as incomplete. Do not guess grouping or count deduplication from equal text, traversal IDs, row counts or projection checkpoints. Only established correspondence can consolidate their contributions to message, tool or usage counts; unresolved evidence is not another established generation. Existing rollout grouping, echo pairing and copy evidence still govern those records. Unsupported schemas remain `no_parser_yet`; native decoding alone does not implement this correspondence.

Compare completed-item identities and payloads using the source version's native decoder, with explicit accounting for excluded inherited items, duplicate/replaced identities, unsupported records and decode errors. Missing optional fields that decode to the same defaults are representation differences. Raw record-count reductions trigger comparison; they do not establish deleted messages. Check turns, timing, reasoning and other record kinds separately from message content.

Record the schema/decoder version, capture time and source digests for each comparison. Each SQLite snapshot is transactionally consistent, but separate databases and active rollouts are not one atomic snapshot. Pin a read transaction before a backup under sustained writes; a page reader must similarly prove one snapshot and reject mixed generations. Recheck active suffixes before classifying missing projection rows as stalled. Neither `quick_check` nor a cursor at EOF proves projection completeness.

Collect without modifying Codex state. A native projection repair is a separate operation: preserve a verified preimage, rehearse on copies, acquire native writer locks, revalidate source bytes and target rows, and prove idempotence and unchanged unrelated history. Failed legacy registration is a different operation from replaying a registered thread's suffix.

### Quirks

- Single lines can be several MB (large tool output). Reading line by line
  with ordinary JSON parsing is fine.
- 2025 thread ids are UUIDv4; 2026 ones are UUIDv7-like. Both are opaque.
- A thread's rollout can be written by several Codex processes over time
  (resumes append to the same file); this is ordinary growth.
- **Codex rewrites rollouts in place**, so a rollout is not reliably
  append-only. 98% of the current files were rewritten at some point, which
  added `ordinal` to every record; in about 7% the per-record timestamps
  collapsed to one or a few values. A history migration
  (`legacy_to_paginated_v1`, recorded in `state_5.sqlite`
  `rollout_migration_state`) also changed content: one file shrank from 217
  to 72 lines, losing `event_msg agent_message` and `user_message`,
  `patch_apply_end` and most `token_count` records, and gaining
  `item_completed` records. Across the 1,439 rewritten rollouts of August
  2026 (the pre-migration copies against today's files) every file had fewer records of some raw types: `user_message` echoes in 1,427 files and `agent_message` events in
  1,426, while about 90 files had larger net raw-count reductions
  (19,643 reasoning records, 3,944 message records and 15,470 custom call
  records with their outputs). These require version-aware payload comparison (`evidence/codex-rewrite-deltas.md`). The
  change classification sees a rewrite, so
  the new bytes become a new revision of the same input, stored as a delta
  against the previous one (under 7% of a plain copy, 0.94% at the median,
  over 1,431 real pairs), and every revision is
  kept and parsed (`storage/blobs.md`). The parser accepts both history modes for the same thread;
  merging by item identity keeps what only the old revision has. When two
  revisions of the same rollout disagree on a record's timestamp, the older
  revision's per-record timestamp wins (the rewrite is what collapsed them;
  "The same conversation through several routes", rule 5). The
  pre-migration copies in `sessions/2026/m8.tar.gz` are imported once by
  command ("Archives of provider files"), to preserve earlier byte representations; raw count differences do not
  establish exclusive ownership of missing conversation content; each member becomes an older
  chain of its rollout's input, so the import gives the index the rewrite
  route gives, in either arrival order; older originals kept by the
  Python archive arrive through the migration import (`plan/migration.md`).
- Wrapper `metadata.client_authored` (2.7% of records) may decide human
  versus harness on user messages more directly than turn matching; check it
  in slice 2 before relying on either.
- `internal_chat_message_metadata_passthrough` (about 10% of response items)
  is opaque: counted, not parsed. `reasoning.content` is always null; only
  `summary` carries reasoning text.

**Times in the August 2025 vintage**. Legacy records carry
no per-record time, only the header's `timestamp`. The first message takes
the header's `timestamp` and the rest have no time; the session's start is the header's `timestamp` and its last
activity the latest time any record carries (for a pure legacy file, the
start); nothing comes from file modification times.

Queue and projection reconciliation rules were checked against upstream Codex `rust-v0.162.1` (`092d3acd6bec3e3a14bdc7e7a2810ab628ab759d`), including `codex-rs/state/queue_migrations/0001_queued_items.sql` and the native thread-history materializer. Recheck version-specific wire mappings when supporting another release.

Provenance (Bead and audit IDs by section) and the Python-version lessons behind some rules: `evidence/spec-provenance-model-sources.md`.

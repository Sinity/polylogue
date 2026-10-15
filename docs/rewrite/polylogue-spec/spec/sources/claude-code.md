# Claude Code

General rules (identity, companions, unrecognized content, merging, spools) are in [README.md](README.md).

## Claude Code

**Where:**
- `~/.claude/projects/<cwd with / replaced by ->/<session id>.jsonl`: one
  file per session (4,650 files, 12.7 GB on this machine in Oct 2026;
  median 283 KB, max 1.6 GB).
- `~/.claude/projects/<slug>/<session id>/subagents/agent-<agent id>.jsonl`
  plus `agent-<agent id>.meta.json`: one file per subagent (11,694 files,
  7.2 GB).
- `~/.claude/projects/<slug>/<session id>/subagents/workflows/<run>/agent-<agent id>.jsonl`
  plus its `.meta.json`: the subagents a workflow run started, the same kind
  of file one folder deeper, owned by the session that ran the workflow (every
  record's `sessionId` is that session's; 480 transcripts and 460 meta files,
  320 MB on 2026-10-12). The run folder also holds `journal.jsonl`, stored and
  not parsed.
- `~/.claude/projects/<slug>/<session id>/tool-results/<toolu id>.txt`:
  large tool outputs moved out of the transcript (16,445 files, 3.5 GB).
- `~/.claude/projects/<slug>/sessions-index.json`: per-project index (first
  prompt, summary, counts); exists in few folders.
- `~/.claude/history.jsonl`: prompt history across sessions.
- `~/.claude/projects/<slug>/memory/*.md`: Claude Code memory files.
- `~/.claude/todos/`, `~/.claude/plans/`, `~/.claude/file-history/`: side data.

Not inputs, never read: `~/.claude/projects/.git/` (a user's own git backup
of the folder; the operator has one), `bridge-pointer.json`. Workflow run
records (`<session id>/workflows/wf_*.json`), workflow journals and memory
files are stored but not yet parsed: each chain gets `parse_state` status
`no_parser_yet` with detail `none planned` (no slice has their parser yet),
so `status` lists them as stored and waiting for a parser, never as
unrecognized records. Companion files (`.meta.json`, `tool-results/*`) are
read with their main input; their own chains get status `companion`, naming
the main input in `detail` (`storage/index-db.md`, `parse_state`). A
`tool-results/` file lives in a session's folder, `<session id>/`, and names
that session's main file, `<session id>.jsonl`, even when one of its
subagents' transcripts refers to it too; the subagent reads it as a
companion as well, and the detail names only the main file.

Each kind of file has its own `input.format` string:

| Files | `input.format` | Parse status |
|---|---|---|
| main and subagent transcripts (`*.jsonl`) | `claude-code-jsonl` | parsed |
| `subagents/*.meta.json` | `claude-code-meta` | `companion` |
| `tool-results/*` | `claude-code-tool-result` | `companion` |
| workflow records, workflow journals, memory files | `claude-code-stored` | `no_parser_yet`, detail `none planned` |
| `history.jsonl` | `claude-code-history` | parsed (prompt history, pastes) |
| `sessions-index.json` | `claude-code-sessions-index` | parsed (titles) |
| the repository's lookup record (`README.md`, the repo-lookup record) | `polylogue-repo-lookup` | `done` (joined by the derive step, not a companion) | A file is a subagent transcript by its place under `subagents/`, at
either depth, never by its name alone (`evidence/real-converge.md`: the
deeper layout was missed and never stored).

Excerpts: `examples/raw/claude-code/main-session-head.jsonl`,
`subagent-transcript-head.jsonl`, `compaction-boundary.jsonl`. The
real-data scan behind this section: `evidence/claude-code-real-data.md`.

**Format:** one JSON object per line. Common keys on most records: `type`,
`uuid`, `parentUuid`, `sessionId`, `timestamp` (ISO string), `cwd`,
`gitBranch`, `version` (CLI version), `isSidechain`, `userType`,
`entrypoint`, `agentId` (subagents). Versioned only by `version`; new record
types appear every few weeks (6 new ones in September 2026 alone).

### Which session a record belongs to

**The file is the session.** A main file `<id>.jsonl` is session
`claude-code:<id>`; its native ID is the file-name stem recorded in its
input's first observation, not any record's `sessionId`. A rename keeps the
input and therefore the ID; a copy at another path is another input, and
when its stem names an existing session it is a further input of that
session and merges by message ID. A subagent file `agent-<agent id>.jsonl`
is session `claude-code:<parent session id>/<agent id>`, where the parent
session ID is the `sessionId` its own records carry (the main session that
owns it, whatever folder the path runs through): agent IDs are short (1,931
files use the 7-character form) and are unique only within their session.
The same qualification names sidechain sessions, nested subagents and the
sibling a copied record's `agentId` refers to, and a hook event's
`agent_id` joins with its `session_id`. A path reached through a symlink is
the same input as its target (`observation.symlink_target`), so a subagent
file symlinked into a second session's `subagents/` folder is one session
with one parent. Every message record in a file belongs to that file's
session, whatever `sessionId` it carries.

A record's `sessionId` is evidence about lineage, not about membership:

- In a subagent file, records carry the parent's `sessionId`; that is
  expected and means nothing further.
- In a main file, records whose `sessionId` is not the file's own are
  copies of that other session's messages (a resume, a usage-limit
  continuation). The file's session gets a `continuation` link (`how:
  resume`) to each such session, with the last copied record as the branch
  point. Each copied message is marked `inherited_from` in one of two cases (`decisions/lineage.md`, when marks are set): the writer sets it when the evidence itself names the source (a copied `sessionId`, an `agentId` that differs from the file's); a copy recognisable only by matching a parent message's `uuid` gets it from the derive step when that parent session arrives. When the other session's file no longer exists
  (Claude Code deleted it; about 680 of 1,080 such files on this machine), the
  link stays pending, and copies of one deleted message held by several
  sessions count once (`decisions/lineage.md`).
- In a subagent file, records whose `agentId` is not the file's own are
  copies of a sibling subagent's records (302 files: a subagent continued
  into a new file). Same rule: a `continuation` link (`how: resume`) from
  `claude-code:<parent session id>/<that agentId>`.
- Records carrying `isSidechain: true` inside a **main** file are an older
  layout (Claude Code 0.2–2.0, 1,050 main files): a subagent's conversation
  written into the parent's file, usually without `agentId`. Each sidechain
  subtree, rooted at a sidechain record whose parent is a main-chain record
  (or null), becomes its own subagent session
  `claude-code:<session id>~side-<root uuid>` (or
  `claude-code:<session id>/<agentId>` when its records carry one), with a `subagent` link whose spawning block is
  the `Task` call whose result that subtree answered (evidence
  `isSidechain`). These records are not messages of the main session.

The same session file sometimes exists in two project folders (135 session
IDs, after the working directory was relocated); in every observed case the
smaller copy is a byte prefix of the larger. Both are inputs of the same
session and merge by message ID (`spec/convergence.md`, several inputs).

### Lineage evidence

Link kinds and `how` values are defined in `model.md` (Link). For each
relationship, the evidence in order of preference, with how often it is
present on this machine (Oct 2026):

| Relationship | Evidence (in order) | Presence |
|---|---|---|
| subagent: parent | 1. meta `parentAgentId` → `claude-code:<parent session id>/<parentAgentId>` (a nested subagent); 2. the folder `<session>/subagents/` → that main session; 3. the `SubagentStart`/`SubagentStop` hook event, which names the subagent's transcript path (see "Hooks") | folder: all 11,694 subagent files; `parentAgentId`: 1,165 (depth up to 5) |
| subagent: spawning block | 1. meta `toolUseId` (the parent's `tool_use.id`); 2. the parent's `tool_result` whose `toolUseResult.agentId` is the child's; 3. a parent `progress` record `agent_progress` naming the agent and its `parentToolUseID`; 4. `fork-context-ref.parentLastUuid` (gives the branch message, not the block); 5. none: the link is stored without a block. The child's parser records the call ID it knows (`spawning_call`) with the field that named it (`spawn_evidence`: `meta.toolUseId`, `toolUseResult.agentId` or `agent_progress.parentToolUseID`); the derive step resolves it to the parent's block (`spawned_by`) when the parent is stored | 1: 5,311; 4: 397 |
| subagent: `how` | meta `isFork: true` → `copied`; otherwise `fresh` | `isFork`: 477 |
| subagent: copied records | a record with a parent message's `uuid`, or an `assistant` record whose `message.id` and `requestId` equal a parent response (fork subagents re-emit the spawning response under a new `uuid`, with its usage); both are copies, marked `inherited_from` by the derive step when the parent's records are present (they are recognised only by matching them; `decisions/lineage.md`), and counted once through `counted_as` in the tree | 40 of 61 sampled forks |
| subagent: instruction | the first `user` message that is not a copy and not a tool result; in a fork subagent, the text block after the copied spawning call's result | |
| subagent: attributes (`link.attrs_json`) | meta `agentType`, `description`, `name`, `teamName`, `spawnDepth`, `model`, `worktreePath`/`worktreeBranch`, `stoppedByUser`; the Agent result's `status`, `totalTokens`, `totalToolUseCount`, `totalDurationMs` (kept for checking, never counted) | meta: 8,727 files (75%) |
| continuation, `resume` | copied records whose `sessionId` (main files) or `agentId` (subagent files) differs from the file's own | 400 main files with the parent present, ~680 with it missing; 302 sibling copies |
| continuation, `fork` | `forkedFrom` (`sessionId`, `messageUuid`) on copied records; the branch message is the last `messageUuid`; a `SessionStart` hook event with source `fork` | 9 files |
| continuation, `after_compaction` | `continued-in.continuedInSessionId` in the parent | 1 file |
| compaction (not a link) | `system` `compact_boundary` (with `logicalParentUuid`, which becomes the `parent` of the first message after the boundary) → event `compaction`; `microcompact_boundary` → event `microcompaction` | 681 and 92 files |
| team membership (not a link) | meta `teamName` and `name` → `session.team`, `session.agent_name` | 263 meta files |
| agent-to-agent message (not a link) | `user` record with `origin.kind = peer`: `written_by=model`, `from_name` = `origin.name`, `origin.from` kept in `attrs`. `from_session`, in order: (1) `verifiedPeerPid` with `verifiedPeerProcStart` matched to the harness process identity of a hook line on the same boot; (2) for a record carrying `senderTaskId` or a team context, the member of the recipient's own team (same lead session) whose `name` equals `origin.name`; otherwise none. A name never resolves across teams | 147 files (about 103 of 503 peer records are socket peers) |

Each row's evidence is stored as `model.md`'s strings: `meta.parentAgentId`,
`folder` or `hook-event` for a subagent's parent (its `spawn_evidence` is
`meta.toolUseId` when the meta file names the spawning call, else the
parent's `toolUseResult.agentId`); `copied-sessionId` or `copied-agentId`
for a resume; `forkedFrom` or `hook-event` for a fork; `continued-in` for
an after-compaction continuation.

A meta file of an unknown shape (one remote-task file has `taskId`,
`command`) is counted as unrecognized, and the subagent still gets its folder
parent. Why "fork" is mostly a `how` and not a kind: `decisions/lineage.md`.

### Session fields

| Provider field | Model field | Rule |
|---|---|---|
| file-name stem at first observation | `native_id` | see above |
| min / max `timestamp` over own message and event records | `started_at` / `last_activity_at` | min and max, not first and last: timestamps go backwards in 6,285 files; copies are left out (`model.md`, Session) |
| `cwd` (first record that has it; `relocated` records correct it) | `cwd` | |
| `gitBranch` | `branch` | as recorded on the first message record |
| repository | `repo` | in order of preference: (1) the `git` block the hook writer records at `SessionStart` (see "Hooks"); (2) `pr-link` records (`prRepository`); (3) the repo-lookup record of the session's repository directory (`README.md`, the repo-lookup record), joined by the derive step, not read by the parser. `repo` takes the first of (1)–(3) that names a remote; `repo_root` comes from (3). Storing the lookup as an input keeps the join a function of stored bytes, so a rebuild gives the same `repo`. Never derived from the folder name. In slice 1. |
| `version` | `harness_version` | `harness` = `claude-code` |
| `custom-title` record | `title` | the user's own rename; within the file it wins over `ai-title` and `summary` (across inputs, `sources/README.md` rule 6 decides) |
| `ai-title` record | `title` | used when there is no `custom-title` |
| `summary` record (older files) | `title` | used when neither of the above exists |

**Subagent sessions** take their parent, spawning block and attributes from
the folder and the companion `agent-<id>.meta.json` ("Lineage evidence"
above). `team` and `agent_name` come from the meta file's `teamName` and
`name`.

### Record types

Records are parsed one at a time; a record that fails to parse, or whose
shape is not the expected one for its `type`, is counted (unrecognized, or a
`parse_state` detail for invalid JSON) and skipped, and the rest of the file
is parsed. One bad line never loses a file.

| `type` | Becomes | Rule |
|---|---|---|
| `user` with string `message.content` or `text`/`image`/`document` blocks | message, `role=user` | `written_by` by the first rule that applies: `origin.kind` (`human` → human; `coordinator` → model, the parent model's message, with `from_name` the parent and `from_session` the parent session; `task-notification`, `auto-continuation` → harness; `peer` → model, see below); `promptSource` (`typed`, `suggestion_accepted` → human; `queued` with `origin.kind` absent → human, a prompt typed while the agent worked; `sdk`, the opening prompt of a non-interactive run → harness, with the source kept in `attrs`; `system` → harness); `isMeta` or `isVisibleInTranscriptOnly` true → harness; a typed slash command, recognised by `<command-name>` (with `<command-message>` and `<command-args>`), or `!` shell input or `#` memory input → **human** with `prompt_kind: command`, because the person typed it; its expansion and output (the command or skill body, `<local-command-stdout>`, `<local-command-stderr>`) → harness; other known harness wrappers (`<system-reminder>`, "[Request interrupted by user]", task notifications) → harness; the first message of a subagent session (the instruction the parent model wrote in its `Agent`/`Task` call) → model; a record with no `promptSource` from a version that predates the field (2.1.161 is the first version writing it, measured over the operator's 16,745 files: 26,672 earlier prompts; `evidence/claude-code-real-accounting.md`) → human; a later record with a `promptId` but no `promptSource` and no `origin`, not meta, not a compaction summary and carrying no harness tag → human, because Claude Code gives every submitted prompt a `promptId` (about 650 such prompts); otherwise `written_by: unknown`, counted under unrecognized with record form `user:<first tag>`. Human messages that are not commands have `prompt_kind: prose`. `image` and `document` blocks become `image` / `file` blocks. |
| `user` with `origin.kind = peer` | message `role=user`, `written_by=model` | a message from another agent: `from_name` and `from_session` on the message ("Lineage evidence") |
| `user` with `tool_result` blocks | message `role=user`, `written_by=harness`; one `tool_result` block per result | paired to the call by `tool_use_id`; see "Tool calls" |
| `assistant` | part of a model message, `role=assistant`, `written_by=model`, `model=message.model` | records sharing `message.id` and `requestId` are one message (below). Blocks: `text` → text, `thinking` → thinking (an empty body with only a `signature` is kept as an empty thinking block), `tool_use` → tool_call (`id`, `name`, `input`), `redacted_thinking` → thinking (encrypted, kept; not seen in Oct 2026 data), `server_tool_use`, `advisor_tool_result`, `fallback` → `other` |
| `assistant` with `isApiErrorMessage: true` (model `<synthetic>`) | message `role=assistant`, `written_by=harness`, no model | an error notice the harness wrote under the assistant role (3,579 records) |
| `assistant` `message.usage` | message usage, once per message | `input_tokens` → input; `output_tokens` → output; `cache_read_input_tokens` → cache read; `cache_creation_input_tokens` → cache write, split by `cache_creation.ephemeral_5m_input_tokens` / `ephemeral_1h_input_tokens` into cache write 5-minute / 1-hour (`cache_write_5m_tokens` / `cache_write_1h_tokens`) when present; `output_tokens_details.thinking_tokens` → reasoning (a part of output, never added to it); `speed` (fast mode), `service_tier`, `server_tool_use.web_search_requests` / `web_fetch_requests` → kept with the usage. Older records' per-record `costUSD` → `provider_cost_usd`, `durationMs` → `duration_ms`. `analysis.md` prices these. |
| `system` with `subtype: compact_boundary` / `microcompact_boundary` | event `compaction` / `microcompaction` | payload keeps `compactMetadata` / `microcompactMetadata` and `logicalParentUuid`. The following `user` record with `isCompactSummary: true` is not a message: its text becomes the compaction event's summary, the event is marked `model_saw=1`, and a record whose `parentUuid` names it resolves, like any event record, to the nearest ancestor record that becomes a message (`sources/README.md`); across the boundary that ancestor is given by the boundary's `logicalParentUuid`, which becomes the `parent` of the first message after it (the compaction row below). |
| `system` with no `subtype` (13 records on this machine, all `isMeta` with `content`) | event `system_note` with its content | a harness note, not a turn of the conversation, so not a message |
| `system` `turn_duration` | event `turn_duration` (`durationMs`) | turn timing for time analysis |
| `system` `api_error` | event `api_error` (`error`, `retryAttempt`, `maxRetries`, `retryInMs`) | |
| `system` `model_refusal_no_fallback` / `model_refusal_fallback` | event `model_refusal` (`apiRefusalCategory`, `refusedUserMessageUuid`, `originalModel`, `fallbackModel`) | |
| `system` `away_summary` | event `away_summary` with the recap text | a harness-written recap shown to the user when they return; not a turn of the conversation, so not a message |
| `system` `stop_hook_summary`, `local_command`, `scheduled_task_fire`, `bridge_status`, `informational`, other subtypes | event of the same name; `scheduled_task_fire` keeps `cron`, `prompt`, `taskId` | text the model saw (`hookAdditionalContext`) is marked `model_saw=1` |
| `attachment` (55 subtypes) | event `attachment` with its subtype; `model_saw` from `rendered` / `renderedRole` / `renderedInHumanTurn` | real files referenced with path and content (`file`, `edited_text_file`, `compact_file_reference`, `plan_file_reference`) also become a `file` block on the next message; `nested_memory`, `invoked_skills`, `skill_listing`, `hook_additional_context`, `queued_command`, `diagnostics` are kept in full for the `context` view; `total_tokens_reminder` and other pings are events only |
| `progress` with `data.type = agent_progress` | evidence for a subagent's spawning block | |
| `progress` with `hook_progress` | event `hook_run`: hook event, hook name, command | which hooks ran, from the transcript alone |
| `progress` with `bash_progress` | counted (per input, in the `unrecognized` table as a counted-only record type, `storage/index-db.md`); payload not stored, **except** for a tool call that never got a result: its last `fullOutput` becomes the call's partial output (the call still has no result) | the partial output is a prefix of the final result, so storing every tick duplicates data; for an interrupted call it is the only output there is |
| `progress` with `waiting_for_task`, `query_update`, `search_results_received`, `mcp_progress` | counted (per input, in the `unrecognized` table as a counted-only record type, `storage/index-db.md`), payload not stored | genuinely useless once the call finishes: each is an intermediate state of a call whose final result carries the same content |
| `fork-context-ref` | evidence for the subagent link | `parentSessionId`, `parentLastUuid` (branch point), `agentId` |
| `started` / `result` (with `key`, `agentId`) | events `workflow_started` / `workflow_result` on the subagent session | workflow subagent start and finish |
| `continued-in` (`continuedInSessionId`) | link `continuation`, `how: after_compaction`, to that session | rare, but explicit |
| `pr-link` | event `pr_link` and session `repo` (see above) | |
| `permission-mode`, `mode` | event `permission_mode` / `mode` | |
| `queue-operation` | event `queue` | the queued prompt text is kept in the payload |
| `last-prompt` | event `last_prompt` (keeps `leafUuid`) | |
| `summary` | title source (above); keeps `leafUuid` | |
| `file-history-snapshot`, `file-history-delta` | event `file_history` | payload kept as-is; the `changes` view uses them |
| `cost-state` | event `provider_cost` | the CLI's own cost total and its "incomplete" flag; not used for cost computation |
| `bridge-session`, `frame-link`, `worktree-state`, `agent-name`, `agent-color`, `relocated` | events of the same names | |
| `atis-latch` (30,047 records; the `atis` string is empty in 94%, 16 characters in 6%, longer in a handful) | counted under unrecognized, string kept in the stored input | not empty, but nothing known to map it to |
| `launched`, `artifact-autoreact-ledger`, `artifact-comment-monitor`, anything else | counted under unrecognized | |

**Skipping heavy records cheaply.** The parser reads each line's `type`
before deserializing the rest. Lines whose payload is only counted
(`progress` other than the kinds kept above) are counted without building
their payload. This matters: the 1.6 GB file is 1.54 GB of `progress` lines,
some 70 MB long. The bytes still go to the blob store unchanged.

### Messages and order

**One model response is one message** (`decisions/one-message-per-response.md`).
Claude Code writes each content block of a response as its own `assistant`
record; the records share `message.id` and `requestId` and repeat the
response's `usage` (partial values while streaming; the last record's is
final). On this machine 2.44 M assistant records are 1.25 M responses.
Records with the same `message.id` become one message when their
`requestId`s are equal or absent on both: its ID is the first record's
`uuid`; its blocks are the records' blocks in record order, numbered by the
block-ID rule of `model.md` (Identity; one counter over every item,
`thinking` included, since Claude Code never drops thinking on a rewrite); `apiBlockIndex`, where present, is kept in `attrs` and checked
against that order, a disagreement counted in `status`, never used to
renumber; its usage, `stop_reason` and model come
from the last record; its `at` is the first record's time and its
`completed_at` the last's. Its `parent` is the message of the first
record's `parentUuid`. Every record's `uuid` maps to that message, so
`parentUuid` links naming any of them resolve to it. A later record whose
`parentUuid` resolves into a message that descends from this one (a tool
result returned while the response was still streaming) is not a tree
edge: that result message is ordered after the record whose `tool_use` it
answers, and no cycle forms. Branches are found on the record tree before
folding (`README.md`, "Parents skip records that are not messages").

**Duplicates:** records with the same `uuid` and identical content are one
(7,866 duplicate `uuid`s within files on this machine).

**Order comes from the record tree and the file, never from timestamps.**
The conversation tree comes from `parentUuid`; within siblings, file order.
Branches (retries, edits) become messages with `on_active_path=false` for
the abandoned side. The active path is the chain from the `leafUuid` of
the last `last-prompt` or `summary` record in file order back to the root; a file with
branches and no such leaf has its branch messages' active-path status
unknown, not guessed from the last record (`model.md`, Message). `forkedFrom` (`sessionId`,
`messageUuid`), present on records of forked conversations, names the source
of a copied message ("Lineage evidence").

### Tool calls

An `assistant` `tool_use` block opens a call (`id`, `name`, `input`; for
`Bash` the command is `input.command`). A later `user` record's
`tool_result` block with the same `tool_use_id` answers it
(`sourceToolAssistantUUID`, when present, names the calling record
directly). Pairing follows the general rule ("How sources work"): a result
answers exactly one call, never a later one, and a `tool_use` re-emitted
under a new message with the same `id` is the same call, paired once.
- `outcome`: `error` if `is_error: true`, or if `toolDenialKind` is present
  (`permission-rule`, `automode-blocked`, `automode-unavailable`,
  `user-rejected`, `automode-parsing-error`; reason `denied:<kind>`), or
  if `toolUseResult` says `interrupted: true` (reason `interrupted`) or
  `timedOutAfterMs` (reason `timed_out`); otherwise `ok`.
- `exit_code`: from `toolUseResult` when present, else parsed from the
  output text (`Exit code N`, `exit code: N`). `returnCodeInterpretation` is
  kept with the result.
- Background tasks: a dispatch acknowledgement for a background command
  (`backgroundTaskId`) has `is_error: false` but is not the result; the
  call has no result until a later background-completion notification
  supplies the real outcome and exit code; one that never reports back is
  `unknown`, reason `background_never_reported`. The same holds for an
  `Agent`/`Task` call whose result has `toolUseResult.status =
  async_launched` (or `isAsync: true`; 56% of sampled Agent results): the
  call stays without a result until the `task-notification` naming its
  agent ID, or the child's `SubagentStop`, supplies the outcome. Its
  duration runs to that notification, and the delegation card's answer is
  the child's last model message.
- Results moved to a side file: `toolUseResult.persistedOutputPath` names
  `tool-results/<id>.txt`. The side file is a companion of the session file
  (see "Companion files" in "How sources work"), so the parser receives it
  and the block's text is its content. If it is missing, the block keeps
  the inline placeholder and `outcome_reason` is `output_not_recorded`.
- Content inside `tool_result.content` may be `text`, `image` (kept as an
  image block) or `tool_reference`. Images and documents nested inside a
  `tool_result`'s content become file blocks of that result; the result is
  never left empty because its content was not plain text.
- An empty `tool_use_id` never pairs with another empty one; an unpaired
  call or result is kept; an unpaired call has no result.
- Kept from `toolUseResult`, in the tool_result block's `data_json`, for
  later views: `structuredPatch` / `originalFile` on edits (the `changes`
  view), and `agentId`, `status`, `totalTokens`, `totalToolUseCount`,
  `totalDurationMs` on Agent results (delegation cards).
- A call with no result is stored as `unknown`: reason
  `passed_without_result` when a later model response exists in the session
  (the harness gives one only after every non-background call of the earlier
  response has its result, so this one will never arrive), otherwise
  `no_result`, which the one rule in `model.md` (tool outcome) reads as
  `pending` while the session is live.

Kept in the message's `attrs` (`model.md`, Message) for agent analytics, on
assistant messages: `attributionAgent`,
`attributionSkill`, `attributionMcpServer` / `attributionMcpTool`,
`attributionPlugin`, `effort` / `perTurnEffort`, `thinkingDurationMs`,
`message.ttftMs`, `advisorModel`, `stop_sequence`, `message.stop_details`,
`usage.iterations`, `usage.fallback_credit`, `quotaLimits`
, `thinkingMetadata` and `maxThinkingTokens`
, and on API-error records
`apiErrorStatus` and `error`. `isAbortedMidStream: true` marks a response
the harness cut off: the message keeps what arrived and gets
`stop_reason = aborted`; it is a finished version for merging
(`README.md`, rule 2). Kept on user records: `promptId`,
`promptSource`, `origin`, `permissionMode`. On tool_use blocks, `caller`
is kept in the block's `data_json`. `system` hook summaries keep `hookCount`, `hookErrors`,
`hookInfos`, `preventedContinuation`, `stopReason`, `toolUseID` in the
event payload; `fork-context-ref.contextLength` is kept with the link.

**Deliberately ignored fields** (present, carry nothing to map, and not
counted at all, so they don't hide real changes; unlike counted-only record
types such as most `progress` records, which are counted): `slug`, snake-case
`session_id`, `message.diagnostics`, `message.container`,
`message.context_management`, `wireToolInputs`, `wireIngestContext`, the
`serverClassifier*` fields, and `rendered*` on attachments. `requestId` is
used only to group a split response.

**Joining `tool-results/` files**. A file
`tool-results/<toolu id>.<ext>` belongs to the result with that tool-use ID.
When the result has no `persistedOutputPath`, its inline preview names the
file in text Claude Code writes: `Full output saved to: <path>` (older:
`Output has been saved to <path>`); the parser uses that path. Claude Code
also writes many small results to this directory while keeping them inline
in full; such a file repeats the inline text and changes nothing (it is
stored like any input). Subagent transcripts write into their parent
session's `tool-results/` directory, so a file there belongs to whichever
transcript, main or subagent, holds its tool-use ID; the directory is
checked once per session, so one file is never counted missing once per
sibling transcript. `hook-*` files in the same
directory are raw hook output, a separate capture, not tool results.

### What an incremental parse carries forward

When a file grows, only the new complete lines are parsed. To give the same
result as a full parse, the parser retains the following logical state from
the earlier part (ids and positions, with open response content where needed):

- the map from every record `uuid` to its message ID, and from each open
  `message.id` + `requestId` to its message and its next block index (a
  split response can continue across the append; its new blocks come back
  as continued blocks, numbered after the earlier ones, an addition to the
  stored message, never a competing version);
- the set of seen `uuid`s (duplicate detection);
- the `parentUuid` of every message (the tree), so the active path can be
  recomputed;
- open tool calls (`tool_use_id` → block ID) and pending background tasks
  (`backgroundTaskId` → block ID);
- a compaction boundary still waiting for its summary record;
- the session fields found so far (first `cwd`, branch, title candidates,
  min and max time).

This state grows with record count; it is not necessarily a small in-memory
object. Large UUID, seen-record and parent maps use addressable disposable
index state with bounded lookup/update, rather than copying, decoding or
serializing every preceding entry on each append. A bounded cache may serve
those lookups. The resume record in `parse_state.carry` identifies the committed
state at its byte position; state updates and that checkpoint commit with the
rows they describe (`convergence.md`, Parse). All retained caches, open responses
and temporary serialization buffers count toward the memory limits. A parser
version change or an index rebuild reconstructs this state from retained inputs;
it is never another source of archive facts.

The state is stored with the input's parse position (`parse_state.carry`,
`storage/index-db.md`), so a restarted daemon continues without re-reading. An append can change rows
written earlier: a new last message moves the active path (earlier messages'
`on_active_path` flips), and a result can close an earlier call. The parse
returns these as row updates, and the writer applies them in the same
transaction. After a head rewrite the parse starts from the subsumed chain's
carry with every offset shifted by the first line's change in length, and
re-reads the first record, whose rows it replaces.

### Typed commands

A user record whose text is one line starting with `/name` (letters, digits,
`-`, `_`, `:`; `/compact`, `/model opus`) is a typed slash command,
`prompt_kind: command`, even without the `<command-name>` wrapper: real
transcripts carry bare ones, which had been read as prose prompts and
reported as unanswered open endings (`evidence/briefing-trial.md`). A path
such as `/realm/…` is not a command name.

### Pastes

The transcript keeps only a marker such as `[Pasted text #1 +40 lines]`. The
full paste exists only in the `UserPromptSubmit` hook event (see "Hooks")
and in `history.jsonl`. When either exists, the full text is attached to the
message as an event `paste` with `model_saw=1`, derived from the
`prompt_submitted` or `prompt_history` event that carried it and naming it
(those stay events of their own: what was submitted, and the history line); the message text itself keeps
what the transcript says. A `prompt_submitted` event joins the user message
of the same session with the same `prompt_id` when present; otherwise the
first user message of that session whose time is within 3 seconds of the
event. The event records how it was joined, `matched_by: prompt_id` or
`matched_by: time_window` (rule R76), so a time match is never presented as
provider evidence. This is matching within one session the hook names,
not linking sessions by time.

A transcript marker is provider-reported paste evidence, not a clipboard-source claim or proof of literary authorship. A time-window join stays a heuristic association. If the provider does not locate the paste in the resolved message text, `text_origin.reported_pastes` keeps `range: null`; neither the attached full text nor a clipboard match invents exact insertion coordinates.

### Attached files

- `sessions-index.json`: per-session summary and first prompt; its summary is
  a display-title source of its own (`index`, `model.md`), below the
  provider's `title`. Read as its own input.
- `history.jsonl`: prompt history (`display`, `pastedContents`, `project`,
  `timestamp`, and `sessionId` on 20,097 of 20,243 lines here). A line that
  carries a session ID becomes an event
  `prompt_history` on that session. A line without one is kept as a
  standalone record of the prompt (shown in project and day views); it is
  never attached to a session by matching project and time, because links
  come only from evidence in the data (`principles.md` #22).
- `memory/*.md`, `todos/`, `plans/`: stored as inputs and shown in reads of
  the session or project they belong to; not parsed into messages.
  `todos/<session id>-agent-<agent id>.json` belongs to the session (and
  subagent) its file name names; files under `plans/` belong to a session
  only when their name or content carries its ID, otherwise to the project.
  Session excision removes the files that belong to the session.
- **Background jobs** (`~/.claude/jobs/`; 2 jobs on 2026-10-12). The
  Python version declares only `jobs/*/adopt.json`, as a workflow fact
  (`adopt_manifest`), and reads neither `state.json` nor `timeline.jsonl`;
  shapes below are from this machine's two jobs (keys and counts only). Each
  job folder `<job id>/` holds:
  - `state.json`, rewritten in place: an input whose every revision is kept.
    It names the job's session (`sessionId`), the session it resumed
    (`resumeSessionId`, a continuation link with evidence `job_resume`), its
    child sessions (`children`, subagent links with evidence `job_child`),
    and the job's `name` (with `nameSource`: `user` or `auto`), `intent`,
    `template`, `cwd`, `state` and `tokens`. The job's name, intent and current
    state go to the session's `attrs`; a `user` name is a
    display-title source of its own (`job`, `model.md`), below the provider's
    `title`. `tokens` is a check
    against the session's counted usage, never a second count. The bridge
    account and organisation IDs go to `attrs` and are never exported.
  - `timeline.jsonl`, append-only: each line (`at`, `state`, `detail`,
    `text`) is one job-state event on the job's session. States seen:
    `working`, `blocked`, `done`, `failed` (3,227 lines, `working` almost
    all). `blocked` is live-state evidence that the session waits on the
    operator; `done` and `failed` end it with that outcome
    (`model.md`, live state).
  - `tmp/`: the job's scratch files written by the agent (278 files,
    58 MB in one job): not collected, like files a model writes in its
    working directory.
  - `adopt.json`, when present (none on this machine, and no sample or
    description of its contents found): stored with parse status
    `no_parser_yet` until a real one shows its shape.
  `~/.claude/jobs/pins.json` (a list, empty here) marks pinned jobs; stored,
  parsed into a `pinned` fact on the job. Parsed in slice 2, with live
  state; until then stored with parse status `no_parser_yet` (slice 2).

### Quirks

- **The first record is mutable.** Claude Code rewrites a session file's
  first record (its header) while appending the rest; in 194 files the only
  difference between two observations was record one.
  The format therefore declares a mutable first record, and such a change is
  classed "head rewritten, rest appended", not a rewrite (`storage/blobs.md`,
  observing a file): it starts a new chain that reuses the previous chain's
  bytes and subsumes it, and the parse continues from the previous chain's
  carry with offsets shifted by the first line's change in length,
  re-reading only the first record.
- **Any other rewrite is lossy** (`sources/README.md`, rule 5): Claude Code
  only appends, so a file rewritten in the middle (an editor, a sync tool, a
  restore) may have lost records. Every chain of the input is merged by
  rules 1–3: messages only an older chain holds stay messages of the
  session, and each message keeps its most complete version, the current
  chain's on a tie (`decisions/merging.md`).
- Some records lack `timestamp`; they get no time.
- A file captured mid-write ends in a partial line: not an error, the line is
  parsed when complete. A corrupt line in the middle of a file (seen once) is
  skipped and counted; the rest of the file is parsed.
- 59 session files are empty: stored, `parse_state.status = no_sessions`.
- Two JSONL files sit one level deeper than expected
  (`<slug>/<id>/<id>.jsonl`); they are parsed as main session files.
- The largest file (1.6 GB, 366 lines) is almost all `progress`; see
  "Skipping heavy records cheaply". It must not need more than a few hundred
  MB of memory beyond the file itself (`behavior.md`).

Provenance (Bead and audit IDs by section) and the Python-version lessons behind some rules: `evidence/spec-provenance-model-sources.md`.

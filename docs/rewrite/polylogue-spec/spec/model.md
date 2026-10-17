# Data model

What Polylogue knows about an AI session, independent of how it is stored
(`storage/README.md`) or which provider it came from (`sources/README.md`). The types here
live in the `polylogue-model` crate; every rule in this file lives there and
nowhere else.

## The entities

```
Input ──(parsed into)──▶ Session ──▶ Message ──▶ Block
                            │           │          ├── tool call / tool result
                            │           │          └── attachment
                            ├── Link (to another session: subagent, continuation, reference)
                            ├── Event (things that are not messages, including compaction)
                            ├── Run (a task and its attempts, when a harness records one)
                            ├── Process (the harness process and its children, from process evidence)
                            └── Delivery (context Polylogue gave the session)
Variant: a transformed copy (translation, summary) of any of the above, tied to its source revision
```

### Input

The exact bytes of one source file, export, capture or hook spool, as stored
in the blob store. An input has:

- an **ID**: the BLAKE3 of its bytes for an input stored complete, or a
  UUIDv7 for a location whose content can change (a growing log, a file
  rewritten in place, a database, a Drive file) (Identity, below);
- the **provider format** it is (`claude-code-jsonl`, `codex-rollout`,
  `chatgpt-export`, …);
- where it was found (machine and path, or export file plus member path, or
  "browser capture") and when it was first seen there;
- its revisions: every distinct content observed, each described by extents
  (appends, rewrites and database snapshots alike), all kept
  (`storage/blobs.md`, inputs, revisions and extents; `decisions/raw-retention.md`);
- the observations that saw it, with full file metadata.

Parsing is a pure function from an input (plus its companion files and,
for an appended file, the state the parser carried from the earlier part;
`convergence.md`) to sessions, messages, blocks, links and events. Running
it twice gives the same result.

### Session

One conversation or one agent run, as the provider understands it.

| Field | Meaning |
|---|---|
| `id` | `<origin>:<native id>`, for example `claude-code:3e0ae36a-57f3-4dbd-8994-7a701d37e958` |
| `origin` | which product it came from (table below) |
| `native_id` | the provider's own ID for the conversation or thread |
| `title` | the provider's title if it has one (`ai-title`, `custom-title`, ChatGPT title, Codex thread name); otherwise none. When routes disagree, `sources/README.md` rule 6 decides, using each route's `title_at` (the provider time stated with that title, `update_time` for ChatGPT; unknown when the route states none, and then it never outranks one that does). |
| display title | not a stored session field, but derived (`session_facts.display_title`) and used by every surface. This is the one rule: the operator's or an agent's current `title` note; else the provider's `title`; else a title the provider keeps outside the transcript (Claude Code's `sessions-index.json` summary, a job's user-given name, Codex's `session_index` thread name; source `index` or `job`); else, for a subagent, the first line of the instruction it was given; else the first line of the first human prose prompt (`prompt_kind: prose`) that is not a **known preamble**; else the session ID. A known preamble is a leading block of at least 3 lines that begins the first human message of 5 or more sessions. The set of known preambles is a derived product of the whole archive, not of one session: when it changes, every session whose first human message starts with an added or removed preamble gets its display title recomputed (`convergence.md`). The thresholds are constants in code. Every displayed title carries its source (`note`, `provider`, `index`, `job`, `instruction`, `first_prompt`, `id`), so surfaces can style a derived title differently. A derived title (`instruction`, `first_prompt`) is shown with a short structural label beside it (duration, tool calls, files touched), because echoed prompts collide: 166 Codex titles were shared by several threads. |
| `started_at`, `last_activity_at` | the first and last time among the messages and events the session counts as its own, leaving out flagged times (Time, below); unknown when there is none. "Own" is the counting rule's (`decisions/lineage.md`): a message that is not a copy, or a copy that counts in this session because its source is absent (`counted_as` unset). A copy whose source is present never moves a session's times; one whose source is absent does (Morrow L8's three copies give it 10:00–10:15, `examples/archive/FACTS.md`). A format whose records carry no per-record time states its session times in its section (Codex August 2025: the header's `timestamp`). |
| `cwd` | the working directory the provider recorded |
| `repo`, `branch`, `commit` | only from git evidence: what the provider recorded (Claude Code `gitBranch`, Codex `git` block, `pr-link` records) or the collect-time lookup of the repository at `cwd` (`sources/README.md`, Repository identity). Never guessed from the directory name. A repository is identified by its **root commit** (`repo_root`, the hash of its first commit), which stays the same across remote renames, forks and URL spellings; `repo` keeps the name or URL for display. The checkout (the worktree path) is a separate value, `checkout`. A session with no git evidence belongs to its directory only and is shown that way. |
| `collection` | the provider's own grouping of chats, when it has one: a ChatGPT project or custom GPT (`gizmo_id`), a claude.ai or Claude Design project (`project_uuid`). Stored as the provider's ID and its name, so "everything in my ChatGPT 'Rust' project" is a query. |
| `harness`, `harness_version` | for agent sessions: `claude-code 2.1.220`, `codex-tui 0.162.1` |
| `machine` | the host the session ran on, by machine ID (`sources/README.md`, Machine). This is the one rule: the provider's own field when it records one (none of the current providers does); else, for a session from a live provider file, the machine where that file was collected (`observation.machine`); else, for an import, the host the imported format records (a Polylogue export carries `machine`); otherwise unknown. |
| `models` | models that produced messages, with message counts |
| `kind` | `main`, `subagent` (the provider says so: Codex `source.subagent`, Gemini CLI `kind: subagent`, Hermes `source`; or it has an incoming `subagent` link), or `chat` (web chats). Whether a session continues another is read from its links, not stored here. |
| `team`, `agent_name` | for an agent that ran as a named member of a team (Claude Code `teamName`, `name`): the team and its name in it |
| `temporary` | ChatGPT temporary chats and similar |
| `inputs` | which inputs contributed to this session |

### Origins

| Origin | Product | Typical location |
|---|---|---|
| `claude-code` | Claude Code CLI | `~/.claude/projects/` |
| `codex` | Codex CLI / TUI / exec | `~/.codex/sessions/` and `~/.codex/*.sqlite` |
| `chatgpt` | ChatGPT web | account export, browser capture |
| `claude-ai` | claude.ai web | account export, browser capture |
| `claude-design` | Claude Design | claude.ai export, browser capture |
| `aistudio` | Google AI Studio | Google Drive folder "Google AI Studio" |
| `gemini` | Gemini app (gemini.google.com, phone app) | Google Takeout, My Activity "Gemini Apps" |
| `gemini-cli` | Gemini CLI | `~/.gemini/tmp/*/chats/` |
| `antigravity` | Antigravity | `~/.gemini/antigravity/` |
| `hermes` | Hermes agent | `~/.hermes/` |
| `grok` | Grok web | account export, browser capture |
| `grok-cli` | Grok CLI | `~/.grok/sessions/` |

The set of origins is open: the table lists those Polylogue parses itself, and a Polylogue export from another tool may carry any other valid origin name. A name matches `[A-Za-z0-9][A-Za-z0-9._-]*`, with case preserved and compared exactly. Unknown names matching this grammar are retained. The same validation applies to constructors and imported wire values. Short names are a convention, with no fixed length cap; native session and message IDs retain their separate opaque-ID rules. The leading alphanumeric excludes `.` and `..` as origin directory components in `export-format.md`.

Origins are chosen fresh for the new archive (the current
`claude-code-session` style is not carried over: the rewrite re-ingests everything and keeps no compatibility path, so short names cost nothing; decision: `decisions/README.md`). Adding a provider adds a
row; nothing else in the model changes.

### Message

One contribution by one author: a prompt, a model response, a tool result.

| Field | Meaning |
|---|---|
| `id` | see "Identity" below |
| `id_kind` | `provider` or `provisional` ("Identity", provider and provisional IDs) |
| `role` | the provider's role, mapped by its section's table to one of `user`, `assistant`, `system`, `tool`; the raw value is kept in `attrs` when it differs |
| `written_by` | the recorded content supplier: `human` (supplied by a person; this sender classification does not establish typing or literary authorship), `model`, `harness` (injected by the agent program: tool output wrapped as a user turn, hook context, skill text, the opening prompt of a non-interactive run), or `unknown` (a record matching no form the provider's section lists; counted per input under unrecognized, so a new harness wrapper shows up in `status` instead of becoming a prompt). Independent of `role`: many `user` messages are written by the harness. The same act gets the same value in every provider. |
| `prompt_kind` | for `written_by: human`: `prose`, or `command` (a typed slash command, `!` shell input, `#` memory input). "Human prompt" everywhere (prompt counts, turns, a briefing's last prompt, display title, `unanswered_prompt`) means `prose`; commands are counted as `human_commands`. A local command that ends a session does not make its turn abandoned. |
| `text_origin` | for `written_by: human`: revision-bound provider-reported paste evidence and separate clipboard content-match candidates, observation metadata and explicitly named proxy metrics. No typed/authored/pasted classification is inferred from clipboard matching. See Text-origin evidence below. |
| `model` | for model messages: `claude-fable-5-1`, `gpt-6.1-sol`, … |
| `at`, `completed_at` | when the provider says it happened; absent if the provider gives no time. A message assembled from several records has `at` = the first record's time and `completed_at` = the last record's. Latency figures use `at`. |
| `parent` | the message it replies to, when the provider records a tree (ChatGPT, Claude branches); across a compaction, the message before the boundary (`logicalParentUuid`), so the tree stays connected |
| `on_active_path` | true on the active path, false for abandoned branches (edits, regenerations; they stay stored and addressable), unknown when no marked leaf decides it (below; `index.db` NULL) |
| `inherited_from` | the message this one is a copy of, as the evidence names it: a message in another session (resume, fork, a subagent started with a copy), or an earlier message of the same session that a harness inserted again after a compaction (Hermes re-inserts the retained history as new rows; `sources/others.md`); see `decisions/lineage.md`. It names the genealogical source even while that source is not in the archive. |
| `copy_origin` | `known` for a copy whose source the evidence names, `unknown` when the provider states the message is a copy but names no source (Codex `subagent_history_start_ordinal` without a parent); absent for a message that is not a copy |
| `counted_as` | for a copy, the message counted in its place: the source when it is stored; while it is not, the copy chosen by the counting rule (`decisions/lineage.md`), which has no `counted_as` itself and counts. A copy of unknown origin is counted nowhere until its source is identified. Separate from `inherited_from` because the counted copy is not necessarily the one the others descend from. Repeated text alone never makes a message a copy: independently recorded usage is never discarded for looking like another message's. |
| `from_session`, `from_name` | for a message one agent sent to another (`written_by: model`): the sending session when it can be resolved, and the sender's name as the provider wrote it |
| `usage` | tokens as the provider reported them: input, output, cache read, cache write, reasoning; plus the provider's own cost figure if it gives one |
| `stop_reason`, `duration_ms` | when recorded |
| `in_progress` | true when the provider marks this version as still being produced (ChatGPT `status: in_progress`, a claude.ai block without `stop_timestamp`); used by the merge (`sources/README.md`, rule 2) and exported, never shown as a finished answer |
| `version_hash` | stored with the preferred version (`message.version_hash`), so a merge that meets the message again compares against it whichever input arrived first (law 4): the BLAKE3 of the record as canonical JSON without its format's volatile fields, the merge's tie-break and its test of "same version" (`sources/README.md`, rule 2) |
| `attrs` | provider fields kept as given that have no field of their own (Claude Code `attributionAgent`, `effort`, Codex `phase`, ChatGPT speech and recap markers, claude.ai citations): one JSON object, keys as the provider wrote them. A field used by a view or query gets its own field instead. |

### Text-origin evidence

`text_origin` is optional for human messages and absent when not evaluated. Its public JSON object has `message_revision`, `reported_pastes`, `clipboard_matches`, `clipboard_observation` and `metrics`. Empty evaluated lists mean no evidence found under the stated state, not evidence of typing. There is no legacy `authorship` or `pasted_spans` alias.

- `reported_pastes` is null when provider extraction was not evaluated, otherwise a list of provider reports: `{provider_ref, matched_by, range}`. `provider_ref` pins the source input revision and event/marker. `matched_by` distinguishes provider identity from a heuristic time-window association (`sources/claude-code.md`). `range` is `{start_byte,end_byte}` only if the report identifies that precise extent in this message revision; otherwise null and displayed at message level. A heuristic association remains labelled as such. A provider report is stronger evidence about a paste action than a content match, without establishing who composed its content.
- `clipboard_matches` is null when matching was not evaluated, otherwise a list of `{start_byte,end_byte,copy_refs,ambiguous,normalization}` candidates from joint decision 30. `copy_refs` holds every qualifying `{id,revision}`, deterministically sorted; `ambiguous` is true when more than one copy qualifies for that range. Even a single candidate does not prove source or insertion route. Overlapping candidates remain separate. `normalization` is `unicode-whitespace-v1`: strip outer Unicode whitespace and collapse inner runs to ASCII space, preserving case, punctuation and Unicode composition; minimum length is 40 normalized characters. Positions map to the complete original UTF-8 ranges, never normalized offsets.
- `clipboard_observation` is null when clipboard matching was not evaluated, otherwise `{coverage,session,through}`. The matching window is the cited Sinex session through the message event time; copies from other sessions or later times do not qualify. `coverage` is `full` only when clipboard and primary-selection observation has no gap in that window, otherwise `gap`. It covers these sources alone. Neither value classifies the unmatched remainder.
- `metrics` contains `operator_supplied_bytes` (all human-supplied text bytes), `provider_unmarked_bytes` (complement of the union of precise reported ranges; null when provider extraction was not evaluated or any associated report has unknown extent), and `clipboard_matched_bytes` (union of candidate ranges, null when matching was not evaluated). These are proxies, never authored or typed byte counts. Overlaps count once within each metric. The two evidence classes can overlap and are not additive partitions.

All ranges are half-open offsets into the exact UTF-8 message text projection: text blocks in block-index order joined by a single LF, with no rendering or normalization applied. Evidence pins the resolved Polylogue revision that defines that projection; it never transfers to an edited message, another preferred version or an excised revision. A copy reference also pins its immutable revision. Late input or changed coverage replaces the current result for the message/revision key, preserving earlier evaluation results. Unknown remainder stays unknown even under full clipboard coverage. Content matching cannot distinguish independent typing, loading a file or pasting an identical string. A causal insertion receipt would need a new instrumented producer binding the action to this document and revision; this spec and the matching prototype do not implement one.


**Active path without a marked leaf.** Only a leaf the provider marks (Claude Code `leafUuid`
in a `last-prompt` or `summary` record, ChatGPT `current_node`) defines the
active path; when routes mark different leaves, the merge rules choose one
(`sources/README.md`, rule 6). Without one, a session with branches has `on_active_path`
unknown on its branch messages rather than a guessed path; a session without
branches is entirely on the path (decision: `decisions/README.md`).

### Block

One piece of a message's content, in order.

| Kind | Holds |
|---|---|
| `text` | prose |
| `thinking` | model reasoning; may be empty with only a signature, or encrypted (kept as such). `data` is the typed encrypted payload; provider extras such as Gemini CLI's `subject` and `timestamp` are kept in `attrs` |
| `tool_call` | tool name, call ID, input (JSON), and for shell-like tools the command |
| `tool_result` | the call ID it answers when known, recorded output text (or a reference to a side file), `outcome`, exit code, duration and structured provider metadata in `data`; an unlinked result stays a result without an invented call |
| `image`, `file` | an attachment: reference to its bytes in the blob store, name, media type, and its **direction**: `input` (the person or harness gave it to the model), `output` (the model or a tool produced it; `produced_by` names the tool call or generation block only when established) or `unknown`. Missing bytes remain explicit acquisition state. Output files are fetched first: provider download links for generated files expire quickly. |
| `other` | a content type Polylogue does not interpret yet, kept with its type name |

Named opaque provider extras use `data` on ToolCall and ToolResult, and
`attrs` on other block kinds. These containers preserve supplied values,
including null, without flattening provider subtrees, replacing typed fields
or creating defaults for missing members. An extra named `data` on a thinking
block is therefore `attrs.data`, distinct from its encrypted `data` payload.
Field-origin names and bounded model metadata use the same containers;
opaque values retain their privacy and presentation qualification.

**Tool outcome** is one of `ok`, `error` or `unknown` with a stated
reason, all facts of the inputs; `pending` is a reading of `unknown`
computed when asked (below). `unknown` is used only when the input truly does not say; where the
outcome is derivable (an exit code in the output, an error marker), it is
derived. `unknown` and `pending` never count as success.

A call with no result is stored as `unknown` with reason `no_result` (or
`background_never_reported`, or a launch acknowledgement still waiting for
its completion notice, `sources/claude-code.md`); that is a fact of the
records and never changes with time. A call with reason `passed_without_result`
is never pending. A `no_result` call is shown and matched as `pending`
at read time while its session's live state (below) is `working`,
`waiting`, `idle` or `unknown`, and as `unknown` otherwise; for an input
stored complete it is always `unknown`. The split is computed with the live
state, never stored. A later append can still turn it into `ok` or `error`.
This is the one rule; the provider sections point here.

`outcome_reason` holds a code from this list (shown with spaces in text
output); a code this version does not know is kept as written:

| Outcome | Reason | When |
|---|---|---|
| `unknown` | `no_result` | the call was recorded, its result never was, and nothing later in the session rules one out |
| `unknown` | `passed_without_result` | the session went on past the call without its result: a later model response exists, which a harness gives only after every non-background call of the earlier response has its result, so this result will never arrive. Never applied to a call whose process is still running by the format's own evidence (a Codex `exec_command` answered `Process running with session ID N`, `sources/codex.md`): that call stays `no_result`, read as pending while its session is live, until a `write_stdin` reports its exit or the session ends (the order of records proves it; Sinex marks unended shell commands followed by a later command the same way) |
| `unknown` | `background_never_reported` | a background task was started and never reported back |
| `unknown` | `result_without_status` | a result exists but says neither success nor failure (no exit status in shell output) |
| `unknown` | `status_field_unreadable` | a status field exists but has the wrong type or an empty value (Hermes) |
| `unknown` | `json_marker_not_decoded` | the result carries a JSON marker whose JSON does not decode (Hermes `\x00json:`) |
| `unknown` | `output_not_recorded` | only a placeholder of the output was kept (an export that drops tool output, a missing side file) |
| `error` | `denied:<kind>` | a permission denial (`<kind>` as the provider names it) |
| `error` | `interrupted` | the person or harness interrupted the call |
| `error` | `timed_out` | the harness stopped it for taking too long |
| `error` | `blocked_by_hook` | a hook refused it |

An `ok` or `error` without a stated reason has no `outcome_reason`.

A structurally recorded tool answer is a result even when its text is empty
or no textual body is supplied. Preserve that recorded presence or absence;
do not invent output text. Empty output alone proves neither success nor
failure: the provider's explicit status, exit code or documented status
default decides, otherwise use `unknown` / `result_without_status`. A known
result is not `no_result` merely because its body is empty. Pairing and
outcome are separate evidence: an unknown call link does not invent a call
or erase a stated status. A provider's error text is kept beside the output,
not in place of it.

A thinking block's `signature` is stored and exported but is not content
when two versions of a message are compared or a copied message is matched
to its original: providers sign the reasoning again on every replay, so the
same reasoning in a resumed or forked session carries a different signature.

### Link

A relationship between two different sessions, established only by
evidence in the data (similar topics or nearby times never create one).
Relationships inside one session (edits, retries, regenerations,
compaction) are not links: they are the message tree (`parent`,
`on_active_path`) and events.

| Kind | Started by | `how` | Branch point |
|---|---|---|---|
| `subagent` | a tool call in the parent (Claude Code `Agent`/`Task`, a Codex spawn, Hermes) | `fresh` (the child starts empty), `copied` (it starts with a copy of the parent's conversation), `summarized` (it starts with a summary of it) | the spawning tool-call block |
| `continuation` | a person or the harness continuing the conversation in a new session | `resume` (the same conversation continued), `fork` (an explicit branch; both may continue), `after_compaction` (a new session after a context limit) | the last parent message the child carries |
| `reference` | something in the child naming the parent | `briefing`, `ping` (context Polylogue handed over; from the delivery rows, see below), `mention` (a full session or message ID, or a provider URL, written in a human or model message) | the referenced message, if one is named |

**Where `reference` links come from** (one rule; `sources/README.md` and
`user-data.md` point here). `how: briefing` and `how: ping` links come only
from delivery rows (Delivery, below): one link from each session a delivered
briefing or ping named to the session that received it, and only when the
delivery's outcome is `delivered`. The reference line the injected text
carries is for people and agents reading the transcript; it is not a second
source of links. `how: mention` links come from a full session or message
ID, or a provider URL that resolves to a session, written in a human or
model message. Short IDs, titles and similar wording never create a link.

"Fork" is mostly an attribute of a spawn, not a kind of its own: in real
data almost every Codex fork is a subagent started with the parent's
context, and Claude Code marks such a subagent `isFork`. The query language
keeps `link:fork` as shorthand for both meanings (`query.md`).

**Retries and how a result was used**. When the
harness records that a subagent call retries an earlier one (a Codex
retry, a job runner retrying a job), the link carries `retry_of` naming the
earlier spawning call; nothing is inferred from similar instructions. How
the parent used the child's result is recorded on the link only from
structure: `returned` (the child's result reached the parent as the
spawning call's result, or as its completion notification), `referenced`
(a later parent message names the child session or its result by ID), or
`unknown`. Neither claims the parent used it; text overlap is never
evidence of use.

A link has: parent and child session IDs (the parent may not be in the
archive yet, or ever: a provider can delete it first), kind, `how`, the
spawning block (subagents), the branch message, the input field that
established it (`evidence`), and the provider's facts about the
relationship as given (agent type, role, depth, worktree). `evidence` is
one of a closed list of strings, each naming the field as the provider
writes it: `folder` (a subagent transcript's place under its parent's
folder), `meta.parentAgentId`, `meta.toolUseId`, `isSidechain`,
`copied-sessionId` and `copied-agentId` (copied records naming another
session or agent), `forkedFrom` (Claude Code), `forked_from_id` (Codex),
`continued-in` (Claude Code's `continued-in` record), `hook-event` (a
`SubagentStart` or `SessionStart` hook line naming the relationship),
`delivery` (a briefing or ping, from the delivery row), `pasted-id` and
`provider-url` (a mention), `job_resume` and `job_child` (Claude Code job
records), `subagent_history_start_ordinal`, `thread_spawn` (the session
meta's `source.subagent.thread_spawn` or `parent_thread_id`) and
`thread_spawn_edges` (`state_5.sqlite`) (Codex), `subagent_spawned` (Grok CLI),
`parent_session_id` and `shared call IDs` (Hermes), `summary_parent`
(Antigravity), `branchParent` (AI Studio), `sessionId+kind` and `directory`
(Gemini CLI), `branching_from_conversation_id` and `shared node id`
(ChatGPT), and `message.id+requestId` (a fork's re-emitted response, matched
by the derive step). A provider section adding a kind of evidence adds its
string here. A link with two sources takes its `evidence` and input by a
fixed per-provider preference, never by arrival order: Codex
`thread_spawn` over `thread_spawn_edges`; Claude Code `meta.toolUseId` and
`folder` over `hook-event`; otherwise the source named first in its provider
section. The field that names a subagent's spawning call is recorded
separately (`spawn_evidence`: `meta.toolUseId`, the parent's
`toolUseResult.agentId`, or a parent `agent_progress` record's
`parentToolUseID`), because it says which call spawned the child, not
which session is its parent. The derive step resolves that call to the
parent's block once the parent is stored, or, when the child names no call
(Morrow #3), finds the parent's tool result whose `toolUseResult.agentId`
is the child's; either way the one link row, whose input is the child's,
gains `spawned_by` from the parent's side. A pair of
sessions has at most one link of each kind and `how` (a `briefing` and a
`mention` reference, or a resume and a fork continuation, can join the same
pair). A session has at most one
incoming `subagent` link, so subagents form trees, and a tree's cost is
summed once (`analysis.md`).

Messages copied from another session and text shared by bytes are
different things. `inherited_from` is set by the writer from the record's own evidence (`copy_evidence`) and by the derive step by matching once the parent is stored; `decisions/lineage.md` (when marks are set) lists every mechanism per provider. Where the evidence names the source, it is set whether or not that message is in the archive. Counting follows `counted_as` (Message, above):
while the source is absent, the copy the counting rule chooses counts as its
session's own (a copy of an excised session's message counts nowhere); when the source arrives, the derive step points every copy's
`counted_as` at it, and the copy takes the source's `written_by` and author
fields, so a copy written before its source arrived is corrected then.
Copies are never counted, costed or searched twice. Shared text (stored once by hash) only
saves storage and says nothing about lineage. A message one agent sent to
another is `written_by: model` with `from_session`; it is not a link. The
same session found in two places is one session, not a link. Per-provider
evidence is in `sources/`; the reasoning in `decisions/lineage.md`.

### Event

Things in a session's timeline that are not messages: `compaction` (the
context replaced by a summary; payload: summary text, before/after tokens),
`microcompaction` (named tool results or attachments removed from context;
payload: their IDs, tokens saved), model switch, permission mode
change, permission decisions and other hook events, working directory
changes, queue operations, file-history snapshots. Each event
has a time, a kind, and its payload as the provider wrote it. The kinds are
an open set: the provider files in `sources/` list the kinds each provider
produces, and that list is the one place they are named. A compaction summary is always the
event's text, never a message, in every provider. Events the
model actually saw as context (hook-injected text, compaction summaries) are
marked as such, so "what did the model see at turn N" can be answered.

### Unrecognized content

Every record type or field a parser does not understand is counted per input
(record type, field path, first and last seen). Nothing is silently dropped.
This is also how format changes are noticed: `polylogue status` reports new
unrecognized record types (`analysis.md`, drift).

### Account records

Inputs that belong to an account or a project rather than a session
(memory files, Codex Cloud tasks, project documents, prompt-history lines
without a session ID) are stored as inputs and parsed into no session. Views
list them by input format and path (`input` views, `rendering.md`); a
prompt-history line without a session is shown in project and day views by
its own time. They never create links or messages.

### Run

A unit of work a harness or job runner tracks across sessions: a **task**
(what was asked: a Codex goal or job, a job started by an external job
runner) and its **attempts** (each try at it, which may be one session, part
of one, or several). Runs come only from recorded evidence, named per
harness in `sources/`: Codex jobs in its state database, Codex goals (the
goal record, which also appears as a `goal` event in the session's
timeline), and a job ID that a job runner passes into the session's
environment, which the hook writer records. They are never inferred from
similar text or nearby times: two sessions serving one goal without such
evidence are not joined, and one session can serve two tasks. A subagent is not a run: its task,
spawning call and retries are already on its `subagent` link (Link, above).

A run has an ID (`<origin>:<native run id>`), its kind (`task` or
`attempt`), its task (for an attempt), the instruction as given, the state
the harness recorded (`running`, `completed`, `failed`, `cancelled`,
`unknown`), its times, and the sessions that carried it out. The state is
what the harness said; it is not an observation that the work was done, nor
a judgment that it was accepted (`analysis.md`, what a record proves).

### Delivery

One occasion on which Polylogue gave context to a session or a notice to the
operator: a hook briefing, a takeover briefing, an MCP `brief` result, a
ping, an advisory (`surfaces/mcp-and-hooks.md`), or a signal's notice
(`user-data.md`, signals; recipient `operator`, with the rule, the rows that
fired it and the rule's author). An advisory's cooldown is read from its earlier
delivery rows, never from a separate timer record. The
prepared content (a briefing, identified by its content hash) and its
delivery are different things: two deliveries of the same bytes are two
deliveries, while recording the same delivery twice is still one.

A delivery has an ID (a UUIDv7 chosen when it is sent and carried in the
reference line of the injected text), the recipient (a session, which may
not be in the archive yet, or the operator), the event that triggered it, its time, the content
hash, the references included and left out, and the outcome the sender
observed: `delivered` (handed to the harness), `failed` (with the error),
`unknown` (no acknowledgement), or `not_sent` (prepared but not handed over,
with the reason). Whether the model read or used it is not recorded: the
host cannot observe that.

### Variant

A transformed copy of some content: a translation, transliteration,
simplification or summary of a session, message, block, text span or note,
tied to the exact revision it was made from (`seen_hash`). Variants are
content, not claims, so they are not notes. They are read
next to the original, kept out of the original's search results unless
asked for, and shown as stale when their source changes.

A variant records its kind, its languages, its author (a person or a model
session), its coverage (whole or partial) and its alignments: which span of
the variant comes from which span of the source, with the relation
(`translates`, `transliterates`, `simplifies`, `summarizes`, `omits`,
`expands`, `reorders`). Exact character maps are kept only where the copy
is exact; translations and summaries align by passage and say so. Offsets
name their unit: bytes, Unicode code points, or UTF-16 positions for text
selected in a browser. Variants arrive in slice
12.

### Configuration in force

Each revision of an instruction file, skill or settings file is identified
by its path and the BLAKE3 of its content. It is valid from the earliest of
an `InstructionsLoaded` event naming its hash (the session's clock) and its
first observation (Polylogue's clock), up to, but not including, the next
revision's start by the same rule. Git history adds revisions that were
never observed, dated by the committer date (`time_quality: intrinsic`, `precision: second`; Time, below);
these never shorten an observed revision's validity. Asking what was in
force at a time gives one revision; `overlap`, when two different revisions
are valid at that moment (starts equal to the millisecond, or from
different clocks within a minute of each other, are both kept and reported
as overlapping, never ordered by a guess); or `gap`, when no revision is known, which is never
filled from the file as it is today. History is read in full. The
evidence comes from `InstructionsLoaded` hook events and git history
(`sources/README.md`); slice 6.

### Process

A process of an agent harness: the harness process itself (`role` harness)
or one of its descendants (`role` child, such as a test run or a build),
identified by its process identity (boot ID, PID, start time), with parent
PID, command name, argument list, working directory, start time and end
time (none while alive). Process facts come only from the process identity in
hook lines and from the daemon's process spool (`sources/README.md`). A child
is linked to a tool call only by evidence: the harness is its ancestor and
its start falls between that call's pre-tool and post-tool hook events;
otherwise it stays unlinked, and it is never matched by command text.
Subagents run inside their harness process (Claude Code) or as threads
(Codex), so they are not processes. Slice 2; tool-call linking slice 6.

### Live state

What a session is doing now, computed when asked from stored facts and the
current time, never rewritten by a clock. It is defined for `now` only, from
facts whose time is not after `now`; a fact dated after `now` (a clock step)
counts as at `now`. Rows apply in order and the table is total:

| State | Rule (the first that applies) | Shown as evidence |
|---|---|---|
| `ended` | the last hook event is `SessionEnd`, the harness process is gone after a `Stop`, or the session's Claude Code job timeline's latest state is `done` or `failed` | the event, process line or job state |
| `stale` | the harness process is gone with no `SessionEnd` and no `Stop` after the last turn (died, killed, the machine rebooted, or it ended while the daemon was down) | "process gone at T, no end event" (a range when the end was found at daemon start) |
| `waiting` | the session is live and waits on the operator | what it waits for, since when |
| `working` | the session is live and the last event is not `Stop` or a wait (a turn is in progress) | the open tool call, or "last event at T" |
| `idle` | the session is live and the last event is `Stop` | since when |
| `unknown` | no hook or process evidence, and the transcript grew within the live window | "inferred from file growth" |
| `stale` | live evidence (a hook line, a process line or file growth) within the last 24 hours, none within the live window | "no live evidence since T" |
| `ended` | no live evidence in the last 24 hours | "no live evidence; last activity at T" |

A session is **live** when its harness process is alive (`sources/README.md`,
Process spool, says when a root counts as alive), or, with no process
evidence, when a hook event came within the **live window** (15 minutes,
`glossary.md`). Each computed state carries `until`: the time at which it
changes with no new fact (the end of the live window or of the 24 hours),
or none. A view that follows the change feed recomputes at `until` as well
as on each feed line.

A session **waits on the operator** when a permission request, a
`Notification` whose type asks for a decision (`permission_prompt`,
`elicitation_dialog`) or a Codex approval request has nothing after it, or
when a `Stop` ends a turn whose final model message asks the operator a
question (the `question` reason of `ending`), or when its Claude Code job
timeline's latest state is `blocked` (`sources/claude-code.md`, jobs). An idle notification
(`idle_prompt`) is evidence for `idle`, not a wait; a notification of
unknown type is kept and decides nothing. Codex reports approvals through
the same permission hook event; its rollouts carry no approval records
(none in 225 rollouts from October 2026).

A subagent is `working` between its `SubagentStart` and `SubagentStop`
events while its parent's process is alive; otherwise it takes its parent's
state.

A session's **live task**, in order of evidence: a job variable in its hook
lines (an AgentCTL job, whose job record is an input of its own kind joined by the derive step, `sources/README.md`, run evidence); the
latest task claim the session itself made by running `bd update <id>
--claim` or `--status in_progress` (recognized from the command's structure,
not from prose), not followed by a `bd close` or release in the same session;
otherwise the harness's in-progress todo item, shown as text and never turned
into a task identity. A Bead ID only mentioned in a prompt or message never
makes a task claim.

**Peers** of a session are the other sessions whose live state is `working`,
`waiting` or `idle` and that share its repository (`repo_root`) or its
checkout, or, when either session has no git evidence, a working directory
where one is the other's or lies below it, by whole path components and
exactly (`query.md`, Fields: `/src/tide` and `/src/tide/web` are peers,
`/src/tide` and `/src/tidepool` are not). They are found by query, never
stored.

### The work graph

Not a store of its own but a view over what is already recorded. Its nodes
are sessions, runs, tool calls, files, commits and PRs, notes and, from
slice 13, issues from a task tracker (Beads first) joined by issue ID. Its edges are links, run membership, tool calls that
read, wrote or edited a file, commit and PR links, and notes that claim,
answer or judge. Three things stay apart: a **claim** (a
note or message saying something is so), an **observed effect** (a tool
outcome, a file edit, a commit), and an **evaluation** (a judgment note).
Being connected in the graph never means serving one goal.

## Identity

**Kinds of sameness.** "The same" means one of these, and the spec names
which; each is established by its own evidence and implies none of the
others, because an unqualified "same" slides from "these records look
alike" to "these are one message":

| Sameness | Established by | Does not establish |
|---|---|---|
| same bytes | BLAKE3 of the bytes | same input, record or message |
| same record location | input, revision chain and byte range (`storage/blobs.md`: offsets are stable within a chain, which only appends; a head rewrite starts a new chain, and a record's range in the new chain is its range in the previous one shifted by the first line's change in length, so the two locations hold the same bytes) | same provider record in another export |
| same provider record | the provider's own ID for it, within its session | same version of it |
| a newer version of a record | the format's revision order; for changing values (title, current node), the provider's update time; never arrival | that the newer version is more complete |
| same message | the message ID: a provider ID, or a provisional ID (below) where the provider gave none | same text in another session |
| counted once | the copy and counting rules (`decisions/lineage.md`, `analysis.md`) | that the other records are discarded |

The merged session view is the **preferred version** of each message under
the merge rules (`sources/README.md`): finished before in progress, then the
most faithful route, then the most complete record, then the hash
tie-break. It is an archival policy for which retained version to show and
count, not a claim about the provider's latest intended state; every other
version stays in its stored input, readable with `--view raw`.

IDs are opaque strings: code never splits an ID to find its parts (a native
ID may itself contain `:`); the session of a message is a field, not a
prefix to parse. An ID a person types or pastes is resolved by looking it up
(an exact match, or a unique prefix of a stored ID), never by splitting it.
When a typed ID prefixes several session IDs and exactly one candidate C is
such that every other candidate begins with C followed by `/` or `~` (its
subagents and sidechains), C is the match. A typed ID containing `/` matches
a stored ID whose part before the `/` starts with the typed part before it
and whose part after starts with the typed part after it (a lookup over the
stored string, not a split).

IDs must be stable: re-parsing, re-importing, a parser improvement, finding
the same session in an export and a live file, or moving a file must never
change an ID, because the operator's notes and tags refer to them. Every ID
is therefore computed from the provider's raw records, never from what the
parser made of them, and from one input revision's bytes plus that
input's own first observation (its first path, its install), never from
other inputs or from what is already stored, so converging inputs in any order, or again after
deleting `index.db`, gives the same IDs. IDs are computed from the bytes as
stored before any excision overwrite (`storage/archive-db.md`), so excising
text never changes an ID.

| Thing | ID | Rule |
|---|---|---|
| Session | `<origin>:<native id>` | The provider's conversation/thread ID. A provider without IDs (Grok web export) gets a hash of its first message's content and time. |
| Claude Code subagent session | `claude-code:<owning session id>/<agentId>`; an older-layout sidechain without `agentId`: `claude-code:<session id>~side-<first record uuid>` | The provider's agent ID is unique only within its owning session, so the owning session's ID is part of it (`sources/claude-code.md`). Polylogue composes this ID; it is still a function of the raw records. |
| Message | `<session id>:<native message id>` | The provider's message ID (Claude Code `uuid`, ChatGPT node ID, or the Codex record ID established by its adapter's grouping and correspondence rules; a generated or turn-scoped UI projection item ID is not automatically a message ID). One model response is one message even when the provider writes it as several records (`decisions/one-message-per-response.md`): its ID is the first record's ID, its blocks are the records' blocks in order, and its usage is counted once. How each provider's records group is in its section (Claude Code by `message.id` and `requestId`; Codex by model request). |
| Message without a native ID | `<session id>:~<hash>-<n>` | A provisional ID, by the fallback rule below. The `~` marks it as Polylogue-made. |
| Block | `<message id>:<index>`; `<message id>:r<index>` for a reasoning item in a format whose rewrites drop reasoning; `<message id>:a:<attachment id>` for an attachment listed outside the content list | The index of the item in the provider's own content list (counting across the records of a split response, in record order), from 0. One counter over every item, thinking included, except in a format whose rewrites drop reasoning items (Codex): there reasoning items have their own counter (`r0`, `r1`, …) and the other items theirs, so a rewrite never shifts another block. The index counts every item the provider wrote of its counter's kind, including ones Polylogue does not show, so a parser that starts or stops keeping a kind of block never shifts the others. Claude Code uses the one counter (the worked example below: thinking `:0`, the tool call `:1`). |
| Event | `<session id>:<native record id>`, or `<session id>:~e<hash>-<n>` | The provider's record ID when the record has one (Claude Code `uuid`); otherwise the fallback rule of messages with prefix `~e`. A hook or process line names its session; one naming a session with no row yet creates a stub session row (no messages; `storage/index-db.md`, `session.stub`), so its live state exists before the transcript does. Its ID: `~h<boot id>-<pid>-<start>-<line>` from the writing process and the line's position in its file, with no session prefix, because such a line can be stored before evidence places it on a session (`storage/index-db.md`, event: `session_pk` NULL until then). |
| Link | none of its own | Named by (parent, child, kind, how), of which a pair has at most one. |
| Run | `<origin>:<native run id>` | The harness's own job, goal or run ID. |
| Delivery | UUIDv7 | Chosen by the sender when it sends, so the same delivery recorded twice stays one. |
| Variant | UUIDv7 | Its alignments are numbered within it. |
| Input | hex BLAKE3 of the bytes, or a UUIDv7 | An imported file that cannot change is named by its content. A location that can grow or change (a session log, a hook spool file, a database) gets a UUIDv7 when first seen and keeps it for life, through appends, rewrites and snapshots; its revisions' bytes are extents over blobs named by their own hashes (`storage/blobs.md`). |

**The fallback rule.** `hash` is the first 12 hex characters of BLAKE3 over
the provider's own record for the message, as canonical JSON (keys sorted,
no whitespace, numbers as the provider wrote them), leaving out only the
wrapper fields its format section declares volatile. `n` counts the earlier
records with the same hash in the same input revision, in record order (0,
1, …). So:

- the same message in two exports gets the same ID, and a parser change
  never moves it;
- identical short turns ("ok") differ, because the record's own time is
  part of the hash, as its raw text: an ID never depends on parsing a time,
  so teaching a parser a new time format never moves an ID;
- a format whose exports relabel one moment in another zone (Sinex found
  Google's do) declares its time field volatile, and the other fields and
  `n` then identify the record;
- inserting a message elsewhere never moves an ID, since `n` counts only
  identical records, never position.

**Provider and provisional IDs.** A message ID is a **provider ID** when it
is built from the provider's own message ID, and **provisional** when
Polylogue computed it (the fallback rule above, or a capture's synthetic turn
ID); `message.id_kind` says which. A provisional ID is stable across
re-parsing, re-import, any arrival order and parser changes, because it is a
function of one revision's bytes. It is not stable across a provider edit:
when a later revision drops an earlier identical record, later duplicates
get a lower `n`, so they can renumber, and the union then holds both
versions under the merge rules rather than guessing which corresponds to
which. Notes on a provisional message carry its text hash (`seen_hash`), so
a renumbered target shows as changed, never silently as another message.
Provisional messages leave a session's union only when matched to a
provider-ID message (`sources/README.md`, rule 1).

**Two known exceptions.** IDs that the provider never assigned and that
cannot be computed from one stable record may move, and their section says
so: a Grok web conversation without `conversation.id` (named by its opening
response, which a later export can change) and Antigravity messages parsed
from a language-server rendering (renamed when the renderer changes). Notes
on such targets attach to the session, and law 10 excludes them.

IDs and content hashes are computed over exact bytes and code points: no
Unicode normalization anywhere, so two strings that look alike stay
different; absent, `null`, empty and a
literal string such as `"null"` are four different values; a path inside an
ID is normalized lexically, never by resolving the filesystem.

A duplicated record with the same native ID and the same content is one
message. Within one input revision, records with the same native ID and
different content are versions of one message in the provider's own order
(Codex rewrites a reasoning item as its summary grows): the last in record
order wins field by field, and the earlier versions stay in the stored
input. A conflict is recorded (a `message_disagreement` row, counted by
`polylogue status` for that input) only when such records disagree on `role` or record type, or an earlier one
carries a field the last lacks, since a provider reusing an ID for
different content is a provider bug worth seeing. Across inputs and
revisions, versions are chosen by the merge rules (`sources/README.md`),
never by which arrived first.
claude.ai exports follow that same rule: none of the operator's four exports repeats a message ID within one export (11,962 to 13,180 messages each, measured 2026-10-12), so no special naming is needed; a repeat would be versions of one message with a conflict recorded, like any format's.

**Lone surrogates in IDs.** A native ID is kept exactly. When the decoded ID contains a lone UTF-16
surrogate, which a Rust string cannot hold, it is stored as `~u:` followed by
the lowercase hex of its WTF-8 bytes (`~` already marks Polylogue-made IDs).
Session, message, call and attachment IDs follow the same rule. Text content
keeps the U+FFFD rule of `sources/README.md`; only identifiers need to stay
distinct (decision: `decisions/README.md`).

**Numbers in canonical JSON**. In the canonical JSON hashed
for a fallback ID, numbers are written as the exact token text the provider
wrote (the lossless number text of `sources/README.md`), never re-formatted, so the
hash depends only on the provider's bytes.

**Attachments listed outside the content list**. ChatGPT `metadata.attachments` and claude.ai `attachments` and
`files` are not in a content list, and their order changes between exports.
Such an attachment gets the block ID `<message id>:a:<attachment id>` when the
provider names one, otherwise `<message id>:a:~<hash>-<n>`, where `hash` is
BLAKE3 over its file name and media type as the provider wrote them (the
bytes never enter the ID: having them is acquisition state), and `n`
follows the rule of fallback message IDs; never its list position. Which
version of a conflicting attachment is kept follows the merge rules, never
arrival.

## Time

Several clocks, kept apart:

- `at`: when the provider says it happened;
- `first_seen`: when Polylogue first had the bytes of the message (`convergence.md`); it depends on when collection ran, so it is never used to order messages;
- for agent sessions, the harness's own timing (durations, `durationMs`).

Every comparison of two times names its clock:

- **provider time** (`at`, `started_at`, `last_activity_at`): ordering
  inside a session uses structure (`convergence.md`); comparisons across
  sessions use provider time and say so.
- **hook time**: a hook or delivery line's `observed_at_ms` is the event's
  and the delivery's time; the line's `timestamp` is kept.
- **Polylogue time**: an observation's `seen_at`, a message's `first_seen`,
  a file's last growth (the `seen_at` of its last appending observation,
  never mtime).
- **writer time**: a note's `created_at`, used for display, and for "added
  since" only on notes written on this machine. An inline-marker note's
  `created_at` is its message's `first_seen`.
- **elapsed time** inside the daemon (stall thresholds, round spacing,
  snapshot caps, sampling): the monotonic clock.

A provider time far beyond the moment Polylogue collected it (a clock set
wrong, a bad field) is kept as given and flagged, but never used as a
session's start, last activity or freshness. Far beyond means more than 7
days after the record's own `first_seen` (the observation that first
covered its bytes; none for up to a minute after arrival, during which
nothing is flagged), so a session resumed weeks later in the same file is
not flagged. A session whose every time is flagged has both times unknown,
with reason `implausible_time`.

A provider timestamp written without a UTC offset is read as its format
declares, by the one wire time type: every format states, as a constant
beside its parser and in its section of `sources/`, whether its offsetless
times are UTC, local, or in a named zone (a server that writes times in its
own zone, whatever machine collected the file). Local times are read with
the zone of the machine that collected the file (the zone recorded with each observation), flagged
`local_naive`; a time in a daylight-saving fold takes the
earlier instant and is flagged ambiguous. A format that declares nothing
gets its offsetless times as unknown, counted as `time_unparsed`, never
guessed as UTC or as the machine's zone: a wrong guess shifts every message
by the offset with nothing to show it (Sinex met this with zoneless export
times three checks could not place). Nothing in parsing, storage or queries
converts through the local time zone (display does).

**Quality and precision.** Every provider time carries two values, in the
vocabulary both projects use (joint glossary; Sinex `model.md` §5), so a
Polylogue row and a Sinex row read the same in a cross-system query:
`time_quality`, from Sinex's closed list (`intrinsic`: the provider's own
time; `file_mtime`: when a file changed; `unknown`), and `precision`
(`instant` | `second` | `minute` | `day` | `month` | `year` | `bounded`,
with `until` for a bounded range). A coarse or bounded time stands for its
whole period, never for its start: a temporal comparison holds only when the
whole period does, and it takes part in a temporal join only when the join's
window is at least as wide as its period (an AI Studio message bounded to
four hours joins `within 4h`, never `within 5m`), then as its period, the
match marked coarse and ranked after precise ones (joint query law "coarse
times"). Polylogue's cases:

| Case | `time_quality` | `precision` |
|---|---|---|
| a provider's own timestamp | `intrinsic` | `instant`, or as written |
| Grok CLI times (whole seconds) | `intrinsic` | `second` |
| AI Studio messages without chunk times, from the Drive file's `createdTime` and `modifiedTime` | `file_mtime` | `bounded` (from created to modified) |
| a Takeout ZIP entry's time | `file_mtime` | `second` (DOS time, zoneless) |
| an offsetless local time (`local_naive`, above) | `intrinsic` | as written; the assumed zone is kept with it |
| an unobserved commit's committer date | `intrinsic` | `second` |

These replace the earlier free-form `at_source` flag; `local_naive`,
`ambiguous` and the implausible-time flag stay, as facts about how a time
was read, not about its precision. Published `polylogue.*` events carry
both values (joint #28).

Display always uses the operator's local time zone.

## Value types

A few values have their own types in `polylogue-model`, so a mistake fails
to compile instead of producing a wrong number:

- **`Usd`** (integer nano-USD) and **`Credits`** (a provider's subscription
  credits) are distinct types with no conversion between them, so credits
  can never be added to dollars (`analysis.md` §1; rule R33).
- **`Ratio`** holds a numerator and a denominator; `Ratio::new` returns
  `None` when the denominator is zero or not exactly known, and a ratio
  without a value is shown as unmeasured, never as 0% or 100%
  (`behavior.md`, Observability; `analysis.md` §11; rule R1).
- Counts, measurements and costs are `Option<_>` from parse to output;
  unknown is never `0` (rule R1).
- The query result's outcome is `polylogue_query::ResultOutcome` (`query.md`,
  Results; rule R2), separate from a tool call's `ToolOutcome` above; a
  parser's answer for one record is a third type, `RecordOutcome`
  (`sources/README.md`). Each is its own name so code never confuses them.

## A worked example

This Claude Code session excerpt (structure as in real files; text is
placeholder):

```jsonl
{"type":"user","uuid":"u1","parentUuid":null,"sessionId":"s1","promptSource":"typed","timestamp":"2026-10-05T09:00:00.000Z","cwd":"/home/op/morrow","gitBranch":"main","version":"2.1.220","message":{"role":"user","content":"Run the tests and fix the failing one."}}
{"type":"assistant","uuid":"a1","parentUuid":"u1","sessionId":"s1","timestamp":"2026-10-05T09:00:04.000Z","message":{"model":"claude-fable-5-1","role":"assistant","content":[{"type":"thinking","thinking":"I'll run the suite first."},{"type":"tool_use","id":"toolu_1","name":"Bash","input":{"command":"cargo test"}}],"usage":{"input_tokens":12,"cache_read_input_tokens":20000,"cache_creation_input_tokens":1500,"output_tokens":80}}}
{"type":"user","uuid":"u2","parentUuid":"a1","sessionId":"s1","timestamp":"2026-10-05T09:00:31.000Z","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":"test parse_dates ... FAILED\nerror: test failed, exit code: 101","is_error":true}]}}
{"type":"assistant","uuid":"a2","parentUuid":"u2","sessionId":"s1","timestamp":"2026-10-05T09:00:40.000Z","message":{"model":"claude-fable-5-1","role":"assistant","content":[{"type":"text","text":"One test fails: parse_dates. Fixing it now."}],"usage":{"input_tokens":15,"cache_read_input_tokens":21500,"cache_creation_input_tokens":300,"output_tokens":20}}}
```

normalizes to (export format, abbreviated):

```json
{
  "session": {
    "id": "claude-code:s1", "origin": "claude-code", "native_id": "s1",
    "kind": "main",
    "display_title": "Run the tests and fix the failing one.", "title_source": "first_prompt",
    "started_at": "2026-10-05T09:00:00.000Z", "last_activity_at": "2026-10-05T09:00:40.000Z",
    "cwd": "/home/op/morrow", "branch": "main",
    "harness": "claude-code", "harness_version": "2.1.220",
    "models": {"claude-fable-5-1": 2}
  },
  "messages": [
    {"id": "claude-code:s1:u1", "role": "user", "written_by": "human", "prompt_kind": "prose",
     "at": "2026-10-05T09:00:00.000Z", "on_active_path": true,
     "blocks": [{"id": "claude-code:s1:u1:0", "kind": "text",
                 "text": "Run the tests and fix the failing one."}]},
    {"id": "claude-code:s1:a1", "role": "assistant", "written_by": "model",
     "model": "claude-fable-5-1", "at": "2026-10-05T09:00:04.000Z", "parent": "claude-code:s1:u1", "on_active_path": true,
     "usage": {"input": 12, "output": 80, "cache_read": 20000, "cache_write": 1500},
     "blocks": [
       {"id": "claude-code:s1:a1:0", "kind": "thinking", "text": "I'll run the suite first."},
       {"id": "claude-code:s1:a1:1", "kind": "tool_call", "tool": "Bash", "call_id": "toolu_1",
        "input": {"command": "cargo test"}, "command": "cargo test"}]},
    {"id": "claude-code:s1:u2", "role": "user", "written_by": "harness",
     "at": "2026-10-05T09:00:31.000Z", "parent": "claude-code:s1:a1", "on_active_path": true,
     "blocks": [{"id": "claude-code:s1:u2:0", "kind": "tool_result", "call_id": "toolu_1",
                 "outcome": "error", "exit_code": 101,
                 "text": "test parse_dates ... FAILED\nerror: test failed, exit code: 101"}]},
    {"id": "claude-code:s1:a2", "role": "assistant", "written_by": "model",
     "model": "claude-fable-5-1", "at": "2026-10-05T09:00:40.000Z", "parent": "claude-code:s1:u2", "on_active_path": true,
     "usage": {"input": 15, "output": 20, "cache_read": 21500, "cache_write": 300},
     "blocks": [{"id": "claude-code:s1:a2:0", "kind": "text",
                 "text": "One test fails: parse_dates. Fixing it now."}]}
  ]
}
```

Things to notice:

- The tool result message has role `user` but `written_by: harness`, because
  Claude Code wraps tool output as a user turn. Human prompt counts use
  `written_by: human` with `prompt_kind: prose` (u1 has both).
- The outcome is `error` from `is_error: true`; the exit code 101 is parsed
  from the output text. Neither is guessed.
- No `repo` appears: the excerpt records a branch and directory but no
  repository URL, and `/home/op/morrow` does not exist on the machine that
  converges the example, so no repo-lookup input exists for it and the
  repository is unknown (`sources/README.md`, the repo-lookup record). `branch` is kept because the provider stated it.
- Token counts are copied as reported. Cost is computed from them with the
  price table and stored as a session fact (`analysis.md`); a corrected price
  table recomputes it without re-parsing.

Provenance (Bead and audit IDs by section) and the Python-version lessons behind some rules: `evidence/spec-provenance-model-sources.md`.

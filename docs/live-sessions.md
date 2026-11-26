# Live sessions

What Polylogue knows about sessions that are running now (live events, live state, processes, peers, edited files, tasks) and which channel each act on a session takes. The acts themselves (`send`, `answer`, `interrupt`, `resume`, `start`) and hosted sessions are [session-control.md](session-control.md); what hooks record is [capture.md#hooks](capture.md#hooks).

## Live state

Live state is computed at read time from stored facts and `now`. Nothing live is kept only in memory, so a rebuild reproduces every past live state. It rests on live events only (a process alive, a harness event, the harness's own registry), never on text.

**Live events** arrive by four routes, each a captured input in the hook spool: `hook` (the hook command's lines), `process` (the daemon's process watcher, below), `registry` (harness registries: Claude Code's `sessions/<pid>.json`, one line per change, the file kept whole; Codex's thread statuses on its shared app-server, read every second while its control socket exists and from `thread/status/changed` notifications between readings, one line per change, the status object kept whole) and `git` (commits seen in a running session's worktree, below). Each line becomes an event on its session (`hook_*`, `process_*`, `registry_*`, `git_*`).

The live-event writers retain failed appends with their original observation times and bytes. A partial line resumes at its unwritten suffix. Process, registry, Codex, Git and hosted-session producers each own a spool file named for the boot, process, producer and a unique writer instance, so one writer cannot interleave bytes with another writer's partial line. Existing spool files remain inputs. Writers retry on their next observation turn and once after their producers stop; an unsuccessful final drain is an explicit shutdown error. Appending does not sync the file. Daemon status counts its own retained lines and reports their write errors; separate hosted-session processes report failures and recovery in their inherited service journal.

**Hook events.** Every event each harness emits is installed and spooled, as [capture.md](capture.md#hooks) lists: Claude Code's full set (including `SessionStart`, `UserPromptSubmit`, `PreToolUse`, `PostToolUse`, `PostToolUseFailure`, `Notification`, `PermissionRequest`, `PermissionDenied`, `Stop`, `StopFailure`, `SubagentStart`, `SubagentStop`, `PreCompact`, `PostCompact`, `Elicitation`, `ElicitationResult`, `CwdChanged`, `FileChanged`, `WorktreeCreate`, `Setup` and `SessionEnd`), Codex's full set including `PermissionRequest`, and Gemini CLI's (`SessionStart`, `BeforeAgent`, `BeforeTool`, `AfterTool`, `AfterAgent`, `Notification`, `PreCompress` and `SessionEnd`). `polylogue install` takes each harness's set from the installed version by probe. These events carry the waits and stalls the live state is computed from.

**Live state.** Each session's live facts are computed from its live events in (time, line) order, written in the transaction that stores each line: the root process, its exit, the hook-derived activity (`working`, `idle`, `waiting`) and wait (`permission`, `question`, `plan`, `input`), the harness registry's own status word, the open tool call, a stall the events state, open subagents, the edited files, the task and the in-progress todo text. A line older than what the session has already applied makes the session recompute from all its stored lines, so arrival order never changes the result.

**Live state**, the first rule that applies:

| State | Rule | Shown with it |
|---|---|---|
| `ended` | a `SessionEnd`, or the root process gone after a `Stop` | the event or process line |
| `stale` | root process gone with no `SessionEnd` (killed, crashed, rebooted); or, with no process events, no hook events for 15 minutes. A day after its latest event a stale session is `ended`: the live table holds sessions with a live root or an event in the last day | "process gone at T, no end event" |
| `waiting` | waiting on the operator: a `PermissionRequest` or a permission or elicitation notification with nothing after it; an `AskUserQuestion` or `ExitPlanMode` call open; a `Stop` whose turn ended with a question (`ending: question`); the registry says `blocked` | what it waits for, since when |
| `working` | a turn in progress: a prompt, a tool call or a subagent after the last stop; or the registry says `busy` | the open tool or last event |
| `idle` | the last turn stopped, nothing waiting; or the registry says `idle` | since when |
| `unknown` | no live events, and the transcript grew in the last 15 minutes or no time is recorded | "inferred from file growth" |

- **The registry outranks hook inference**: when the harness's own word (`busy`, `idle`, `blocked`; Codex `active`, `idle`, `waitingOnApproval`, `waitingOnUserInput`) states an activity, that is the state shown, with the hook events kept beside it.
- **Without process events** (lines from before hook lines carried the process identity), a hook event holds for 15 minutes after the last line; afterwards the session is `stale`, and a day after the last line `ended`. With a known root process that is alive, no passage of time changes the state; only events do.
- **Stalls** are waits on something other than the operator: `rate_limit` and `api_error` (a `StopFailure`, by its error type), `long_call` (a tool call open for 30 minutes), `queue` (a message sent to a session Polylogue hosts waits behind the turn it is running; the daemon writes the number waiting to the registry route as `HostedQueued` lines, and the stall ends when the number returns to zero, the session goes idle or it ends). A stall is shown with the state and never counts as waiting.
- `live_until` is the next instant at which the state or stall changes with no new event; empty when only events can change them.
- **The live clock.** The daemon wakes at the earliest `live_until` of the sessions in the live table and gives each session whose instant passed a new change counter, so a transition that comes from time alone reaches the change feed, `read --follow` and open web views like any other session change. Nothing polls.

Every live read includes a live root regardless of the age of its latest event, including edited files and the next live-state deadline.

A subagent's live state is its parent's process plus its own `SubagentStart`/`SubagentStop`. Query fields on sessions: `live`, `running` (`live` in `working`, `waiting`, `idle`, `unknown`), `waiting`, `waiting_on`, `stall`, `process_alive`, `harness_status`, `open_tool`, `edited_files`, `task`, `todo`, `live_since`, `live_until`, `last_event_at`, `live_rank`, `peers`.

**Live intervals.** Applying events also closes intervals of each session's activity, wait and stall: `(session, aspect, value, start, end, closing event)`, kept in `live_interval`. They depend only on events: an interval stays open until an event closes it, and a rebuild gives the same intervals.

**Process identity.** Each hook line carries the harness's process identity: its pid, its start time (field 22 of `/proc/<pid>/stat`, `harness_start`) and the boot ID. A process or registry line whose pid matches but whose start time or boot differs never attaches to the session, and a control never acts on a pid whose start time or boot differs.

**The process watcher.**
- Its **roots** are the root processes that sessions' events say are alive. Each root gets a pidfd; its exit writes `RootGone` at once. A root already gone when first seen is written gone then; a root of an earlier boot is written gone at that boot's end.
- **Descendants** are read from `/proc/<pid>/task/*/children`, every second while the session has an open tool call and every 15 seconds otherwise; only differences are written to the process spool (`process/*.ndjson`): `Started` (pid, parent pid, start, `comm`, `argv`, `cwd`, role), `Ended`, `RootGone`.
- With no roots nothing under `/proc` is read; a sample with no change writes nothing; `environ` is never read.
- **`processes` rows:** session, identity, parent pid, `comm`, `argv`, `cwd`, `started_at`, `ended_at`, `role` (`harness` or `child`), and `tool_call`, the call that was open when the child was first seen, by time only. A session's end closes the rows of its processes.

**Peers** are other sessions whose live state is `working`, `waiting` or `idle` in the same repository (by its identity) or, when either has no repository identity, the same working directory. Peers are a query, symmetric at any `now`.

**Edited files.** Per live session, the files it changed through structured edit tools (Claude Code `Edit`, `Write`, `MultiEdit`, `NotebookEdit`; Gemini CLI `write_file`, `replace`; the file headers of Codex's `apply_patch`), from the finished call's structured input. Shell commands contribute nothing. A file leaves the edited files when a commit in the session's worktree contains it, and the whole set is cleared at the session's end. A root process gone mid-turn leaves the session stale with its edited files.

**The commit watcher.** For the worktrees of running sessions with edited files, the daemon checks each worktree's HEAD reflog once per step (one `stat`). When HEAD moved, Git lists the new commits and their files, and each session whose edited files a commit contains gets one `git` line `Commit`, dated at the commit, naming the commit, the worktree and exactly those files; an edit after the commit puts its file back. No command text is read. These lines carry the files a commit contains, which `commit_link` rows do not hold. Commits made while the daemon was not watching the worktree (before its first step there) are not seen.

**Task.** A session's live task is set only by an explicit structured task event: a hook line `TaskClaimed` with `{"task": ID}` sent for the session (`polylogue internal hook TaskClaimed --harness HARNESS`), cleared by `TaskReleased` naming the same task. A command's name or arguments, including a tracker's claim command, establish no task state. A tracker ID mentioned in prose stays a reference. The harness's in-progress todo item is shown as text (`todo`), never as a task.

**Surfaces.**
- `polylogue running` lists running sessions, waiting first; `polylogue waiting` lists what waits on the operator, the longest wait first, held web-chat turns included. Both are shipped saved queries ([saved-queries.md](saved-queries.md#shipped-saved-queries)).
- `polylogue running read --follow` (any selection of sessions before `read --follow`) prints one line per live-state change (`14:02:11 claude-code:5f2e… → waiting: permission`) and per new message or tool call of the selected sessions, until interrupted or, with `--until-idle`, until they settle. It wakes on each committed change, reads the selection again, so sessions that start matching are followed too, and starts with each session's current state rather than its history. Each daemon request runs until its deadline and ends with the marks it reached, which the next request carries, so lines are neither repeated nor skipped. Over MCP there is no following (a model cannot hold a stream); the refusal gives the `read` call with `from` that continues from where the caller is.
- `read --view live` shows the state and how long, what it waits on or its stall, the open tool call, the task, the peers, and the process tree: the harness, then each child with its command, age, working directory and tool call.
- `status` lists the running sessions, waiting first, each with its state, since, wait, stall, open tool, task, edited file count and directory; MCP `status` returns the same list. `status --line` prints one line for status bars ([operations.md](operations.md#status)).
- The brief's other-agents section lists each live session here ([agent-context.md](agent-context.md#the-brief-for-a-directory)).
- Live-state changes are session changes on the change feed, so followers see `waiting` without polling.

**Latency:** from a hook line or transcript append to the change being visible in `read --follow`, the running list and an open web view: 1 s p95, warm.

## Control channels

**Channels.** Which channel an act takes is one pure function of the act, the session's live state and what the installed harness is known to support (`polylogue_model::live::channel`), in this order of preference:

1. **protocol**: a session Polylogue hosts ([Hosted sessions](session-control.md#hosted-sessions)): `send` steers or prompts it, `answer` replies to its oldest pending request, `interrupt` stops its turn.
2. **hook**: the next context-accepting hook event of a running session whose events show the hook installed, through the mailbox; for `answer`, Claude Code's synchronous `PermissionRequest` hook, where the probe confirmed it.
3. **resume**: the harness's headless resume (`claude --resume ID -p TEXT`, `codex exec resume ID TEXT`) for a session that ended or is stale.

A mechanism no probe confirmed is refused (`refused`, `unsupported`, naming the harness). A channel is never chosen by trying one and falling back: a failed channel records `not_sent:failed(<reason>)` and nothing else is tried.

What each harness offers (its context-accepting hook events, its permission hook, its headless resume, its own fork for `resume --fork`, whether it lets a running session be reopened in a second client for `resume --anyway`, and the protocol its hosted sessions run on) is declared once, with the channel function; the hook command reads its context events from the same declarations. The table below is generated from them, and a test keeps it equal to what the function returns: per harness, a session Polylogue observes (its hook installed) and one it hosts, the channel each act takes, or why it is refused.

<!-- channel table: generated by polylogue_model::live::channel::channel_table -->
| Harness | Session | send, running | send, ended | notice | answer | interrupt | resume, ended | context events |
|---|---|---|---|---|---|---|---|---|
| claude-code | observed | hook | resume | hook | hook | refused: unsupported | resume | `PostToolUse`, `UserPromptSubmit`, `PermissionRequest` (answers) |
| claude-code | hosted (claude_stream_json) | protocol | resume | protocol | protocol | protocol | resume | `PostToolUse`, `UserPromptSubmit`, `PermissionRequest` (answers) |
| codex | observed | hook | resume | hook | refused: unsupported | refused: unsupported | resume | `PostToolUse`, `UserPromptSubmit` |
| codex | hosted (codex_app_server) | protocol | resume | protocol | protocol | protocol | resume | `PostToolUse`, `UserPromptSubmit` |
| gemini-cli | observed | hook | refused: unsupported | hook | refused: unsupported | refused: unsupported | refused: unsupported | `AfterTool`, `BeforeAgent` |
| gemini-cli | hosted (acp) | protocol | refused: unsupported | protocol | protocol | protocol | refused: unsupported | `AfterTool`, `BeforeAgent` |
| hermes | observed | hook | refused: unsupported | hook | refused: unsupported | refused: unsupported | refused: unsupported | `pre_llm_call` |
| hermes | hosted (acp) | protocol | refused: unsupported | protocol | protocol | protocol | refused: unsupported | `pre_llm_call` |
<!-- end of channel table -->

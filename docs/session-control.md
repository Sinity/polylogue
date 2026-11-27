# Session control

How the operator and agents reach live sessions: `send` (to sessions, roles, member aliases and web chats, and through bracketed sends in messages), `answer`, `interrupt`, `resume` (with `--fork`, `--anyway`, and `--in` for another harness), `read --follow`, `start`, roles, hosted sessions, the mailbox, notices to sessions, and delegation records. Live states, peers and the control channel of each harness are [live-sessions.md](live-sessions.md#live-state); the line grammar is [interfaces.md](interfaces.md); the sessions waiting on you and the running ones are the shipped saved queries `waiting` and `running` ([saved-queries.md](saved-queries.md#shipped-saved-queries)).

Every act here is a step of a line or an operation of the `polylogue_act` tool ([agents.md](agents.md#mcp-server)), the same from the CLI, MCP and the web. Each act is recorded as a [delivery](agent-context.md#deliveries) with its computed author ([authorship.md](authorship.md)); an act on several rows of a query needs `--all` ([interfaces.md](interfaces.md#steps)); `--dry-run` shows each recipient and channel and reaches nothing. Polylogue records what happened and does not coordinate: it never locks, assigns or schedules work.

An act's reply reports what its recipient or channel acknowledged. Its outcome record is queued for the writer; status reports any pending outcome writes and their last failure ([delivery recording](agent-context.md#deliveries)). Retrying that record never repeats the act.

## Send

`send TEXT` sends text to each selected session; the rows are the recipients. It covers agent sessions, roles and web chats.

```console
$ polylogue @2 send 'rebase on master first'
$ polylogue role:reviewer send 'review the diff on branch docs/command-surface'
$ polylogue @reviewer send 'look at the findings' --page flaky-ci --section Findings
$ polylogue 'sessions where running:true repo:sinex' send 'stop before touching the writer' --all
```

| Recipient | Delivery |
|---|---|
| a running session Polylogue hosts | over its control connection, at once ([Hosted sessions](#hosted-sessions)) |
| a running session with hooks | at its next tool result or prompt, through its [mailbox](#the-mailbox) |
| one ended session, named alone | resumed once, headlessly, with the text (this starts a harness process) |
| ended sessions among several | skipped, each listed with its live state and the line that sends to it alone |
| `@alias` | the session the [member](pages.md#members) names, resolved when the message is sent; an alias follows its session's successor, so a resumed session gets it |
| `role:NAME` | the role's running instance, or a new one started with the role's harness and instructions ([Roles](#roles)) |
| a web-chat session, or a pasted conversation URL not yet archived | the browser extension writes the turn and holds it for the operator's approval ([Answer](#answer)); with the setting `browser.actions` off it is refused with the `set` line |

- **The record.** A sent message is one delivery per recipient: sender `send`, event `message`, with the text, the recipient, the `--about` reference and the attachments. It writes no annotation. The text the recipient receives carries the delivery reference line and says how to reply: a note with `--replies-to delivery:<id>` (over MCP, `replies_to`), or `send` back to the sender.
- **Replies** are the recipient's regular output, captured as usual and linked to the message by a `--replies-to delivery:<id>` relation or by the delivery reference found in the recipient's transcript. A session resumed by a sent message records a `reference` link to the sending session with `how: send` ([sessions.md](sessions.md#session-links)).
- **What happens depends on the recipient's live state:**
  - running (`working`, `waiting`, `idle`, `unknown`): over the hosted connection, or at its next context-accepting hook event through the mailbox, never interrupting a running tool call; if the session stops running first, `not_sent:session_ended`;
  - `ended` or `stale`, named alone: resumed once, headlessly, through the harness's resume command in the session's [resume directory](#resume), with the text as its prompt; without one the delivery is `not_sent:directory_unknown` or `not_sent:directory_missing(DIR)`. No confirmation is asked, whoever sent it and through whichever channel. The reply says `started` once the harness process starts, or `queued` once the harness and directory have been checked and a running resume delays it. A delivery is `delivered` once the process starts. If it exits unsuccessfully, the delivery also records `failed` with its exit status and the end of its standard error;
  - a session not in the archive gives `not_sent:session_missing`; a harness with no headless resume gives `not_sent:unsupported`.
- **Where the reply goes.** `--page P --section S` records on the delivery the page and heading where a reply belongs; an annotation replying to the delivery that names no page of its own is placed there ([pages.md](pages.md#reply-placement)).
- `--about REF` says what the message is about; `--attach FILE` (repeatable) attaches files, over MCP as paths on the daemon's host under `import`'s rule.
- The sender is never its own recipient. Each recipient's channel and outcome is in the reply. Unsent and skipped recipients carry a reason with a `kind` and any details that belong to it, such as the missing directory's path or the session's live state. A send with a skipped or unsent recipient returns a degraded reply (exit 5). A sent message cannot be taken back, and the reply says so.
- **Retries never send twice.** Two identical sends to one session without a request ID are one send ([agents.md](agents.md#retries-never-write-twice)); sending the same text again on purpose is a new line, which the CLI gives a fresh request ID.
- **Concurrency:** one resumed run per session at a time; further messages to it queue as `prepared` deliveries and run in order, unless the resumed session's own hook claims them first. There is no rate limit.
- **Bracketed sends.** A bracketed address in a message the operator or an agent writes sends the text after it, as `send` does ([markup.md](markup.md#sends) has the syntax):
  - `@[codex:019d0f2a] rebase first` sends to that session, with the delivery rules above (one ended session named alone is resumed);
  - `@[@reviewer] look at the findings` sends to the member alias, resolved as [pages.md](pages.md#members) says;
  - `@[sessions where running:true repo:sinex] stop before touching the writer` reaches the running sessions the query selects when the message is sent, as `send --all` does; ended ones are skipped, and sessions that start matching later receive nothing.

  The send's ID derives from the marker as an annotation's does ([markup.md](markup.md#sends)), so re-parsing, quoting, translating or importing the message again never sends twice. Each recipient gets one delivery like any send, from the message's author. A bare `@reviewer` without brackets is plain text.
- **Background triggers never start or resume agents:** a signal or watch firing has no recipient named by an author, so it only notifies ([Notices to sessions](#notices-to-sessions)).

## Answer

`answer VALUE` responds to what a session is waiting for, and only that. `VALUE` is `allow` or `deny` for a permission request (the harnesses' own words), or quoted text for a question the session asked. `--standing` answers the same tool and input whenever the session asks next; `--reason TEXT` gives a reason with a denial.

```console
$ polylogue waiting
$ polylogue @1 answer allow
$ polylogue @3 answer 'use the staging bucket'
```

- **The first answer wins**, from the terminal, the web, the harness itself or the extension's popup; a later one is refused `conflict` naming what won, and its delivery records `not_sent:already_answered`.
- **Hosted sessions.** A hosted session's requests come to Polylogue's connection, and `answer` replies to the oldest one.
- **Observed sessions.** Polylogue does not hold a harness's own permission prompt, so an answer reaches an observed Claude Code session only ahead of time: `answer allow --standing` records the decision for the pending request's tool and input, and the next `PermissionRequest` hook for the same tool and input in that session returns it once and records `delivered` after writing it to the harness; with none, the hook returns nothing and the harness's own prompt stands. A plain `answer` to an observed session is refused with "answer it in the terminal".
- **Held web-chat turns.** A turn Polylogue writes into an existing web conversation is always held, whoever sent it, and so is a new web conversation submitted by anyone but the operator ([Start](#start)); the held turn is what that web-chat session waits for. `polylogue @1 answer allow` releases it and `deny` cancels it; the extension's popup does the same ([extension.md](extension.md#chat-actions)). Over MCP, `answer` on a held web-chat turn is refused with "approve it in the browser or terminal": the hold exists so that the operator sees what is posted under the operator's account (default; open operator question).
- Every answer is a delivery with sender `act` and event `control`.

## Interrupt

`interrupt` stops a session's current turn, not the session, through the control connection of a session Polylogue hosts (`polylogue @2 interrupt`). For any other session it is refused (exit 3) with the reason. Signals to processes are never used. Interrupting a hosted Gemini CLI or Hermes session also answers every pending permission request of it `cancelled`.

## Resume

`resume` reopens a session in its own harness, in the directory where it ran, using the harness's resume command from its origin declaration (`claude --resume ID`, `codex resume ID`, `gemini --resume ID`).

```console
$ polylogue resume
$ polylogue codex:019d0f2a resume
$ polylogue codex:019d0f2a resume --dry-run
$ polylogue claude-code:7f3e2a9b resume --fork
```

- **On a terminal** it runs that command; **off a terminal** it prints it, since an agent's tool shell has no terminal to run an interactive harness in. `--dry-run` prints it on a terminal too. Run on a terminal, it exits with the harness's own status; a session with no harness resume or no resume directory here is refused (exit 3) with the context-assisted line, and a harness that cannot be started is `io` (exit 4).
- **Alone**, `polylogue resume` opens the picker over the shipped saved query `unfinished`: sessions in this repository that ended open, each row with why it is open (an unanswered prompt, an unfinished tool call, an open todo or handoff, a partial outcome; [saved-queries.md](saved-queries.md#shipped-saved-queries)). Successors replace predecessors; forks stay separate; sessions judged `solved` or `abandoned` come last. Off a terminal it is refused with the candidates as ready lines ([interfaces.md](interfaces.md#a-verb-with-no-selection)).
- **The resume directory** is the directory the session was launched in, or the one its harness relocated it to (a Claude Code `relocated` record), never a directory it moved to while working. A session with no recorded directory, or one that does not exist here, is refused (`directory_unknown`, `directory_missing`); the command never runs in the caller's or the daemon's directory.
- **A running session** is refused `session_running`, with `polylogue SESSION read --follow`, `polylogue SESSION send 'TEXT'` and, where the harness allows it, the `--anyway` line offered. **`--anyway`** reopens it in a second client where the harness lets a running session be resumed again, as its origin declaration says; where it does not, the refusal stands.
- **`--fork`** opens the harness's own fork of the session where the harness has one (Claude Code: `claude --resume ID --fork-session`): a new session sharing the source's history up to now, which the archive links to the source as a `fork` ([sessions.md](sessions.md#session-links)). A harness with no fork of its own is refused `unsupported` with the `resume --in` line, which continues the work in a new session instead. `--fork` works on running and ended sessions, and runs or prints like any resume.
- `resume` takes one row: several rows open the picker on a terminal and are refused `ambiguous` off it.

### Resume in another harness

`resume --in HARNESS` continues the session's work in a new hosted session of that harness (`claude-code`, `codex`, `gemini-cli`, `hermes`):

```console
$ polylogue self resume --in codex --stop-source
$ polylogue codex:019d0f2a resume --in claude-code --at message:codex:019d0f2a:m41 --cwd ~/morrow
```

- The new session's first turn is the source's brief (its takeover, [agent-context.md](agent-context.md#the-brief-of-a-session)) with a line naming the source and the message to continue from (`--at MESSAGE`, default the latest); the turn opens with its delivery reference. It starts in the source's resume directory, or `--cwd DIR`.
- The first turn is a delivery (sender `act`, event `first_turn`) that refers to the source session and the message; that is how the new session is recorded as briefed from the source. It is a new session briefed from the source, unlike `--fork`, which uses the harness's own fork.
- `--stop-source` interrupts the source through its channel once the new session runs, recorded like any interrupt. If the interrupt fails, the reply still names the new session and reports the interrupt failure inside it. A retry with the same request ID reuses the recorded first turn.
- It works on running and ended sessions, on and off a terminal, since it starts a hosted session and runs nothing interactive in the caller's shell. `--dry-run` shows the brief and the channel and starts nothing.
- Over MCP: `{"arguments": {"selection": "self", "steps": [{"verb": "resume", "in": "codex", "stop_source": true}]}}`.

## Following

`read --follow` prints one line per new message, tool call or live-state change of the selected sessions until stopped, like `tail -f`:

```console
$ polylogue running read --follow
$ polylogue @2 read --follow --until-idle --hide thinking
```

- Each session's items print in the collapsed reading form, then what is appended. `--show` and `--hide` choose the item kinds as in any reading ([reading.md](reading.md#show-and-hide)).
- The stream ends each observed batch with its current marks, after all selected sessions have been read. Clients decide whether every session has settled at that boundary.
- **`--until-idle`** ends it when every selected session settles: nothing new arrives and the session's live state (the query field `live`; for `root:REF`, of its newest physical session) is neither `working` nor `waiting`.
- **On a selection that is not sessions** it prints the query's rows, then each row (by its reference) as it starts matching, reading the query again whenever the archive's read position moves.
- **Deadline.** A caller's deadline ends the follow with the read timeout described in [interfaces.md](interfaces.md#exit-codes).
- **Cost.** A session reading wakes on each commit and looks again only at the selected sessions whose own rows or live state changed since its previous look; between commits it waits without holding a reader, and a client that goes away ends it at once.
- **Alone**, `polylogue read --follow` opens the picker over `running`; Tab marks several.
- `read --follow` must be the last step of its line. HTTP streams it with the shared server-sent event framing.
- **Not over MCP:** a model cannot hold a stream. The refusal gives the `read` call with `from` set after what the caller has, which is also the call the last page of every session reading carries ([transcripts.md](transcripts.md#rendering-for-models)).

## Start

`start TEXT` starts a session Polylogue hosts on the harness's control protocol, so its approvals, questions and turns can be answered, steered and interrupted from anywhere, and it survives a daemon restart.

```console
$ polylogue start 'fix the flaky attachment test' --in codex --cwd ~/morrow
$ polylogue start 'review the open pull request' --role reviewer
$ polylogue start 'compare these two retry strategies' --in chatgpt --model gpt-5 --project morrow
$ polylogue start 'summarise this thread' --in claude-ai --submit
```

- `--in HARNESS` (`claude-code`, `codex`, `gemini-cli`, `hermes`; default the setting `start.harness`) or `--role NAME` (an instance of a [role](#roles): its harness with its instructions); `--cwd DIR`; `--attach FILE` (repeatable).
- The first turn is recorded as a delivery (sender `act`, event `first_turn`) to the new session, whose text opens with the delivery reference. For Claude Code, startup waits for the first harness protocol line after the prompt is submitted. For ACP, it waits for the first update, permission request or successful prompt response. Codex startup waits for the app-server to accept the turn. A stdio harness that exits before that output fails startup with its exit status and the end of its standard error.
- **Web chats.** `--in chatgpt|claude-ai|grok` drafts a new web conversation through the browser extension; `--model`, `--effort` and `--project` choose what the provider's menus offer. `--submit` sends it when the caller is the operator; from any other author (an agent over MCP or a shell) the draft is held like a turn into an existing conversation until the operator's `answer allow` ([Answer](#answer)). The extension's side is [extension.md](extension.md#chat-actions). A turn into an existing conversation is `send`, always held for approval. Drafts and held turns are rows of the unit `chat_actions`.
- Older web conversations come in through `import PROVIDER` ([extension.md](extension.md)).

## Roles

A role is a named kind of agent: instructions (a SOUL.md-style text, or a `file:` reference) and the harness that runs it. It is a `role` annotation keyed by its name, with value `{instructions, harness}`, brought in by `import` of annotation lines ([annotations.md](annotations.md#annotation-lines)); `role:NAME` names it, and `edit` on the role's annotation writes a new revision. Roles are optional; a session without one runs the harness's default.

- **Instances.** A session is an instance of a role, or of no role. Starting an instance (`polylogue start 'TEXT' --role reviewer --cwd ~/morrow`, or `start` with `role` over MCP) starts a hosted session of the role's harness with the role's instructions, then the text, as its first turn. That delivery refers to `role:<name>` and to the role's exact revision, so editing the role later does not change what earlier sessions ran under. A role may have several instances running at once.
- **Members.** A member names one session under an alias, optionally as an instance of a role and on a page; `@alias` is then a selection and a recipient ([pages.md](pages.md#members)).
- **`role:NAME` as a recipient.** `send` to `role:reviewer` reaches the role's newest instance that Polylogue hosts or that runs. With none, it starts one in the directory the role's latest instance ran in, with the sent text as the first turn; a role that never ran has no directory, and the send is refused (`no_live_instance`) with the line `polylogue start 'TEXT' --role reviewer --cwd DIR`.

## Hosted sessions

`start`, `send` to a role with no running instance, and `resume --in` start a session Polylogue hosts: it runs on the harness's own control protocol, and the [control channels](live-sessions.md#control-channels) decide how each act reaches it.

- **Codex** runs on its shared app-server: Polylogue starts a thread (`thread/start`) and holds the connection; the app-server's approval and question requests for the thread come to it, a sent message steers the running turn (`turn/steer`) or starts one (`turn/start`), `answer` replies to the oldest request (which stays pending until the app-server has been sent the answer, so a failed answer can be given again), and `interrupt` sends `turn/interrupt`. The app-server stores the thread, so after a daemon restart Polylogue connects again and subscribes with `thread/resume`.
- **Claude Code** runs as `claude -p --input-format stream-json --output-format stream-json --include-hook-events --permission-prompts host --session-id ID`: a sent message is a user message (run after the current turn), `answer` is the control response to its oldest `can_use_tool` request (allow with the tool's input, or deny with the reason), `interrupt` is the `interrupt` control request.
- **Gemini CLI and Hermes** run in ACP mode (`gemini --acp`, `hermes acp`): Polylogue initializes, creates the session (`session/new`) and sends each message as `session/prompt`; a message while a prompt runs waits for it, since ACP has no steering, and its delivery is recorded `delivered` when it is sent (`not_sent` when an interrupt or a closed host connection drops it before submission; a daemon restart in between leaves it `prepared`). A connection closing after submission does not turn that sent delivery into `not_sent` or cause a resend. `answer` selects the request's `allow_once` or `allow_always` option (deny: `reject_once` or `reject_always`; with no such option the reply reports `cancelled` and lists the option kinds the harness offered), and `interrupt` sends `session/cancel`. Polylogue offers the agent no file system or terminal methods.
- **The per-session host.** Claude Code's stream-json and ACP run over the agent's standard input and output, so a small per-session `polylogue internal host` process holds them, not the daemon: a daemon restart never ends the agent. The host relays lines between the agent and the daemon's connection; what the agent writes while no daemon is connected waits in the host, and a new connection first gets a line with the session's status, then every request of the agent nobody answered, then what waited. The host writes the session's status (`busy`, `blocked`, `idle`) to the registry route as `HostedStatus` lines naming the agent process it started, which the live state takes as the session's root, and `HostedEnded` when the agent exits, which ends the session. It reads no archive, and exits with the agent. It runs in its own process group, and the service stops the daemon alone (`KillMode=process`), so hosts outlive a restart; they stay in the service's control group with its CPU and I/O weights. A host whose session never started (the protocol refused or timed out) is stopped with its agent.
- **One act at a time per connection.** Each hosted session's connection is used by one act at a time, and only that session's acts wait for it: an adapter slow to reply delays no other session. What a connection reports later (a message sent after the running prompt, a resumed run's outcome) is recorded by the daemon's acts worker.
- **Reattachment.** Each hosted session has a record in the archive's runtime directory (`hosts/<key>.json`, with the host's socket or the Codex thread). At start the daemon connects to every recorded session again; a record whose host or thread is gone is removed, and one whose service is briefly unavailable (Codex's app-server not up yet) stays. Failed reattachments, mailbox errors and rejected delivery records are logged and counted in daemon status. A hosted session whose live state is `ended` is let go, its record with it.
- Worktree and permissions are explicit request context, never an ambient directory. An agent saying "I started the reviewer" does not show a delivery; the delivery and the recipient's transcript do.

## The mailbox

A message or notice to a running session that Polylogue does not host waits as a `prepared` delivery and raises the empty flag file `mailbox/<origin>/<session>` in the archive's runtime directory. At a context-accepting event (Claude Code and Codex `PostToolUse` and `UserPromptSubmit`, Gemini CLI `AfterTool` and `BeforeAgent`, Hermes `pre_llm_call`), the hook checks that one path; only when it exists does it ask the daemon (`internal mail-claim`), which hands over every pending item, oldest first, and lowers the flag. The hook records each `delivered` once it has written it to the harness.

- Claims run one at a time in the daemon, and an item handed to a hook or a resumed run is never handed out again while the daemon runs, so concurrent hooks of one session never take the same item; a claim whose hook never confirms stays without an outcome. Nothing is claimed before a tool runs.
- Pending items of a session that stopped running reach no hook: they are recorded `not_sent:session_ended` or `not_sent:session_stale`. At start the daemon raises the flags of running sessions with pending items and records the rest the same way, including a resumed run that was queued when it stopped.

## Notices to sessions

Signal rules and their shipped table are [operations.md](operations.md#notices). A rule may address sessions (`"to": "sessions where repo:x live:working"`, a session reference or `role:NAME`): the addressed sessions are resolved when the notice fires, and each gets one delivery (sender `signal`, event `notice`, referring to `signal:<rule>`) through the notice's channel: over the hosted connection, where it is a message the agent takes up in a turn, or at its next context-accepting hook event through the mailbox. It appears in that session's next brief with the other notices. A notice never starts or resumes a session: to a session that is not running it is recorded `not_sent:session_not_live`, and to a role with no running instance nothing is sent.

## Delegation records

A **delegation** is a parent session handing work to a child session, recorded from structured records only:
- harness spawns: Claude Code `Agent` and `Task` calls with their subagent transcripts; Codex `spawn_agent` and its spawn edges; Hermes delegation;
- launches from a shell: a child harness started by a parent's tool call (`claude -p`, `codex exec`, or through any runner), shown by the parent's session variables in the child's first hook line or by a process link ([live sessions](live-sessions.md#live-state)).

Text similarity and prompt matching never establish one.

The `delegations` unit holds, per row: parent session and spawning tool call; child session; instruction (the child's first message); model requested and used; result (the child's final message and the parent's tool result); outcome (`completed`, `failed`, `interrupted`, `running`); child cost, tokens and wall time; `how` (`harness`, `inherited_env`, `process`).

Deliveries and delegations together are the context-transfer edges between sessions, which the web context-flow view reads.

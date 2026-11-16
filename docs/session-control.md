# Session control

Resuming, controlling, pinging and handing off sessions use the
[agent contract and live states](agents.md#live-sessions).

## Resume

- **`self`** is the calling session's reference. `polylogue self read --until compaction` (MCP `read` with `ref: self, until: compaction`) re-reads what came before the last compaction.
- **Resume candidates:** the shipped saved query `query:resume_candidates` lists sessions in the current repository whose records end open, with the reason by name: `unanswered_prompt`, `unfinished_tool_call`, `open_note`, `outcome_partial`, `question`. Successors replace predecessors; forks stay separate; sessions judged `solved` or `abandoned` come last. Candidates page until enough are found, with no look-back limit.
- **`REF resume`** (or `@N resume` on a listed candidate; MCP `act`) runs the session's harness resume command from its origin declaration (`claude --resume ID`, `codex resume ID`, `gemini --resume ID`) in the session's **resume directory**: the directory it was launched in, or the one its tool relocated it to (a Claude Code `relocated` record), never a directory it moved to while working. A session with no recorded directory, or one that does not exist here, is refused (`directory_unknown`, `directory_missing`); the command never runs in the caller's or the daemon's directory. `--print` prints the command and the context-assisted alternative (`polylogue REF pack takeover`) instead. A running session is refused (`refused`, `session_running`) unless `--anyway`. Without a reference, `resume` offers the current repository's candidates in the picker ([terminal.md#picker](terminal.md#picker)).
- **A harness resume is never confused with a briefed new session.** A new session started with a pack records a `reference` link with `how: briefing`, never `resume` ([sessions.md#session-links](sessions.md#session-links)).
- **Resume from a message:** `polylogue message:<id> resume --in claude-code|codex [--print] [--copy]` starts a **new** session in the chosen harness, seeded with the pack of `context --at <message>` plus the message text, in the source session's cwd, and records a `reference` link (`how: briefing`) to the message. `--print` prints the command and the pack; `--copy` puts the pack on the clipboard for any other agent. Where a harness offers its own fork of a whole session (`claude --resume ID --fork-session`), `REF resume --fork` uses it and the link is `fork`.

## Live-session control

From the CLI, MCP or the web UI alike, an author acts on a session; each act is a command-table row, recorded as a delivery (event `ping` or `control`, sender `ping` or `act`) with its computed author.

| Operation | Meaning |
|---|---|
| `ping TARGET TEXT [--live] [--about REF]` | an addressed message in any state ([Pings](#pings)); `--live` refuses (`not_sent:session_not_live`) rather than resume a session that is not running |
| `answer SESSION approve\|deny [--text T] [--standing]` | answer the session's pending permission request; `--standing` answers the same tool and input whenever the session asks next in this session |
| `interrupt SESSION` | stop the session's current turn |
| `start TEXT --harness H --cwd DIR` | start a session Polylogue hosts ([Hosted sessions](session-control.md#hosted-sessions)) |
| `start TEXT --role NAME --cwd DIR` | start an instance of a role ([Pings](#pings), Roles) |
| `hand-off SESSION --to H [--at MESSAGE] [--cwd DIR] [--stop-source] [--print]` | hand a session's work to a new hosted session ([Hand-off](#hand-off)) |
| `continue SESSION` | harness resume ([Resume](#resume)) |

The [control channels and harness table](agents.md#control-channels) determine
how each act reaches its session.

**The mailbox.** A ping or notice to a running session waits as a `prepared` delivery and raises the empty flag file `mail/<harness>/<session>` in the archive's runtime directory. At a context-accepting event (Claude Code and Codex `PostToolUse` and `UserPromptSubmit`, Gemini CLI `AfterTool` and `BeforeAgent`, Hermes `pre_llm_call`), the hook checks that one path; only when it exists does it ask the daemon, which hands over every pending item, oldest first, and lowers the flag. The hook records each `delivered` once it has written it to the harness. Claims run one at a time in the daemon, and an item handed to a hook or a resumed run is never handed out again while the daemon runs, so concurrent hooks of one session never take the same item; a claim whose hook never confirms stays without an outcome. Nothing is claimed before a tool runs. Pending items of a session that stopped running reach no hook: they are recorded `not_sent:session_<state>` (`session_ended`, `session_stale`). At start the daemon raises the flags of running sessions with pending items and records the rest the same way, including a resumed run that was queued when it stopped.

**Answers.** A hosted session's requests come to Polylogue's connection, and `answer` replies to the oldest one. Polylogue does not hold a harness's permission prompt (an open product question), so an answer reaches an observed Claude Code session only ahead of time: `answer --standing` records the decision for the pending request's tool and input, and the next `PermissionRequest` hook for the same tool and input in that session returns it at once (recorded `delivered`), once; with none, the hook returns nothing and the harness's own prompt stands. A plain `answer` to an observed session is refused with "answer it in the terminal". A request is `request:<session>#<instant>`; it is answered once, and a later answer records `not_sent:already_answered` naming the first.

### Hosted sessions

`start` (with `--harness`, or `--role`), a ping to a role with no running instance, and `hand-off` start a session Polylogue hosts: it runs on the harness's own control protocol, with the first turn recorded as a `pack` delivery (sender `act`) to the new session whose text opens with the delivery reference.
- **Codex** runs on its shared app-server: Polylogue starts a thread (`thread/start`) and keeps the connection; the app-server's approval and question requests for the thread come to it, a ping steers the running turn (`turn/steer`) or starts one (`turn/start`), `answer` replies to the oldest request, and `interrupt` sends `turn/interrupt`. The app-server keeps the thread, so after a daemon restart Polylogue connects again and subscribes with `thread/resume`.
- **Claude Code** runs as `claude -p --input-format stream-json --output-format stream-json --include-hook-events --permission-prompts host --session-id ID`: a ping is a user message (run after the current turn), `answer` is the control response to its oldest `can_use_tool` request (allow with the tool's input, or deny with the answer's text), `interrupt` is the `interrupt` control request.
- **Gemini CLI and Hermes** run in ACP mode (`gemini --acp`, `hermes acp`): Polylogue initializes, creates the session (`session/new`) and sends each ping as `session/prompt`; a ping while a prompt runs waits for it, since ACP has no steering, and its delivery is recorded `delivered` when it is sent (`not_sent` when an interrupt or a closed host connection drops it before submission; a daemon restart in between leaves it `prepared`). A connection closing after submission does not turn that sent delivery into `not_sent` or cause a resend. `answer` selects the request's `allow_once` or `allow_always` option (deny: `reject_once` or `reject_always`; with no such option the request is cancelled), and `interrupt` sends `session/cancel`, then answers every pending permission request `cancelled`. Polylogue offers the agent no file system or terminal methods.
- **`polylogue host`.** Claude Code's stream-json and ACP run over the agent's stdin and stdout, so a small per-session `polylogue host` process holds them, not the daemon: a daemon restart never ends the agent. The host relays lines between the agent and the daemon's connection; what the agent writes while no daemon is connected waits in the host, and a new connection first gets a line with the session's status, then every request of the agent nobody answered, then what waited. The host writes the session's status (`busy`, `blocked`, `idle`) to the registry route as `HostedStatus` lines naming the agent process it started, which the live state takes as the session's root, and `HostedEnded` when the agent exits, which ends the session. It reads no archive, and exits with the agent. It runs in its own process group, and the service stops the daemon alone (`KillMode=process`), so hosts outlive a restart; they stay in the service's control group with its CPU and I/O weights. A host whose session never started (the protocol refused or timed out) is stopped with its agent.
- **One session at a time per connection.** Each hosted session's connection is used by one act at a time, and only that session's acts wait for it: an adapter slow to answer delays no other session. What a connection reports later (a ping sent after the running prompt, a resumed run's outcome) is recorded by the daemon's acts worker.
- **Reattachment.** Each hosted session has a record in the archive's runtime directory (`hosts/<key>.json`, with the host's socket or the Codex thread). At start the daemon connects to every recorded session again; a record whose host or thread is gone is removed, and one whose service is briefly unavailable (Codex's app-server not up yet) is kept. A hosted session whose live state is `ended` is let go, its record with it.

**Interrupt** goes only through a hosted session's connection; signals to processes are never used. Worktree and permissions are explicit request context, never an ambient directory; a resumed run uses the session's resume directory ([Resume](#resume)). An agent saying "I started the reviewer" does not show a delivery; the delivery and the recipient's transcript do.

## Pings

A **ping** is an explicit address with text: `@[address] text` in an assertion, a page slot or a transcript, the web ping composer, `polylogue ADDRESS ping TEXT`, or MCP `act` `ping`. It is stored as an assertion ([pages.md#ping-records](pages.md#ping-records)) and recorded as one delivery (sender `ping`).

- **The address is explicit:** a session reference (`claude-code:…` or `@[claude-code:…]`), `@[role:NAME]` (the role's newest instance that Polylogue still hosts or that runs; with none, a new instance, below), or `@[QUERY]` for every session a sessions query selects when the ping is sent (a broadcast: one ping assertion about each recipient and one delivery each; sessions that start matching later receive nothing). Page-local member aliases need the member records of [pages.md](pages.md#members-and-roles) and are refused until pages carry them.
- **Execution follows the recipient's live state and the channel function:**
  - running (`working`, `waiting`, `idle`, `unknown`): over the hosted connection, or at its next context-accepting hook event through the mailbox, never interrupting a running tool call; if the session stops running first, `not_sent:session_ended`;
  - `ended` or `stale`: resumed once, headlessly, through the harness's resume command, in the session's resume directory ([Resume](#resume)), with the ping pack as its prompt; without one the delivery is `not_sent:directory_unknown` or `not_sent:directory_missing(DIR)`. No confirmation is asked, whoever sent the ping and through whichever channel. The delivery is `delivered` when the run starts and `failed(<first error line>)` if it exits unsuccessfully;
  - a session not in the archive gives `not_sent:session_missing`; a harness with no headless resume gives `not_sent:unsupported`.
- **The ping pack:** the text, the subject's content and references, the page's table of contents when the ping came from a page, and the instruction to answer through the MCP `assert` operation with a `finding` whose `about` and `answers` links name `assertion:<ping>@<ping>` (or to reach the sender with `ping`), with the delivery reference line. The pack names the markup forms only once markup is parsed.
- **Once per ping.** Re-reading, quoting, translating or re-importing text that contains `@[address]` never sends it again; deliberately submitting the same text again does. The key is the message ID and the markup's position.
- **The reply** is the recipient's regular output, ingested as usual and linked to the ping by an assertion's `answers` link or by the delivery reference found in the recipient's transcript. The resumed session's link to the pinging session is `reference` with `how: ping`.
- **Concurrency:** one resumed run per session at a time; further pings to it queue as `prepared` deliveries and run in order, unless the resumed session's own hook claims them first. There is no rate limit. A sent ping has no undo.
- **Background triggers never start or resume agents** (default; open operator question): a signal or watch firing has no explicit address and no author act, so it only notifies.

**Roles.** A role is a `role` assertion: instructions (a SOUL.md-style text) and the harness to launch. Roles are optional; a session without one runs the harness's default.
- A session records the role it runs: the first-turn `pack` delivery of an instance refers to `role:<name>` and to the role revision (`assertion:<id>`), so editing the role later does not change what earlier sessions ran under.
- Starting a role instance (`polylogue start --role NAME --cwd DIR TEXT`, also over MCP) starts a hosted session of the role's harness with the role's instructions, then the text, as its first turn. A ping to `@[role:NAME]` with no running instance starts one in the directory the role's latest instance ran in, with the ping as the text; a role that never ran has no directory, and the ping is refused (`no_live_instance`) naming the start command.
- Page members map aliases to sessions that are instances of roles ([pages.md](pages.md#members-and-roles)). A role may have several sessions running at once.

## Notices to sessions

Signal rules and their shipped table are [operations.md#notices](operations.md#notices). A rule may address sessions (`"to": "sessions where repo:x live:working"`, a session reference or `role:NAME`): the addressed sessions are resolved when the notice fires, as a ping's are, and each gets one delivery (sender `signal`, event `notice`, referring to `signal:<rule>`) through the notice's channel: over the hosted connection (where it is a message the agent takes up in a turn, as a ping is), or at its next context-accepting hook event through the mailbox. It appears in that session's next brief with the other notices. A notice never starts or resumes a session: to a session that is not running it is recorded `not_sent:session_not_live`, and to a role with no running instance nothing is sent.

## Hand-off

`polylogue hand-off SESSION --to codex|claude-code|gemini-cli|hermes [--at MESSAGE] [--cwd DIR] [--stop-source] [--print]` (MCP `act`) hands a session's work to a new session in any harness Polylogue hosts. The new session starts in the source's recorded directory (or `--cwd`), and its first turn is the source's takeover pack (the whole session's; the pack builder has no cut at a message yet) with a line naming the source and the message to continue from; the turn opens with the pack's delivery reference. The `pack` delivery refers to `session:<source>` and the message, which is how the new session is recorded as briefed from the source: a hand-off is a new session, never a resume or a fork of the source. `--stop-source` then interrupts the source through its channel, recorded like any interrupt; `--print` shows the pack and the channel and starts nothing.

## Delegation records

A **delegation** is a parent session handing work to a child session, recorded from structured records only:
- harness spawns: Claude Code `Agent`/`Task` calls with their subagent transcripts; Codex `spawn_agent` and its spawn edges; Hermes delegation;
- launches from a shell: a child harness started by a parent's tool call (`claude -p`, `codex exec`, or through any runner), evidenced by the parent's session variables in the child's first hook line or by a process link ([Live sessions](agents.md#live-sessions)).

Text similarity and prompt matching never establish one.

The `delegations` unit holds, per row: parent session and spawning tool call; child session; instruction (the child's first message); model requested and used; result (the child's final answer and the parent's tool result); outcome (`completed`, `failed`, `interrupted`, `running`); child cost, tokens and wall time; `how` (`harness`, `inherited_env`, `process`).

Deliveries and delegations together are the context-transfer edges between sessions, which the web context-flow view reads.

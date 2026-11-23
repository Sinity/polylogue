# Authorship

Every authored record in Polylogue (annotations, alternates, withdrawals, judgments, settings, kept results, deliveries, chat actions, provider import jobs) stores who wrote it and how it arrived. The records themselves are [annotations.md](annotations.md); how captured messages record who wrote them is [sessions.md](sessions.md#message-authorship). Agents are primary users with no fewer powers than the operator; what distinguishes the operator is authorship, which the daemon computes and no caller supplies.

## The record

- **`author`** is one of:
  - `operator`;
  - `agent:<session>`, the most specific session known, for example `claude-code:agent-a7f3a2c1000000003` for a Claude subagent or `codex:<thread>`;
  - `agent:<origin>`, when the harness is known and its session is not, for example `agent:gemini-cli`;
  - `agent`, when it is known to be an agent but neither its harness nor its session is;
  - `script:<unit>`, a named non-interactive program, named by its cgroup unit (a systemd unit or a scope);
  - `unattributed`, when nothing shows either way.
- **`via`** is the channel the write arrived on: `mcp`, `cli`, `web`, `extension`, `desktop`, `hook`, `markup`, `import` or `daemon`. There is no attribution blob and no confidence score.

**No surface accepts an author from the caller.** A request that names an author is refused as `bad_request`. A selection or a `--for` option names a subject or a recipient, never the author.

## How each channel decides

The client process gathers its identity and sends it as `caller` in the request envelope: harness, session ID, tool call ID, harness pid, whether a controlling terminal is present, and the systemd unit or scope it runs in. The daemon's one function decides the author from `caller`, `via` and the hook spool, and every answer that used an identity names it under `caller`.

| Channel | Author |
|---|---|
| **MCP**, the agents' primary channel (`polylogue mcp`, a child of the harness) | At start: `CLAUDE_CODE_SESSION_ID` gives `claude-code:<id>`; `CODEX_THREAD_ID` gives `codex:<id>`; `HERMES_SESSION_ID` is resolved through Hermes's state database; otherwise the parent executable names the harness, giving `agent:<origin>`. When hooks are installed, the hook spool's latest `SessionStart` for the server's parent pid wins over the start-up variable, so resumed and cleared sessions are followed. Per call, for Claude Code: `_meta["claudecode/toolUseId"]` is joined to the spooled `PreToolUse` with that `tool_use_id`; when that event carries `agent_id`, the author is `claude-code:agent-<agent_id>`, otherwise its `session_id`. MCP writes are never `operator`. |
| **CLI**, the second channel | Harness identities come from the environment (`CLAUDE_CODE_SESSION_ID` with `CLAUDE_PID`; `CODEX_THREAD_ID` with `CODEX_TOOL_CALL_ID`; `HERMES_SESSION_ID`; `GEMINI_CLI=1`, which has no session). With several present, the nearest harness ancestor in the process tree wins; with several and no harness ancestor, `unattributed`. A harness ancestor is recognised by its executable or by the harness variables it set in the processes below it (an ancestor whose own environment lacks the `CLAUDECODE`, `CODEX_THREAD_ID`, `HERMES_SESSION_ID` or `GEMINI_CLI=1` its children carry), and the harness pid is that ancestor's. A CLI write from a Claude subagent's Bash call is attributed to the main session, which is correct as the root of the work: nothing recorded per process ties the command to one tool call, and an open Bash call in the spool does not tell which process it started. With no harness identity, a controlling terminal gives `operator`; no terminal gives `script:<unit>` when the process's cgroup names a unit, else `unattributed`. |
| **Web UI and HTTP API** | `operator`, `via: web`, with no inspection of the browser. The web UI is the operator's channel; agents are directed to MCP. |
| **Browser extension** | `operator`, `via: extension`, for acts in the extension's own UI. Acts it performs for a request (a turn sent to a web chat, a provider import) carry the requester's author from the request record ([extension.md](extension.md#authorship-from-the-extension)). |
| **Desktop notifications** | `operator`, `via: desktop`, for actions taken on a desktop notification ([operations.md](operations.md#notices)); a rule cannot supply this channel |
| **Hooks** | records a hook writes for a session are `agent:<that session>` or `daemon`, never `operator` |
| **Markup** | the author of the message the marker is in ([markup.md](markup.md)) |
| **Import of user state** | each record keeps its stored author |
| **Launched work** (job runners, schedulers, scripts) | the CLI rule: `script:<unit>` from the cgroup, or the requesting session when the launcher forwards the harness variables. Polylogue recognises no job runner by name and requires no forwarding. |

Harness variable names are harness internals. A test pins the variables seen by a probe of each supported harness version; a renamed variable degrades the author to `agent:<origin>` or to the terminal rule, and never silently to `operator` except through an interactive terminal.

## What the author changes

- **No capability depends on the author.** Every author may write every kind, judgments and settings included, and an agent's writes apply directly, with no proposal step. Pages display text by anyone but the operator that the operator has not judged as proposed, with accept and reject ([pages.md](pages.md#authorship-and-proposals)); that is a reading of judgments and holds nothing back. Settings that spend money or send data out of the machine are no exception; changes by anyone but the operator are listed in `status` until the operator next looks (default; open operator question). The one act that depends on the author is posting into a web chat under the operator's account: a turn into an existing conversation is held whoever sends it, and so is a new conversation submitted by another author, for the operator's `answer allow` ([session-control.md](session-control.md#answer)).
- "Operator" means exactly `operator`; `agent:*`, `script:*` and `unattributed` are best-effort attributions.
- Readers show author and `via` on every record. Query fields: `author`, `by_operator` (exactly `operator`), `agent` (any `agent:*`), `unattributed`, `via`. Unattributed records are labelled "author unknown", never shown as the operator's.
- **Making a record one's own.** The operator makes an unattributed or mis-attributed record their own by a revision authored `operator`. The original stays in history; there is no re-attribution verb.
- Briefs and signals read the author: how briefs treat agents' annotations is [agent-context.md](agent-context.md#the-brief-for-a-directory); which rules may interrupt is [operations.md](operations.md#notices).

## MCP completeness

Every write is a step or operation of the `polylogue_write` or `polylogue_act` tool ([agents.md](agents.md#mcp-server)): `note`, `tag`, `edit`, `withdraw`, `judge`, `save`, `set`, `import` of annotation lines, `send`, `answer`, `interrupt`, `resume`, `start`. Only local commands (`daemon`, `install`, `demo`, `web`, `status --verify`) stay CLI-only, so an agent never needs the CLI to write.

## What authorship does not guarantee

Authorship is for honest attribution, not a security boundary: all agents run as the operator's user, and the threat it addresses is mistakes, not adversaries. Not guaranteed: deliberate impersonation (unsetting variables, typing into a terminal an agent opened, driving the web UI, forging `caller`); an agent using the operator's channels in good faith (recorded as `operator`); the operator running `polylogue` inside a harness's shell mode (recorded as that session; a revision by the operator makes it theirs); attribution to the right subagent for two concurrent CLI calls from one Claude process (recorded as the main session); Gemini sessions (harness only, `agent:gemini-cli`); detached work whose launcher does not forward the harness variables.

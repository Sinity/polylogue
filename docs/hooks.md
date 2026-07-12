[← Back to README](../README.md)

# Hook Integration

Polylogue integrates with AI coding agents (Claude Code, Codex) via their hook
systems to capture session lifecycle data at creation time. This provides 100%
data coverage for events that are not recorded in post-hoc session JSONL.

## How It Works

1. The AI agent invokes `polylogue-hook <event-type>` on each hook event,
   passing the event payload on stdin as JSON.
2. `polylogue-hook` validates the event, enriches it with metadata (provider,
   timestamp, session_id), and atomically writes one immutable envelope to the
   hook spool's `pending/` directory.
3. The daemon watches that directory, persists the envelope in
   `source.db.raw_hook_events`, and moves it to `acknowledged/` only after the
   source-tier transaction commits. Failed writes remain pending for retry.
4. The hook command also retains its legacy per-session JSONL journal for local
   compatibility; the daemon's durable capture route consumes the spool
   envelopes rather than that journal.

## Supported Events

### Claude Code (16 events)

| Event | Trigger | Captured Data | Use case — why wire it |
|-------|---------|---------------|------------------------|
| `SessionStart` | New session starts | session_id, cwd, model, permission_mode | Anchors every session to its initial state — model, working directory, permission posture. Enables real-time context injection if the hook script chooses to echo polylogue summaries back to the agent on session bootstrap. **Recommended.** |
| `Setup` | One-time setup (first CC run) | setup metadata | One-off install marker. Low value for ongoing capture; wire only if you want first-run diagnostics. |
| `UserPromptSubmit` | User submits a prompt | prompt text with paste references BEFORE expansion | **Highest-value event.** The only place `[Pasted text #N]` markers are observable before Claude Code expands them; post-hoc `history.jsonl` extraction misses ~70%. Powers accurate paste detection and the AC2 wiring tracked in [#1654](https://github.com/Sinity/polylogue/issues/1654). **Recommended.** |
| `PreToolUse` | Before tool execution | tool_name, tool_input, tool_call_id | Tool annotations (read_only/destructive), structured input — none of this lands in JSONL. Pairs with `PostToolUse` for end-to-end tool execution audit. **Recommended.** |
| `PostToolUse` | After successful tool execution | tool_name, tool_output, tool_call_id | Tool output before any MCP modification. Pairs with `PreToolUse`. **Recommended.** |
| `PostToolUseFailure` | After failed tool execution | tool_name, error message, is_interrupt flag | Structured error subtypes (transient vs permanent, interrupt vs error). Powers retry/diagnostic heuristics. Wire if you need to attribute failures to specific tools. |
| `PermissionRequest` | Tool needs permission | tool_name, proposed command | Half of the permission audit pair. Records what the agent wanted to do. |
| `PermissionDenied` | User denied permission | tool_name | Other half — records the decision. Together they reconstruct the full permission decision log, which session JSONL never captures. |
| `Notification` | System notification | message, severity | Operator messages routed through the agent UI. Wire if you want a record of what the agent surfaced to the user. |
| `Elicitation` | Modal dialog shown | prompt, options | Modal interaction prompts. Pairs with `ElicitationResult`. |
| `ElicitationResult` | User responds to dialog | selected option | User's modal answer. With `Elicitation` reconstructs interactive sessions. |
| `CwdChanged` | Working directory changed | old_cwd, new_cwd | Mid-session cwd shifts (vs only the initial cwd from `SessionStart`). Wire for accurate per-file attribution in long sessions that change directories. |
| `FileChanged` | File modified by tool | file path, diff stats | Per-tool-call file diff stats. Powers per-session change-magnitude rollups. |
| `WorktreeCreate` | Git worktree created | path | Tool-driven worktree creation events. Wire when running multi-agent workflows that spawn worktrees. |
| `SubagentStart` | Subagent spawned | subagent_type, prompt | Subagent dispatch. Pairs with the subagent's own session id to reconstruct the parent->child lineage. |
| `Stop` | Session ending | session_id, reason | Final session state + termination reason. Pairs with `SessionStart` to bracket every session. **Recommended.** |

### Codex (6 events)

| Event | Trigger | Captured Data | Use case — why wire it |
|-------|---------|---------------|------------------------|
| `SessionStart` | New session starts | session_id, cwd, source | Same role as Claude Code's `SessionStart` — anchors session identity to its initial state. **Recommended.** |
| `UserPromptSubmit` | User submits a prompt | prompt text | Codex prompt capture before any expansion. **Recommended.** |
| `PreToolUse` | Before tool execution | tool_name, tool_input | Pairs with `PostToolUse` for tool execution audit. **Recommended.** |
| `PostToolUse` | After tool execution | tool_name, tool_output | Pairs with `PreToolUse`. **Recommended.** |
| `PermissionRequest` | Tool needs permission | proposed action | Permission audit trail (Codex emits only the request, not a separate denial event — the absence of a follow-up tool execution is the denial signal). |
| `Stop` | Session ending | session_id | Final session state. **Recommended.** |

### Recommended starter set

The five events marked **Recommended** above cover ~95% of the data
not visible in post-hoc JSONL: session lifecycle (`SessionStart` +
`Stop`), paste ground truth (`UserPromptSubmit`), and tool execution
metadata (`PreToolUse` + `PostToolUse`). Wire these first; add the
remaining events when their specific use case applies. The Setup
section below shows configuration for every event — comment out the
ones you don't need.

## Enriched Event Record Format

Each hook event is written as a single JSON line with this structure:

```json
{
  "event_type": "PreToolUse",
  "session_id": "abc123...",
  "timestamp": "2026-05-07T12:00:00Z",
  "provider": "claude-code",
  "payload": {
    "...": "original hook payload from the agent"
  }
}
```

The `payload` field contains the original event data as received from the AI
agent on stdin. The wrapper fields (`event_type`, `session_id`, `timestamp`,
`provider`) are added by `polylogue-hook` for routing and discovery.

## Sidecar Directory Layout

```
~/.local/share/polylogue/hooks/         # Default (XDG_DATA_HOME/polylogue/hooks)
├── pending/
│   └── <event-id>.json                 # Atomic producer envelopes
├── acknowledged/
│   └── <event-id>.json                 # Source-tier receipt after commit
├── claude-code-<session-id>.jsonl      # Legacy Claude journal
└── codex-<session-id>.jsonl            # Legacy Codex journal
```

The daemon creates and watches the pending spool directory on startup, so the
first hook event after a cold start is captured automatically. The documented
spool-root override applies to both the producer and daemon.

## Configuration

### polylogue.toml

```toml
[hooks]
enabled = true
sidecar_dir = "/home/user/.local/share/polylogue/hooks"
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `POLYLOGUE_HOOK_SIDECAR_DIR` | Override the shared producer/daemon spool root |
| `POLYLOGUE_HOOK_PROVIDER` | Force provider detection to `claude-code` or `codex` |

### PolylogueConfig Properties

| Property | Type | Default |
|----------|------|---------|
| `hooks_enabled` | `bool` | `False` |
| `hooks_sidecar_dir` | `str` | `~/.local/share/polylogue/hooks` |

## Installation

`polylogue-hook` is available in three forms:

| Form | Install | Runtime deps |
| --- | --- | --- |
| Standalone PyPI package | `pip install polylogue-hooks` | none (stdlib only) |
| Bundled in main package | `pip install polylogue` | full archive runtime |
| Bash script | copy `contrib/polylogue-hook` from the repository | `bash`, `python3` |

The `polylogue-hooks` package is the recommended path for environments where
you do not want the full polylogue runtime closure (for example, inside the AI
coding agent's own Python environment). The script behaviour is identical
across all three forms and the version is kept in sync with the main package
via release-please (#1309).

The main `polylogue` distribution also installs the `polylogue-hook` entry
point. Once either distribution is installed, wire the recommended starter set
without editing harness settings by hand:

```bash
polylogue hooks install --harness claude-code --events recommended
polylogue hooks install --harness codex --events recommended
```

The command performs a structured, idempotent merge. Existing matcher groups
and handlers are preserved, and a second identical invocation produces no
file diff. Use `--dry-run` to inspect the exact JSON diff first. `--events all`
means every event in Polylogue's current harness catalog; a comma-separated
event list is also accepted. Harness catalogs evolve, so `hooks status --json`
is the authoritative installed-version view rather than a count embedded in
this document.

Inspect wiring and trailing-seven-day evidence with:

```bash
polylogue hooks status
polylogue hooks status --coverage
polylogue hooks status --json
```

Status distinguishes configured wiring from observed events. The coverage
table reports the enrichment role for each event and only treats missing event
types as a liveness gap when the archive supplies a defensible opportunity
denominator (session start, authored prompt, or tool use). Conditional events
such as `Stop` remain observational.

Uninstall is symmetric and removes only handlers whose command invokes
`polylogue-hook`; unrelated hooks in the same event group remain intact:

```bash
polylogue hooks uninstall --harness claude-code
polylogue hooks uninstall --harness codex
```

## Setup

### Claude Code

`polylogue hooks install --harness claude-code` updates
`~/.claude/settings.json` using Claude Code's current three-level hook shape
(event, matcher group, handler). The managed entries look like:

```json
{
  "hooks": {
    "SessionStart": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "polylogue-hook SessionStart --provider claude-code",
            "timeout": 5
          }
        ]
      }
    ],
    "UserPromptSubmit": [
      {"hooks": [{"type": "command", "command": "polylogue-hook UserPromptSubmit --provider claude-code", "timeout": 5}]}
    ],
    "PreToolUse": [
      {"hooks": [{"type": "command", "command": "polylogue-hook PreToolUse --provider claude-code", "timeout": 5}]}
    ],
    "PostToolUse": [
      {"hooks": [{"type": "command", "command": "polylogue-hook PostToolUse --provider claude-code", "timeout": 5}]}
    ],
    "PostToolUseFailure": [
      {"hooks": [{"type": "command", "command": "polylogue-hook PostToolUseFailure --provider claude-code", "timeout": 5}]}
    ],
    "PermissionRequest": [
      {"hooks": [{"type": "command", "command": "polylogue-hook PermissionRequest --provider claude-code", "timeout": 5}]}
    ],
    "PermissionDenied": [
      {"hooks": [{"type": "command", "command": "polylogue-hook PermissionDenied --provider claude-code", "timeout": 5}]}
    ],
    "Notification": [
      {"hooks": [{"type": "command", "command": "polylogue-hook Notification --provider claude-code", "timeout": 5}]}
    ],
    "Stop": [
      {"hooks": [{"type": "command", "command": "polylogue-hook Stop --provider claude-code", "timeout": 5}]}
    ]
  }
}
```

Not all 16 events need to be configured — choose the events relevant to your
capture needs. `UserPromptSubmit` is the most impactful for capturing paste
ground truth before expansion.

### Codex

Current Codex supports both inline TOML and a dedicated `hooks.json` layer.
Polylogue writes `~/.codex/hooks.json`, which avoids reserializing or
reformatting unrelated `config.toml` settings. Existing inline Polylogue hooks
are detected so install does not duplicate them. The generated JSON uses the
same event/matcher-group/handler structure:

```json
{
  "hooks": {
    "SessionStart": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "polylogue-hook SessionStart --provider codex",
            "timeout": 5
          }
        ]
      }
    ]
  }
}
```

Codex requires review of new or changed non-managed command hooks. After
installation, open `/hooks` in Codex and trust the generated definitions. If
`[features].hooks = false` is set in `config.toml`, status reports the harness
as disabled; Polylogue does not silently override that explicit operator
choice.

## Key Data Captured

### Paste Ground Truth (UserPromptSubmit)

The `UserPromptSubmit` hook fires BEFORE paste expansion. The hook payload
contains the raw prompt text with `[Pasted text #N]` markers intact. This
provides 100% paste detection coverage, compared to ~30% from post-hoc
`history.jsonl` extraction.

### Tool Execution Metadata (PreToolUse + PostToolUse)

Tool annotations (read_only/destructive), structured input/output parameters,
and MCP tool output modifications. Data that exists at hook time but is never
written to session JSONL.

### Error Subtypes (PostToolUseFailure)

Structured error information including error type, whether the failure is
transient or permanent, and interrupt flags.

### Permission Audit Trail (PermissionRequest + PermissionDenied)

Complete record of permission decisions: what was proposed, what was decided,
who decided. Currently invisible in session JSONL.

### Session Lifecycle (SessionStart + Stop)

Working directory, model configuration, and permission mode at session start.
Final state at session end.

When a SessionStart hook calls the MCP `compose_context_preamble` tool, it
should forward the provider-supplied session identity as
`successor_session_id`. The session need not be ingested yet: the value names
the new recipient of the context and makes the durable MCP call record
queryable against that successor without guessing a predecessor from time.

### File Change Tracking (FileChanged)

Per-file diff statistics (lines added/removed) for each tool call.

### Working Directory Tracking (CwdChanged)

All directory changes during the session, not just the initial cwd.

## Security and Privacy

- Hook artifacts are stored in the Polylogue blob store (content-addressed,
  immutable) alongside other archive data.
- No hook data is transmitted off-machine. The entire pipeline runs locally.
- Session IDs in filenames are derived from provider-generated identifiers, not
  user-provided data.
- The sidecar directory is under the Polylogue XDG data directory, inheriting
  filesystem permissions.
- Hook payloads may contain sensitive data (prompts, tool inputs/outputs, file
  paths). Treat the hook sidecar directory and blob store with the same security
  posture as the archive database.

## Troubleshooting

**Hook script exits with code 2 ("unsupported event type"):** The event name
passed to `polylogue-hook` must match exactly (case-sensitive). Check the
supported event list above.

**Hook script exits with code 1 ("could not extract session_id"):** The stdin
payload must contain a `session_id`, `sessionId`, or `session` field.

**Hook files not being ingested:** Ensure the Polylogue daemon is running
(`polylogued run`) and that the hooks sidecar directory exists. The daemon
watcher creates it on startup if needed.

**Provider detection fails:** Set `POLYLOGUE_HOOK_PROVIDER=claude-code` or
`POLYLOGUE_HOOK_PROVIDER=codex` in the hook command environment.

---

**See also:** [Configuration](configuration.md) · [Data Model](data-model.md) · [Daemon](daemon.md)

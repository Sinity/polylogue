[← Back to README](../README.md)

# Hook Integration

Polylogue integrates with AI coding agents (Claude Code, Codex) via their hook
systems to capture session lifecycle data at creation time. This provides 100%
data coverage for events that are not recorded in post-hoc session JSONL.

## How It Works

1. The AI agent invokes `polylogue-hook <event-type>` on each hook event,
   passing the event payload on stdin as JSON.
2. `polylogue-hook` validates the event, enriches it with metadata (provider,
   timestamp, session_id), and writes a structured JSONL record to the Polylogue
   hook sidecar directory.
3. The Polylogue daemon watcher picks up the sidecar file and ingests it through
   the standard archive/blob/artifact pipeline.
4. Hook events are classified as `ArtifactKind.HOOK_EVENT` and linked to the
   parent session via `link_group_key`.

## Supported Events

### Claude Code (16 events)

| Event | Trigger | Captured Data |
|-------|---------|---------------|
| `SessionStart` | New session starts | session_id, cwd, model, permission_mode |
| `Setup` | One-time setup (first CC run) | setup metadata |
| `UserPromptSubmit` | User submits a prompt | prompt text with paste references BEFORE expansion |
| `PreToolUse` | Before tool execution | tool_name, tool_input, tool_call_id |
| `PostToolUse` | After successful tool execution | tool_name, tool_output, tool_call_id |
| `PostToolUseFailure` | After failed tool execution | tool_name, error message, is_interrupt flag |
| `PermissionRequest` | Tool needs permission | tool_name, proposed command |
| `PermissionDenied` | User denied permission | tool_name |
| `Notification` | System notification | message, severity |
| `Elicitation` | Modal dialog shown | prompt, options |
| `ElicitationResult` | User responds to dialog | selected option |
| `CwdChanged` | Working directory changed | old_cwd, new_cwd |
| `FileChanged` | File modified by tool | file path, diff stats |
| `WorktreeCreate` | Git worktree created | path |
| `SubagentStart` | Subagent spawned | subagent_type, prompt |
| `Stop` | Session ending | session_id, reason |

### Codex (6 events)

| Event | Trigger | Captured Data |
|-------|---------|---------------|
| `SessionStart` | New session starts | session_id, cwd, source |
| `UserPromptSubmit` | User submits a prompt | prompt text |
| `PreToolUse` | Before tool execution | tool_name, tool_input |
| `PostToolUse` | After tool execution | tool_name, tool_output |
| `PermissionRequest` | Tool needs permission | proposed action |
| `Stop` | Session ending | session_id |

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
├── claude-code-<session-id>.jsonl      # Claude Code events for one session
├── codex-<session-id>.jsonl            # Codex events for one session
└── ...
```

The sidecar directory is watched by the Polylogue daemon. Hook files are
ingested and linked to the parent session automatically.

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
| `POLYLOGUE_HOOK_SIDECAR_DIR` | Override default sidecar directory (in the hook script) |
| `POLYLOGUE_HOOK_PROVIDER` | Force provider detection to `claude-code` or `codex` |

### PolylogueConfig Properties

| Property | Type | Default |
|----------|------|---------|
| `hooks_enabled` | `bool` | `False` |
| `hooks_sidecar_dir` | `str` | `~/.local/share/polylogue/hooks` |

## Setup

### Claude Code

Add hook configuration to Claude Code settings (`.claude/settings.json` or
`~/.claude/settings.json`):

```json
{
  "hooks": {
    "SessionStart": [
      {
        "command": "polylogue-hook SessionStart"
      }
    ],
    "UserPromptSubmit": [
      {
        "command": "polylogue-hook UserPromptSubmit"
      }
    ],
    "PreToolUse": [
      {
        "command": "polylogue-hook PreToolUse"
      }
    ],
    "PostToolUse": [
      {
        "command": "polylogue-hook PostToolUse"
      }
    ],
    "PostToolUseFailure": [
      {
        "command": "polylogue-hook PostToolUseFailure"
      }
    ],
    "PermissionRequest": [
      {
        "command": "polylogue-hook PermissionRequest"
      }
    ],
    "PermissionDenied": [
      {
        "command": "polylogue-hook PermissionDenied"
      }
    ],
    "Notification": [
      {
        "command": "polylogue-hook Notification"
      }
    ],
    "Stop": [
      {
        "command": "polylogue-hook Stop"
      }
    ]
  }
}
```

Not all 16 events need to be configured — choose the events relevant to your
capture needs. `UserPromptSubmit` is the most impactful for capturing paste
ground truth before expansion.

### Codex

Codex hook configuration uses TOML (`.codex/config.toml`):

```toml
[hooks]
SessionStart = ["polylogue-hook SessionStart"]
UserPromptSubmit = ["polylogue-hook UserPromptSubmit"]
PreToolUse = ["polylogue-hook PreToolUse"]
PostToolUse = ["polylogue-hook PostToolUse"]
PermissionRequest = ["polylogue-hook PermissionRequest"]
Stop = ["polylogue-hook Stop"]
```

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

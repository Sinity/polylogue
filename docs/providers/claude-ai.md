# Claude.ai Export Notes

Claude workspace exports (`claude-ai-data-*.zip`) provide a linear conversation log that Polylogue converts into Markdown with attachment metadata and paired tool usage traces.

## Bundle Layout

- `conversations.json` contains ordered `chat_messages` for each conversation.
- Messages expose:
  - `sender` (`user`, `assistant`, `system`).
  - `content` segments (`text`, `tool_use`, `tool_result`) that represent individual blocks within a turn.
  - `attachments` / `files` arrays with file metadata.
- Workspace-level metadata (account, timestamps, optional summaries) sits alongside the conversation data.

## Import Flow

- Iterate `chat_messages` chronologically. Emit one Markdown section per content block, preserving the original order inside each turn.
- Render `tool_use` blocks with the tool name and the JSON arguments; pair them with the subsequent `tool_result` block that references the same ID.
- Capture attachments into `_attachments/`, recording the filename, type, and size so downstream summaries can surface them.
- Populate YAML front matter with conversation metadata (`name`, timestamps, source bundle path) to support reruns and UI summaries.

## Automation Considerations

- Anthropic does not expose a public API for claude.ai history. Users must initiate exports by hand or automate a browser session (Playwright/Selenium) with a stored login.
- Re-imports are idempotent: Polylogue reuses the cached slug and skips conversations whose content hash matches the previous run.

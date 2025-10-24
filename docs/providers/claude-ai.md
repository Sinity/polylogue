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
- Supply `--force` if you need to clobber a hand-edited Markdown file; without it, Polylogue preserves your edits and reports the conversation as dirty.
- Escaped inline markers such as `\[3]` are normalised to `[3]` during import so citations and numbered annotations remain legible.
- Token stats in the front matter are mirrored by `totalWordsApprox`/`inputWordsApprox`, making the approximate word count explicit wherever tokens are reported.
- Conversation metadata is persisted in `XDG_STATE_HOME/polylogue/polylogue.db`, and each import writes a branch-aware tree next to the canonical transcript: `<slug>.md` for the default view, plus `<slug>/conversation.md`, `<slug>/conversation.common.md`, and `branches/<branch-id>/{<branch-id>.md, overlay.md}` capturing alternate paths when forks exist.

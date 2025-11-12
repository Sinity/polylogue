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
- Capture attachments into `attachments/`, recording the filename, type, and size so downstream summaries can surface them.
- Populate YAML front matter with conversation metadata (`name`, timestamps, source bundle path) to support reruns and UI summaries.

### Branch Layout

Claude imports always emit the canonical transcript plus the branch tree: `<slug>/conversation.md`, `<slug>/conversation.common.md`, and `branches/<branch-id>/{<branch-id>.md, overlay.md}`. The registrar keeps branch metadata and overlays in sync whether you import a single bundle or run the sync/watch commands.

## Automation Considerations

- Anthropic does not expose a public API for claude.ai history. Users must initiate exports by hand or automate a browser session (Playwright/Selenium) with a stored login.
- Re-imports are idempotent: Polylogue reuses the cached slug and skips conversations whose content hash matches the previous run.
- Supply `--force` if you need to clobber a hand-edited Markdown file; without it, Polylogue preserves your edits and reports the conversation as dirty.
- Escaped inline markers such as `\[3]` are normalised to `[3]` during import so citations and numbered annotations remain legible.
- Token stats in the front matter are mirrored by `totalWordsApprox`/`inputWordsApprox`, making the approximate word count explicit wherever tokens are reported.
- Conversation metadata is persisted in `XDG_STATE_HOME/polylogue/polylogue.db`, and each import writes a branch-aware tree next to the canonical transcript: `<slug>.md` for the default view, plus `<slug>/conversation.md`, `<slug>/conversation.common.md`, and `branches/<branch-id>/{<branch-id>.md, overlay.md}` capturing alternate paths when forks exist.

## Local Sync Workflow

To keep recurring Claude exports in lockstep with the rest of your archive:

1. Stash every bundle (ZIP or extracted directory with `conversations.json`) under `$XDG_DATA_HOME/polylogue/exports/claude` — typically `~/.local/share/polylogue/exports/claude`.
2. Run `polylogue sync claude` to import the new bundles. Without `--all`, the CLI offers an interactive picker so you can cherry-pick which exports to process.
   - Use `--session /path/to/export.zip` (repeatable) when you want to target specific bundles without touching the picker.
3. Use `--base-dir` to point at another directory, and reuse familiar options like `--html` and `--prune` to control the output layout.
4. Launch `polylogue watch claude` to monitor the export directory continuously; every new ZIP or refreshed `conversations.json` automatically kicks off the same pipeline as a manual sync.

Whether you sync on demand or rely on `polylogue watch claude`, the registrar keeps slugs, token stats, and branch metadata perfectly aligned with the latest export.

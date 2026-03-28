# Multi-Provider Import & Capture Notes

## Export Pipelines

- **OpenAI ChatGPT (consumer web)**
  - Official export bundle contains `conversations.json`, `user.json`, attachments, and shared threads. Each conversation stores a `mapping` graph keyed by node IDs; messages expose `content.parts` with `content_type` values (`text`, `code`, `image_file`, etc.) and metadata (citations, attachments).
  - Import strategy:
    - Flatten the mapping via parent/children edges (topological walk following `current_node` to reconstruct the chosen branch).
    - Decode `parts` preserving callouts, lists, code fences, tables, and citation markers (private-use unicode `\uE200` series) by consulting `message.metadata.citations`.
    - Copy bundled attachments into the Markdown `_attachments` directory or fall back to remote URLs when missing.
    - Treat tool/function messages (`recipient="assistant"` with tool payloads) as dedicated chunks.
  - Automation: ChatGPT supports scheduled exports via Account Settings → Data Controls → Export; user can cron a headless browser request or rely on OpenAI’s emailed download link.

- **Anthropic Claude.ai**
  - Export bundles (`claude-ai-data-*.zip`) include `conversations.json` with ordered `chat_messages`. Each message provides `sender`, `content` segments (`text`, `tool_use`, `tool_result`), plus attachments.
  - Import steps:
    - Iterate `chat_messages` chronologically, rendering each `content` block (text, code, tool logs) into Markdown sections.
    - Ingest attachments (`attachments`, `files`) into local storage or remote links, capturing metadata for the summary section.
    - Preserve conversation metadata (`name`, timestamps) for YAML front matter.
  - Export automation: Claude workspace settings provide manual exports; there is no public API. Users can schedule downloads via headless login (Playwright/Selenium) with stored session cookies, though this is brittle.

- **Other Providers**
  - `cody-chat-history-*.json` (sunsetting): flatten `interactions` arrays into user/assistant turns, embedding repository references.
  - Keep importer architecture modular so additional sources (perplexity, notebooks) can plug in via a common chunk schema.

- **OpenAI Codex CLI Logs (`~/.codex`)**
  - Global state: `config.toml` (model, trust levels), `auth.json` (OAuth tokens), `history.jsonl` (80 MB+ index of all sessions), and `log/codex-tui.log` (CLI telemetry).
  - Session transcripts live under `sessions/` in two formats:
    - Legacy `rollout-*.jsonl` files at the top level (single metadata line).
    - Date-partitioned subdirectories (`sessions/YYYY/MM/DD/...jsonl`) with full JSONL traces: `session_meta`, environment context, user messages, tool calls, assistant replies. Each `session_id` in `history.jsonl` points to the matching file.
  - Import plan: read the JSONL stream, extract `response_item` entries of type `message`, map the `role` (`user`/`assistant`) and `content[].text` fields into canonical chunks, and feed them to our Markdown builder. Attachments are rare; tool outputs appear as separate message types.

- **Claude Code (`~/.config/claude`)**
  - Directory structure captures IDE/workflow state: `projects/` holds per-project JSONL transcripts grouped by workspace name (e.g., `-home-sinity/...jsonl`). Each file is a chronological log with `summary` entries, user prompts, tool runs, and assistant responses.
  - Additional folders (`extras/`, `commands/`, `ide/`, `shell-snapshots/`) store command history, settings, and tool metadata that can enrich imports (e.g., executed shell commands or file snapshots).
  - Import strategy mirrors Codex: parse the JSONL files, normalise message events into chunks, and pull out attachments or summaries when present.

## Live Capture Considerations

- **ChatGPT**: The official REST API only covers developer-created assistants, not consumer chatgpt.com history. Capturing live logs would require browser automation against chatgpt.com, imitating private backend endpoints (`conversation`, `backend-api/conversation`). This is fragile and may breach ToS; treat as opt-in tooling with explicit warnings if pursued.
- **Claude.ai**: Similar story—anthropics’s public APIs target Claude for developers, not claude.ai web traffic. Any live capture would need browser automation using session tokens, with the same caveats.

## Import Adapter Architecture

- Introduce `geminimd/importers/` with provider-specific modules exposing a common interface (e.g., `Iterable[Chunk]`). Each adapter normalises exports into the internal chunk schema before handing off to the Markdown pipeline.
- Validate export payloads via Pydantic/jsonschema to catch format drift per provider.
- Reuse the new MarkdownDocument pipeline for formatting; attachments, stats, and HTML previews become provider-agnostic.

## Future Enhancements

- CLI command `gmd import <path>` detects provider type (zip/json) and delegates to the right importer.
- Optional workflows to push rendered Markdown into knowledge bases (Obsidian vaults, Git repos) or build searchable indices.
- Document recommended export schedules for both services so users can automate regular backups feeding into `gmd`.

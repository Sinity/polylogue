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
  - Global state: `config.toml` (model, trust levels), `auth.json` (OAuth tokens), `history.jsonl` (80 MB+ index of all sessions), and `log/codex-tui.log` (CLI telemetry — client-side `FunctionCall`, errors, etc.).
  - Session transcripts live under `sessions/` in two formats:
    - Legacy `rollout-*.jsonl` files at the top level (single metadata line).
    - Date-partitioned subdirectories (`sessions/YYYY/MM/DD/...jsonl`) with full JSONL traces: `session_meta`, environment context, `response_item` records (`message`, `function_call`, `function_call_output`, `reasoning`), plus auxiliary `event_msg`/`turn_context` lines. Every `session_id` in `history.jsonl` points to one of these files.
  - Import guidelines:
    - Parse the JSONL stream and normalise `response_item:type == "message"` payloads into user/assistant chunks (skip `<user_instructions>`/`<environment_context>` entries so we don’t duplicate boilerplate).
    - `response_item:type == "function_call"`/`"function_call_output"` describe tool invocations and their stdout/stderr. Large argument/result blobs can be written to `_attachments` files and referenced in the Markdown body to keep the conversation readable.
    - `response_item:type == "reasoning"` currently carries encrypted content, so omit them until OpenAI surfaces plaintext traces.
    - The helper script used during analysis writes `/tmp/codex-session.md` and a companion `_attachments/` folder, demonstrating how to wire Codex logs into our existing Markdown pipeline without touching Gemini behaviour.

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

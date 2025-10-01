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
  - Global state: `config.toml` (model, trust levels), `auth.json` (OAuth tokens), `history.jsonl` (~6–7 MB index of all sessions), and `log/codex-tui.log` (~80 MB runtime telemetry with `FunctionCall` traces, errors, etc.).
  - Session transcripts live under `sessions/` in two formats:
    - Legacy `rollout-*.jsonl` files at the top level (single metadata line).
    - Date-partitioned subdirectories (`sessions/YYYY/MM/DD/...jsonl`) with full JSONL traces: `session_meta`, environment context, `response_item` records (`message`, `function_call`, `function_call_output`, `reasoning`), plus auxiliary `event_msg`/`turn_context` lines. Every `session_id` in `history.jsonl` points to one of these files.
  - Import guidelines:
    - Parse the JSONL stream and normalise `response_item:type == "message"` payloads into user/assistant chunks (skip `<user_instructions>`/`<environment_context>` entries so we don’t duplicate boilerplate).
    - `response_item:type == "function_call"`/`"function_call_output"` describe tool invocations and their stdout/stderr. These often contain multi-megabyte JSON blobs; extract anything beyond configurable thresholds into `_attachments` while keeping links in the Markdown body. Consider inlining the first/last N lines for quick context.
    - `response_item:type == "reasoning"` currently carries encrypted content, so omit them until OpenAI surfaces plaintext traces.
    - A helper script (see `/tmp/codex-session.md` during analysis) already demonstrates this flow: it harvests messages, extracts oversized payloads, and feeds the result into the shared Markdown pipeline.

- **Claude Code (`~/.config/claude`)**
  - Directory structure captures IDE/workflow state: `projects/<workspace>/*.jsonl` contains session logs with `summary` nodes (context/compaction checkpoints), `user` prompts, `assistant` replies, and tool interactions. Each record includes `parentUuid`, `cwd`, and `sessionId`, so conversations can branch; “compacted” summaries signal that earlier nodes were rolled up.
  - Supplemental folders—`commands/`, `extras/`, `ide/`, `shell-snapshots/`, `todos/`, `tools/`—hold shell transcripts, file snapshots, and configuration that may need separate parsing or attachment treatment.
  - Import approach:
    - Traverse the JSONL entries in order, reconstruct parent/child relationships when necessary, and normalise `user` / `assistant` messages into chunks (extracting embedded code diffs or logs when large).
    - Treat summaries as front-matter notes or inline callouts; attach file snapshots when present.
    - Apply the same attachment heuristics as Codex so mega-byte diffs/tool outputs are captured without overwhelming the Markdown body.

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

## Requirements & UX Considerations

- **Provider parity**: Normalise ChatGPT, Claude, Codex, Claude Code, etc. into a shared chunk schema, but preserve provider-specific nuances (tool metadata, attachments, reasoning traces) so output can be compared side-by-side.
- **Attachment extraction**:
  - Provide sensible defaults for splitting large inputs/outputs (e.g., extract anything beyond N lines/bytes, keep first/last K lines inline).
  - Allow provider/tool-specific overrides without overwhelming the user—surface high-level toggles (extract tool outputs, keep inline) rather than exposing every knob.
- **Interactive tuning**:
  - After parsing, offer an interactive review (sorted by size, type) so outliers can be inlined, elided, or moved to attachments before writing Markdown.
  - Persist aggregated stats (total size, attachment counts, tokens) and show them dynamically while the user tweaks thresholds.
- **Formatting fidelity**:
  - Render nested JSON/JSONL as fenced code blocks with language hints; keep tables, lists, and code fragments intact.
  - Include provider badges or metadata tags in both Markdown and HTML previews to indicate provenance.
- **Validation & safety**:
  - Validate each provider’s JSON payloads via Pydantic/jsonschema (with a bypass flag if files drift from spec).
  - Plan for optional PII/“sensitive content” detection that redacts or moves such material to attachments.
- **Automation targets**:
  - Support a “sync” mode for local stores (`~/.codex`, `~/.config/claude/projects/`) akin to the Drive sync—watch directories, ingest new sessions automatically, and write Markdown incrementally.
  - Keep automation friendly to systemd timers/services: non-interactive defaults should honour the same extraction rules as the interactive flow.
- **Future GUI**: The interactive workflow may benefit from a richer UI (TUI/HTML) to visualise chunk sizes, attachments, and preview Markdown/HTML side by side.

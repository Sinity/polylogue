# Multi-Provider Import & Capture Notes

## Export Pipelines

- **OpenAI ChatGPT (consumer web)**
  - Official export bundle contains `conversations.json`, `user.json`, attachments, and shared threads. Each conversation stores a `mapping` graph keyed by node IDs; messages expose `content.parts` with `content_type` values (`text`, `code`, `image_file`, etc.) and metadata (citations, attachments).
  - Import strategy:
    - Flatten the mapping via parent/children edges (depth-first/topological walk following `current_node` so the chosen branch is reconstructed in order). Preserve `create_time`/`update_time` as chunk metadata when useful.
    - Decode `content.parts` preserving lists (`list` parts with nested `items`), tables (convert `table` rows to Markdown tables), code blocks (`code` with `mime_type` or language hints), and citation markers (private-use unicode `\uE200` sequences referencing `message.metadata.citations`).
    - Attachments referenced via `attachment://` URIs or metadata should be copied into the `_attachments` directory when present in the ZIP; otherwise link to the remote Drive URL in the export.
    - Treat tool/function messages (`recipient="assistant"` with tool payloads or `content_type == tool_results`) as dedicated chunks, optionally pairing call/response when the export associates them.
  - Automation: ChatGPT supports scheduled exports via Account Settings → Data Controls → Export; user can cron a headless browser request or rely on OpenAI’s emailed download link.

- **Anthropic Claude.ai**
  - Export bundles (`claude-ai-data-*.zip`) include `conversations.json` with ordered `chat_messages`. Each message provides `sender`, `content` segments (`text`, `tool_use`, `tool_result`), plus attachments.
  - Import steps:
    - Iterate `chat_messages` chronologically, rendering each `content` block (text, code diffs, `tool_use`, `tool_result`) into Markdown sections. Tool invocations specify the tool name, arguments (JSON), and an ID linking to the subsequent result.
    - Ingest attachments (`attachments`, `files`) into local storage or remote links, capturing metadata (`file_name`, `file_type`, size) for the summary section.
    - Preserve conversation metadata (`name`, timestamps, account) for YAML front matter; note that `tool_use` blocks can contain multiple operations (command execution, file edits) that may warrant attachments when long.
  - Export automation: Claude workspace settings provide manual exports; there is no public API. Users can schedule downloads via headless login (Playwright/Selenium) with stored session cookies, though this is brittle.

- **Other Providers**
  - `cody-chat-history-*.json` (sunsetting): flatten `interactions` arrays into user/assistant turns, embedding repository references.
  - Keep importer architecture modular so additional sources (perplexity, notebooks) can plug in via a common chunk schema.

- **OpenAI Codex CLI Logs (`~/.codex`)**
  - Global state: `config.toml` (model, trust levels), `auth.json` (OAuth tokens), `history.jsonl` (~6–7 MB index of all sessions), and `log/codex-tui.log` (~80 MB runtime telemetry with `FunctionCall` traces, errors, etc.).
  - Session transcripts live under `sessions/` in two formats:
    - Legacy `rollout-*.jsonl` files at the top level (single metadata line).
    - Date-partitioned subdirectories (`sessions/YYYY/MM/DD/...jsonl`) with full JSONL traces: `session_meta`, environment context, `response_item` records (`message`, `function_call`, `function_call_output`, `reasoning`), plus auxiliary `event_msg`/`turn_context` lines. Every `session_id` in `history.jsonl` points to one of these files.
  - `history.jsonl` exposes (`session_id`, timestamp, summary). Resolve each record by globbing `sessions/**/<session_id>.jsonl`; `internal_storage.json` and `config.json` only hold UI state, not transcripts.
  - `log/codex-tui.log` is an ANSI-coloured append-only log capturing `LocalShellCall`/`FunctionCall` telemetry with timestamps aligned to the JSONL stream, useful for attaching long command outputs as separate artefacts.
  - Import guidelines:
    - Parse the JSONL stream and normalise `response_item:type == "message"` payloads into user/assistant chunks (skip `<user_instructions>`/`<environment_context>` entries so we don’t duplicate boilerplate).
    - `response_item:type == "function_call"`/`"function_call_output"` describe tool invocations and their stdout/stderr. Calls are paired via `call_id`; combine them into a single chunk with the tool name in the header, storing oversized arguments/results in `_attachments` while keeping the first/last N lines inline for context.
    - `response_item:type == "reasoning"` currently carries encrypted content, so omit them until OpenAI surfaces plaintext traces.
    - A helper script (see `/tmp/codex-session.md` during analysis) already demonstrates this flow: it harvests messages, extracts oversized payloads, and feeds the result into the shared Markdown pipeline.

- **Claude Code (`~/.config/claude`)**
  - Directory structure captures IDE/workflow state: `projects/<workspace>/*.jsonl` contains session logs with `summary` nodes (context/compaction checkpoints), `user` prompts, `assistant` replies, and tool interactions. Each record includes `parentUuid`, `cwd`, and `sessionId`, so conversations can branch; “compacted” summaries signal that earlier nodes were rolled up.
  - Workspace folders mirror absolute paths with `/` replaced by `-` (e.g., `/realm/project/sinnix` → `-realm-project-sinnix`). Rows mix `summary`, `user`, `assistant`, `tool_use`, and `tool_result` payloads; use `parentUuid`/`leafUuid` to stitch branches and pair tool calls with their results.
  - Supplemental folders—`commands/` (prompt macros), `extras/` (binary assets), `ide/` (editor state), `shell-snapshots/` (timestamped shell transcripts and `.lock` sentinels), `todos/`, and `tools/` (Python helpers)—hold artefacts that may need attachment or metadata treatment during import.
  - Import approach:
    - Traverse the JSONL entries in order, reconstruct parent/child relationships when necessary (sidechains, branching uuids), and normalise `user` / `assistant` messages into chunks (extracting embedded code diffs or logs when large). Treat `summary` entries as front-matter notes describing the compaction intervals.
    - Treat summaries as front-matter notes or inline callouts; attach file snapshots when present.
    - Apply the same attachment heuristics as Codex so mega-byte diffs/tool outputs are captured without overwhelming the Markdown body.
  - Additional indexes: `history.json` tracks recent sessions per workspace, `activeProject.json` points to the open workspace, and `recentCommands.json` enumerates macro usage. These can seed sync dashboards without reparsing every JSONL.
  - The IDE also keeps `shell-snapshots/<session>/<timestamp>.jsonl` deltas that mirror `tool_result` payloads; when present, link them as attachments so command histories remain accessible even after compaction.

## Live Capture Considerations

- **ChatGPT**: The official REST API only covers developer-created assistants, not consumer chatgpt.com history. Capturing live logs would require browser automation against chatgpt.com, imitating private backend endpoints (`conversation`, `backend-api/conversation`). This is fragile and may breach ToS; treat as opt-in tooling with explicit warnings if pursued.
- **Claude.ai**: Similar story—anthropics’s public APIs target Claude for developers, not claude.ai web traffic. Any live capture would need browser automation using session tokens, with the same caveats.

## Import Adapter Architecture

- Introduce `polylogue/importers/` with provider-specific modules exposing a common interface (e.g., `Iterable[Chunk]`). Each adapter normalises exports into the internal chunk schema before handing off to the Markdown pipeline.
- Validate export payloads via Pydantic/jsonschema to catch format drift per provider.
- Reuse the new MarkdownDocument pipeline for formatting; attachments, stats, and HTML previews become provider-agnostic.

## Future Enhancements

- CLI command `polylogue import <path>` detects provider type (zip/json) and delegates to the right importer.
  - Implemented: `polylogue import chatgpt`, `polylogue import claude`, `polylogue import claude-code`, and `polylogue import codex` share a common Markdown pipeline with interactive skim pickers and consistent attachment policies.
- Local sync parity: `polylogue sync-codex` and `polylogue sync-claude-code` mirror local session stores with the same folding/attachment rules as cloud sync, including skip/prune logic and optional HTML previews.
- Optional workflows to push rendered Markdown into knowledge bases (Obsidian vaults, Git repos) or build searchable indices.
- Document recommended export schedules for both services so users can automate regular backups feeding into Polylogue.
- Explore optional “import assistant” prompts: use an LLM to propose extraction thresholds/tool policies based on a sample session, while still letting the user review diffs before writing files.

## Requirements & UX Considerations

- **Provider parity**: Normalise ChatGPT, Claude, Codex, Claude Code, etc. into a shared chunk schema, but preserve provider-specific nuances (tool metadata, attachments, reasoning traces) so output can be compared side-by-side.
- **Attachment extraction**:
  - Provide sensible defaults for splitting large inputs/outputs (e.g., extract anything beyond N lines/bytes, keep first/last K lines inline) while enforcing a reasonable floor (≈10 lines/blocks) so extraction never triggers on tiny payloads.
  - Extracted files should carry the full original payload—including the previewed leading/trailing lines and tool metadata—so attachments remain authoritative references.
  - Allow provider/tool-specific overrides without overwhelming the user—surface high-level toggles (extract tool outputs, keep inline) rather than exposing every knob.
- **Tool interactions**:
  - Treat a tool call and its output as one logical chunk when the provider links them (include the tool name/kind in the callout header, with arguments/output either inline or in attachments depending on size).
  - Keep reasoning traces separate unless the provider explicitly associates them with a tool call (avoid concatenating consecutive assistant messages blindly).
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
  - When dealing with repeated exports (e.g., recurring ChatGPT take-outs), track per-conversation state so prior formatting decisions persist; if state tracking is unavailable initially, document the wipe-and-rebuild behaviour clearly.
- **Future GUI**: The interactive workflow may benefit from a richer UI (TUI/HTML) to visualise chunk sizes, attachments, and preview Markdown/HTML side by side.

## Data Directory Map

| Provider / Action | Source of Truth | Default Markdown Output |
| --- | --- | --- |
| Gemini render | Local JSON exports / Drive API | `/realm/data/chatlog/markdown/gemini-render` |
| Gemini sync (Drive) | Google Drive “AI Studio” folder | `/realm/data/chatlog/markdown/gemini-sync` |
| ChatGPT import | chat.openai.com export ZIP | `/realm/data/chatlog/markdown/chatgpt` |
| Claude.ai import | claude.ai export bundle | `/realm/data/chatlog/markdown/claude` |
| Codex sync/import | `~/.codex/sessions/*.jsonl` | `/realm/data/chatlog/markdown/codex` |
| Claude Code sync/import | `~/.config/claude/projects/**` | `/realm/data/chatlog/markdown/claude-code` |

Run caches (`state.json`, `runs.json`) stay under `$XDG_STATE_HOME/polylogue/`, while credentials and tokens live in `$XDG_CONFIG_HOME/polylogue/`.

## Supporting Libraries & Integrations

- **Markdown & HTML**: `markdown-it-py` powers Markdown → HTML previews; `python-frontmatter` keeps YAML headers round-trippable. Consider `mdformat` or `markdown-it-attrs` for future formatting controls.
- **Templating**: `jinja2` renders the HTML shell today; it can also drive provider-specific report templates or homepage dashboards. If richer components are needed, evaluate `jinja_partials` or `mako`.
- **Validation**: `pydantic` models keep importer payloads honest; schema drift detectors (e.g., `jsonschema`) can guard against provider changes.
- **Terminal UX**: `rich`, `gum`, `skim`, `bat`, `glow`, and `delta` cover current needs. For more advanced TUIs, `textual` or `prompt_toolkit` could stage a live preview/tuning panel.
- **Automation**: Use `systemd` timers or `cron` to trigger `polylogue sync-*` commands; browser automation (Playwright/Selenium) can schedule ChatGPT / Claude exports when APIs are absent.
- **Data post-processing**: Tools like `ripgrep`, `jq`, or `sqlite-utils` can index the rendered Markdown/HTML; consider wiring an optional SQLite catalogue for cross-provider search.

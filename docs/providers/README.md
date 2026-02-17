[← Back to docs](../README.md)

# Provider Documentation

Polylogue auto-detects provider format from file content — no configuration needed. Each provider has its own parser that converts raw exports into the unified conversation model.

Detection happens in `sources/source.py:detect_provider()` via `looks_like()` probe functions that inspect file structure rather than filenames.

## Supported Providers

| Provider | Format | Detection Signal | Documentation |
|----------|--------|-----------------|---------------|
| **ChatGPT** | `conversations.json` | `mapping` field with UUID graph | [chatgpt.md](chatgpt.md) |
| **Claude (web)** | `.jsonl` / `.json` | `chat_messages` array | [claude-ai.md](claude-ai.md) |
| **Claude Code** | `.jsonl` | `parentUuid`/`sessionId` markers | [claude-code.md](claude-code.md) |
| **Codex** | `.jsonl` | Session envelope structure | [openai-codex.md](openai-codex.md) |
| **Gemini** | Google Drive API | `chunkedPrompt.chunks` structure | [gemini.md](gemini.md) |

ZIP archives are supported (nested ZIPs too, with bomb protection). Encoding fallback handles UTF-8, UTF-8-sig, UTF-16, and UTF-32.

## Session-Level Integration

For detailed documentation on Claude Code session storage, querying, and integration with Polylogue's MCP server, see [Claude Code Session Integration](claude-code-sessions.md).

---

**See also:** [CLI Reference](../cli-reference.md) · [Architecture](../architecture.md) · [Data Model](../data-model.md)

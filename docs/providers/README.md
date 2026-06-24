[← Back to docs](../README.md)

# Provider Documentation

Polylogue auto-detects provider format from file content — no configuration needed. Each provider has its own parser that converts raw exports into the unified session model.

Detection happens in `sources/dispatch.py:detect_provider()` via `looks_like()`
probe functions that inspect file structure.

## Supported Providers

| Provider | Format | Detection Signal | Documentation |
|----------|--------|-----------------|---------------|
| **ChatGPT** | `sessions.json` | `mapping` field with UUID graph | [chatgpt.md](chatgpt.md) |
| **Claude (web)** | `.jsonl` / `.json` | `chat_messages` array | [claude-ai.md](claude-ai.md) |
| **Claude Code** | `.jsonl` | `parentUuid`/`sessionId` markers | [claude-code.md](claude-code.md) |
| **Codex** | `.jsonl` | Session envelope structure | [openai-codex.md](openai-codex.md) |
| **Gemini** | Google Drive API | `chunkedPrompt.chunks` structure | [gemini.md](gemini.md) |


## Usage Telemetry Coverage

Provider detection and transcript parsing do not imply exact usage accounting.
Polylogue declares usage coverage by origin and reports the observed state in
`polylogue diagnostics usage`. Exact provider telemetry, transcript-derived
estimates, unsupported origins, source acquisition debt, and stale rollups are
separate states.

| Origin | Declared usage coverage | Cache semantics |
| --- | --- | --- |
| `claude-code-session` | Exact when `message.usage` records are present | `cache_read_input_tokens` and `cache_creation_input_tokens` stay in cached/cache-write lanes |
| `codex-session` | Exact when `token_count` records are present | `cached_input_tokens` and cache-write/cache-creation aliases stay in separate lanes |
| `chatgpt-export` | Estimate-only transcript text | No provider cache token lanes in the export |
| `claude-ai-export` | Estimate-only transcript text | No provider cache token lanes in the export |
| `aistudio-drive` | Partial message token fields | No cache token lanes in Drive prompt exports |
| `gemini-cli-session` | Partial generic usage dictionaries | Generic cache fields are preserved when present, but provider semantics are not independently declared |
| `hermes-session` | Partial generic usage dictionaries | Generic cache fields are preserved when present, but provider semantics are not independently declared |
| `antigravity-session` | Unsupported | No provider cache token lanes |
| `unknown-export` | Unsupported | No provider cache token lanes |

For exact origins, provider event rows remain distinct from transcript text and
from rebuildable `session_model_usage` rows. `last_token_usage` and
`total_token_usage` are also distinct for Codex: current/request-window counters
may be summed by request, while cumulative counters use the latest per
session/model total to avoid double-counting.

ZIP archives are supported (nested ZIPs too, with bomb protection). Encoding fallback handles UTF-8, UTF-8-sig, UTF-16, and UTF-32.

---

**See also:** [CLI Reference](../cli-reference.md) · [Architecture](../architecture.md) · [Data Model](../data-model.md)

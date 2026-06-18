[← Back to Docs](../README.md)

# Providers

Polylogue detects and parses six known AI chat providers plus an unknown
fallback.

## Known Providers

| Provider | Enum Value | Detection Method | Parser |
|----------|-----------|------------------|--------|
| ChatGPT | `chatgpt` | `mapping` dict with message graph (bundle record) | `sources/parsers/chatgpt.py` |
| Claude.ai | `claude-ai` | `chat_messages` list (bundle record) | `sources/parsers/claude.py` |
| Claude Code | `claude-code` | `parentUuid`/`sessionId` in record array (grouped records) | `sources/parsers/claude.py` (code path) |
| Codex | `codex` | Session envelope structure (grouped records) | `sources/parsers/codex.py` |
| Gemini | `gemini` | `chunkedPrompt.chunks` structure (chunked prompt) | `sources/parsers/drive.py` |
| Drive | `drive` | Google Takeout export structure (chunked prompt) | `sources/parsers/drive.py` |
| Unknown | `unknown` | Fallback: generic message list extraction | `sources/parsers/base.py` |

## Provider Identity

All provider identity flows through the `Provider` enum in
`polylogue/types.py`. The enum normalizes string values via
`canonical_runtime_provider()` and defaults to `UNKNOWN` for unrecognized
strings.

Session IDs use the composite format `provider:provider_id`:

```
claude-code:abc123
chatgpt:def456
claude-ai:ghi789
codex:jkl012
```

## Detection Flow

`detect_provider()` in `sources/dispatch.py` calls each parser's `looks_like()`
function in priority order. Detection is shape-based, not filename-based.

Three payload modes:

| Mode | Providers | Description |
|------|-----------|-------------|
| Bundle record | ChatGPT, Claude.ai | Single JSON object containing a complete session |
| Grouped records | Claude Code, Codex | Array of records with session grouping keys |
| Chunked prompt | Gemini, Drive | Multi-part prompt chunk structure |

## Provider-Specific Parsing

### ChatGPT

Parses the ChatGPT export format: `mapping` dictionary with message nodes,
parent/children graph, and author roles. Extracts session title from the
first user message or the export metadata.

### Claude.ai / Claude Code

Claude.ai exports use a `chat_messages` array with structured content blocks
(text, tool_use, tool_result, thinking). Claude Code exports use a record array
with `parentUuid` / `sessionId` fields to group messages into sessions. Both
paths go through the same parser with a branch point.

### Codex

Parses the OpenAI Codex session envelope format. Detects session boundaries
from the envelope structure and extracts messages with their content blocks and
tool use records.

### Gemini / Drive

Gemini and Google Drive Takeout exports use a chunked prompt structure. The
parser reconstructs sessions from `chunkedPrompt.chunks` arrays. Drive
exports additionally handle Google Docs metadata.

## Browser Capture

In addition to the six known providers, the daemon's browser capture receiver
accepts live capture payloads from local browser extensions. These are parsed
through `sources/parsers/browser_capture.py` and use the same provider detection
pipeline.

## Adding a Provider

1. **Add a `Provider` variant** in `polylogue/types.py`
2. **Implement `looks_like()`** in a new or existing parser under
   `polylogue/sources/parsers/`
3. **Register the detector** in `polylogue/sources/dispatch.py` at the
   appropriate priority level
4. **Add a provider schema bundle** under `polylogue/schemas/providers/`
   (run `devtools lab schema generate` to bootstrap)
5. **Update schema inference** if the new provider introduces novel content
   block types or message structures

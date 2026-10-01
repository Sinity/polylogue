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
`polylogue analyze usage`. Exact provider telemetry, transcript-derived
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
| `hermes-session` | Exact when `state.db` session counters are present | `cache_read_tokens`, `cache_write_tokens`, and reasoning tokens stay in separate lanes |
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

## Grok

The ordinary parser accepts account exports and original app-chat endpoint bundles. Account exports without native IDs retain their declared intrinsic identity. A native bundle contains the original conversation reply, responses reply and optional response-node reply; native `conversationId`, `responseId` and `parentResponseId` remain provider identity. Nested conversation replies and list-shaped responses are accepted alongside the endpoint wrappers.

Native responses remain messages when their prose is empty or absent. Steps become thinking and tool-result blocks; search lists and tool responses retain their structured evidence. Tool outcomes use explicit boolean error or integer exit-code fields. An outcome-free result records `not_reported`. Attachment descriptors retain native file IDs and source locators; the capture owner separately merges acquired bytes using message owner coordinates. Missing response IDs remain absent for the archive's intrinsic identity calculation. Repeated IDs retain separate variant and attachment coordinates.

Parent edges preserve forks. A single leaf establishes its ancestor path; multiple leaves carry no guessed selection. The original response-node reply remains a session event because the retained producer examples do not establish a selected-leaf contract. Conversation revision timestamps and response interruption state survive as session evidence. Raw acquired replies remain the authority for unrecognized fields.

`parse_native_bundle` delegates through `parse_conversation`, the shared ordinary semantic owner. This route materializes Python strings, message lists and blocks. Parser fidelity tests do not establish scalar-independent memory, registered browser backfill transport or live acquisition qualification.

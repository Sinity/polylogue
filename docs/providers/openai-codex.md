# OpenAI Codex CLI Sessions

Polylogue mirrors `~/.codex` session logs, pairing tool calls with their outputs and capturing large payloads as attachments.

## Directory Anatomy

- `config.toml`, `auth.json`: CLI configuration and OAuth tokens (not required for transcript parsing).
- `history.jsonl`: index of all sessions, each row containing `session_id`, timestamps, and summaries.
- `sessions/`: JSONL transcripts, either legacy `rollout-*.jsonl` files or modern date-partitioned paths (`sessions/YYYY/MM/DD/<session_id>.jsonl`).
- `log/codex-tui.log`: append-only ANSI log with `LocalShellCall`/`FunctionCall` traces that supplement the JSONL stream.

## Transcript Structure

- JSONL rows include:
  - `session_meta`: metadata for the session.
  - `response_item` entries with `type` values such as `message`, `function_call`, `function_call_output`, `reasoning`, or auxiliary `event_msg`.
  - `turn_context` and environment records that describe the shell or editor state.
- Every `session_id` in `history.jsonl` maps to one transcript file under `sessions/`.

## Import Guidelines

- Parse the JSONL stream, skipping boilerplate rows like `<user_instructions>` or `<environment_context>` that are repeated in every file.
- Normalise `response_item:type == "message"` rows into user/assistant turns.
- Pair `function_call` entries with their corresponding `function_call_output` (linked via `call_id`). Render them as a single tool block with arguments and results; spill oversized payloads into `_attachments/` while keeping a short inline preview.
- Ignore encrypted `reasoning` payloads until OpenAI exposes a readable form.
- When `log/codex-tui.log` contains extended shell output, attach it alongside the Markdown so the full transcript remains accessible.

## Automation

- `polylogue sync-codex` and `polylogue watch codex` traverse `sessions/`, apply the import pipeline, and preserve modification times on generated Markdown/HTML.
- Repeated runs reuse stored slugs and hashes, skipping untouched sessions while logging summary statistics to `polylogue status`.

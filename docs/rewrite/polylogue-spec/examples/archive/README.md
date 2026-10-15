# Morrow example archive: raw files

Synthetic provider files that produce exactly the facts in `FACTS.md`.

- `build.py` generates every file here (plus `../prices.toml`). Edit it, not
  the generated files, then run `python3 build.py`.
- `check_facts.py` re-derives the facts from the raw files with a naive
  reader and compares them with `FACTS.md`; it exits non-zero on any
  mismatch. Run `python3 check_facts.py`.

| Path | What |
|---|---|
| `claude-code/projects/-home-op-morrow/` | sessions #1, #2, #4 (`<session id>.jsonl`); subagent #3 under `<#2>/subagents/agent-<agentId>.jsonl` with its `.meta.json` |
| `codex/sessions/2026/10/0{5,6}/rollout-*.jsonl` | sessions #5 and #6 (2026 envelope format) |
| `hooks/carriers/claude-code/2026-10-03/` | the slice-2 hook spool: one delivery to #4 naming #2 (`FACTS.md`, "Hook spool (slice 2)") |
| `codex/state_5.sqlite` | Codex state database; a small subset of the real `threads` columns (as in real data, `title` is the first user message and `name` the name the user set) |
| `claude-code-only/claude-code/projects/` | the Claude-Code-only home for slice 1: #1–#4 again, plus the lineage set L1–L8 (`FACTS.md`, "Lineage set"), including a second project folder `-home-op-morrow-old` |
| `live/` | the live set V1–V3 (`FACTS.md`, "Live set"): `hooks/carriers/<provider>/2026-10-09/<pid>-<start>.ndjson`, `procs/2026-10-09.ndjson`, and minimal transcripts |
| `chatgpt/export/conversations.json` | session #7, a `mapping` tree with one abandoned branch |
| `claude-ai/export/conversations.json` | session #8 |

## Counting rules

These are the rules `FACTS.md` uses; the parsers are expected to agree.

- **What is a message.**
  - Claude Code: each `user` record is one message, and each model response
    is one message: `assistant` records sharing `message.id` and `requestId`
    are one message, with usage counted once from the last of them
    (`decisions/one-message-per-response.md`). #2's first response is split
    across two records this way (a text record with partial usage, then a
    record with the two `Read` calls and the final usage). Records
    of other types (`ai-title`, `atis-latch`, `system` compaction boundaries,
    `fork-context-ref`) are not messages. A `user` record with
    `isCompactSummary: true` is part of the compaction event, not a message.
  - Codex: the items one model request produced, up to the `token_count`
    event that closes it, are one model message; each tool output item is
    its own `role=tool` message written by the harness; each user `message`
    item paired with its echo is one human message (`spec/sources/codex.md`).
  - ChatGPT: each node with a `user` or `assistant` message. Messages on the
    path from `current_node` to the root are "on the active path"; the rest
    are abandoned branches.
  - claude.ai: each entry in `chat_messages`.
- **Human prompts** are messages typed by a person: Claude Code `user`
  records with string content (except the first record of a subagent file,
  which is the parent model's instruction and is `written_by: model`,
  `spec/sources/claude-code.md`); Codex user `message` items paired with an
  `event_msg user_message` echo of the same text; ChatGPT `user` nodes;
  claude.ai `human` entries. `user` records carrying `tool_result` blocks
  are written by the harness. Typed prompts carry `promptSource: typed`, as
  current Claude Code writes them; without it a current-version record is
  `written_by: unknown` (`spec/sources/claude-code.md`).
- **Copies in the lineage set.** A main-file record whose `sessionId` is
  not the file's own, a subagent-file record whose `agentId` is not the
  file's own, and a record with `forkedFrom` are copies of another session's
  message. They always name it in `inherited_from`; they count as inherited
  when that message is in the home, and as the session's own while it is
  not (`counted_as`, `spec/model.md`). `isSidechain` records without `agentId` in a
  main file belong to a sidechain session named `<session>~side-<root uuid>`.
  A `user` record with `origin.kind = peer` is written by a model, not a
  human.
- **Inherited messages.** Session #4's file starts with the 19 records (18
  messages) of #2, unchanged, still carrying #2's `sessionId`. They count as
  inherited, not as #4's own messages, and their tokens are not counted
  again.
- **Tool calls** are `tool_use` blocks (Claude Code) and `custom_tool_call`
  items (Codex). Codex's `exec` is a code-mode wrapper: each child call
  recorded by an `item_completed` `CommandExecution` item between the
  wrapper and its output is counted as a tool call named `exec_command`
  with action `shell`, and the wrapper itself is not counted
  (`spec/sources/codex.md`, code mode). A call is an error when its result
  has `is_error: true` (Claude Code) or its `item_completed` item has a
  non-zero `exit_code` (Codex).
- **Tokens.** Claude Code: sum of `message.usage` over the session's own
  assistant records; `cache_creation_input_tokens` is cache write. Codex:
  sum of `last_token_usage` over `token_count` events; Codex's
  `input_tokens` includes cached tokens, so uncached input =
  `input_tokens - cached_input_tokens`.
- **Cost** = tokens × `../prices.toml`, computed exactly, rounded to cents
  only for display. Models missing from the price table (and web chats
  without token counts) have unknown cost.

History of changes to `FACTS.md` while these files were built:
`evidence/plan-history.md`.

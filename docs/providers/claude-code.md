# Claude Code Sessions

Polylogue ingests Claude Code JSONL sessions via typed validation using ClaudeCodeRecord model with semantic content block extraction.

## Supported Inputs

- JSONL files with per-event records (messages, summaries, metadata snapshots).
- Optional `sessions-index.json` for enriched metadata (summary, git branch, project path).

## Semantic Content Extraction

The parser extracts structured semantic data from content_blocks:

- **Thinking traces**: `type: "thinking"` blocks with token approximations.
- **Tool invocations**: Structured `type: "tool_use"` blocks with input parameters and semantic categories (file operations, search, subagent spawns).
- **Tool results**: `type: "tool_result"` blocks with error states.
- **Git operations**: Parsed from `Bash` tool commands (commit, push, checkout, add, rm).
- **File changes**: Operations from Read/Write/Edit tools with path tracking.
- **Subagent spawns**: Task tool invocations with agent type and prompt.
- **Context compaction**: Summary records capturing context truncation events.

## Model & Token Extraction

- Extracts model slug, token usage (input/output/cache), and stop reason from message content.
- Builds conversation-level cost and duration aggregates from message metadata.
- Detects sidechain sessions and branch types.

## Current Behavior

- Normalizes roles via ClaudeCodeRecord.role property (user/assistant/system/unknown).
- Extracts text via content_blocks_raw property with fallback to structured fields.
- Captures working directories and models used across the session.
- Enriches conversation metadata from sessions-index.json when available.

## Limitations

- Workspace file history snapshots are not reconstructed; only captured as metadata.

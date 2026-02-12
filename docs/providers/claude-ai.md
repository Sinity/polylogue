# Claude.ai Exports

Polylogue ingests Claude exports via typed validation using ClaudeAIConversation model, with fallback to untyped extraction.

## Supported Inputs

- ZIP archives containing `conversations.json` (filtered during extraction).
- JSON payloads containing `chat_messages` (Claude AI export format).
- JSON lists with `messages` arrays (generic format fallback).
- JSONL streams containing per-message entries.

## Typed Model Extraction

When validation succeeds via ClaudeAIConversation model:
- Extracts typed message metadata: `uuid`, `text`, `sender`, `created_at`, `updated_at`.
- Normalizes roles via `role_normalized` property.
- Parses timestamps to datetime objects.
- Converts messages to harmonized content blocks.

## Current Behavior

- Converts `chat_messages` into ordered message text with role preservation.
- Captures attachment metadata from `attachments`/`files` lists.
- Falls back to untyped extraction for non-standard exports.

## Limitations

- Attachment binaries are not copied from Claude exports; only metadata is recorded.

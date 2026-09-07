# Claude.ai Exports

Polylogue ingests Claude exports via typed validation using ClaudeAISession model, with fallback to untyped extraction.

## Supported Inputs

- ZIP archives containing `sessions.json` (filtered during extraction).
- JSON payloads containing `chat_messages` (Claude AI export format).
- JSON lists with `messages` arrays (generic format fallback).
- JSONL streams containing per-message entries.

## Typed Model Extraction

When validation succeeds via ClaudeAISession model:
- Extracts typed message metadata: `uuid`, `text`, `sender`, `created_at`, `updated_at`.
- Normalizes roles via `role_normalized` property.
- Parses timestamps to datetime objects.
- Converts messages to harmonized content blocks.

## Current Behavior

- Converts `chat_messages` into ordered message text with role preservation.
- Captures attachment metadata from `attachments`/`files` lists.
- Reads an optional `content_base64` attachment field for exact binary bytes;
  `extracted_content` remains the UTF-8 fallback when no binary carrier is
  supplied.
- Falls back to untyped extraction for non-standard exports.

## Limitations

- A malformed `content_base64` carrier fails parsing rather than being treated
  as extracted text.

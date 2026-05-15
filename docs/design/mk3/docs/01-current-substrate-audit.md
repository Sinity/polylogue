# Current substrate audit for MK3

This is based on the uploaded HEAD snapshot and issue material. It is intentionally terse; the goal is to say what MK3 can assume and what it cannot assume.

## Already real enough to design on

The daemon web shell exists as a vanilla single-page reader with search, facets, conversation list, conversation header, message list, simple inspector tabs, and keyboard navigation. The current HTTP surface exposes health/status, conversations, conversation detail, messages, raw conversation artifacts, facets, sources, reset, ingest, and maintenance endpoints.

The core archive schema already has:

- `conversations` with `parent_conversation_id`, `branch_type`, `source_name`, `working_directories_json`, `git_branch`, `git_repository_url`, and `raw_id`.
- `messages` with `parent_message_id`, `branch_index`, `has_tool_use`, `has_thinking`, `has_paste`, token fields, and `message_type`.
- `content_blocks` with block type, text, tool fields, media type, metadata, and semantic type.
- `attachments` and `attachment_refs` with provider-native identity columns.
- `conversation_stats` with paste/tool/thinking counts.
- `user_marks`, `saved_views`, and `recall_packs` scaffolding.
- `identity_ledger`, which tracks raw-to-conversation identity across resets/reimports.

## Not real enough yet

Current HTTP conversation detail serializes messages mostly as `id`, `role`, `text`, timestamp, message type, word count, and flags. It does not yet expose the full content-block graph, attachment refs, paste spans, annotations, topology edges, or per-message provenance in the reader contract.

`user_marks` are conversation-only and only allow `star`, `pin`, and `archive`; MK3 needs conversation and message targets, plus `important` and `read_later`. There is no durable annotation table visible in the current DDL. Saved views exist as `query_json`, but MK3 still needs an explicit query-spec roundtrip contract and UI/API integration.

Topology is still under-modeled. `conversations.parent_conversation_id` and `messages.parent_message_id` exist, but there is no durable `topology_edges` table in the uploaded snapshot. The preparation path only fills `parent_conversation_id` when the canonical parent is already known; unresolved native parents can be lost as graph facts. That blocks trustworthy session continuation, late-parent repair, fork trees, and sidechain/subagent maps.

Paste handling is message-level (`has_paste`) rather than segment-level. That is enough for filtering and chips, but not enough to render typed preface vs pasted payload, deduplicate pasted artifacts, or copy “typed-only” and “paste-only” separately.

Attachments have a good identity/storage base, but not a product-level rendering contract: preview availability, missing-file status, text extraction, thumbnail status, duplicate/shared evidence, and safe-open/copy actions are not yet modeled for the reader.

## MK3 design posture

The UI should show current facts immediately, but data-model docs should be explicit about needed substrate additions. The first implementation slice should not try to ship every view; it should make the transcript renderer, message action model, target refs, paste spans, attachments, and topology contracts coherent enough that every later screen reuses the same vocabulary.

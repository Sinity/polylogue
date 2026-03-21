# Schema Composition + Quarantine Report (2026-03-06)

## 1) What "quarantine" means here

`quarantine` marks malformed raw payload rows in `raw_conversations` as explicit validation failures so they are visible/auditable and excluded from parse-stage ingestion.

When quarantine is applied, affected rows are updated with:
- `validation_status='failed'`
- `validation_error='<decode/malformed reason>'`
- `validation_mode='strict'`
- `validation_provider='<provider used for validation>'`
- `validated_at='<timestamp>'`
- `parse_error` populated for unparsed rows

Implementation:
- `polylogue/schemas/verification.py` (`verify_raw_corpus(..., quarantine_malformed=True)`)
- CLI surface: `polylogue check --schemas --schema-quarantine-malformed`

## 2) Data model layering and composition

Polylogue keeps two distinct data layers:

1. **Raw layer** (`raw_conversations`)
- Stores provider wire bytes exactly as acquired (`raw_content` BLOB).
- This is the source of truth for schema validation and replayability.

2. **Parsed/normalized layer** (`conversations`, `messages`, `attachments`, `attachment_refs`)
- Derived from raw bytes by provider parsers.
- `conversations.raw_id` links normalized conversation back to raw source row.

### Critical distinction: `raw_content` vs `messages.provider_meta`

- `raw_content` is provider wire format (JSON or JSONL) and validated against provider schemas.
- `messages.provider_meta` is parser-emitted metadata for a normalized message row.
- For most providers, `provider_meta` includes `raw` for re-extraction convenience.
- For **Claude Code**, `provider_meta` intentionally omits `raw` to prevent metadata bloat; raw bytes remain in `raw_conversations.raw_content`.

This is why message-record detection logic that expects `provider_meta.raw` must use raw corpus for Claude Code audits.

## 3) Stage interactions (acquire -> validate -> parse)

### Acquire
- Reads source files and stores bytes to `raw_conversations`.
- No parsing happens here.
- Code: `polylogue/pipeline/services/acquisition.py`.

### Validate
- Decodes `raw_content`, validates against provider schema, persists validation status.
- Strict mode marks failures and records parse_error hints.
- Code: `polylogue/pipeline/services/validation.py`, `polylogue/schemas/validator.py`.

### Parse
- Reads parse-ready raw rows (`passed` / `skipped`) and writes normalized records.
- Persists `conversations.raw_id` FK linkage to original raw row.
- Code: `polylogue/pipeline/services/parsing.py`, `polylogue/pipeline/prepare.py`.

## 4) Where each schema lives

1. **SQLite schema**
- Declared in `polylogue/storage/backends/schema.py`.
- Live database dump generated to:
  - `qa_outputs/Q24_db_schema_dump.sql`
  - `qa_outputs/Q24_db_schema_tables.md`
  - `qa_outputs/Q24_db_schema_inventory.txt`

2. **Provider raw payload JSON Schemas**
- Packaged baselines: `polylogue/schemas/providers/*.schema.json.gz`
- Runtime registry versions: `~/.local/share/polylogue/schemas/<provider>/vN.schema.json.gz`
- Dumped/compared in:
  - `qa_outputs/Q24_schema_dumps/Q24_provider_schema_inventory.md`
  - `qa_outputs/Q24_schema_dumps/packaged/*.schema.json`
  - `qa_outputs/Q24_schema_dumps/runtime_latest/*.schema.json`

## 5) Key code-path map (raw -> parsed)

- Raw ingestion write: `SQLiteBackend.save_raw_conversation(...)`
- Validation persistence: `SQLiteBackend.mark_raw_validated(...)`
- Parse-state persistence: `SQLiteBackend.mark_raw_parsed(...)`
- Parse materialization: `prepare_records(...)` in `polylogue/pipeline/prepare.py`
- Meta extraction behavior for claude-code vs others: `extract_from_provider_meta(...)` in `polylogue/schemas/unified.py`
- Claude parser extracted-meta strategy (no message raw in provider_meta): `polylogue/sources/parsers/claude.py`
- Codex parser raw-preserving strategy (`provider_meta['raw']`): `polylogue/sources/parsers/codex.py`

---

## Appendix A: Live DB table/field dump

# DB Schema Tables (Live)

Database: `/home/sinity/.local/share/polylogue/polylogue.db`

## attachment_refs

| cid | name | type | notnull | dflt_value | pk |
| --- | --- | --- | --- | --- | --- |
| 0 | ref_id | TEXT | 0 |  | 1 |
| 1 | attachment_id | TEXT | 1 |  | 0 |
| 2 | conversation_id | TEXT | 1 |  | 0 |
| 3 | message_id | TEXT | 0 |  | 0 |
| 4 | provider_meta | TEXT | 0 |  | 0 |

Foreign keys:
- `message_id` -> `messages.message_id` on_update=NO ACTION on_delete=SET NULL
- `conversation_id` -> `conversations.conversation_id` on_update=NO ACTION on_delete=CASCADE
- `attachment_id` -> `attachments.attachment_id` on_update=NO ACTION on_delete=CASCADE

## attachments

| cid | name | type | notnull | dflt_value | pk |
| --- | --- | --- | --- | --- | --- |
| 0 | attachment_id | TEXT | 0 |  | 1 |
| 1 | mime_type | TEXT | 0 |  | 0 |
| 2 | size_bytes | INTEGER | 0 |  | 0 |
| 3 | path | TEXT | 0 |  | 0 |
| 4 | ref_count | INTEGER | 1 | 0 | 0 |
| 5 | provider_meta | TEXT | 0 |  | 0 |

## conversations

| cid | name | type | notnull | dflt_value | pk |
| --- | --- | --- | --- | --- | --- |
| 0 | conversation_id | TEXT | 0 |  | 1 |
| 1 | provider_name | TEXT | 1 |  | 0 |
| 2 | provider_conversation_id | TEXT | 1 |  | 0 |
| 3 | title | TEXT | 0 |  | 0 |
| 4 | created_at | TEXT | 0 |  | 0 |
| 5 | updated_at | TEXT | 0 |  | 0 |
| 6 | sort_key | REAL | 0 |  | 0 |
| 7 | content_hash | TEXT | 1 |  | 0 |
| 8 | provider_meta | TEXT | 0 |  | 0 |
| 9 | metadata | TEXT | 0 | '{}' | 0 |
| 10 | version | INTEGER | 1 |  | 0 |
| 11 | parent_conversation_id | TEXT | 0 |  | 0 |
| 12 | branch_type | TEXT | 0 |  | 0 |
| 13 | raw_id | TEXT | 0 |  | 0 |

Foreign keys:
- `raw_id` -> `raw_conversations.raw_id` on_update=NO ACTION on_delete=NO ACTION
- `parent_conversation_id` -> `conversations.conversation_id` on_update=NO ACTION on_delete=NO ACTION

## embedding_status

| cid | name | type | notnull | dflt_value | pk |
| --- | --- | --- | --- | --- | --- |
| 0 | conversation_id | TEXT | 0 |  | 1 |
| 1 | message_count_embedded | INTEGER | 0 | 0 | 0 |
| 2 | last_embedded_at | TEXT | 0 |  | 0 |
| 3 | needs_reindex | INTEGER | 0 | 1 | 0 |
| 4 | error_message | TEXT | 0 |  | 0 |

Foreign keys:
- `conversation_id` -> `conversations.conversation_id` on_update=NO ACTION on_delete=CASCADE

## embeddings_meta

| cid | name | type | notnull | dflt_value | pk |
| --- | --- | --- | --- | --- | --- |
| 0 | target_id | TEXT | 0 |  | 1 |
| 1 | target_type | TEXT | 1 |  | 0 |
| 2 | model | TEXT | 1 |  | 0 |
| 3 | dimension | INTEGER | 1 |  | 0 |
| 4 | embedded_at | TEXT | 1 |  | 0 |
| 5 | content_hash | TEXT | 0 |  | 0 |

## message_embeddings

| cid | name | type | notnull | dflt_value | pk |
| --- | --- | --- | --- | --- | --- |
| 0 | message_id |  | 1 |  | 1 |
| 1 | embedding |  | 0 |  | 0 |
| 2 | provider_name |  | 0 |  | 0 |
| 3 | conversation_id |  | 0 |  | 0 |

## message_embeddings_auxiliary

| cid | name | type | notnull | dflt_value | pk |
| --- | --- | --- | --- | --- | --- |
| 0 | rowid | INTEGER | 0 |  | 1 |
| 1 | value00 |  | 0 |  | 0 |
| 2 | value01 |  | 0 |  | 0 |

## message_embeddings_chunks

| cid | name | type | notnull | dflt_value | pk |
| --- | --- | --- | --- | --- | --- |
| 0 | chunk_id | INTEGER | 0 |  | 1 |
| 1 | size | INTEGER | 1 |  | 0 |
| 2 | validity | BLOB | 1 |  | 0 |
| 3 | rowids | BLOB | 1 |  | 0 |

## message_embeddings_info

| cid | name | type | notnull | dflt_value | pk |
| --- | --- | --- | --- | --- | --- |
| 0 | key | TEXT | 0 |  | 1 |
| 1 | value | ANY | 0 |  | 0 |

## message_embeddings_rowids

| cid | name | type | notnull | dflt_value | pk |
| --- | --- | --- | --- | --- | --- |
| 0 | rowid | INTEGER | 0 |  | 1 |
| 1 | id | TEXT | 1 |  | 0 |
| 2 | chunk_id | INTEGER | 0 |  | 0 |
| 3 | chunk_offset | INTEGER | 0 |  | 0 |

## message_embeddings_vector_chunks00

| cid | name | type | notnull | dflt_value | pk |
| --- | --- | --- | --- | --- | --- |
| 0 | rowid |  | 0 |  | 1 |
| 1 | vectors | BLOB | 1 |  | 0 |

## messages

| cid | name | type | notnull | dflt_value | pk |
| --- | --- | --- | --- | --- | --- |
| 0 | message_id | TEXT | 0 |  | 1 |
| 1 | conversation_id | TEXT | 1 |  | 0 |
| 2 | provider_message_id | TEXT | 0 |  | 0 |
| 3 | role | TEXT | 0 |  | 0 |
| 4 | text | TEXT | 0 |  | 0 |
| 5 | timestamp | TEXT | 0 |  | 0 |
| 6 | sort_key | REAL | 0 |  | 0 |
| 7 | content_hash | TEXT | 1 |  | 0 |
| 8 | provider_meta | TEXT | 0 |  | 0 |
| 9 | version | INTEGER | 1 |  | 0 |
| 10 | parent_message_id | TEXT | 0 |  | 0 |
| 11 | branch_index | INTEGER | 0 | 0 | 0 |

Foreign keys:
- `conversation_id` -> `conversations.conversation_id` on_update=NO ACTION on_delete=CASCADE

## messages_fts

| cid | name | type | notnull | dflt_value | pk |
| --- | --- | --- | --- | --- | --- |
| 0 | message_id |  | 0 |  | 0 |
| 1 | conversation_id |  | 0 |  | 0 |
| 2 | content |  | 0 |  | 0 |

## messages_fts_config

| cid | name | type | notnull | dflt_value | pk |
| --- | --- | --- | --- | --- | --- |
| 0 | k |  | 1 |  | 1 |
| 1 | v |  | 0 |  | 0 |

## messages_fts_content

| cid | name | type | notnull | dflt_value | pk |
| --- | --- | --- | --- | --- | --- |
| 0 | id | INTEGER | 0 |  | 1 |
| 1 | c0 |  | 0 |  | 0 |
| 2 | c1 |  | 0 |  | 0 |
| 3 | c2 |  | 0 |  | 0 |

## messages_fts_data

| cid | name | type | notnull | dflt_value | pk |
| --- | --- | --- | --- | --- | --- |
| 0 | id | INTEGER | 0 |  | 1 |
| 1 | block | BLOB | 0 |  | 0 |

## messages_fts_docsize

| cid | name | type | notnull | dflt_value | pk |
| --- | --- | --- | --- | --- | --- |
| 0 | id | INTEGER | 0 |  | 1 |
| 1 | sz | BLOB | 0 |  | 0 |

## messages_fts_idx

| cid | name | type | notnull | dflt_value | pk |
| --- | --- | --- | --- | --- | --- |
| 0 | segid |  | 1 |  | 1 |
| 1 | term |  | 1 |  | 2 |
| 2 | pgno |  | 0 |  | 0 |

## raw_conversations

| cid | name | type | notnull | dflt_value | pk |
| --- | --- | --- | --- | --- | --- |
| 0 | raw_id | TEXT | 0 |  | 1 |
| 1 | provider_name | TEXT | 1 |  | 0 |
| 2 | source_name | TEXT | 0 |  | 0 |
| 3 | source_path | TEXT | 1 |  | 0 |
| 4 | source_index | INTEGER | 0 |  | 0 |
| 5 | raw_content | BLOB | 1 |  | 0 |
| 6 | acquired_at | TEXT | 1 |  | 0 |
| 7 | file_mtime | TEXT | 0 |  | 0 |
| 8 | parsed_at | TEXT | 0 |  | 0 |
| 9 | parse_error | TEXT | 0 |  | 0 |
| 10 | validated_at | TEXT | 0 |  | 0 |
| 11 | validation_status | TEXT | 0 |  | 0 |
| 12 | validation_error | TEXT | 0 |  | 0 |
| 13 | validation_drift_count | INTEGER | 0 | 0 | 0 |
| 14 | validation_provider | TEXT | 0 |  | 0 |
| 15 | validation_mode | TEXT | 0 |  | 0 |

## runs

| cid | name | type | notnull | dflt_value | pk |
| --- | --- | --- | --- | --- | --- |
| 0 | run_id | TEXT | 0 |  | 1 |
| 1 | timestamp | TEXT | 1 |  | 0 |
| 2 | plan_snapshot | TEXT | 0 |  | 0 |
| 3 | counts_json | TEXT | 0 |  | 0 |
| 4 | drift_json | TEXT | 0 |  | 0 |
| 5 | indexed | INTEGER | 0 |  | 0 |
| 6 | duration_ms | INTEGER | 0 |  | 0 |



---

## Appendix B: Provider schema inventory + top-level field dump

# Provider Schema Inventory (Packaged + Runtime Latest)

| provider | packaged_file | packaged_sha256 | runtime_latest | runtime_sha256 |
| --- | --- | --- | --- | --- |
| chatgpt | `polylogue/schemas/providers/chatgpt.schema.json.gz` | `6e72af508fd3adb8cb7a49e9e8041f72cbac711531eec88e299a82bb6b994b2b` | `v4.schema.json.gz` | `d957126313addf5f85b0cad7f60910f5e61089060a4864de323559ed2c0498d8` |
| claude-ai | `polylogue/schemas/providers/claude-ai.schema.json.gz` | `6f7d6d3b5de958168b86ce4c619361b6c24f2b6346c0be734f19dca7cdda9e2f` | `v2.schema.json.gz` | `b5dfadae1cde4b06e3b65c0610c5cfbbeeb6b9efcbea74723e6da4617588a529` |
| claude-code | `polylogue/schemas/providers/claude-code.schema.json.gz` | `f8cd43f0b018515aebfc0fbee2eebd723c37ddd8be356b84a15b1d93b6236087` | `v2.schema.json.gz` | `e69525416f68cad71f6a50e4656074d7351498bc2d4c706f5cc679da1ceea775` |
| codex | `polylogue/schemas/providers/codex.schema.json.gz` | `4d18f7cdcc19d6254b04c38bfc5071ebcc77633b1599ac8fab7a02444d88e655` | `v9.schema.json.gz` | `786c074f05a6279f0f97c9de3a522fb020b8ee5237f98fe51f30408acd193f78` |
| gemini | `polylogue/schemas/providers/gemini.schema.json.gz` | `f7fdfe818c56960e26baffad28d2f6a11875a10e64f9f53f3646625699a6c4ff` | `v2.schema.json.gz` | `63420108c4be4d1dd989bb161bbe76ad01825fe05f969250d59f1fd0c88abb9e` |

## Top-level schema summaries (packaged)

### chatgpt
- type: `object`
- properties (39): async_status, blocked_urls, chatgpt_plus_user, content, context_scopes, conversation_id, conversation_origin, conversation_template_id, create_time, current_node, default_model_slug, disabled_tool_ids, email, evaluation_name, evaluation_treatment, gizmo_id, gizmo_type, id, is_anonymous, is_archived, is_do_not_remember, is_read_only, is_starred, is_study_mode, mapping, memory_scope, moderation_results, owner, phone_number, plugin_ids, rating, safe_urls, sugar_item_id, sugar_item_visible, title, update_time, user_id, voice, workspace_id
- required (1): id

### claude-ai
- type: `object`
- properties (7): account, chat_messages, created_at, name, summary, updated_at, uuid
- required (7): account, chat_messages, created_at, name, summary, updated_at, uuid

### claude-code
- type: `object`
- properties (56): agentId, apiError, cause, compactMetadata, content, costUSD, customTitle, cwd, data, durationMs, error, gitBranch, hasOutput, hookCount, hookErrors, hookInfos, imagePasteIds, isApiErrorMessage, isCompactSummary, isMeta, isSidechain, isSnapshotUpdate, isVisibleInTranscriptOnly, leafUuid, level, logicalParentUuid, maxRetries, message, messageId, microcompactMetadata, operation, parentToolUseID, parentUuid, permissionMode, planContent, preventedContinuation, requestId, retryAttempt, retryInMs, sessionId, slug, snapshot, sourceToolAssistantUUID, sourceToolUseID, stopReason, subtype, summary, thinkingMetadata, timestamp, todos, toolUseID, toolUseResult, type, userType, uuid, version
- required (1): type

### codex
- type: `object`
- properties (15): arguments, call_id, content, encrypted_content, git, id, instructions, name, output, payload, record_type, role, summary, timestamp, type
- required (0): (none)

### gemini
- type: `object`
- properties (5): applets, chunkedPrompt, citations, runSettings, systemInstruction
- required (0): (none)



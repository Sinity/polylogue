# MK3 data model proposal

The MK3 UI cannot be correct unless these data contracts exist. This document proposes the minimum domain vocabulary, not a broad ORM or generic knowledge graph.

## A. TargetRef: one UI/action target vocabulary

Many MK3 details depend on stable targets: copy/mark/annotate a message, open an attachment, inspect a paste span, jump to a topology edge, save a stack workspace. The current user-state issue says conversation and message targets are the stable first scope; MK3 should still name the full target vocabulary so disabled/deferred states are coherent.

Proposed target ref shape:

```json
{
  "target_type": "conversation | message | content_block | attachment | paste_span | raw_artifact | topology_edge | saved_view | workspace",
  "target_id": "stable id",
  "conversation_id": "optional parent conversation id",
  "message_id": "optional parent message id",
  "block_index": "optional for content_block fallback",
  "identity_key": "optional durable identity for user-state retention"
}
```

Implementation rule: ship conversation/message first. Other target types can render as read-only targets until their identity contract is proven.

## B. Topology edges

Current parent columns are useful fast paths, but insufficient for continuation/fork/subagent modeling. MK3 needs a durable topology table that preserves unresolved native parents and supports late repair.

Proposed table:

```sql
CREATE TABLE topology_edges (
  edge_id TEXT PRIMARY KEY,
  edge_scope TEXT NOT NULL CHECK (edge_scope IN ('conversation','message','artifact')),
  edge_type TEXT NOT NULL CHECK (edge_type IN (
    'continuation','fork','sidechain','subagent','parent','branch','reply','raw_artifact','hook_event'
  )),
  source_kind TEXT NOT NULL CHECK (source_kind IN ('conversation','message','raw_artifact','hook_event','attachment')),
  source_id TEXT NOT NULL,
  target_kind TEXT NOT NULL CHECK (target_kind IN ('conversation','message','raw_artifact','hook_event','attachment')),
  target_id TEXT,
  source_provider_name TEXT,
  source_provider_id TEXT,
  target_provider_id TEXT,
  raw_id TEXT,
  evidence_json TEXT NOT NULL DEFAULT '{}',
  confidence TEXT NOT NULL CHECK (confidence IN ('resolved','inferred','unresolved','ambiguous','cyclic','invalid')),
  resolution_status TEXT NOT NULL CHECK (resolution_status IN ('resolved','unresolved','repaired','rejected','quarantined')),
  observed_source TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  resolved_at TEXT
);
```

Indexes:

```sql
CREATE INDEX idx_topology_source ON topology_edges(source_kind, source_id);
CREATE INDEX idx_topology_target ON topology_edges(target_kind, target_id) WHERE target_id IS NOT NULL;
CREATE INDEX idx_topology_unresolved ON topology_edges(target_provider_id, source_provider_name) WHERE target_id IS NULL;
CREATE INDEX idx_topology_type ON topology_edges(edge_type, confidence);
```

Rules:

- `conversations.parent_conversation_id` remains a denormalized fast pointer for resolved single-parent cases.
- Unknown native parent becomes an unresolved edge, not `NULL` with lost evidence.
- Late parent ingest repairs edges by native provider id and provider/source context.
- Cycle detection marks or rejects the edge with cycle evidence.
- `work_threads` becomes a derived topology summary, not the source of truth.
- The reader consumes topology through archive operations, not raw table queries.

## C. Paste spans

`messages.has_paste` is enough for filtering, but not enough for rendering. MK3 needs paste spans.

```sql
CREATE TABLE message_paste_spans (
  paste_id TEXT PRIMARY KEY,
  conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
  message_id TEXT NOT NULL REFERENCES messages(message_id) ON DELETE CASCADE,
  start_offset INTEGER,
  end_offset INTEGER,
  content_hash TEXT,
  label TEXT,
  paste_type TEXT CHECK (paste_type IN ('text','code','image','file','unknown')),
  line_count INTEGER NOT NULL DEFAULT 0,
  char_count INTEGER NOT NULL DEFAULT 0,
  detection_source TEXT NOT NULL CHECK (detection_source IN ('hook','history','provider_marker','heuristic','manual')),
  confidence TEXT NOT NULL CHECK (confidence IN ('explicit','high','medium','low')),
  created_at TEXT NOT NULL
);
```

Fallback rule: when only `has_paste=1` exists, the renderer creates a synthetic whole-message paste block with `confidence=low`. It must visually say “paste boundaries unknown.”

Useful queries:

- `GET /api/conversations/{id}/pastes`
- `GET /api/pastes?hash=...`
- `GET /api/pastes?provider=claude-code&min_lines=100`

## D. Attachment rendering contract

The current attachment tables are a good identity base. MK3 needs a product layer for preview/extraction and safe display.

```sql
CREATE TABLE attachment_derivatives (
  derivative_id TEXT PRIMARY KEY,
  attachment_id TEXT NOT NULL REFERENCES attachments(attachment_id) ON DELETE CASCADE,
  derivative_type TEXT NOT NULL CHECK (derivative_type IN ('thumbnail','text_extract','metadata','safe_preview')),
  status TEXT NOT NULL CHECK (status IN ('ready','missing','failed','unsupported','too_large','quarantined')),
  mime_type TEXT,
  size_bytes INTEGER,
  blob_ref TEXT,
  error_kind TEXT,
  redacted_error TEXT,
  generated_at TEXT NOT NULL
);
```

Reader attachment states:

- available: can preview/open/copy metadata.
- missing: referenced but no local path/blob.
- remote-only: provider id known, content not local.
- unsupported: no safe preview renderer.
- too_large: preview intentionally skipped.
- quarantined: extractor/sanitizer blocked it.
- shared: same attachment id/hash appears in multiple messages/conversations.

HTML/raw safety rule: attachment previews render text, image, PDF thumbnails, or sanitized isolated preview only. Raw HTML from archives is never injected into the main DOM.

## E. User-state tables

Current marks/saved-views/recall-pack scaffolding should become target-ref based and identity-preserving.

```sql
CREATE TABLE reader_marks (
  mark_id TEXT PRIMARY KEY,
  target_type TEXT NOT NULL CHECK (target_type IN ('conversation','message')),
  target_id TEXT NOT NULL,
  identity_key TEXT,
  mark_kind TEXT NOT NULL CHECK (mark_kind IN ('star','pin','important','read_later','archive')),
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  UNIQUE(target_type, target_id, mark_kind)
);

CREATE TABLE reader_annotations (
  annotation_id TEXT PRIMARY KEY,
  target_type TEXT NOT NULL CHECK (target_type IN ('conversation','message')),
  target_id TEXT NOT NULL,
  identity_key TEXT,
  note_text TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE reader_saved_views (
  view_id TEXT PRIMARY KEY,
  name TEXT NOT NULL UNIQUE,
  query_json TEXT NOT NULL,
  layout_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE reader_workspaces (
  workspace_id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  mode TEXT NOT NULL CHECK (mode IN ('tabs','stack','compare','timeline')),
  query_json TEXT,
  open_targets_json TEXT NOT NULL DEFAULT '[]',
  layout_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
```

Identity retention rule: when `identity_ledger` can map a reimported logical conversation, marks/annotations should be carried by `identity_key`; concrete target IDs are the current realization, not the only durable identity.

## F. Message render envelope

The reader needs a richer message payload than current `text` and flags.

```json
{
  "id": "message id",
  "conversation_id": "conversation id",
  "role": "user|assistant|tool|system",
  "message_type": "message|context|protocol|tool_use|tool_result|summary|unknown",
  "timestamp": "iso timestamp",
  "text": "plain text fallback",
  "content_blocks": [
    {"block_id":"...", "block_index":0, "type":"text|code|tool_use|tool_result|thinking|image|attachment", "text":"...", "semantic_type":"file_read|shell|subagent|thinking|other", "fold_policy":"open|closed|auto"}
  ],
  "attachments": [{"attachment_id":"...", "ref_id":"...", "name":"...", "mime_type":"...", "state":"available|missing|unsupported"}],
  "paste_spans": [{"paste_id":"...", "start_offset":0, "end_offset":123, "confidence":"explicit|low"}],
  "metrics": {"word_count":123, "tokens":{}, "cost":{}},
  "provenance": {"raw_id":"...", "source_path":"redacted", "parser":"claude-code", "confidence":"canonical|inferred"},
  "actions": {"copy_text":true, "copy_markdown":true, "annotate":true, "open_raw":true}
}
```

This envelope lets the UI render message cards consistently, and it avoids doing all classification in frontend heuristics.

"""Concrete column specifications for archive_tiers tables.

Defines the single source of truth for:
  - messages table structure (29 writable columns + 1 GENERATED message_id)
  - blocks table structure (14 writable columns + 1 GENERATED block_id)
  - Other key tables (sessions, session_events, etc.)

Each spec drives INSERT/SELECT generation and typed row extraction.

Extractors are typically defined at module level or passed as callables at spec
construction time. This approach consolidates the column triplicates (column list,
placeholder string, tuple order) into a single spec that drives INSERT/SELECT.
"""

from __future__ import annotations

from polylogue.storage.sqlite.archive_tiers.column_spec import ColumnSpec, TableColumnSpec

# Note: extractors are None here; they will be set in write.py where the
# actual extraction logic lives. This file defines the STRUCTURE only.


def _make_messages_spec() -> TableColumnSpec:
    """Create the messages table column specification.

    The messages table structure (from schema):
      session_id, native_id, parent_message_id, position, role, message_type,
      material_origin, model_name, model_effort, sender_name, recipient,
      delivery_status, end_turn, user_context_text, has_tool_use, has_thinking,
      has_paste, paste_boundary, variant_index, is_active_path, is_active_leaf,
      word_count, input_tokens, output_tokens, cache_read_tokens,
      cache_write_tokens, duration_ms, content_hash, occurred_at_ms

    GENERATED (not writable): message_id

    Special handling: parent_message_id is always NULL on INSERT (no tuple value).
    """
    all_columns = (
        ColumnSpec("message_id", "TEXT", is_generated=True),
        ColumnSpec("session_id", "TEXT"),
        ColumnSpec("native_id", "TEXT"),
        ColumnSpec("parent_message_id", "TEXT", extract_placeholder="NULL"),
        ColumnSpec("position", "INTEGER"),
        ColumnSpec("role", "TEXT"),
        ColumnSpec("message_type", "TEXT"),
        ColumnSpec("material_origin", "TEXT"),
        ColumnSpec("model_name", "TEXT"),
        ColumnSpec("model_effort", "TEXT"),
        ColumnSpec("sender_name", "TEXT"),
        ColumnSpec("recipient", "TEXT"),
        ColumnSpec("delivery_status", "TEXT"),
        ColumnSpec("end_turn", "INTEGER"),
        ColumnSpec("user_context_text", "TEXT"),
        ColumnSpec("has_tool_use", "INTEGER"),
        ColumnSpec("has_thinking", "INTEGER"),
        ColumnSpec("has_paste", "INTEGER"),
        ColumnSpec("paste_boundary", "TEXT"),
        ColumnSpec("variant_index", "INTEGER"),
        ColumnSpec("is_active_path", "INTEGER"),
        ColumnSpec("is_active_leaf", "INTEGER"),
        ColumnSpec("word_count", "INTEGER"),
        ColumnSpec("input_tokens", "INTEGER"),
        ColumnSpec("output_tokens", "INTEGER"),
        ColumnSpec("cache_read_tokens", "INTEGER"),
        ColumnSpec("cache_write_tokens", "INTEGER"),
        ColumnSpec("duration_ms", "INTEGER"),
        ColumnSpec("content_hash", "BLOB"),
        ColumnSpec("occurred_at_ms", "INTEGER"),
    )

    writable_columns = tuple(col for col in all_columns if not col.is_generated)

    return TableColumnSpec(
        table_name="messages",
        all_columns=all_columns,
        writable_columns=writable_columns,
    )


def _make_blocks_spec() -> TableColumnSpec:
    """Create the blocks table column specification.

    The blocks table structure (from schema):
      session_id, message_id, position, block_type, text, tool_name, tool_id,
      tool_input, semantic_type, media_type, language, tool_result_is_error,
      tool_result_exit_code, content_hash

    GENERATED (not writable):
      block_id, tool_command, tool_path, search_text, tool_detail_text
    """
    all_columns = (
        ColumnSpec("block_id", "TEXT", is_generated=True),
        ColumnSpec("message_id", "TEXT"),
        ColumnSpec("session_id", "TEXT"),
        ColumnSpec("position", "INTEGER"),
        ColumnSpec("block_type", "TEXT"),
        ColumnSpec("text", "TEXT"),
        ColumnSpec("tool_name", "TEXT"),
        ColumnSpec("tool_id", "TEXT"),
        ColumnSpec("tool_input", "TEXT"),
        ColumnSpec("semantic_type", "TEXT"),
        ColumnSpec("media_type", "TEXT"),
        ColumnSpec("language", "TEXT"),
        ColumnSpec("tool_result_is_error", "INTEGER"),
        ColumnSpec("tool_result_exit_code", "INTEGER"),
        ColumnSpec("content_hash", "BLOB"),
        ColumnSpec("tool_command", "TEXT", is_generated=True),
        ColumnSpec("tool_path", "TEXT", is_generated=True),
        ColumnSpec("search_text", "TEXT", is_generated=True),
        ColumnSpec("tool_detail_text", "TEXT", is_generated=True),
    )

    writable_columns = tuple(col for col in all_columns if not col.is_generated)

    return TableColumnSpec(
        table_name="blocks",
        all_columns=all_columns,
        writable_columns=writable_columns,
    )


# Global registry of table specs
MESSAGES_SPEC = _make_messages_spec()
BLOCKS_SPEC = _make_blocks_spec()

TABLE_SPECS = {
    "messages": MESSAGES_SPEC,
    "blocks": BLOCKS_SPEC,
}

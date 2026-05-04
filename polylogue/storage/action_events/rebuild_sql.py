"""SQL fragments used by action-event read-model rebuilds."""

from __future__ import annotations

from polylogue.core.common import SQL_ACTION_EVENT_INSERT as _ACTION_EVENT_INSERT_SQL

ACTION_EVENT_REPAIR_CANDIDATE_IDS_SQL = """
    SELECT DISTINCT cb.conversation_id
    FROM content_blocks cb
    JOIN conversations c ON c.conversation_id = cb.conversation_id
    WHERE cb.type = 'tool_use'
      AND (
          NOT EXISTS (
              SELECT 1
              FROM action_events ae
              WHERE ae.conversation_id = cb.conversation_id
          )
          OR EXISTS (
              SELECT 1
              FROM action_events ae
              WHERE ae.conversation_id = cb.conversation_id
                AND ae.materializer_version != ?
          )
      )
    ORDER BY cb.conversation_id
"""
ACTION_EVENT_VALID_SOURCE_IDS_SQL = """
    SELECT DISTINCT cb.conversation_id
    FROM content_blocks cb
    JOIN conversations c ON c.conversation_id = cb.conversation_id
    WHERE cb.type = 'tool_use'
    ORDER BY cb.conversation_id
"""
ACTION_EVENT_CONVERSATION_IDS_SQL = """
    SELECT DISTINCT conversation_id
    FROM content_blocks
    WHERE type = 'tool_use'
    ORDER BY conversation_id
"""
# Re-export from canonical source so rebuild callers and ingest_batch both
# use the same SQL template.
ACTION_EVENT_INSERT_SQL = _ACTION_EVENT_INSERT_SQL

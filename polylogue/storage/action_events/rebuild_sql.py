"""SQL fragments used by action-event read-model rebuilds."""

from __future__ import annotations

from polylogue.core.common import SQL_ACTION_EVENT_INSERT as _ACTION_EVENT_INSERT_SQL

ACTION_EVENT_REPAIR_CANDIDATE_IDS_SQL = """
    SELECT DISTINCT cb.session_id
    FROM content_blocks cb
    JOIN sessions c ON c.session_id = cb.session_id
    WHERE cb.type = 'tool_use'
      AND (
          NOT EXISTS (
              SELECT 1
              FROM action_events ae
              WHERE ae.session_id = cb.session_id
          )
          OR EXISTS (
              SELECT 1
              FROM action_events ae
              WHERE ae.session_id = cb.session_id
                AND ae.materializer_version != ?
          )
      )
    ORDER BY cb.session_id
"""
ACTION_EVENT_VALID_SOURCE_IDS_SQL = """
    SELECT DISTINCT cb.session_id
    FROM content_blocks cb
    JOIN sessions c ON c.session_id = cb.session_id
    WHERE cb.type = 'tool_use'
    ORDER BY cb.session_id
"""
ACTION_EVENT_SESSION_IDS_SQL = """
    SELECT DISTINCT session_id
    FROM content_blocks
    WHERE type = 'tool_use'
    ORDER BY session_id
"""
# Re-export from canonical source so rebuild callers and ingest_batch both
# use the same SQL template.
ACTION_EVENT_INSERT_SQL = _ACTION_EVENT_INSERT_SQL

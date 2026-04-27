"""SQL fragments used by action-event read-model rebuilds."""

from __future__ import annotations

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
ACTION_EVENT_INSERT_SQL = """
    INSERT INTO action_events (
        event_id,
        conversation_id,
        message_id,
        materializer_version,
        source_block_id,
        timestamp,
        sort_key,
        sequence_index,
        provider_name,
        action_kind,
        tool_name,
        normalized_tool_name,
        tool_id,
        affected_paths_json,
        cwd_path,
        branch_names_json,
        command,
        query_text,
        url,
        output_text,
        search_text
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

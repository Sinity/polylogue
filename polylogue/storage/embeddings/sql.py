"""SQL fragments used by embedding-statistics readers."""

from __future__ import annotations

EMBEDDED_SESSIONS_SQL = "SELECT COUNT(*) FROM embedding_status WHERE needs_reindex = 0"
PENDING_SESSIONS_SQL = "SELECT COUNT(*) FROM embedding_status WHERE needs_reindex = 1"
PENDING_MESSAGES_SQL = """
    SELECT COUNT(*)
    FROM messages m
    JOIN sessions c ON c.session_id = m.session_id
    LEFT JOIN embedding_status e ON e.session_id = c.session_id
    WHERE e.session_id IS NULL OR e.needs_reindex = 1
"""
EMBEDDED_MESSAGES_SQL = "SELECT COUNT(*) FROM message_embeddings"
MISSING_META_MESSAGES_SQL = """
    SELECT COUNT(*)
    FROM message_embeddings me
    LEFT JOIN embeddings_meta em
      ON em.target_id = me.message_id
     AND em.target_type = 'message'
    WHERE em.target_id IS NULL
"""
STALE_MESSAGES_SQL = """
    SELECT COUNT(*)
    FROM message_embeddings me
    JOIN messages m ON m.message_id = me.message_id
    LEFT JOIN embeddings_meta em
      ON em.target_id = me.message_id
     AND em.target_type = 'message'
    WHERE em.target_id IS NULL
       OR (em.content_hash IS NOT NULL AND em.content_hash != m.content_hash)
"""
EMBEDDED_AT_BOUNDS_SQL = """
    SELECT MIN(embedded_at) AS oldest_embedded_at, MAX(embedded_at) AS newest_embedded_at
    FROM embeddings_meta
    WHERE target_type = 'message'
"""
MODEL_COUNTS_SQL = """
    SELECT model, COUNT(*) AS count
    FROM embeddings_meta
    WHERE target_type = 'message'
    GROUP BY model
    ORDER BY count DESC, model ASC
"""
DIMENSION_COUNTS_SQL = """
    SELECT dimension, COUNT(*) AS count
    FROM embeddings_meta
    WHERE target_type = 'message'
    GROUP BY dimension
    ORDER BY count DESC, dimension ASC
"""
SESSIONS_EXISTS_SQL = "SELECT name FROM sqlite_master WHERE type='table' AND name='sessions'"
EMBEDDING_FAILURE_COUNT_SQL = "SELECT COUNT(*) FROM embedding_status WHERE error_message IS NOT NULL"
STORED_MODEL_SQL = """
    SELECT DISTINCT model
    FROM embeddings_meta
    WHERE target_type = 'message'
    ORDER BY model
"""
TOTAL_MESSAGES_SQL = """
    SELECT COUNT(*) AS message_count
    FROM messages
"""

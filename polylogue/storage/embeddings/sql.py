"""SQL fragments used by embedding-statistics readers."""

from __future__ import annotations

EMBEDDED_CONVERSATIONS_SQL = "SELECT COUNT(*) FROM embedding_status WHERE needs_reindex = 0"
PENDING_CONVERSATIONS_SQL = "SELECT COUNT(*) FROM embedding_status WHERE needs_reindex = 1"
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
CONVERSATIONS_EXISTS_SQL = "SELECT name FROM sqlite_master WHERE type='table' AND name='conversations'"

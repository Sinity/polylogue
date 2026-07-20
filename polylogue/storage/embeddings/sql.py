"""SQL fragments used by embedding-statistics readers."""

from __future__ import annotations

EMBEDDABLE_MESSAGE_WHERE = """
    m.message_type = 'message'
    AND m.role IN ('user', 'assistant')
    AND m.material_origin IN ('human_authored', 'assistant_authored')
    AND m.word_count > 0
"""
ELIGIBLE_SESSIONS_SQL = f"""
    SELECT COUNT(*)
    FROM (
        SELECT m.session_id
        FROM messages m
        WHERE {EMBEDDABLE_MESSAGE_WHERE}
        GROUP BY m.session_id
    ) eligible_sessions
"""
EMBEDDED_SESSIONS_SQL = "SELECT COUNT(*) FROM embedding_status WHERE needs_reindex = 0"
PENDING_SESSIONS_SQL = f"""
    WITH eligible_sessions AS (
        SELECT m.session_id, COUNT(*) AS message_count
        FROM messages m
        WHERE {EMBEDDABLE_MESSAGE_WHERE}
        GROUP BY m.session_id
    )
    SELECT COUNT(*)
    FROM eligible_sessions es
    LEFT JOIN embedding_status e ON e.session_id = es.session_id
    WHERE e.session_id IS NULL
       OR e.needs_reindex = 1
       OR e.message_count_embedded < es.message_count
"""
PENDING_MESSAGES_SQL = """
    SELECT COUNT(*)
    FROM messages m
    JOIN sessions c ON c.session_id = m.session_id
    LEFT JOIN embedding_status e ON e.session_id = c.session_id
    WHERE (
            e.session_id IS NULL
         OR e.needs_reindex = 1
    )
      AND m.message_type = 'message'
      AND m.role IN ('user', 'assistant')
      AND m.material_origin IN ('human_authored', 'assistant_authored')
      AND m.word_count > 0
"""
# v4 (polylogue-q88p): a message's vector is looked up through
# message_embedding_refs (message_id -> embedding_input_hash), not directly
# by message_id -- message_embeddings/message_embeddings_meta are keyed by
# the content-addressed hash and are shared/deduped across messages.
EMBEDDED_MESSAGES_SQL = "SELECT COUNT(*) FROM message_embedding_refs"
MISSING_META_MESSAGES_SQL = """
    SELECT COUNT(*)
    FROM message_embedding_refs r
    LEFT JOIN message_embeddings_meta em
      ON em.embedding_input_hash = r.embedding_input_hash
    WHERE em.embedding_input_hash IS NULL
"""
EMBEDDED_AT_BOUNDS_SQL = """
    SELECT MIN(embedded_at_ms) AS oldest_embedded_at, MAX(embedded_at_ms) AS newest_embedded_at
    FROM message_embeddings_meta
"""
MODEL_COUNTS_SQL = """
    SELECT model, COUNT(*) AS count
    FROM message_embeddings_meta
    GROUP BY model
    ORDER BY count DESC, model ASC
"""
DIMENSION_COUNTS_SQL = """
    SELECT dimension, COUNT(*) AS count
    FROM message_embeddings_meta
    GROUP BY dimension
    ORDER BY count DESC, dimension ASC
"""
SESSIONS_EXISTS_SQL = "SELECT name FROM sqlite_master WHERE type='table' AND name='sessions'"
EMBEDDING_FAILURE_COUNT_SQL = "SELECT COUNT(*) FROM embedding_status WHERE error_message IS NOT NULL"
STORED_MODEL_SQL = """
    SELECT DISTINCT model
    FROM message_embeddings_meta
    ORDER BY model
"""
TOTAL_MESSAGES_SQL = """
    SELECT COUNT(*) AS message_count
    FROM messages m
    WHERE m.message_type = 'message'
      AND m.role IN ('user', 'assistant')
      AND m.material_origin IN ('human_authored', 'assistant_authored')
      AND m.word_count > 0
"""

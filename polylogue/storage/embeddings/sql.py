"""SQL fragments used by embedding-statistics readers."""

from __future__ import annotations

from polylogue.storage.embeddings.identity import EMBEDDING_INPUT_HASH_SQL_FUNCTION, sql_string_literal

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


def stale_messages_sql(model: str) -> str:
    """Count refs whose recorded hash no longer matches the message's live text.

    There is no per-vector "stale" state under content-addressing (a hash
    with a meta row is fresh by construction); "stale" instead means the
    *ref* is out of date -- the message's current embedder input text hashes
    differently than what was recorded the last time it was embedded, so a
    fresh vector for the new hash has not been written yet even though an
    old ref row still points at the superseded one. Requires
    ``register_embedding_identity_sql(conn)`` to already be installed on the
    connection this runs against.
    """
    model_literal = sql_string_literal(model)
    prose_expr = (
        "(SELECT GROUP_CONCAT(b.text, char(10)||char(10)) "
        "FROM blocks b WHERE b.message_id = m.message_id "
        "AND b.block_type = 'text' AND b.text IS NOT NULL ORDER BY b.position)"
    )
    return f"""
    SELECT COUNT(*)
    FROM message_embedding_refs r
    JOIN messages m ON m.message_id = r.message_id
    WHERE m.message_type = 'message'
      AND m.role IN ('user', 'assistant')
      AND m.material_origin IN ('human_authored', 'assistant_authored')
      AND m.word_count > 0
      AND r.embedding_input_hash != {EMBEDDING_INPUT_HASH_SQL_FUNCTION}({model_literal}, {prose_expr})
    """


# Legacy (v3) shape, kept for readers that may still be pointed at a
# not-yet-rebuilt embeddings.db: message_embeddings/message_embeddings_meta
# were message_id-keyed and compared against messages.content_hash directly.
STALE_MESSAGES_SQL_LEGACY = """
    SELECT COUNT(*)
    FROM message_embeddings me
    JOIN messages m ON m.message_id = me.message_id
    LEFT JOIN message_embeddings_meta em
      ON em.message_id = me.message_id
    WHERE (
            em.message_id IS NULL
         OR (em.content_hash IS NOT NULL AND em.content_hash != m.content_hash)
    )
      AND m.message_type = 'message'
      AND m.role IN ('user', 'assistant')
      AND m.material_origin IN ('human_authored', 'assistant_authored')
      AND m.word_count > 0
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

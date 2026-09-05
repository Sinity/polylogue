"""Session-scoped materialization of deterministic tool action pairs."""

from __future__ import annotations

import sqlite3


def action_pairs_refresh_sql(session_expr: str, *, session_index_hint: str = "") -> str:
    """Return the bounded insert used by writers and fixture-maintaining triggers."""
    return f"""
        INSERT INTO action_pairs (
            tool_use_block_id, session_id, message_id, tool_id, use_rank,
            tool_name, semantic_type, tool_command, tool_path,
            tool_result_block_id, is_error, exit_code
        )
        WITH ranked_uses AS (
            SELECT u.session_id, u.message_id, u.block_id AS tool_use_block_id,
                   u.tool_name, u.semantic_type, u.tool_command, u.tool_path,
                   u.tool_id,
                   ROW_NUMBER() OVER (
                       PARTITION BY u.session_id, u.tool_id
                       ORDER BY um.position, um.variant_index, u.position
                   ) AS use_rank
            FROM blocks u{session_index_hint} JOIN messages um ON um.message_id = u.message_id
            WHERE u.session_id = {session_expr} AND u.block_type = 'tool_use'
              AND u.tool_id IS NOT NULL AND u.tool_id != ''
        ), ranked_results AS (
            SELECT r.session_id, r.tool_id, r.block_id AS tool_result_block_id,
                   r.tool_result_is_error AS is_error,
                   r.tool_result_exit_code AS exit_code,
                   ROW_NUMBER() OVER (
                       PARTITION BY r.session_id, r.tool_id
                       ORDER BY rm.position, rm.variant_index, r.position
                   ) AS result_rank
            FROM blocks r{session_index_hint} JOIN messages rm ON rm.message_id = r.message_id
            WHERE r.session_id = {session_expr} AND r.block_type = 'tool_result'
              AND r.tool_id IS NOT NULL AND r.tool_id != ''
        )
        SELECT u.tool_use_block_id, u.session_id, u.message_id, u.tool_id, u.use_rank,
               u.tool_name, u.semantic_type, u.tool_command, u.tool_path,
               r.tool_result_block_id, r.is_error, r.exit_code
        FROM ranked_uses u LEFT JOIN ranked_results r
          ON r.session_id = u.session_id AND r.tool_id = u.tool_id AND r.result_rank = u.use_rank
        UNION ALL
        SELECT u.block_id, u.session_id, u.message_id, u.tool_id, NULL,
               u.tool_name, u.semantic_type, u.tool_command, u.tool_path,
               NULL, NULL, NULL
        FROM blocks u{session_index_hint}
        WHERE u.session_id = {session_expr} AND u.block_type = 'tool_use'
          AND (u.tool_id IS NULL OR u.tool_id = '')
    """


def refresh_action_pairs(conn: sqlite3.Connection, session_id: str) -> None:
    """Rebuild action pairs for one changed session inside its write transaction."""
    conn.execute("DELETE FROM action_pairs WHERE session_id = ?", (session_id,))
    has_session_index = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'index' AND name = 'idx_blocks_session_position'"
    ).fetchone()
    session_index_hint = " INDEXED BY idx_blocks_session_position" if has_session_index is not None else ""
    conn.execute(
        action_pairs_refresh_sql("?", session_index_hint=session_index_hint), (session_id, session_id, session_id)
    )


def action_pairs_refresh_all_sql() -> str:
    """Return the set-based archive-wide action-pair population statement."""
    return (
        action_pairs_refresh_sql("u.session_id")
        .replace("WHERE u.session_id = u.session_id AND", "WHERE")
        .replace("WHERE r.session_id = u.session_id AND", "WHERE")
    )


def rebuild_all_action_pairs_sync(conn: sqlite3.Connection) -> None:
    """Repopulate action pairs once for a bulk generation build."""
    conn.execute("DELETE FROM action_pairs")
    conn.execute(action_pairs_refresh_all_sql())


__all__ = [
    "action_pairs_refresh_all_sql",
    "action_pairs_refresh_sql",
    "rebuild_all_action_pairs_sync",
    "refresh_action_pairs",
]

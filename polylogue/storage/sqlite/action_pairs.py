"""Session-scoped materialization of deterministic tool action pairs."""

from __future__ import annotations

import sqlite3


def action_pairs_refresh_sql(session_expr: str) -> str:
    """Return the bounded insert used by writers and fixture-maintaining triggers."""
    return f"""
        INSERT INTO action_pairs (
            tool_use_block_id, session_id, message_id, tool_id, use_rank,
            tool_name, semantic_type, tool_command, tool_path, tool_input,
            tool_result_block_id, output_text, is_error, exit_code
        )
        WITH ranked_uses AS (
            SELECT u.session_id, u.message_id, u.block_id AS tool_use_block_id,
                   u.tool_name, u.semantic_type, u.tool_command, u.tool_path,
                   u.tool_input, u.tool_id,
                   ROW_NUMBER() OVER (
                       PARTITION BY u.session_id, u.tool_id
                       ORDER BY um.position, um.variant_index, u.position
                   ) AS use_rank
            FROM blocks u JOIN messages um ON um.message_id = u.message_id
            WHERE u.session_id = {session_expr} AND u.block_type = 'tool_use'
              AND u.tool_id IS NOT NULL AND u.tool_id != ''
        ), ranked_results AS (
            SELECT r.session_id, r.tool_id, r.block_id AS tool_result_block_id,
                   r.text AS output_text, r.tool_result_is_error AS is_error,
                   r.tool_result_exit_code AS exit_code,
                   ROW_NUMBER() OVER (
                       PARTITION BY r.session_id, r.tool_id
                       ORDER BY rm.position, rm.variant_index, r.position
                   ) AS result_rank
            FROM blocks r JOIN messages rm ON rm.message_id = r.message_id
            WHERE r.session_id = {session_expr} AND r.block_type = 'tool_result'
              AND r.tool_id IS NOT NULL AND r.tool_id != ''
        )
        SELECT u.tool_use_block_id, u.session_id, u.message_id, u.tool_id, u.use_rank,
               u.tool_name, u.semantic_type, u.tool_command, u.tool_path, u.tool_input,
               r.tool_result_block_id, r.output_text, r.is_error, r.exit_code
        FROM ranked_uses u LEFT JOIN ranked_results r
          ON r.session_id = u.session_id AND r.tool_id = u.tool_id AND r.result_rank = u.use_rank
        UNION ALL
        SELECT u.block_id, u.session_id, u.message_id, u.tool_id, NULL,
               u.tool_name, u.semantic_type, u.tool_command, u.tool_path, u.tool_input,
               NULL, NULL, NULL, NULL
        FROM blocks u
        WHERE u.session_id = {session_expr} AND u.block_type = 'tool_use'
          AND (u.tool_id IS NULL OR u.tool_id = '')
    """


def refresh_action_pairs(conn: sqlite3.Connection, session_id: str) -> None:
    """Rebuild action pairs for one changed session inside its write transaction."""
    conn.execute("DELETE FROM action_pairs WHERE session_id = ?", (session_id,))
    conn.execute(action_pairs_refresh_sql("?"), (session_id, session_id, session_id))


__all__ = ["action_pairs_refresh_sql", "refresh_action_pairs"]

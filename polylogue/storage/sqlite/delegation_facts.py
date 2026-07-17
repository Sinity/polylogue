"""Session-scoped materialization of delegation facts."""

from __future__ import annotations

import sqlite3

_FACT_COLUMNS = (
    "delegation_id, parent_session_id, child_session_id, mapping_state, link_confidence, link_method, inheritance, "
    "branch_point_message_id, instruction_message_id, instruction_tool_use_block_id, instruction_payload, "
    "dispatch_turn_model, requested_model, artifact_block_id, artifact_text, result_is_error, result_exit_code, "
    "result_status, parent_origin, parent_session_dominant_model, parent_session_dominant_model_family, "
    "parent_terminal_state, child_session_dominant_model, child_session_dominant_model_family, child_cost_usd, "
    "child_cost_is_estimated, child_tokens, child_wall_ms, child_terminal_state"
)


def delegation_facts_insert_sql(parent_expr: str) -> str:
    return f"""
        INSERT INTO delegation_facts ({_FACT_COLUMNS})
        SELECT
            COALESCE(instruction_tool_use_block_id, parent_session_id || ':' || child_session_id),
            parent_session_id, child_session_id, mapping_state, link_confidence, link_method, inheritance,
            branch_point_message_id, instruction_message_id, instruction_tool_use_block_id, instruction_payload,
            dispatch_turn_model, requested_model, artifact_block_id, artifact_text, result_is_error, result_exit_code,
            result_status, parent_origin, parent_session_dominant_model, parent_session_dominant_model_family,
            parent_terminal_state, child_session_dominant_model, child_session_dominant_model_family, child_cost_usd,
            child_cost_is_estimated, child_tokens, child_wall_ms, child_terminal_state
        FROM delegation_facts_source
    """


def refresh_delegation_facts(conn: sqlite3.Connection, parent_session_id: str) -> None:
    """Refresh one parent cohort; the compatibility source is filtered by parent."""
    conn.execute("DELETE FROM delegation_facts WHERE parent_session_id = ?", (parent_session_id,))
    conn.execute("INSERT OR REPLACE INTO delegation_refresh_scope(parent_session_id) VALUES (?)", (parent_session_id,))
    try:
        conn.execute(delegation_facts_insert_sql("?"))
    finally:
        conn.execute("DELETE FROM delegation_refresh_scope")


def refresh_delegation_facts_for_session(conn: sqlite3.Connection, session_id: str) -> None:
    """Refresh the session and every parent cohort affected by its links."""
    parent_ids = {session_id}
    row = conn.execute("SELECT parent_session_id FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
    if row is not None and row[0] is not None:
        parent_ids.add(str(row[0]))
    parent_ids.update(
        str(row[0])
        for row in conn.execute(
            "SELECT resolved_dst_session_id FROM session_links WHERE src_session_id = ? AND resolved_dst_session_id IS NOT NULL",
            (session_id,),
        ).fetchall()
        if row[0] is not None
    )
    for parent_id in sorted(parent_ids):
        refresh_delegation_facts(conn, parent_id)


__all__ = ["delegation_facts_insert_sql", "refresh_delegation_facts", "refresh_delegation_facts_for_session"]

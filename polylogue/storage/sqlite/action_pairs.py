"""One causal-association owner for action reads, materialization and events."""

from __future__ import annotations

import sqlite3

from polylogue.core.tool_association import tool_association_ctes_sql


def action_pairing_ctes_sql(*, use_bound: str = "", result_bound: str = "", session_index_hint: str = "") -> str:
    """Project original stored facts into the shared association owner.

    Exact resolved parent chains can prove multiple replies to one invocation.
    Without that proof, only clean alternating reuse is associated; gaps,
    duplicate replies and conflicting variants leave the suffix unknown.
    Bounded scans retain the caller's original session-index hint.
    """
    association = tool_association_ctes_sql()
    return f"""
    association_messages AS (
        SELECT session_id AS session_key,message_id AS message_key,parent_message_id AS parent_key,
               role,position AS message_position,variant_index FROM messages
    ), association_blocks AS (
        SELECT u.session_id AS session_key,u.block_id AS block_key,u.message_id AS message_key,
               u.tool_id,1 AS is_use,u.tool_outcome AS outcome,
               u.tool_result_outcome_unknown_reason AS unknown_reason,u.position AS block_position
        FROM blocks u{session_index_hint} WHERE u.block_type='tool_use'{use_bound}
        UNION ALL
        SELECT r.session_id,r.block_id,r.message_id,r.tool_id,0,r.tool_outcome,
               r.tool_result_outcome_unknown_reason,r.position
        FROM blocks r{session_index_hint} WHERE r.block_type='tool_result'{result_bound}
    ), {association}, paired_uses AS (
        SELECT session_key AS session_id,tool_id,use_rank,use_key AS tool_use_block_id,
               result_key AS candidate_result_id,result_count,ambiguous,verdict,verdict_reason,association_state
        FROM tool_associations
    )
    """.strip()


def action_pairs_select_sql(*, use_bound: str = "", result_bound: str = "", session_index_hint: str = "") -> str:
    """Project the same association into the materialized action-pair shape."""
    ctes = action_pairing_ctes_sql(
        use_bound=use_bound, result_bound=result_bound, session_index_hint=session_index_hint
    )
    return f"""
        WITH RECURSIVE {ctes}
        SELECT u.block_id AS tool_use_block_id, u.session_id, u.message_id, u.tool_id, pair.use_rank,
               u.tool_name, u.semantic_type, u.tool_command, u.tool_path,
               r.block_id AS tool_result_block_id,
               r.tool_result_is_error AS is_error,
               r.tool_result_exit_code AS exit_code,
               pair.verdict AS tool_outcome,pair.verdict_reason AS outcome_unknown_reason
        FROM paired_uses pair
        JOIN blocks u ON u.block_id = pair.tool_use_block_id
        LEFT JOIN blocks r ON r.block_id = pair.candidate_result_id AND pair.ambiguous = 0
        UNION ALL
        SELECT u.block_id, u.session_id, u.message_id, u.tool_id, NULL,
               u.tool_name, u.semantic_type, u.tool_command, u.tool_path,
               NULL, NULL, NULL, 'no_result', NULL
        FROM blocks u{session_index_hint}
        WHERE u.block_type = 'tool_use' AND (u.tool_id IS NULL OR u.tool_id = ''){use_bound}
    """.strip()


def action_pairs_refresh_sql(session_expr: str | None, *, session_index_hint: str = "") -> str:
    """Return the insert used by writers, fixture triggers and bulk builds."""
    select = action_pairs_select_sql(
        use_bound=f" AND u.session_id = {session_expr}" if session_expr is not None else "",
        result_bound=f" AND r.session_id = {session_expr}" if session_expr is not None else "",
        session_index_hint=session_index_hint,
    )
    return f"""
        INSERT INTO action_pairs (
            tool_use_block_id, session_id, message_id, tool_id, use_rank,
            tool_name, semantic_type, tool_command, tool_path,
            tool_result_block_id, is_error, exit_code,
            tool_outcome, outcome_unknown_reason
        )
        {select}
    """


def refresh_action_pairs(conn: sqlite3.Connection, session_id: str, *, prior_rows: bool = True) -> None:
    """Rebuild action pairs for one changed session inside its write transaction.

    ``prior_rows`` is false when the session's messages are all new, so it
    can hold no pairs to delete first.
    """
    if prior_rows:
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
    return action_pairs_refresh_sql(None)


def rebuild_all_action_pairs_sync(conn: sqlite3.Connection) -> None:
    """Repopulate action pairs once for a bulk generation build."""
    conn.execute("DELETE FROM action_pairs")
    conn.execute(action_pairs_refresh_all_sql())


__all__ = [
    "action_pairing_ctes_sql",
    "action_pairs_select_sql",
    "action_pairs_refresh_all_sql",
    "action_pairs_refresh_sql",
    "rebuild_all_action_pairs_sync",
    "refresh_action_pairs",
]

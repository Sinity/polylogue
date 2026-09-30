"""One causal-association owner for action reads, materialization and events."""

from __future__ import annotations

import sqlite3


def action_pairing_ctes_sql(*, use_bound: str = "", result_bound: str = "", session_index_hint: str = "") -> str:
    """Return narrow CTEs ending in one ``paired_uses`` row per named use.

    Leading orphan receipts belong to no invocation. After that prefix, a
    clean alternating use/result stream supports sequential ID reuse. A gap
    or duplicate destroys that alignment: the affected use and remaining
    same-ID suffix stay unknown, not nearest-paired. A final outstanding use
    alone is simply resultless. Variant creation order is not causal order:
    cross-message reuse involving variants also stays unresolved.

    Windows carry identifiers and positions only, never transcript payloads.
    Both bounded physical scans retain the caller's session-index hint.
    """
    return f"""
    tool_events AS (
        SELECT u.session_id, u.tool_id, u.block_id, u.message_id,
               um.position AS message_position, um.variant_index,
               u.position AS block_position, 1 AS is_use
        FROM blocks u{session_index_hint}
        JOIN messages um ON um.message_id = u.message_id
        WHERE u.block_type = 'tool_use' AND u.tool_id IS NOT NULL AND u.tool_id != ''{use_bound}
        UNION ALL
        SELECT r.session_id, r.tool_id, r.block_id, r.message_id,
               rm.position, rm.variant_index, r.position, 0 AS is_use
        FROM blocks r{session_index_hint}
        JOIN messages rm ON rm.message_id = r.message_id
        WHERE r.block_type = 'tool_result' AND r.tool_id IS NOT NULL AND r.tool_id != ''{result_bound}
    ), numbered_events AS (
        SELECT *,
               SUM(is_use) OVER (
                   PARTITION BY session_id, tool_id
                   ORDER BY message_position, variant_index, block_position
                   ROWS UNBOUNDED PRECEDING
               ) AS use_rank,
               MAX(variant_index) OVER (PARTITION BY session_id, tool_id) AS max_variant
        FROM tool_events
    ), invocation_windows AS (
        SELECT session_id, tool_id, use_rank,
               MAX(CASE WHEN is_use = 1 THEN block_id END) AS tool_use_block_id,
               MAX(CASE WHEN is_use = 1 THEN message_id END) AS use_message_id,
               MAX(CASE WHEN is_use = 0 THEN message_id END) AS result_message_id,
               MAX(CASE WHEN is_use = 0 THEN block_id END) AS candidate_result_id,
               SUM(1 - is_use) AS result_count,
               MAX(max_variant) AS max_variant
        FROM numbered_events
        WHERE use_rank > 0
        GROUP BY session_id, tool_id, use_rank
    ), classified_windows AS (
        SELECT *,
               CASE
                   WHEN result_count > 1 THEN 1
                   WHEN result_count = 0 AND
                        LEAD(use_rank) OVER (PARTITION BY session_id, tool_id ORDER BY use_rank) IS NOT NULL THEN 1
                   WHEN result_count = 1 AND max_variant > 0 AND use_message_id != result_message_id THEN 1
                   ELSE 0
               END AS ambiguous
        FROM invocation_windows
    ), paired_uses AS (
        SELECT session_id, tool_id, use_rank, tool_use_block_id, candidate_result_id,
               MAX(ambiguous) OVER (
                   PARTITION BY session_id, tool_id ORDER BY use_rank ROWS UNBOUNDED PRECEDING
               ) AS ambiguous
        FROM classified_windows
    )
    """.strip()


def action_pairs_select_sql(*, use_bound: str = "", result_bound: str = "", session_index_hint: str = "") -> str:
    """Project the same association into the materialized action-pair shape."""
    ctes = action_pairing_ctes_sql(
        use_bound=use_bound, result_bound=result_bound, session_index_hint=session_index_hint
    )
    return f"""
        WITH {ctes}
        SELECT u.block_id AS tool_use_block_id, u.session_id, u.message_id, u.tool_id, pair.use_rank,
               u.tool_name, u.semantic_type, u.tool_command, u.tool_path,
               r.block_id AS tool_result_block_id,
               r.tool_result_is_error AS is_error, r.tool_result_exit_code AS exit_code,
               CASE WHEN pair.ambiguous = 1 THEN 'unknown'
                    WHEN r.block_id IS NULL THEN 'no_result'
                    ELSE COALESCE(r.tool_outcome, 'unknown') END AS tool_outcome,
               CASE WHEN pair.ambiguous = 1 THEN 'ambiguous_tool_id_reuse'
                    ELSE r.tool_result_outcome_unknown_reason END AS outcome_unknown_reason
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

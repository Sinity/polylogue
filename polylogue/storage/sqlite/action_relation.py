"""Canonical tool-action reads through the shared causal-association owner."""

from __future__ import annotations

from polylogue.storage.sqlite.action_pairs import action_pairs_select_sql


def action_relation_select_sql(
    *,
    session_placeholders: str | None = None,
    empty: bool = False,
) -> str:
    """Return actions with all three physical branches bounded before pairing.

    Fresh archives have no ANALYZE statistics; pin the known session index
    rather than letting SQLite choose an archive-wide block-type scan.
    """
    if empty and session_placeholders is not None:
        raise ValueError("An empty action relation cannot also declare session placeholders")
    select = action_pairs_select_sql(
        use_bound=" AND 0"
        if empty
        else f" AND u.session_id IN ({session_placeholders})"
        if session_placeholders
        else "",
        result_bound=" AND 0"
        if empty
        else f" AND r.session_id IN ({session_placeholders})"
        if session_placeholders
        else "",
        session_index_hint=" INDEXED BY idx_blocks_session_position" if session_placeholders else "",
    )
    return f"""
        WITH paired_actions AS ({select})
        SELECT ap.session_id, ap.message_id, ap.tool_use_block_id,
               ap.tool_name, ap.semantic_type, ap.tool_command, ap.tool_path,
               u.tool_input, r.text AS output_text, ap.is_error, ap.exit_code,
               ap.tool_result_block_id, ap.tool_outcome, ap.outcome_unknown_reason,
               CASE ap.tool_outcome
                   WHEN 'no_result' THEN 'no_result'
                   WHEN 'unknown' THEN 'outcome_unknown'
                   WHEN 'error' THEN 'outcome_error'
                   WHEN 'ok' THEN 'outcome_success'
               END AS result_state
        FROM paired_actions ap
        JOIN blocks u ON u.block_id = ap.tool_use_block_id
        LEFT JOIN blocks r ON r.block_id = ap.tool_result_block_id
    """.strip()


def bounded_action_relation_cte(*, relation_name: str, session_count: int) -> str:
    """Return one named CTE whose three branches share the same session set."""
    if session_count < 0:
        raise ValueError("A bounded action relation cannot have a negative session count")
    placeholders = ", ".join("?" for _ in range(session_count)) or None
    select_sql = action_relation_select_sql(
        session_placeholders=placeholders,
        empty=session_count == 0,
    )
    return f"{relation_name} AS ({select_sql})"


__all__ = ["action_relation_select_sql", "bounded_action_relation_cte"]

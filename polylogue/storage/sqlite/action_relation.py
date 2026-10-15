"""Canonical tool-action reads through the shared causal-association owner."""

from __future__ import annotations

from polylogue.storage.sqlite.action_pairs import action_pairing_ctes_sql


def action_relation_select_sql(
    *,
    session_placeholders: str | None = None,
    empty: bool = False,
    context_relation: str | None = None,
) -> str:
    """Project every owned result without retaining a second payload table."""
    if empty and session_placeholders is not None:
        raise ValueError("An empty action relation cannot also declare session placeholders")
    bound = (
        " AND 0" if empty else f" AND {{alias}}.session_id IN ({session_placeholders})" if session_placeholders else ""
    )
    ctes = action_pairing_ctes_sql(
        use_bound=bound.format(alias="u"),
        result_bound=bound.format(alias="r"),
        session_index_hint=" INDEXED BY idx_blocks_session_position" if session_placeholders else "",
        context_relation=context_relation,
    )
    use_join = (
        "JOIN blocks u INDEXED BY idx_blocks_session_position "
        "ON u.session_id=m.source_session_id AND u.message_id=m.message_key"
        if context_relation is not None
        else "JOIN blocks u ON u.message_id=m.message_key"
    )
    return f"""
        WITH RECURSIVE {ctes}, result_rows AS MATERIALIZED (
            SELECT ar.session_key,ar.assigned_use_key,ar.message_position,ar.variant_index,ar.block_position,
                   b.block_id,b.text,b.tool_result_is_error AS is_error,b.tool_result_exit_code AS exit_code,
                   b.tool_outcome,b.tool_result_outcome_unknown_reason AS outcome_unknown_reason
            FROM association_results ar JOIN blocks b ON b.block_id=ar.block_key
            JOIN paired_uses p ON p.session_id=ar.session_key AND p.tool_use_block_id=ar.assigned_use_key
            WHERE p.ambiguous=0
        ), selected_uses AS (
            SELECT m.session_key,u.block_id,u.message_id,u.tool_name,u.semantic_type,u.tool_command,u.tool_path,
                   u.tool_input,p.verdict,p.verdict_reason
            FROM association_messages m {use_join}
            LEFT JOIN paired_uses p ON p.session_id=m.session_key AND p.tool_use_block_id=u.block_id
            WHERE u.block_type='tool_use'{bound.format(alias="u") if context_relation is None else ""}
        )
        SELECT u.session_key AS session_id,u.message_id,u.block_id AS tool_use_block_id,
               u.tool_name,u.semantic_type,u.tool_command,u.tool_path,u.tool_input,
               (SELECT json_group_array(json_object(
                    'tool_result_block_id',r.block_id,'output_text',r.text,'is_error',r.is_error,
                    'exit_code',r.exit_code,'tool_outcome',r.tool_outcome,
                    'outcome_unknown_reason',r.outcome_unknown_reason))
                FROM (SELECT * FROM result_rows r WHERE r.session_key=u.session_key AND r.assigned_use_key=u.block_id
                      ORDER BY r.message_position,r.variant_index,r.block_position,r.block_id) r) AS results_json,
               (SELECT group_concat(r.text,char(10))
                FROM (SELECT * FROM result_rows r WHERE r.session_key=u.session_key AND r.assigned_use_key=u.block_id
                      ORDER BY r.message_position,r.variant_index,r.block_position,r.block_id) r) AS output_text,
               COALESCE(u.verdict,'no_result') AS tool_outcome,u.verdict_reason AS outcome_unknown_reason,
               CASE COALESCE(u.verdict,'no_result') WHEN 'error' THEN 1 WHEN 'ok' THEN 0 ELSE NULL END AS is_error,
               CASE COALESCE(u.verdict,'no_result')
                   WHEN 'no_result' THEN 'no_result' WHEN 'unknown' THEN 'outcome_unknown'
                   WHEN 'error' THEN 'outcome_error' WHEN 'ok' THEN 'outcome_success'
               END AS result_state
        FROM selected_uses u
    """.strip()


def bounded_action_relation_cte(*, relation_name: str, session_count: int) -> str:
    """Return one named CTE whose branches share the same session set."""
    if session_count < 0:
        raise ValueError("A bounded action relation cannot have a negative session count")
    placeholders = ", ".join("?" for _ in range(session_count)) or None
    return f"{relation_name} AS ({action_relation_select_sql(session_placeholders=placeholders, empty=session_count == 0)})"


__all__ = ["action_relation_select_sql", "bounded_action_relation_cte"]

"""Canonical deterministic tool-use/result pairing SQL."""

from __future__ import annotations


def action_relation_select_sql(
    *,
    session_placeholders: str | None = None,
    empty: bool = False,
) -> str:
    """Return the actions SELECT, optionally bounding every physical branch."""
    if empty and session_placeholders is not None:
        raise ValueError("An empty action relation cannot also declare session placeholders")
    use_bound = " AND 0" if empty else f" AND u.session_id IN ({session_placeholders})" if session_placeholders else ""
    result_bound = (
        " AND 0" if empty else f" AND r.session_id IN ({session_placeholders})" if session_placeholders else ""
    )
    null_id_bound = (
        " AND 0" if empty else f" AND u.session_id IN ({session_placeholders})" if session_placeholders else ""
    )
    return f"""
WITH ranked_uses AS (
    SELECT
        u.session_id,
        u.message_id,
        u.block_id AS tool_use_block_id,
        u.tool_name,
        u.semantic_type,
        u.tool_command,
        u.tool_path,
        u.tool_input,
        u.tool_id,
        ROW_NUMBER() OVER (
            PARTITION BY u.session_id, u.tool_id
            ORDER BY um.position, um.variant_index, u.position
        ) AS use_rank
    FROM blocks u
    JOIN messages um ON um.message_id = u.message_id
    WHERE u.block_type = 'tool_use' AND u.tool_id IS NOT NULL AND u.tool_id != ''{use_bound}
),
ranked_results AS (
    SELECT
        r.session_id,
        r.tool_id,
        r.block_id AS tool_result_block_id,
        r.text AS output_text,
        r.tool_result_is_error AS is_error,
        r.tool_result_exit_code AS exit_code,
        ROW_NUMBER() OVER (
            PARTITION BY r.session_id, r.tool_id
            ORDER BY rm.position, rm.variant_index, r.position
        ) AS result_rank
    FROM blocks r
    JOIN messages rm ON rm.message_id = r.message_id
    WHERE r.block_type = 'tool_result' AND r.tool_id IS NOT NULL AND r.tool_id != ''{result_bound}
)
SELECT
    ranked_uses.session_id,
    ranked_uses.message_id,
    ranked_uses.tool_use_block_id,
    ranked_uses.tool_name,
    ranked_uses.semantic_type,
    ranked_uses.tool_command,
    ranked_uses.tool_path,
    ranked_uses.tool_input,
    ranked_results.output_text,
    ranked_results.is_error,
    ranked_results.exit_code,
    ranked_results.tool_result_block_id
FROM ranked_uses
LEFT JOIN ranked_results
    ON ranked_results.session_id = ranked_uses.session_id
   AND ranked_results.tool_id = ranked_uses.tool_id
   AND ranked_results.result_rank = ranked_uses.use_rank

UNION ALL

SELECT
    u.session_id,
    u.message_id,
    u.block_id AS tool_use_block_id,
    u.tool_name,
    u.semantic_type,
    u.tool_command,
    u.tool_path,
    u.tool_input,
    NULL AS output_text,
    NULL AS is_error,
    NULL AS exit_code,
    NULL AS tool_result_block_id
FROM blocks u
WHERE u.block_type = 'tool_use' AND (u.tool_id IS NULL OR u.tool_id = ''){null_id_bound}
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

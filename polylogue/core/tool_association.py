"""Resolve invocations from original occurrence and resolved parent facts."""

from __future__ import annotations


def tool_association_ctes_sql() -> str:
    """Lower caller-owned association_messages/association_blocks relations.

    Both relations carry session_key. Message keys and resolved parent keys
    identify globally unique exact occurrences in the caller relation, not reusable provider IDs. Parent traversal
    crosses tool replies only; UNION terminates cycles without a depth cap.
    The caller keeps the original SQL creator and all physical result rows.
    """
    return """
    association_events AS (
        SELECT b.session_key,b.block_key,b.message_key,b.tool_id,b.is_use,b.outcome,b.unknown_reason,
               m.message_position,m.variant_index,m.parent_key,b.block_position
        FROM association_blocks b JOIN association_messages m
          ON m.session_key=b.session_key AND m.message_key=b.message_key
        WHERE b.tool_id IS NOT NULL AND b.tool_id!=''
    ), association_ancestors(session_key,result_key,tool_id,parent_key) AS (
        SELECT b.session_key,b.block_key,b.tool_id,m.parent_key
        FROM association_blocks b JOIN association_messages m
          ON m.session_key=b.session_key AND m.message_key=b.message_key
        WHERE b.is_use=0
        UNION
        SELECT a.session_key,a.result_key,a.tool_id,m.parent_key
        FROM association_ancestors a JOIN association_messages m
          ON m.message_key=a.parent_key
        WHERE m.role='tool' AND m.parent_key IS NOT NULL
    ), association_owner_candidates AS (
        SELECT a.session_key,a.result_key,COUNT(DISTINCT b.block_key) AS owner_count,
               MIN(b.block_key) AS owner_key
        FROM association_ancestors a JOIN association_blocks b
          ON b.session_key=a.session_key AND b.message_key=a.parent_key
        WHERE b.is_use=1 AND b.tool_id=a.tool_id
        GROUP BY a.session_key,a.result_key
    ), association_numbered AS (
        SELECT e.*,o.owner_key,o.owner_count,
               SUM(is_use) OVER (
                   PARTITION BY e.session_key,tool_id
                   ORDER BY message_position,variant_index,block_position,block_key
                   ROWS UNBOUNDED PRECEDING
               ) AS use_rank,
               MAX(variant_index) OVER (PARTITION BY e.session_key,tool_id) AS max_variant
        FROM association_events e LEFT JOIN association_owner_candidates o
          ON o.session_key=e.session_key AND o.result_key=e.block_key
    ), association_uses AS (
        SELECT * FROM association_numbered WHERE is_use=1
    ), association_results AS (
        SELECT r.*,u.block_key AS assigned_use_key,
               CASE WHEN r.owner_count=1 THEN 1 ELSE 0 END AS parent_proved
        FROM association_numbered r JOIN association_uses u
          ON u.session_key=r.session_key AND u.tool_id=r.tool_id
         AND ((r.owner_count=1 AND u.block_key=r.owner_key)
              OR (COALESCE(r.owner_count,0)!=1 AND u.use_rank=r.use_rank))
        WHERE r.is_use=0
    ), association_siblings AS (
        SELECT session_key,assigned_use_key,parent_key,
               COUNT(DISTINCT variant_index)>1 AS alternatives
        FROM association_results WHERE parent_key IS NOT NULL
        GROUP BY session_key,assigned_use_key,parent_key
    ), association_windows AS (
        SELECT u.session_key,u.tool_id,u.use_rank,u.block_key AS use_key,
               u.message_key AS use_message_key,u.max_variant,
               COUNT(r.block_key) AS result_count,
               MAX(r.message_key) AS result_message_key,
               CASE WHEN COUNT(r.block_key)=1 THEN MAX(r.block_key) END AS result_key,
               COALESCE(SUM(r.parent_proved),0) AS proved_count,
               COALESCE(SUM(sibling.alternatives),0) AS alternative_count,
               COALESCE(SUM(CASE WHEN r.owner_count>1 THEN 1 ELSE 0 END),0) AS conflicting_count,
               COALESCE(SUM(CASE WHEN r.outcome='error' THEN 1 ELSE 0 END),0) AS errors,
               COALESCE(SUM(CASE WHEN r.block_key IS NOT NULL
                    AND COALESCE(r.outcome,'unknown')='unknown' THEN 1 ELSE 0 END),0) AS unknowns,
               MIN(CASE WHEN COALESCE(r.outcome,'unknown')='unknown' THEN r.unknown_reason END) AS first_reason,
               MAX(CASE WHEN COALESCE(r.outcome,'unknown')='unknown' THEN r.unknown_reason END) AS last_reason,
               COALESCE(SUM(CASE WHEN r.block_key IS NOT NULL AND COALESCE(r.outcome,'unknown')='unknown'
                    AND r.unknown_reason IS NULL THEN 1 ELSE 0 END),0) AS missing_reasons
        FROM association_uses u LEFT JOIN association_results r
          ON r.session_key=u.session_key AND r.assigned_use_key=u.block_key
        LEFT JOIN association_siblings sibling
          ON sibling.session_key=r.session_key AND sibling.assigned_use_key=r.assigned_use_key
         AND sibling.parent_key=r.parent_key
        GROUP BY u.session_key,u.tool_id,u.use_rank,u.block_key,u.message_key,u.max_variant
    ), association_classified AS (
        SELECT *,CASE
          WHEN conflicting_count>0 OR alternative_count>0 THEN 1
          WHEN result_count>1 AND proved_count!=result_count THEN 1
          WHEN result_count=0 AND LEAD(use_rank) OVER (
              PARTITION BY session_key,tool_id ORDER BY use_rank) IS NOT NULL THEN 1
          WHEN result_count=1 AND proved_count=0 AND max_variant>0
               AND use_message_key!=result_message_key THEN 1
          ELSE 0 END AS unresolved
        FROM association_windows
    ), association_resolved AS (
        SELECT *,CASE WHEN result_count>0 AND proved_count=result_count
          AND alternative_count=0 AND conflicting_count=0 THEN 0
          ELSE MAX(unresolved) OVER (
            PARTITION BY session_key,tool_id ORDER BY use_rank ROWS UNBOUNDED PRECEDING
          ) END AS ambiguous FROM association_classified
    ), tool_associations AS (
        SELECT *,CASE WHEN ambiguous=1 THEN 'unknown'
                      WHEN result_count=0 THEN 'no_result'
                      WHEN errors>0 THEN 'error'
                      WHEN unknowns>0 THEN 'unknown' ELSE 'ok' END AS verdict,
          CASE WHEN ambiguous=1 THEN 'ambiguous_tool_id_reuse'
               WHEN errors=0 AND unknowns>0 AND missing_reasons=0 AND first_reason=last_reason
               THEN first_reason ELSE NULL END AS verdict_reason,
          CASE WHEN ambiguous=1 THEN 'ambiguous'
               WHEN result_count=0 THEN 'no_result'
               WHEN result_count>1 THEN 'parent_fanout' ELSE 'paired' END AS association_state
        FROM association_resolved
    )
    """.strip()

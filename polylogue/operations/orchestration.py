"""Read retained usage observations for the orchestration evidence operation."""

from __future__ import annotations

import sqlite3


def read_orchestration_usage(conn: sqlite3.Connection, session_id: str) -> list[dict[str, object]]:
    """Read native counter columns; missing wire fields are not reconstructed."""
    rows = conn.execute(
        """
        SELECT session_id, position, source_message_id, provider_event_type,
               model_name, occurred_at_ms,
               last_input_tokens, last_output_tokens, last_cached_input_tokens,
               last_cache_write_tokens, last_reasoning_output_tokens, last_total_tokens,
               total_input_tokens, total_output_tokens, total_cached_input_tokens,
               total_cache_write_tokens, total_reasoning_output_tokens, total_tokens
        FROM session_provider_usage_events
        WHERE session_id = ?
        ORDER BY position
        """,
        (session_id,),
    ).fetchall()
    return [dict(row) for row in rows]

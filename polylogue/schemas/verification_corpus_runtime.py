"""Runtime helpers for schema verification."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone


def apply_quarantine_updates(
    conn: sqlite3.Connection,
    *,
    updates: list[tuple[str, str, str, str | None]],
) -> None:
    validated_at = datetime.now(tz=timezone.utc).isoformat()
    for raw_id, reason, provider, payload_provider in updates:
        conn.execute(
            """
            UPDATE raw_conversations
            SET validation_status = 'failed',
                validation_error = ?,
                validation_drift_count = 0,
                validation_provider = ?,
                validation_mode = 'strict',
                validated_at = ?,
                payload_provider = COALESCE(?, payload_provider)
            WHERE raw_id = ?
            """,
            (reason, provider, validated_at, payload_provider, raw_id),
        )
        conn.execute(
            """
            UPDATE raw_conversations
            SET parse_error = COALESCE(parse_error, ?)
            WHERE raw_id = ? AND parsed_at IS NULL
            """,
            (reason, raw_id),
        )
    conn.commit()


__all__ = ["apply_quarantine_updates"]

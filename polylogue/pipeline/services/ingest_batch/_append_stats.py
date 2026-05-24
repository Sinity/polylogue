"""Append-only conversation stats helpers."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Sequence

from polylogue.pipeline.services.ingest_worker import MessageTuple

_WRITE_SELECT_CHUNK_SIZE = 900
MessageSignature = tuple[str, int, int, int, int]
FullStatsRecount = Callable[[sqlite3.Connection, str, str], None]


def existing_message_signatures(conn: sqlite3.Connection, message_ids: Sequence[str]) -> dict[str, MessageSignature]:
    if not message_ids:
        return {}
    signatures: dict[str, MessageSignature] = {}
    for offset in range(0, len(message_ids), _WRITE_SELECT_CHUNK_SIZE):
        chunk = message_ids[offset : offset + _WRITE_SELECT_CHUNK_SIZE]
        placeholders = ", ".join("?" for _ in chunk)
        rows = conn.execute(
            f"""
            SELECT
                message_id,
                content_hash,
                word_count,
                has_tool_use,
                has_thinking,
                has_paste
            FROM messages
            WHERE message_id IN ({placeholders})
            """,
            tuple(chunk),
        ).fetchall()
        signatures.update(
            {
                str(row["message_id"]): (
                    str(row["content_hash"]),
                    int(row["word_count"] or 0),
                    int(row["has_tool_use"] or 0),
                    int(row["has_thinking"] or 0),
                    int(row["has_paste"] or 0),
                )
                for row in rows
            }
        )
    return signatures


def upsert_stats_for_append(
    conn: sqlite3.Connection,
    conversation_id: str,
    provider_name: str,
    changed_messages: Sequence[MessageTuple],
    existing_messages: dict[str, MessageSignature],
    *,
    full_recount: FullStatsRecount,
) -> None:
    current = conn.execute(
        "SELECT 1 FROM conversation_stats WHERE conversation_id = ?",
        (conversation_id,),
    ).fetchone()
    if current is None:
        full_recount(conn, conversation_id, provider_name)
        return
    if not changed_messages:
        return

    message_delta = 0
    word_delta = 0
    tool_delta = 0
    thinking_delta = 0
    paste_delta = 0
    for message in changed_messages:
        old = existing_messages.get(str(message[0]))
        message_delta += 1 if old is None else 0
        word_delta += int(message[11]) - (old[1] if old is not None else 0)
        tool_delta += int(message[12]) - (old[2] if old is not None else 0)
        thinking_delta += int(message[13]) - (old[3] if old is not None else 0)
        paste_delta += int(message[14]) - (old[4] if old is not None else 0)

    conn.execute(
        """
        UPDATE conversation_stats
        SET provider_name = ?,
            message_count = message_count + ?,
            word_count = word_count + ?,
            tool_use_count = tool_use_count + ?,
            thinking_count = thinking_count + ?,
            paste_count = paste_count + ?
        WHERE conversation_id = ?
        """,
        (
            provider_name,
            message_delta,
            word_delta,
            tool_delta,
            thinking_delta,
            paste_delta,
            conversation_id,
        ),
    )


__all__ = ["existing_message_signatures", "upsert_stats_for_append"]

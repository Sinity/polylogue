"""Append-only session stats helpers."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Sequence

from polylogue.pipeline.services.ingest_worker import MessageTuple

_WRITE_SELECT_CHUNK_SIZE = 900
MessageSignature = tuple[str, int, int, int, int, str]
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
                has_paste,
                role
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
                    str(row["role"] or ""),
                )
                for row in rows
            }
        )
    return signatures


def upsert_stats_for_append(
    conn: sqlite3.Connection,
    session_id: str,
    source_name: str,
    changed_messages: Sequence[MessageTuple],
    existing_messages: dict[str, MessageSignature],
    *,
    full_recount: FullStatsRecount,
) -> None:
    current = conn.execute(
        "SELECT 1 FROM session_stats WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    if current is None:
        full_recount(conn, session_id, source_name)
        return
    if not changed_messages:
        return

    message_delta = 0
    word_delta = 0
    tool_delta = 0
    thinking_delta = 0
    paste_delta = 0
    user_delta = 0
    assistant_delta = 0
    system_delta = 0
    tool_msg_delta = 0
    user_word_delta = 0
    assistant_word_delta = 0
    for message in changed_messages:
        old = existing_messages.get(str(message[0]))
        is_new = old is None
        old_role = old[5] if old is not None else ""
        new_role = str(message[3]) if message[3] else ""
        old_word = old[1] if old is not None else 0
        new_word = int(message[11])

        message_delta += 1 if is_new else 0
        word_delta += new_word - old_word
        tool_delta += int(message[12]) - (old[2] if old is not None else 0)
        thinking_delta += int(message[13]) - (old[3] if old is not None else 0)
        paste_delta += int(message[14]) - (old[4] if old is not None else 0)

        if not is_new and old_role != new_role:
            if old_role == "user":
                user_delta -= 1
                user_word_delta -= old_word
            elif old_role == "assistant":
                assistant_delta -= 1
                assistant_word_delta -= old_word
            elif old_role == "system":
                system_delta -= 1
            elif old_role == "tool":
                tool_msg_delta -= 1

        if new_role == "user":
            user_delta += 1
            user_word_delta += new_word
        elif new_role == "assistant":
            assistant_delta += 1
            assistant_word_delta += new_word
        elif new_role == "system":
            system_delta += 1
        elif new_role == "tool":
            tool_msg_delta += 1

    conn.execute(
        """
        UPDATE session_stats
        SET source_name = ?,
            message_count = message_count + ?,
            word_count = word_count + ?,
            tool_use_count = tool_use_count + ?,
            thinking_count = thinking_count + ?,
            paste_count = paste_count + ?,
            user_msg_count = user_msg_count + ?,
            assistant_msg_count = assistant_msg_count + ?,
            system_msg_count = system_msg_count + ?,
            tool_msg_count = tool_msg_count + ?,
            user_word_count = user_word_count + ?,
            assistant_word_count = assistant_word_count + ?
        WHERE session_id = ?
        """,
        (
            source_name,
            message_delta,
            word_delta,
            tool_delta,
            thinking_delta,
            paste_delta,
            user_delta,
            assistant_delta,
            system_delta,
            tool_msg_delta,
            user_word_delta,
            assistant_word_delta,
            session_id,
        ),
    )


__all__ = ["existing_message_signatures", "upsert_stats_for_append"]

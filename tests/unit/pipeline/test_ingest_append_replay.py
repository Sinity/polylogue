"""Append-only ingest replay regression tests."""

from __future__ import annotations

from pathlib import Path

from polylogue.pipeline.services.ingest_batch import _write_conversation
from polylogue.storage.sqlite.connection import open_connection
from polylogue.types import ConversationId
from tests.unit.pipeline.test_ingest_batch import (
    _block_tuple,
    _conversation_data,
    _message_tuple,
)


def test_append_mode_filters_unchanged_replayed_rows(tmp_path: Path) -> None:
    with open_connection(tmp_path / "ingest.db") as conn:
        initial = _conversation_data(
            "codex:append-replay",
            content_hash="hash-v1",
            message_tuples=[
                _message_tuple(
                    "msg-1",
                    "codex:append-replay",
                    role="user",
                    text="first",
                    content_hash="msg-v1-1",
                    sort_key=1.0,
                )
            ],
            block_tuples=[
                _block_tuple(
                    block_id="blk-msg-1-0",
                    message_id="msg-1",
                    conversation_id="codex:append-replay",
                    block_index=0,
                    text="first block",
                )
            ],
            stats_tuple=(ConversationId("codex:append-replay"), "codex", 1, 1, 0, 0, 0),
        )
        replay = _conversation_data(
            "codex:append-replay",
            content_hash="hash-replayed-full-file",
            message_tuples=[
                _message_tuple(
                    "msg-1",
                    "codex:append-replay",
                    role="user",
                    text="first should not be rewritten",
                    content_hash="msg-v1-1",
                    sort_key=1.0,
                ),
                _message_tuple(
                    "msg-2",
                    "codex:append-replay",
                    role="assistant",
                    text="second",
                    content_hash="msg-v2-2",
                    sort_key=2.0,
                ),
            ],
            block_tuples=[
                _block_tuple(
                    block_id="blk-msg-1-0",
                    message_id="msg-1",
                    conversation_id="codex:append-replay",
                    block_index=0,
                    text="rewritten block should be ignored",
                ),
                _block_tuple(
                    block_id="blk-msg-2-0",
                    message_id="msg-2",
                    conversation_id="codex:append-replay",
                    block_index=0,
                    text="second block",
                ),
            ],
            append_only=True,
        )

        changed_initial, _initial_counts = _write_conversation(conn, initial)
        changed_tail, tail_counts = _write_conversation(conn, replay)
        conn.commit()

        rows = conn.execute(
            "SELECT message_id, text FROM messages WHERE conversation_id = ? ORDER BY sort_key",
            ("codex:append-replay",),
        ).fetchall()
        block_rows = conn.execute(
            "SELECT message_id, text FROM content_blocks WHERE conversation_id = ? ORDER BY message_id",
            ("codex:append-replay",),
        ).fetchall()

    assert changed_initial is True
    assert changed_tail is True
    assert tail_counts["messages"] == 1
    assert tail_counts["skipped_messages"] == 1
    assert [(row["message_id"], row["text"]) for row in rows] == [
        ("msg-1", "first"),
        ("msg-2", "second"),
    ]
    assert [(row["message_id"], row["text"]) for row in block_rows] == [
        ("msg-1", "first block"),
        ("msg-2", "second block"),
    ]

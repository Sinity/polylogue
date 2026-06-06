"""Append-only ingest replay regression tests."""

from __future__ import annotations

from pathlib import Path

import pytest

import polylogue.pipeline.services.ingest_batch._core as ingest_batch_core
from polylogue.pipeline.services.ingest_batch import _write_session
from polylogue.storage.sqlite.connection import open_connection
from polylogue.types import SessionId
from tests.unit.pipeline.test_ingest_batch import (
    _block_tuple,
    _message_tuple,
    _session_data,
)


def test_append_mode_filters_unchanged_replayed_rows(tmp_path: Path) -> None:
    with open_connection(tmp_path / "ingest.db") as conn:
        initial = _session_data(
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
                    session_id="codex:append-replay",
                    block_index=0,
                    text="first block",
                )
            ],
            stats_tuple=(SessionId("codex:append-replay"), "codex", 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        )
        replay = _session_data(
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
                    session_id="codex:append-replay",
                    block_index=0,
                    text="rewritten block should be ignored",
                ),
                _block_tuple(
                    block_id="blk-msg-2-0",
                    message_id="msg-2",
                    session_id="codex:append-replay",
                    block_index=0,
                    text="second block",
                ),
            ],
            append_only=True,
        )

        changed_initial, _initial_counts = _write_session(conn, initial)
        changed_tail, tail_counts = _write_session(conn, replay)
        conn.commit()

        rows = conn.execute(
            "SELECT message_id, text FROM messages WHERE session_id = ? ORDER BY sort_key",
            ("codex:append-replay",),
        ).fetchall()
        block_rows = conn.execute(
            "SELECT message_id, text FROM content_blocks WHERE session_id = ? ORDER BY message_id",
            ("codex:append-replay",),
        ).fetchall()
        stats = conn.execute(
            """
            SELECT message_count, word_count, tool_use_count, thinking_count, paste_count
            FROM session_stats
            WHERE session_id = ?
            """,
            ("codex:append-replay",),
        ).fetchone()

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
    assert stats is not None
    assert tuple(stats) == (2, 2, 0, 0, 0)


def test_append_mode_updates_stats_without_full_message_recount(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with open_connection(tmp_path / "ingest.db") as conn:
        initial = _session_data(
            "codex:append-stats",
            content_hash="hash-v1",
            message_tuples=[
                _message_tuple(
                    "msg-1",
                    "codex:append-stats",
                    role="user",
                    text="first message",
                    content_hash="msg-v1-1",
                    sort_key=1.0,
                )
            ],
            stats_tuple=(SessionId("codex:append-stats"), "codex", 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        )
        changed_initial, _initial_counts = _write_session(conn, initial)
        conn.commit()

        def fail_full_recount(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("append stats should be updated incrementally")

        monkeypatch.setattr(ingest_batch_core, "_upsert_stats_from_messages", fail_full_recount)

        replay = _session_data(
            "codex:append-stats",
            content_hash="hash-v2",
            message_tuples=[
                _message_tuple(
                    "msg-1",
                    "codex:append-stats",
                    role="user",
                    text="rewritten first message",
                    content_hash="msg-v1-1",
                    sort_key=1.0,
                ),
                _message_tuple(
                    "msg-2",
                    "codex:append-stats",
                    role="assistant",
                    text="second message here",
                    content_hash="msg-v2-2",
                    sort_key=2.0,
                ),
            ],
            append_only=True,
        )

        changed_tail, tail_counts = _write_session(conn, replay)
        conn.commit()
        stats = conn.execute(
            """
            SELECT message_count, word_count, tool_use_count, thinking_count, paste_count
            FROM session_stats
            WHERE session_id = ?
            """,
            ("codex:append-stats",),
        ).fetchone()

    assert changed_initial is True
    assert changed_tail is True
    assert tail_counts["messages"] == 1
    assert tail_counts["skipped_messages"] == 1
    assert stats is not None
    assert tuple(stats) == (2, 5, 0, 0, 0)


def test_append_mode_repairs_missing_stats_on_unchanged_replay(tmp_path: Path) -> None:
    with open_connection(tmp_path / "ingest.db") as conn:
        initial = _session_data(
            "codex:append-stats-repair",
            content_hash="hash-v1",
            message_tuples=[
                _message_tuple(
                    "msg-1",
                    "codex:append-stats-repair",
                    role="user",
                    text="first message",
                    content_hash="msg-v1-1",
                    sort_key=1.0,
                )
            ],
            stats_tuple=(SessionId("codex:append-stats-repair"), "codex", 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        )
        changed_initial, _initial_counts = _write_session(conn, initial)
        conn.execute(
            "DELETE FROM session_stats WHERE session_id = ?",
            ("codex:append-stats-repair",),
        )
        conn.commit()

        replay = _session_data(
            "codex:append-stats-repair",
            content_hash="hash-v1",
            message_tuples=[
                _message_tuple(
                    "msg-1",
                    "codex:append-stats-repair",
                    role="user",
                    text="first message",
                    content_hash="msg-v1-1",
                    sort_key=1.0,
                )
            ],
            append_only=True,
        )

        changed_tail, tail_counts = _write_session(conn, replay)
        conn.commit()
        stats = conn.execute(
            """
            SELECT message_count, word_count, tool_use_count, thinking_count, paste_count
            FROM session_stats
            WHERE session_id = ?
            """,
            ("codex:append-stats-repair",),
        ).fetchone()

    assert changed_initial is True
    assert changed_tail is False
    assert tail_counts["messages"] == 0
    assert tail_counts["skipped_messages"] == 1
    assert stats is not None
    assert tuple(stats) == (1, 2, 0, 0, 0)

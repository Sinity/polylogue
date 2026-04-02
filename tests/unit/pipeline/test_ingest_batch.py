"""Focused tests for sync ingest-batch DB writes."""

from __future__ import annotations

from pathlib import Path

from polylogue.pipeline.services.ingest_batch import (
    _topo_sort_conversation_entries,
    _write_conversation,
)
from polylogue.pipeline.services.ingest_worker import ConversationData
from polylogue.storage.backends.connection import open_connection


def _conversation_data(
    conversation_id: str,
    *,
    content_hash: str,
    parent_conversation_id: str | None = None,
) -> ConversationData:
    conversation_tuple = (
        conversation_id,
        "codex",
        conversation_id.split(":", 1)[-1],
        "Conversation",
        "2026-04-02T00:00:00Z",
        "2026-04-02T00:00:00Z",
        0.0,
        content_hash,
        None,
        "{}",
        1,
        parent_conversation_id,
        None,
        None,
    )
    return ConversationData(
        conversation_id=conversation_id,
        content_hash=content_hash,
        provider_name="codex",
        conversation_tuple=conversation_tuple,
    )


def test_topo_sort_conversation_entries_orders_parent_before_child() -> None:
    parent = _conversation_data("codex:parent", content_hash="hash-parent")
    child = _conversation_data(
        "codex:child",
        content_hash="hash-child",
        parent_conversation_id="codex:parent",
    )

    ordered = _topo_sort_conversation_entries([
        ("raw-child", child),
        ("raw-parent", parent),
    ])

    assert [entry[1].conversation_id for entry in ordered] == [
        "codex:parent",
        "codex:child",
    ]


def test_write_conversation_clears_missing_parent_fk(tmp_path: Path) -> None:
    with open_connection(tmp_path / "ingest.db") as conn:
        child = _conversation_data(
            "codex:child",
            content_hash="hash-child",
            parent_conversation_id="codex:missing-parent",
        )

        _write_conversation(conn, child)
        conn.commit()

        row = conn.execute(
            "SELECT parent_conversation_id FROM conversations WHERE conversation_id = ?",
            ("codex:child",),
        ).fetchone()
        assert row is not None
        assert row["parent_conversation_id"] is None


def test_write_conversation_preserves_existing_parent_fk(tmp_path: Path) -> None:
    with open_connection(tmp_path / "ingest.db") as conn:
        parent = _conversation_data("codex:parent", content_hash="hash-parent")
        child = _conversation_data(
            "codex:child",
            content_hash="hash-child",
            parent_conversation_id="codex:parent",
        )

        _write_conversation(conn, parent)
        _write_conversation(conn, child)
        conn.commit()

        row = conn.execute(
            "SELECT parent_conversation_id FROM conversations WHERE conversation_id = ?",
            ("codex:child",),
        ).fetchone()
        assert row is not None
        assert row["parent_conversation_id"] == "codex:parent"

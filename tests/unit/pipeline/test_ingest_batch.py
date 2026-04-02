"""Focused tests for sync ingest-batch DB writes."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from polylogue.pipeline.services.ingest_batch import (
    _drain_ready_conversation_entries,
    _IngestBatchSummary,
    _topo_sort_conversation_entries,
    _write_conversation,
    refresh_session_products_bulk,
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


def test_drain_ready_conversation_entries_preserves_late_parent_fk(tmp_path: Path) -> None:
    with open_connection(tmp_path / "ingest.db") as conn:
        parent = _conversation_data("codex:parent", content_hash="hash-parent")
        child = _conversation_data(
            "codex:child",
            content_hash="hash-child",
            parent_conversation_id="codex:parent",
        )

        summary = _IngestBatchSummary()
        materialized_ids: set[str] = set()
        pending_by_parent: dict[str, list[tuple[str, ConversationData]]] = {}

        _drain_ready_conversation_entries(
            conn,
            [("raw-child", child)],
            summary=summary,
            materialized_ids=materialized_ids,
            pending_by_parent=pending_by_parent,
        )
        assert list(pending_by_parent) == ["codex:parent"]

        _drain_ready_conversation_entries(
            conn,
            [("raw-parent", parent)],
            summary=summary,
            materialized_ids=materialized_ids,
            pending_by_parent=pending_by_parent,
        )
        conn.commit()

        row = conn.execute(
            "SELECT parent_conversation_id FROM conversations WHERE conversation_id = ?",
            ("codex:child",),
        ).fetchone()
        assert row is not None
        assert row["parent_conversation_id"] == "codex:parent"


@pytest.mark.asyncio
async def test_refresh_session_products_bulk_dedupes_related_refreshes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_conn = SimpleNamespace(commit=AsyncMock())

    @asynccontextmanager
    async def _connection():
        yield fake_conn

    fake_backend = SimpleNamespace(connection=_connection)

    async def _fake_apply(conn, conversation_id: str, *, transaction_depth: int):
        del conn, transaction_depth
        if conversation_id == "conv-1":
            return SimpleNamespace(
                affected_groups={("chatgpt", "2026-04-02")},
                thread_root_id="root-a",
            )
        if conversation_id == "conv-2":
            return SimpleNamespace(
                affected_groups={("chatgpt", "2026-04-02")},
                thread_root_id="root-a",
            )
        return SimpleNamespace(
            affected_groups={("chatgpt", "2026-04-03")},
            thread_root_id="root-b",
        )

    refresh_thread_root = AsyncMock(return_value=1)
    refresh_aggregates = AsyncMock()

    monkeypatch.setattr(
        "polylogue.storage.session_product_refresh._apply_session_product_conversation_update_async",
        _fake_apply,
    )
    monkeypatch.setattr(
        "polylogue.storage.session_product_refresh._refresh_thread_root_async",
        refresh_thread_root,
    )
    monkeypatch.setattr(
        "polylogue.storage.session_product_refresh.refresh_async_provider_day_aggregates",
        refresh_aggregates,
    )

    await refresh_session_products_bulk(
        fake_backend,
        ["conv-1", "conv-2", "conv-3"],
    )

    assert refresh_thread_root.await_count == 2
    refreshed_roots = sorted(call.args[1] for call in refresh_thread_root.await_args_list)
    assert refreshed_roots == ["root-a", "root-b"]
    refresh_aggregates.assert_awaited_once()
    aggregate_args = refresh_aggregates.await_args
    assert aggregate_args.args[0] is fake_conn
    assert aggregate_args.args[1] == {
        ("chatgpt", "2026-04-02"),
        ("chatgpt", "2026-04-03"),
    }
    assert aggregate_args.kwargs["transaction_depth"] == 1
    fake_conn.commit.assert_awaited_once()

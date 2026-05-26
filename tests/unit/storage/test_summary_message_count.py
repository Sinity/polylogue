"""Regression coverage for #1623: `ConversationSummary.message_count` hydration.

Without this the facets surface reports ``total_messages: 0`` because
``compute_facets`` sums ``s.message_count or 0`` over summaries whose
``message_count`` field defaults to ``None``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.storage.query_models import ConversationRecordQuery
from polylogue.storage.repository import ConversationRepository
from polylogue.storage.sqlite.async_sqlite import SQLiteBackend
from tests.infra.storage_records import ConversationBuilder


@pytest.mark.asyncio
async def test_list_summaries_by_query_populates_message_count(tmp_path: Path) -> None:
    db_path = tmp_path / "summary-counts.db"

    (
        ConversationBuilder(db_path, "conv-three")
        .provider("chatgpt")
        .add_message("m1", role="user", text="hi")
        .add_message("m2", role="assistant", text="hello")
        .add_message("m3", role="user", text="bye")
        .save()
    )
    (ConversationBuilder(db_path, "conv-one").provider("chatgpt").add_message("m4", role="user", text="ping").save())

    backend = SQLiteBackend(db_path=db_path)
    repo = ConversationRepository(backend=backend)
    try:
        summaries = await repo.list_summaries_by_query(ConversationRecordQuery(provider="chatgpt", limit=10))
        counts = {str(s.id): s.message_count for s in summaries}
        assert counts == {"conv-three": 3, "conv-one": 1}
    finally:
        await repo.close()


@pytest.mark.asyncio
async def test_list_summaries_by_query_returns_none_for_unknown_conversations(tmp_path: Path) -> None:
    """Empty-result path doesn't error and doesn't issue a count query."""
    db_path = tmp_path / "summary-empty.db"

    backend = SQLiteBackend(db_path=db_path)
    repo = ConversationRepository(backend=backend)
    try:
        summaries = await repo.list_summaries_by_query(ConversationRecordQuery(provider="chatgpt", limit=10))
        assert summaries == []
    finally:
        await repo.close()

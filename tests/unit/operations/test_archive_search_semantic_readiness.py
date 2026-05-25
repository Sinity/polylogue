"""Semantic-readiness gates for archive search operations."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from polylogue.archive.query.spec import ConversationQuerySpec
from polylogue.errors import DatabaseError
from polylogue.operations.archive import ArchiveOperations


def _ops(repo: MagicMock, *, db_path: Path) -> ArchiveOperations:
    backend = MagicMock(db_path=db_path)
    config = MagicMock(db_path=db_path)
    return ArchiveOperations(config=config, repository=repo, backend=backend)


@pytest.mark.asyncio
async def test_semantic_count_rejects_unready_embeddings(tmp_path: Path) -> None:
    repo = MagicMock()
    repo.get_archive_stats = AsyncMock(
        return_value=SimpleNamespace(
            retrieval_ready=False,
            embedding_readiness_status="stale",
            embedded_messages=10,
            stale_embedding_messages=10,
        )
    )

    with pytest.raises(DatabaseError, match="retrieval-ready embeddings"):
        await _ops(repo, db_path=tmp_path / "archive.db").count_conversations(
            ConversationQuerySpec(similar_text="memory leak")
        )

    repo.get_archive_stats.assert_awaited_once()


@pytest.mark.asyncio
async def test_explicit_hybrid_rejects_missing_vector_provider(tmp_path: Path) -> None:
    repo = MagicMock()
    repo.get_archive_stats = AsyncMock(return_value=SimpleNamespace(retrieval_ready=True))

    with (
        patch("polylogue.storage.search_providers.create_vector_provider", return_value=None),
        pytest.raises(DatabaseError, match="vector search support"),
    ):
        await _ops(repo, db_path=tmp_path / "archive.db").count_conversations(
            ConversationQuerySpec(query_terms=("fts",), retrieval_lane="hybrid")
        )


@pytest.mark.asyncio
async def test_semantic_count_passes_ready_vector_provider(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = MagicMock()
    repo.get_archive_stats = AsyncMock(return_value=SimpleNamespace(retrieval_ready=True))
    provider = MagicMock(name="vector_provider")
    seen: dict[str, object] = {}

    async def fake_count(
        self: ConversationQuerySpec,
        repository: object,
        *,
        vector_provider: object | None = None,
    ) -> int:
        seen["repository"] = repository
        seen["vector_provider"] = vector_provider
        return 7

    monkeypatch.setattr(ConversationQuerySpec, "count", fake_count)
    with patch("polylogue.storage.search_providers.create_vector_provider", return_value=provider):
        total = await _ops(repo, db_path=tmp_path / "archive.db").count_conversations(
            ConversationQuerySpec(similar_text="memory leak")
        )

    assert total == 7
    assert seen == {"repository": repo, "vector_provider": provider}

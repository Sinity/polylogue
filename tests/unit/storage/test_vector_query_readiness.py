"""Readiness belongs to the provider's current-recipe query, not configuration."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.core.errors import EmbeddingRetrievalNotReadyError
from polylogue.storage.search_providers.sqlite_vec import SqliteVecProvider


def _snapshot(root: Path, state: str) -> sqlite3.Connection:
    from polylogue.storage.embeddings.identity import EmbeddingRecipe
    from polylogue.storage.search_providers.sqlite_vec_runtime import open_vector_read_snapshot
    from tests.infra.vector_archive import seed_vector_archive

    seed_vector_archive(
        root,
        [("seed", "m1", "Synthetic current recipe prose.", [1.0] + [0.0] * 1023)],
        model="voyage-3" if state == "stale" else "voyage-4",
    )
    if state == "empty":
        with sqlite3.connect(root / "embeddings.db") as connection:
            connection.execute("DELETE FROM message_embeddings_meta")
    return open_vector_read_snapshot(
        embeddings_path=root / "embeddings.db",
        index_path=root / "index.db",
        recipe=EmbeddingRecipe.current(model="voyage-4", dimensions=1024),
    )


@pytest.mark.parametrize("state", ["empty", "stale"], ids=["empty-store", "stale-recipe-only"])
def test_snapshot_query_refuses_no_current_vectors_before_embedding(
    tmp_path: Path,
    state: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F878: without the readiness probe the query embedding is purchased and the spy fails.

    An empty or wholly stale store must refuse typed ("empty"), not return a
    confident empty KNN page.
    """
    connection = _snapshot(tmp_path, state)
    provider = SqliteVecProvider.from_vector_read_snapshot(
        voyage_key="fixture", connection=connection, model="voyage-4"
    )

    def refuse_purchase(*args: object, **kwargs: object) -> list[list[float]]:
        raise AssertionError("no current vectors: query embedding must not be purchased")

    monkeypatch.setattr(provider, "_get_embeddings", refuse_purchase)
    try:
        with pytest.raises(EmbeddingRetrievalNotReadyError) as failure:
            provider.query("needle")
        assert failure.value.readiness_status == "empty"
        # The archive owns this handle, including its lifetime.
        assert tuple(connection.execute("SELECT 1").fetchone()) == (1,)
    finally:
        connection.close()


def test_snapshot_query_with_a_current_vector_proceeds_to_embedding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Anti-vacuity for the probe: one current vector admits the query route."""
    connection = _snapshot(tmp_path, "current")
    provider = SqliteVecProvider.from_vector_read_snapshot(
        voyage_key="fixture", connection=connection, model="voyage-4"
    )
    purchases: list[str] = []

    def record_purchase(texts: list[str], **_kwargs: object) -> list[list[float]]:
        purchases.extend(texts)
        return []

    monkeypatch.setattr(provider, "_get_embeddings", record_purchase)
    try:
        assert provider.query("needle") == []
        assert purchases == ["needle"]
    finally:
        connection.close()

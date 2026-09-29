"""Readiness belongs to the provider's current-recipe query, not configuration."""

from __future__ import annotations

import sqlite3

import pytest

from polylogue.core.errors import EmbeddingRetrievalNotReadyError
from polylogue.storage.search_providers.sqlite_vec import SqliteVecProvider

_CURRENT_HASH = bytes.fromhex("11" * 32)
_STALE_HASH = bytes.fromhex("22" * 32)


def _snapshot(stored_hashes: tuple[bytes, ...]) -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.executescript(
        """
        CREATE TABLE current_embedding_messages (vector_derivation_hash BLOB NOT NULL);
        CREATE TABLE message_embeddings_meta (vector_derivation_hash BLOB PRIMARY KEY);
        """
    )
    connection.execute("INSERT INTO current_embedding_messages VALUES (?)", (_CURRENT_HASH,))
    for stored in stored_hashes:
        connection.execute("INSERT INTO message_embeddings_meta VALUES (?)", (stored,))
    return connection


@pytest.mark.parametrize("stored_hashes", [(), (_STALE_HASH,)], ids=["empty-store", "stale-recipe-only"])
def test_snapshot_query_refuses_no_current_vectors_before_embedding(
    stored_hashes: tuple[bytes, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F878: without the readiness probe the query embedding is purchased and the spy fails.

    An empty or wholly stale store must refuse typed ("empty"), not return a
    confident empty KNN page.
    """
    connection = _snapshot(stored_hashes)
    provider = SqliteVecProvider.from_vector_read_snapshot(voyage_key="fixture", connection=connection)
    monkeypatch.setattr(provider, "_ensure_vec_available", lambda: None)
    monkeypatch.setattr(provider, "_ensure_tables", lambda: None)

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


def test_snapshot_query_with_a_current_vector_proceeds_to_embedding(monkeypatch: pytest.MonkeyPatch) -> None:
    """Anti-vacuity for the probe: one current vector admits the query route."""
    connection = _snapshot((_CURRENT_HASH,))
    provider = SqliteVecProvider.from_vector_read_snapshot(voyage_key="fixture", connection=connection)
    monkeypatch.setattr(provider, "_ensure_vec_available", lambda: None)
    monkeypatch.setattr(provider, "_ensure_tables", lambda: None)
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

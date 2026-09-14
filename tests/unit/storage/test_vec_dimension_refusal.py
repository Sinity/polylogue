"""A semantic read must never destroy the expensive-to-rebuild vector tier.

``_ensure_tables`` runs on every semantic query through the direct CLI/API
provider.  It used to reconcile a configured/stored dimension mismatch by
executing ``DROP TABLE IF EXISTS message_embeddings`` -- so editing
``embedding_dimension`` and then running one query physically destroyed every
stored vector, while ``message_embeddings_meta`` survived carrying
``CHECK(dimension = <old>)`` and made re-embedding fail outright.

These tests prove the read can no longer drop vectors by counting the stored
vectors after the read, rather than asserting the absence of a call.
"""

from __future__ import annotations

import sqlite3
import struct
from pathlib import Path
from typing import Protocol

import pytest

from polylogue.core.errors import SchemaSkewError
from polylogue.storage.search_providers.sqlite_vec import SqliteVecProvider


class EmbeddingFetcher(Protocol):
    def __call__(self, texts: list[str], input_type: str = "document") -> list[list[float]]: ...


from polylogue.storage.search_providers.sqlite_vec_runtime import drop_vec0_for_dimension_change
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.sqlite_vec_extension import try_load_sqlite_vec

_STORED_DIMENSION = 1024
_RECONFIGURED_DIMENSION = 512
_HASH = "a" * 64


def _embeddings_db_with_one_vector(tmp_path: Path) -> Path:
    db_path = tmp_path / "embeddings.db"
    initialize_archive_database(db_path, ArchiveTier.EMBEDDINGS)
    conn = sqlite3.connect(db_path)
    try:
        loaded, error = try_load_sqlite_vec(conn)
        if not loaded:
            pytest.skip(f"sqlite-vec unavailable: {error}")
        conn.execute(
            "INSERT INTO message_embeddings (vector_derivation_hash, embedding, model) VALUES (?, ?, ?)",
            (_HASH, struct.pack(f"<{_STORED_DIMENSION}f", *([0.01] * _STORED_DIMENSION)), "voyage-4"),
        )
        conn.commit()
    finally:
        conn.close()
    return db_path


def _stored_vector_count(db_path: Path) -> int:
    conn = sqlite3.connect(db_path)
    try:
        try_load_sqlite_vec(conn)
        return int(conn.execute("SELECT COUNT(*) FROM message_embeddings").fetchone()[0])
    finally:
        conn.close()


class _MutableSqliteVecProvider(SqliteVecProvider):
    """Assignable-attribute subclass so a test can set the configured dimension."""

    _get_embeddings: EmbeddingFetcher


def _provider(db_path: Path, dimension: int) -> _MutableSqliteVecProvider:
    provider = _MutableSqliteVecProvider(voyage_key="test-voyage-key", db_path=db_path, model="voyage-4")
    provider.dimension = dimension
    provider._get_embeddings = lambda texts, input_type="document": [[0.01] * dimension for _ in texts]
    return provider


def test_semantic_query_at_a_changed_dimension_refuses_and_keeps_every_vector(tmp_path: Path) -> None:
    """Anti-vacuity: restoring the ``DROP TABLE IF EXISTS message_embeddings``
    branch in ``_assert_vec0_dimension`` makes this red twice over -- the read
    stops raising, and the post-read vector count falls to zero."""

    db_path = _embeddings_db_with_one_vector(tmp_path)
    assert _stored_vector_count(db_path) == 1

    provider = _provider(db_path, _RECONFIGURED_DIMENSION)

    with pytest.raises(SchemaSkewError) as raised:
        provider.query("any semantic text")

    message = str(raised.value)
    assert str(_STORED_DIMENSION) in message
    assert str(_RECONFIGURED_DIMENSION) in message
    # The proof: the vectors physically survived the read.
    assert _stored_vector_count(db_path) == 1


def test_repeated_reads_never_erode_the_vector_tier(tmp_path: Path) -> None:
    """No once-per-process guard hides the destruction on later reads either.

    Anti-vacuity: the former code had no ``_tables_ensured`` short-circuit, so
    each read re-entered the drop; restoring it makes the count fall to zero.
    """

    db_path = _embeddings_db_with_one_vector(tmp_path)
    provider = _provider(db_path, _RECONFIGURED_DIMENSION)

    for _ in range(3):
        with pytest.raises(SchemaSkewError):
            provider.query("any semantic text")
    assert _stored_vector_count(db_path) == 1


def test_matching_dimension_still_serves_the_read(tmp_path: Path) -> None:
    """Anti-vacuity: refusing unconditionally makes this red."""

    db_path = _embeddings_db_with_one_vector(tmp_path)
    provider = _provider(db_path, _STORED_DIMENSION)

    assert provider.query("any semantic text") == []
    assert _stored_vector_count(db_path) == 1


def test_discarding_vectors_is_an_explicit_route_that_also_clears_stale_meta(tmp_path: Path) -> None:
    """Destruction remains available, but only as a deliberate operator action.

    Anti-vacuity: dropping the meta cleanup makes this red -- the archive would
    keep ``CHECK(dimension = 1024)`` metadata and re-embedding at the new
    dimension would fail outright.
    """

    db_path = _embeddings_db_with_one_vector(tmp_path)
    conn = sqlite3.connect(db_path)
    # The provider's helpers read columns by name, as the provider's own
    # connections do.
    conn.row_factory = sqlite3.Row
    try:
        try_load_sqlite_vec(conn)
        conn.execute(
            "INSERT INTO message_embeddings_meta "
            "(vector_derivation_hash, model, dimension, embedded_at_ms, recipe_hash, output_contract_hash) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (bytes.fromhex(_HASH), "voyage-4", _STORED_DIMENSION, 0, b"r" * 32, b"o" * 32),
        )
        conn.commit()
        assert drop_vec0_for_dimension_change(conn, _RECONFIGURED_DIMENSION) == _STORED_DIMENSION
        remaining_meta = conn.execute(
            "SELECT COUNT(*) FROM message_embeddings_meta WHERE dimension = ?", (_STORED_DIMENSION,)
        ).fetchone()[0]
        assert remaining_meta == 0
        assert (
            conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='message_embeddings'").fetchone()
            is None
        )
    finally:
        conn.close()

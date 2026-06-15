"""Storage-primitive contracts for ``VectorProvider.query_by_session`` (#1842).

``query_by_session`` powers ``near:id:<ref>`` session-seeded similarity: it reads
a stored session's own vectors and KNN-searches them against the store, excluding
the seed session's own messages. These tests pin the primitive directly against a
real archive-shaped ``embeddings.db`` so the exclusion and no-embedding contracts
fail here rather than silently at the query surface.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.core.enums import Origin
from polylogue.storage.search_providers.sqlite_vec import SqliteVecProvider
from polylogue.storage.search_providers.sqlite_vec_support import SqliteVecError
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.embedding_write import upsert_message_embedding
from polylogue.storage.sqlite.archive_tiers.embeddings import EMBEDDING_DIMENSION
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


def _unit_vector(*, axis0: float, axis1: float) -> list[float]:
    """Build a 1024-dim vector concentrated on the first two axes."""
    vec = [0.0] * EMBEDDING_DIMENSION
    vec[0] = axis0
    vec[1] = axis1
    return vec


@pytest.fixture
def embeddings_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "embeddings.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        initialize_archive_tier(conn, ArchiveTier.EMBEDDINGS)
    except sqlite3.OperationalError as exc:
        if "vec0" in str(exc) or "sqlite-vec" in str(exc):
            pytest.skip("sqlite-vec extension is unavailable")
        raise
    # Session A (seed): two messages on the x-axis.
    # Session B: very close to A (small angle) -> small L2 distance.
    # Session C: orthogonal to A -> large L2 distance.
    seeds = [
        ("sess-A", "sess-A:m1", _unit_vector(axis0=1.0, axis1=0.0)),
        ("sess-A", "sess-A:m2", _unit_vector(axis0=0.999, axis1=0.045)),
        ("sess-B", "sess-B:m1", _unit_vector(axis0=0.99, axis1=0.141)),
        ("sess-C", "sess-C:m1", _unit_vector(axis0=0.0, axis1=1.0)),
    ]
    for session_id, message_id, vector in seeds:
        upsert_message_embedding(
            conn,
            message_id=message_id,
            session_id=session_id,
            origin=Origin.CODEX_SESSION,
            embedding=vector,
            model="voyage-4",
            embedded_at_ms=1_767_225_700_000,
            content_hash=b"x" * 32,
        )
    conn.close()
    return db_path


def _provider(db_path: Path) -> SqliteVecProvider:
    provider = SqliteVecProvider(voyage_key="test-key", db_path=db_path, model="voyage-4")
    provider.dimension = EMBEDDING_DIMENSION
    provider._vec_available = None
    return provider


def test_query_by_session_excludes_seed_and_ranks_by_similarity(embeddings_db: Path) -> None:
    provider = _provider(embeddings_db)

    hits = provider.query_by_session("sess-A", limit=10)

    returned_ids = [message_id for message_id, _distance in hits]
    # The seed session's own messages never appear.
    assert all(not message_id.startswith("sess-A:") for message_id in returned_ids)
    # Both other sessions' messages surface, B (closer) ahead of C (orthogonal).
    assert "sess-B:m1" in returned_ids
    assert "sess-C:m1" in returned_ids
    assert returned_ids.index("sess-B:m1") < returned_ids.index("sess-C:m1")
    # Distances are non-decreasing (ascending similarity ranking).
    distances = [distance for _id, distance in hits]
    assert distances == sorted(distances)


def test_query_by_session_raises_typed_when_seed_has_no_embeddings(embeddings_db: Path) -> None:
    provider = _provider(embeddings_db)

    with pytest.raises(SqliteVecError):
        provider.query_by_session("sess-unembedded", limit=10)

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
from polylogue.storage.embeddings.identity import vector_derivation_hash
from polylogue.storage.search_providers.sqlite_vec import SqliteVecProvider
from polylogue.storage.search_providers.sqlite_vec_support import SqliteVecError
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.embeddings import EMBEDDING_DIMENSION
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from tests.infra.identity import fixture_block_content_identity


def _unit_vector(*, axis0: float, axis1: float) -> list[float]:
    """Build a 1024-dim vector concentrated on the first two axes."""
    vec = [0.0] * EMBEDDING_DIMENSION
    vec[0] = axis0
    vec[1] = axis1
    return vec


@pytest.fixture
def embeddings_db(tmp_path: Path) -> Path:
    from tests.infra.vector_archive import seed_vector_archive

    seed_vector_archive(
        tmp_path,
        [
            ("sess-A", "m1", "Alpha first retained message for similarity lookup.", _unit_vector(axis0=1.0, axis1=0.0)),
            (
                "sess-A",
                "m2",
                "Alpha second retained message for similarity lookup.",
                _unit_vector(axis0=0.999, axis1=0.045),
            ),
            ("sess-B", "m1", "Beta retained message for similarity lookup.", _unit_vector(axis0=0.99, axis1=0.141)),
            ("sess-C", "m1", "Gamma retained message for similarity lookup.", _unit_vector(axis0=0.0, axis1=1.0)),
        ],
    )
    return tmp_path / "embeddings.db"


def _provider(db_path: Path) -> SqliteVecProvider:
    provider = SqliteVecProvider(voyage_key="test-key", db_path=db_path, model="voyage-4", archive_root=db_path.parent)
    provider.dimension = EMBEDDING_DIMENSION
    provider._vec_available = None
    return provider


def test_query_by_session_excludes_seed_and_ranks_by_similarity(embeddings_db: Path) -> None:
    provider = _provider(embeddings_db)

    hits = provider.query_by_session("codex-session:sess-A", limit=10)

    returned_ids = [message_id for message_id, _distance in hits]
    # The seed session's own messages never appear.
    assert all(not message_id.startswith("codex-session:sess-A:") for message_id in returned_ids)
    # Both other sessions' messages surface, B (closer) ahead of C (orthogonal).
    assert "codex-session:sess-B:n:m1" in returned_ids
    assert "codex-session:sess-C:n:m1" in returned_ids
    assert returned_ids.index("codex-session:sess-B:n:m1") < returned_ids.index("codex-session:sess-C:n:m1")
    # Distances are non-decreasing (ascending similarity ranking).
    distances = [distance for _id, distance in hits]
    assert distances == sorted(distances)


def test_query_by_session_raises_typed_when_seed_has_no_embeddings(embeddings_db: Path) -> None:
    provider = _provider(embeddings_db)

    with pytest.raises(SqliteVecError):
        provider.query_by_session("sess-unembedded", limit=10)


def test_managed_connection_retains_current_index_embeddings(tmp_path: Path) -> None:
    """The managed current-message projection must register its hash function.

    Anti-vacuity: removing ``register_embedding_identity_sql`` from the managed
    connection path makes this fail with SQLite's ``no such function`` error.
    """
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    index_db = archive_root / "index.db"
    embeddings_db = archive_root / "embeddings.db"
    session_id = "codex-session:managed"
    message_id = f"{session_id}:n:m1"
    text = "A managed archive message long enough for retention projection."

    conn = sqlite3.connect(index_db)
    try:
        initialize_archive_tier(conn, ArchiveTier.INDEX)
        conn.execute(
            """
            INSERT INTO sessions (native_id, origin, title, content_hash)
            VALUES ('managed', 'codex-session', 'Managed', ?)
            """,
            (b"s" * 32,),
        )
        conn.execute(
            """
            INSERT INTO messages (
                session_id, native_id, position, role, material_origin,
                word_count, content_hash
            ) VALUES (?, 'm1', 0, 'user', 'human_authored', ?, ?)
            """,
            (session_id, len(text.split()), b"m" * 32),
        )
        conn.execute(
            "INSERT INTO blocks ( session_id, message_id, position, block_type, text, content_hash , content_identity, content_occurrence) VALUES (?, ?, 0, 'text', ?, ?, ?, 0)",
            (
                session_id,
                message_id,
                text,
                b"b" * 32,
                fixture_block_content_identity("text", text),
            ),
        )
        conn.commit()
    finally:
        conn.close()

    conn = sqlite3.connect(embeddings_db)
    try:
        try:
            initialize_archive_tier(conn, ArchiveTier.EMBEDDINGS)
        except sqlite3.OperationalError as exc:
            if "vec0" in str(exc) or "sqlite-vec" in str(exc):
                pytest.skip("sqlite-vec extension is unavailable")
            raise
        from polylogue.storage.sqlite.archive_tiers.embedding_write import upsert_message_embedding

        upsert_message_embedding(
            conn,
            message_id=message_id,
            session_id=session_id,
            origin=Origin.CODEX_SESSION,
            embedding=[1.0] + [0.0] * (EMBEDDING_DIMENSION - 1),
            model="voyage-4",
            embedded_at_ms=1_767_225_700_000,
            vector_derivation_hash=vector_derivation_hash(model="voyage-4", input_text=text),
        )
    finally:
        conn.close()

    provider = SqliteVecProvider(
        voyage_key="test-key",
        db_path=embeddings_db,
        model="voyage-4",
        archive_root=archive_root,
    )
    provider.dimension = EMBEDDING_DIMENSION
    provider._vec_available = None

    assert provider.count_session_embeddings(session_id) == 1

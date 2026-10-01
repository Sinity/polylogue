"""Python API embedding readiness/preflight contracts (#1503)."""

from __future__ import annotations

from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

import pytest

from polylogue.api import Polylogue
from polylogue.core.errors import VectorRuntimeUnavailableError


@pytest.mark.asyncio
async def test_embedding_status_returns_canonical_payload(tmp_path: Path) -> None:
    archive = Polylogue(archive_root=tmp_path, db_path=tmp_path / "index.db")
    payload = {
        "status": "none",
        "retrieval_ready": False,
        "next_action": {"code": "enable_embeddings", "command": "polylogue ops embed enable --yes"},
    }
    try:
        with patch(
            "polylogue.storage.embeddings.status_payload.embedding_status_payload", return_value=payload
        ) as mock_status:
            result = archive.embedding_status(detail=True)
    finally:
        await archive.close()

    assert result == payload
    mock_status.assert_called_once_with(archive, include_retrieval_bands=True, include_detail=True)


@pytest.mark.asyncio
async def test_embedding_preflight_returns_canonical_payload(tmp_path: Path) -> None:
    archive = Polylogue(archive_root=tmp_path, db_path=tmp_path / "index.db")
    report = MagicMock(name="preflight_report")
    payload = {
        "pending_sessions": 2,
        "pending_messages": 100,
        "backfill_command": "polylogue ops embed backfill --yes --max-sessions 2",
    }
    try:
        with (
            patch("polylogue.storage.embeddings.preflight.build_preflight_report", return_value=report) as mock_build,
            patch("polylogue.storage.embeddings.preflight.preflight_payload", return_value=payload) as mock_payload,
        ):
            result = archive.embedding_preflight(max_sessions=2, max_cost_usd=0.05)
    finally:
        await archive.close()

    assert result == payload
    mock_build.assert_called_once_with(
        tmp_path / "index.db",
        rebuild=False,
        max_sessions=2,
        max_messages=None,
        max_cost_usd=0.05,
    )
    mock_payload.assert_called_once_with(report)


@pytest.mark.asyncio
async def test_search_similar_sessions_fails_closed_without_vector_provider(tmp_path: Path) -> None:
    archive = Polylogue(archive_root=tmp_path, db_path=tmp_path / "index.db")
    try:
        with (
            patch("polylogue.storage.search_providers.create_vector_provider", return_value=None),
            pytest.raises(VectorRuntimeUnavailableError),
        ):
            await archive.search_similar_sessions("missing-session")
    finally:
        await archive.close()


@pytest.mark.asyncio
async def test_retained_similarity_uses_explicit_archive_recipe_without_acquisition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Replacing the explicit Config with ambient settings loses these retained hits."""
    from polylogue.config import Config, PolylogueConfig
    from polylogue.storage.search_providers.sqlite_vec import SqliteVecProvider
    from tests.infra.vector_archive import seed_vector_archive

    root = tmp_path / "explicit-archive"
    seed_vector_archive(
        root,
        [
            ("seed", "m1", "Synthetic seed prose for an explicitly configured recipe.", [1.0] + [0.0] * 1023),
            (
                "near",
                "m1",
                "Synthetic neighbor prose for an explicitly configured recipe.",
                [0.99, 0.141] + [0.0] * 1022,
            ),
        ],
        model="voyage-4",
    )
    config = Config(archive_root=root, render_root=root / "render", sources=[], embedding_model="voyage-4")
    monkeypatch.setattr("polylogue.config.load_polylogue_config", lambda: PolylogueConfig())
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    provider_call = MagicMock(side_effect=AssertionError("retained reads must not acquire vectors"))
    monkeypatch.setattr(SqliteVecProvider, "_get_embeddings", provider_call)
    async with Polylogue(config=config) as archive:
        result = await archive.search_similar_sessions("codex-session:seed")
    assert result["source_embedded_messages"] == 1
    hits = cast(list[dict[str, object]], result["results"])
    assert [hit["session_id"] for hit in hits] == ["codex-session:near"]
    provider_call.assert_not_called()


@pytest.mark.asyncio
async def test_supplied_public_vector_provider_projects_retained_similarity(tmp_path: Path) -> None:
    """The public provider protocol includes the operation the API invokes."""
    from polylogue.core.protocols import VectorProvider
    from polylogue.storage.search_providers.sqlite_vec import SqliteVecProvider
    from tests.infra.vector_archive import seed_vector_archive

    seed_vector_archive(
        tmp_path,
        [
            ("seed", "m1", "Synthetic seed prose for a supplied provider.", [1.0] + [0.0] * 1023),
            ("near", "m1", "Synthetic neighbor prose for a supplied provider.", [0.99, 0.141] + [0.0] * 1022),
        ],
    )
    provider: VectorProvider = SqliteVecProvider(
        voyage_key=None, db_path=tmp_path / "embeddings.db", archive_root=tmp_path, model="voyage-4"
    )
    assert isinstance(provider, VectorProvider)
    with patch.object(provider, "_get_embeddings", side_effect=AssertionError("no acquisition")) as acquisition:
        async with Polylogue(archive_root=tmp_path, db_path=tmp_path / "index.db") as archive:
            result = await archive.search_similar_sessions("codex-session:seed", vector_provider=provider)
    assert result["source_embedded_messages"] == 1
    assert [hit["session_id"] for hit in cast(list[dict[str, object]], result["results"])] == ["codex-session:near"]
    acquisition.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("supplied_snapshot", [False, True])
async def test_supplied_vector_provider_refuses_a_different_archive_index(
    tmp_path: Path, supplied_snapshot: bool
) -> None:
    """An archive-A provider cannot attach its hits to archive-B metadata."""
    from polylogue.storage.embeddings.identity import EmbeddingRecipe
    from polylogue.storage.search_providers.sqlite_vec import SqliteVecError, SqliteVecProvider
    from polylogue.storage.search_providers.sqlite_vec_runtime import open_vector_read_snapshot
    from tests.infra.vector_archive import seed_vector_archive

    root_a, root_b = tmp_path / "a", tmp_path / "b"
    for root in (root_a, root_b):
        seed_vector_archive(root, [("seed", "m1", "Synthetic selected archive prose.", [1.0] + [0.0] * 1023)])
    connection = None
    if supplied_snapshot:
        connection = open_vector_read_snapshot(
            embeddings_path=root_a / "embeddings.db",
            index_path=root_a / "index.db",
            recipe=EmbeddingRecipe.current(model="voyage-4", dimensions=1024),
        )
        provider = SqliteVecProvider.from_vector_read_snapshot(voyage_key=None, connection=connection, model="voyage-4")
    else:
        provider = SqliteVecProvider(None, db_path=root_a / "embeddings.db", archive_root=root_a, model="voyage-4")
    try:
        with patch.object(provider, "_get_embeddings", side_effect=AssertionError("no acquisition")) as acquisition:
            async with Polylogue(archive_root=root_b, db_path=root_b / "index.db") as archive:
                with pytest.raises(SqliteVecError):
                    await archive.search_similar_sessions("codex-session:seed", vector_provider=provider)
            acquisition.assert_not_called()
        if connection is not None:
            assert connection.execute("SELECT 1").fetchone()[0] == 1
    finally:
        if connection is not None:
            connection.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("construct_on_worker", [False, True])
async def test_supplied_thread_affine_snapshot_remains_owned_by_its_creator(
    tmp_path: Path, construct_on_worker: bool
) -> None:
    """Moving this public API call to a worker violates SQLite thread affinity."""
    import asyncio
    import threading

    from polylogue.storage.embeddings.identity import EmbeddingRecipe
    from polylogue.storage.search_providers.sqlite_vec import SqliteVecProvider
    from polylogue.storage.search_providers.sqlite_vec_runtime import open_vector_read_snapshot
    from tests.infra.vector_archive import seed_vector_archive

    seed_vector_archive(
        tmp_path,
        [
            ("seed", "m1", "Synthetic retained seed.", [1.0] + [0.0] * 1023),
            ("near", "m1", "Synthetic retained neighbor.", [0.99, 0.141] + [0.0] * 1022),
        ],
    )
    connection = open_vector_read_snapshot(
        embeddings_path=tmp_path / "embeddings.db",
        index_path=tmp_path / "index.db",
        recipe=EmbeddingRecipe.current(model="voyage-4", dimensions=1024),
    )
    creator = threading.get_ident()
    if construct_on_worker:
        provider = await asyncio.to_thread(
            SqliteVecProvider.from_vector_read_snapshot, voyage_key=None, connection=connection, model="voyage-4"
        )
    else:
        provider = SqliteVecProvider.from_vector_read_snapshot(voyage_key=None, connection=connection, model="voyage-4")
    try:
        with patch.object(provider, "_get_embeddings", side_effect=AssertionError("no acquisition")) as acquisition:
            async with Polylogue(archive_root=tmp_path, db_path=tmp_path / "index.db") as archive:
                result = await archive.search_similar_sessions("codex-session:seed", vector_provider=provider)
            assert result["source_embedded_messages"] == 1
            assert [hit["session_id"] for hit in cast(list[dict[str, object]], result["results"])] == [
                "codex-session:near"
            ]
            assert threading.get_ident() == creator
            assert connection.execute("SELECT 1").fetchone()[0] == 1
            acquisition.assert_not_called()
    finally:
        connection.close()


@pytest.mark.asyncio
async def test_public_provider_projection_owns_rows_without_changing_connection(tmp_path: Path) -> None:
    """A public provider can project over SQLite's ordinary tuple connection."""
    import sqlite3
    from collections.abc import Callable
    from contextlib import closing

    from polylogue.storage.search_providers.sqlite_vec import SqliteVecProvider
    from tests.infra.vector_archive import seed_vector_archive

    identities = seed_vector_archive(
        tmp_path,
        [
            ("seed", "m1", "Synthetic external provider seed.", [1.0] + [0.0] * 1023),
            ("near", "m1", "Synthetic external provider neighbor.", [0.99, 0.141] + [0.0] * 1022),
        ],
    )

    class TupleProvider(SqliteVecProvider):
        async def read_session_similarity(
            self,
            session_id: str,
            *,
            index_path: Path,
            limit: int = 10,
            project: Callable[[sqlite3.Connection, int, list[tuple[str, float]]], dict[str, object]],
        ) -> dict[str, object]:
            with closing(sqlite3.connect(":memory:")) as connection:
                connection.execute("ATTACH DATABASE ? AS archive_index", (str(index_path),))
                assert connection.row_factory is None
                result = project(connection, 1, [(identities[("near", "m1")][1], 0.1)])
                assert connection.row_factory is None
                assert connection.execute("SELECT 1").fetchone() == (1,)
                return result

    provider = TupleProvider(None, archive_root=tmp_path, db_path=tmp_path / "embeddings.db", model="voyage-4")
    async with Polylogue(archive_root=tmp_path, db_path=tmp_path / "index.db") as archive:
        result = await archive.search_similar_sessions("codex-session:seed", vector_provider=provider)
    assert [hit["session_id"] for hit in cast(list[dict[str, object]], result["results"])] == ["codex-session:near"]


@pytest.mark.asyncio
@pytest.mark.parametrize("replace_before_provider", [False, True])
async def test_supplied_snapshot_refuses_index_identity_replaced_before_api_read(
    tmp_path: Path, replace_before_provider: bool
) -> None:
    """An already held conventional index cannot hydrate through its replacement."""
    import shutil

    from polylogue.storage.embeddings.identity import EmbeddingRecipe
    from polylogue.storage.search_providers.sqlite_vec import SqliteVecError, SqliteVecProvider
    from polylogue.storage.search_providers.sqlite_vec_runtime import open_vector_read_snapshot
    from tests.infra.vector_archive import seed_vector_archive

    seed_vector_archive(tmp_path, [("seed", "m1", "Synthetic pinned index prose.", [1.0] + [0.0] * 1023)])
    connection = open_vector_read_snapshot(
        embeddings_path=tmp_path / "embeddings.db",
        index_path=tmp_path / "index.db",
        recipe=EmbeddingRecipe.current(model="voyage-4", dimensions=1024),
    )
    provider = None
    if not replace_before_provider:
        provider = SqliteVecProvider.from_vector_read_snapshot(voyage_key=None, connection=connection, model="voyage-4")
    try:
        (tmp_path / "index.db").rename(tmp_path / "held-index.db")
        shutil.copyfile(tmp_path / "held-index.db", tmp_path / "index.db")
        if replace_before_provider:
            provider = SqliteVecProvider.from_vector_read_snapshot(
                voyage_key=None, connection=connection, model="voyage-4"
            )
        assert provider is not None
        async with Polylogue(archive_root=tmp_path, db_path=tmp_path / "index.db") as archive:
            with pytest.raises(SqliteVecError):
                await archive.search_similar_sessions("codex-session:seed", vector_provider=provider)
        assert connection.execute("SELECT COUNT(*) FROM archive_index.sessions").fetchone()[0] == 1
    finally:
        connection.close()


@pytest.mark.asyncio
async def test_retained_similarity_distinguishes_missing_seed_from_present_unembedded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Returning an empty result for an absent source turns this red."""
    from polylogue.config import Config
    from polylogue.core.errors import SessionNotFoundError
    from polylogue.storage.search_providers.sqlite_vec import SqliteVecProvider
    from tests.infra.vector_archive import record_owned_vector_closes, seed_vector_archive

    seed_vector_archive(tmp_path, [("seed", "m1", "Synthetic seed prose.", [1.0] + [0.0] * 1023)])
    import sqlite3

    with sqlite3.connect(tmp_path / "index.db") as index:
        index.execute(
            "INSERT INTO sessions (native_id, origin, title, content_hash) VALUES (?, ?, ?, ?)",
            ("unembedded", "codex-session", "Unembedded", b"u" * 32),
        )
    config = Config(archive_root=tmp_path, render_root=tmp_path / "render", sources=[], embedding_model="voyage-4")
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    provider_call = MagicMock(side_effect=AssertionError("retained reads must not acquire vectors"))
    monkeypatch.setattr(SqliteVecProvider, "_get_embeddings", provider_call)
    closed = record_owned_vector_closes(monkeypatch)
    async with Polylogue(config=config) as archive:
        with pytest.raises(SessionNotFoundError):
            await archive.search_similar_sessions("codex-session:missing")
        result = await archive.search_similar_sessions("codex-session:unembedded")
    assert result == {"source_embedded_messages": 0, "results": [], "unresolved_message_hits": 0}
    assert closed == [True, True]
    provider_call.assert_not_called()

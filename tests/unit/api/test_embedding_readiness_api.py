"""Python API embedding readiness/preflight contracts (#1503)."""

from __future__ import annotations

from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

import pytest

from polylogue.api import Polylogue


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
            pytest.raises(ValueError, match="No vector provider configured"),
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

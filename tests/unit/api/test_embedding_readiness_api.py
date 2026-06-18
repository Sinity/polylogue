"""Python API embedding readiness/preflight contracts (#1503)."""

from __future__ import annotations

from pathlib import Path
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

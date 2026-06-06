"""Contracts for query miss diagnostics."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from polylogue.archive.query.miss_diagnostics import QueryMissDiagnostics, diagnose_query_miss
from polylogue.archive.query.spec import SessionQuerySpec
from polylogue.archive.stats import ArchiveStats
from polylogue.config import Config
from polylogue.core.outcomes import OutcomeCheck, OutcomeStatus
from polylogue.readiness import ReadinessReport
from polylogue.storage.action_events.artifacts import ActionEventArtifactState


def _codes(diagnostics: QueryMissDiagnostics) -> list[str]:
    return [reason.code for reason in diagnostics.reasons]


@pytest.mark.asyncio
async def test_diagnose_query_miss_reports_empty_archive_scope() -> None:
    repo = MagicMock()
    repo.get_archive_stats = AsyncMock(return_value=ArchiveStats(total_sessions=0, total_messages=0))
    repo.get_raw_session_count = AsyncMock(return_value=0)

    diagnostics = await diagnose_query_miss(repo, SessionQuerySpec())

    assert _codes(diagnostics) == ["archive_empty"]
    assert diagnostics.archive_session_count == 0
    assert diagnostics.raw_session_count == 0
    repo.get_raw_session_count.assert_awaited_once_with(provider=None)


@pytest.mark.asyncio
async def test_diagnose_query_miss_reports_raw_backlog_for_selected_origin() -> None:
    repo = MagicMock()
    repo.get_archive_stats = AsyncMock(
        return_value=ArchiveStats(
            total_sessions=3,
            total_messages=12,
            origins={"chatgpt-export": 3},
        )
    )
    repo.get_raw_session_count = AsyncMock(return_value=2)
    selection = SessionQuerySpec(origins=("claude-ai-export",))

    diagnostics = await diagnose_query_miss(repo, selection)

    assert _codes(diagnostics) == ["archive_empty", "raw_ingest_backlog"]
    assert diagnostics.archive_session_count == 0
    assert diagnostics.raw_session_count == 2
    repo.get_raw_session_count.assert_awaited_once_with(provider="claude-ai")


@pytest.mark.asyncio
async def test_diagnose_query_miss_reports_degraded_action_readiness() -> None:
    repo = MagicMock()
    repo.get_archive_stats = AsyncMock(return_value=ArchiveStats(total_sessions=5, total_messages=20))
    repo.get_raw_session_count = AsyncMock(return_value=0)
    repo.get_action_event_artifact_state = AsyncMock(
        return_value=ActionEventArtifactState(
            source_sessions=5,
            materialized_sessions=3,
            materialized_rows=4,
            fts_rows=1,
        )
    )
    selection = SessionQuerySpec(action_terms=("file_edit",))

    diagnostics = await diagnose_query_miss(repo, selection)

    assert _codes(diagnostics) == ["action_read_model_degraded"]
    reason = diagnostics.reasons[0]
    assert reason.count == 5
    assert "missing sessions" in str(reason.detail)


@pytest.mark.asyncio
async def test_diagnose_query_miss_reports_degraded_message_index() -> None:
    repo = MagicMock()
    repo.get_archive_stats = AsyncMock(return_value=ArchiveStats(total_sessions=5, total_messages=20))
    repo.get_raw_session_count = AsyncMock(return_value=0)
    report = ReadinessReport(
        checks=[
            OutcomeCheck(
                "index",
                OutcomeStatus.WARNING,
                summary="messages indexed: 1/20",
                count=1,
            )
        ]
    )

    with patch("polylogue.archive.query.miss_diagnostics.get_readiness", return_value=report) as mock_readiness:
        diagnostics = await diagnose_query_miss(
            repo,
            SessionQuerySpec(query_terms=("needle",)),
            config=cast(Config, SimpleNamespace(db_path=Path("archive.sqlite"))),
        )

    assert _codes(diagnostics) == ["message_index_degraded"]
    assert diagnostics.reasons[0].detail == "messages indexed: 1/20"
    assert diagnostics.reasons[0].count == 1
    mock_readiness.assert_called_once()


@pytest.mark.asyncio
async def test_diagnose_query_miss_falls_back_to_matching_absence() -> None:
    repo = MagicMock()
    repo.get_archive_stats = AsyncMock(return_value=ArchiveStats(total_sessions=5, total_messages=20))
    repo.get_raw_session_count = AsyncMock(return_value=0)

    diagnostics = await diagnose_query_miss(repo, SessionQuerySpec(query_terms=("absent",)))

    assert _codes(diagnostics) == ["no_matching_session"]
    assert diagnostics.reasons[0].count == 5

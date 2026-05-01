"""Contracts for query miss diagnostics."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from polylogue.archive.query.miss_diagnostics import QueryMissDiagnostics, diagnose_query_miss
from polylogue.archive.query.spec import ConversationQuerySpec
from polylogue.archive.stats import ArchiveStats
from polylogue.config import Config
from polylogue.core.outcomes import OutcomeCheck, OutcomeStatus
from polylogue.readiness import ReadinessReport
from polylogue.storage.action_events.artifacts import ActionEventArtifactState
from polylogue.types import Provider


def _codes(diagnostics: QueryMissDiagnostics) -> list[str]:
    return [reason.code for reason in diagnostics.reasons]


@pytest.mark.asyncio
async def test_diagnose_query_miss_reports_empty_archive_scope() -> None:
    repo = MagicMock()
    repo.get_archive_stats = AsyncMock(return_value=ArchiveStats(total_conversations=0, total_messages=0))
    repo.get_raw_conversation_count = AsyncMock(return_value=0)

    diagnostics = await diagnose_query_miss(repo, ConversationQuerySpec())

    assert _codes(diagnostics) == ["archive_empty"]
    assert diagnostics.archive_conversation_count == 0
    assert diagnostics.raw_conversation_count == 0
    repo.get_raw_conversation_count.assert_awaited_once_with(provider=None)


@pytest.mark.asyncio
async def test_diagnose_query_miss_reports_raw_backlog_for_selected_provider() -> None:
    repo = MagicMock()
    repo.get_archive_stats = AsyncMock(
        return_value=ArchiveStats(
            total_conversations=3,
            total_messages=12,
            providers={"chatgpt": 3},
        )
    )
    repo.get_raw_conversation_count = AsyncMock(return_value=2)
    selection = ConversationQuerySpec(providers=(Provider.CLAUDE_AI,))

    diagnostics = await diagnose_query_miss(repo, selection)

    assert _codes(diagnostics) == ["archive_empty", "raw_ingest_backlog"]
    assert diagnostics.archive_conversation_count == 0
    assert diagnostics.raw_conversation_count == 2
    repo.get_raw_conversation_count.assert_awaited_once_with(provider="claude-ai")


@pytest.mark.asyncio
async def test_diagnose_query_miss_reports_degraded_action_readiness() -> None:
    repo = MagicMock()
    repo.get_archive_stats = AsyncMock(return_value=ArchiveStats(total_conversations=5, total_messages=20))
    repo.get_raw_conversation_count = AsyncMock(return_value=0)
    repo.get_action_event_artifact_state = AsyncMock(
        return_value=ActionEventArtifactState(
            source_conversations=5,
            materialized_conversations=3,
            materialized_rows=4,
            fts_rows=1,
        )
    )
    selection = ConversationQuerySpec(action_terms=("file_edit",))

    diagnostics = await diagnose_query_miss(repo, selection)

    assert _codes(diagnostics) == ["action_read_model_degraded"]
    reason = diagnostics.reasons[0]
    assert reason.count == 5
    assert "missing conversations" in str(reason.detail)


@pytest.mark.asyncio
async def test_diagnose_query_miss_reports_degraded_message_index() -> None:
    repo = MagicMock()
    repo.get_archive_stats = AsyncMock(return_value=ArchiveStats(total_conversations=5, total_messages=20))
    repo.get_raw_conversation_count = AsyncMock(return_value=0)
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
            ConversationQuerySpec(query_terms=("needle",)),
            config=cast(Config, SimpleNamespace(db_path=Path("archive.sqlite"))),
        )

    assert _codes(diagnostics) == ["message_index_degraded"]
    assert diagnostics.reasons[0].detail == "messages indexed: 1/20"
    assert diagnostics.reasons[0].count == 1
    mock_readiness.assert_called_once()


@pytest.mark.asyncio
async def test_diagnose_query_miss_falls_back_to_matching_absence() -> None:
    repo = MagicMock()
    repo.get_archive_stats = AsyncMock(return_value=ArchiveStats(total_conversations=5, total_messages=20))
    repo.get_raw_conversation_count = AsyncMock(return_value=0)

    diagnostics = await diagnose_query_miss(repo, ConversationQuerySpec(query_terms=("absent",)))

    assert _codes(diagnostics) == ["no_matching_conversation"]
    assert diagnostics.reasons[0].count == 5

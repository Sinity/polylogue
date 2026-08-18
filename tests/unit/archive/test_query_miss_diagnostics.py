"""Contracts for query miss diagnostics."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from polylogue.archive.query.miss_diagnostics import (
    QueryMissDiagnostics,
    diagnose_named_source_miss,
    diagnose_query_miss,
)
from polylogue.archive.query.source_freshness import NamedSourceFreshness
from polylogue.archive.query.spec import SessionQuerySpec
from polylogue.archive.stats import ArchiveStats
from polylogue.config import Config
from polylogue.core.outcomes import OutcomeCheck, OutcomeStatus
from polylogue.readiness import ReadinessReport


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
    repo.get_raw_session_count.assert_awaited_once_with(origin=None)


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
    repo.get_raw_session_count.assert_awaited_once_with(origin="claude-ai-export")


@pytest.mark.asyncio
async def test_diagnose_query_miss_does_not_require_action_read_model() -> None:
    repo = MagicMock()
    repo.get_archive_stats = AsyncMock(return_value=ArchiveStats(total_sessions=5, total_messages=20))
    repo.get_raw_session_count = AsyncMock(return_value=0)
    repo.get_action_artifact_state = AsyncMock(side_effect=AssertionError("old action readiness must not be read"))
    selection = SessionQuerySpec(action_terms=("file_edit",))

    diagnostics = await diagnose_query_miss(repo, selection)

    assert _codes(diagnostics) == ["no_matching_session"]
    repo.get_action_artifact_state.assert_not_awaited()


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


@pytest.mark.parametrize(
    ("stage", "code"),
    [
        ("unseen", "named_source_unseen"),
        ("acquired-unparsed", "named_source_acquired_unparsed"),
        ("parsed-unindexed", "named_source_parsed_unindexed"),
        ("indexed-unconverged", "named_source_indexed_unconverged"),
        ("searchable", "named_source_searchable"),
    ],
)
def test_named_source_miss_diagnostics_distinguish_pipeline_stage(stage: str, code: str) -> None:
    freshness = SimpleNamespace(
        source_path="/exact/source.jsonl",
        stage=SimpleNamespace(value=stage),
        operational_state=SimpleNamespace(value="idle"),
        cursor=SimpleNamespace(excluded=False, pending_bytes=0),
        retry=SimpleNamespace(reason=None),
        index=SimpleNamespace(session_count_lower_bound=1, broken_head=False),
        raw_revisions=(object(),),
    )

    diagnostics = diagnose_named_source_miss(cast(NamedSourceFreshness, freshness))

    assert _codes(diagnostics) == [code]
    detail = diagnostics.reasons[0].detail or ""
    assert f"stage={stage}" in detail
    assert "source_indexed_session_lower_bound=1" in detail
    assert "source_raw_revision_sample_count=1" in detail
    assert diagnostics.archive_session_count is None
    assert diagnostics.raw_session_count is None


@pytest.mark.asyncio
async def test_action_scoped_miss_attributes_an_empty_action_read_model() -> None:
    """An action filter over an archive with no indexed actions is attributable.

    ``actions`` is a view over tool_use/tool_result blocks. An archive holding
    sessions but no tool blocks makes every action-scoped query miss for a reason
    the predicate breakdown cannot express -- the filter is not too narrow, the
    relation is empty. This producer returned ``None`` unconditionally, so that
    miss was silently unattributable.
    """
    repo = MagicMock()
    repo.get_archive_stats = AsyncMock(return_value=ArchiveStats(total_sessions=12, total_messages=340))
    repo.get_raw_session_count = AsyncMock(return_value=12)
    repo.count_actions = AsyncMock(return_value=0)

    diagnostics = await diagnose_query_miss(repo, SessionQuerySpec(action_terms=("Edit",)))

    assert "action_read_model_empty" in _codes(diagnostics)
    repo.count_actions.assert_awaited_once_with(origin=None)


@pytest.mark.asyncio
async def test_action_read_model_reason_is_silent_when_actions_exist() -> None:
    repo = MagicMock()
    repo.get_archive_stats = AsyncMock(return_value=ArchiveStats(total_sessions=12, total_messages=340))
    repo.get_raw_session_count = AsyncMock(return_value=12)
    repo.count_actions = AsyncMock(return_value=57)

    diagnostics = await diagnose_query_miss(repo, SessionQuerySpec(action_terms=("Edit",)))

    assert "action_read_model_empty" not in _codes(diagnostics)


@pytest.mark.asyncio
async def test_action_read_model_reason_skipped_for_non_action_queries() -> None:
    """A plain text query must not pay for -- or be explained by -- the action relation."""
    repo = MagicMock()
    repo.get_archive_stats = AsyncMock(return_value=ArchiveStats(total_sessions=12, total_messages=340))
    repo.get_raw_session_count = AsyncMock(return_value=12)
    repo.count_actions = AsyncMock(return_value=0)

    diagnostics = await diagnose_query_miss(repo, SessionQuerySpec(query_terms=("needle",)))

    assert "action_read_model_empty" not in _codes(diagnostics)
    repo.count_actions.assert_not_awaited()

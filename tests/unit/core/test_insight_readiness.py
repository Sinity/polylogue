"""Tests for insight readiness report construction."""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.analysis.readiness import (
    InsightReadinessEntry,
    InsightReadinessQuery,
    InsightReadinessReport,
)
from polylogue.api import Polylogue
from polylogue.storage.runtime.store_constants import SESSION_INSIGHT_MATERIALIZER_VERSION
from tests.infra.storage_records import SessionBuilder


def _entry_by_name(report: InsightReadinessReport, name: str) -> InsightReadinessEntry:
    return next(insight for insight in report.insights if insight.insight_name == name)


def _seed_readiness_sessions(db_path: Path) -> None:
    (
        SessionBuilder(db_path, "ready-root")
        .provider("codex")
        .title("Ready Root")
        .created_at("2026-04-01T09:00:00+00:00")
        .updated_at("2026-04-01T09:10:00+00:00")
        .add_message(
            "u1",
            role="user",
            text="Plan insight readiness reporting.",
            timestamp="2026-04-01T09:00:00+00:00",
        )
        .add_message(
            "a1",
            role="assistant",
            text="Implement readiness report and tests.",
            timestamp="2026-04-01T09:05:00+00:00",
        )
        .save()
    )


async def _rebuild(db_path: Path) -> None:
    archive = Polylogue(archive_root=db_path.parent, db_path=db_path)
    try:
        await archive.rebuild_insights()
    finally:
        await archive.close()


def _provider_native_id(token: str, origin: str = "claude-code-session") -> str:
    return f"{origin}:ext-{token}"


@pytest.mark.asyncio
async def test_insight_readiness_report_marks_rebuilt_insights_ready(cli_workspace: dict[str, Path]) -> None:
    db_path = cli_workspace["db_path"]
    _seed_readiness_sessions(db_path)
    await _rebuild(db_path)

    archive = Polylogue(archive_root=cli_workspace["archive_root"], db_path=db_path)
    report = await archive.insight_readiness_report()

    # The sparse seed deliberately lacks the evidence needed for a fully
    # grounded profile. Rebuild is complete, so the rows do not diverge from
    # their sources, but coverage must still surface the fallback.
    assert {insight.insight_name for insight in report.insights} >= {
        "session_profiles",
        "session_work_events",
        "session_phases",
        "threads",
        "session_tag_rollups",
        "archive_coverage",
    }
    profile = _entry_by_name(report, "session_profiles")
    assert not profile.diverged
    assert profile.degraded_count == 1
    assert profile.fallback_reason_counts
    assert profile.row_count == 1
    assert profile.origin_coverage[0].origin == "codex-session"


@pytest.mark.asyncio
async def test_insight_readiness_report_marks_empty_insights(cli_workspace: dict[str, Path]) -> None:
    archive = Polylogue(archive_root=cli_workspace["archive_root"], db_path=cli_workspace["db_path"])

    report = await archive.insight_readiness_report(InsightReadinessQuery(insights=("session_profiles",)))

    profile = _entry_by_name(report, "session_profiles")
    assert profile.table_present
    assert not profile.diverged
    assert not profile.incomplete
    assert profile.row_count == 0
    assert profile.expected_row_count == 0


@pytest.mark.asyncio
async def test_insight_readiness_report_marks_partial_and_incompatible_insights(
    cli_workspace: dict[str, Path],
) -> None:
    import sqlite3

    db_path = cli_workspace["db_path"]
    _seed_readiness_sessions(db_path)
    (
        SessionBuilder(db_path, "ready-second")
        .provider("codex")
        .title("Ready Second")
        .created_at("2026-04-01T10:00:00+00:00")
        .updated_at("2026-04-01T10:05:00+00:00")
        .add_message("u2", role="user", text="Second session.", timestamp="2026-04-01T10:00:00+00:00")
        .save()
    )
    await _rebuild(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "DELETE FROM session_profiles WHERE session_id = ?",
            (_provider_native_id("ready-second", "codex-session"),),
        )
        conn.commit()

    archive = Polylogue(archive_root=cli_workspace["archive_root"], db_path=db_path)
    partial = await archive.insight_readiness_report(InsightReadinessQuery(insights=("session_profiles",)))
    incomplete = _entry_by_name(partial, "session_profiles")
    assert incomplete.incomplete
    assert not incomplete.diverged

    await _rebuild(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE session_profiles SET materializer_version = ?",
            (SESSION_INSIGHT_MATERIALIZER_VERSION - 1,),
        )
        conn.commit()

    stale = await archive.insight_readiness_report(InsightReadinessQuery(insights=("session_profiles",)))
    profile = _entry_by_name(stale, "session_profiles")
    assert profile.diverged
    assert profile.stale_count == 2


@pytest.mark.asyncio
async def test_insight_readiness_report_marks_stale_insights(cli_workspace: dict[str, Path]) -> None:
    import sqlite3

    db_path = cli_workspace["db_path"]
    _seed_readiness_sessions(db_path)
    await _rebuild(db_path)
    with sqlite3.connect(db_path) as conn:
        # sort_key_ms is a generated column (COALESCE(updated_at_ms, created_at_ms));
        # bump the source updated_at_ms so the derived high-water mark advances past
        # the value captured at materialization time.
        conn.execute(
            "UPDATE sessions SET updated_at_ms = COALESCE(updated_at_ms, created_at_ms, 0) + 1000 WHERE session_id = ?",
            (_provider_native_id("ready-root", "codex-session"),),
        )
        conn.commit()

    archive = Polylogue(archive_root=cli_workspace["archive_root"], db_path=db_path)
    report = await archive.insight_readiness_report(InsightReadinessQuery(insights=("session_profiles",)))

    profile = _entry_by_name(report, "session_profiles")
    assert profile.diverged
    assert profile.stale_count == 1


def test_absent_table_is_divergence_not_an_honest_zero() -> None:
    """``table_present`` decides divergence; an empty present table does not.

    This is the derivation that replaced the ``missing``/``empty`` verdicts.
    Goes red if ``diverged`` stops reading ``table_present`` -- an absent table
    would then be indistinguishable from a table that legitimately holds no rows.
    """
    absent = InsightReadinessEntry(
        insight_name="session_profiles",
        display_name="Session Profiles",
        table_present=False,
    )
    empty = InsightReadinessEntry(
        insight_name="session_profiles",
        display_name="Session Profiles",
        table_present=True,
        expected_row_count=0,
    )

    assert absent.diverged
    assert not empty.diverged
    assert not empty.incomplete
    assert absent.row_count == empty.row_count == 0


@pytest.mark.asyncio
async def test_permanently_absent_storage_artifact_reports_absent(cli_workspace: dict[str, Path]) -> None:
    """The presence probe reads sqlite_master rather than assuming presence.

    ``session_runs`` is a source-derived CTE relation whose legacy table never
    exists, so its storage artifact must report ``present=False`` while the
    surface itself still counts rows. Goes red if artifact presence is
    hardcoded true, which would hide a genuinely missing table.
    """
    db_path = cli_workspace["db_path"]
    _seed_readiness_sessions(db_path)
    await _rebuild(db_path)

    archive = Polylogue(archive_root=cli_workspace["archive_root"], db_path=db_path)
    report = await archive.insight_readiness_report(InsightReadinessQuery(insights=("session_runs",)))

    runs = _entry_by_name(report, "session_runs")
    assert [artifact.name for artifact in runs.storage_artifacts] == ["session_runs"]
    assert runs.storage_artifacts[0].present is False
    # The surface is still reported; only its legacy cache table is absent.
    assert runs.table_present

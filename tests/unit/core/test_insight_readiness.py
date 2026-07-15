"""Tests for insight readiness report construction."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import aiosqlite
import pytest

from polylogue.api import Polylogue
from polylogue.insights.readiness import (
    InsightReadinessEntry,
    InsightReadinessQuery,
    InsightReadinessReport,
    build_insight_readiness_report,
)
from polylogue.storage.insights.session.status import session_insight_status_sync
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

    # The sparse seed now materializes a complete deterministic insight set;
    # missing rich workflow events do not make the rebuilt read model degraded.
    assert report.aggregate_verdict == "ready"
    assert {insight.insight_name for insight in report.insights} >= {
        "session_profiles",
        "session_work_events",
        "session_phases",
        "threads",
        "session_tag_rollups",
        "archive_coverage",
    }
    profile = _entry_by_name(report, "session_profiles")
    assert profile.verdict == "ready"
    assert profile.degraded_count == 0
    assert profile.fallback_reason_counts == {}
    assert profile.row_count == 1
    assert profile.provider_coverage[0].source_name == "codex"
    assert profile.version_coverage[0].versions[str(SESSION_INSIGHT_MATERIALIZER_VERSION)] == 1


@pytest.mark.asyncio
async def test_insight_readiness_report_marks_empty_insights(cli_workspace: dict[str, Path]) -> None:
    archive = Polylogue(archive_root=cli_workspace["archive_root"], db_path=cli_workspace["db_path"])

    report = await archive.insight_readiness_report(InsightReadinessQuery(insights=("session_profiles",)))

    profile = _entry_by_name(report, "session_profiles")
    assert report.aggregate_verdict == "empty"
    assert profile.verdict == "empty"
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
    assert _entry_by_name(partial, "session_profiles").verdict == "partial"

    await _rebuild(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE insight_materialization SET materializer_version = ?",
            (SESSION_INSIGHT_MATERIALIZER_VERSION - 1,),
        )
        conn.commit()

    incompatible = await archive.insight_readiness_report(InsightReadinessQuery(insights=("session_profiles",)))
    profile = _entry_by_name(incompatible, "session_profiles")
    assert profile.verdict == "incompatible"
    assert profile.incompatible_count == 2


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
    assert report.aggregate_verdict == "stale"
    assert profile.verdict == "stale"
    assert profile.stale_count == 1


@pytest.mark.asyncio
async def test_insight_readiness_report_marks_missing_insight_tables(tmp_path: Path) -> None:
    db_path = tmp_path / "missing.db"
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        conn.executescript(
            """
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                parent_session_id TEXT,
                source_name TEXT,
                origin TEXT,
                branch_type TEXT,
                title TEXT,
                git_branch TEXT,
                native_id TEXT,
                message_count INTEGER,
                tool_use_count INTEGER,
                created_at_ms INTEGER,
                updated_at_ms INTEGER,
                sort_key REAL,
                updated_at TEXT
            );
            CREATE TABLE blocks (
                block_id TEXT PRIMARY KEY,
                session_id TEXT,
                block_type TEXT,
                message_id TEXT,
                position INTEGER,
                semantic_type TEXT,
                tool_command TEXT,
                tool_id TEXT,
                tool_name TEXT,
                tool_result_exit_code INTEGER,
                tool_result_is_error INTEGER,
                search_text TEXT
            );
            INSERT INTO sessions (session_id, parent_session_id, source_name, sort_key, updated_at)
            VALUES ('missing-root', NULL, 'codex', 1.0, '2026-04-01T00:00:00Z');
            """
        )
        status = session_insight_status_sync(conn)

    async with aiosqlite.connect(db_path) as conn:
        conn.row_factory = aiosqlite.Row
        report = await build_insight_readiness_report(
            conn,
            status,
            InsightReadinessQuery(insights=("session_profiles",)),
        )

    profile = _entry_by_name(report, "session_profiles")
    assert profile.verdict == "missing"
    assert profile.row_count == 0
    assert profile.expected_row_count == 1

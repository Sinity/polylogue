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
from polylogue.storage.insights.session.rebuild import rebuild_session_insights_sync
from polylogue.storage.insights.session.status import session_insight_status_sync
from polylogue.storage.runtime.store_constants import SESSION_INSIGHT_MATERIALIZER_VERSION
from polylogue.storage.sqlite.connection import open_connection
from tests.infra.storage_records import ConversationBuilder


def _entry_by_name(report: InsightReadinessReport, name: str) -> InsightReadinessEntry:
    return next(insight for insight in report.insights if insight.insight_name == name)


def _seed_readiness_conversations(db_path: Path) -> None:
    (
        ConversationBuilder(db_path, "ready-root")
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


@pytest.mark.asyncio
async def test_insight_readiness_report_marks_rebuilt_insights_ready(cli_workspace: dict[str, Path]) -> None:
    db_path = cli_workspace["db_path"]
    _seed_readiness_conversations(db_path)
    with open_connection(db_path) as conn:
        rebuild_session_insights_sync(conn)

    archive = Polylogue(archive_root=cli_workspace["archive_root"], db_path=db_path)
    report = await archive.insight_readiness_report()

    assert report.aggregate_verdict == "ready"
    assert {insight.insight_name for insight in report.insights} >= {
        "session_profiles",
        "session_enrichments",
        "session_work_events",
        "session_phases",
        "work_threads",
        "session_tag_rollups",
        "day_session_summaries",
        "week_session_summaries",
        "provider_analytics",
    }
    profile = _entry_by_name(report, "session_profiles")
    assert profile.verdict == "ready"
    assert profile.row_count == 1
    assert profile.provider_coverage[0].provider_name == "codex"
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
async def test_insight_readiness_report_marks_partial_and_legacy_insights(cli_workspace: dict[str, Path]) -> None:
    db_path = cli_workspace["db_path"]
    _seed_readiness_conversations(db_path)
    (
        ConversationBuilder(db_path, "ready-second")
        .provider("codex")
        .title("Ready Second")
        .created_at("2026-04-01T10:00:00+00:00")
        .updated_at("2026-04-01T10:05:00+00:00")
        .add_message("u2", role="user", text="Second session.", timestamp="2026-04-01T10:00:00+00:00")
        .save()
    )
    with open_connection(db_path) as conn:
        rebuild_session_insights_sync(conn)
        conn.execute("DELETE FROM session_profiles WHERE conversation_id = ?", ("ready-second",))
        conn.commit()

    archive = Polylogue(archive_root=cli_workspace["archive_root"], db_path=db_path)
    partial = await archive.insight_readiness_report(InsightReadinessQuery(insights=("session_profiles",)))
    assert _entry_by_name(partial, "session_profiles").verdict == "partial"

    with open_connection(db_path) as conn:
        rebuild_session_insights_sync(conn)
        conn.execute(
            "UPDATE session_profiles SET materializer_version = ?", (SESSION_INSIGHT_MATERIALIZER_VERSION - 1,)
        )
        conn.commit()

    legacy = await archive.insight_readiness_report(InsightReadinessQuery(insights=("session_profiles",)))
    profile = _entry_by_name(legacy, "session_profiles")
    assert profile.verdict == "legacy"
    assert profile.legacy_incompatible_count == 2


@pytest.mark.asyncio
async def test_insight_readiness_report_marks_stale_insights(cli_workspace: dict[str, Path]) -> None:
    db_path = cli_workspace["db_path"]
    _seed_readiness_conversations(db_path)
    with open_connection(db_path) as conn:
        rebuild_session_insights_sync(conn)
        conn.execute("UPDATE conversations SET sort_key = sort_key + 1 WHERE conversation_id = ?", ("ready-root",))
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
            CREATE TABLE conversations (
                conversation_id TEXT PRIMARY KEY,
                parent_conversation_id TEXT,
                provider_name TEXT,
                sort_key REAL,
                updated_at TEXT
            );
            INSERT INTO conversations (conversation_id, parent_conversation_id, provider_name, sort_key, updated_at)
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

"""Unit contracts for readiness reporting (consolidated module)."""

from __future__ import annotations

import importlib
import sqlite3
from collections.abc import Mapping
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.mark.parametrize(
    ("name", "status_name", "detail", "expected"),
    [
        ("test", "OK", "All good", {"name": "test", "status": "ok", "count": 0, "detail": "All good", "breakdown": {}}),
        (
            "database",
            "ERROR",
            "Connection failed",
            {"name": "database", "status": "error", "count": 0, "detail": "Connection failed", "breakdown": {}},
        ),
        (
            "archive",
            "WARNING",
            "Directory missing",
            {"name": "archive", "status": "warning", "count": 0, "detail": "Directory missing", "breakdown": {}},
        ),
    ],
)
def test_health_check_dataclass_contract(name: str, status_name: str, detail: str, expected: dict[str, object]) -> None:
    from polylogue.readiness import ReadinessCheck, VerifyStatus

    check = ReadinessCheck(name=name, status=getattr(VerifyStatus, status_name), summary=detail)
    assert {
        "name": check.name,
        "status": check.status.value,
        "count": check.count,
        "detail": check.summary,
        "breakdown": check.breakdown,
    } == expected


@pytest.mark.parametrize(
    ("deep", "expected_checks"),
    [
        (
            False,
            {
                "config",
                "archive_root",
                "render_root",
                "database",
                "index",
                "orphaned_messages",
                "action_event_read_model",
                "action_event_fts",
                "fts_sync",
                "transcript_embeddings",
                "transcript_embedding_freshness",
                "retrieval_evidence",
                "retrieval_inference",
                "retrieval_enrichment",
                "session_profile_enrichment_fts",
                "schemas_coverage",
                "schemas_freshness",
            },
        ),
        (
            True,
            {
                "config",
                "archive_root",
                "render_root",
                "database",
                "sqlite_integrity",
                "index",
                "orphaned_messages",
                "orphaned_content_blocks",
                "action_event_read_model",
                "action_event_fts",
                "fts_sync",
                "transcript_embeddings",
                "transcript_embedding_freshness",
                "retrieval_evidence",
                "retrieval_inference",
                "retrieval_enrichment",
                "session_profile_enrichment_fts",
                "schemas_coverage",
                "schemas_freshness",
            },
        ),
    ],
)
def test_run_health_core_contract(
    cli_workspace: Mapping[str, Path],
    deep: bool,
    expected_checks: set[str],
) -> None:
    from polylogue.config import get_config
    from polylogue.readiness import run_archive_readiness

    report = run_archive_readiness(get_config(), deep=deep)
    names = {check.name for check in report.checks}
    assert report.timestamp > 0
    assert expected_checks.issubset(names)
    assert sum(report.summary.values()) == len(report.checks)


@pytest.mark.parametrize(
    ("embedded_conversations", "embedded_messages", "pending_conversations", "expected_status", "expected_text"),
    [
        (0, 0, 0, "warning", "Transcript embeddings pending"),
        (2, 50, 1, "warning", "Transcript embeddings pending"),
        (3, 90, 0, "ok", "Transcript embeddings ready"),
    ],
)
def test_run_health_embedding_status_contract(
    cli_workspace: Mapping[str, Path],
    embedded_conversations: int,
    embedded_messages: int,
    pending_conversations: int,
    expected_status: str,
    expected_text: str,
) -> None:
    from polylogue.config import get_config
    from polylogue.maintenance.models import DerivedModelStatus
    from polylogue.readiness import run_archive_readiness
    from tests.infra.storage_records import ConversationBuilder

    ConversationBuilder(cli_workspace["db_path"], "health-embed-1").add_message(text="hello").save()
    ConversationBuilder(cli_workspace["db_path"], "health-embed-2").add_message(text="hello").save()
    ConversationBuilder(cli_workspace["db_path"], "health-embed-3").add_message(text="hello").save()

    embeddings_ready = embedded_conversations == 3 and pending_conversations == 0
    expected_detail = (
        f"Transcript embeddings ready ({embedded_conversations:,}/3 conversations, {embedded_messages:,} messages)"
        if embeddings_ready
        else (
            f"Transcript embeddings pending ({embedded_conversations:,}/3 conversations, "
            f"pending {pending_conversations:,}, stale 0, missing provenance 0)"
        )
    )
    with patch(
        "polylogue.storage.derived.derived_status.collect_derived_model_statuses_sync",
        return_value={
            "messages_fts": DerivedModelStatus(
                name="messages_fts",
                ready=True,
                detail="Messages FTS ready",
                source_rows=3,
                materialized_rows=3,
            ),
            "action_events": DerivedModelStatus(
                name="action_events",
                ready=True,
                detail="Action events ready",
                source_documents=3,
                materialized_documents=3,
                materialized_rows=0,
            ),
            "action_events_fts": DerivedModelStatus(
                name="action_events_fts",
                ready=True,
                detail="Action-event FTS ready",
                source_rows=0,
                materialized_rows=0,
            ),
            "transcript_embeddings": DerivedModelStatus(
                name="transcript_embeddings",
                ready=embeddings_ready,
                detail=expected_detail,
                source_documents=3,
                materialized_documents=embedded_conversations,
                materialized_rows=embedded_messages,
                pending_documents=pending_conversations,
            ),
            "retrieval_evidence": DerivedModelStatus(
                name="retrieval_evidence",
                ready=True,
                detail="Evidence retrieval ready",
                source_rows=3,
                materialized_rows=3,
            ),
            "retrieval_inference": DerivedModelStatus(
                name="retrieval_inference",
                ready=True,
                detail="Inference retrieval ready",
                source_rows=3,
                materialized_rows=3,
            ),
            "retrieval_enrichment": DerivedModelStatus(
                name="retrieval_enrichment",
                ready=True,
                detail="Enrichment retrieval ready",
                source_rows=3,
                materialized_rows=3,
            ),
            "session_profile_enrichment_fts": DerivedModelStatus(
                name="session_profile_enrichment_fts",
                ready=True,
                detail="Session-profile enrichment FTS ready",
                source_rows=3,
                materialized_rows=3,
            ),
        },
    ):
        report = run_archive_readiness(get_config())

    check = next(c for c in report.checks if c.name == "transcript_embeddings")
    assert check.status.value == expected_status
    assert expected_text in check.summary


@pytest.mark.parametrize(
    ("path_name", "missing"),
    [("archive_root", True), ("render_root", True), ("archive_root", False), ("render_root", False)],
)
def test_run_health_path_contracts(tmp_path: Path, path_name: str, missing: bool) -> None:
    from polylogue.config import Config, Source
    from polylogue.readiness import VerifyStatus, run_archive_readiness

    archive_root = tmp_path / "archive"
    render_root = tmp_path / "render"
    if path_name != "archive_root" or not missing:
        archive_root.mkdir(parents=True, exist_ok=True)
    if path_name != "render_root" or not missing:
        render_root.mkdir(parents=True, exist_ok=True)

    report = run_archive_readiness(
        Config(archive_root=archive_root, render_root=render_root, sources=[Source(name="test", path=tmp_path)])
    )
    check = next(c for c in report.checks if c.name == path_name)
    assert check.status == (VerifyStatus.WARNING if missing else VerifyStatus.OK)


def test_run_health_includes_source_checks(cli_workspace: Mapping[str, Path]) -> None:
    from polylogue.config import get_config
    from polylogue.readiness import run_archive_readiness

    config = get_config()
    report = run_archive_readiness(config)
    source_checks = [check for check in report.checks if check.name.startswith("source:")]
    assert len(source_checks) >= len(config.sources)


def test_run_archive_readiness_reports_busy_archive_with_operator_message(tmp_path: Path) -> None:
    from polylogue.config import Config, Source
    from polylogue.readiness import run_archive_readiness

    archive_root = tmp_path / "archive"
    render_root = tmp_path / "render"
    archive_root.mkdir(parents=True, exist_ok=True)
    render_root.mkdir(parents=True, exist_ok=True)

    config = Config(archive_root=archive_root, render_root=render_root, sources=[Source(name="test", path=tmp_path)])

    with patch(
        "polylogue.storage.sqlite.connection.open_connection",
        side_effect=sqlite3.OperationalError("database is locked"),
    ):
        report = run_archive_readiness(config)

    database_check = next(check for check in report.checks if check.name == "database")
    index_check = next(check for check in report.checks if check.name == "index")

    assert (
        database_check.summary
        == "DB error: database is locked (archive is busy; retry after the current run completes)"
    )
    assert (
        index_check.summary
        == "Skipped: database unavailable (database is locked (archive is busy; retry after the current run completes))"
    )


def test_run_archive_readiness_reports_legacy_inline_raw_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import polylogue.paths
    from polylogue.config import get_config
    from polylogue.readiness import VerifyStatus, run_archive_readiness

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    importlib.reload(polylogue.paths)

    db_path = polylogue.paths.db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE raw_conversations (
            raw_id TEXT PRIMARY KEY,
            provider_name TEXT NOT NULL,
            payload_provider TEXT,
            source_name TEXT,
            source_path TEXT NOT NULL,
            source_index INTEGER,
            raw_content BLOB NOT NULL,
            acquired_at TEXT NOT NULL,
            file_mtime TEXT,
            parsed_at TEXT,
            parse_error TEXT,
            validated_at TEXT,
            validation_status TEXT,
            validation_error TEXT,
            validation_drift_count INTEGER DEFAULT 0,
            validation_provider TEXT,
            validation_mode TEXT
        );
        PRAGMA user_version = 1;
        """
    )
    conn.commit()
    conn.close()

    report = run_archive_readiness(get_config())

    database_check = next(check for check in report.checks if check.name == "database")
    index_check = next(check for check in report.checks if check.name == "index")
    assert database_check.status == VerifyStatus.ERROR
    assert "legacy inline raw-content layout" in database_check.summary
    assert index_check.status == VerifyStatus.WARNING


@pytest.mark.parametrize("deep", [False, True])
def test_get_readiness_contract(cli_workspace: Mapping[str, Path], deep: bool) -> None:
    from polylogue.config import get_config
    from polylogue.readiness import get_readiness

    config = get_config()
    report = get_readiness(config, deep=deep)
    assert report.timestamp > 0


def test_quick_readiness_summary_returns_live_status(tmp_path: Path) -> None:
    from polylogue.readiness import quick_readiness_summary

    result = quick_readiness_summary(tmp_path)
    # With a valid DB, returns "OK (N conversations)"; without, returns "unavailable (...)"
    assert isinstance(result, str)
    assert "OK" in result or "unavailable" in result or "schema" in result


def test_verify_status_contract() -> None:
    from polylogue.readiness import READINESS_TTL_SECONDS, VerifyStatus

    assert str(VerifyStatus.OK) == "ok"
    assert str(VerifyStatus.WARNING) == "warning"
    assert str(VerifyStatus.ERROR) == "error"
    assert 60 <= READINESS_TTL_SECONDS <= 3600
